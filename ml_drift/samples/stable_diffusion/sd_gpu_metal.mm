// Copyright 2026 The ML Drift Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import <Metal/Metal.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/flops_util.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/kernels/elementwise.h"
#include "ml_drift/common/kernels/mean_stddev_normalization.h"
#include "ml_drift/common/kernels/random_philox.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/model_transformer.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/selectors/convolution_selector.h"
#include "ml_drift/common/selectors/simple_selectors.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/transformations/model_transformations.h"
#include "ml_drift/common/util.h"
#include "ml_drift/common/winograd_util.h"
#include "ml_drift/metal/inference_context.h"
#include "ml_drift/metal/metal_device.h"
#include "ml_drift/samples/stable_diffusion/autoencoder_kl_builder.h"
#include "ml_drift/samples/stable_diffusion/bpe_tokenizer.h"
#include "ml_drift/samples/stable_diffusion/text_guidance_builder.h"
#include "ml_drift/samples/stable_diffusion/unet_builder.h"
#include "ml_drift/samples/stable_diffusion/util.h"

namespace ml_drift {
namespace metal {
namespace {

class OpHolder {
 public:
  struct ExecutionParams {
    std::vector<MetalSpatialTensor*> src;
    std::vector<MetalSpatialTensor*> dst;
    std::map<std::string, float> float_params;
    std::map<std::string, half> half_params;
    std::map<std::string, int> int_params;
  };

  absl::Status Init(MetalDevice* device, GPUOperation&& operation) {
    return InitMetalOp(device, std::move(operation));
  }

  absl::Status InitElementwiseOneInput(OperationType op_type, const TensorDescriptor& src_desc,
                                       const TensorDescriptor& dst_desc, MetalDevice* device) {
    OperationDef op_def;
    op_def.src_tensors.push_back(src_desc);
    op_def.dst_tensors.push_back(dst_desc);
    GPUOperation operation = CreateElementwiseOneInput(device->GetInfo(), op_def, op_type);
    return InitMetalOp(device, std::move(operation));
  }

  absl::Status Encode(id<MTLComputeCommandEncoder> encoder, float scalar, MetalSpatialTensor* src,
                      MetalSpatialTensor* dst) {
    metal_op_.SetSrcTensor(src, 0);
    metal_op_.SetDstTensor(dst, 0);
    ABSL_RETURN_IF_ERROR(metal_op_.SetHalf("scalar", scalar));
    ABSL_RETURN_IF_ERROR(metal_op_.UpdateParams());
    metal_op_.Encode(encoder);
    return absl::OkStatus();
  }

  absl::Status Encode(id<MTLComputeCommandEncoder> encoder, const ExecutionParams& params) {
    for (int i = 0; i < params.src.size(); ++i) {
      metal_op_.SetSrcTensor(params.src[i], i);
    }
    for (int i = 0; i < params.dst.size(); ++i) {
      metal_op_.SetDstTensor(params.dst[i], i);
    }
    for (const auto& float_param : params.float_params) {
      ABSL_RETURN_IF_ERROR(metal_op_.SetFloat(float_param.first, float_param.second));
    }
    for (const auto& half_param : params.half_params) {
      ABSL_RETURN_IF_ERROR(metal_op_.SetHalf(half_param.first, half_param.second));
    }
    for (const auto& int_param : params.int_params) {
      ABSL_RETURN_IF_ERROR(metal_op_.SetInt(int_param.first, int_param.second));
    }
    ABSL_RETURN_IF_ERROR(metal_op_.UpdateParams());
    metal_op_.Encode(encoder);
    return absl::OkStatus();
  }

  absl::Status Encode(id<MTLComputeCommandEncoder> encoder, MetalSpatialTensor* src,
                      MetalSpatialTensor* dst) {
    metal_op_.SetSrcTensor(src, 0);
    metal_op_.SetDstTensor(dst, 0);
    ABSL_RETURN_IF_ERROR(metal_op_.UpdateParams());
    metal_op_.Encode(encoder);
    return absl::OkStatus();
  }

  absl::Status Encode(id<MTLComputeCommandEncoder> encoder, MetalSpatialTensor* src0,
                      MetalSpatialTensor* src1, MetalSpatialTensor* dst) {
    metal_op_.SetSrcTensor(src0, 0);
    metal_op_.SetSrcTensor(src1, 1);
    metal_op_.SetDstTensor(dst, 0);
    ABSL_RETURN_IF_ERROR(metal_op_.UpdateParams());
    metal_op_.Encode(encoder);
    return absl::OkStatus();
  }

 private:
  absl::Status InitMetalOp(MetalDevice* device, GPUOperation&& operation) {
    auto gpu_op = std::make_unique<GPUOperation>(std::move(operation));

    ABSL_RETURN_IF_ERROR(gpu_op->AssembleCode(device->GetInfo()));

    metal_op_.Init(std::move(gpu_op));
    ABSL_RETURN_IF_ERROR(metal_op_.Compile(device));
    return absl::OkStatus();
  }
  ComputeTask metal_op_;
};

class DiffusionStepper {
 public:
  absl::Status Init(MetalDevice* device, const std::string& dir_folder) {
    alphas_ = LoadF16(dir_folder + "alphas_cumprod.bin", /*variable=*/1000);
    alphas_prev_ = alphas_;
    alphas_prev_.insert(alphas_prev_.begin(), half(1.0f));

    TensorDescriptor desc(DataType::FLOAT16, GetFastestStorageType(device->GetInfo()), Layout::HWC);

    ABSL_RETURN_IF_ERROR(custom_op_.Init(device, CreateDiffusionStepOp(desc, desc, desc, desc)));
    return absl::OkStatus();
  }
  absl::Status StepCustomOp(id<MTLComputeCommandEncoder> encoder, MetalSpatialTensor* xIn,
                            MetalSpatialTensor* etaUncondIn, MetalSpatialTensor* etaCondIn,
                            MetalSpatialTensor* dst, int tIn, int tPrevIn, float guidanceScaleIn) {
    float alphaIn = alphas_[tIn];
    int tPrevInOffset = std::max(0, tPrevIn + 1);
    float alphaPrevIn = alphas_prev_[tPrevInOffset];
    half sqrt_alpha = half(sqrt(alphaIn));
    half sqrt_alpha_prev = half(sqrt(alphaPrevIn));
    half sqrt_one_minus_alpha = half(sqrt(1.0f - alphaIn));
    half sqrt_one_minus_alpha_prev = half(sqrt(1.0f - alphaPrevIn));

    OpHolder::ExecutionParams exec_params;
    exec_params.src = {xIn, etaUncondIn, etaCondIn};
    exec_params.dst = {dst};
    exec_params.half_params = {{"guidance_scale", half(guidanceScaleIn)},
                               {"sqrt_alpha", half(sqrt_alpha)},
                               {"sqrt_alpha_prev", half(sqrt_alpha_prev)},
                               {"sqrt_one_minus_alpha", half(sqrt_one_minus_alpha)},
                               {"sqrt_one_minus_alpha_prev", half(sqrt_one_minus_alpha_prev)}};
    return custom_op_.Encode(encoder, exec_params);
  }

 private:
  OpHolder custom_op_;
  std::vector<half> alphas_;
  std::vector<half> alphas_prev_;
};

class TextGuidance {
 public:
  absl::Status Init(MetalDevice* device, const std::string& file_folder) {
    const auto& gpu_info = device->GetInfo();

    CreateGpuModelInfo create_info;
    create_info.precision = CalculationsPrecision::F16;
    create_info.storage_type = GetFastestStorageType(gpu_info);
    create_info.hints.Add(ModelHints::kFastTuning);
    create_info.hints.Add(ModelHints::kNoWinogradOptimizations);
    GpuModel gpu_model;
    TextGuidanceBuilder::Config config;
    config.embedding_size = 768;
    config.num_layers = 12;
    config.num_heads = 12;
    config.file_folder = file_folder;
    TextGuidanceBuilder builder;
    ABSL_RETURN_IF_ERROR(builder.Build(config, gpu_info, create_info, &gpu_model, &src_,
                                       /*mask_ptr=*/nullptr, &dst_));

    const auto start_init = std::chrono::high_resolution_clock::now();
    ABSL_RETURN_IF_ERROR(
        inference_context_.InitFromGpuModel(create_info, &gpu_model, device->device(), nullptr));
    const auto end_init = std::chrono::high_resolution_clock::now();
    std::cout << "TextGuidance initialization time: " << (end_init - start_init).count() * 1e-6f
              << " ms." << std::endl;

    return absl::OkStatus();
  }

  MetalSpatialTensor* GetGuidanceTensor() { return inference_context_.GetTensor(dst_.id); }

  absl::Status SetInput(const ml_drift::Tensor<BHWC, DataType::INT32>& src) {
    return inference_context_.SetInputTensor(src_.id, src);
  }
  absl::Status Execute(id<MTLCommandBuffer> command_buffer) {
    inference_context_.EncodeWithCommandBuffer(command_buffer);
    return absl::OkStatus();
  }

 private:
  GpuModelBuilder::TensorHandle src_, dst_;
  InferenceContext inference_context_;
};

class UNet {
 public:
  absl::Status Init(int width, int height, MetalDevice* device, const std::string& file_folder) {
    const auto& gpu_info = device->GetInfo();

    CreateGpuModelInfo create_info;
    create_info.precision = CalculationsPrecision::F16;
    create_info.storage_type = GetFastestStorageType(gpu_info);
    create_info.hints.Add(ModelHints::kFastTuning);
    create_info.hints.Add(ModelHints::kNoWinogradOptimizations);
    GpuModel gpu_model;
    UnetBuilder builder;
    UnetBuilder::Config config;
    config.in_channels = 4;
    config.out_channels = 4;
    config.model_channels = 320;
    config.num_res_blocks = {2, 2, 2, 2};
    config.num_transformer_blocks = {{1, 1}, {1, 1},    {1, 1},    {},
                                     {},     {1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    config.channel_mult = {1, 2, 4, 4};
    config.num_attn_heads = {8, 8, 8, 8};
    config.use_spatial_transformer = true;
    config.transformer_depth = 1;
    config.context_dim = 768;
    config.unet_file_dir = file_folder;
    GpuModelBuilder gpu_builder(gpu_info, create_info.hints, create_info.precision,
                                create_info.storage_type);

    ABSL_RETURN_IF_ERROR(builder.Build(config, gpu_info, create_info, width, height, &gpu_builder,
                                       &src_, &temb_, &guidance_, /*text_proj_ptr=*/nullptr,
                                       /*masked_image_latent_ptr=*/nullptr, &eta0_, &eta1_,
                                       nullptr));

    std::vector<GpuModelBuilder::ValueId> input_ids = {src_.id, temb_.id, guidance_.id};
    std::vector<GpuModelBuilder::ValueId> output_ids = {eta0_.id, eta1_.id, src_.id, guidance_.id};
    ABSL_RETURN_IF_ERROR(gpu_builder.GetGpuModel(input_ids, output_ids, &gpu_model));

    const auto start_init = std::chrono::high_resolution_clock::now();
    ABSL_RETURN_IF_ERROR(
        inference_context_.InitFromGpuModel(create_info, &gpu_model, device->device(), nullptr));
    const auto end_init = std::chrono::high_resolution_clock::now();
    std::cout << "UNet initialization time: " << (end_init - start_init).count() * 1e-6f << " ms."
              << std::endl;

    return absl::OkStatus();
  }

  MetalSpatialTensor* GetLatentTensor() { return inference_context_.GetTensor(src_.id); }
  MetalSpatialTensor* GetTembTensor() { return inference_context_.GetTensor(temb_.id); }
  MetalSpatialTensor* GetGuidanceTensor() { return inference_context_.GetTensor(guidance_.id); }
  MetalSpatialTensor* GetEtaUncondTensor() { return inference_context_.GetTensor(eta0_.id); }
  MetalSpatialTensor* GetEtaCondTensor() { return inference_context_.GetTensor(eta1_.id); }

  absl::Status Execute(id<MTLCommandBuffer> command_buffer) {
    inference_context_.EncodeWithCommandBuffer(command_buffer);
    return absl::OkStatus();
  }

 private:
  GpuModelBuilder::TensorHandle src_, temb_, guidance_, eta0_, eta1_;
  InferenceContext inference_context_;
};

class Decoder {
 public:
  absl::Status Init(int width, int height, MetalDevice* device, const std::string& file_folder) {
    const auto& gpu_info = device->GetInfo();

    CreateGpuModelInfo create_info;
    create_info.precision = CalculationsPrecision::F16;
    create_info.storage_type = GetFastestStorageType(gpu_info);
    create_info.hints.Add(ModelHints::kFastTuning);
    create_info.hints.Add(ModelHints::kNoWinogradOptimizations);
    GpuModel gpu_model;
    AutoencoderKLBuilder builder;
    AutoencoderKLBuilder::Config config;
    config.double_z = true;
    config.z_channels = 4;
    config.resolution = 256;
    config.in_channels = 3;
    config.out_ch = 3;
    config.ch = 128;
    config.ch_mult = {1, 2, 4, 4};
    config.num_res_blocks = 2;
    ABSL_RETURN_IF_ERROR(builder.BuildDecoder(config, gpu_info, create_info, width, height,
                                              &gpu_model, &src_, &dst_, file_folder));

    const auto start_init = std::chrono::high_resolution_clock::now();
    ABSL_RETURN_IF_ERROR(
        inference_context_.InitFromGpuModel(create_info, &gpu_model, device->device(), nullptr));
    const auto end_init = std::chrono::high_resolution_clock::now();
    std::cout << "Decoder initialization time: " << (end_init - start_init).count() * 1e-6f
              << " ms." << std::endl;

    return absl::OkStatus();
  }

  MetalSpatialTensor* GetInputTensor() { return inference_context_.GetTensor(src_.id); }

  absl::Status GetOutput(TensorFloat32* dst) {
    return inference_context_.GetOutputTensor(dst_.id, dst);
  }

  absl::Status Execute(id<MTLCommandBuffer> command_buffer) {
    inference_context_.EncodeWithCommandBuffer(command_buffer);
    return absl::OkStatus();
  }

 private:
  GpuModelBuilder::TensorHandle src_, dst_;
  InferenceContext inference_context_;
};

absl::Status TestStableDiffusion(std::string weights_path) {
  BPETokenizer bpe_tokenizer;
  bpe_tokenizer.Init(weights_path);

  MetalDevice device_;
  const auto& gpu_info = device_.GetInfo();

  TensorDescriptor default_desc(DataType::FLOAT16, GetFastestStorageType(gpu_info), Layout::HWC);

  OpHolder temb_generation_op;
  ABSL_RETURN_IF_ERROR(temb_generation_op.Init(
      &device_, CreateTembGenerationOp(gpu_info, default_desc, weights_path)));

  OpHolder noise_op;
  {
    OperationDef op_def;
    op_def.dst_tensors.push_back(default_desc);
    ABSL_RETURN_IF_ERROR(noise_op.Init(&device_, CreateRandomNormalPhilox(gpu_info, op_def)));
  }

  DiffusionStepper diffusion;
  ABSL_RETURN_IF_ERROR(diffusion.Init(&device_, weights_path));

  TextGuidance text_guidance_graph;
  ABSL_RETURN_IF_ERROR(text_guidance_graph.Init(&device_, weights_path));

  const int final_image_width = 512;
  const int final_image_height = 512;

  const int latent_image_width = final_image_width / 8;
  const int latent_image_height = final_image_height / 8;

  UNet unet;
  ABSL_RETURN_IF_ERROR(unet.Init(latent_image_width, latent_image_height, &device_, weights_path));

  Decoder decoder;
  ABSL_RETURN_IF_ERROR(
      decoder.Init(latent_image_width, latent_image_height, &device_, weights_path));

  OpHolder copier;
  ABSL_RETURN_IF_ERROR(
      copier.InitElementwiseOneInput(OperationType::COPY, default_desc, default_desc, &device_));

  ml_drift::Tensor<BHWC, DataType::INT32> tokens;
  tokens.shape = BHWC(1, 1, 2, 77);
  tokens.data.resize(tokens.shape.DimensionsProduct());

  MetalSpatialTensor latent_copy;
  {
    MetalSpatialTensor* unet_latent = unet.GetLatentTensor();
    TensorDescriptor descriptor_with_shape = unet_latent->GetDescriptor();
    descriptor_with_shape.SetBHWCShape(BHWC(unet_latent->Batch(), unet_latent->Height(),
                                            unet_latent->Width(), unet_latent->Channels()));
    ABSL_RETURN_IF_ERROR(CreateTensor(device_.device(), descriptor_with_shape, &latent_copy));
  }

  id<MTLCommandQueue> command_queue = [device_.device() newCommandQueue];

  unsigned int base_seed = 0;

  while (true) {
    std::string prompt = "a photo of an astronaut riding a horse on mars";
    std::string negative_prompt = "";
    std::cout << "Enter phrase: ";
    std::getline(std::cin, prompt);
    int steps = 50;
    std::cout << "Enter steps: ";
    std::cin >> steps;
    std::string dummy;
    std::getline(std::cin, dummy);

    const auto p0 = std::chrono::high_resolution_clock::now();

    auto bpe_tokens = bpe_tokenizer.Encode(prompt);
    auto bpe_base_tokens = bpe_tokenizer.Encode(negative_prompt);

    for (int i = 0; i < 77; ++i) {
      tokens.data[i] = bpe_base_tokens[i];
      tokens.data[i + 77] = bpe_tokens[i];
    }
    ABSL_RETURN_IF_ERROR(text_guidance_graph.SetInput(tokens));

    const auto p1 = std::chrono::high_resolution_clock::now();

    {
      id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
      ABSL_RETURN_IF_ERROR(text_guidance_graph.Execute(command_buffer));
      [command_buffer commit];
    }

    {
      int seed = rand_r(&base_seed);
      OpHolder::ExecutionParams exec_params;
      exec_params.dst = {unet.GetLatentTensor()};
      exec_params.int_params = {{"seed", seed}};
      id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      ABSL_RETURN_IF_ERROR(noise_op.Encode(encoder, exec_params));
      [encoder endEncoding];
      [command_buffer commit];
    }

    {
      id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      ABSL_RETURN_IF_ERROR(copier.Encode(encoder, text_guidance_graph.GetGuidanceTensor(),
                                         unet.GetGuidanceTensor()));
      [encoder endEncoding];
      [command_buffer commit];
    }

    const int stride = 1000 / steps;
    for (int t = steps - 1; t >= 0; --t) {
      int ts = t * stride + 1;
      int tsPrev = ts - stride;

      {
        OpHolder::ExecutionParams exec_params;
        exec_params.dst = {unet.GetTembTensor()};
        exec_params.float_params = {{"index_val", ts}};
        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        ABSL_RETURN_IF_ERROR(temb_generation_op.Encode(encoder, exec_params));
        [encoder endEncoding];
        [command_buffer commit];
      }

      {
        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        ABSL_RETURN_IF_ERROR(copier.Encode(encoder, unet.GetLatentTensor(), &latent_copy));
        [encoder endEncoding];
        [command_buffer commit];
      }

      {
        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        ABSL_RETURN_IF_ERROR(unet.Execute(command_buffer));
        [command_buffer commit];
      }

      float guidance_scale = 7.5f;
      {
        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        ABSL_RETURN_IF_ERROR(diffusion.StepCustomOp(
            encoder, &latent_copy, unet.GetEtaUncondTensor(), unet.GetEtaCondTensor(),
            unet.GetLatentTensor(), ts, tsPrev, guidance_scale));
        [encoder endEncoding];
        [command_buffer commit];
      }
      std::cout << "step " << steps - t << "/" << steps << std::endl;
    }  // loop t

    {
      id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      ABSL_RETURN_IF_ERROR(
          copier.Encode(encoder, unet.GetLatentTensor(), decoder.GetInputTensor()));
      [encoder endEncoding];
      [command_buffer commit];
    }

    {
      id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
      ABSL_RETURN_IF_ERROR(decoder.Execute(command_buffer));
      [command_buffer commit];
      [command_buffer waitUntilCompleted];  // Required sync.
    }

    const auto p2 = std::chrono::high_resolution_clock::now();
    TensorFloat32 result;
    ABSL_RETURN_IF_ERROR(decoder.GetOutput(&result));
    const auto p3 = std::chrono::high_resolution_clock::now();

    std::cout << "Cpu begin time: " << (p1 - p0).count() * 1e-6f << " ms." << std::endl;

    std::cout << "Gpu time: " << (p2 - p1).count() * 1e-6f << " ms." << std::endl;

    std::cout << "Cpu end time: " << (p3 - p2).count() * 1e-6f << " ms." << std::endl;

    std::cout << "Total time: " << (p3 - p0).count() * 1e-6f << " ms." << std::endl;

    GenerateImage(result, "result.bmp");
    std::cout << "ready" << std::endl;
  }

  return absl::OkStatus();
}

}  // namespace
}  // namespace metal
}  // namespace ml_drift

int main(int argc, char** argv) {
  @autoreleasepool {
    std::string weights_folder = "sd_1_5/";
    if (argc >= 2) {
      weights_folder = argv[1];
      if (!weights_folder.empty() && weights_folder.back() != '/') {
        weights_folder += '/';
      }
    }
    auto status = ml_drift::metal::TestStableDiffusion(weights_folder);
    if (!status.ok()) {
      std::cout << "Failed test." << status.message() << std::endl;
      return -1;
    }
  }
  return 0;
}
