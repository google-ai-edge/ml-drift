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

#include <algorithm>
#include <chrono>  // NOLINT(build/c++11)
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/time/time.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/kernels/elementwise.h"
#include "ml_drift/common/kernels/random_philox.h"
#include "ml_drift/common/model_hints.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/samples/stable_diffusion/autoencoder_kl_builder.h"
#include "ml_drift/samples/stable_diffusion/bpe_tokenizer.h"
#include "ml_drift/samples/stable_diffusion/text_guidance_builder.h"
#include "ml_drift/samples/stable_diffusion/unet_builder.h"
#include "ml_drift/samples/stable_diffusion/util.h"
#include "ml_drift/webgpu/compute_task.h"
#include "ml_drift/webgpu/environment.h"
#include "ml_drift/webgpu/inference_context.h"
#include "ml_drift/webgpu/spatial_tensor.h"
#include "ml_drift/webgpu/webgpu_api_util.h"
#include "ml_drift/webgpu/webgpu_headers.h"

namespace ml_drift {
namespace webgpu {
namespace {

absl::Status WaitUntilCompleted(Environment* env) {
  // Can use this to wait at each step, or can just skip all waiting by
  // commenting out.
  return ml_drift::webgpu::WaitUntilCompleted(env->queue(), env->device(),
                                              absl::Minutes(1));
  // return absl::OkStatus();
}

// Hardcoded to Texture2D on web for now.
TensorStorageType GetFastestStorageType(GpuInfo const& info) {
  return TensorStorageType::TEXTURE_2D;
}

// Helper class for running the one-off ops (OpHolder & DiffusionStepper)
class ComputePassHelper {
 public:
  explicit ComputePassHelper(const Environment* env) : env_(env) {
    command_encoder_ = env_->device().CreateCommandEncoder();
    compute_pass_encoder_ = command_encoder_.BeginComputePass();
  }

  const wgpu::ComputePassEncoder& compute_pass_encoder() const {
    return compute_pass_encoder_;
  }

  wgpu::CommandBuffer FinalizeCommandBuffer() {
    compute_pass_encoder_.End();
    return command_encoder_.Finish();
  }

  void Submit() {
    wgpu::CommandBuffer cb = FinalizeCommandBuffer();
    env_->queue().Submit(1, &cb);
  }

 private:
  const Environment* env_;
  wgpu::CommandEncoder command_encoder_;
  wgpu::ComputePassEncoder compute_pass_encoder_;
};

class OpHolder {
 public:
  struct ExecutionParams {
    std::vector<SpatialTensor*> src;
    std::vector<SpatialTensor*> dst;
    std::map<std::string, float> float_params;
    std::map<std::string, half> half_params;
    std::map<std::string, int> int_params;
  };
  absl::Status Init(Environment* env, GPUOperation&& operation) {
    return InitWebgpuOp(env, std::move(operation));
  }

  absl::Status InitElementwiseOneInput(OperationType op_type,
                                       const TensorDescriptor& src_desc,
                                       const TensorDescriptor& dst_desc,
                                       Environment* env) {
    OperationDef op_def;
    op_def.src_tensors.push_back(src_desc);
    op_def.dst_tensors.push_back(dst_desc);
    GPUOperation operation =
        CreateElementwiseOneInput(env->GetInfo(), op_def, op_type);
    return InitWebgpuOp(env, std::move(operation));
  }

  absl::Status Encode(wgpu::ComputePassEncoder encoder, Environment* env,
                      const ExecutionParams& params) {
    for (int i = 0; i < params.src.size(); ++i) {
      RETURN_IF_ERROR(webgpu_op_.SetSrcTensor(i, params.src[i]));
    }
    for (int i = 0; i < params.dst.size(); ++i) {
      RETURN_IF_ERROR(webgpu_op_.SetDstTensor(i, params.dst[i]));
    }
    for (const auto& float_param : params.float_params) {
      RETURN_IF_ERROR(
          webgpu_op_.SetFloat(float_param.first, float_param.second));
    }
    for (const auto& half_param : params.half_params) {
      RETURN_IF_ERROR(webgpu_op_.SetHalf(half_param.first, half_param.second));
    }
    for (const auto& int_param : params.int_params) {
      RETURN_IF_ERROR(webgpu_op_.SetInt(int_param.first, int_param.second));
    }
    RETURN_IF_ERROR(webgpu_op_.Update(env->device()));
    webgpu_op_.UpdateGpuObjectBindings(env->device());
    return webgpu_op_.Encode(encoder);
  }

  absl::Status Encode(wgpu::ComputePassEncoder encoder, Environment* env,
                      SpatialTensor* src, SpatialTensor* dst) {
    RETURN_IF_ERROR(webgpu_op_.SetSrcTensor(0, src));
    RETURN_IF_ERROR(webgpu_op_.SetDstTensor(0, dst));
    RETURN_IF_ERROR(webgpu_op_.Update(env->device()));
    webgpu_op_.UpdateGpuObjectBindings(env->device());
    return webgpu_op_.Encode(encoder);
  }

 private:
  absl::Status InitWebgpuOp(Environment* env, GPUOperation&& operation) {
    auto gpu_op = std::make_unique<GPUOperation>(std::move(operation));

    RETURN_IF_ERROR(gpu_op->AssembleCode(env->GetInfo()));

    RETURN_IF_ERROR(webgpu_op_.Init(*env, std::move(gpu_op)));
    RETURN_IF_ERROR(webgpu_op_.Compile(*env));
    return absl::OkStatus();
  }
  ComputeTask webgpu_op_;
};

class DiffusionStepper {
 public:
  absl::Status Init(Environment* env, const std::string& dir_folder) {
    alphas_ = LoadF16(dir_folder + "alphas_cumprod.bin", /*variable=*/1000);
    alphas_prev_ = alphas_;
    alphas_prev_.insert(alphas_prev_.begin(), half(1.0f));

    TensorDescriptor desc(DataType::FLOAT16,
                          GetFastestStorageType(env->GetInfo()), Layout::HWC);

    RETURN_IF_ERROR(
        custom_op_.Init(env, CreateDiffusionStepOp(desc, desc, desc, desc)));
    return absl::OkStatus();
  }
  absl::Status StepCustomOp(wgpu::ComputePassEncoder encoder, Environment* env,
                            SpatialTensor* xIn, SpatialTensor* etaUncondIn,
                            SpatialTensor* etaCondIn, SpatialTensor* dst,
                            int tIn, int tPrevIn, float guidanceScaleIn) {
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
    exec_params.half_params = {
        {"guidance_scale", half(guidanceScaleIn)},
        {"sqrt_alpha", half(sqrt_alpha)},
        {"sqrt_alpha_prev", half(sqrt_alpha_prev)},
        {"sqrt_one_minus_alpha", half(sqrt_one_minus_alpha)},
        {"sqrt_one_minus_alpha_prev", half(sqrt_one_minus_alpha_prev)}};
    return custom_op_.Encode(encoder, env, exec_params);
  }

 private:
  OpHolder custom_op_;
  std::vector<half> alphas_;
  std::vector<half> alphas_prev_;
};

class TextGuidance {
 public:
  absl::Status Init(Environment* env, const std::string& file_folder) {
    const auto& gpu_info = env->GetInfo();

    CreateGpuModelInfo create_info;
    create_info.precision = CalculationsPrecision::F16;
    create_info.storage_type = GetFastestStorageType(gpu_info);
    create_info.hints.Add(ModelHints::kFastTuning);
    create_info.hints.Add(ModelHints::kNoWinogradOptimizations);
    create_info.hints.Add(ModelHints::kPreferTextureWeights);  // new
    GpuModel gpu_model;
    TextGuidanceBuilder::Config config;
    config.embedding_size = 768;
    config.num_layers = 12;
    config.num_heads = 12;
    config.file_folder = file_folder;
    TextGuidanceBuilder builder;
    RETURN_IF_ERROR(builder.Build(config, gpu_info, create_info, &gpu_model,
                                  &src_, /*mask_ptr=*/nullptr, &dst_));

    const auto start_init = std::chrono::high_resolution_clock::now();
    RETURN_IF_ERROR(
        inference_context_.InitFromGpuModel(*env, create_info, &gpu_model));
    const auto end_init = std::chrono::high_resolution_clock::now();
    std::cout << "TextGuidance initialization time: "
              << (end_init - start_init).count() * 1e-6f << " ms." << std::endl;

    return absl::OkStatus();
  }

  SpatialTensor* GetGuidanceTensor() {
    return inference_context_.GetTensor(dst_.id);
  }

  absl::Status SetInput(Environment* env,
                        const ml_drift::Tensor<BHWC, DataType::INT32>& src) {
    return inference_context_.SetInputTensor(*env, src_.id, src);
  }
  absl::Status Execute(Environment* env) {
    return inference_context_.Execute(*env);
  }

 private:
  GpuModelBuilder::TensorHandle src_, dst_;
  InferenceContext inference_context_;
};

class UNet {
 public:
  absl::Status Init(int width, int height, Environment* env,
                    const std::string& file_folder) {
    const auto& gpu_info = env->GetInfo();

    CreateGpuModelInfo create_info;
    create_info.precision = CalculationsPrecision::F16;
    create_info.storage_type = GetFastestStorageType(gpu_info);
    create_info.hints.Add(ModelHints::kFastTuning);
    create_info.hints.Add(ModelHints::kNoWinogradOptimizations);
    create_info.hints.Add(ModelHints::kPreferTextureWeights);  // new
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
    RETURN_IF_ERROR(builder.Build(config, gpu_info, create_info, width, height,
                                  &gpu_model, &src_, &temb_, &guidance_,
                                  /*text_proj_ptr=*/nullptr,
                                  /*masked_image_latent_ptr=*/nullptr, &eta0_,
                                  &eta1_, nullptr));

    const auto start_init = std::chrono::high_resolution_clock::now();
    RETURN_IF_ERROR(
        inference_context_.InitFromGpuModel(*env, create_info, &gpu_model));
    const auto end_init = std::chrono::high_resolution_clock::now();
    std::cout << "UNet initialization time: "
              << (end_init - start_init).count() * 1e-6f << " ms." << std::endl;

    return absl::OkStatus();
  }

  SpatialTensor* GetLatentTensor() {
    return inference_context_.GetTensor(src_.id);
  }
  SpatialTensor* GetTembTensor() {
    return inference_context_.GetTensor(temb_.id);
  }
  SpatialTensor* GetGuidanceTensor() {
    return inference_context_.GetTensor(guidance_.id);
  }
  SpatialTensor* GetEtaUncondTensor() {
    return inference_context_.GetTensor(eta0_.id);
  }
  SpatialTensor* GetEtaCondTensor() {
    return inference_context_.GetTensor(eta1_.id);
  }

  absl::Status Execute(Environment* env) {
    return inference_context_.Execute(*env);
  }

 private:
  GpuModelBuilder::TensorHandle src_, temb_, guidance_, eta0_, eta1_;
  InferenceContext inference_context_;
};

class Decoder {
 public:
  absl::Status Init(int width, int height, Environment* env,
                    const std::string& file_folder) {
    const auto& gpu_info = env->GetInfo();

    CreateGpuModelInfo create_info;
    create_info.precision = CalculationsPrecision::F16;
    create_info.storage_type = GetFastestStorageType(gpu_info);
    create_info.hints.Add(ModelHints::kFastTuning);
    create_info.hints.Add(ModelHints::kNoWinogradOptimizations);
    create_info.hints.Add(ModelHints::kPreferTextureWeights);  // new
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
    RETURN_IF_ERROR(builder.BuildDecoder(config, gpu_info, create_info, width,
                                         height, &gpu_model, &src_, &dst_,
                                         file_folder));

    const auto start_init = std::chrono::high_resolution_clock::now();
    RETURN_IF_ERROR(
        inference_context_.InitFromGpuModel(*env, create_info, &gpu_model));
    const auto end_init = std::chrono::high_resolution_clock::now();
    std::cout << "Decoder initialization time: "
              << (end_init - start_init).count() * 1e-6f << " ms." << std::endl;

    return absl::OkStatus();
  }

  SpatialTensor* GetInputTensor() {
    return inference_context_.GetTensor(src_.id);
  }

  absl::Status GetOutput(Environment* env, TensorFloat32* dst) {
    return inference_context_.GetOutputTensor(*env, dst_.id, dst);
  }

  absl::Status Execute(Environment* env) {
    return inference_context_.Execute(*env);
  }

 private:
  GpuModelBuilder::TensorHandle src_, dst_;
  InferenceContext inference_context_;
};

absl::Status TestStableDiffusion(std::string weights_path) {
  BPETokenizer bpe_tokenizer;
  bpe_tokenizer.Init(weights_path);

  Environment env;
  env.RequestExtension("shader_f16");
  RETURN_IF_ERROR(env.Initialize());
  const auto& gpu_info = env.GetInfo();

  TensorDescriptor default_desc(DataType::FLOAT16,
                                GetFastestStorageType(gpu_info), Layout::HWC);

  OpHolder temb_generation_op;
  RETURN_IF_ERROR(temb_generation_op.Init(
      &env, CreateTembGenerationOp(gpu_info, default_desc, weights_path)));

  OpHolder noise_op;
  {
    OperationDef op_def;
    op_def.dst_tensors.push_back(default_desc);
    RETURN_IF_ERROR(
        noise_op.Init(&env, CreateRandomNormalPhilox(gpu_info, op_def)));
  }

  DiffusionStepper diffusion;
  RETURN_IF_ERROR(diffusion.Init(&env, weights_path));

  TextGuidance text_guidance_graph;
  RETURN_IF_ERROR(text_guidance_graph.Init(&env, weights_path));

  const int final_image_width = 512;
  const int final_image_height = 512;

  const int latent_image_width = final_image_width / 8;
  const int latent_image_height = final_image_height / 8;

  UNet unet;
  RETURN_IF_ERROR(
      unet.Init(latent_image_width, latent_image_height, &env, weights_path));

  Decoder decoder;
  RETURN_IF_ERROR(decoder.Init(latent_image_width, latent_image_height, &env,
                               weights_path));

  OpHolder copier;
  RETURN_IF_ERROR(copier.InitElementwiseOneInput(
      OperationType::COPY, default_desc, default_desc, &env));

  ml_drift::Tensor<BHWC, DataType::INT32> tokens;
  tokens.shape = BHWC(1, 1, 2, 77);
  tokens.data.resize(tokens.shape.DimensionsProduct());

  SpatialTensor latent_copy;
  {
    SpatialTensor* unet_latent = unet.GetLatentTensor();
    TensorDescriptor descriptor_with_shape = unet_latent->GetDescriptor();
    descriptor_with_shape.SetBHWCShape(
        BHWC(unet_latent->Batch(), unet_latent->Height(), unet_latent->Width(),
             unet_latent->Channels()));
    RETURN_IF_ERROR(
        CreateTensor(env.device(), descriptor_with_shape, &latent_copy));
  }

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
    RETURN_IF_ERROR(text_guidance_graph.SetInput(&env, tokens));

    const auto p1 = std::chrono::high_resolution_clock::now();

    {
      RETURN_IF_ERROR(text_guidance_graph.Execute(&env));
    }

    {
      int seed = rand_r(&base_seed);
      OpHolder::ExecutionParams exec_params;
      exec_params.dst = {unet.GetLatentTensor()};
      exec_params.int_params = {{"seed", seed}};
      ComputePassHelper helper(&env);
      RETURN_IF_ERROR(
          noise_op.Encode(helper.compute_pass_encoder(), &env, exec_params));
      helper.Submit();
    }

    {
      ComputePassHelper helper(&env);
      RETURN_IF_ERROR(copier.Encode(helper.compute_pass_encoder(), &env,
                                    text_guidance_graph.GetGuidanceTensor(),
                                    unet.GetGuidanceTensor()));
      helper.Submit();
    }

    const int stride = 1000 / steps;
    for (int t = steps - 1; t >= 0; --t) {
      int ts = t * stride + 1;
      int tsPrev = ts - stride;

      {
        OpHolder::ExecutionParams exec_params;
        exec_params.dst = {unet.GetTembTensor()};
        exec_params.float_params = {{"index_val", ts}};
        ComputePassHelper helper(&env);
        RETURN_IF_ERROR(temb_generation_op.Encode(helper.compute_pass_encoder(),
                                                  &env, exec_params));
        helper.Submit();
      }

      {
        ComputePassHelper helper(&env);
        RETURN_IF_ERROR(copier.Encode(helper.compute_pass_encoder(), &env,
                                      unet.GetLatentTensor(), &latent_copy));
        helper.Submit();
      }

      {
        RETURN_IF_ERROR(unet.Execute(&env));
      }

      float guidance_scale = 7.5f;
      {
        ComputePassHelper helper(&env);
        RETURN_IF_ERROR(diffusion.StepCustomOp(
            helper.compute_pass_encoder(), &env, &latent_copy,
            unet.GetEtaUncondTensor(), unet.GetEtaCondTensor(),
            unet.GetLatentTensor(), ts, tsPrev, guidance_scale));
        helper.Submit();
      }
      std::cout << "step " << steps - t << "/" << steps << std::endl;
    }  // loop t

    {
      ComputePassHelper helper(&env);
      RETURN_IF_ERROR(copier.Encode(helper.compute_pass_encoder(), &env,
                                    unet.GetLatentTensor(),
                                    decoder.GetInputTensor()));
      helper.Submit();
    }

    {
      RETURN_IF_ERROR(decoder.Execute(&env));
    }
    RETURN_IF_ERROR(WaitUntilCompleted(&env));  // Required sync.

    const auto p2 = std::chrono::high_resolution_clock::now();
    TensorFloat32 result;
    RETURN_IF_ERROR(decoder.GetOutput(&env, &result));
    const auto p3 = std::chrono::high_resolution_clock::now();

    std::cout << "Cpu begin time: " << (p1 - p0).count() * 1e-6f << " ms."
              << std::endl;

    std::cout << "Gpu time: " << (p2 - p1).count() * 1e-6f << " ms."
              << std::endl;

    std::cout << "Cpu end time: " << (p3 - p2).count() * 1e-6f << " ms."
              << std::endl;

    std::cout << "Total time: " << (p3 - p0).count() * 1e-6f << " ms."
              << std::endl;

    GenerateImage(result, "result.bmp");
    std::cout << "ready" << std::endl;
  }

  return absl::OkStatus();
}

}  // namespace
}  // namespace webgpu
}  // namespace ml_drift

int main(int argc, char** argv) {
  auto status = ml_drift::webgpu::TestStableDiffusion("sd_1_5/");
  if (!status.ok()) {
    std::cout << "Failed test." << status.message() << std::endl;
    return -1;
  }
  return 0;
}
