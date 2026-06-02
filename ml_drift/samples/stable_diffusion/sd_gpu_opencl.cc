// Copyright 2023 The ML Drift Authors.
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
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "ml_drift/cl/cl_command_queue.h"
#include "ml_drift/cl/cl_operation.h"
#include "ml_drift/cl/environment.h"
#include "ml_drift/cl/inference_context.h"
#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/cl/tensor.h"
#include "ml_drift/common/data_type.h"
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
#include "ml_drift/common/task/profiling_info.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/samples/stable_diffusion/autoencoder_kl_builder.h"
#include "ml_drift/samples/stable_diffusion/bpe_tokenizer.h"
#include "ml_drift/samples/stable_diffusion/text_guidance_builder.h"
#include "ml_drift/samples/stable_diffusion/unet_builder.h"
#include "ml_drift/samples/stable_diffusion/util.h"

namespace ml_drift {
namespace cl {
namespace {

class OpHolder {
 public:
  struct ExecutionParams {
    std::vector<Tensor*> src;
    std::vector<Tensor*> dst;
    std::map<std::string, float> float_params;
    std::map<std::string, half> half_params;
    std::map<std::string, int> int_params;
  };

  absl::Status Init(Environment* env, GPUOperation&& operation) {
    return InitClOp(env, std::move(operation));
  }

  absl::Status InitElementwiseOneInput(OperationType op_type,
                                       const TensorDescriptor& src_desc,
                                       const TensorDescriptor& dst_desc,
                                       Environment* env) {
    OperationDef op_def;
    op_def.src_tensors.push_back(src_desc);
    op_def.dst_tensors.push_back(dst_desc);
    GPUOperation operation =
        CreateElementwiseOneInput(env->device().GetInfo(), op_def, op_type);
    return InitClOp(env, std::move(operation));
  }

  absl::Status Execute(CLCommandQueue* queue, float scalar, Tensor* src,
                       Tensor* dst) {
    RETURN_IF_ERROR(cl_op_.SetSrcTensor(0, src));
    RETURN_IF_ERROR(cl_op_.SetDstTensor(0, dst));
    RETURN_IF_ERROR(cl_op_.SetHalf("scalar", scalar));
    RETURN_IF_ERROR(cl_op_.UpdateParams());
    RETURN_IF_ERROR(cl_op_.AddToQueue(queue));
    return absl::OkStatus();
  }

  absl::Status Execute(CLCommandQueue* queue, const ExecutionParams& params) {
    for (int i = 0; i < params.src.size(); ++i) {
      RETURN_IF_ERROR(cl_op_.SetSrcTensor(i, params.src[i]));
    }
    for (int i = 0; i < params.dst.size(); ++i) {
      RETURN_IF_ERROR(cl_op_.SetDstTensor(i, params.dst[i]));
    }
    for (const auto& float_param : params.float_params) {
      RETURN_IF_ERROR(cl_op_.SetFloat(float_param.first, float_param.second));
    }
    for (const auto& half_param : params.half_params) {
      RETURN_IF_ERROR(cl_op_.SetHalf(half_param.first, half_param.second));
    }
    for (const auto& int_param : params.int_params) {
      RETURN_IF_ERROR(cl_op_.SetInt(int_param.first, int_param.second));
    }
    RETURN_IF_ERROR(cl_op_.UpdateParams());
    RETURN_IF_ERROR(cl_op_.AddToQueue(queue));
    return absl::OkStatus();
  }

  absl::Status Execute(CLCommandQueue* queue, Tensor* src, Tensor* dst) {
    RETURN_IF_ERROR(cl_op_.SetSrcTensor(0, src));
    RETURN_IF_ERROR(cl_op_.SetDstTensor(0, dst));
    RETURN_IF_ERROR(cl_op_.UpdateParams());
    RETURN_IF_ERROR(cl_op_.AddToQueue(queue));
    return absl::OkStatus();
  }

  absl::Status Execute(CLCommandQueue* queue, Tensor* src0, Tensor* src1,
                       Tensor* dst) {
    RETURN_IF_ERROR(cl_op_.SetSrcTensor(0, src0));
    RETURN_IF_ERROR(cl_op_.SetSrcTensor(1, src1));
    RETURN_IF_ERROR(cl_op_.SetDstTensor(0, dst));
    RETURN_IF_ERROR(cl_op_.UpdateParams());
    RETURN_IF_ERROR(cl_op_.AddToQueue(queue));
    return absl::OkStatus();
  }

 private:
  absl::Status InitClOp(Environment* env, GPUOperation&& operation) {
    auto gpu_op = std::make_unique<GPUOperation>(std::move(operation));

    RETURN_IF_ERROR(gpu_op->AssembleCode(env->device().GetInfo()));

    cl_op_.Init(std::move(gpu_op));
    {
      CreationContext creation_context;
      creation_context.device = env->GetDevicePtr();
      creation_context.context = &env->context();
      creation_context.queue = env->queue();
      creation_context.cache = env->program_cache();
      RETURN_IF_ERROR(cl_op_.Compile(creation_context));
    }
    return absl::OkStatus();
  }
  ClOperation cl_op_;
};

class DiffusionStepper {
 public:
  absl::Status Init(Environment* env,
                    const std::string& file_folder) {
    alphas_ = LoadF16(file_folder + "alphas_cumprod.bin", 1000);
    alphas_prev_ = alphas_;
    alphas_prev_.insert(alphas_prev_.begin(), half(1.0f));

    TensorDescriptor desc(DataType::FLOAT16,
                          GetFastestStorageType(env->device().GetInfo()),
                          Layout::HWC);

    RETURN_IF_ERROR(
        custom_op_.Init(env, CreateDiffusionStepOp(desc, desc, desc, desc)));
    return absl::OkStatus();
  }
  absl::Status StepCustomOp(CLCommandQueue* queue, Tensor* xIn,
                            Tensor* etaUncondIn, Tensor* etaCondIn, Tensor* dst,
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
    return custom_op_.Execute(queue, exec_params);
  }

 private:
  OpHolder custom_op_;
  std::vector<half> alphas_;
  std::vector<half> alphas_prev_;
};

void PrintMemoryUsage(const InferenceContext& context) {
  const uint64_t runtime_mem_bytes =
      context.GetSizeOfMemoryAllocatedForIntermediateTensors();
  std::cout << "Memory for intermediate tensors - "
            << runtime_mem_bytes / 1024.0 / 1024.0 << " MB" << std::endl;
  const uint64_t const_mem_bytes = context.GetConstantTensorsSize();
  std::cout << "Memory for constant tensors - "
            << const_mem_bytes / 1024.0 / 1024.0 << " MB" << std::endl;
  std::cout << "Total tensors memory(const + intermediate) - "
            << (const_mem_bytes + runtime_mem_bytes) / 1024.0 / 1024.0 << " MB"
            << std::endl;
}

void PrintProfilingInfo(InferenceContext& context,
                        ProfilingCommandQueue* queue) {
  ProfilingInfo profiling_info;
  if (!context.Profile(queue, &profiling_info).ok()) {
    std::cout << "Profiling failed." << std::endl;
    return;
  }

  ProfilingInfo::DetailedReportOptions profiling_options = {.add_shapes_info =
                                                                false};

  std::cout << profiling_info.GetDetailedReport(profiling_options) << std::endl;
}

class TextGuidance {
 public:
  absl::Status Init(Environment* env,
                    const std::string& file_folder) {
    const auto& gpu_info = env->GetDevicePtr()->GetInfo();

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
    RETURN_IF_ERROR(builder.Build(config, gpu_info, create_info, &gpu_model,
                                  &src_, /*mask_ptr=*/nullptr, &dst_));

    const auto start_init = std::chrono::high_resolution_clock::now();
    RETURN_IF_ERROR(inference_context_.InitFromGpuModel(create_info, &gpu_model,
                                                        env, nullptr));
    const auto end_init = std::chrono::high_resolution_clock::now();
    std::cout << "TextGuidance initialization time: "
              << (end_init - start_init).count() * 1e-6f << " ms." << std::endl;

    if (/* DISABLES CODE */ (false)) {
      PrintProfilingInfo(inference_context_, env->profiling_queue());
    }
    PrintMemoryUsage(inference_context_);

    return absl::OkStatus();
  }

  Tensor* GetGuidanceTensor() { return inference_context_.GetTensor(dst_.id); }

  absl::Status SetInput(Environment* env,
                        const ml_drift::Tensor<BHWC, DataType::INT32>& src) {
    return inference_context_.SetInputTensor(src_.id, src, env->queue());
  }
  absl::Status Execute(CLCommandQueue* queue) {
    return inference_context_.AddToQueue(queue);
  }

 private:
  GpuModelBuilder::TensorHandle src_, dst_;
  InferenceContext inference_context_;
};

class UNet {
 public:
  absl::Status Init(int width, int height, Environment* env,
                    const std::string& file_folder) {
    const auto& gpu_info = env->GetDevicePtr()->GetInfo();

    CreateGpuModelInfo create_info;
    create_info.precision = CalculationsPrecision::F16;
    create_info.storage_type = GetFastestStorageType(gpu_info);
    create_info.hints.Add(ModelHints::kFastTuning);
    create_info.hints.winograd_runtime_weights_conversion = false;
    // Comment out for potentially faster performance, but more memory usage.
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
    RETURN_IF_ERROR(builder.Build(config, gpu_info, create_info, width, height,
                                  &gpu_model, &src_, &temb_, &guidance_,
                                  /*text_proj_ptr=*/nullptr,
                                  /*masked_image_latent_ptr=*/nullptr, &eta0_,
                                  &eta1_, nullptr));

    const auto start_init = std::chrono::high_resolution_clock::now();
    RETURN_IF_ERROR(inference_context_.InitFromGpuModel(create_info, &gpu_model,
                                                        env, nullptr));
    const auto end_init = std::chrono::high_resolution_clock::now();
    std::cout << "UNet initialization time: "
              << (end_init - start_init).count() * 1e-6f << " ms." << std::endl;

    if (/* DISABLES CODE */ (false)) {
      PrintProfilingInfo(inference_context_, env->profiling_queue());
    }
    PrintMemoryUsage(inference_context_);

    return absl::OkStatus();
  }

  Tensor* GetLatentTensor() { return inference_context_.GetTensor(src_.id); }
  Tensor* GetTembTensor() { return inference_context_.GetTensor(temb_.id); }
  Tensor* GetGuidanceTensor() {
    return inference_context_.GetTensor(guidance_.id);
  }
  Tensor* GetEtaUncondTensor() {
    return inference_context_.GetTensor(eta0_.id);
  }
  Tensor* GetEtaCondTensor() { return inference_context_.GetTensor(eta1_.id); }

  absl::Status Execute(CLCommandQueue* queue) {
    return inference_context_.AddToQueue(queue);
  }

 private:
  GpuModelBuilder::TensorHandle src_, temb_, guidance_, eta0_, eta1_;
  InferenceContext inference_context_;
};



class Decoder {
 public:
  absl::Status Init(int width, int height, Environment* env,
                    const std::string& file_folder) {
    const auto& gpu_info = env->GetDevicePtr()->GetInfo();

    CreateGpuModelInfo create_info;
    create_info.precision = CalculationsPrecision::F16;
    create_info.storage_type = GetFastestStorageType(gpu_info);
    create_info.hints.Add(ModelHints::kFastTuning);
    create_info.hints.winograd_runtime_weights_conversion = false;
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
    RETURN_IF_ERROR(builder.BuildDecoder(config, gpu_info, create_info, width,
                                         height, &gpu_model, &src_, &dst_,
                                         file_folder));

    const auto start_init = std::chrono::high_resolution_clock::now();
    RETURN_IF_ERROR(inference_context_.InitFromGpuModel(create_info, &gpu_model,
                                                        env, nullptr));
    const auto end_init = std::chrono::high_resolution_clock::now();
    std::cout << "Decoder initialization time: "
              << (end_init - start_init).count() * 1e-6f << " ms." << std::endl;

    if (/* DISABLES CODE */ (false)) {
      PrintProfilingInfo(inference_context_, env->profiling_queue());
    }
    PrintMemoryUsage(inference_context_);

    return absl::OkStatus();
  }

  Tensor* GetInputTensor() { return inference_context_.GetTensor(src_.id); }

  absl::Status GetOutput(Environment* env, TensorFloat32* dst) {
    return inference_context_.GetOutputTensor(dst_.id, env->queue(), dst);
  }

  absl::Status Execute(CLCommandQueue* queue) {
    return inference_context_.AddToQueue(queue);
  }

 private:
  GpuModelBuilder::TensorHandle src_, dst_;
  InferenceContext inference_context_;
};

absl::Status TestStableDiffusion(std::string weights_path) {
  BPETokenizer bpe_tokenizer;
  bpe_tokenizer.Init(weights_path);

  Environment env;
  RETURN_IF_ERROR(CreateEnvironment(&env));
  const auto& gpu_info = env.GetDevicePtr()->GetInfo();

  TensorDescriptor default_desc(DataType::FLOAT16,
                                GetFastestStorageType(gpu_info), Layout::HWC);

  const int final_image_width = 512;
  const int final_image_height = 512;

  const int latent_image_width = final_image_width / 8;
  const int latent_image_height = final_image_height / 8;

  // For some strange reason, on Adreno 8xx, if noise_op initialized first,
  // performance of unet/decoder is much worse. Difference is significant,
  // ~ + 50% in total latency.
  UNet unet;
  RETURN_IF_ERROR(
      unet.Init(latent_image_width, latent_image_height, &env, weights_path));

  Decoder decoder;
  RETURN_IF_ERROR(decoder.Init(latent_image_width, latent_image_height, &env,
                               weights_path));

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

  OpHolder copier;
  RETURN_IF_ERROR(copier.InitElementwiseOneInput(
      OperationType::COPY, default_desc, default_desc, &env));

  ml_drift::Tensor<BHWC, DataType::INT32> tokens;
  tokens.shape = BHWC(1, 1, 2, 77);
  tokens.data.resize(tokens.shape.DimensionsProduct());

  Tensor latent_copy;
  {
    Tensor* unet_latent = unet.GetLatentTensor();
    TensorDescriptor descriptor_with_shape = unet_latent->GetDescriptor();
    descriptor_with_shape.SetBHWCShape(
        BHWC(unet_latent->Batch(), unet_latent->Height(), unet_latent->Width(),
             unet_latent->Channels()));
    RETURN_IF_ERROR(
        CreateTensor(env.context(), descriptor_with_shape, &latent_copy));
  }

  unsigned int base_seed = 1;
  std::srand(base_seed);

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

    RETURN_IF_ERROR(text_guidance_graph.Execute(env.queue()));

    {
      int seed = std::rand();
      OpHolder::ExecutionParams exec_params;
      exec_params.dst = {unet.GetLatentTensor()};
      exec_params.int_params = {{"seed", seed}};
      RETURN_IF_ERROR(noise_op.Execute(env.queue(), exec_params));
    }

    RETURN_IF_ERROR(copier.Execute(env.queue(),
                                   text_guidance_graph.GetGuidanceTensor(),
                                   unet.GetGuidanceTensor()));

    const int stride = 1000 / steps;
    for (int t = steps - 1; t >= 0; --t) {
      int ts = t * stride + 1;
      int tsPrev = ts - stride;

      {
        OpHolder::ExecutionParams exec_params;
        exec_params.dst = {unet.GetTembTensor()};
        exec_params.float_params = {{"index_val", ts}};
        RETURN_IF_ERROR(temb_generation_op.Execute(env.queue(), exec_params));
      }

      RETURN_IF_ERROR(
          copier.Execute(env.queue(), unet.GetLatentTensor(), &latent_copy));

      RETURN_IF_ERROR(unet.Execute(env.queue()));

      float guidance_scale = 7.5f;
      RETURN_IF_ERROR(diffusion.StepCustomOp(
          env.queue(), &latent_copy, unet.GetEtaUncondTensor(),
          unet.GetEtaCondTensor(), unet.GetLatentTensor(), ts, tsPrev,
          guidance_scale));
      std::cout << "step " << steps - t << "/" << steps << std::endl;
    }  // loop t

    RETURN_IF_ERROR(copier.Execute(env.queue(), unet.GetLatentTensor(),
                                   decoder.GetInputTensor()));

    RETURN_IF_ERROR(decoder.Execute(env.queue()));
    RETURN_IF_ERROR(env.queue()->WaitForCompletion());  // Required sync.

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
}  // namespace cl
}  // namespace ml_drift

int main(int argc, char** argv) {
  auto load_status = ml_drift::cl::LoadOpenCL();
  if (!load_status.ok()) {
    std::cout << load_status.message();
    return -1;
  }

  auto status = ml_drift::cl::TestStableDiffusion("sd_1_5/");
  if (!status.ok()) {
    std::cout << "Failed test." << status.message() << std::endl;
    return -1;
  }
  return 0;
}
