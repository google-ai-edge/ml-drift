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

#include <chrono>  // NOLINT(build/c++11)
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/time/time.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/kernels/cast.h"
#include "ml_drift/common/kernels/random_philox.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/model_hints.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"
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
  return TensorStorageType::kTexture2D;
}

// Helper class for running the one-off ops
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

class NoiseGenerator {
 public:
  absl::Status Init(Environment* env, const TensorDescriptor& dst_desc) {
    OperationDef op_def;
    op_def.dst_tensors.push_back(dst_desc);

    auto gpu_op = std::make_unique<GPUOperation>(
        CreateRandomNormalPhilox(env->GetInfo(), op_def));
    ABSL_RETURN_IF_ERROR(gpu_op->AssembleCode(env->GetInfo()));

    ABSL_RETURN_IF_ERROR(webgpu_op_.Init(*env, std::move(gpu_op)));
    ABSL_RETURN_IF_ERROR(webgpu_op_.Compile(*env));
    return absl::OkStatus();
  }

  absl::Status Execute(Environment* env, uint32_t seed, SpatialTensor* dst) {
    ABSL_RETURN_IF_ERROR(webgpu_op_.SetInt("seed", seed));
    ABSL_RETURN_IF_ERROR(webgpu_op_.SetInt("seed2", seed));
    ABSL_RETURN_IF_ERROR(webgpu_op_.SetDstTensor(0, dst));
    ABSL_RETURN_IF_ERROR(webgpu_op_.Update(env->device()));
    webgpu_op_.UpdateGpuObjectBindings(env->device());

    return webgpu_op_.Execute(*env);
  }

 private:
  ComputeTask webgpu_op_;
};

class Copier {
 public:
  absl::Status Init(const TensorDescriptor& src_desc,
                    const TensorDescriptor& dst_desc, Environment* env) {
    OperationDef op_def;
    op_def.src_tensors.push_back(src_desc);
    op_def.dst_tensors.push_back(dst_desc);

    auto gpu_op =
        std::make_unique<GPUOperation>(CreateCast(op_def, env->GetInfo()));
    ABSL_RETURN_IF_ERROR(gpu_op->AssembleCode(env->GetInfo()));

    ABSL_RETURN_IF_ERROR(webgpu_op_.Init(*env, std::move(gpu_op)));
    ABSL_RETURN_IF_ERROR(webgpu_op_.Compile(*env));
    return absl::OkStatus();
  }

  absl::Status Execute(Environment* env, SpatialTensor* src,
                       SpatialTensor* dst) {
    ABSL_RETURN_IF_ERROR(webgpu_op_.SetSrcTensor(0, src));
    ABSL_RETURN_IF_ERROR(webgpu_op_.SetDstTensor(0, dst));
    ABSL_RETURN_IF_ERROR(webgpu_op_.Update(env->device()));
    webgpu_op_.UpdateGpuObjectBindings(env->device());

    return webgpu_op_.Execute(*env);
  }

 private:
  ComputeTask webgpu_op_;
};

class TextGuidance {
 public:
  absl::Status Init(Environment* env, const std::string& file_folder) {
    const auto& gpu_info = env->GetInfo();

    CreateGpuModelInfo create_info;
    create_info.precision = CalculationsPrecision::kF16;
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
    ABSL_RETURN_IF_ERROR(builder.Build(config, gpu_info, create_info,
                                       &gpu_model, &src_, /*mask_ptr=*/nullptr,
                                       &dst_));
    const auto start_init = std::chrono::high_resolution_clock::now();

    ABSL_RETURN_IF_ERROR(
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
                        const ml_drift::Tensor<BHWC, DataType::kInt32>& src) {
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
    create_info.precision = CalculationsPrecision::kF16;
    create_info.storage_type = GetFastestStorageType(gpu_info);
    create_info.hints.Add(ModelHints::kFastTuning);
    create_info.hints.Add(ModelHints::kNoWinogradOptimizations);
    GpuModel gpu_model;
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
    GpuModelBuilder gpu_builder(gpu_info, create_info.hints,
                                create_info.precision,
                                create_info.storage_type);

    index_val_ = gpu_builder.AddTensor(BHWC(1, 1, 1, 1), DataType::kFloat32);
    ABSL_ASSIGN_OR_RETURN(
        auto temb_res,
        gpu_builder.AppendOp(
            "TembGeneration", {index_val_},
            {{"channels", 320},
             {"storage_type",
              static_cast<int>(GetFastestStorageType(gpu_info))}}));
    temb_ = temb_res[0];

    UnetBuilder builder;
    ABSL_RETURN_IF_ERROR(builder.Build(
        config, gpu_info, create_info, width, height, &gpu_builder, &src_,
        &temb_, &guidance_,
        /*text_proj_ptr=*/nullptr,
        /*masked_image_latent_ptr=*/nullptr, &eta0_, &eta1_, nullptr));

    guidance_scale_ =
        gpu_builder.AddTensor(BHWC(1, 1, 1, 4), DataType::kFloat32);
    sqrt_alpha_ = gpu_builder.AddTensor(BHWC(1, 1, 1, 4), DataType::kFloat32);
    sqrt_alpha_prev_ =
        gpu_builder.AddTensor(BHWC(1, 1, 1, 4), DataType::kFloat32);
    sqrt_one_minus_alpha_ =
        gpu_builder.AddTensor(BHWC(1, 1, 1, 4), DataType::kFloat32);
    sqrt_one_minus_alpha_prev_ =
        gpu_builder.AddTensor(BHWC(1, 1, 1, 4), DataType::kFloat32);
    src_input_ = src_;
    ABSL_ASSIGN_OR_RETURN(
        auto _step_res,
        gpu_builder.AppendOp(
            "DiffusionStep",
            {src_, eta0_, eta1_, guidance_scale_, sqrt_alpha_, sqrt_alpha_prev_,
             sqrt_one_minus_alpha_, sqrt_one_minus_alpha_prev_}));
    src_ = _step_res[0];

    std::vector<ValueId> in_ids = {src_input_.id,
                                   index_val_.id,
                                   guidance_.id,
                                   guidance_scale_.id,
                                   sqrt_alpha_.id,
                                   sqrt_alpha_prev_.id,
                                   sqrt_one_minus_alpha_.id,
                                   sqrt_one_minus_alpha_prev_.id};
    std::vector<ValueId> out_ids = {src_.id, eta0_.id, eta1_.id};
    ABSL_RETURN_IF_ERROR(gpu_builder.GetGpuModel(in_ids, out_ids, &gpu_model));

    const auto start_init = std::chrono::high_resolution_clock::now();
    ABSL_RETURN_IF_ERROR(
        inference_context_.InitFromGpuModel(*env, create_info, &gpu_model));
    const auto end_init = std::chrono::high_resolution_clock::now();
    std::cout << "UNet initialization time: "
              << (end_init - start_init).count() * 1e-6f << " ms." << std::endl;

    return absl::OkStatus();
  }

  SpatialTensor* GetLatentTensor() {
    return inference_context_.GetTensor(src_input_.id);
  }
  SpatialTensor* GetOutputLatentTensor() {
    return inference_context_.GetTensor(src_.id);
  }
  SpatialTensor* GetInputLatentTensor() {
    return inference_context_.GetTensor(src_input_.id);
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

  absl::Status Execute(Environment* env, int step_index, float guidance_scale,
                       float sqrt_alpha, float sqrt_alpha_prev,
                       float sqrt_one_minus_alpha,
                       float sqrt_one_minus_alpha_prev) {
    TensorFloat32 scalar_tensor;
    scalar_tensor.shape = BHWC(1, 1, 1, 4);
    scalar_tensor.data.resize(4, 0.0f);
    scalar_tensor.data[0] = static_cast<float>(step_index);
    ABSL_RETURN_IF_ERROR(
        inference_context_.SetInputTensor(*env, index_val_.id, scalar_tensor));
    scalar_tensor.data[0] = guidance_scale;
    ABSL_RETURN_IF_ERROR(inference_context_.SetInputTensor(
        *env, guidance_scale_.id, scalar_tensor));
    scalar_tensor.data[0] = sqrt_alpha;
    ABSL_RETURN_IF_ERROR(
        inference_context_.SetInputTensor(*env, sqrt_alpha_.id, scalar_tensor));
    scalar_tensor.data[0] = sqrt_alpha_prev;
    ABSL_RETURN_IF_ERROR(inference_context_.SetInputTensor(
        *env, sqrt_alpha_prev_.id, scalar_tensor));
    scalar_tensor.data[0] = sqrt_one_minus_alpha;
    ABSL_RETURN_IF_ERROR(inference_context_.SetInputTensor(
        *env, sqrt_one_minus_alpha_.id, scalar_tensor));
    scalar_tensor.data[0] = sqrt_one_minus_alpha_prev;
    ABSL_RETURN_IF_ERROR(inference_context_.SetInputTensor(
        *env, sqrt_one_minus_alpha_prev_.id, scalar_tensor));
    return inference_context_.Execute(*env);
  }

 private:
  GpuModelBuilder::TensorHandle src_, src_input_, temb_, guidance_, eta0_,
      eta1_;
  GpuModelBuilder::TensorHandle index_val_, guidance_scale_;
  GpuModelBuilder::TensorHandle sqrt_alpha_, sqrt_alpha_prev_,
      sqrt_one_minus_alpha_, sqrt_one_minus_alpha_prev_;
  InferenceContext inference_context_;
};

class Decoder {
 public:
  absl::Status Init(int width, int height, Environment* env,
                    const std::string& file_folder) {
    const auto& gpu_info = env->GetInfo();

    CreateGpuModelInfo create_info;
    create_info.precision = CalculationsPrecision::kF16;
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
    ABSL_RETURN_IF_ERROR(builder.BuildDecoder(config, gpu_info, create_info,
                                              width, height, &gpu_model, &src_,
                                              &dst_, file_folder));
    const auto start_init = std::chrono::high_resolution_clock::now();

    ABSL_RETURN_IF_ERROR(
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

absl::Status RunStableDiffusion(std::string weights_path) {
  BPETokenizer bpe_tokenizer;
  bpe_tokenizer.Init(weights_path);

  Environment env;
  env.RequestExtension("shader_f16");
  ABSL_RETURN_IF_ERROR(env.Initialize());
  const auto& gpu_info = env.GetInfo();

  TensorDescriptor default_desc(DataType::kFloat16,
                                GetFastestStorageType(gpu_info), Layout::kHWC);

  const int final_image_width = 512;
  const int final_image_height = 512;
  const int latent_image_width = final_image_width / 8;
  const int latent_image_height = final_image_height / 8;

  TensorDescriptor latent_desc = default_desc;
  latent_desc.SetBHWCShape(
      ml_drift::BHWC(1, latent_image_height, latent_image_width, 4));

  NoiseGenerator noise_gen;
  ABSL_RETURN_IF_ERROR(noise_gen.Init(&env, latent_desc));

  TextGuidance text_guidance_graph;
  ABSL_RETURN_IF_ERROR(text_guidance_graph.Init(&env, weights_path));

  UNet unet;
  ABSL_RETURN_IF_ERROR(
      unet.Init(latent_image_width, latent_image_height, &env, weights_path));

  Decoder decoder;
  ABSL_RETURN_IF_ERROR(decoder.Init(latent_image_width, latent_image_height,
                                    &env, weights_path));

  auto* text_g = text_guidance_graph.GetGuidanceTensor();
  auto* unet_g = unet.GetGuidanceTensor();
  auto* unet_out = unet.GetOutputLatentTensor();
  auto* unet_in = unet.GetInputLatentTensor();

  Copier copier_guidance;
  ABSL_RETURN_IF_ERROR(copier_guidance.Init(text_g->GetDescriptor(),
                                            unet_g->GetDescriptor(), &env));

  Copier copier_latent;
  ABSL_RETURN_IF_ERROR(copier_latent.Init(unet_out->GetDescriptor(),
                                          unet_in->GetDescriptor(), &env));

  ml_drift::Tensor<BHWC, DataType::kInt32> tokens;
  tokens.shape = BHWC(1, 1, 2, 77);
  tokens.data.resize(tokens.shape.DimensionsProduct());


  unsigned int base_seed = 0;

  while (true) {
    std::string prompt = "a photo of an astronaut riding a horse on mars";
    std::string negative_prompt = "";
    std::cout << "Enter phrase: ";
    if (!std::getline(std::cin, prompt)) break;
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
    ABSL_RETURN_IF_ERROR(text_guidance_graph.SetInput(&env, tokens));

    const auto p1 = std::chrono::high_resolution_clock::now();

    ABSL_RETURN_IF_ERROR(text_guidance_graph.Execute(&env));

    int seed = rand_r(&base_seed);
    ABSL_RETURN_IF_ERROR(
        noise_gen.Execute(&env, seed, unet.GetInputLatentTensor()));
    ABSL_RETURN_IF_ERROR(
        copier_guidance.Execute(&env, text_guidance_graph.GetGuidanceTensor(),
                                unet.GetGuidanceTensor()));

    auto alphas_cumprod = LoadF16(weights_path + "alphas_cumprod.bin", 1000);

    const int stride = 1000 / steps;
    for (int t = steps - 1; t >= 0; --t) {
      int ts = t * stride + 1;
      int tsPrev = ts - stride;

      float guidance_scale = 7.5f;
      float alpha_cumprod = static_cast<float>(alphas_cumprod[ts]);
      float alpha_cumprod_prev =
          tsPrev >= 0 ? static_cast<float>(alphas_cumprod[tsPrev]) : 1.0f;
      float sqrt_alpha = std::sqrt(alpha_cumprod);
      float sqrt_alpha_prev = std::sqrt(alpha_cumprod_prev);
      float sqrt_one_minus_alpha = std::sqrt(1.0f - alpha_cumprod);
      float sqrt_one_minus_alpha_prev = std::sqrt(1.0f - alpha_cumprod_prev);

      ABSL_RETURN_IF_ERROR(unet.Execute(&env, ts, guidance_scale, sqrt_alpha,
                                        sqrt_alpha_prev, sqrt_one_minus_alpha,
                                        sqrt_one_minus_alpha_prev));

      ABSL_RETURN_IF_ERROR(copier_latent.Execute(
          &env, unet.GetOutputLatentTensor(), unet.GetInputLatentTensor()));
      std::cout << "step " << steps - t << "/" << steps << std::endl;
    }

    ABSL_RETURN_IF_ERROR(copier_latent.Execute(
        &env, unet.GetOutputLatentTensor(), decoder.GetInputTensor()));

    ABSL_RETURN_IF_ERROR(decoder.Execute(&env));
    ABSL_RETURN_IF_ERROR(WaitUntilCompleted(&env));  // Required sync.

    const auto p2 = std::chrono::high_resolution_clock::now();
    TensorFloat32 result;
    ABSL_RETURN_IF_ERROR(decoder.GetOutput(&env, &result));
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
  std::string weights_folder = "sd_1_5/";
  if (argc >= 2) {
    weights_folder = argv[1];
    if (!weights_folder.empty() && weights_folder.back() != '/') {
      weights_folder += '/';
    }
  }
  auto status = ml_drift::webgpu::RunStableDiffusion(weights_folder);
  if (!status.ok()) {
    std::cout << "Failed test." << status.message() << std::endl;
    return -1;
  }
  return 0;
}
