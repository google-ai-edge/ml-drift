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

#include "ml_drift/samples/stable_diffusion/diffuser.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/cl/cl_command_queue.h"
#include "ml_drift/cl/cl_operation.h"
#include "ml_drift/cl/environment.h"
#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/cl/tensor.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_model.h"
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

namespace ml_drift {
namespace cl {
namespace stable_diffusion {
namespace {

constexpr std::array<int, 9> kDistilledModelTimesteps = {
    999, 871, 743, 615, 487, 359, 231, 103, 0,
};

constexpr std::array<int, 5> kTigoFourTimesteps = {999, 925, 800, 650, 450};

absl::Status SanityCheck(const Diffuser::Config& config) {
  if (config.image_width != config.image_height ||
      config.image_width % 8 != 0 || config.image_height % 8 != 0 ||
      config.image_height <= 0 || config.image_width <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid output image width/height: ", config.image_width,
                     "/", config.image_height));
  }
  if (config.run_unet_with_plugins) {
    if (config.image_width != 512) {
      return absl::InvalidArgumentError(
          "Diffusion plugins mode must use output image width/height: 512.");
    }
  }
  if (!config.lora_dir.empty() && !config.lora_weights_layer_mapping.empty()) {
    return absl::InvalidArgumentError(
        "Only one of lora_dir and lora_weights_layer_mapping should "
        "be set.");
  }
  return absl::OkStatus();
}

bool IsGldmOrTigo(Diffuser::ModelType model_type) {
  return model_type == Diffuser::ModelType::kGldm ||
         model_type == Diffuser::ModelType::kDistilledGldm ||
         model_type == Diffuser::ModelType::kTigo ||
         model_type == Diffuser::ModelType::kTigoUfo;
}

bool IsTigo(Diffuser::ModelType model_type) {
  return model_type == Diffuser::ModelType::kTigo ||
         model_type == Diffuser::ModelType::kTigoUfo;
}

int GetRandomInt(uint32_t size, uint32_t seed) {
  std::default_random_engine generator;
  generator.seed(seed);
  std::uniform_int_distribution<> distribution(0, size - 1);
  return distribution(generator);
}

}  // namespace

absl::StatusOr<std::unique_ptr<Diffuser>> Diffuser::Create(
    const Diffuser::Config& config) {
  RETURN_IF_ERROR(SanityCheck(config));
  auto load_status = LoadOpenCL();
  if (!load_status.ok()) {
    return absl::FailedPreconditionError(load_status.message());
  }
  auto env = std::make_unique<Environment>();
  RETURN_IF_ERROR(CreateEnvironment(env.get(), config.env_options));
  ABSL_LOG(INFO) << "Environment created";

  const auto& gpu_info = env->GetDevicePtr()->GetInfo();
  TensorDescriptor default_desc(DataType::FLOAT16,
                                GetFastestStorageType(gpu_info), Layout::HWC);

  auto text_guidance_graph = std::make_unique<TextGuidance>();
  RETURN_IF_ERROR(text_guidance_graph->Init(config, env.get()));
  ABSL_LOG(INFO) << "TextGuidance is created";

  auto copier = std::make_unique<Diffuser::OpHolder>();
  RETURN_IF_ERROR(copier->InitElementwiseOneInput(
      OperationType::COPY, default_desc, default_desc, env.get()));
  ABSL_LOG(INFO) << "Copier is created";

  int latent_image_width = config.image_width / 8;
  int latent_image_height = config.image_height / 8;
  auto unet = std::make_unique<UNet>();
  RETURN_IF_ERROR(
      unet->Init(config, latent_image_width, latent_image_height, env.get()));
  ABSL_LOG(INFO) << "UNet is created";

  auto diffusion = std::make_unique<DiffusionStepper>();
  RETURN_IF_ERROR(diffusion->Init(config, env.get()));
  ABSL_LOG(INFO) << "DiffusionStepper is created";

  auto decoder = std::make_unique<Decoder>();
  RETURN_IF_ERROR(decoder->Init(config, latent_image_width, latent_image_height,
                                env.get()));
  ABSL_LOG(INFO) << "Decoder is created";

  Tensor latent_copy;
  {
    Tensor* unet_latent = unet->GetLatentTensor();
    TensorDescriptor descriptor_with_shape =
        unet_latent->GetDescriptor();
    descriptor_with_shape.SetBHWCShape(
        BHWC(unet_latent->Batch(), unet_latent->Height(), unet_latent->Width(),
             unet_latent->Channels()));
    RETURN_IF_ERROR(
        CreateTensor(env->context(), descriptor_with_shape, &latent_copy));
  }
  ABSL_LOG(INFO) << "latent_copy is created";

  return absl::WrapUnique(
      new Diffuser(config, std::move(env), std::move(text_guidance_graph),
                   std::move(unet), std::move(diffusion), std::move(decoder),
                   std::move(copier), std::move(latent_copy)));
}

Diffuser::Diffuser(const Diffuser::Config& config,
                   std::unique_ptr<Environment> env,
                   std::unique_ptr<TextGuidance> text_guidance_graph,
                   std::unique_ptr<UNet> unet,
                   std::unique_ptr<DiffusionStepper> diffusion_stepper,
                   std::unique_ptr<Decoder> decoder,
                   std::unique_ptr<OpHolder> copier, Tensor latent_copy) {
  config_ = config;
  env_ = std::move(env);
  text_guidance_graph_ = std::move(text_guidance_graph);
  unet_ = std::move(unet);
  diffusion_stepper_ = std::move(diffusion_stepper);
  decoder_ = std::move(decoder);
  copier_ = std::move(copier);
  latent_copy_ = std::move(latent_copy);

  bpe_tokenizer_ = std::make_unique<BPETokenizer>();
  switch (config.model_type) {
    case ModelType::kGldm:
    case ModelType::kDistilledGldm:
    case ModelType::kSd2Base:
      bpe_tokenizer_->Init(config.model_dir, /*padding_token=*/0);
      break;
    case ModelType::kTigo:
    case ModelType::kTigoUfo:
    case ModelType::kSd1:
    default:
      bpe_tokenizer_->Init(config.model_dir, /*padding_token=*/49407);
  }

  tokens_.shape = BHWC(1, 1, 2, 77);
  tokens_.data.resize(tokens_.shape.DimensionsProduct());
}

absl::StatusOr<TensorFloat32> Diffuser::Diffuse(
    const std::string& prompt, int total_steps, std::optional<uint> rand_seed,
    std::optional<float> plugins_strength,
    const std::vector<TensorFloat32>& plugin_tensors) {
  RETURN_IF_ERROR(RunInitStep(prompt, total_steps, rand_seed, plugins_strength,
                              plugin_tensors));
  for (int i = 0; i < total_steps; ++i) {
    RETURN_IF_ERROR(RunIterationStep(total_steps, i));
  }
  return RunDecodeStep();
}

absl::Status Diffuser::RunInitStep(
    const std::string& prompt, int total_steps, std::optional<uint> rand_seed,
    std::optional<float> plugins_strength,
    const std::vector<TensorFloat32>& plugin_tensors) {
  std::string negative_prompt = "";
  auto bpe_tokens = bpe_tokenizer_->Encode(prompt);
  auto bpe_base_tokens = bpe_tokenizer_->Encode(negative_prompt);
  for (int i = 0; i < 77; ++i) {
    tokens_.data[i] = bpe_base_tokens[i];
    tokens_.data[i + 77] = bpe_tokens[i];
  }
  RETURN_IF_ERROR(text_guidance_graph_->SetInput(env_.get(), tokens_));
  RETURN_IF_ERROR(text_guidance_graph_->Execute(env_->queue()));
  if (config_.run_unet_with_plugins) {
    int num_plugins = unet_->GetPluginTensors().size();
    for (int i = 0; i < num_plugins; ++i) {
      // Uploads plugin tensor data from a ml drift tensor to a mc tensor.
      Tensor* mc_tensor = unet_->GetPluginTensors()[i];
      TensorDescriptor descriptor_with_data = mc_tensor->GetDescriptor();
      descriptor_with_data.UploadData(plugin_tensors[i]);
      RETURN_IF_ERROR(
          mc_tensor->UploadDescriptorData(descriptor_with_data, env_->queue()));
      RETURN_IF_ERROR(env_->queue()->WaitForCompletion());
    }
    ABSL_LOG(INFO) << "plugin tensor data is uploaded";
  }
  if (config_.run_unet_with_masked_image) {
    Tensor* masked_image = unet_->GetMaskedImageTensor();
    TensorDescriptor descriptor_with_data = masked_image->GetDescriptor();
    TensorFloat32 cpu_encoded_image = {
        .shape = BHWC(1, config_.image_height / 8, config_.image_width / 8, 9),
    };
    int total_size = cpu_encoded_image.shape.DimensionsProduct();
    cpu_encoded_image.data.resize(total_size);
    std::vector<half> default_encoded = LoadF16(
        absl::StrCat(config_.model_dir, "masked_image_latent_constant.bin"),
        (config_.image_height / 8) * (config_.image_width / 8) * 8);
    int k = 0;
    for (int i = 0; i < total_size; i += 9) {
      for (int j = 0; j < 8; ++j) {
        cpu_encoded_image.data[i + j] =
            static_cast<float>(default_encoded[k++]);
      }
      cpu_encoded_image.data[i + 8] = 0.0f;
    }
    descriptor_with_data.UploadData(cpu_encoded_image);
    RETURN_IF_ERROR(masked_image->UploadDescriptorData(descriptor_with_data,
                                                       env_->queue()));
    RETURN_IF_ERROR(env_->queue()->WaitForCompletion());
    ABSL_LOG(INFO) << "constant masked image tensor data is uploaded.";
  }
  RETURN_IF_ERROR(unet_->GenerateNoiseInput(env_->queue(),
                                            rand_seed.value_or(config_.seed)));
  RETURN_IF_ERROR(copier_->Execute(env_->queue(),
                                   text_guidance_graph_->GetGuidanceTensor(),
                                   unet_->GetGuidanceTensor()));
  if (IsTigo(config_.model_type)) {
    RETURN_IF_ERROR(copier_->Execute(
        env_->queue(), text_guidance_graph_->GetTextProjectionTensor(),
        unet_->GetTextProjectionTensor()));
  }
  plugins_strength_ = plugins_strength.value_or(0.0f);
  return absl::OkStatus();
}

absl::Status Diffuser::RunIterationStep(int total_steps, int curr_iteration) {
  int ts, ts_prev;
  if (total_steps == 8) {
    // The distilled models are usually run with 8 steps, and the timestep
    // indices are carefully selected to match its progressive distillation.
    ts = kDistilledModelTimesteps[curr_iteration];
    ts_prev = kDistilledModelTimesteps[curr_iteration + 1];
  } else if (total_steps == 1 && config_.model_type == ModelType::kTigoUfo) {
    ts = 999;
    ts_prev = -1;
  } else if (total_steps == 4 && config_.model_type == ModelType::kTigo) {
    ts = kTigoFourTimesteps[curr_iteration];
    ts_prev = kTigoFourTimesteps[curr_iteration + 1];
  } else {
    const int stride = 1000 / total_steps;
    int t = total_steps - curr_iteration - 1;
    if (t < 0) {
      return absl::InvalidArgumentError(
          "iteration is larget than targeted steps, something went wrong.");
    }
    ts = t * stride + 1;
    ts_prev = ts - stride;
  }
  RETURN_IF_ERROR(
      copier_->Execute(env_->queue(), unet_->GetLatentTensor(), &latent_copy_));

  if (config_.run_unet_with_plugins) {
    Tensor* mc_tensor = unet_->GetPluginsStrengthTensor();
    TensorDescriptor descriptor_with_data = mc_tensor->GetDescriptor();
    TensorFloat32 dst_tensor = {
        .shape = BHWC(1, 1, 1, 1),
    };
    dst_tensor.data.resize(dst_tensor.shape.DimensionsProduct());
    dst_tensor.data[0] = plugins_strength_;
    descriptor_with_data.UploadData(dst_tensor);
    RETURN_IF_ERROR(
        mc_tensor->UploadDescriptorData(descriptor_with_data, env_->queue()));
    RETURN_IF_ERROR(env_->queue()->WaitForCompletion());
  }
  RETURN_IF_ERROR(unet_->Execute(env_->queue(), ts));

  float guidance_scale = 7.5f;
  if (config_.model_type == ModelType::kDistilledGldm ||
      config_.model_type == ModelType::kTigoUfo) {
    guidance_scale = 1.0f;
  } else if (config_.model_type == ModelType::kTigo &&
             config_.run_unet_with_masked_image) {
    guidance_scale = 1.75f;
  }
  RETURN_IF_ERROR(diffusion_stepper_->StepCustomOp(
      env_->queue(), &latent_copy_, unet_->GetEtaUncondTensor(),
      unet_->GetEtaCondTensor(), unet_->GetLatentTensor(), ts, ts_prev,
      guidance_scale));
  RETURN_IF_ERROR(env_->queue()->WaitForCompletion());
  return absl::OkStatus();
}

absl::StatusOr<TensorFloat32> Diffuser::RunDecodeStep() {
  RETURN_IF_ERROR(copier_->Execute(env_->queue(), unet_->GetLatentTensor(),
                                   decoder_->GetInputTensor()));
  RETURN_IF_ERROR(decoder_->Execute(env_->queue()));
  RETURN_IF_ERROR(env_->queue()->WaitForCompletion());
  TensorFloat32 result;
  RETURN_IF_ERROR(decoder_->GetOutput(env_.get(), &result));
  return result;
}

absl::Status Diffuser::TextGuidance::Init(const Diffuser::Config& runner_config,
                                          Environment* env) {
  const auto& gpu_info = env->GetDevicePtr()->GetInfo();

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::F32;
  create_info.storage_type = GetFastestStorageType(gpu_info);
  create_info.hints.Add(ModelHints::kFastTuning);
  GpuModel gpu_model;
  TextGuidanceBuilder builder;
  openclip_ = runner_config.model_type != ModelType::kSd1 &&
              !IsTigo(runner_config.model_type);
  const auto start_init = std::chrono::high_resolution_clock::now();
  if (openclip_) {
    TextGuidanceBuilder::Config config;
    config.embedding_size = 1024;
    config.num_layers =
        runner_config.model_type == ModelType::kSd2Base ? 23 : 24;
    config.num_heads = 16;
    config.file_folder = runner_config.model_dir;
    TextGuidanceBuilder builder;
    RETURN_IF_ERROR(builder.Build(config, gpu_info, create_info, &gpu_model,
                                  &src_, &mask_, &dst_));
  } else {
    TextGuidanceBuilder::Config config;
    config.embedding_size = 768;
    config.num_layers = 12;
    config.num_heads = 12;
    config.skip_final_layer_norm = false;
    config.file_folder = runner_config.model_dir;
    TextGuidanceBuilder builder;
    if (IsTigo(runner_config.model_type)) {
      RETURN_IF_ERROR(builder.Build(config, gpu_info, create_info, &gpu_model,
                                    &src_, /*mask_ptr=*/nullptr, &dst_,
                                    &text_proj_));
    } else {
      RETURN_IF_ERROR(builder.Build(config, gpu_info, create_info, &gpu_model,
                                    &src_, /*mask_ptr=*/nullptr, &dst_));
    }
  }
  RETURN_IF_ERROR(inference_context_.InitFromGpuModel(create_info, &gpu_model,
                                                      env, nullptr));
  const auto end_init = std::chrono::high_resolution_clock::now();
  std::cout << "TextGuidance initialization time: "
            << (end_init - start_init).count() * 1e-6f << " ms." << std::endl;
  return absl::OkStatus();
}

absl::Status Diffuser::TextGuidance::SetInput(
    Environment* env, const ml_drift::Tensor<BHWC, DataType::INT32>& src) {
  if (openclip_) {
    ASSIGN_OR_RETURN(auto mask_tensor, GenerateOpenClipMaskTensor(src));
    RETURN_IF_ERROR(
        inference_context_.SetInputTensor(mask_.id, mask_tensor, env->queue()));
  }
  return inference_context_.SetInputTensor(src_.id, src, env->queue());
}

absl::Status Diffuser::UNet::Init(const Diffuser::Config& runner_config,
                                  int width, int height, Environment* env) {
  const auto& gpu_info = env->GetDevicePtr()->GetInfo();

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::F16;
  create_info.storage_type = GetFastestStorageType(gpu_info);
  create_info.hints.Add(ModelHints::kFastTuning);
  GpuModel gpu_model;
  UnetBuilder builder;
  UnetBuilder::Config config;
  config.use_spatial_transformer = true;
  config.transformer_depth = 1;
  if (IsTigo(runner_config.model_type)) {
    config.num_res_blocks = {1, 1, 2};
    config.model_channels = 32;
    config.channel_mult = {10, 20, 32};  // {320, 640, 1024}
    config.num_transformer_blocks = {{}, {1}, {3, 3}, {3, 3, 3}, {1, 1}, {}};
    config.use_self_attn = {false, false, true};
    config.use_convnext = {false, false, true};
    config.self_attn_share_kv_proj = true;
    config.ff_multiplier = 6;
    config.skip_middle_blocks = true;
    config.activation_function =
        UnetBuilder::Config::ActivationFunction::kGatedSiLU;
  } else {
    config.model_channels = 320;
    config.num_res_blocks = {2, 2, 2, 2};
    config.channel_mult = {1, 2, 4, 4};
    config.num_transformer_blocks = {{1, 1}, {1, 1},    {1, 1},    {},
                                     {},     {1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
  }
  switch (runner_config.model_type) {
    case ModelType::kSd2Base:
      config.num_attn_heads = {5, 10, 20, 20};
      break;
    case ModelType::kTigo:
    case ModelType::kTigoUfo:
      config.num_attn_heads = {5, 10, 16};
      break;
    case ModelType::kGldm:
    case ModelType::kDistilledGldm:
    case ModelType::kSd1:
    default:
      config.num_attn_heads = {8, 8, 8, 8};
  }
  switch (runner_config.model_type) {
    case ModelType::kGldm:
    case ModelType::kTigo:
    case ModelType::kTigoUfo:
      config.in_channels = 8;
      config.out_channels = 8;
      break;
    case ModelType::kSd1:
    case ModelType::kSd2Base:
    case ModelType::kDistilledGldm:
    default:
      config.in_channels = 4;
      config.out_channels = 4;
  }
  switch (runner_config.model_type) {
    case ModelType::kGldm:
    case ModelType::kDistilledGldm:
    case ModelType::kSd2Base:
      config.context_dim = 1024;
      break;
    case ModelType::kSd1:
    case ModelType::kTigo:
    case ModelType::kTigoUfo:
    default:
      config.context_dim = 768;
  }
  config.unet_file_dir = runner_config.model_dir;
  config.lora_file_dir = runner_config.lora_dir;
  config.lora_weights_layer_mapping = runner_config.lora_weights_layer_mapping;
  config.lora_rank = runner_config.lora_rank;
  config.lora_scale = 1;
  const auto start_init = std::chrono::high_resolution_clock::now();
  if (!config.num_transformer_blocks.empty() &&
      config.num_transformer_blocks.size() !=
          config.num_res_blocks.size() * 2) {
    return absl::InvalidArgumentError(
        "the size of num_transformer_blocks must be twice as the size of "
        "num_res_blocks.");
  }
  if (runner_config.run_unet_with_plugins) {
    RETURN_IF_ERROR(builder.BuildUNetWithPlugins(
        config, gpu_info, create_info, width, height, &gpu_model, &src_,
        &plugin_tensors_, &temb_, &guidance_, &eta0_, &eta1_,
        &plugins_strength_, &dbg_));
  } else if (runner_config.run_unet_with_masked_image) {
    RETURN_IF_ERROR(builder.Build(
        config, gpu_info, create_info, width, height, &gpu_model, &src_, &temb_,
        &guidance_, IsTigo(runner_config.model_type) ? &text_proj_ : nullptr,
        &masked_image_, &eta0_, &eta1_, &dbg_));
  } else {
    RETURN_IF_ERROR(builder.Build(
        config, gpu_info, create_info, width, height, &gpu_model, &src_, &temb_,
        &guidance_, IsTigo(runner_config.model_type) ? &text_proj_ : nullptr,
        nullptr, &eta0_, &eta1_, &dbg_));
  }
  RETURN_IF_ERROR(inference_context_.InitFromGpuModel(create_info, &gpu_model,
                                                      env, nullptr));
  const auto end_init = std::chrono::high_resolution_clock::now();
  std::cout << "UNet initialization time: "
            << (end_init - start_init).count() * 1e-6f << " ms." << std::endl;

  TensorDescriptor default_desc(DataType::FLOAT16,
                                GetFastestStorageType(gpu_info), Layout::HWC);
  RETURN_IF_ERROR(temb_generation_op_.Init(
      env,
      CreateTembGenerationOp(gpu_info, default_desc, runner_config.model_dir)));

  {
    OperationDef op_def;
    op_def.dst_tensors.push_back(default_desc);
    RETURN_IF_ERROR(
        noise_op_.Init(env, CreateRandomNormalPhilox(gpu_info, op_def)));
  }
  return absl::OkStatus();
}

absl::Status Diffuser::UNet::GenerateNoiseInput(CLCommandQueue* queue,
                                                uint32_t base_seed) {
  int seed = GetRandomInt(60000, base_seed);
  OpHolder::ExecutionParams exec_params;
  exec_params.dst = {GetLatentTensor()};
  exec_params.int_params = {{"seed", seed}};
  return noise_op_.Execute(queue, exec_params);
}

absl::Status Diffuser::UNet::Execute(CLCommandQueue* queue, int step_index) {
  {
    OpHolder::ExecutionParams exec_params;
    exec_params.dst = {GetTembTensor()};
    exec_params.float_params = {{"index_val", step_index}};
    RETURN_IF_ERROR(temb_generation_op_.Execute(queue, exec_params));
  }
  return inference_context_.AddToQueue(queue);
}

absl::Status Diffuser::UNet::LogDebugTensor(CLCommandQueue* queue) {
  ml_drift::TensorFloat32 dbg_result;
  RETURN_IF_ERROR(
      inference_context_.GetOutputTensor(dbg_.id, queue, &dbg_result));
  ABSL_LOG(INFO) << "debug tensor shape:" << dbg_result.shape.b << "x"
                 << dbg_result.shape.h << "x" << dbg_result.shape.w << "x"
                 << dbg_result.shape.c;
  float min_v = dbg_result.data[0];
  float max_v = dbg_result.data[0];
  int num_nan = 0;
  for (float v : dbg_result.data) {
    num_nan += isnan(v) ? 1 : 0;
    min_v = std::min(min_v, v);
    max_v = std::max(max_v, v);
  }
  ABSL_LOG(INFO) << "debug tensor value min:" << min_v
                 << ", value max:" << max_v << ", num nan:" << num_nan;
  return absl::OkStatus();
}

absl::Status Diffuser::Decoder::Init(const Diffuser::Config& runner_config,
                                     int width, int height, Environment* env) {
  const auto& gpu_info = env->GetDevicePtr()->GetInfo();

  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::F16;
  create_info.storage_type = GetFastestStorageType(gpu_info);
  create_info.hints.Add(ModelHints::kFastTuning);
  GpuModel gpu_model;
  if (IsGldmOrTigo(runner_config.model_type)) {
  } else {
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
                                         runner_config.model_dir));

    const auto start_init = std::chrono::high_resolution_clock::now();
    RETURN_IF_ERROR(inference_context_.InitFromGpuModel(create_info, &gpu_model,
                                                        env, nullptr));
    const auto end_init = std::chrono::high_resolution_clock::now();
    std::cout << "SD decoder initialization time: "
              << (end_init - start_init).count() * 1e-6f << " ms." << std::endl;
  }

  return absl::OkStatus();
}

absl::Status Diffuser::DiffusionStepper::Init(
    const Diffuser::Config& runner_config, Environment* env) {
  alphas_ =
      LoadF16(runner_config.model_dir + "alphas_cumprod.bin", 1000);
  alphas_prev_ = alphas_;
  alphas_prev_.insert(alphas_prev_.begin(), half(1.0f));

  TensorDescriptor desc(
      DataType::FLOAT16,
      GetFastestStorageType(env->device().GetInfo()), Layout::HWC);

  if (IsGldmOrTigo(runner_config.model_type)) {
    if (runner_config.run_unet_with_masked_image) {
      OperationDef op_def;
      op_def.dst_tensors.push_back(desc);
      RETURN_IF_ERROR(noise_op_.Init(
          env,
          CreateRandomNormalPhilox(env->GetDevicePtr()->GetInfo(), op_def)));
      run_lcm_stepper_ = true;
      desc.SetBHWCShape(BHWC(1, runner_config.image_width / 8,
                             runner_config.image_height / 8, 8));
      RETURN_IF_ERROR(CreateTensor(env->context(), desc, &noise_));
      RETURN_IF_ERROR(custom_op_.Init(
          env, CreateLcmVPredictionStepOp(desc, desc, desc, desc, desc)));
    } else {
      RETURN_IF_ERROR(custom_op_.Init(
          env, CreateVPredictionStepOp(desc, desc, desc, desc)));
    }
  } else {
    RETURN_IF_ERROR(
        custom_op_.Init(env, CreateDiffusionStepOp(desc, desc, desc, desc)));
  }
  return absl::OkStatus();
}

absl::Status Diffuser::DiffusionStepper::StepCustomOp(
    CLCommandQueue* queue, Tensor* xIn, Tensor* etaUncondIn, Tensor* etaCondIn,
    Tensor* dst, int tIn, int tPrevIn, float guidanceScaleIn) {
  float alphaIn = alphas_[tIn];
  int tPrevInOffset = std::max(0, tPrevIn + 1);
  float alphaPrevIn = alphas_prev_[tPrevInOffset];
  half sqrt_alpha = half(sqrt(alphaIn));
  half sqrt_alpha_prev = half(sqrt(alphaPrevIn));
  half sqrt_one_minus_alpha = half(sqrt(1.0f - alphaIn));
  half sqrt_one_minus_alpha_prev = half(sqrt(1.0f - alphaPrevIn));

  if (run_lcm_stepper_) {
    OpHolder::ExecutionParams exec_params;
    exec_params.dst = {&noise_};
    exec_params.int_params = {{"seed", rand()}};
    RETURN_IF_ERROR(noise_op_.Execute(queue, exec_params));
    RETURN_IF_ERROR(queue->WaitForCompletion());
  }

  OpHolder::ExecutionParams exec_params;
  exec_params.src = {xIn, etaUncondIn, etaCondIn};
  exec_params.dst = {dst};
  exec_params.half_params = {
      {"guidance_scale", half(guidanceScaleIn)},
      {"sqrt_alpha", half(sqrt_alpha)},
      {"sqrt_alpha_prev", half(sqrt_alpha_prev)},
      {"sqrt_one_minus_alpha", half(sqrt_one_minus_alpha)},
      {"sqrt_one_minus_alpha_prev", half(sqrt_one_minus_alpha_prev)}};
  if (run_lcm_stepper_) {
    exec_params.src.push_back(&noise_);
    float sq_sigma = 0.25;
    float timestep_scaling = 10;
    float scaled_timestamp = tIn * timestep_scaling;
    float sq_scaled_timestamp = scaled_timestamp * scaled_timestamp;
    float c_skip = sq_sigma / (sq_scaled_timestamp + sq_sigma);
    float c_out = scaled_timestamp / sqrt(sq_scaled_timestamp + sq_sigma);
    exec_params.int_params = {{"step_idx", tIn}};
    exec_params.half_params.insert({"c_skip", half(c_skip)});
    exec_params.half_params.insert({"c_out", half(c_out)});
  }
  return custom_op_.Execute(queue, exec_params);
}

GPUOperation Diffuser::DiffusionStepper::CreateVPredictionStepOp(
    const TensorDescriptor& xIn, const TensorDescriptor& etaUncondIn,
    const TensorDescriptor& etaCondIn, const TensorDescriptor& dst) {
  GPUOperation op;
  op.AddSrcTensor("xIn", xIn);
  op.AddSrcTensor("etaUncondIn", etaUncondIn);
  op.AddSrcTensor("etaCondIn", etaCondIn);
  op.AddDstTensor("dst", dst);
  op.args_.AddHalf("guidance_scale", half(1.0f));
  op.args_.AddHalf("sqrt_alpha", half(1.0f));
  op.args_.AddHalf("sqrt_alpha_prev", half(1.0f));
  op.args_.AddHalf("sqrt_one_minus_alpha", half(1.0f));
  op.args_.AddHalf("sqrt_one_minus_alpha_prev", half(1.0f));
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  std::string c;
  c += R"(MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.dst.Width() || Y >= args.dst.Height() || S >= args.dst.Slices()) return;
  Type eta_cond = args.etaCondIn.Read(X, Y, S);
  Type eta_uncond = args.etaUncondIn.Read(X, Y, S);
  Type x_in = args.xIn.Read(X, Y, S);

  Type delta_cond = (eta_cond - eta_uncond) * args.guidance_scale;
  Type eta = eta_uncond + delta_cond;

  Type predX0Scaled = args.sqrt_alpha * x_in;
  Type deltaX0 = args.sqrt_one_minus_alpha * eta;
  Type predX0 = predX0Scaled - deltaX0;
  Type etaScaled = args.sqrt_alpha * eta;
  Type etaDelta = args.sqrt_one_minus_alpha * x_in;
  eta = etaScaled + etaDelta;

  Type dirX = eta * args.sqrt_one_minus_alpha_prev;
  Type xPrevBase = predX0 * args.sqrt_alpha_prev;
  Type result = xPrevBase + dirX;
  args.dst.Write(result, X, Y, S);
})";
  absl::StrReplaceAll({{"Type", ToUclDataType(dst.GetDataType(), 4)}}, &c);
  op.code_ = std::move(c);
  return op;
}

GPUOperation Diffuser::DiffusionStepper::CreateLcmVPredictionStepOp(
    const TensorDescriptor& xIn, const TensorDescriptor& etaUncondIn,
    const TensorDescriptor& etaCondIn, const TensorDescriptor& noise,
    const TensorDescriptor& dst) {
  GPUOperation op;
  op.AddSrcTensor("xIn", xIn);
  op.AddSrcTensor("etaUncondIn", etaUncondIn);
  op.AddSrcTensor("etaCondIn", etaCondIn);
  op.AddSrcTensor("noise", noise);
  op.AddDstTensor("dst", dst);
  op.args_.AddHalf("guidance_scale", half(1.0f));
  op.args_.AddHalf("sqrt_alpha", half(1.0f));
  op.args_.AddHalf("sqrt_alpha_prev", half(1.0f));
  op.args_.AddHalf("sqrt_one_minus_alpha", half(1.0f));
  op.args_.AddHalf("sqrt_one_minus_alpha_prev", half(1.0f));
  op.args_.AddInt("step_idx", 1);
  op.args_.AddHalf("c_out", half(1.0f));
  op.args_.AddHalf("c_skip", half(0.0f));
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  std::string c;
  c += R"(MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.dst.Width() || Y >= args.dst.Height() || S >= args.dst.Slices()) return;
  Type eta_cond = args.etaCondIn.Read(X, Y, S);
  Type eta_uncond = args.etaUncondIn.Read(X, Y, S);
  Type x_in = args.xIn.Read(X, Y, S);

  Type eta = eta_cond * args.guidance_scale + (1 - args.guidance_scale) * eta_uncond;

  Type predX0Scaled = args.sqrt_alpha * x_in;
  Type deltaX0 = args.sqrt_one_minus_alpha * eta;
  Type predX0 = predX0Scaled - deltaX0;

  Type denoised = args.c_out * predX0 + args.c_skip * x_in;
  if (args.step_idx == 650) {
    args.dst.Write(denoised, X, Y, S);
    return;
  }

  Type noise = args.noise.Read(X, Y, S);
  Type prev_sample_compute = args.sqrt_alpha_prev * denoised + args.sqrt_one_minus_alpha_prev * noise;
  args.dst.Write(prev_sample_compute, X, Y, S);
})";
  absl::StrReplaceAll({{"Type", ToUclDataType(dst.GetDataType(), 4)}}, &c);
  op.code_ = std::move(c);
  return op;
}

absl::Status Diffuser::OpHolder::InitElementwiseOneInput(
    OperationType op_type, const TensorDescriptor& src_desc,
    const TensorDescriptor& dst_desc, Environment* env) {
  OperationDef op_def;
  op_def.src_tensors.push_back(src_desc);
  op_def.dst_tensors.push_back(dst_desc);
  GPUOperation operation =
      CreateElementwiseOneInput(env->device().GetInfo(), op_def, op_type);
  return Initialize(env, std::move(operation));
}

absl::Status Diffuser::OpHolder::Execute(CLCommandQueue* queue,
                                         const ExecutionParams& params) {
  for (int i = 0; i < params.src.size(); ++i) {
    RETURN_IF_ERROR(op_.SetSrcTensor(i, params.src[i]));
  }
  for (int i = 0; i < params.dst.size(); ++i) {
    RETURN_IF_ERROR(op_.SetDstTensor(i, params.dst[i]));
  }
  for (const auto& float_param : params.float_params) {
    RETURN_IF_ERROR(op_.SetFloat(float_param.first, float_param.second));
  }
  for (const auto& half_param : params.half_params) {
    RETURN_IF_ERROR(op_.SetHalf(half_param.first, half_param.second));
  }
  for (const auto& int_param : params.int_params) {
    RETURN_IF_ERROR(op_.SetInt(int_param.first, int_param.second));
  }
  RETURN_IF_ERROR(op_.UpdateParams());
  RETURN_IF_ERROR(op_.AddToQueue(queue));
  return absl::OkStatus();
}

absl::Status Diffuser::OpHolder::Execute(CLCommandQueue* queue, Tensor* src,
                                         Tensor* dst) {
  RETURN_IF_ERROR(op_.SetSrcTensor(0, src));
  RETURN_IF_ERROR(op_.SetDstTensor(0, dst));
  RETURN_IF_ERROR(op_.UpdateParams());
  RETURN_IF_ERROR(op_.AddToQueue(queue));
  return absl::OkStatus();
}

absl::Status Diffuser::OpHolder::Initialize(Environment* env,
                                            GPUOperation&& operation) {
  auto gpu_op = std::make_unique<GPUOperation>(std::move(operation));

  RETURN_IF_ERROR(gpu_op->AssembleCode(env->device().GetInfo()));

  op_.Init(std::move(gpu_op));
  {
    CreationContext creation_context;
    creation_context.device = env->GetDevicePtr();
    creation_context.context = &env->context();
    creation_context.queue = env->queue();
    creation_context.cache = env->program_cache();
    RETURN_IF_ERROR(op_.Compile(creation_context));
  }
  return absl::OkStatus();
}

}  // namespace stable_diffusion
}  // namespace cl
}  // namespace ml_drift
