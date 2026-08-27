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
#include <chrono>  // NOLINT(build/c++11)
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <ratio>  // NOLINT(build/c++11)
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ml_drift/cl/cl_command_queue.h"
#include "ml_drift/cl/cl_operation.h"
#include "ml_drift/cl/environment.h"
#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/cl/tensor.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/kernels/elementwise.h"
#include "ml_drift/common/kernels/random_philox.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/model_hints.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
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

int GetRandomInt(uint32_t size, uint32_t seed) {
  std::default_random_engine generator;
  generator.seed(seed);
  std::uniform_int_distribution<> distribution(0, size - 1);
  return distribution(generator);
}

}  // namespace

absl::StatusOr<std::unique_ptr<Diffuser>> Diffuser::Create(
    const Diffuser::Config& config) {
  ABSL_RETURN_IF_ERROR(SanityCheck(config));
  auto load_status = LoadOpenCL();
  if (!load_status.ok()) {
    return absl::FailedPreconditionError(load_status.message());
  }
  auto env = std::make_unique<Environment>();
  ABSL_RETURN_IF_ERROR(CreateEnvironment(env.get(), config.env_options));
  ABSL_LOG(INFO) << "Environment created";

  const auto& gpu_info = env->GetDevicePtr()->GetInfo();
  TensorDescriptor default_desc(DataType::FLOAT16,
                                GetFastestStorageType(gpu_info), Layout::HWC);

  auto text_guidance_graph = std::make_unique<TextGuidance>();
  ABSL_RETURN_IF_ERROR(text_guidance_graph->Init(config, env.get()));
  ABSL_LOG(INFO) << "TextGuidance is created";

  auto copier = std::make_unique<Diffuser::OpHolder>();
  ABSL_RETURN_IF_ERROR(copier->InitElementwiseOneInput(
      OperationType::COPY, default_desc, default_desc, env.get()));
  ABSL_LOG(INFO) << "Copier is created";

  int latent_image_width = config.image_width / 8;
  int latent_image_height = config.image_height / 8;

  auto decoder = std::make_unique<Decoder>();
  ABSL_RETURN_IF_ERROR(decoder->Init(config, latent_image_width,
                                     latent_image_height, env.get()));
  ABSL_LOG(INFO) << "Decoder is created";

  auto unet = std::make_unique<UNet>();
  ABSL_RETURN_IF_ERROR(
      unet->Init(config, latent_image_width, latent_image_height, env.get()));
  ABSL_LOG(INFO) << "UNet is created";

  Tensor latent_copy;
  {
    Tensor* unet_latent = unet->GetLatentTensor();
    TensorDescriptor descriptor_with_shape =
        unet_latent->GetDescriptor();
    descriptor_with_shape.SetBHWCShape(
        BHWC(unet_latent->Batch(), unet_latent->Height(), unet_latent->Width(),
             unet_latent->Channels()));
    ABSL_RETURN_IF_ERROR(
        CreateTensor(env->context(), descriptor_with_shape, &latent_copy));
  }
  ABSL_LOG(INFO) << "latent_copy is created";

  return absl::WrapUnique(new Diffuser(
      config, std::move(env), std::move(text_guidance_graph), std::move(unet),
      std::move(decoder), std::move(copier), std::move(latent_copy)));
}

Diffuser::Diffuser(const Diffuser::Config& config,
                   std::unique_ptr<Environment> env,
                   std::unique_ptr<TextGuidance> text_guidance_graph,
                   std::unique_ptr<UNet> unet,
                   std::unique_ptr<Decoder> decoder,
                   std::unique_ptr<OpHolder> copier, Tensor latent_copy) {
  config_ = config;
  env_ = std::move(env);
  text_guidance_graph_ = std::move(text_guidance_graph);
  unet_ = std::move(unet);
  decoder_ = std::move(decoder);
  copier_ = std::move(copier);
  latent_copy_ = std::move(latent_copy);

  bpe_tokenizer_ = std::make_unique<BPETokenizer>();
  bpe_tokenizer_->Init(config.model_dir, /*padding_token=*/49407);

  tokens_.shape = BHWC(1, 1, 2, 77);
  tokens_.data.resize(tokens_.shape.DimensionsProduct());
}

absl::StatusOr<TensorFloat32> Diffuser::Diffuse(
    const std::string& prompt, int total_steps, std::optional<uint> rand_seed,
    std::optional<float> plugins_strength,
    const std::vector<TensorFloat32>& plugin_tensors, bool show_progress) {
  const auto p0 = std::chrono::high_resolution_clock::now();
  ABSL_RETURN_IF_ERROR(RunInitStep(prompt, total_steps, rand_seed,
                                   plugins_strength, plugin_tensors));
  const auto p1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < total_steps; ++i) {
    ABSL_RETURN_IF_ERROR(RunIterationStep(total_steps, i));
    if (show_progress) {
      std::cout << "step " << (i + 1) << "/" << total_steps << std::endl;
    }
  }
  const auto p2 = std::chrono::high_resolution_clock::now();
  auto result = RunDecodeStep();
  const auto p3 = std::chrono::high_resolution_clock::now();

  std::cout << "Init time: "
            << std::chrono::duration<float, std::milli>(p1 - p0).count()
            << " ms." << std::endl;
  std::cout << "Loop time: "
            << std::chrono::duration<float, std::milli>(p2 - p1).count()
            << " ms." << std::endl;
  std::cout << "Decode time: "
            << std::chrono::duration<float, std::milli>(p3 - p2).count()
            << " ms." << std::endl;
  std::cout << "Total diffuser time: "
            << std::chrono::duration<float, std::milli>(p3 - p0).count()
            << " ms." << std::endl;

  return result;
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
  ABSL_RETURN_IF_ERROR(text_guidance_graph_->SetInput(env_.get(), tokens_));
  ABSL_RETURN_IF_ERROR(text_guidance_graph_->Execute(env_->queue()));
  if (config_.run_unet_with_plugins) {
    int num_plugins = unet_->GetPluginTensors().size();
    for (int i = 0; i < num_plugins; ++i) {
      // Uploads plugin tensor data from a ml drift tensor to a mc tensor.
      Tensor* mc_tensor = unet_->GetPluginTensors()[i];
      TensorDescriptor descriptor_with_data = mc_tensor->GetDescriptor();
      descriptor_with_data.UploadData(plugin_tensors[i]);
      ABSL_RETURN_IF_ERROR(
          mc_tensor->UploadDescriptorData(descriptor_with_data, env_->queue()));
      ABSL_RETURN_IF_ERROR(env_->queue()->WaitForCompletion());
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
    ABSL_RETURN_IF_ERROR(masked_image->UploadDescriptorData(
        descriptor_with_data, env_->queue()));
    ABSL_RETURN_IF_ERROR(env_->queue()->WaitForCompletion());
    ABSL_LOG(INFO) << "constant masked image tensor data is uploaded.";
  }
  ABSL_RETURN_IF_ERROR(unet_->GenerateNoiseInput(
      env_->queue(), rand_seed.value_or(config_.seed)));
  ABSL_RETURN_IF_ERROR(
      copier_->Execute(env_->queue(), text_guidance_graph_->GetGuidanceTensor(),
                       unet_->GetGuidanceTensor()));

  plugins_strength_ = plugins_strength.value_or(0.0f);
  return absl::OkStatus();
}

absl::Status Diffuser::RunIterationStep(int total_steps, int curr_iteration) {
  int ts, ts_prev;
  {
    const int stride = 1000 / total_steps;
    int t = total_steps - curr_iteration - 1;
    if (t < 0) {
      return absl::InvalidArgumentError(
          "iteration is larget than targeted steps, something went wrong.");
    }
    ts = std::min(t * stride + 1, 999);
    ts_prev = ts - stride;
  }

  if (config_.run_unet_with_plugins) {
    Tensor* mc_tensor = unet_->GetPluginsStrengthTensor();
    TensorDescriptor descriptor_with_data = mc_tensor->GetDescriptor();
    TensorFloat32 dst_tensor = {
        .shape = BHWC(1, 1, 1, 1),
    };
    dst_tensor.data.resize(dst_tensor.shape.DimensionsProduct());
    dst_tensor.data[0] = plugins_strength_;
    descriptor_with_data.UploadData(dst_tensor);
    ABSL_RETURN_IF_ERROR(
        mc_tensor->UploadDescriptorData(descriptor_with_data, env_->queue()));
    ABSL_RETURN_IF_ERROR(env_->queue()->WaitForCompletion());
  }
  float guidance_scale = 7.5f;

  float alphaIn = unet_->alphas_[ts];
  int tPrevInOffset = std::max(0, ts_prev + 1);
  float alphaPrevIn = unet_->alphas_prev_[tPrevInOffset];
  float sqrt_alpha = std::sqrt(alphaIn);
  float sqrt_alpha_prev = std::sqrt(alphaPrevIn);
  float sqrt_one_minus_alpha = std::sqrt(1.0f - alphaIn);
  float sqrt_one_minus_alpha_prev = std::sqrt(1.0f - alphaPrevIn);

  ABSL_RETURN_IF_ERROR(unet_->Execute(
      env_->queue(), ts, guidance_scale, sqrt_alpha, sqrt_alpha_prev,
      sqrt_one_minus_alpha, sqrt_one_minus_alpha_prev));
  ABSL_RETURN_IF_ERROR(copier_->Execute(env_->queue(),
                                        unet_->GetOutputLatentTensor(),
                                        unet_->GetInputLatentTensor()));
  ABSL_RETURN_IF_ERROR(env_->queue()->WaitForCompletion());
  return absl::OkStatus();
}

absl::StatusOr<TensorFloat32> Diffuser::RunDecodeStep() {
  ABSL_RETURN_IF_ERROR(copier_->Execute(env_->queue(),
                                        unet_->GetOutputLatentTensor(),
                                        decoder_->GetInputTensor()));
  ABSL_RETURN_IF_ERROR(decoder_->Execute(env_->queue()));
  ABSL_RETURN_IF_ERROR(env_->queue()->WaitForCompletion());
  TensorFloat32 result;
  ABSL_RETURN_IF_ERROR(decoder_->GetOutput(env_.get(), &result));
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

  const auto start_init = std::chrono::high_resolution_clock::now();
  {
    TextGuidanceBuilder::Config config;
    config.embedding_size = 768;
    config.num_layers = 12;
    config.num_heads = 12;
    config.skip_final_layer_norm = false;
    config.file_folder = runner_config.model_dir;
    TextGuidanceBuilder builder;

      ABSL_RETURN_IF_ERROR(builder.Build(config, gpu_info, create_info,
                                         &gpu_model, &src_,
                                         /*mask_ptr=*/nullptr, &dst_));
  }
  ABSL_RETURN_IF_ERROR(inference_context_.InitFromGpuModel(
      create_info, &gpu_model, env, nullptr));
  const auto end_init = std::chrono::high_resolution_clock::now();
  std::cout
      << "TextGuidance initialization time: "
      << std::chrono::duration<float, std::milli>(end_init - start_init).count()
      << " ms." << std::endl;

  return absl::OkStatus();
}

absl::Status Diffuser::TextGuidance::SetInput(
    Environment* env, const ml_drift::Tensor<BHWC, DataType::INT32>& src) {
  return inference_context_.SetInputTensor(src_.id, src, env->queue());
}

absl::Status Diffuser::UNet::Init(const Diffuser::Config& runner_config,
                                  int width, int height, Environment* env) {
  alphas_ = LoadF16(absl::StrCat(runner_config.model_dir, "alphas_cumprod.bin"),
                    1000);
  alphas_prev_ = alphas_;
  alphas_prev_.insert(alphas_prev_.begin(), half(1.0f));
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

  config.model_channels = 320;
  config.num_res_blocks = {2, 2, 2, 2};
  config.channel_mult = {1, 2, 4, 4};
  config.num_transformer_blocks = {{1, 1}, {1, 1},    {1, 1},    {},
                                   {},     {1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
  config.num_attn_heads = {8, 8, 8, 8};
  config.in_channels = 4;
  config.out_channels = 4;
  config.context_dim = 768;
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
  GpuModelBuilderOptions options;
  options.hints = create_info.hints;
  options.storage = create_info.storage_type;
  options.use_f32_accum_for_f16_convolutions =
      (create_info.precision == CalculationsPrecision::F32_F16);
  ml_drift::GpuModelBuilder step_builder(gpu_info, options);

  index_val_ = step_builder.AddTensor(BHWC(1, 1, 1, 4), DataType::FLOAT32);
  ABSL_ASSIGN_OR_RETURN(
      auto result_or,
      step_builder.AppendOp(
          "TembGeneration", {index_val_},
          {{"channels", 320},
           {"storage_type",
            static_cast<int>(GetFastestStorageType(gpu_info))}}));
  temb_ = result_or[0];

  guidance_scale_ = step_builder.AddTensor(BHWC(1, 1, 1, 4), DataType::FLOAT32);
  sqrt_alpha_ = step_builder.AddTensor(BHWC(1, 1, 1, 4), DataType::FLOAT32);
  sqrt_alpha_prev_ =
      step_builder.AddTensor(BHWC(1, 1, 1, 4), DataType::FLOAT32);
  sqrt_one_minus_alpha_ =
      step_builder.AddTensor(BHWC(1, 1, 1, 4), DataType::FLOAT32);
  sqrt_one_minus_alpha_prev_ =
      step_builder.AddTensor(BHWC(1, 1, 1, 4), DataType::FLOAT32);
  if (runner_config.run_unet_with_plugins) {
    ABSL_RETURN_IF_ERROR(builder.BuildUNetWithPlugins(
        config, gpu_info, create_info, width, height, &step_builder, &src_,
        &plugin_tensors_, &temb_, &guidance_, &eta0_, &eta1_,
        &plugins_strength_, &dbg_));
  } else if (runner_config.run_unet_with_masked_image) {
    ABSL_RETURN_IF_ERROR(builder.Build(
        config, gpu_info, create_info, width, height, &step_builder, &src_,
        &temb_, &guidance_, nullptr, &masked_image_, &eta0_, &eta1_, &dbg_));
  } else {
    ABSL_RETURN_IF_ERROR(builder.Build(
        config, gpu_info, create_info, width, height, &step_builder, &src_,
        &temb_, &guidance_, nullptr, nullptr, &eta0_, &eta1_, &dbg_));
  }

  std::vector<ValueId> inputs = {src_.id,
                                 guidance_.id,
                                 index_val_.id,
                                 guidance_scale_.id,
                                 sqrt_alpha_.id,
                                 sqrt_alpha_prev_.id,
                                 sqrt_one_minus_alpha_.id,
                                 sqrt_one_minus_alpha_prev_.id};
  if (runner_config.run_unet_with_masked_image) {
    inputs.push_back(masked_image_.id);
  }
  if (runner_config.run_unet_with_plugins) {
    for (const auto& tensor : plugin_tensors_) {
      inputs.push_back(tensor.id);
    }
    inputs.push_back(plugins_strength_.id);
  }

  src_input_ = src_;
  ABSL_ASSIGN_OR_RETURN(
      auto step_result,
      step_builder.AppendOp(
          "DiffusionStep",
          {src_, eta0_, eta1_, guidance_scale_, sqrt_alpha_, sqrt_alpha_prev_,
           sqrt_one_minus_alpha_, sqrt_one_minus_alpha_prev_}));
  src_ = step_result[0];

  std::vector<ValueId> outputs = {src_.id};
  outputs.insert(outputs.end(), inputs.begin(), inputs.end());

  ABSL_RETURN_IF_ERROR(step_builder.GetGpuModel(inputs, outputs, &gpu_model));

  ABSL_RETURN_IF_ERROR(inference_context_.InitFromGpuModel(
      create_info, &gpu_model, env, nullptr));
  const auto end_init = std::chrono::high_resolution_clock::now();
  std::cout
      << "UNet initialization time: "
      << std::chrono::duration<float, std::milli>(end_init - start_init).count()
      << " ms." << std::endl;

  TensorDescriptor default_desc(DataType::FLOAT16,
                                GetFastestStorageType(gpu_info), Layout::HWC);

  {
    OperationDef op_def;
    op_def.dst_tensors.push_back(default_desc);
    ABSL_RETURN_IF_ERROR(
        noise_op_.Init(env, CreateRandomNormalPhilox(gpu_info, op_def)));
  }
  return absl::OkStatus();
}

absl::Status Diffuser::UNet::GenerateNoiseInput(CLCommandQueue* queue,
                                                uint32_t base_seed) {
  int seed = GetRandomInt(60000, base_seed);
  OpHolder::ExecutionParams exec_params;
  exec_params.dst = {GetInputLatentTensor()};
  exec_params.int_params = {{"seed", seed}};
  return noise_op_.Execute(queue, exec_params);
}

absl::Status Diffuser::UNet::LogDebugTensor(CLCommandQueue* queue) {
  ml_drift::TensorFloat32 dbg_result;
  ABSL_RETURN_IF_ERROR(
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
                                              &dst_, runner_config.model_dir));

    const auto start_init = std::chrono::high_resolution_clock::now();
    ABSL_RETURN_IF_ERROR(inference_context_.InitFromGpuModel(
        create_info, &gpu_model, env, nullptr));
    const auto end_init = std::chrono::high_resolution_clock::now();
    std::cout << "SD decoder initialization time: "
              << std::chrono::duration<float, std::milli>(end_init - start_init)
                     .count()
              << " ms." << std::endl;

    return absl::OkStatus();
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
    ABSL_RETURN_IF_ERROR(op_.SetSrcTensor(i, params.src[i]));
  }
  for (int i = 0; i < params.dst.size(); ++i) {
    ABSL_RETURN_IF_ERROR(op_.SetDstTensor(i, params.dst[i]));
  }
  for (const auto& float_param : params.float_params) {
    ABSL_RETURN_IF_ERROR(op_.SetFloat(float_param.first, float_param.second));
  }
  for (const auto& half_param : params.half_params) {
    ABSL_RETURN_IF_ERROR(op_.SetHalf(half_param.first, half_param.second));
  }
  for (const auto& int_param : params.int_params) {
    ABSL_RETURN_IF_ERROR(op_.SetInt(int_param.first, int_param.second));
  }
  ABSL_RETURN_IF_ERROR(op_.UpdateParams());
  ABSL_RETURN_IF_ERROR(op_.AddToQueue(queue));
  return absl::OkStatus();
}

absl::Status Diffuser::OpHolder::Execute(CLCommandQueue* queue, Tensor* src,
                                         Tensor* dst) {
  ABSL_RETURN_IF_ERROR(op_.SetSrcTensor(0, src));
  ABSL_RETURN_IF_ERROR(op_.SetDstTensor(0, dst));
  ABSL_RETURN_IF_ERROR(op_.UpdateParams());
  ABSL_RETURN_IF_ERROR(op_.AddToQueue(queue));
  return absl::OkStatus();
}

absl::Status Diffuser::OpHolder::Initialize(Environment* env,
                                            GPUOperation&& operation) {
  auto gpu_op = std::make_unique<GPUOperation>(std::move(operation));

  ABSL_RETURN_IF_ERROR(gpu_op->AssembleCode(env->device().GetInfo()));

  op_.Init(std::move(gpu_op));
  ABSL_RETURN_IF_ERROR(
      op_.Compile(env->GetDevicePtr(), &env->context(), env->program_cache()));
  return absl::OkStatus();
}

absl::Status Diffuser::UNet::Execute(CLCommandQueue* queue, int step_index,
                                     float guidance_scale, float sqrt_alpha,
                                     float sqrt_alpha_prev,
                                     float sqrt_one_minus_alpha,
                                     float sqrt_one_minus_alpha_prev) {
  {
    TensorFloat32 scalar_tensor;
    scalar_tensor.shape = BHWC(1, 1, 1, 4);
    scalar_tensor.data.resize(4, 0.0f);
    scalar_tensor.data[0] = static_cast<float>(step_index);
    ABSL_RETURN_IF_ERROR(
        inference_context_.SetInputTensor(index_val_.id, scalar_tensor, queue));
    scalar_tensor.data[0] = guidance_scale;
    ABSL_RETURN_IF_ERROR(inference_context_.SetInputTensor(
        guidance_scale_.id, scalar_tensor, queue));
    scalar_tensor.data[0] = sqrt_alpha;
    ABSL_RETURN_IF_ERROR(inference_context_.SetInputTensor(
        sqrt_alpha_.id, scalar_tensor, queue));
    scalar_tensor.data[0] = sqrt_alpha_prev;
    ABSL_RETURN_IF_ERROR(inference_context_.SetInputTensor(
        sqrt_alpha_prev_.id, scalar_tensor, queue));
    scalar_tensor.data[0] = sqrt_one_minus_alpha;
    ABSL_RETURN_IF_ERROR(inference_context_.SetInputTensor(
        sqrt_one_minus_alpha_.id, scalar_tensor, queue));
    scalar_tensor.data[0] = sqrt_one_minus_alpha_prev;
    ABSL_RETURN_IF_ERROR(inference_context_.SetInputTensor(
        sqrt_one_minus_alpha_prev_.id, scalar_tensor, queue));
  }

  return inference_context_.AddToQueue(queue);
}

}  // namespace stable_diffusion
}  // namespace cl
}  // namespace ml_drift
