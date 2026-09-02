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
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/string_view.h"
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
#include "ml_drift/common/model.h"
#include "ml_drift/common/model_hints.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/profiling_info.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/samples/stable_diffusion/autoencoder_kl_builder.h"
#include "ml_drift/samples/stable_diffusion/diffuser.h"
#include "ml_drift/samples/stable_diffusion/text_guidance_builder.h"
#include "ml_drift/samples/stable_diffusion/unet_builder.h"
#include "ml_drift/samples/stable_diffusion/util.h"

namespace ml_drift {
namespace cl {

class OpHolder {
 public:
  struct ExecutionParams {
    std::vector<Tensor*> src;
    std::vector<Tensor*> dst;
    std::map<std::string, float> float_params;
    std::map<std::string, half> half_params;
    std::map<std::string, int> int_params;
  };
  absl::Status InitElementwiseScalar(OperationType op_type,
                                     const TensorDescriptor& src_desc,
                                     const TensorDescriptor& dst_desc,
                                     Environment* env) {
    ElementwiseAttributes attr;
    attr.param = 0.3f;
    OperationDef op_def;
    op_def.src_tensors.push_back(src_desc);
    op_def.dst_tensors.push_back(dst_desc);
    GPUOperation operation =
        CreateElementwise(env->device().GetInfo(), op_def, op_type, attr);
    return InitClOp(env, std::move(operation));
  }
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

  absl::Status InitElementwiseTwoInput(OperationType op_type,
                                       const TensorDescriptor& src0_desc,
                                       const TensorDescriptor& src1_desc,
                                       const TensorDescriptor& dst_desc,
                                       Environment* env) {
    OperationDef op_def;
    op_def.src_tensors.push_back(src0_desc);
    op_def.src_tensors.push_back(src1_desc);
    op_def.dst_tensors.push_back(dst_desc);
    GPUOperation operation =
        CreateElementwiseTwoInput(env->device().GetInfo(), op_def, op_type,
                                  BHWC(2, 2, 2, 2), BHWC(2, 2, 2, 2));
    return InitClOp(env, std::move(operation));
  }

  absl::Status Execute(CLCommandQueue* queue, float scalar, Tensor* src,
                       Tensor* dst) {
    ABSL_RETURN_IF_ERROR(cl_op_.SetSrcTensor(0, src));
    ABSL_RETURN_IF_ERROR(cl_op_.SetDstTensor(0, dst));
    ABSL_RETURN_IF_ERROR(cl_op_.SetHalf("scalar", scalar));
    ABSL_RETURN_IF_ERROR(cl_op_.UpdateParams());
    ABSL_RETURN_IF_ERROR(cl_op_.AddToQueue(queue));
    return absl::OkStatus();
  }

  absl::Status Execute(CLCommandQueue* queue, const ExecutionParams& params) {
    for (int i = 0; i < params.src.size(); ++i) {
      ABSL_RETURN_IF_ERROR(cl_op_.SetSrcTensor(i, params.src[i]));
    }
    for (int i = 0; i < params.dst.size(); ++i) {
      ABSL_RETURN_IF_ERROR(cl_op_.SetDstTensor(i, params.dst[i]));
    }
    for (const auto& float_param : params.float_params) {
      ABSL_RETURN_IF_ERROR(
          cl_op_.SetFloat(float_param.first, float_param.second));
    }
    for (const auto& half_param : params.half_params) {
      ABSL_RETURN_IF_ERROR(cl_op_.SetHalf(half_param.first, half_param.second));
    }
    for (const auto& int_param : params.int_params) {
      ABSL_RETURN_IF_ERROR(cl_op_.SetInt(int_param.first, int_param.second));
    }
    ABSL_RETURN_IF_ERROR(cl_op_.UpdateParams());
    ABSL_RETURN_IF_ERROR(cl_op_.AddToQueue(queue));
    return absl::OkStatus();
  }

  absl::Status Execute(CLCommandQueue* queue, Tensor* src, Tensor* dst) {
    ABSL_RETURN_IF_ERROR(cl_op_.SetSrcTensor(0, src));
    ABSL_RETURN_IF_ERROR(cl_op_.SetDstTensor(0, dst));
    ABSL_RETURN_IF_ERROR(cl_op_.UpdateParams());
    ABSL_RETURN_IF_ERROR(cl_op_.AddToQueue(queue));
    return absl::OkStatus();
  }

  absl::Status Execute(CLCommandQueue* queue, Tensor* src0, Tensor* src1,
                       Tensor* dst) {
    ABSL_RETURN_IF_ERROR(cl_op_.SetSrcTensor(0, src0));
    ABSL_RETURN_IF_ERROR(cl_op_.SetSrcTensor(1, src1));
    ABSL_RETURN_IF_ERROR(cl_op_.SetDstTensor(0, dst));
    ABSL_RETURN_IF_ERROR(cl_op_.UpdateParams());
    ABSL_RETURN_IF_ERROR(cl_op_.AddToQueue(queue));
    return absl::OkStatus();
  }

 private:
  absl::Status InitClOp(Environment* env, GPUOperation&& operation) {
    auto gpu_op = std::make_unique<GPUOperation>(std::move(operation));

    ABSL_RETURN_IF_ERROR(gpu_op->AssembleCode(env->device().GetInfo()));

    cl_op_.Init(std::move(gpu_op));
    ABSL_RETURN_IF_ERROR(cl_op_.Compile(env->GetDevicePtr(), &env->context(),
                                        env->program_cache()));
    return absl::OkStatus();
  }
  ClOperation cl_op_;
};


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
  absl::Status Init(Environment* env, const std::string& file_folder) {
    const auto& gpu_info = env->GetDevicePtr()->GetInfo();

    CreateGpuModelInfo create_info;
    create_info.precision = CalculationsPrecision::F16;
    create_info.storage_type = GetFastestStorageType(gpu_info);
    create_info.hints.Add(ModelHints::kFastTuning);
    // create_info.hints.Add(ModelHints::kNoWinogradOptimizations);
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
    ABSL_RETURN_IF_ERROR(inference_context_.InitFromGpuModel(
        create_info, &gpu_model, env, nullptr));
    const auto end_init = std::chrono::high_resolution_clock::now();
    std::cout << "TextGuidance initialization time: "
              << (end_init - start_init).count() * 1e-6f << " ms." << std::endl;

    if (/* DISABLES CODE */ (false)) {
      PrintProfilingInfo(inference_context_, env->profiling_queue());
    }

    return absl::OkStatus();
  }

  Tensor* GetGuidanceTensor() { return inference_context_.GetTensor(dst_.id); }

  absl::Status Execute(Environment* env,
                       const ml_drift::Tensor<BHWC, DataType::INT32>& src,
                       TensorFloat32* dst) {
    ABSL_RETURN_IF_ERROR(
        inference_context_.SetInputTensor(src_.id, src, env->queue()));

    ABSL_RETURN_IF_ERROR(inference_context_.AddToQueue(env->queue()));
    ABSL_RETURN_IF_ERROR(env->queue()->WaitForCompletion());

    ABSL_RETURN_IF_ERROR(
        inference_context_.GetOutputTensor(dst_.id, env->queue(), dst));
    return absl::OkStatus();
  }

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

    GpuModelBuilderOptions options;
    options.hints = create_info.hints;
    options.storage = create_info.storage_type;
    options.use_f32_accum_for_f16_convolutions =
        (create_info.precision == CalculationsPrecision::F32_F16);
    ml_drift::GpuModelBuilder model_builder(gpu_info, options);

    index_val_ = model_builder.AddTensor(BHWC(1, 1, 1, 4), DataType::FLOAT32);
    ABSL_ASSIGN_OR_RETURN(
        auto result,
        model_builder.AppendOp(
            "TembGeneration", {index_val_},
            {{"channels", 320},
             {"storage_type",
              static_cast<int>(GetFastestStorageType(gpu_info))}}));
    temb_ = result[0];

    GpuModel gpu_model;
    ABSL_RETURN_IF_ERROR(builder.Build(
        config, gpu_info, create_info, width, height, &model_builder, &src_,
        &temb_, &guidance_,
        /*text_proj_ptr=*/nullptr,
        /*masked_image_latent_ptr=*/nullptr, &eta0_, &eta1_, nullptr));

    guidance_scale_ =
        model_builder.AddTensor(BHWC(1, 1, 1, 4), DataType::FLOAT32);
    sqrt_alpha_ = model_builder.AddTensor(BHWC(1, 1, 1, 4), DataType::FLOAT32);
    sqrt_alpha_prev_ =
        model_builder.AddTensor(BHWC(1, 1, 1, 4), DataType::FLOAT32);
    sqrt_one_minus_alpha_ =
        model_builder.AddTensor(BHWC(1, 1, 1, 4), DataType::FLOAT32);
    sqrt_one_minus_alpha_prev_ =
        model_builder.AddTensor(BHWC(1, 1, 1, 4), DataType::FLOAT32);
    src_input_ = src_;
    ABSL_ASSIGN_OR_RETURN(
        auto _step_res,
        model_builder.AppendOp(
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
    ABSL_RETURN_IF_ERROR(
        model_builder.GetGpuModel(in_ids, out_ids, &gpu_model));

    const auto start_init = std::chrono::high_resolution_clock::now();
    ABSL_RETURN_IF_ERROR(inference_context_.InitFromGpuModel(
        create_info, &gpu_model, env, nullptr));
    const auto end_init = std::chrono::high_resolution_clock::now();
    std::cout << "UNet initialization time: "
              << (end_init - start_init).count() * 1e-6f << " ms." << std::endl;

    if (/* DISABLES CODE */ (false)) {
      PrintProfilingInfo(inference_context_, env->profiling_queue());
    }

    return absl::OkStatus();
  }

  Tensor* GetLatentTensor() {
    return inference_context_.GetTensor(src_input_.id);
  }
  Tensor* GetOutputLatentTensor() {
    return inference_context_.GetTensor(src_.id);
  }
  Tensor* GetInputLatentTensor() {
    return inference_context_.GetTensor(src_input_.id);
  }
  Tensor* GetTembTensor() { return inference_context_.GetTensor(temb_.id); }
  Tensor* GetGuidanceTensor() {
    return inference_context_.GetTensor(guidance_.id);
  }
  Tensor* GetEtaUncondTensor() {
    return inference_context_.GetTensor(eta0_.id);
  }
  Tensor* GetEtaCondTensor() { return inference_context_.GetTensor(eta1_.id); }

  absl::Status Execute(Environment* env, const TensorFloat32& latent,
                       const TensorFloat32& guidance,
                       const TensorFloat32& temb_in, TensorFloat32* eta_uncond,
                       TensorFloat32* eta_cond) {
    ABSL_RETURN_IF_ERROR(
        inference_context_.SetInputTensor(src_input_.id, latent, env->queue()));
    ABSL_RETURN_IF_ERROR(
        inference_context_.SetInputTensor(temb_.id, temb_in, env->queue()));
    ABSL_RETURN_IF_ERROR(inference_context_.SetInputTensor(
        guidance_.id, guidance, env->queue()));

    ABSL_RETURN_IF_ERROR(inference_context_.AddToQueue(env->queue()));
    ABSL_RETURN_IF_ERROR(env->queue()->WaitForCompletion());

    ABSL_RETURN_IF_ERROR(
        inference_context_.GetOutputTensor(eta0_.id, env->queue(), eta_uncond));
    ABSL_RETURN_IF_ERROR(
        inference_context_.GetOutputTensor(eta1_.id, env->queue(), eta_cond));
    return absl::OkStatus();
  }

  absl::Status Execute(CLCommandQueue* queue, int step_index,
                       float guidance_scale, float sqrt_alpha,
                       float sqrt_alpha_prev, float sqrt_one_minus_alpha,
                       float sqrt_one_minus_alpha_prev) {
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
    return inference_context_.AddToQueue(queue);
  }

 private:
  GpuModelBuilder::TensorHandle index_val_, guidance_scale_, sqrt_alpha_,
      sqrt_alpha_prev_, sqrt_one_minus_alpha_, sqrt_one_minus_alpha_prev_;
  GpuModelBuilder::TensorHandle src_, src_input_, temb_, guidance_, eta0_,
      eta1_;
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
    // create_info.hints.Add(ModelHints::kNoWinogradOptimizations);
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
    ABSL_RETURN_IF_ERROR(inference_context_.InitFromGpuModel(
        create_info, &gpu_model, env, nullptr));
    const auto end_init = std::chrono::high_resolution_clock::now();
    std::cout << "Decoder initialization time: "
              << (end_init - start_init).count() * 1e-6f << " ms." << std::endl;

    if (/* DISABLES CODE */ (false)) {
      PrintProfilingInfo(inference_context_, env->profiling_queue());
    }

    return absl::OkStatus();
  }

  Tensor* GetInputTensor() { return inference_context_.GetTensor(src_.id); }

  absl::Status Execute(Environment* env, const TensorFloat32& src,
                       TensorFloat32* dst) {
    ABSL_RETURN_IF_ERROR(
        inference_context_.SetInputTensor(src_.id, src, env->queue()));

    ABSL_RETURN_IF_ERROR(inference_context_.AddToQueue(env->queue()));
    ABSL_RETURN_IF_ERROR(env->queue()->WaitForCompletion());

    ABSL_RETURN_IF_ERROR(
        inference_context_.GetOutputTensor(dst_.id, env->queue(), dst));
    return absl::OkStatus();
  }

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

absl::Status RunStableDiffusion(absl::string_view pipeline_type) {
  ml_drift::cl::stable_diffusion::Diffuser::Config config;
  config.model_dir = std::string(pipeline_type);

  ABSL_ASSIGN_OR_RETURN(
      auto diffuser, ml_drift::cl::stable_diffusion::Diffuser::Create(config));

  unsigned int base_seed = 1;
  std::srand(base_seed);

  while (true) {
    std::string prompt = "a photo of an astronaut riding a horse on mars";
    std::string negative_prompt = "";
    int steps = 50;

    std::cout << "Enter phrase: ";
    if (!std::getline(std::cin, prompt)) break;
    if (prompt.empty()) continue;
    std::cout << "Enter steps: ";
    if (!(std::cin >> steps)) break;
    std::string dummy;
    std::getline(std::cin, dummy);

    int seed = std::rand() % 60000;

    ABSL_ASSIGN_OR_RETURN(
        auto result,
        diffuser->Diffuse(prompt, steps, seed, std::nullopt, {}, true));

    GenerateImage(result, "result.bmp");
    std::cout << "ready" << std::endl;
  }

  return absl::OkStatus();
}

}  // namespace cl
}  // namespace ml_drift

int main() {
  auto load_status = ml_drift::cl::LoadOpenCL();
  if (!load_status.ok()) {
    std::cout << load_status.message();
    return -1;
  }

  auto status = ml_drift::cl::RunStableDiffusion("sd_1_5/");
  if (!status.ok()) {
    std::cout << status.message() << std::endl;
    return -1;
  }
  return EXIT_SUCCESS;
}
