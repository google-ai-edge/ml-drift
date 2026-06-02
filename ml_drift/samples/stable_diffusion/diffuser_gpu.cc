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

#include "ml_drift/samples/stable_diffusion/diffuser_gpu.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "ml_drift/cl/util_types.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/samples/stable_diffusion/diffuser.h"

namespace {

using ::ml_drift::BHWC;
using ::ml_drift::TensorFloat32;
using ::ml_drift::cl::PerformanceHint;
using ::ml_drift::cl::PriorityHint;
using ::ml_drift::cl::stable_diffusion::Diffuser;

Diffuser::ModelType ToModelType(DiffuserModelType model_type) {
  switch (model_type) {
    case kDiffuserModelTypeSd1:
      return Diffuser::ModelType::kSd1;
    case kDiffuserModelTypeGldm:
      return Diffuser::ModelType::kGldm;
    case kDiffuserModelTypeDistilledGldm:
      return Diffuser::ModelType::kDistilledGldm;
    case kDiffuserModelTypeSd2Base:
      return Diffuser::ModelType::kSd2Base;
    case kDiffuserModelTypeTigo:
      return Diffuser::ModelType::kTigo;
    case kDiffuserModelTypeTigoUfo:
      return Diffuser::ModelType::kTigoUfo;
  }
  return Diffuser::ModelType::kSd1;
}

PriorityHint ToPriorityHint(DiffuserPriorityHint priority_hint) {
  switch (priority_hint) {
    case kDiffuserPriorityHintHigh:
      return PriorityHint::kHigh;
    case kDiffuserPriorityHintNormal:
      return PriorityHint::kNormal;
    case kDiffuserPriorityHintLow:
      return PriorityHint::kLow;
  }
  return PriorityHint::kNormal;
}

PerformanceHint ToPerformanceHint(DiffuserPerformanceHint priority_hint) {
  switch (priority_hint) {
    case kDiffuserPerformanceHintHigh:
      return PerformanceHint::kHigh;
    case kDiffuserPerformanceHintNormal:
      return PerformanceHint::kNormal;
    case kDiffuserPerformanceHintLow:
      return PerformanceHint::kLow;
  }
  return PerformanceHint::kHigh;
}

}  // namespace

DiffuserContext* DiffuserCreate(const DiffuserConfig* config) {
  Diffuser::Config config2;
  config2.model_type = ToModelType(config->model_type);
  config2.model_dir = config->model_dir;
  config2.lora_dir = config->lora_dir;
  if (config->lora_weights_layer_mapping) {
    config2.lora_weights_layer_mapping =
        *static_cast<const std::map<std::string, const char*>*>(
            config->lora_weights_layer_mapping);
  }
  config2.lora_rank = config->lora_rank;
  config2.seed = config->seed;
  config2.image_width = config->image_width;
  config2.image_height = config->image_height;
  config2.run_unet_with_plugins = config->run_unet_with_plugins;
  config2.run_unet_with_masked_image = config->run_unet_with_masked_image;
  config2.env_options.priority =
      ToPriorityHint(config->env_options.priority_hint);
  config2.env_options.performance =
      ToPerformanceHint(config->env_options.performance_hint);
  auto diffuser_status = Diffuser::Create(config2);
  if (!diffuser_status.ok()) {
    ABSL_LOG(ERROR) << diffuser_status.status();
    return nullptr;
  }
  auto* context = new DiffuserContext();
  context->diffuser = diffuser_status.value().release();
  return context;
}

int DiffuserReset(DiffuserContext* context, const char* prompt, int total_steps,
                  int rand_seed, float plugins_strength,
                  const void* plugin_tensors) {
  auto* diffuser = static_cast<Diffuser*>(context->diffuser);
  const auto& src_tensors =
      *static_cast<const std::vector<DiffuserPluginTensor>*>(plugin_tensors);
  if (src_tensors.empty()) {
    const auto status =
        diffuser->RunInitStep(prompt, total_steps, rand_seed, plugins_strength);
    if (!status.ok()) ABSL_LOG(ERROR) << status;
    return status.ok();
  } else {
    std::vector<TensorFloat32> dst_tensors;
    dst_tensors.reserve(src_tensors.size());
    for (const auto& src_tensor : src_tensors) {
      TensorFloat32 dst_tensor = {
          .shape = BHWC(src_tensor.shape[0], src_tensor.shape[1],
                        src_tensor.shape[2], src_tensor.shape[3]),
      };
      dst_tensor.data.resize(dst_tensor.shape.DimensionsProduct());
      std::memcpy(&dst_tensor.data[0], src_tensor.data,
                  dst_tensor.shape.DimensionsProduct() * sizeof(float));
      dst_tensors.push_back(std::move(dst_tensor));
    }
    const auto status = diffuser->RunInitStep(prompt, total_steps, rand_seed,
                                              plugins_strength, dst_tensors);
    if (!status.ok()) ABSL_LOG(ERROR) << status;
    return status.ok();
  }
}

int DiffuserIterate(DiffuserContext* context, int total_steps,
                    int curr_iteration) {
  auto* diffuser = static_cast<Diffuser*>(context->diffuser);
  return diffuser->RunIterationStep(total_steps, curr_iteration).ok();
}

int DiffuserDecode(DiffuserContext* context, uint8_t* pixel_data) {
  auto* diffuser = static_cast<Diffuser*>(context->diffuser);
  auto tensor_status = diffuser->RunDecodeStep();
  if (!tensor_status.ok()) {
    ABSL_LOG(ERROR) << tensor_status.status();
    return 0;
  }
  const auto& tensor = tensor_status.value();
  const float* src = tensor.data.data();
  for (int i = 0; i < 3 * tensor.shape.w * tensor.shape.h; ++i) {
    *pixel_data++ = 255 * std::min(std::max(0.0f, *src++), 1.0f);
  }
  return 1;
}

void DiffuserDelete(DiffuserContext* context) {
  if (context) delete static_cast<Diffuser*>(context->diffuser);
  delete context;
}
