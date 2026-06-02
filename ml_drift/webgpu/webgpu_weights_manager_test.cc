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

#include "ml_drift/webgpu/webgpu_weights_manager.h"

#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status_matchers.h"
#include "absl/time/time.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/webgpu/execution_environment.h"
#include "ml_drift/webgpu/inference_context.h"
#include "ml_drift/webgpu/spatial_tensor.h"
#include "ml_drift/webgpu/webgpu_api_util.h"
#include "ml_drift/webgpu/webgpu_headers.h"

namespace ml_drift::webgpu {
namespace {

wgpu::BackendType GetBackendType() {
  wgpu::BackendType backend_type = wgpu::BackendType::Vulkan;
#ifdef __APPLE__
  backend_type = wgpu::BackendType::Metal;
#elif defined(_WIN32)
  backend_type = wgpu::BackendType::DX12;
#endif
  return backend_type;
}

TEST(WebGpuWeightsManagerTest,
     BatchExecutionOutputsSameWithInferenceContextExecution) {
  // Testing environment setup.
  ExecutionEnvironment env(GetBackendType());
  MLD_ASSERT_OK(env.Initialize({.enable_host_mapped_pointer = true}));
  const GpuInfo& gpu_info = env.GetInfo();

  // Define testing parameters.
  CreateGpuModelInfo create_info;
  create_info.storage_type = TensorStorageType::BUFFER;
  create_info.precision = CalculationsPrecision::F32;
  size_t num_weights_to_prepare = 8;
  OHWI weights_shape = OHWI(100, 1, 1, 100);
  size_t num_weights_elements = weights_shape.DimensionsProduct();
  WeightsDescription weights_desc = {
      .type = DataType::FLOAT32,
      .layout = WeightsLayout::kOSpatialIOGroupO4I4,
      .output_group_size = 16,
  };

  // Register weights conversion to WeightsManager.
  WebGpuWeightsManager weights_manager;
  std::vector<std::vector<float>> raw_data_vecs;
  raw_data_vecs.reserve(num_weights_to_prepare);
  for (int w = 0; w < num_weights_to_prepare; ++w) {
    ValueId main_model_weight_id = w;
    std::vector<float> raw_data(num_weights_elements);
    for (int i = 0; i < num_weights_elements; ++i) {
      raw_data[i] = std::sin(i * 0.1f + w) + w;
    }
    raw_data_vecs.push_back(std::move(raw_data));
    weights_manager.RegisterWeightsConversion(
        {main_model_weight_id}, weights_desc, weights_shape, DataType::FLOAT32,
        raw_data_vecs.back().data());
  }

  // InferenceContext execution.
  absl::flat_hash_map<ValueId, ValueId> preparation_id_to_main_id_map;
  std::vector<WeightsManager::UploadWeightsInfo> upload_weights_info;
  GpuModel conversion_gpu_model;
  MLD_ASSERT_OK(weights_manager.CreateConversionGpuModel(
      gpu_info, &conversion_gpu_model, &preparation_id_to_main_id_map,
      &upload_weights_info));
  absl::flat_hash_map<ValueId, ValueId> main_id_to_preparation_id_map;
  for (const auto& [preparation_id, main_id] : preparation_id_to_main_id_map) {
    main_id_to_preparation_id_map[main_id] = preparation_id;
  }

  InferenceContext conversion_context;
  MLD_ASSERT_OK(conversion_context.InitFromGpuModel(env, create_info,
                                                &conversion_gpu_model));
  for (const auto& upload_info : upload_weights_info) {
    auto tensor = conversion_context.GetTensor(upload_info.input_id);
    MLD_ASSERT_OK(tensor->WriteData(env.queue(), upload_info.data));
  }
  MLD_ASSERT_OK(conversion_context.AddToQueue(env));
  MLD_ASSERT_OK(WaitUntilCompleted(env.queue(), env.device(), absl::Seconds(10)));

  // Batch execution.
  MLD_ASSERT_OK_AND_ASSIGN(
      auto weights_map,
      weights_manager.PrepareWeightsInBatches(
          env, WeightsManager::ScheduleStrategy::kDefaultBatch));

  for (int i = 0; i < num_weights_to_prepare; ++i) {
    TensorFloat32 ic_exec_output;
    MLD_ASSERT_OK(conversion_context.GetOutputTensor(
        env, main_id_to_preparation_id_map[i], &ic_exec_output));

    auto tensor = static_cast<SpatialTensor*>(weights_map[i].get());
    TensorDescriptor desc;
    MLD_ASSERT_OK(tensor->ToDescriptor(env.device(), &desc));
    TensorFloat32 batch_exec_output;
    desc.DownloadData(&batch_exec_output);

    EXPECT_THAT(
        batch_exec_output.data,
        testing::Pointwise(testing::FloatNear(2e-5), ic_exec_output.data));
  }
}

}  // namespace
}  // namespace ml_drift::webgpu
