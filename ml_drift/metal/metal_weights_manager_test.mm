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

#include "ml_drift/metal/metal_weights_manager.h"

#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>

#import <XCTest/XCTest.h>
#import <Metal/Metal.h>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
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
#include "ml_drift/metal/environment.h"
#include "ml_drift/metal/inference_context.h"
#include "ml_drift/metal/metal_spatial_tensor.h"

@interface MetalWeightsManagerTest : XCTestCase
@end

@implementation MetalWeightsManagerTest

using ml_drift::metal::Environment;
using ml_drift::GpuInfo;
using ml_drift::metal::InferenceContext;
using ml_drift::metal::MetalSpatialTensor;
using ml_drift::metal::MetalWeightsManager;
using ml_drift::TensorFloat32;
using ml_drift::TensorStorageType;
using ml_drift::WeightsDescription;
using ml_drift::WeightsLayout;
using ml_drift::CreateGpuModelInfo;
using ml_drift::CalculationsPrecision;
using ml_drift::DataType;
using ml_drift::OHWI;
using ml_drift::ValueId;
using ml_drift::WeightsManager;
using ml_drift::GpuModel;
using ml_drift::TensorDescriptor;

- (void)testBatchExecutionOutputsSameWithInferenceContextExecution {
  // Testing environment setup.
  Environment env;
  const GpuInfo& gpu_info = env.GetInfo();
  id<MTLCommandQueue> command_queue = [env.device() newCommandQueue];

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
  MetalWeightsManager weights_manager;
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
  absl::Status status = weights_manager.CreateConversionGpuModel(
      gpu_info, &conversion_gpu_model, &preparation_id_to_main_id_map,
      &upload_weights_info);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());

  absl::flat_hash_map<ValueId, ValueId> main_id_to_preparation_id_map;
  for (const auto& [preparation_id, main_id] : preparation_id_to_main_id_map) {
    main_id_to_preparation_id_map[main_id] = preparation_id;
  }

  InferenceContext conversion_context;
  status = conversion_context.InitFromGpuModel(create_info, &conversion_gpu_model, &env);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());

  for (const auto& upload_info : upload_weights_info) {
    auto tensor = conversion_context.GetTensor(upload_info.input_id);
    status = tensor->WriteData(command_queue, upload_info.data, /*wait_for_completion=*/false);
    XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  }

  id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
  conversion_context.EncodeWithCommandBuffer(command_buffer);
  [command_buffer commit];
  [command_buffer waitUntilCompleted];

  // Batch execution.
  auto weights_map_or = weights_manager.PrepareWeightsInBatches(
      env, WeightsManager::ScheduleStrategy::kDefaultBatch, 0);
  XCTAssertTrue(weights_map_or.ok(), @"%s", std::string(weights_map_or.status().message()).c_str());
  auto weights_map = std::move(weights_map_or).value();

  // Compare results of InferenceContext execution and Batch execution.
  for (int i = 0; i < num_weights_to_prepare; ++i) {
    TensorFloat32 ic_exec_output;
    status = conversion_context.GetOutputTensor(main_id_to_preparation_id_map[i], &ic_exec_output);
    XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());

    auto tensor = static_cast<MetalSpatialTensor*>(weights_map[i].get());
    TensorDescriptor desc;
    status = tensor->ToDescriptor(&desc, env.device());
    XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
    TensorFloat32 batch_exec_output;
    desc.DownloadData(&batch_exec_output);

    XCTAssertEqual(ic_exec_output.data.size(), batch_exec_output.data.size());
    for (size_t j = 0; j < ic_exec_output.data.size(); ++j) {
      XCTAssertEqualWithAccuracy(ic_exec_output.data[j], batch_exec_output.data[j], 2e-5);
    }
  }

}

@end
