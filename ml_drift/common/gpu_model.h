// Copyright 2024 The ML Drift Authors.
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

#ifndef ML_DRIFT_COMMON_GPU_MODEL_H_
#define ML_DRIFT_COMMON_GPU_MODEL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "flatbuffers/buffer.h"
#include "flatbuffers/flatbuffer_builder.h"
#include "ml_drift/common/gpu_model_generated.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/model_hints.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/common/task/tensor_desc.h"

namespace ml_drift {

struct GpuNode {
  std::unique_ptr<GPUOperation> gpu_operation;
  std::vector<ValueId> inputs;
  std::vector<ValueId> outputs;
  // Both `name` and `op_name` are optional fields mostly for debugging.
  // `name` is populated from `gpu_model_builder.cc`, and often only reflects
  // the characteristics of op, like type, stride, input/output channels
  // `op_name` provides users a way to set unique names for each node for easier
  // debugging.
  std::string name;
  std::string op_name;

  // A set of tag values set if this is an optional node. The node should only
  // run if all tags in the set are enabled.
  absl::flat_hash_set<int> optional_tag;
  std::string subgraph_id;

  GpuNode() = default;
  GpuNode(GpuNode&& node) = default;
  GpuNode& operator=(GpuNode&& node) = default;
  GpuNode(const GpuNode&) = delete;
  GpuNode& operator=(const GpuNode&) = delete;
};

struct CreateGpuModelInfo {
  CalculationsPrecision precision;
  TensorStorageType storage_type;
  ModelHints hints;

  // User can require specific layout for some tensors.
  // This will guarantee that tensors with specific ids have exact specified
  // layout.
  // Some restrictions apply:
  //   1) ValueId must be input or output id of GraphFloat32
  //   2) data_type must be equal to DeduceDataTypeFromPrecision(precision);
  //      for example for precision F16, data_type must be FLOAT16
  //   3) Layout must be without Batch dimension if tensor.shape.b == 1
  //      Layout must be with Batch dimension if tensor.shape.b != 1
  // InitFromGraph will fail if gpu can not allocate tensor with requested
  // tensor descriptor
  // WARNING: This is an experimental API and subject to change.
  // IMPORTANT: tensors ids from predefined / external_immutable_tensors /
  // external_mutable_tensors should not intersect.
  absl::flat_hash_map<ValueId, TensorDescriptor> predefined;

  // User can provide immutable external tensors for inference context.
  // Some restrictions apply:
  //   1) ValueId must be input or output id of GraphFloat32
  //   2) Provided ptrs must be valid during life of InferenceContext.
  //   3) data_type must be equal to DeduceDataTypeFromPrecision(precision);
  //      for example for precision F16, data_type must be FLOAT16
  //   4) Layout must be without Batch dimension if tensor.shape.b == 1
  //      Layout must be with Batch dimension if tensor.shape.b != 1
  // InitFromGraph will fail if gpu can not allocate tensor with requested
  // tensor descriptor
  // WARNING: This is an experimental API and subject to change.
  // IMPORTANT: tensors ids from predefined / external_immutable_tensors /
  // external_mutable_tensors should not intersect.
  absl::flat_hash_map<ValueId, GpuSpatialTensor*> external_immutable_tensors;

  // User can provide mutable external tensors for inference context.
  // HINT: Highly recommended to use other options if possible, this options
  // will be with the worst performance.
  // Some restrictions apply:
  //   1) ValueId must be input or output id of GraphFloat32
  //   2) data_type must be equal to DeduceDataTypeFromPrecision(precision);
  //      for example for precision F16, data_type must be FLOAT16
  //   3) Layout must be without Batch dimension if tensor.shape.b == 1
  //      Layout must be with Batch dimension if tensor.shape.b != 1
  // InitFromGraph will fail if gpu can not allocate tensor with requested
  // tensor descriptor
  // WARNING: This is an experimental API and subject to change.
  // IMPORTANT: tensors ids from predefined / external_immutable_tensors /
  // external_mutable_tensors should not intersect.
  absl::flat_hash_map<ValueId, TensorDescriptor> external_mutable_tensors;
};

// TODO(sorokin): Remove the same fields from CreateGpuModelInfo.
struct ExternalTensorsInfo {
  absl::flat_hash_map<ValueId, GpuSpatialTensor*> immutable_tensors;
  absl::flat_hash_map<ValueId, TensorDescriptor> mutable_tensors;
};

struct GpuModel {
  std::vector<std::pair<ValueId, ValueId>> input_ids_and_refs;
  std::vector<std::pair<ValueId, ValueId>> output_ids_and_refs;
  std::vector<GpuNode> nodes;
  absl::flat_hash_map<ValueId, TensorDescriptor> tensors;
  absl::flat_hash_map<ValueId, TensorDescriptor> const_tensors;
  absl::flat_hash_map<std::string, GpuModel> subgraphs;
};

flatbuffers::Offset<data::GpuModel> Encode(
    const GpuModel& gpu_model, flatbuffers::FlatBufferBuilder* builder);

absl::Status Decode(const data::GpuModel* fb_gpu_model, GpuModel* gpu_model);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_GPU_MODEL_H_
