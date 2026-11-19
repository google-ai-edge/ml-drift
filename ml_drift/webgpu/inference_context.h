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

#ifndef ML_DRIFT_WEBGPU_INFERENCE_CONTEXT_H_
#define ML_DRIFT_WEBGPU_INFERENCE_CONTEXT_H_

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "flatbuffers/buffer.h"
#include "flatbuffers/flatbuffer_builder.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_generated.h"
#include "ml_drift/common/ir_model.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/profiling_info.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/webgpu/compute_task.h"
#include "ml_drift/webgpu/environment.h"
#include "ml_drift/webgpu/memory_manager.h"
#include "ml_drift/webgpu/metrics_collector_internal.h"
#include "ml_drift/webgpu/serialization_generated.h"
#include "ml_drift/webgpu/spatial_tensor.h"
#include "ml_drift/webgpu/webgpu_headers.h"

namespace ml_drift {
namespace webgpu {

struct WebGpuNode {
  ComputeTask webgpu_operation;
  std::vector<ValueId> inputs;
  std::vector<ValueId> outputs;

  // Mostly for debug purposes.
  std::string name;

  absl::flat_hash_set<int> optional_tag;
  std::string subgraph_id;

  WebGpuNode() = default;

  WebGpuNode(WebGpuNode&& node) = default;
  WebGpuNode& operator=(WebGpuNode&& node) = default;
  WebGpuNode(const WebGpuNode&) = delete;
  WebGpuNode& operator=(const WebGpuNode&) = delete;
};

class InferenceContext {
 public:
  InferenceContext();
  explicit InferenceContext(MemoryManager* memory_manager);

  absl::Status InitFromGpuModel(
      const Environment& env, const CreateGpuModelInfo& create_info,
      GpuModel* gpu_model, std::vector<uint8_t>* serialized_model = nullptr);
  absl::Status InitFromGpuModel(
      const CreateGpuModelInfo& create_info, GpuModel* gpu_model,
      Environment* env, std::vector<uint8_t>* serialized_model = nullptr);

  // Creates command buffers which hold up to `num_nodes_per_encoder` nodes per
  // command buffer.
  // If `submit_command_buffers` is false, the command buffers will be returned
  // in a vector and the caller must submit them to the queue.
  // If `submit_command_buffers` is true, the command buffers will be submitted
  // to the queue within the function and returns an empty vector.
  struct CommandBufferInfo {
    std::set<ValueId> src_tensors;
  };
  absl::StatusOr<std::vector<wgpu::CommandBuffer>> CreateCommandBuffers(
      const Environment& environment, int num_nodes_per_encoder,
      std::vector<CommandBufferInfo>* command_buffer_infos = nullptr,
      bool submit_command_buffers = false);

  absl::Status Execute(const Environment& environment);
  absl::Status AddToQueue(const Environment& environment);
  absl::Status ProfilingAddToQueue(const Environment& environment,
                                   InternalMetricsCollector* metrics_collector);

  absl::Status Profile(const Environment& env, ProfilingInfo* result);
  uint64_t GetSizeOfMemoryAllocatedForIntermediateTensors() const;
  uint64_t GetSizeOfMemoryAllocatedForConstTensors() const;

  absl::Status SetInputTensor(const Environment& environment, ValueId id,
                              const TensorFloat32& tensor);
  absl::Status SetInputTensor(
      const Environment& environment, ValueId id,
      const ml_drift::Tensor<BHWC, DataType::kInt32>& tensor);
  absl::Status GetOutputTensor(const Environment& environment, ValueId id,
                               TensorFloat32* result);
  absl::Status GetOutputTensor(const Environment& environment, ValueId id,
                               TensorInt32* result);

  // Can be used only with ids from external_mutable_tensors in create_info
  absl::Status SetTensor(const ValueId& tensor_id, SpatialTensor* tensor);

  // It will work only with input/output tensor ids. For all other ids we don't
  // have any guarantees.
  SpatialTensor* GetTensor(ValueId id) {
    return memory_manager_.GetSpatialTensor(GetKey(id));
  }
  const TensorDescriptor& GetTensorDescriptor(ValueId id) {
    return memory_manager_.GetTensorDescriptor(GetKey(id));
  }

  const std::vector<ValueId>& GetInputIds() const { return input_ids_; }
  const std::vector<ValueId>& GetOutputIds() const { return output_ids_; }

  // Enables optional nodes with the given tags that were defined via
  // `GpuModelBuilder::BeginOptionalNodes`.
  void EnableOptionalNodes(absl::flat_hash_set<int> tags);

  absl::Status UpdateMutableObjectsBindings(const Environment& environment);
  absl::Status RestoreDeserialized(
      absl::Span<const uint8_t> serialized_model, const Environment& env,
      const CreateGpuModelInfo* create_info = nullptr);

 private:
  absl::Status Compile(const Environment& env, TuningType tuning_type);
  // Compiles the model for a serialized model. Uses the code from the
  // decoded flatbuffer instead of the code in the GPUOperation.
  absl::Status CompileForSerializedModel(
      const Environment& env, const data::InferenceContext& decoded_fb);
  // Encodes the InferenceContext to a flatbuffer.
  flatbuffers::Offset<data::InferenceContext> Encode(
      const Environment& env,
      flatbuffers::Offset<ml_drift::data::GpuModel> gpu_model_fb,
      flatbuffers::FlatBufferBuilder& builder);
  absl::Status ProfilingEncode(wgpu::CommandEncoder* encoder,
                               InternalMetricsCollector* metrics_collector);
  absl::Status ProfilingEncodeSingleCommandBuffer(
      wgpu::CommandEncoder* encoder,
      InternalMetricsCollector* metrics_collector);
  absl::Status ProfilingEncodePerOpCommandBuffer(
      wgpu::CommandEncoder* encoder,
      InternalMetricsCollector* metrics_collector);
  absl::Status ProfileWithTimestamps(const Environment& env,
                                     ProfilingInfo* result,
                                     const std::vector<int>& ops_per_encoder,
                                     int node_first_index, int node_last_index);
  // Initializes the InferenceContext from a GpuModel. If
  // `from_serialized_model` is true, code in each GPUOperation will not be
  // updated.
  absl::Status InitFromGpuModel(const Environment& env, GpuModel& gpu_model,
                                bool from_serialized_model);
  // Updates the mutable objects in the GpuModel. Allocates new memory for the
  // mutable objects if needed.
  absl::StatusOr<std::vector<std::unique_ptr<SpatialTensor>>>
  UpdateMutableObjects(const Environment& env,
                       const CreateGpuModelInfo& create_info,
                       GpuModel* gpu_model);

  absl::Status BindGpuMemory(const Environment& environment);

  MemoryManager::Key GetKey(ValueId id) const {
    return MemoryManager::Key(model_id_, id);
  }

  // Initializes the ComputeTask for all nodes referencing subgraphs.
  // If `tuning_type` is provided, the subgraphs will be tuned and compiled
  // with the given tuning type.
  absl::Status InitSubgraphNodes(
      const Environment& env,
      absl::flat_hash_map<std::string, std::vector<WebGpuNode>> subgraphs,
      TuningType* tuning_type = nullptr);

  // Initializes nodes from [start_index, start_index + subgraph_nodes.size())
  // from the nodes in `subgraph_nodes`. If this subgraph has not been compiled
  // yet, the subgraph will be compiled.
  // If `tuning_type` is provided, the subgraph will be tuned and compiled with
  // the given tuning type.
  absl::Status InitFromSubgraph(
      const Environment& env, int start_index,
      std::vector<WebGpuNode>& subgraph_nodes,
      absl::flat_hash_set<std::string>& compiled_subgraphs,
      TuningType* tuning_type = nullptr);

  std::vector<WebGpuNode> nodes_;
  std::vector<int> mutable_node_indexes_;
  struct MutableObject {
    int node_index;
    int object_index;
  };
  // value_id -> list of nodes with mutable objects
  absl::flat_hash_map<ValueId, std::vector<MutableObject>> mutable_objects_;
  std::vector<ValueId> updated_mutable_objects_;

  std::unique_ptr<MemoryManager> owned_memory_manager_;
  MemoryManager& memory_manager_;
  MemoryManager::ModelId model_id_;

  std::vector<ValueId> input_ids_;
  std::vector<ValueId> output_ids_;

  absl::flat_hash_set<int> enabled_tags_;
};

absl::Status IrModelToInferenceContext(
    Environment* env, const ::ml_drift::ir::IrModel& ir_model,
    ::ml_drift::CreateGpuModelInfo& create_info, InferenceContext* result);

}  // namespace webgpu
}  // namespace ml_drift

#endif  // ML_DRIFT_WEBGPU_INFERENCE_CONTEXT_H_
