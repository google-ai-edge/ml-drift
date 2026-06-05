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

#ifndef ML_DRIFT_METAL_INFERENCE_CONTEXT_H_
#define ML_DRIFT_METAL_INFERENCE_CONTEXT_H_

#import <Metal/Metal.h>

#include <cstdint>
#include <list>
#include <map>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_generated.h"
#include "ml_drift/common/ir_model.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/profiling_info.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/metal/compute_task.h"
#include "ml_drift/metal/environment.h"
#include "ml_drift/metal/inference_context_generated.h"
#include "ml_drift/metal/memory_manager.h"
#include "ml_drift/metal/metal_spatial_tensor.h"

namespace ml_drift {
namespace metal {

struct MetalNode {
  ComputeTask task;
  std::vector<ValueId> inputs;
  std::vector<ValueId> outputs;

  // Mostly for debug purposes.
  std::string name;

  absl::flat_hash_set<int> optional_tag;

  MetalNode() = default;

  MetalNode(MetalNode&& node) = default;
  MetalNode& operator=(MetalNode&& node) = default;
  MetalNode(const MetalNode&) = delete;
  MetalNode& operator=(const MetalNode&) = delete;
};

class InferenceContext {
 public:
  InferenceContext();
  explicit InferenceContext(MemoryManager* memory_manager);

  absl::Status InitFromGpuModel(
      const CreateGpuModelInfo& create_info, GpuModel* gpu_model,
      id<MTLDevice> device_id,
      std::vector<uint8_t>* serialized_model = nullptr);
  absl::Status InitFromGpuModel(
      const CreateGpuModelInfo& create_info, GpuModel* gpu_model,
      Environment* env, std::vector<uint8_t>* serialized_model = nullptr);

  absl::Status RestoreDeserialized(
      const absl::Span<const uint8_t> serialized_model, id<MTLDevice> device_id,
      const CreateGpuModelInfo* create_info = nullptr);

  /// Inserts all GPU compute kernels into the command encoder.
  /// @param inputOutputBuffers Must be created and passed into the method
  /// with pairs ID:buffer
  /// @discussion No GPU synchronization functions are used inside. All GPU
  /// resources must be created
  ///             with the same device which has been used in
  ///             compileModelWithDevice() method.
  void EncodeWithEncoder(id<MTLComputeCommandEncoder> command_encoder);

  /// Inserts all GPU compute kernels into the command buffer. For every task
  /// will be used separate
  ///   encoder.
  /// @param inputOutputBuffers Must be created and passed into the method with
  /// pairs ID:buffer
  /// @discussion No GPU synchronization functions are used inside. All GPU
  /// resources must be created
  ///             with the same device which has been used in
  ///             compileModelWithDevice() method.
  void EncodeWithCommandBuffer(id<MTLCommandBuffer> command_buffer,
                               int tasks_per_encoder = 1);

  /// Adds all GPU compute kernels to the command queue. For every task will be
  /// used separate
  ///   encoder. Few encoders(flushPeriod) batched into compute buffer that sent
  ///   for execution.
  /// @param inputOutputBuffers Must be created and passed into the method with
  /// pairs ID:buffer
  /// @discussion No GPU synchronization functions are used inside. All GPU
  /// resources must be created
  ///             with the same device which has been used in
  ///             compileModelWithDevice() method.
  void EncodeWithCommandQueue(id<MTLCommandQueue> command_queue,
                              int flush_period);

  API_AVAILABLE(ios(13.0), macos(11.00), tvos(13.0))
  void AddResources(id<MTLComputeCommandEncoder> command_encoder);
  API_AVAILABLE(ios(13.0), macos(11.00), tvos(13.0))
  void EncodeWithICB(id<MTLComputeCommandEncoder> command_encoder);

  void Profile(id<MTLDevice> device, ProfilingInfo* result);
  // Returns size in bytes for all intermediate(runtime) tensors that owned by
  // this inference context. Do not include constant tensors.
  uint64_t GetIntermediateTensorsSize() const;
  uint64_t GetConstantTensorsSize() const;

  // Can be used only with ids from external_mutable_tensors in create_info
  // Must be called after initialization and before execution
  absl::Status SetTensor(const ValueId& tensor_id,
                         MetalSpatialTensor* tensor_ptr);

  MetalSpatialTensor* GetTensor(ValueId tensor_id);
  absl::Status SetInputTensor(ValueId id, const TensorFloat32& tensor);
  absl::Status SetInputTensor(ValueId id, const TensorInt32& tensor);
  absl::Status GetOutputTensor(ValueId id, TensorFloat32* result);
  absl::Status GetOutputTensor(ValueId id, TensorInt32* result);

  // Enables optional nodes with the given tags that were defined via
  // `GpuModelBuilder::BeginOptionalNodes`.
  void EnableOptionalNodes(absl::flat_hash_set<int> tags);

 private:
  flatbuffers::Offset<data::InferenceContext> Encode(
      Environment* env,
      flatbuffers::Offset<ml_drift::data::GpuModel> gpu_model_fb,
      flatbuffers::FlatBufferBuilder* builder);

  absl::Status DecodeTasks(Environment* env,
                           const data::InferenceContext* fb_inference);

  void CopyFromGpuModel(GpuModel* gpu_model, bool use_arguments_buffer);
  absl::Status CompileOperations(Environment* env);
  void GetMutableNodes(const absl::flat_hash_map<ValueId, TensorDescriptor>&
                           external_mutable_tensors);

  void BindTensorsToOperations();
  absl::Status UpdateParams(const GpuInfo& gpu_info);
  absl::Status Tune(TuningType tuning_type, Environment* env);

  bool HasOptionalNode() const;

  void ProfileTime(id<MTLDevice> device, ProfilingInfo* result);

  API_AVAILABLE(macos(11.00), ios(14.0))
  absl::Status ProfileTimeWithTimestamps(id<MTLDevice> device,
                                         ProfilingInfo* result);

  // MTLBuffers need to be destroyed before the MTLDevice. This field is
  // first to ensure all other fields are destroyed first.
  id<MTLDevice> device_ = nullptr;

  std::vector<MetalNode> nodes_;
  std::vector<ValueId> input_ids_;
  std::vector<ValueId> output_ids_;

  absl::flat_hash_set<int> enabled_tags_;

  absl::flat_hash_map<ValueId, std::vector<int>> external_tensor_to_nodes_;

  std::unique_ptr<MemoryManager> owned_memory_manager_;
  MemoryManager& memory_manager_;
  MemoryManager::ModelId model_id_ = 0;

  MemoryManager::Key GetKey(ValueId id) const {
    return MemoryManager::Key(model_id_, id);
  }

  id<MTLIndirectCommandBuffer> icb_ = nullptr;
};

absl::Status IrModelToInferenceContext(
    const ::ml_drift::ir::IrModel& ir_model,
    ::ml_drift::CreateGpuModelInfo& create_info,
    const ::ml_drift::GpuInfo& gpu_info, id<MTLDevice> device,
    InferenceContext* result);

}  // namespace metal
}  // namespace ml_drift

#endif  // ML_DRIFT_METAL_INFERENCE_CONTEXT_H_
