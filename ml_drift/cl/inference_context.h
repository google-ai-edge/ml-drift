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

#ifndef ML_DRIFT_CL_INFERENCE_CONTEXT_H_
#define ML_DRIFT_CL_INFERENCE_CONTEXT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "flatbuffers/buffer.h"
#include "flatbuffers/flatbuffer_builder.h"
#include "ml_drift/cl/buffer.h"
#include "ml_drift/cl/cl_command_queue.h"
#include "ml_drift/cl/cl_device.h"
#include "ml_drift/cl/cl_event.h"
#include "ml_drift/cl/cl_operation.h"
#include "ml_drift/cl/command_buffer.h"
#include "ml_drift/cl/environment.h"
#include "ml_drift/cl/memory_manager.h"
#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/cl/program_cache.h"
#include "ml_drift/cl/serialization_generated.h"
#include "ml_drift/cl/tensor.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_generated.h"
#include "ml_drift/common/ir_model.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/model_hints.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/profiling_info.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {
namespace cl {

struct CLNode {
  ClOperation cl_operation;
  std::vector<ValueId> inputs;
  std::vector<ValueId> outputs;

  // Both `name` and `op_name` are optional fields mostly for debugging.
  // `name` is populated from `gpu_model_builder.cc`, and often only reflects
  // the characteristics of op, like type, stride, input/output channels
  // `op_name` provides users a way to set unique names for each node for easier
  // debugging.
  std::string name;
  std::string op_name;

  absl::flat_hash_set<int> optional_tag;

  CLNode() = default;

  CLNode(CLNode&& node) = default;
  CLNode& operator=(CLNode&& node) = default;
  CLNode(const CLNode&) = delete;
  CLNode& operator=(const CLNode&) = delete;
};

struct CreateInferenceInfo {
  struct Options {
    TuningType tuning_type = TuningType::kFast;
    // Can be unstable on some drivers, but can improve performance on some.
    bool allow_cl_khr_command_buffer = false;
  };
  Options options;
  ExternalTensorsInfo external_tensors;
};

CreateInferenceInfo::Options CreateFromModelHints(const ModelHints& hints);

class InferenceContext {
 public:
  explicit InferenceContext(MemoryManager* memory_manager = nullptr);
  absl::Status InitFromGpuModel(
      const CreateGpuModelInfo& create_info, GpuModel* gpu_model,
      Environment* env, std::vector<uint8_t>* serialized_model = nullptr,
      Buffer* shared_buffer = nullptr);
  absl::Status InitFromGpuModel(
      const CreateInferenceInfo& create_info, GpuModel* gpu_model,
      Environment* env, std::vector<uint8_t>* serialized_model = nullptr,
      Buffer* shared_buffer = nullptr);

  absl::Status AddToCommandBuffer(cl_command_buffer_khr cb);

  // event will be used with last enqueue operation.
  absl::Status AddToQueue(CLCommandQueue* queue, CLEvent* event = nullptr);

  // Experimental API.
  // This version of AddToQueue adds kernels one by one without any flush or
  // command buffer optimizations.
  // Context can have optional nodes. In this case they are skipped. Function
  // returns new offset after submitting N(count) active kernels. Function
  // returns -1 if all nodes were enqueued. Recommended to use new offset in
  // next call, so as it takes into account skipped optional nodes.
  // If provided, event will be used with last enqueued kernel.
  absl::StatusOr<int> AddToQueue(CLCommandQueue* queue, int offset, int count,
                                 CLEvent* event = nullptr);

  absl::Status Profile(ProfilingCommandQueue* queue, ProfilingInfo* result);
  // for profiling and memory statistics
  uint64_t GetSizeOfMemoryAllocatedForIntermediateTensors() const;
  uint64_t GetConstantTensorsSize() const;
  uint64_t GetExternalTensorsSize() const;

  // Warning: These SetInputTensor() functions don't work performantly so it's
  // not recommended to use them in production.
  absl::Status SetInputTensor(ValueId id, const TensorFloat32& tensor,
                              CLCommandQueue* queue);
  absl::Status SetInputTensor(ValueId id, const TensorBool& tensor,
                              CLCommandQueue* queue);

  absl::Status SetInputTensor(
      ValueId id, const ml_drift::Tensor<BHWC, DataType::INT32>& tensor,
      CLCommandQueue* queue);

  // It will work only with input/output tensor ids. For all other ids we don't
  // have any guarantees.
  Tensor* GetTensor(ValueId id);

  // Includes CPU-GPU synchronization.
  // Warning: These GetOutputTensor() functions don't work performantly so it's
  // not recommended to use them in production.
  absl::Status GetOutputTensor(ValueId id, CLCommandQueue* queue,
                               TensorFloat32* result);
  // Includes CPU-GPU synchronization.
  absl::Status GetOutputTensor(ValueId id, CLCommandQueue* queue,
                               TensorBool* result);
  // Includes CPU-GPU synchronization.
  absl::Status GetOutputTensor(ValueId id, CLCommandQueue* queue,
                               TensorInt32* result);

  const std::vector<ValueId>& GetInputIds() const { return input_ids_; }
  const std::vector<ValueId>& GetOutputIds() const { return output_ids_; }

  // Enables optional nodes with the given tags that were defined via
  // `GpuModelBuilder::BeginOptionalNodes`.
  void EnableOptionalNodes(absl::flat_hash_set<int> tags);

  absl::Status RestoreDeserialized(
      absl::Span<const uint8_t> serialized_model, Environment* env,
      const CreateGpuModelInfo* create_info = nullptr);

  // Can be used only with ids from external_mutable_tensors in create_info
  // Must be called after initialization and before execution
  absl::Status SetTensor(const ValueId& tensor_id, Tensor* tensor_ptr);

 private:
  flatbuffers::Offset<data::InferenceContext> Encode(
      const CLDevice& device, const ProgramCache& program_cache,
      flatbuffers::Offset<ml_drift::data::GpuModel> gpu_model_fb,
      flatbuffers::FlatBufferBuilder* builder);

  void InitFromGpuModelInternal(GpuModel* gpu_model);

  absl::Status BindMemoryToOperations();
  absl::Status Compile(const CreationContext& creation_context);
  absl::Status Tune(TuningType tuning_type, const GpuInfo& gpu_info,
                    ProfilingCommandQueue* profiling_queue);
  absl::Status UpdateParams();
  void GetMutableNodes(const absl::flat_hash_map<ValueId, TensorDescriptor>&
                           external_mutable_tensors);

  absl::Status BuildExecutionPlan(Environment* env,
                                  bool allow_cl_khr_command_buffer);

  absl::Status ProfileTime(ProfilingCommandQueue* queue, ProfilingInfo* result);
  absl::Status ClarifyTimeMultipleEnqueue(double ops_total_duration_ms,
                                          int min_ops, int max_ops,
                                          ProfilingCommandQueue* queue,
                                          ProfilingInfo* result);
  absl::Status ClarifyTimeWithCommandBuffer(ProfilingCommandQueue* queue,
                                            ProfilingInfo* result);

  bool IsMutable(const CLNode& node) const;
  void UpdateActiveTiles();

  bool HasOptionalNode() const;

  struct ExecutionHints {
    bool need_flush = false;

    bool flush_periodically = false;
    int flush_period = 1;

    // In order to reduce memory leak on Mali a pipeline needs to be
    // synchronized with CPU to prevent growing internal global OpenCL kernel
    // pool. One trick is to enqueue an event from a previous run. Most of the
    // time is should already be executed on GPU and should not stall the
    // pipeline.
    bool need_manual_release = false;
    CLEvent prev_enqueue_start_point;

    void Init(const GpuInfo& gpu_info);
  };
  ExecutionHints execution_hints_;

  // Directly mapped nodes from graph, but some of them "inactive" due
  //  to fusion (inactive = fused).
  // Memory is allocated only once, in ConvertOperations, and is not modified
  //  anywhere.
  std::vector<CLNode> nodes_;

  absl::flat_hash_map<ValueId, std::vector<int>> external_tensor_to_nodes_;

  std::vector<ValueId> input_ids_;
  std::vector<ValueId> output_ids_;

  absl::flat_hash_set<int> enabled_tags_;

  std::unique_ptr<MemoryManager> owned_memory_manager_;
  MemoryManager& memory_manager_;
  MemoryManager::ModelId model_id_ = 0;

  MemoryManager::Key GetKey(ValueId id) const {
    return MemoryManager::Key(model_id_, id);
  }

  std::vector<std::unique_ptr<CommandBuffer>> command_buffers_;

  struct ExecutionTile {
    enum class Type {
      kKernel,
      kCommandBuffer,
    };
    Type type = Type::kKernel;
    int index = 0;
    absl::flat_hash_set<int> optional_tag;
  };
  std::vector<ExecutionTile> execution_tiles_;
  std::vector<int> active_tiles_;
  absl::Status AddToQueue(const ExecutionTile& tile, CLCommandQueue* queue,
                          CLEvent* event);

  GpuInfo gpu_info_;
};

absl::Status GetInOutRefs(absl::Span<const uint8_t> serialized_model,
                          std::vector<int64_t>* in_refs,
                          std::vector<int64_t>* out_refs);

absl::Status GraphToInferenceContext(const GpuInfo& gpu_info,
                                     const GraphFloat32& graph,
                                     Environment* env,
                                     CreateGpuModelInfo& create_info,
                                     InferenceContext* context,
                                     InferenceContext* weights_prep_context);

absl::Status IrModelToInferenceContext(const GpuInfo& gpu_info,
                                       const ir::IrModel& ir_model,
                                       Environment* env,
                                       CreateGpuModelInfo& create_info,
                                       InferenceContext* context);

}  // namespace cl
}  // namespace ml_drift

#endif  // ML_DRIFT_CL_INFERENCE_CONTEXT_H_
