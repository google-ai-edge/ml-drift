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

#ifndef ML_DRIFT_GL_GL_INFERENCE_CONTEXT_H_
#define ML_DRIFT_GL_GL_INFERENCE_CONTEXT_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "flatbuffers/buffer.h"
#include "flatbuffers/flatbuffer_builder.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_generated.h"
#include "ml_drift/common/ir_model.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/profiling_info.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/gl/gl_inference_context_generated.h"
#include "ml_drift/gl/gl_operation.h"
#include "ml_drift/gl/gl_spatial_tensor.h"
#include "ml_drift/gl/memory_manager.h"
#include "ml_drift/gl/portable_gl31.h"

namespace ml_drift {
namespace gl {

struct GlNode {
  GlOperation gl_operation;
  std::vector<ValueId> inputs;
  std::vector<ValueId> outputs;

  // Mostly for debug purposes.
  std::string name;

  GlNode() = default;

  GlNode(GlNode&& node) = default;
  GlNode& operator=(GlNode&& node) = default;
  GlNode(const GlNode&) = delete;
  GlNode& operator=(const GlNode&) = delete;
};

class GlInferenceContext {
 public:
  absl::Status InitFromGpuModel(
      const CreateGpuModelInfo& create_info, GpuModel* gpu_model,
      std::vector<uint8_t>* serialized_model = nullptr);

  absl::Status RestoreDeserialized(
      const absl::Span<const uint8_t> serialized_model,
      CreateGpuModelInfo* create_info = nullptr);

  absl::Status AddToQueue();
  absl::Status Profile(ProfilingInfo* result);
  // absl::Status Profile(ProfilingCommandQueue* queue, ProfilingInfo* result);
  // for profiling and memory statistics
  uint64_t GetSizeOfMemoryAllocatedForIntermediateTensors() const;

  absl::Status SetInputTensor(ValueId id, const TensorFloat32& tensor);

  absl::Status SetInputTensor(ValueId id,
                              const Tensor<BHWC, DataType::kInt32>& tensor);

  // It will work only with input/output tensor ids. For all other ids we don't
  // have any guarantees.
  GlSpatialTensor* GetTensor(ValueId id);

  absl::Status GetOutputTensor(ValueId id, TensorFloat32* result);
  absl::Status GetOutputTensor(ValueId id, TensorInt32* result);

  const std::vector<ValueId>& GetInputIds() const { return input_ids_; }
  const std::vector<ValueId>& GetOutputIds() const { return output_ids_; }

 private:
  absl::Status Encode(
      flatbuffers::Offset<ml_drift::data::GpuModel> gpu_model_fb,
      flatbuffers::FlatBufferBuilder* builder,
      flatbuffers::Offset<data::InferenceContext>* inf_context_fb);

  absl::Status Decode(const GpuInfo& gpu_info,
                      const data::InferenceContext* fb_inference);

  void CopyFromGpuModel(GpuModel* gpu_model);

  absl::Status BindMemoryToOperations();
  absl::Status Compile(const GpuInfo& gpu_info);

  void InitBarriers();

  // performance hacks
  bool need_flush_ = false;

  bool flush_periodically_ = false;
  int flush_period_ = 1;

  // add GL_TEXTURE_FETCH_BARRIER_BIT for TensorStorageType::BUFFER
  // bug on some mali devices(?), see b/428712357 for more details
  bool add_texture_fetch_barrier_for_buffer_ = false;

  // Directly mapped nodes from graph, but some of them "inactive" due
  //  to fusion (inactive = fused).
  // Memory is allocated only once, in ConvertOperations, and is not modified
  //  anywhere.
  std::vector<GlNode> nodes_;
  // memory barriers for glMemoryBarrier barriers_.size() = nodes_.size()
  GLbitfield in_barrier_;
  std::vector<GLbitfield> barriers_;

  MemoryManager memory_manager_;

  std::vector<ValueId> input_ids_;
  std::vector<ValueId> output_ids_;

  ProgramCache program_cache_;
};

TensorStorageType GetFastestStorageType(const GpuInfo& gpu_info);

// Converts the provided GraphFloat32 instance into a GL InferenceContext
// instance.
//
// If `weights_prep_context` is provided (not null), the function may utilize
// it to prepare weights on GPU, depending on if GPU preparation is faster than
// CPU preparation.
//
// If `weights_prep_context` is null, the weights are prepared on CPU and then
// uploaded to GPU.
absl::Status GraphToInferenceContext(
    const GpuInfo& gpu_info, const GraphFloat32& graph,
    CreateGpuModelInfo& create_info, GlInferenceContext* context,
    GlInferenceContext* weights_prep_context = nullptr);

absl::Status IrModelToInferenceContext(
    const GpuInfo& gpu_info, const ir::IrModel& ir_model,
    CreateGpuModelInfo& create_info, GlInferenceContext* context,
    GlInferenceContext* weights_prep_context = nullptr);

}  // namespace gl
}  // namespace ml_drift

#endif  // ML_DRIFT_GL_GL_INFERENCE_CONTEXT_H_
