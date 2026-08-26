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

#ifndef ML_DRIFT_COMMON_TASK_GPU_OPERATION_H_
#define ML_DRIFT_COMMON_TASK_GPU_OPERATION_H_

#include <stdint.h>

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "flatbuffers/buffer.h"
#include "flatbuffers/flatbuffer_builder.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/compiler_options.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/common/task/serialization_base_generated.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/types.h"

namespace ml_drift {
// kCustom: default value
//   GPUOperation::GetGridSize must be overloaded
// kWBToX_HDToY_SToZ:
//   grid_x = dst_[0]->Width() * dst_[0]->Batch();
//   grid_y = dst_[0]->Height() * dst_[0]->Depth();
//   grid_z = dst_[0]->Slices();
// kWBToX_HDToY_ZIs1:
//   grid_x = dst_[0]->Width() * dst_[0]->Batch();
//   grid_y = dst_[0]->Height() * dst_[0]->Depth();
//   grid_z = 1;
// kWBToX_HToY_DToZ:
//   grid_x = dst_[0]->Width() * dst_[0]->Batch();
//   grid_y = dst_[0]->Height();
//   grid_z = dst_[0]->Depth();
// kBToX_YIs1_ZIs1:
//   grid_x = dst_[0]->Batch();
//   grid_y = 1;
//   grid_z = 1;
enum class TensorToGrid {
  kCustom,
  kWBToX_HDToY_SToZ,
  kWBToX_HDToY_ZIs1,
  kWBToX_HToY_DToZ,
  kBToX_YIs1_ZIs1
};

struct OperationDef {
  std::vector<TensorDescriptor> src_tensors;
  std::vector<TensorDescriptor> dst_tensors;
};

struct ElementwiseDescriptor {
  Arguments args;
  std::string code;
};

struct CodeInfo {
  bool uses_global_id = false;
  bool uses_local_id = false;
  bool uses_group_id = false;
  bool uses_group_size = false;
  bool uses_sub_group_local_id = false;
  bool uses_sub_group_id = false;
  bool uses_sub_group_size = false;
};

struct LinkableContext {
  std::string code;
  TensorDescriptor* tensor_desc = nullptr;
  TensorDescriptor* final_tensor_desc = nullptr;  // for reorder.
};

class GPUOperation {
 public:
  GPUOperation() = default;
  virtual ~GPUOperation() = default;
  // Move only
  GPUOperation(GPUOperation&& operation);
  GPUOperation& operator=(GPUOperation&& operation);
  GPUOperation(const GPUOperation&) = delete;
  GPUOperation& operator=(const GPUOperation&) = delete;

  absl::Status AddOperation(const GpuInfo& gpu_info, GPUOperation* operation);

  // fuse 2 reorder operations in single one
  absl::Status AddReorderOperation(const BHWC& interm_shape,
                                   GPUOperation* operation);

  int GetElementwiseInputsCount() const { return elementwise_inputs_; }

  void SetSrc(GpuSpatialTensor* ptr, size_t index = 0);
  void SetDst(GpuSpatialTensor* ptr, size_t index = 0);

  struct DispatchInfo {
    int3 work_group_size;
    int3 work_groups_count;
  };
  void GetPossibleDispatches(TuningType tuning_type, const GpuInfo& gpu_info,
                             const KernelInfo& kernel_info,
                             std::vector<DispatchInfo>* dispatches) const;

  const std::vector<std::string>& GetSrcTensorsNames() const {
    return src_objects_names_;
  }
  const std::vector<std::string>& GetDstTensorsNames() const {
    return dst_objects_names_;
  }
  const std::vector<GpuSpatialTensor*>& GetSrcTensors() const { return src_; }
  const std::vector<GpuSpatialTensor*>& GetDstTensors() const { return dst_; }
  const int3& GetWorkGroupsCount() const { return work_groups_count_; }

  absl::Status GetTensorDescriptor(const std::string& tensor_name,
                                   TensorDescriptor* result) const;

  absl::Status SetOutputDescriptor(size_t index,
                                   const TensorDescriptor& new_tensor_desc);

  absl::Status AssembleCode(const GpuInfo& gpu_info);

  void AddSrcTensor(const std::string& tensor_name,
                    const TensorDescriptor& desc);
  void AddSrcBuffer(const std::string& buffer_name,
                    const BufferDescriptor& desc);
  void AddDstTensor(const std::string& tensor_name,
                    const TensorDescriptor& desc);
  void AddDstBuffer(const std::string& buffer_name,
                    const BufferDescriptor& desc);

  bool IsLinkable() const { return elementwise_; }
  void SetReorderCode(const std::string& code) {
    reorder_code_ = code;
    reorder_op_ = true;
  }
  const std::string& GetReorderCode() const { return reorder_code_; }
  void AddInputReorder(const std::string& input_name, std::string reorder_code,
                       TensorDescriptor* src_tensor_desc,
                       TensorDescriptor* dst_tensor_desc);
  bool IsReorderOp() const { return reorder_op_; }
  void ResolveReorderFinalShape(const BHWC& final_shape);
  void ResolveReorderFinalShape(const BHWDC& final_shape);

  void AllowFuseInputReorder(bool value = true) {
    allow_fuse_input_reorder_ = value;
  }
  bool FuseInputReorderAllowed() const { return allow_fuse_input_reorder_; }

  virtual absl::Status BindArguments(ArgumentsBinder* args) {
    return absl::OkStatus();
  }
  void RecalculateGridSize() { grid_size_ = GetGridSize(); }
  void SetGridSize(const int3& grid_size) {
    grid_size_ = grid_size;
    tensor_to_grid_ = TensorToGrid::kCustom;
  }
  void RecalculateWorkGroupsCount();

  virtual std::string_view GetDebugName() const { return ""; }

  Arguments args_;
  std::string code_;
  int3 work_group_size_ = int3(8, 4, 1);
  std::vector<CompilerOptions> compiler_options_;
  // not applicable to elementwise
  TensorToGrid tensor_to_grid_ = TensorToGrid::kCustom;

  // for profiling
  uint64_t flops_ = 0;
  // size in bytes of constant gpu_objects inside args_
  uint64_t const_args_size_ = 0;
  uint64_t read_size_ = -1;
  uint64_t write_size_ = -1;

  // Must be called before const generic objects in args_ released.
  void CalculateConstArgsSize();

  CodeInfo code_info_;

 protected:
  friend flatbuffers::Offset<data::GPUOperation> Encode(
      const GPUOperation& op, flatbuffers::FlatBufferBuilder* builder);
  friend absl::Status Decode(const data::GPUOperation* fb_op, GPUOperation* op);
  friend GPUOperation CreateGpuOperation(const OperationDef& definition,
                                         ElementwiseDescriptor&& descriptor);
  friend GPUOperation CreateGpuOperation(const OperationDef& definition,
                                         ElementwiseDescriptor&& descriptor,
                                         const BHWDC& second_shape,
                                         const BHWDC& dst_shape);
  friend GPUOperation CreateGpuOperation(
      const TensorDescriptor& in_out_descriptor,
      const BufferDescriptor& src_buffer_descriptor,
      ElementwiseDescriptor&& descriptor);
  friend GPUOperation CreateReorderGpuOperation(const OperationDef& definition,
                                                std::string&& code);

  friend absl::Status FuseElemWithElemInternal(
      const GpuInfo& gpu_info, GPUOperation&& elem0, GPUOperation&& elem1,
      const std::vector<std::pair<std::string, std::string>>& replacements,
      GPUOperation* result);
  friend absl::Status FuseSimpleElemWithSimpleElem(const GpuInfo& gpu_info,
                                                   GPUOperation&& elem0,
                                                   GPUOperation&& elem1,
                                                   GPUOperation* result);
  friend absl::Status Fuse2InputElemWithSimpleElemAsFirstInput(
      const GpuInfo& gpu_info, GPUOperation&& elem0, GPUOperation&& elem1,
      GPUOperation* result);
  friend absl::Status Fuse2InputElemWithSimpleElemAsSecondInput(
      const GpuInfo& gpu_info, GPUOperation&& elem0, GPUOperation&& elem1,
      GPUOperation* result);
  friend absl::Status Fuse2InputElemWith2SimpleElem(const GpuInfo& gpu_info,
                                                    GPUOperation&& elem0,
                                                    GPUOperation&& elem1,
                                                    GPUOperation&& elem_root,
                                                    GPUOperation* result);

  virtual int3 GetGridSize() const;
  virtual std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const;

  std::vector<GpuSpatialTensor*> src_;
  std::vector<GpuSpatialTensor*> dst_;
  int grid_dimension_ = 3;  // can be 1, 2 or 3
  int3 work_group_launch_order_ = int3(0, 1, 2);
  int3 grid_size_ = int3(0, 0, 0);

 private:
  absl::Status GetTensorDescriptor(const std::string& tensor_name,
                                   TensorDescriptor** result);
  absl::Status ResolveSecondElementwiseInput(const GpuInfo& gpu_info);

  std::vector<std::string> src_objects_names_;
  std::vector<std::string> dst_objects_names_;

  int3 work_groups_count_ = int3(0, 0, 0);
  bool elementwise_ = false;      // temporary, used during op construction
  int elementwise_inputs_ = 0;    // can be {0, 1, 2}
  std::string
      second_elementwise_tensor_name_;  // used with elementwise_inputs_ = 2
  int linkable_count_ = 0;        // temporary, used during op construction
  std::string elementwise_code_;  // temporary, used during op construction

  bool reorder_op_ = false;   // temporary, used during op construction
  int reorder_op_count_ = 0;  // temporary, used during op construction
  std::string reorder_code_;  // temporary, used during op construction
  absl::flat_hash_map<std::string, LinkableContext> input_reorder_linkables_;

  bool const_expr_resolved_ = false;  // temporary, used during op construction
  bool allow_fuse_input_reorder_ =
      false;  // temporary, used during op construction
};

GPUOperation CreateGpuOperation(const TensorDescriptor& src,
                                const TensorDescriptor& dst,
                                ElementwiseDescriptor&& descriptor);
GPUOperation CreateGpuOperation(const OperationDef& definition,
                                ElementwiseDescriptor&& descriptor);

// For creating elementwise operations with 2 runtime inputs
GPUOperation CreateGpuOperation(const OperationDef& definition,
                                ElementwiseDescriptor&& descriptor,
                                const BHWDC& second_shape,
                                const BHWDC& dst_shape);

GPUOperation CreateGpuOperation(const OperationDef& definition,
                                ElementwiseDescriptor&& descriptor,
                                const BHWC& second_shape,
                                const BHWC& dst_shape);

GPUOperation CreateGpuOperation(const TensorDescriptor& in_out_descriptor,
                                const BufferDescriptor& src_buffer_descriptor,
                                ElementwiseDescriptor&& descriptor);

GPUOperation CreateReorderGpuOperation(const OperationDef& definition,
                                       std::string&& code);

absl::Status FuseElemWithElemInternal(
    const GpuInfo& gpu_info, GPUOperation&& elem0, GPUOperation&& elem1,
    const std::vector<std::pair<std::string, std::string>>& replacements,
    GPUOperation* result);

//    input       input
//      |           |
//    elem0         |
//      |    -->  elem
//    elem1         |
//      |           |
//    output      output
absl::Status FuseSimpleElemWithSimpleElem(const GpuInfo& gpu_info,
                                          GPUOperation&& elem0,
                                          GPUOperation&& elem1,
                                          GPUOperation* result);

//      input           input
//     /    \             |
//  elem0    |            |
//     \    /      -->  elem
//     elem1              |
//       |                |
//     output           output
absl::Status Fuse2InputElemWithSimpleElemAsFirstInput(const GpuInfo& gpu_info,
                                                      GPUOperation&& elem0,
                                                      GPUOperation&& elem1,
                                                      GPUOperation* result);

//      input           input
//     /    \             |
//    |    elem0          |
//     \    /      -->  elem
//     elem1              |
//       |                |
//     output           output
absl::Status Fuse2InputElemWithSimpleElemAsSecondInput(const GpuInfo& gpu_info,
                                                       GPUOperation&& elem0,
                                                       GPUOperation&& elem1,
                                                       GPUOperation* result);

//      input           input
//     /    \             |
//  elem0  elem1          |
//     \    /      -->  elem
//   elem_root            |
//       |                |
//     output           output
absl::Status Fuse2InputElemWith2SimpleElem(const GpuInfo& gpu_info,
                                           GPUOperation&& elem0,
                                           GPUOperation&& elem1,
                                           GPUOperation&& elem_root,
                                           GPUOperation* result);

struct ExternalWeights {
  WeightsDescription desc;
  OHWI shape;
  // scale and zero_point(optional) are for quantized weights
  OHWI scale_zp_shape = OHWI(1, 1, 1, 1);
  const TensorDescriptor* scale = nullptr;
  const TensorDescriptor* zero_point = nullptr;
  std::optional<float> scalar_scale = std::nullopt;
  std::optional<float> scalar_zero_point = std::nullopt;
};

// A struct to allow runtime-configurable channel bounds with conv or
// fully-connected ops.
//
// If `src_end_ch_index` is provided, then the end slice will contain the
// channel value at `src_end_ch_index` for the given parameters tensor rather
// than the channel size of the source tensor.
//
// If `dst_end_ch_index` is provided, then the value at `dst_end_ch_index` for
// the parameters tensor will be used as the end channel rather than the channel
// size of the destination tensor. NOTE: Tensor values outside the range [0, dst
// end channel) will not be well-defined. It is the developer's responsibility
// to use tensor values that are within bounds.
struct ConvRuntimeCheckDesc {
  std::optional<int> src_end_ch_index = std::nullopt;
  std::optional<int> dst_end_ch_index = std::nullopt;
  static constexpr int kChannelsAlignment = 32;

  bool HasValues() const {
    return src_end_ch_index.has_value() || dst_end_ch_index.has_value();
  }
  int GetSlicesAlignment() const;
  std::string GetRuntimeEndSlice(const std::string& channels,
                                 const std::string& max_slices) const;

  // Experimental feature.
  // for using with batched weights. if offset is set, then runtime sizes for
  // every group can be read from the tensor with the given offset.
  struct PackedGroups {
    int params_offset;  // runtime sizes/offset for every group in param buffer.
                        // [num_groups] sizes + [num_groups] offsets
    int num_groups;     // equal to batch size of the weights
    int max_group_size;
  };
  std::optional<PackedGroups> packed_groups;

  // Experimental feature.
  std::optional<int> ring_o_offset_index = std::nullopt;
  std::optional<int> ring_i_offset_index = std::nullopt;
  std::optional<int> ring_size = std::nullopt;
};

// A struct to allow runtime-configurable channel bounds with softmax ops.
//
// If `end_ch_index` is provided, then the value at `end_ch_index` for a given
// parameters tensor will be used as the end channel rather than the channel
// size of the source tensor.
//
// Note that tensor values outside of the range [0, end channel)
// will not be well-defined. It is the developer's responsibility to use
// tensor values that are within bounds.
struct SoftmaxRuntimeCheckDesc {
  std::optional<int> end_ch_index = std::nullopt;
};

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_TASK_GPU_OPERATION_H_
