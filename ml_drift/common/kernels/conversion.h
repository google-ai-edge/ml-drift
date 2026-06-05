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

#ifndef ML_DRIFT_COMMON_KERNELS_CONVERSION_H_
#define ML_DRIFT_COMMON_KERNELS_CONVERSION_H_

#include <vector>

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/work_group_picking.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

class TensorToTensor : public GPUOperation {
 public:
  TensorToTensor() = default;

  // Returns the grid size for the given tensor shape.
  int3 GetGridSizeFromShape(const BHWC& tensor_shape) const {
    return int3(tensor_shape.w * tensor_shape.b, tensor_shape.h,
                DivideRoundUp(tensor_shape.c, 4));
  }

  // Recalculates the work group size for the given tensor shape.
  void RecalculateWorkGroupsSize(const GpuInfo& gpu_info,
                                 const BHWC& tensor_shape) {
    int3 grid = GetGridSizeFromShape(tensor_shape);
    if (grid.x == 0 || grid.y == 0 || grid.z == 0) {
      work_group_size_ = int3(8, 4, 1);
      return;
    }
    KernelInfo kernel_info;
    kernel_info.max_work_group_size = gpu_info.GetMaxWorkGroupTotalSize();
    kernel_info.private_memory_size = 0;
    const auto work_groups = GetPossibleWorkGroupsConv(
        TuningType::kFast, gpu_info, kernel_info, grid);
    work_group_size_ = work_groups[0];
  }

  // Returns the work group size.
  int3 GetWorkGroupSize() const { return work_group_size_; }

  // Move only
  TensorToTensor(TensorToTensor&& kernel) = default;
  TensorToTensor& operator=(TensorToTensor&& kernel) = default;
  TensorToTensor(const TensorToTensor&) = delete;
  TensorToTensor& operator=(const TensorToTensor&) = delete;
};

class TensorToBhwcBuffer : public TensorToTensor {
 public:
  TensorToBhwcBuffer() = default;
  int3 GetGridSize() const override {
    if (is_aligned_) {
      const int total_size = src_[0]->Width() * src_[0]->Batch() *
                             src_[0]->Height() * src_[0]->Channels();
      return int3(DivideRoundUp(total_size, 4), 1, 1);
    } else {
      return int3(src_[0]->Width() * src_[0]->Batch(), src_[0]->Height(),
                  src_[0]->Slices());
    }
  }

  // Move only
  TensorToBhwcBuffer(TensorToBhwcBuffer&& kernel) = default;
  TensorToBhwcBuffer& operator=(TensorToBhwcBuffer&& kernel) = default;
  TensorToBhwcBuffer(const TensorToBhwcBuffer&) = delete;
  TensorToBhwcBuffer& operator=(const TensorToBhwcBuffer&) = delete;

  bool is_aligned_ = false;
};

class BhwcBufferToTensor : public TensorToTensor {
 public:
  BhwcBufferToTensor() = default;

  // Move only
  BhwcBufferToTensor(BhwcBufferToTensor&& kernel) = default;
  BhwcBufferToTensor& operator=(BhwcBufferToTensor&& kernel) = default;
  BhwcBufferToTensor(const BhwcBufferToTensor&) = delete;
  BhwcBufferToTensor& operator=(const BhwcBufferToTensor&) = delete;
};

// Creates a GPU operation for converting a tensor from one layout to another.
TensorToTensor CreateTensorToTensorOp(const GpuInfo& gpu_info,
                                      const TensorDescriptor& src_desc,
                                      const TensorDescriptor& dst_desc);

// Creates a GPU operation for converting a tensor to a BHWC buffer.
TensorToBhwcBuffer CreateTensorToBhwcBufferOp(const GpuInfo& gpu_info,
                                              const TensorDescriptor& src_desc,
                                              const BufferDescriptor& dst_desc);
// Creates a GPU operation for converting a tensor to an aligned BHWC buffer.
TensorToBhwcBuffer CreateTensorToBhwcBufferAlignedOp(
    const GpuInfo& gpu_info, const TensorDescriptor& src_desc,
    const BufferDescriptor& dst_desc);

// Creates a GPU operation for converting a BHWC buffer to a tensor.
BhwcBufferToTensor CreateBhwcBufferToTensorOp(const GpuInfo& gpu_info,
                                              const BufferDescriptor& src_desc,
                                              const TensorDescriptor& dst_desc);
// Creates a GPU operation for converting an aligned BHWC buffer to a tensor.
BhwcBufferToTensor CreateBhwcBufferAlignedToTensorOp(
    const GpuInfo& gpu_info, const BufferDescriptor& src_desc,
    const TensorDescriptor& dst_desc);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_CONVERSION_H_
