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

#ifndef ML_DRIFT_COMMON_KERNELS_WINOGRAD_H_
#define ML_DRIFT_COMMON_KERNELS_WINOGRAD_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

// You can read https://arxiv.org/pdf/1509.09308.pdf for understanding of basic
// principles. In this kernels used different matrices for transformations than
// in original work.
class Winograd4x4To36 : public GPUOperation {
 public:
  Winograd4x4To36() = default;
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }
  int3 GetGridSize() const override;
  absl::Status BindArguments(ArgumentsBinder* args) override;

  // Move only
  Winograd4x4To36(Winograd4x4To36&& kernel) = default;
  Winograd4x4To36& operator=(Winograd4x4To36&& kernel) = default;
  Winograd4x4To36(const Winograd4x4To36&) = delete;
  Winograd4x4To36& operator=(const Winograd4x4To36&) = delete;

 private:
  Winograd4x4To36(const OperationDef& definition, const Padding2D& padding)
      : padding_(padding) {}
  friend Winograd4x4To36 CreateWinograd4x4To36(const OperationDef& definition,
                                               const Padding2D& padding,
                                               const GpuInfo& gpu_info);

  Padding2D padding_;
};

// Creates a Winograd 4x4 to 36 operation.
Winograd4x4To36 CreateWinograd4x4To36(const OperationDef& definition,
                                      const Padding2D& padding,
                                      const GpuInfo& gpu_info);

class Winograd3x3TiledXForward : public GPUOperation {
 public:
  Winograd3x3TiledXForward() = default;
  Winograd3x3TiledXForward(const OperationDef& definition,
                           const Padding2D& padding, const GpuInfo& gpu_info,
                           int tile_size);
  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override;

  // Move only
  Winograd3x3TiledXForward(Winograd3x3TiledXForward&& operation) = default;
  Winograd3x3TiledXForward& operator=(Winograd3x3TiledXForward&& operation) =
      default;
  Winograd3x3TiledXForward(const Winograd3x3TiledXForward&) = delete;
  Winograd3x3TiledXForward& operator=(const Winograd3x3TiledXForward&) = delete;

 private:
  friend Winograd3x3TiledXForward CreateWinograd3x3TiledXForward(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const Padding2D& padding, int tile_size);

  void UploadBt(const OperationDef& op_def);

  std::string GetCode(const OperationDef& op_def, const GpuInfo& gpu_info);

  // Must be called after kernel compilation
  int3 SelectBestWorkGroup(const KernelInfo& kernel_info) const;

  int tile_size_inner_;
  int tile_size_outer_;
  // caching Width&Height elements to local memory
  bool use_spatial_caching_ = false;
  Padding2D padding_;
};

// Creates a Winograd 3x3 tiled forward operation.
Winograd3x3TiledXForward CreateWinograd3x3TiledXForward(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const Padding2D& padding, int tile_size);

class Winograd36To4x4 : public GPUOperation {
 public:
  Winograd36To4x4() = default;
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }
  int3 GetGridSize() const override;

  // Move only
  Winograd36To4x4(Winograd36To4x4&& kernel) = default;
  Winograd36To4x4& operator=(Winograd36To4x4&& kernel) = default;
  Winograd36To4x4(const Winograd36To4x4&) = delete;
  Winograd36To4x4& operator=(const Winograd36To4x4&) = delete;

 private:
  friend Winograd36To4x4 CreateWinograd36To4x4(
      const OperationDef& definition,
      const Tensor<Linear, DataType::kFloat32>& biases);
};

// Creates a Winograd 36 to 4x4 operation.
Winograd36To4x4 CreateWinograd36To4x4(
    const OperationDef& definition,
    const Tensor<Linear, DataType::kFloat32>& biases);

class Winograd3x3TiledXBackward : public GPUOperation {
 public:
  Winograd3x3TiledXBackward() = default;
  Winograd3x3TiledXBackward(const OperationDef& definition,
                            const GpuInfo& gpu_info, int tile_size);
  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override;

  // Move only
  Winograd3x3TiledXBackward(Winograd3x3TiledXBackward&& operation) = default;
  Winograd3x3TiledXBackward& operator=(Winograd3x3TiledXBackward&& operation) =
      default;
  Winograd3x3TiledXBackward(const Winograd3x3TiledXBackward&) = delete;
  Winograd3x3TiledXBackward& operator=(const Winograd3x3TiledXBackward&) =
      delete;

 private:
  friend Winograd3x3TiledXBackward CreateWinograd3x3TiledXBackward(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const Tensor<Linear, DataType::kFloat32>& biases, int tile_size);

  void UploadAt(const OperationDef& op_def);

  std::string GetCode(const OperationDef& op_def, const GpuInfo& gpu_info);

  // Must be called after kernel compilation
  int3 SelectBestWorkGroup(const KernelInfo& kernel_info) const;

  int tile_size_inner_;
  int tile_size_outer_;
};

// Creates a Winograd 3x3 tiled backward operation.
Winograd3x3TiledXBackward CreateWinograd3x3TiledXBackward(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const Tensor<Linear, DataType::kFloat32>& biases, int tile_size);

class Winograd3x3To36 : public GPUOperation {
 public:
  Winograd3x3To36() = default;
  Winograd3x3To36(const TensorDescriptor& src_desc,
                  const TensorDescriptor& dst_desc);
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }
  int3 GetGridSize() const override {
    return int3(dst_[0]->Batch(), dst_[0]->Slices(), dst_[0]->Height());
  }

  // Move only
  Winograd3x3To36(Winograd3x3To36&& kernel) = default;
  Winograd3x3To36& operator=(Winograd3x3To36&& kernel) = default;
  Winograd3x3To36(const Winograd3x3To36&) = delete;
  Winograd3x3To36& operator=(const Winograd3x3To36&) = delete;

 private:
  void AddTransformMatrix();
};

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_WINOGRAD_H_
