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

#ifndef ML_DRIFT_COMMON_KERNELS_RESIZE_H_
#define ML_DRIFT_COMMON_KERNELS_RESIZE_H_

#include <string>

#include "ml_drift/common/operations.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class Resize : public GPUOperation {
 public:
  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;

  // Move only
  Resize(Resize&& operation);
  Resize& operator=(Resize&& operation);
  Resize(const Resize&) = delete;
  Resize& operator=(const Resize&) = delete;

  friend Resize CreateResize(const OperationDef& definition,
                             const Resize2DAttributes& attr);

 private:
  Resize(const OperationDef& definition, const Resize2DAttributes& attr);

  std::string GetResizeCode(const OperationDef& op_def,
                            const Resize2DAttributes& attr);

  Resize2DAttributes attr_;
};

// Returns the resize code for 2D tensors.
std::string GetResize2dCode(const Resize2DAttributes& attr,
                            const std::string& src_tensor_name,
                            const std::string& x_coord,
                            const std::string& y_coord,
                            const std::string& s_coord,
                            const std::string& result_name);

// Creates a GPU operation for the 2D resize operation.
Resize CreateResize(const OperationDef& definition,
                    const Resize2DAttributes& attr);

class Resize3D : public GPUOperation {
 public:
  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;

  // Move only
  Resize3D(Resize3D&& operation);
  Resize3D& operator=(Resize3D&& operation);
  Resize3D(const Resize3D&) = delete;
  Resize3D& operator=(const Resize3D&) = delete;

  friend Resize3D CreateResize3D(const OperationDef& definition,
                                 const Resize3DAttributes& attr);

 private:
  Resize3D(const OperationDef& definition, const Resize3DAttributes& attr);

  std::string GetResize3DCode(const OperationDef& op_def,
                              const Resize3DAttributes& attr);

  Resize3DAttributes attr_;
};

// Creates a GPU operation for the 3D resize operation.
Resize3D CreateResize3D(const OperationDef& definition,
                        const Resize3DAttributes& attr);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_RESIZE_H_
