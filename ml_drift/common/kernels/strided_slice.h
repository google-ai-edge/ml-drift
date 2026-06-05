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

#ifndef ML_DRIFT_COMMON_KERNELS_STRIDED_SLICE_H_
#define ML_DRIFT_COMMON_KERNELS_STRIDED_SLICE_H_

#include <string>

#include "ml_drift/common/operations.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class StridedSlice : public GPUOperation {
 public:
  StridedSlice(const OperationDef& definition, const SliceAttributes& attr);
  StridedSlice(const OperationDef& definition, const Slice3DAttributes& attr);
  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;

  // Move only
  StridedSlice(StridedSlice&& operation);
  StridedSlice& operator=(StridedSlice&& operation);
  StridedSlice(const StridedSlice&) = delete;
  StridedSlice& operator=(const StridedSlice&) = delete;

 private:
  std::string GetStridedSliceCode(const OperationDef& op_def, bool alignedx4);
  Slice3DAttributes attributes_;
};

// Creates a StridedSlice operation.
StridedSlice CreateStridedSlice(const OperationDef& definition,
                                const SliceAttributes& attr);

StridedSlice CreateStridedSlice(const OperationDef& definition,
                                const Slice3DAttributes& attr);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_STRIDED_SLICE_H_
