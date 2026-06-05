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

#ifndef ML_DRIFT_COMMON_KERNELS_REDUCE_H_
#define ML_DRIFT_COMMON_KERNELS_REDUCE_H_

#include <map>
#include <set>
#include <string>
#include <vector>

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class Reduce : public GPUOperation {
 public:
  enum class Type {
    kAll,  // AND operation. Only works with bool
    kAny,  // OR operation. Only works with bool
    kMean,
    kMeanSquares,
    kSum,
    kProduct,
    kMaximum,
    kMinimum,
    kMaximumIndex  // Works with single dimension only
  };

  // Returns a string representation of the Reduce type.
  static inline const char* TypeToString(Type type) {
    switch (type) {
      case Type::kAll:
        return "reduce_all";
      case Type::kAny:
        return "reduce_any";
      case Type::kMean:
        return "reduce_mean";
      case Type::kMeanSquares:
        return "reduce_mean_squares";
      case Type::kSum:
        return "reduce_sum";
      case Type::kProduct:
        return "reduce_product";
      case Type::kMaximum:
        return "reduce_maximum";
      case Type::kMinimum:
        return "reduce_minimum";
      case Type::kMaximumIndex:
        return "reduce_maximum_index";
      default:
        return "reduce_unknown";
    }
  }

  Reduce() = default;
  // Creates a new Reduce operation.
  Reduce(const std::map<Axis, int>& axis_to_reduce, Type reduce_type,
         const OperationDef& definition, const GpuInfo& gpu_info,
         bool add_input = false);

  // Returns the possible kernel work groups for the given tuning type and GPU
  // info.
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override;
  // Binds the arguments for the GPU operation.
  absl::Status BindArguments(ArgumentsBinder* args) override;
  // Returns the grid size for the GPU operation.
  int3 GetGridSize() const override;

  // Move only
  Reduce(Reduce&& operation);
  Reduce& operator=(Reduce&& operation);
  Reduce(const Reduce&) = delete;
  Reduce& operator=(const Reduce&) = delete;

 private:
  std::string GetReduceKernelCode(const OperationDef& op_def,
                                  const GpuInfo& gpu_info,
                                  const int3& work_group_size,
                                  const std::vector<Axis>& axis_to_reduce,
                                  Type reduce_type, bool add_input = false);

  bool use_wg_reduction_;
};

// Returns the Reduce type from the given OperationType.
Reduce::Type GetReduceTypeFromOperationType(OperationType op_type);

// Creates a Reduce operation for a BHWC tensor.
Reduce CreateReduce(const std::set<Axis>& axis_to_reduce, const BHWC& src_shape,
                    OperationType op_type, const OperationDef& definition,
                    const GpuInfo& gpu_info);

// Creates a Reduce operation for a BHWC tensor.
Reduce CreateReduce(const std::set<Axis>& axis_to_reduce, const BHWC& src_shape,
                    Reduce::Type reduce_type, const OperationDef& definition,
                    const GpuInfo& gpu_info);

// Creates a Reduce operation for a BHWDC tensor.
Reduce CreateReduce(const std::set<Axis>& axis_to_reduce,
                    const BHWDC& src_shape, OperationType op_type,
                    const OperationDef& definition, const GpuInfo& gpu_info);

// Creates a Reduce operation for a BHWDC tensor.
Reduce CreateReduce(const std::set<Axis>& axis_to_reduce,
                    const BHWDC& src_shape, Reduce::Type reduce_type,
                    const OperationDef& definition, const GpuInfo& gpu_info);

// Creates a Reduce operation with two inputs for a BHWC tensor.
Reduce Create2InputReduce(const std::set<Axis>& axis_to_reduce,
                          const BHWC& src_shape, OperationType op_type,
                          const OperationDef& definition,
                          const GpuInfo& gpu_info);
}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_REDUCE_H_
