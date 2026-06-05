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

#include "ml_drift/common/kernels/relu.h"

#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {

ElementwiseDescriptor CreateReLU(const ReLUAttributes& attr,
                                 DataType in_out_value_type) {
  ElementwiseDescriptor result;
  std::string min_func;
  if (attr.alpha != 0.0f) {
    result.args.AddFloat("alpha", attr.alpha, in_out_value_type);
    min_func = "min(in_value * args.alpha, ucl::Init<Type>(0.0f))";
  } else {
    result.args.AddFloat("activation_min", attr.activation_min,
                         in_out_value_type);
    min_func = "ucl::Init<Type>(args.activation_min)";
  }
  if (attr.activation_max != 0.0f) {
    result.args.AddFloat("activation_max", attr.activation_max,
                         in_out_value_type);
    result.code = absl::StrCat("out_value = clamp(in_value, " + min_func +
                               ", ucl::Init<Type>(args.activation_max));");
  } else {
    result.code = absl::StrCat("out_value = max(in_value, ", min_func, ");");
  }
  absl::StrReplaceAll({{"Type", ToUclDataType(in_out_value_type, 4)}},
                      &result.code);
  return result;
}

GPUOperation CreateReLU(const OperationDef& definition,
                        const ReLUAttributes& attr) {
  ElementwiseDescriptor op_desc =
      CreateReLU(attr, definition.src_tensors[0].GetDataType());
  return CreateGpuOperation(definition, std::move(op_desc));
}

}  // namespace ml_drift
