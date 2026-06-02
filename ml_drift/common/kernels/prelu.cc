// Copyright 2024 The ML Drift Authors.
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

#include "ml_drift/common/kernels/prelu.h"

#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/types/variant.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

ElementwiseDescriptor CreatePReLU(const PReLUAttributes& attr,
                                  TensorDescriptor tensor_desc) {
  ElementwiseDescriptor op_desc;
  std::string alpha_read;
  auto alpha_linear =
      std::get_if<Tensor<Linear, DataType::FLOAT32>>(&attr.alpha);
  if (alpha_linear) {
    TensorDescriptor alpha_tensor_desc = CreateConstantLinearTensorDescriptor(
        tensor_desc.GetDataType(), tensor_desc.GetStorageType(), *alpha_linear);
    op_desc.args.AddObject("alpha", std::make_unique<TensorDescriptor>(
                                        std::move(alpha_tensor_desc)));
    alpha_read = "Type alpha_val = args.alpha.Read(S_COORD);\n";
  }

  auto alpha_hwc = std::get_if<Tensor<HWC, DataType::FLOAT32>>(&attr.alpha);
  if (alpha_hwc) {
    const BHWC shape =
        BHWC(1, alpha_hwc->shape.h, alpha_hwc->shape.w, alpha_hwc->shape.c);
    TensorDescriptor const_tensor_desc = tensor_desc;
    const_tensor_desc.UploadData(*alpha_hwc);
    op_desc.args.AddObject("alpha", std::make_unique<TensorDescriptor>(
                                        std::move(const_tensor_desc)));
    const std::string x_coord = shape.w == 1 ? "0" : "X_COORD";
    const std::string y_coord = shape.h == 1 ? "0" : "Y_COORD";
    const std::string s_coord = shape.c == 1 ? "0" : "S_COORD";
    alpha_read = absl::StrCat("Type alpha_val = args.alpha.Read(", x_coord,
                              ", ", y_coord, ", ", s_coord, ");\n");
    if (shape.c == 1) {
      alpha_read += "  alpha_val.y = alpha_val.x;\n";
      alpha_read += "  alpha_val.z = alpha_val.x;\n";
      alpha_read += "  alpha_val.w = alpha_val.x;\n";
    }
  }

  op_desc.code = alpha_read +
                 "out_value = max(ucl::Init<Type>(0.0f), in_value) + "
                 "min(ucl::Init<Type>(0.0f), "
                 "in_value) * alpha_val;";
  absl::StrReplaceAll({{"Type", ToUclDataType(tensor_desc.GetDataType(), 4)}},
                      &op_desc.code);
  return op_desc;
}

GPUOperation CreatePReLU(const GpuInfo& gpu_info,
                         const OperationDef& definition,
                         const PReLUAttributes& attr) {
  ElementwiseDescriptor op_desc = CreatePReLU(attr, definition.src_tensors[0]);
  return CreateGpuOperation(definition, std::move(op_desc));
}

}  // namespace ml_drift
