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

#include "ml_drift/common/kernels/cast.h"

#include <string>
#include <utility>
#include <vector>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {
GPUOperation CreateCast(const OperationDef& definition,
                        const GpuInfo& gpu_info) {
  ElementwiseDescriptor op_desc;
  DataType dst_type = definition.dst_tensors[0].GetDataType();
  if (!(gpu_info.IsApiMetal() &&
        gpu_info.metal_info.IsNativeBfloatSupported())) {
    dst_type = dst_type == DataType::BFLOAT16 ? DataType::FLOAT32 : dst_type;
  }
  op_desc.code = "out_value = ucl::Convert<" + ToUclDataType(dst_type, 4) +
                 ">(in_value);\n";
  return CreateGpuOperation(definition, std::move(op_desc));
}

}  // namespace ml_drift
