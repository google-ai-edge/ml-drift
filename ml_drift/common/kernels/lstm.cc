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

#include "ml_drift/common/kernels/lstm.h"

#include <string>

#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {

namespace {
std::string GetLSTMCode(const OperationDef& op_def, const GpuInfo& gpu_info) {
  std::string c;
  c += "MAIN_FUNCTION(\n";
  c += "$0) {\n";
  c += "  int B = ucl::GetGlobalId<0>();\n";
  c += "  int Z = ucl::GetGlobalId<2>();\n";
  c += "  if (Z >= args.activation.Slices() || B >= args.activation.Batch()) "
       "return;\n";
  c += "  Type prev_st = args.prev_state.Read(0, 0, Z, B);\n";
  c += "  Type r0 = args.intermediate.Read(0, 0, Z, B);\n";
  c += "  int state_stride = args.activation.Slices();\n";
  c += "  Type r1 = args.intermediate.Read(0, 0, Z + state_stride, B);\n";
  c += "  Type r2 = args.intermediate.Read(0, 0, Z + state_stride * 2, B);\n";
  c += "  Type r3 = args.intermediate.Read(0, 0, Z + state_stride * 3, B);\n";
  if (gpu_info.IsApiOpenCl() &&
      op_def.src_tensors[0].GetDataType() != DataType::FLOAT32 &&
      gpu_info.IsAdreno()) {
    c += "  Type input_gate;\n";
    c += "  Type new_input;\n";
    c += "  Type forget_gate;\n";
    c += "  Type output_gate;\n";
    c += "  input_gate.x = native_recip(1.0h + native_exp(-r0.x));\n";
    c += "  input_gate.y = native_recip(1.0h + native_exp(-r0.y));\n";
    c += "  input_gate.z = native_recip(1.0h + native_exp(-r0.z));\n";
    c += "  input_gate.w = native_recip(1.0h + native_exp(-r0.w));\n";
    c += "  new_input.x = 1.0h - 2.0h * native_recip(1.0h + native_exp(2.0h * "
         "r1.x));\n";
    c += "  new_input.y = 1.0h - 2.0h * native_recip(1.0h + native_exp(2.0h * "
         "r1.y));\n";
    c += "  new_input.z = 1.0h - 2.0h * native_recip(1.0h + native_exp(2.0h * "
         "r1.z));\n";
    c += "  new_input.w = 1.0h - 2.0h * native_recip(1.0h + native_exp(2.0h * "
         "r1.w));\n";
    c += "  forget_gate.x = native_recip(1.0h + native_exp(-r2.x));\n";
    c += "  forget_gate.y = native_recip(1.0h + native_exp(-r2.y));\n";
    c += "  forget_gate.z = native_recip(1.0h + native_exp(-r2.z));\n";
    c += "  forget_gate.w = native_recip(1.0h + native_exp(-r2.w));\n";
    c += "  output_gate.x = native_recip(1.0h + native_exp(-r3.x));\n";
    c += "  output_gate.y = native_recip(1.0h + native_exp(-r3.y));\n";
    c += "  output_gate.z = native_recip(1.0h + native_exp(-r3.z));\n";
    c += "  output_gate.w = native_recip(1.0h + native_exp(-r3.w));\n";
  } else {
    c +=
        "  Type input_gate  = ucl::Init<Type>(1.0f) / (ucl::Init<Type>(1.0f) + "
        "exp(ucl::Init<Type>(-1.0f) "
        "* r0));\n";
    c += "  Type new_input   = tanh(r1);\n";
    c +=
        "  Type forget_gate = ucl::Init<Type>(1.0f) / (ucl::Init<Type>(1.0f) + "
        "exp(ucl::Init<Type>(-1.0f) "
        "* r2));\n";
    c +=
        "  Type output_gate = ucl::Init<Type>(1.0f) / (ucl::Init<Type>(1.0f) + "
        "exp(ucl::Init<Type>(-1.0f) "
        "* r3));\n";
  }
  c += "  Type new_st = input_gate * new_input + forget_gate * prev_st;\n";
  c += "  Type act_value = output_gate * tanh(new_st);\n";
  c += "  args.activation.Write(act_value, 0, 0, Z, B);\n";
  c += "  args.new_state.Write(new_st, 0, 0, Z, B);\n";
  c += "}\n";
  absl::StrReplaceAll(
      {{"Type", ToUclDataType(op_def.dst_tensors[0].GetDataType(), 4)}}, &c);
  return c;
}

}  // namespace

GPUOperation CreateLSTM(const OperationDef& definition,
                        const GpuInfo& gpu_info) {
  GPUOperation op;
  op.AddSrcTensor("intermediate", definition.src_tensors[0]);
  op.AddSrcTensor("prev_state", definition.src_tensors[1]);
  op.AddDstTensor("new_state", definition.dst_tensors[0]);
  op.AddDstTensor("activation", definition.dst_tensors[1]);
  op.code_ = GetLSTMCode(definition, gpu_info);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace ml_drift
