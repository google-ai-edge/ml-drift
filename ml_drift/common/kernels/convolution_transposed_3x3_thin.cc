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

#include "ml_drift/common/kernels/convolution_transposed_3x3_thin.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace {
std::string ConvInstr(CalculationsPrecision precision, bool is_i4_o4,
                      const std::string& dst_name, const std::string& src_name,
                      int weights_offset) {
  std::string c;
  if (is_i4_o4) {
    switch (precision) {
      case CalculationsPrecision::F32:
      case CalculationsPrecision::F16:
        c += "  $0 += $1.x * args.weights.Read($2); \n";
        c += "  $0 += $1.y * args.weights.Read($3); \n";
        c += "  $0 += $1.z * args.weights.Read($4); \n";
        c += "  $0 += $1.w * args.weights.Read($5); \n";
        break;
      case CalculationsPrecision::F32_F16:
        c += "  $0 += ucl::Convert<AccType>($1.x * args.weights.Read($2) + "
             "$1.y * "
             "args.weights.Read($3) + $1.z * args.weights.Read($4) + $1.w * "
             "args.weights.Read($5)); \n";
        break;
    }
  } else {
    // O4I4
    c += "  $0.x += dot($1, args.weights.Read($2)); \n";
    c += "  $0.y += dot($1, args.weights.Read($3)); \n";
    c += "  $0.z += dot($1, args.weights.Read($4)); \n";
    c += "  $0.w += dot($1, args.weights.Read($5)); \n";
  }
  return absl::Substitute(c, dst_name, src_name, weights_offset,
                          weights_offset + 1, weights_offset + 2,
                          weights_offset + 3);
}

MemoryType GetWeightsMemoryType(const GpuInfo& gpu_info) {
  return gpu_info.IsApiWebGpu() ? MemoryType::GLOBAL : MemoryType::CONSTANT;
}
}  // namespace

ConvolutionTransposed3x3Thin::ConvolutionTransposed3x3Thin(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const ConvolutionTransposedAttributes& attr) {
  if (gpu_info.IsDotPreferred()) {
    weights_layout_ = WeightsLayout::kOICustomSpatialO4I4;
  } else {
    weights_layout_ = WeightsLayout::kOICustomSpatialI4O4;
  }
  weights_data_type_ = DeduceDataTypeFromPrecision(precision);
  code_ = GenerateConvolutionTransposedCode(
      definition, precision, gpu_info, DivideRoundUp(attr.weights.shape.i, 4),
      DivideRoundUp(attr.weights.shape.o, 4), !attr.bias.data.empty());
}

std::string ConvolutionTransposed3x3Thin::GenerateConvolutionTransposedCode(
    const OperationDef& op_def, CalculationsPrecision precision,
    const GpuInfo& gpu_info, int src_depth, int dst_depth, bool has_bias) {
  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);

  if (op_def.src_tensors.size() == 2) {
    // dynamic weights
    BufferDescriptor desc;
    desc.element_type = op_def.src_tensors[1].GetDataType();
    desc.element_size = 4;
    desc.memory_type = GetWeightsMemoryType(gpu_info);
    AddSrcBuffer("weights", desc);
  }

  std::string c;

  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = ucl::GetGlobalId<0>();\n";
  }
  c += "  int Y = ucl::GetGlobalId<1>();\n";
  c += "  if (X >= args.src_tensor.Width() || Y >= args.src_tensor.Height()) "
       "return;\n";
  for (int d = 0; d < dst_depth; ++d) {
    const std::string layer = std::to_string(d);
    c += "  AccType r" + layer + "[2][2];\n";
    c += "  r" + layer + "[0][0] = ucl::Init<AccType>(0.0f);\n";
    c += "  r" + layer + "[0][1] = ucl::Init<AccType>(0.0f);\n";
    c += "  r" + layer + "[1][0] = ucl::Init<AccType>(0.0f);\n";
    c += "  r" + layer + "[1][1] = ucl::Init<AccType>(0.0f);\n";
  }
  for (int s = 0; s < src_depth; ++s) {
    const std::string z = std::to_string(s);
    c += "  {\n";
    if (op_def.src_tensors[0].SupportsZeroClamp(Axis::WIDTH, gpu_info) &&
        op_def.src_tensors[0].SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
      c += "  Type src0 = args.src_tensor.Read(X, Y, " + z + ");\n";
      c += "  Type src1 = args.src_tensor.Read(X + 1, Y, " + z + ");\n";
      c += "  Type src2 = args.src_tensor.Read(X, Y + 1, " + z + ");\n";
      c += "  Type src3 = args.src_tensor.Read(X + 1, Y + 1, " + z + ");\n";
    } else if (op_def.src_tensors[0].IsLinear() &&
               op_def.src_tensors[0].ReturnsZeroForNegOneRead(gpu_info)) {
      c += "  int c0 = args.src_tensor.GetAddress(X, Y, " + z + ");\n";
      c += "  int c1 = args.src_tensor.GetAddress(X + 1, Y, " + z + ");\n";
      c += "  int c2 = args.src_tensor.GetAddress(X, Y + 1, " + z + ");\n";
      c += "  int c3 = args.src_tensor.GetAddress(X + 1, Y + 1, " + z + ");\n";
      c += "  bool x_in = X + 1 < args.src_tensor.Width();\n";
      c += "  bool y_in = Y + 1 < args.src_tensor.Height();\n";
      c += "  c1 = select(-1, c1, x_in);\n";
      c += "  c2 = select(-1, c2, y_in);\n";
      c += "  c3 = select(-1, c3, x_in && y_in);\n";
      c += "  Type src0 = args.src_tensor.Read(c0);\n";
      c += "  Type src1 = args.src_tensor.Read(c1);\n";
      c += "  Type src2 = args.src_tensor.Read(c2);\n";
      c += "  Type src3 = args.src_tensor.Read(c3);\n";
    } else {
      // Manual zero clamp
      c += "  bool x_in = X + 1 < args.src_tensor.Width();\n";
      c += "  bool y_in = Y + 1 < args.src_tensor.Height();\n";
      c += "  Type src0 = args.src_tensor.Read(X, Y, " + z + ");\n";
      c += "  Type src1 = ucl::Init<Type>(0.0f);\n";
      c += "  Type src2 = ucl::Init<Type>(0.0f);\n";
      c += "  Type src3 = ucl::Init<Type>(0.0f);\n";
      c += "  if (x_in) {\n";
      c += "    src1 = args.src_tensor.Read(X + 1, Y, " + z + ");\n";
      c += "  }\n";
      c += "  if (y_in) {\n";
      c += "    src2 = args.src_tensor.Read(X, Y + 1, " + z + ");\n";
      c += "  }\n";
      c += "  if (x_in && y_in) {\n";
      c += "    src3 = args.src_tensor.Read(X + 1, Y + 1, " + z + ");\n";
      c += "  }\n";
    }
    for (int d = 0; d < dst_depth; ++d) {
      const std::string layer = std::to_string(d);
      const int filters_index = (s * dst_depth + d) * 36;
      const bool is_i4_o4 = GetWeightsDescription().IsI4O4();
      c += ConvInstr(precision, is_i4_o4, "r" + layer + "[0][0]", "src0",
                     filters_index);
      c += ConvInstr(precision, is_i4_o4, "r" + layer + "[0][1]", "src0",
                     filters_index + 4);
      c += ConvInstr(precision, is_i4_o4, "r" + layer + "[0][1]", "src1",
                     filters_index + 8);
      c += ConvInstr(precision, is_i4_o4, "r" + layer + "[1][0]", "src0",
                     filters_index + 12);
      c += ConvInstr(precision, is_i4_o4, "r" + layer + "[1][0]", "src2",
                     filters_index + 16);
      c += ConvInstr(precision, is_i4_o4, "r" + layer + "[1][1]", "src0",
                     filters_index + 20);
      c += ConvInstr(precision, is_i4_o4, "r" + layer + "[1][1]", "src1",
                     filters_index + 24);
      c += ConvInstr(precision, is_i4_o4, "r" + layer + "[1][1]", "src2",
                     filters_index + 28);
      c += ConvInstr(precision, is_i4_o4, "r" + layer + "[1][1]", "src3",
                     filters_index + 32);
    }
    c += "  }\n";
  }
  c += "  X *= 2;\n";
  c += "  Y *= 2;\n";
  for (int d = 0; d < dst_depth; ++d) {
    const std::string layer = std::to_string(d);
    c += "  {\n";
    if (has_bias) {
      c += "  Type bias_val = args.biases.Read(" + layer + ");\n";
    } else {
      c += "  Type bias_val = ucl::Init<Type>(0.0f);\n";
    }
    for (int y = 0; y < 2; ++y) {
      for (int x = 0; x < 2; ++x) {
        const std::string x_coord = "X + " + std::to_string(x);
        const std::string y_coord = "Y + " + std::to_string(y);
        c += "  {\n";
        c += "    Type result = ucl::Convert<Type>(r" + layer + "[" +
             std::to_string(y) + "][" + std::to_string(x) + "]) + bias_val;\n";
        c += "    args.dst_tensor.Write(result, " + x_coord + ", " + y_coord +
             ", " + layer + ");\n";
        c += "  }\n";
      }
    }
    c += "  }\n";
  }
  c += "}\n";
  const DataType acc_type = precision == CalculationsPrecision::F16
                                ? DataType::FLOAT16
                                : DataType::FLOAT32;
  const DataType type = op_def.src_tensors[0].GetDataType();
  absl::StrReplaceAll({{"AccType", ToUclDataType(acc_type, 4)},
                       {"Type", ToUclDataType(type, 4)}},
                      &c);

  return c;
}

int3 ConvolutionTransposed3x3Thin::GetGridSize() const {
  const int grid_x = src_[0]->Width() * dst_[0]->Batch();
  const int grid_y = src_[0]->Height();
  const int grid_z = 1;
  return int3(grid_x, grid_y, grid_z);
}

std::vector<int> ConvolutionTransposed3x3Thin::GetSpatialWeightsRemap() const {
  return std::vector<int>{4, 5, 3, 7, 1, 8, 6, 2, 0};
}

void ConvolutionTransposed3x3Thin::UploadWeights(
    const GpuInfo& gpu_info, const Tensor<OHWI, DataType::FLOAT32>& weights) {
  const auto weights_desc = GetWeightsDescription();
  const int flt_count =
      GetTotalElementsCountForLayout(weights_desc, weights.shape);

  BufferDescriptor desc;
  desc.element_type = weights_desc.type;
  desc.element_size = 4;
  desc.memory_type = GetWeightsMemoryType(gpu_info);
  desc.size = flt_count * SizeOf(desc.element_type);
  desc.data.resize(desc.size);

  RearrangeWeights(weights, weights_desc, absl::MakeSpan(desc.data));

  args_.AddObject("weights",
                  std::make_unique<BufferDescriptor>(std::move(desc)));
}

bool IsConvolutionTransposed3x3ThinSupported(
    const ConvolutionTransposedAttributes& attr) {
  return attr.weights.shape.o <= 8 && attr.weights.shape.w == 3 &&
         attr.weights.shape.h == 3 && attr.stride.w == 2 &&
         attr.stride.h == 2 && attr.padding.prepended.w == 1 &&
         attr.padding.prepended.h == 1 && attr.padding.appended.w == 1 &&
         attr.padding.appended.h == 1;
}

ConvolutionTransposed3x3Thin CreateConvolutionTransposed3x3Thin(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const ConvolutionTransposedAttributes& attr) {
  ConvolutionTransposed3x3Thin result(gpu_info, definition, precision, attr);
  result.UploadWeights(gpu_info, attr.weights);

  if (!attr.bias.data.empty()) {
    TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
        gpu_info, definition.src_tensors[0].GetDataType(), attr.bias);
    result.args_.AddObject("biases", std::make_unique<TensorDescriptor>(
                                         std::move(bias_tensor_desc)));
  }
  return result;
}

ConvolutionTransposed3x3Thin CreateConvolutionTransposed3x3ThinDynamicWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const ConvolutionTransposedAttributes& attr) {
  OperationDef new_def = definition;
  new_def.src_tensors = {
      definition.src_tensors[0]};  // leaving only src_tensor def, weights defs
                                   // will be added later
  const DataType weights_type = definition.src_tensors[0].GetDataType();
  // add 1 src_tensor(buffer) for weights
  new_def.src_tensors.push_back(
      {weights_type, TensorStorageType::BUFFER, Layout::HWC});
  ConvolutionTransposed3x3Thin result(gpu_info, new_def, precision, attr);

  if (!attr.bias.data.empty()) {
    TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
        gpu_info, definition.src_tensors[0].GetDataType(), attr.bias);
    result.args_.AddObject("biases", std::make_unique<TensorDescriptor>(
                                         std::move(bias_tensor_desc)));
  }
  return result;
}

}  // namespace ml_drift
