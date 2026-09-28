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

#include "ml_drift/common/kernels/accumulate_input_channels.h"

#include <string>

#include "absl/log/absl_log.h"
#include "absl/strings/substitute.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"

namespace ml_drift {

namespace {

std::string GetSrcReader(const DataType src_data_type,
                         const TensorStorageType storage_type,
                         std::string linear_index) {
  if (src_data_type == DataType::kInt4) {
    return absl::Substitute(R"(
  int i32_value = ucl::Init<int>(0);
  args.src_tensor.ReadPerChannel(i32_value, ($0));
  src_value += ucl::I16ToVec4I4<int>(((i32_value << 16u) >> 16u));
  src_value += ucl::I16ToVec4I4<int>((i32_value >> 16u));
)",
                            linear_index);
  } else if (src_data_type == DataType::kInt2) {
    return absl::Substitute(R"(
  char i8_value = ucl::Init<char>(0);
  args.src_tensor.ReadPerChannel(i8_value, ($0));
  src_value = ucl::I8ToVec4I2<int>(i8_value);
)",
                            linear_index);
  }
  return absl::Substitute(R"(
  src_value = ucl::Convert<int4>(args.src_tensor.Read($0));
)",
                          linear_index);
}

std::string GetAccumulateInputChannelsCode(
    const DataType src_data_type, const TensorStorageType storage_type) {
  // TODO: b/430312327 - Calculate single element per thread with work group
  // size as WGX_x_4.
  std::string c = absl::Substitute(
      R"(
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int DST_S = ucl::GetGlobalId<2>();
  if (X > 0 || Y > 0 || DST_S >= args.dst_tensor_slices) return;
  int4 result = ucl::Init<int4>(0);
  int4 reducer_o0 = ucl::Init<int4>(0);
  for (int SRC_S = 0; SRC_S < args.input_channel_slices; SRC_S += 1) {
    int4 src_value = ucl::Init<int4>(0);
    $0
    reducer_o0 += src_value;
  }
  result.x = reducer_o0.x + reducer_o0.y + reducer_o0.z + reducer_o0.w;
  int4 reducer_o1 = ucl::Init<int4>(0);
  for (int SRC_S = 0; SRC_S < args.input_channel_slices; SRC_S += 1) {
    int4 src_value = ucl::Init<int4>(0);
    $1
    reducer_o1 += src_value;
  }
  result.y = reducer_o1.x + reducer_o1.y + reducer_o1.z + reducer_o1.w;
  int4 reducer_o2 = ucl::Init<int4>(0);
  for (int SRC_S = 0; SRC_S < args.input_channel_slices; SRC_S += 1) {
    int4 src_value = ucl::Init<int4>(0);
    $2
    reducer_o2 += src_value;
  }
  result.z = reducer_o2.x + reducer_o2.y + reducer_o2.z + reducer_o2.w;
  int4 reducer_o3 = ucl::Init<int4>(0);
  for (int SRC_S = 0; SRC_S < args.input_channel_slices; SRC_S += 1) {
    int4 src_value = ucl::Init<int4>(0);
    $3
    reducer_o3 += src_value;
  }
  result.w = reducer_o3.x + reducer_o3.y + reducer_o3.z + reducer_o3.w;

  result.x = DST_S * 4 + 0 < args.output_channels ? result.x : ucl::Init<int>(0);
  result.y = DST_S * 4 + 1 < args.output_channels ? result.y : ucl::Init<int>(0);
  result.z = DST_S * 4 + 2 < args.output_channels ? result.z : ucl::Init<int>(0);
  result.w = DST_S * 4 + 3 < args.output_channels ? result.w : ucl::Init<int>(0);
  args.dst_tensor.WriteLinear(result, DST_S);
}
)",
      GetSrcReader(src_data_type, storage_type,
                   "(DST_S * 4) * args.input_channel_slices + SRC_S"),
      GetSrcReader(src_data_type, storage_type,
                   "(DST_S * 4 + 1) * args.input_channel_slices + SRC_S"),
      GetSrcReader(src_data_type, storage_type,
                   "(DST_S * 4 + 2) * args.input_channel_slices + SRC_S"),
      GetSrcReader(src_data_type, storage_type,
                   "(DST_S * 4 + 3) * args.input_channel_slices + SRC_S"));
  return "MAIN_FUNCTION($0) {\n" + c;
}

}  // namespace

GPUOperation CreateAccumulateInputChannels(const OperationDef& definition,
                                           const OHWI& input_shape,
                                           const DataType& input_data_type) {
  const TensorDescriptor& src_desc = definition.src_tensors[0];
  const TensorDescriptor& dst_desc = definition.dst_tensors[0];

  if (src_desc.GetLayout() != Layout::kLinear ||
      dst_desc.GetLayout() != Layout::kLinear) {
    ABSL_LOG(FATAL) << "Only linear layout is supported for accumulate input "
                       "channels operation.";
  }
  if (input_data_type != DataType::kInt8 &&
      input_data_type != DataType::kInt4 &&
      input_data_type != DataType::kInt2) {
    ABSL_LOG(FATAL) << "Only int8, int4 and int2 input data types are "
                       "supported for accumulate input channels operation.";
  }
  if (dst_desc.GetDataType() != DataType::kInt32) {
    ABSL_LOG(FATAL) << "Only int32 output data type is supported for accumulate"
                       "input channels operation.";
  }

  GPUOperation op;
  int input_channel_slices;
  if (input_data_type == DataType::kInt4) {
    if (input_shape.i % 8 != 0) {
      ABSL_LOG(FATAL)
          << "Input channels must be a multiple of 8 for int4 data type.";
    }
    input_channel_slices = input_shape.i / 8;
  } else if (input_data_type == DataType::kInt2) {
    if (input_shape.i % 4 != 0) {
      ABSL_LOG(FATAL)
          << "Input channels must be a multiple of 16 for int2 data type.";
    }
    input_channel_slices = input_shape.i / 4;
  } else {
    if (input_shape.i % 4 != 0) {
      ABSL_LOG(FATAL) << "Input channels must be a multiple of 4.";
    }
    input_channel_slices = input_shape.i / 4;
  }
  op.AddSrcTensor("src_tensor", src_desc);
  op.AddDstTensor("dst_tensor", dst_desc);
  op.args_.AddInt("input_channels", input_shape.i);
  op.args_.AddInt("output_channels", input_shape.o);
  op.args_.AddInt("input_channel_slices", input_channel_slices);
  op.code_ = GetAccumulateInputChannelsCode(input_data_type,
                                            src_desc.GetStorageType());
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace ml_drift
