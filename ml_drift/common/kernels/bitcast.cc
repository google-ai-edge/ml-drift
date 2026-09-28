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

#include "ml_drift/common/kernels/bitcast.h"

#include <string>
#include <vector>

#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {

namespace {
std::string GetBitcastCode(const OperationDef& op_def,
                           const GpuInfo& gpu_info) {
  const DataType src_type = op_def.src_tensors[0].GetDataType();
  const DataType dst_type = op_def.dst_tensors[0].GetDataType();

  // boilerplate
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::kBatch)) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = ucl::GetGlobalId<0>();\n";
  }
  c += "  int Y = ucl::GetGlobalId<1>();\n";
  c += "  int S = ucl::GetGlobalId<2>();\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";

  // Convert from bool special bc stored as char
  if (src_type == DataType::kBool) {
    switch (SizeOf(dst_type)) {
      case 1: {
        c += "  uchar accum = 0u;\n";
        c += "  uchar bits = 0u;\n";
        break;
      }
      case 2: {
        c += "  ushort accum = 0u;\n";
        c += "  ushort bits = 0u;\n";
        break;
      }
      default: {  // 4
        c += "  uint accum = 0u;\n";
        c += "  uint bits = 0u;\n";
        break;
      }
    }
    c += "  for (int slice = 0; slice < args.src_tensor.Slices(); ++slice) {\n";
    if (gpu_info.IsApiWebGpu() ||
        gpu_info.IsGlsl()) {  // doesn't support bool shifts
      c += "    args.src_tensor::type read_bool = args.src_tensor.Read(X, Y, "
           "slice);\n";
      c += "    uint4 val = ucl::Convert<uint4>(read_bool);\n";
    } else {
      c += "    args.src_tensor::type val = args.src_tensor.Read(X, Y, "
           "slice);\n";
    }
    c += "    bits = val.x + (val.y << 1) + (val.z << 2) + (val.w << 3);\n";
    c += "    uint to_shift = ucl::Convert<uint>(slice * 4);\n";
    c += "    accum += bits << to_shift;\n";
    c += "  }\n";
    c += "  args.dst_tensor::type to_write;\n";
    if (dst_type == DataType::kUint8 || dst_type == DataType::kUint16 ||
        dst_type == DataType::kUint32) {
      c += "  to_write.x = accum;\n";
    } else {
      c += "  to_write.x = ucl::Reinterpret<uint, dst_type>(accum);\n";
    }
    c += "  args.dst_tensor.Write(to_write, X, Y, 0);\n";
    c += "}\n";
    c = absl::StrReplaceAll(c, {{"dst_type", ToUclDataType(dst_type, 1)}});
    return c;
  }

  // Convert to bool special bc stored as char
  if (dst_type == DataType::kBool) {
    c += "  args.src_tensor::scalar_type read = args.src_tensor.Read(X, Y, "
         "0).x;\n";
    switch (SizeOf(src_type)) {
      case 1: {
        if (src_type == DataType::kUint8) {
          c += "  uchar expanded = read;\n";
        } else {
          c += "  uchar expanded = ucl::Reinterpret<" +
               ToUclDataType(src_type) + ", uchar>(read);\n";
        }
        c += "  uchar4 to_write;\n";
        break;
      }
      case 2: {
        if (src_type == DataType::kUint16) {
          c += "  ushort expanded = read;\n";
        } else {
          c += "  ushort expanded = ucl::Reinterpret<" +
               ToUclDataType(src_type) + ", ushort>(read);\n";
        }
        c += "  ushort4 to_write;\n";
        break;
      }
      default: {  // 4
        if (src_type == DataType::kUint32) {
          c += "  uint expanded = read;\n";
        } else {
          c += "  uint expanded = ucl::Reinterpret<" + ToUclDataType(src_type) +
               ", uint>(read);\n";
        }
        c += "  uint4 to_write;\n";
      }
    }
    c += "  uint slices_to_shift = ucl::Convert<uint>(4 * S);\n";
    c += "  to_write.x = (expanded >> (slices_to_shift + 0u)) & 1u;\n";
    c += "  to_write.y = (expanded >> (slices_to_shift + 1u)) & 1u;\n";
    c += "  to_write.z = (expanded >> (slices_to_shift + 2u)) & 1u;\n";
    c += "  to_write.w = (expanded >> (slices_to_shift + 3u)) & 1u;\n";
    c += "  args.dst_tensor.Write(to_write, X, Y, S);\n";
    c += "}\n";
    return c;
  }

  // Normal bitcast
  std::string src_type_name;
  std::string dst_type_name;
  if (SizeOf(src_type) < SizeOf(dst_type)) {
    c += "  args.src_tensor::type src_read = args.src_tensor.Read(X, Y, S);\n";
    c += "  args.dst_tensor::type res;\n";
    if (SizeOf(dst_type) == SizeOf(src_type) * 2) {
      c += "  res.x = ucl::Reinterpret<src_type, dst_type>(src_read.xy);\n";
    } else {
      c += "  res.x = ucl::Reinterpret<src_type, dst_type>(src_read.xyzw);\n";
    }
    c += "  args.dst_tensor.Write(res, X, Y, S);\n";
    src_type_name =
        ToUclDataType(src_type, SizeOf(dst_type) / SizeOf(src_type));
    dst_type_name = ToUclDataType(dst_type, 1);
  } else if (SizeOf(src_type) > SizeOf(dst_type)) {
    c += "  args.src_tensor::scalar_type scalar_res = args.src_tensor.Read(X, "
         "Y, 0).x;\n";
    if (SizeOf(src_type) == SizeOf(dst_type) * 2) {
      c += "  args.dst_tensor::type res;\n";
      c += "  res.wz = ucl::Reinterpret<src_type, "
           "dst_type>(scalar_res);\n";
    } else {
      c += "  args.dst_tensor::type res = ucl::Reinterpret<src_type, "
           "dst_type>(scalar_res);\n";
    }
    c += "  args.dst_tensor.Write(res, X, Y, S);\n";
    src_type_name = ToUclDataType(src_type, 1);
    dst_type_name =
        ToUclDataType(dst_type, SizeOf(src_type) / SizeOf(dst_type));
  } else {
    c += "  args.src_tensor::type res = args.src_tensor.Read(X, Y, S);\n";
    c += "  args.dst_tensor.Write(ucl::Reinterpret<src_type, dst_type>(res), "
         "X, Y, S);\n";
    src_type_name = ToUclDataType(src_type, 4);
    dst_type_name = ToUclDataType(dst_type, 4);
  }
  c += "}\n";
  c = absl::StrReplaceAll(
      c, {{"src_type", src_type_name}, {"dst_type", dst_type_name}});
  return c;
}
}  // namespace

GPUOperation CreateBitcast(const OperationDef& op_def,
                           const GpuInfo& gpu_info) {
  GPUOperation op;
  op.AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  op.AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  op.code_ = GetBitcastCode(op_def, gpu_info);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace ml_drift
