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

#include "ml_drift/common/task/gpu_operation.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "ml_drift/common/access_type.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/util.h"
#include "ml_drift/common/task/wave_matrix_util.h"
#include "ml_drift/common/task/wave_memory_util.h"
#include "ml_drift/common/task/work_group_picking.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace {
int3 GetWorkGroupsCountInternal(int grid_dimension, const int3& grid_size,
                                const int3& work_group_size,
                                const int3& work_group_launch_order) {
  int3 work_groups_count;
  if (grid_dimension == 1) {
    work_groups_count.x = DivideRoundUp(grid_size.x, work_group_size.x);
    work_groups_count.y = 1;
    work_groups_count.z = 1;
  } else if (grid_dimension == 2) {
    int3 wgs;
    wgs.x = DivideRoundUp(grid_size.x, work_group_size.x);
    wgs.y = DivideRoundUp(grid_size.y, work_group_size.y);
    work_groups_count.x = wgs[work_group_launch_order[0]];
    work_groups_count.y = wgs[work_group_launch_order[1]];
    work_groups_count.z = 1;
  } else {  // grid_dimension == 3
    int3 wgs;
    wgs.x = DivideRoundUp(grid_size.x, work_group_size.x);
    wgs.y = DivideRoundUp(grid_size.y, work_group_size.y);
    wgs.z = DivideRoundUp(grid_size.z, work_group_size.z);
    work_groups_count.x = wgs[work_group_launch_order[0]];
    work_groups_count.y = wgs[work_group_launch_order[1]];
    work_groups_count.z = wgs[work_group_launch_order[2]];
  }
  return work_groups_count;
}

std::string GetElementWiseCode(const TensorDescriptor& dst_desc) {
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (dst_desc.HasAxis(Axis::BATCH)) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = ucl::GetGlobalId<0>();\n";
  }

  std::string coords = "X, Y";
  if (dst_desc.HasAxis(Axis::DEPTH)) {
    c += "  int linear_y = ucl::GetGlobalId<1>();\n";
    c += "  int Y = linear_y / args.dst_tensor.Depth();\n";
    c += "  int Z = linear_y % args.dst_tensor.Depth();\n";
    coords += ", Z";
  } else {
    c += "  int Y = ucl::GetGlobalId<1>();\n";
  }
  c += "  int S = ucl::GetGlobalId<2>();\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  coords += ", S";
  c += "  args.src_tensor::type src = args.src_tensor.Read(" + coords + ");\n";
  c += "  args.dst_tensor.Write(src, " + coords + ");\n";
  c += "} \n";
  return c;
}

bool NeedsBroadcast(const TensorDescriptor& src_desc, const BHWDC& src_shape,
                    const BHWDC& dst_shape) {
  bool needs_broadcast = src_shape.w < dst_shape.w ||
                         src_shape.h < dst_shape.h ||
                         src_shape.c < dst_shape.c || src_shape.d < dst_shape.d;
  if (src_desc.HasAxis(Axis::BATCH)) {
    needs_broadcast = needs_broadcast || src_shape.b < dst_shape.b;
  }
  return needs_broadcast;
}

std::string GetStringWithoutComments(absl::string_view code) {
  size_t code_size = code.size();
  std::string result;
  result.reserve(code_size);
  int last_index = 0;
  size_t i = 0;
  for (; i < code_size; ++i) {
    // skip long comment /*...*/
    if (code[i] == '/' && i + 1 < code.size() && code[i + 1] == '*') {
      result += code.substr(last_index, i - last_index);
      i = i + 3;
      for (; i < code_size && (code[i - 1] != '*' && code[i] != '/'); ++i) {
      }
      last_index = i + 1;
      continue;
    }
    // skip short comment //...
    if (code[i] == '/' && i + 1 < code.size() && code[i + 1] == '/') {
      result += code.substr(last_index, i - last_index);
      i = i + 2;
      for (; i < code_size && code[i] != '\n'; ++i) {
      }
      last_index = i;
      continue;
    }
  }
  result += code.substr(last_index, i - last_index);
  return result;
}

absl::Status AddConvertFromBfloat(const GpuInfo& gpu_info,
                                  const std::vector<std::string>& template_args,
                                  std::string* code) {
  if (template_args.size() != 1) {
    return absl::InvalidArgumentError(
        "ConvertFromBfloat requires exactly 1 template argument\n");
  }
  int type_size = 0;
  DataType dst_data_type;
  ABSL_RETURN_IF_ERROR(
      DataTypeFromTemplateArg(template_args[0], &dst_data_type, &type_size));
  const std::string dst_ucl = ToUclDataType(dst_data_type, type_size);
  if (gpu_info.IsApiOpenCl()) {
    const std::string dst_cl = ToCLDataType(dst_data_type, type_size);
    if (absl::StrContains(*code, dst_cl + " ConvertFromBfloatTo" + dst_ucl)) {
      return absl::OkStatus();
    }
    std::string read = dst_cl + " ConvertFromBfloatTo$0(" +
                       ToUclDataType(DataType::UINT16, type_size) + " src) {\n";
    read += "  " + ToCLDataType(DataType::UINT32, type_size) +
            " src_ui32 = ucl::Convert<$1>(src);\n";
    read += "  src_ui32 <<= 16;\n";
    if (dst_data_type == DataType::FLOAT32) {
      read += "  return ucl::Reinterpret<$1, $2>(src_ui32);\n";
    } else {
      read += "  " + ToCLDataType(DataType::FLOAT32, type_size) +
              " dst_float ucl::Reinterpret<$1, $2>(src_ui32);\n";
      read += "  return ucl::Convert<$0>(dst_float);\n";
    }
    read += "}\n\n";
    read = absl::Substitute(read, dst_ucl,
                            ToUclDataType(DataType::UINT32, type_size),
                            ToUclDataType(DataType::FLOAT32, type_size));
    *code = read + *code;
  } else if (gpu_info.IsApiWebGpu()) {
    if (absl::StrContains(*code, "fn ConvertFromBfloatTo" + dst_ucl)) {
      return absl::OkStatus();
    }
    std::string read = "fn ConvertFromBfloatTo$0(src : " +
                       ToWebGpuDataType(DataType::UINT32, type_size) + ") -> " +
                       ToWebGpuDataType(dst_data_type, type_size) + " {\n";
    read += "  var src_ui32 : $3 = src;\n";
    read += "  const shift_vec : $3 = ucl::Init<$1>(16);\n";
    read += "  src_ui32 <<= shift_vec;\n";
    if (dst_data_type == DataType::FLOAT32) {
      read += "  return ucl::Reinterpret<$1, $2>(src_ui32);\n";
    } else {
      read += "  var dst_float : " +
              ToWebGpuDataType(DataType::FLOAT32, type_size) +
              " = ucl::Reinterpret<$1, $2>(src_ui32);\n";
      read += "  return ucl::Convert<$0>(dst_float);\n";
    }
    read += "}\n\n";
    read = absl::Substitute(read, dst_ucl,
                            ToUclDataType(DataType::UINT32, type_size),
                            ToUclDataType(DataType::FLOAT32, type_size),
                            ToWebGpuDataType(DataType::UINT32, type_size));
    *code = read + *code;
  } else if (gpu_info.IsGlsl()) {
    const std::string dst_gl = ToGlslShaderDataType(dst_data_type, type_size);
    if (absl::StrContains(*code, dst_gl + " ConvertFromBfloatTo" + dst_ucl)) {
      return absl::OkStatus();
    }
    std::string read = dst_gl + " ConvertFromBfloatTo$0(" +
                       ToGlslShaderDataType(DataType::UINT16, type_size) +
                       " src) {\n";
    read += "  " + ToGlslShaderDataType(DataType::UINT32, type_size) +
            " src_ui32 = ucl::Convert<$1>(src);\n";
    read += "  src_ui32 <<= 16;\n";
    read += "  " + ToGlslShaderDataType(DataType::FLOAT32, type_size) +
            " dst_float;\n";
    for (int i = 0; i < type_size; ++i) {
      read += "  dst_float[" + std::to_string(i) + "] = ucl::Reinterpret<" +
              ToUclDataType(DataType::UINT32, 1) + ", " +
              ToUclDataType(DataType::FLOAT32, 1) + ">(src_ui32[" +
              std::to_string(i) + "]);\n";
    }
    if (dst_data_type == DataType::FLOAT32) {
      read += "  return dst_float;\n";
    } else {
      read += "  return ucl::Convert<$0>(dst_float);\n";
    }
    read += "}\n\n";
    read = absl::Substitute(read, dst_ucl,
                            ToUclDataType(DataType::UINT32, type_size),
                            ToUclDataType(DataType::FLOAT32, type_size));
    *code = read + *code;
  } else if (gpu_info.IsApiMetal()) {
    if (gpu_info.metal_info.IsNativeBfloatSupported()) {
      return absl::OkStatus();
    }
    const std::string dst_metal = ToMetalDataType(dst_data_type, type_size);
    if (absl::StrContains(*code,
                          dst_metal + " ConvertFromBfloatTo" + dst_ucl)) {
      return absl::OkStatus();
    }
    std::string read = dst_metal + " ConvertFromBfloatTo" + dst_ucl + "(" +
                       ToMetalDataType(DataType::UINT16, type_size) +
                       " src) {\n";
    read += "  $0 src_ui32 = ucl::Convert<$0>(src);\n";
    read += "  src_ui32 <<= 16;\n";
    if (dst_data_type == DataType::FLOAT32) {
      read += "  return ucl::Reinterpret<$0, $1>(src_ui32);\n";
    } else {
      read += "  $1 dst_float = ucl::Convert<$1>(src_ui32);\n";
      read += "  return ucl::Reinterpret<$1, " + dst_metal + ">(dst_float);\n";
    }
    read += "}\n\n";
    read = absl::Substitute(read, ToMetalDataType(DataType::UINT32, type_size),
                            ToMetalDataType(DataType::FLOAT32, type_size));
    *code = read + *code;
  } else {
    return absl::UnimplementedError(
        "ConvertFromBfloat is not implemented for this backend");
  }
  return absl::OkStatus();
}

std::string GetU8ToVec4I2Function(const GpuInfo& gpu_info) {
  std::string c;
  if (gpu_info.IsApiWebGpu()) {
    c += "fn UclU8ToVec4I2(src: u32) -> vec4<i32> {\n";
  } else {
    c += "int4 UclU8ToVec4I2(uint src) {\n";
  }
  c += R"(  int s = ucl::Convert<int>(src);
  return ucl::Init<int4>((s << 30u) >> 30u, (s << 28u) >> 30u, (s << 26u) >> 30u, (s << 24u) >> 30u);
}
)";
  return c;
}

absl::Status AddConvertToBfloat(const GpuInfo& gpu_info,
                                const std::vector<std::string>& template_args,
                                std::string* code) {
  if (template_args.size() != 1) {
    return absl::InvalidArgumentError(
        "ConvertToBfloat requires exactly 1 template argument\n");
  }
  int type_size = 0;
  DataType src_data_type;
  ABSL_RETURN_IF_ERROR(
      DataTypeFromTemplateArg(template_args[0], &src_data_type, &type_size));
  const std::string src_ucl = ToUclDataType(src_data_type, type_size);
  if (gpu_info.IsApiOpenCl()) {
    const std::string src_cl = ToCLDataType(src_data_type, type_size);
    if (absl::StrContains(*code,
                          "ConvertToBfloatFrom" + src_ucl + "(" + src_cl)) {
      return absl::OkStatus();
    }
    const std::string ui32_cl = ToCLDataType(DataType::UINT32, type_size);
    const std::string float_vec = ToUclDataType(DataType::FLOAT32, type_size);
    std::string write = ToCLDataType(DataType::UINT16, type_size) +
                        " ConvertToBfloatFrom" + src_ucl + "(" + src_cl +
                        " src) {\n";
    if (src_data_type != DataType::FLOAT32) {
      write += "  " + ui32_cl +
               " src_ui32 = ucl::Reinterpret<$1, "
               "$0>(ucl::Convert<$1>(src));\n";
    } else {
      write += "  " + ui32_cl + " src_ui32 = ucl::Reinterpret<$1, $0>(src);\n";
    }
    write += "  " + ToCLDataType(DataType::INT32, type_size) +
             " to_round = (src_ui32 & 0x00018000) == 0x00018000;\n";
    write += "  src_ui32 = to_round ? src_ui32 + 0x00008000 : src_ui32;\n";
    write += "  src_ui32 >>= 16;\n";
    write += "  return ucl::Convert<" +
             ToUclDataType(DataType::UINT16, type_size) + ">(src_ui32);\n";
    write += "}\n\n";
    write = absl::Substitute(write, ToUclDataType(DataType::UINT32, type_size),
                             float_vec);
    *code = write + *code;
  } else if (gpu_info.IsApiWebGpu()) {
    if (absl::StrContains(*code, "fn ConvertToBfloatFrom" + src_ucl)) {
      return absl::OkStatus();
    }
    const std::string float_vec = ToUclDataType(DataType::FLOAT32, type_size);
    std::string write = "fn ConvertToBfloatFrom" + src_ucl +
                        "(src : " + ToWebGpuDataType(src_data_type, type_size) +
                        ") -> $2 {\n";
    if (src_data_type != DataType::FLOAT32) {
      write +=
          "  var src_ui32 : $2 = ucl::Reinterpret<$1, "
          "$0>(ucl::Convert<$1>(src));\n";
    } else {
      write += "  var src_ui32 : $2 = ucl::Reinterpret<$1, $0>(src);\n";
    }
    write += "  const magic_vec : $2 = ucl::Init<$0>(0x00018000);\n";
    write += "  let to_round : " + ToWebGpuDataType(DataType::BOOL, type_size) +
             " = (src_ui32 & magic_vec) == magic_vec;\n";
    write += "  src_ui32 = to_round ? src_ui32 + 0x00008000 : src_ui32;\n";
    write += "  const shift_vec : $2 = ucl::Init<$0>(16);\n";
    write += "  return src_ui32 >> shift_vec;\n";
    write += "}\n\n";
    write = absl::Substitute(write, ToUclDataType(DataType::UINT32, type_size),
                             float_vec,
                             ToWebGpuDataType(DataType::UINT32, type_size));
    *code = write + *code;
  } else if (gpu_info.IsGlsl()) {
    const std::string src_gl = ToGlslShaderDataType(src_data_type, type_size);
    if (absl::StrContains(*code,
                          "ConvertToBfloatFrom" + src_ucl + "(" + src_gl)) {
      return absl::OkStatus();
    }
    const std::string ushort_gl =
        ToGlslShaderDataType(DataType::UINT16, type_size);
    const std::string float_gl =
        ToGlslShaderDataType(DataType::FLOAT32, type_size);
    std::string write = ushort_gl + " ConvertToBfloatFrom" + src_ucl + "(" +
                        src_gl + " src) {\n";
    if (src_data_type != DataType::FLOAT32) {
      write += "  $0 src_float = ucl::Convert<" +
               ToUclDataType(DataType::FLOAT32, type_size) + ">(src);\n";
    } else {
      write += "  $0 src_float = src;\n";
    }
    write += "  " + ToGlslShaderDataType(DataType::UINT32, type_size) +
             " src_ui32;\n";
    write +=
        "  " + ToGlslShaderDataType(DataType::BOOL, type_size) + " to_round;\n";
    write +=
        "  for (int i = 0; i < " + std::to_string(type_size) + "; ++i) {\n";
    write += R"(
        src_ui32[i] = ucl::Reinterpret<float, uint>(src_float[i]);
        to_round[i] = (src_ui32[i] & 98304u) == 98304u;
        src_ui32[i] = to_round[i] ? src_ui32[i] + 32768u : src_ui32[i];
      }
    )";
    write += "  src_ui32 >>= 16;\n";
    write += "  return ucl::Convert<" +
             ToUclDataType(DataType::UINT16, type_size) + ">(src_ui32);\n";
    write += "}\n\n";
    write = absl::Substitute(write, float_gl);
    *code = write + *code;
  } else if (gpu_info.IsApiMetal()) {
    if (gpu_info.metal_info.IsNativeBfloatSupported()) {
      return absl::OkStatus();
    }
    const std::string src_metal = ToMetalDataType(src_data_type, type_size);
    if (absl::StrContains(*code,
                          "ConvertToBfloatFrom" + src_ucl + "(" + src_metal)) {
      return absl::OkStatus();
    }
    const std::string ushort_metal =
        ToMetalDataType(DataType::UINT16, type_size);
    const std::string float_metal =
        ToMetalDataType(DataType::FLOAT32, type_size);
    const std::string uint_metal = ToMetalDataType(DataType::UINT32, type_size);
    std::string write =
        "$0 ConvertToBfloatFrom" + src_ucl + "(" + src_metal + " src) {\n";
    if (src_data_type != DataType::FLOAT32) {
      write += "  " + float_metal + " src_float = ucl::Convert<" +
               ToUclDataType(DataType::FLOAT32, type_size) + src_ucl + ">(" +
               src_metal + ")(src);\n";
    } else {
      write += "  " + float_metal + " src_float = src;\n";
    }
    write += "  $1 src_ui32 = ucl::Reinterpret<" + float_metal + ", " +
             uint_metal + ">(src_float);\n";
    write += "  " + ToUclDataType(DataType::BOOL, type_size) + " to_round = " +
             "(src_ui32 & ($1)(0x00018000)) == ($1)(0x00018000);\n";
    write +=
        "  for (int i = 0; i < " + std::to_string(type_size) + "; ++i) {\n";
    write += R"(
      src_ui32[i] = to_round[i] ? src_ui32[i] + 0x0000800 : src_ui32[i];
    }
  )";
    write += "  src_ui32 >>=16;\n";
    write += "  return ucl::Convert<$0>(src_ui32);\n}\n\n";
    write = absl::Substitute(write, ushort_metal, uint_metal);
    *code = write + *code;
  } else {
    return absl::UnimplementedError(
        "ConvertToBfloat is not implemented for this backend");
  }
  return absl::OkStatus();
}

absl::Status AddQuantizedBufferWrite(const GpuInfo& gpu_info, DataType type,
                                     std::string* code) {
  std::string fcn;
  if (gpu_info.IsGlsl()) {
    if (type == DataType::UINT16 || type == DataType::INT16 ||
        type == DataType::BFLOAT16) {
      if (type == DataType::UINT16 || type == DataType::BFLOAT16) {
        fcn += "uvec2 QuantizedBufferWrite(ushort4 src) {\n";
        if (absl::StrContains(*code, fcn)) {
          return absl::OkStatus();
        }
        fcn += "  uvec2 dst;\n";
      } else {
        fcn += "ivec2 QuantizedBufferWrite(short4 src) {\n";
        if (absl::StrContains(*code, fcn)) {
          return absl::OkStatus();
        }
        fcn += "  ivec2 dst;\n";
      }
      fcn += "  dst.x = bitfieldInsert(dst.x, src.x, 0, 16);\n";
      fcn += "  dst.x = bitfieldInsert(dst.x, src.y, 16, 16);\n";
      fcn += "  dst.y = bitfieldInsert(dst.y, src.z, 0, 16);\n";
      fcn += "  dst.y = bitfieldInsert(dst.y, src.w, 16, 16);\n";
      fcn += "  return dst;\n}\n";
      *code = fcn + *code;
      return absl::OkStatus();
    } else if (type == DataType::UINT8 || type == DataType::INT8 ||
               type == DataType::BOOL) {
      if (type == DataType::UINT8 || type == DataType::BOOL) {
        fcn += "uint QuantizedBufferWrite(uchar4 src) {\n";
        if (absl::StrContains(*code, fcn)) {
          return absl::OkStatus();
        }
        fcn += "  uint dst;\n";
      } else {
        fcn += "int QuantizedBufferWrite(char4 src) {\n";
        if (absl::StrContains(*code, fcn)) {
          return absl::OkStatus();
        }
        fcn += "  int dst;\n";
      }
      fcn += "  dst = bitfieldInsert(dst, src.x, 0, 8);\n";
      fcn += "  dst = bitfieldInsert(dst, src.y, 8, 8);\n";
      fcn += "  dst = bitfieldInsert(dst, src.z, 16, 8);\n";
      fcn += "  dst = bitfieldInsert(dst, src.w, 24, 8);\n";
      fcn += "  return dst;\n}\n";
      *code = fcn + *code;
      return absl::OkStatus();
    } else {
      return absl::UnimplementedError(
          "QuantizedBufferWrite is not used for GlSl for this type");
    }
  } else if (gpu_info.IsApiWebGpu()) {
    if (type == DataType::UINT16 || type == DataType::INT16 ||
        type == DataType::BFLOAT16) {
      if (type == DataType::UINT16 || type == DataType::BFLOAT16) {
        fcn += "fn QuantizedBufferWrite(src : vec4<u32>) -> vec2<u32> {\n";
        if (absl::StrContains(*code, fcn)) {
          return absl::OkStatus();
        }
        fcn += "  var dst: vec2<u32>;\n";
      } else {
        fcn += "fn QuantizedBufferWrite(src : vec4<i32>) -> vec2<i32> {\n";
        if (absl::StrContains(*code, fcn)) {
          return absl::OkStatus();
        }
        fcn += "  var dst: vec2<i32>;\n";
      }
      fcn += "  dst.x = insertBits(dst.x, src.x, 0, 16);\n";
      fcn += "  dst.x = insertBits(dst.x, src.y, 16, 16);\n";
      fcn += "  dst.y = insertBits(dst.y, src.z, 0, 16);\n";
      fcn += "  dst.y = insertBits(dst.y, src.w, 16, 16);\n";
      fcn += "  return dst;\n}\n";
      *code = fcn + *code;
      return absl::OkStatus();
    } else if (type == DataType::UINT8 || type == DataType::INT8 ||
               type == DataType::BOOL) {
      if (type == DataType::UINT8 || type == DataType::BOOL) {
        fcn += "fn QuantizedBufferWrite(src : vec4<u32>) -> u32 {\n";
        if (absl::StrContains(*code, fcn)) {
          return absl::OkStatus();
        }
        fcn += "  var dst: u32;\n";
      } else {
        fcn += "fn QuantizedBufferWrite(src : vec4<i32>) -> i32 {\n";
        if (absl::StrContains(*code, fcn)) {
          return absl::OkStatus();
        }
        fcn += "  var dst: i32;\n";
      }
      fcn += "  dst = insertBits(dst, src.x, 0, 8);\n";
      fcn += "  dst = insertBits(dst, src.y, 8, 8);\n";
      fcn += "  dst = insertBits(dst, src.z, 16, 8);\n";
      fcn += "  dst = insertBits(dst, src.w, 24, 8);\n";
      fcn += "  return dst;\n}\n";
      *code = fcn + *code;
      return absl::OkStatus();
    } else {
      return absl::UnimplementedError(
          "QuantizedBufferWrite is not used for WebGpu for this type");
    }
  }
  return absl::UnimplementedError(
      "QuantizedBufferWrite is not implemented for this backend");
}

absl::Status AddGlslReinterpretUvec2ToHalf4(const GpuInfo& gpu_info,
                                            std::string* code) {
  if (!gpu_info.IsGlsl()) {
    return absl::UnimplementedError(
        "GlslReinterpretUvec2ToHalf4 is not implemented for this backend");
  }

  if (absl::StrContains(*code, "half4 ReinterpretUvec2ToHalf4")) {
    return absl::OkStatus();
  }
  const std::string unpack_fnc = gpu_info.IsGlslSupportsExplicitFp16()
                                     ? "unpackFloat2x16"
                                     : "unpackHalf2x16";
  const std::string half_type =
      gpu_info.IsGlslSupportsExplicitFp16() ? "half4" : "vec4";
  const std::string fnc =
      absl::Substitute(R"(half4 ReinterpretUvec2ToHalf4(uvec2 weights) {
  half2 lo = $0(weights.x);
  half2 hi = $0(weights.y);
  return $1(lo.x, lo.y, hi.x, hi.y);
}
)",
                       unpack_fnc, half_type);
  *code = fnc + *code;
  return absl::OkStatus();
}

absl::Status AddGlslReinterpretHalf4ToUvec2(const GpuInfo& gpu_info,
                                            std::string* code) {
  if (!gpu_info.IsGlsl()) {
    return absl::UnimplementedError(
        "GlslReinterpretHalf4ToUvec2 is not implemented for this backend");
  }

  if (absl::StrContains(*code, "uvec2 ReinterpretHalf4ToUvec2")) {
    return absl::OkStatus();
  }
  const std::string unpack_fnc =
      gpu_info.IsGlslSupportsExplicitFp16() ? "packFloat2x16" : "packHalf2x16";
  const std::string fnc =
      absl::Substitute(R"(uvec2 ReinterpretHalf4ToUvec2(half4 value) {
  uint lo = $0(value.xy);
  uint hi = $0(value.zw);
  return uvec2(lo, hi);
}
)",
                       unpack_fnc);
  *code = fnc + *code;
  return absl::OkStatus();
}

absl::Status AddGlslBitsToVec(const GpuInfo& gpu_info,
                              absl::string_view fcn_name,
                              const std::vector<std::string>& template_args,
                              std::string* code) {
  // Check correct conditions
  if (!gpu_info.IsGlsl()) {
    return absl::UnimplementedError(
        "GlslBitsToVec is not implemented for this backend");
  }
  if (absl::StrContains(*code, absl::StrCat(template_args[1], " ", fcn_name))) {
    return absl::OkStatus();
  }

  // Parse
  std::string fcn = absl::StrCat(template_args[1], " ", fcn_name, "(",
                                 template_args[0], " src) {\n");
  DataType src_data_type;
  int src_type_size;
  ABSL_RETURN_IF_ERROR(DataTypeFromTemplateArg(template_args[0], &src_data_type,
                                               &src_type_size));
  DataType dst_data_type;
  int dst_type_size;
  ABSL_RETURN_IF_ERROR(DataTypeFromTemplateArg(template_args[1], &dst_data_type,
                                               &dst_type_size));
  if (src_type_size != 1 && dst_type_size != 1 &&
      src_type_size != dst_type_size) {
    return absl::UnimplementedError(
        "GlslBitsToVec for vec <-> vec only supported for same size vectors");
  }

  // Base case for vectors of the same size
  if (src_type_size == dst_type_size) {
    fcn += "  " + template_args[1] + " dst;\n";
    for (int i = 0; i < src_type_size; ++i) {
      fcn += "  dst[" + std::to_string(i) + "] = ucl::Reinterpret<" +
             ToUclDataType(src_data_type) + ", " +
             ToUclDataType(dst_data_type) + ">(src[" + std::to_string(i) +
             "]);\n";
    }
    fcn += "  return dst;\n}\n";
    *code = fcn + *code;
    return absl::OkStatus();
  }

  // Reinterpret for vectors of different sizes
  // Convert from float to int if necessary (bitfield fcns only work with ints)
  if (src_data_type == DataType::FLOAT16 ||
      src_data_type == DataType::FLOAT32 ||
      src_data_type == DataType::FLOAT64) {  // float src type
    std::string src_int_type_str;
    if (dst_data_type == DataType::INT8 || dst_data_type == DataType::INT16 ||
        dst_data_type == DataType::INT32 || dst_data_type == DataType::INT64) {
      src_int_type_str = "int";
    } else {
      src_int_type_str = "uint";
    }
    if (src_type_size == 1) {
      fcn += "  " + src_int_type_str + " src_int = ucl::Reinterpret<float, " +
             src_int_type_str + ">(src);\n";
    } else {
      fcn += "  " + src_int_type_str + std::to_string(src_type_size) +
             " src_int;\n";
      for (int i = 0; i < src_type_size; ++i) {
        fcn += "  src_int[" + std::to_string(i) +
               "] = ucl::Reinterpret<float, " + src_int_type_str + ">(src[" +
               std::to_string(i) + "]);\n";
      }
    }
  } else {  // not a float src type
    fcn += "  " + template_args[0] + " src_int = src;\n";
  }

  // Determine type of int to use for bitfield operations
  bool signed_convert =
      src_data_type == DataType::INT8 || src_data_type == DataType::INT16 ||
      src_data_type == DataType::INT32 || src_data_type == DataType::INT64;
  const std::string dst_int_type_str =
      signed_convert ? ToGlslShaderDataType(DataType::INT32, dst_type_size)
                     : ToGlslShaderDataType(DataType::UINT32, dst_type_size);
  fcn += "  " + dst_int_type_str + " dst_int;\n";

  // bitfieldExtract
  if (src_type_size == 1) {
    const int bits = SizeOf(dst_data_type) * 8;
    int offset = 0;
    for (int i = 0; i < dst_type_size; ++i) {
      fcn += "  dst_int[" + std::to_string(i) +
             "] = bitfieldExtract(src_int, " + std::to_string(offset) + ", " +
             std::to_string(bits) + ");\n";
      offset += bits;
    }
  } else {  // bitfieldInsert
    const int bits = SizeOf(src_data_type) * 8;
    int offset = 0;
    for (int i = 0; i < src_type_size; ++i) {
      fcn += "  dst_int = bitfieldInsert(dst_int, src_int[" +
             std::to_string(i) + "], " + std::to_string(offset) + ", " +
             std::to_string(bits) + ");\n";
      offset += bits;
    }
  }

  // Convert from int to float if necessary (bitfield fcns only work with ints)
  if (dst_data_type == DataType::FLOAT16 ||
      dst_data_type == DataType::FLOAT32 ||
      dst_data_type == DataType::FLOAT64) {  // float dst type
    std::string dst_int_type_str;
    if (signed_convert) {
      dst_int_type_str = "int";
    } else {
      dst_int_type_str = "uint";
    }
    if (dst_type_size == 1) {
      fcn += "  float dst = ucl::Reinterpret<" + dst_int_type_str +
             ", float>(dst_int);\n";
    } else {
      fcn += "  " + ToGlslShaderDataType(DataType::FLOAT32, dst_type_size) +
             " dst;\n";
      for (int i = 0; i < src_type_size; ++i) {
        fcn += "  dst[" + std::to_string(i) + "] = ucl::Reinterpret<" +
               dst_int_type_str + ", float>(dst_int[" + std::to_string(i) +
               "]);\n";
      }
    }
  } else {  // not a float src type
    fcn += "  " + template_args[1] + " dst = dst_int;\n";
  }

  fcn += "  return dst;\n}\n";
  *code = fcn + *code;
  return absl::OkStatus();
}

bool IsWordSymbol(char symbol) {
  return absl::ascii_isalnum(symbol) || symbol == '_';
}

// Word is a token consisting of ascii_isalnum symbols or '_'
// (see above IsWordSymbol(char symbol))
// ReplaceAllWords replace all specified word-tokens(old_word) with new_word
void ReplaceAllWords(const std::string& old_word, const std::string& new_word,
                     std::string* str) {
  for (size_t position = str->find(old_word); position != std::string::npos;
       position = str->find(old_word, position)) {
    const char prev = position == 0 ? '.' : (*str)[position - 1];
    const char next = position + old_word.size() < str->size()
                          ? (*str)[position + old_word.size()]
                          : '.';
    if (IsWordSymbol(prev) || IsWordSymbol(next)) {
      position += 1;
      continue;
    }
    str->replace(position, old_word.size(), new_word);
    position += new_word.size();
  }
}

struct LinkableContext {
  std::string code;
  TensorDescriptor* tensor_desc;
};

absl::Status ResolveLinking(const GpuInfo& gpu_info,
                            const LinkableContext& linkable_context,
                            std::vector<std::string>* function_args,
                            std::string* result) {
  std::string value_name, x_coord, y_coord, z_coord, s_coord, b_coord;
  ABSL_RETURN_IF_ERROR(
      linkable_context.tensor_desc->GetLinkingContextFromWriteSelector(
          *function_args, &value_name, &x_coord, &y_coord, &z_coord, &s_coord,
          &b_coord));
  const std::string new_value_name = value_name + "_final";
  DataType type = linkable_context.tensor_desc->GetDataType();
  if (!(gpu_info.IsApiMetal() &&
        gpu_info.metal_info.IsNativeBfloatSupported())) {
    type = type == DataType::BFLOAT16 ? DataType::FLOAT32 : type;
  }
  const std::string type_decl = ToUclDataType(type, 4);
  const std::string out_var_declaration =
      "\n " + type_decl + " " + new_value_name + ";\n";
  *result = out_var_declaration +
            "{  // elementwise code with input:" + value_name +
            " output:" + new_value_name + "\n" +
            absl::Substitute(linkable_context.code, "") + "\n}\n";
  *result = absl::StrReplaceAll(*result, {{"\n", "\n  "},
                                          {"in_value", value_name},
                                          {"out_value", new_value_name},
                                          {"X_COORD", x_coord},
                                          {"Y_COORD", y_coord},
                                          {"Z_COORD", z_coord},
                                          {"S_COORD", s_coord},
                                          {"B_COORD", b_coord}});

  (*function_args)[0] = new_value_name;
  return absl::OkStatus();
}

// resolve constructions of type: args.object_name::const_expr_name
// Example: 'args.dst_tensor::type' can be replaced with 'float4'
absl::Status ResolveConstExprPass(const GpuInfo& gpu_info,
                                  const Arguments& args, std::string* code) {
  std::string result;
  size_t position = 0;
  constexpr char kArgsPrefix[] = "args.";
  size_t next_position = code->find(kArgsPrefix);
  while (next_position != std::string::npos) {
    size_t arg_pos = next_position;
    next_position += strlen(kArgsPrefix);
    absl::string_view object_name = GetNextWord(*code, next_position);
    if (next_position + object_name.size() > code->size() - 2) {
      next_position = code->find(kArgsPrefix, next_position);
      continue;
    }
    char next0 = (*code)[next_position + object_name.size()];
    char next1 = (*code)[next_position + object_name.size() + 1];
    if (next0 == ':' && next1 == ':') {
      next_position += object_name.size() + 2;
      absl::string_view const_expr_name = GetNextWord(*code, next_position);
      next_position += const_expr_name.size();
      std::string patch;
      GPUObjectDescriptor* desc_ptr;
      ABSL_RETURN_IF_ERROR(args.GetDescriptor(object_name, &desc_ptr));
      ABSL_RETURN_IF_ERROR(
          desc_ptr->PerformConstExpr(gpu_info, const_expr_name, &patch));
      code->replace(arg_pos, next_position - arg_pos, patch);
      position = arg_pos + patch.size();
    } else {
      position = arg_pos + strlen(kArgsPrefix);
    }
    next_position = code->find(kArgsPrefix, position);
  }
  return absl::OkStatus();
}

// resolve constructions of type: args.object_name.method_name(list of args)
// Example: 'args.bias.Read(S)' can be replaced with 'args.bias_buffer[S]'
absl::Status ResolveSelectorsPass(
    const GpuInfo& gpu_info,
    const std::map<std::string, LinkableContext, std::less<>>& linkables,
    const Arguments& args, std::string* code) {
  std::string result;
  size_t position = 0;
  constexpr char kArgsPrefix[] = "args.";
  size_t next_position = code->find(kArgsPrefix);
  std::string write_fcns = "";
  // Declare arg vectors here to avoid reallocating each loop.
  std::vector<std::string> template_args;
  std::vector<std::string> function_args;
  while (next_position != std::string::npos) {
    size_t arg_pos = next_position;
    next_position += strlen(kArgsPrefix);
    std::string object_name = GetNextWordStr(*code, next_position);
    char next = (*code)[next_position + object_name.size()];
    if (next == '.') {
      next_position += object_name.size() + 1;
      std::string selector_name = GetNextWordStr(*code, next_position);
      next_position += selector_name.size();
      next = (*code)[next_position];
      template_args.clear();
      if (next == '<') {
        size_t close_bracket_pos;
        ABSL_RETURN_IF_ERROR(ParseArguments(
            *code, next_position, &close_bracket_pos, &template_args));
        next_position = close_bracket_pos + 1;
        next = (*code)[next_position];
      }
      if (next != '(') {
        return absl::NotFoundError(absl::StrCat(
            "Expected ( after ", object_name, ".", selector_name, " call"));
      }
      function_args.clear();
      size_t close_bracket_pos;
      ABSL_RETURN_IF_ERROR(ParseArguments(*code, next_position,
                                          &close_bracket_pos, &function_args));
      for (auto& arg : function_args) {
        ABSL_RETURN_IF_ERROR(ResolveSelectorsPass(gpu_info, {}, args, &arg));
      }
      std::string patch;
      std::string linkable_patch;
      GPUObjectDescriptor* desc_ptr;
      ABSL_RETURN_IF_ERROR(args.GetDescriptor(object_name, &desc_ptr));
      auto names = desc_ptr->GetGPUResources(gpu_info).GetNames();
      if (desc_ptr && !linkables.empty()) {
        if (selector_name == "WriteLinear") {
          return absl::InternalError("Can not link with WriteLinear.");
        } else if (selector_name == "Write") {
          auto it = linkables.find(object_name);
          if (it != linkables.end()) {
            ABSL_RETURN_IF_ERROR(ResolveLinking(
                gpu_info, it->second, &function_args, &linkable_patch));
            ABSL_RETURN_IF_ERROR(
                ResolveConstExprPass(gpu_info, args, &linkable_patch));
            ABSL_RETURN_IF_ERROR(
                ResolveSelectorsPass(gpu_info, {}, args, &linkable_patch));
          }
        } else if (selector_name == "Read") {
          auto it = linkables.find(object_name);
          if (it != linkables.end()) {
            std::string x_coord, y_coord, z_coord, s_coord, b_coord;
            ABSL_RETURN_IF_ERROR(
                it->second.tensor_desc->GetLinkingContextFromReadSelector(
                    function_args, &x_coord, &y_coord, &z_coord, &s_coord,
                    &b_coord));
            std::string reorder_patch = absl::StrReplaceAll(
                it->second.code, {{"SRC_X", "r_s_x"},
                                  {"SRC_Y", "r_s_y"},
                                  {"SRC_S", "r_s_s"},
                                  {"SRC_B", "r_s_b"},
                                  {"SRC_WIDTH", "args.src_tensor.Width()"},
                                  {"SRC_HEIGHT", "args.src_tensor.Height()"},
                                  {"SRC_SLICES", "args.src_tensor.Slices()"},
                                  {"SRC_BATCH", "args.src_tensor.Batch()"},
                                  {"DST_X", x_coord},
                                  {"DST_Y", y_coord},
                                  {"DST_S", s_coord},
                                  {"DST_B", "B"}});
            ABSL_RETURN_IF_ERROR(
                ResolveConstExprPass(gpu_info, args, &reorder_patch));
            ABSL_RETURN_IF_ERROR(
                ResolveSelectorsPass(gpu_info, {}, args, &reorder_patch));
            reorder_patch = absl::StrCat("{\n", reorder_patch, "\n}");

            // find previous first ';' or '}'
            size_t prev_pos = arg_pos;
            while (prev_pos > 0 && (*code)[prev_pos] != ';' &&
                   (*code)[prev_pos] != '}') {
              prev_pos--;
            }
            code->insert(prev_pos + 1, reorder_patch);
            arg_pos += reorder_patch.size();
            close_bracket_pos += reorder_patch.size();
            function_args.clear();
            function_args = {"r_s_x", "r_s_y", "r_s_s"};
            if (it->second.tensor_desc->HasAxis(Axis::BATCH)) {
              function_args.push_back("r_s_b");
            }
          }
        }
      }
      // Check if we need to add quantized write function
      if (desc_ptr && desc_ptr->IsTensorDescriptor()) {
        TensorDescriptor* tensor_desc =
            static_cast<TensorDescriptor*>(desc_ptr);
        bool add_quantized_write = true;
        add_quantized_write &= (gpu_info.IsGlsl() || gpu_info.IsApiWebGpu());
        add_quantized_write &=
            (tensor_desc->GetStorageType() == TensorStorageType::BUFFER ||
             tensor_desc->GetStorageType() == TensorStorageType::IMAGE_BUFFER);
        DataType type = tensor_desc->GetDataType();
        add_quantized_write &=
            (type == DataType::BOOL || type == DataType::UINT8 ||
             type == DataType::INT8 || type == DataType::UINT16 ||
             type == DataType::INT16 || type == DataType::BFLOAT16);
        add_quantized_write &=
            (selector_name == "Write" || selector_name == "WriteLinear");
        if (add_quantized_write) {
          ABSL_RETURN_IF_ERROR(
              AddQuantizedBufferWrite(gpu_info, type, &write_fcns));
        }
      }
      absl::Status selector_status = desc_ptr->PerformSelector(
          gpu_info, selector_name, function_args, template_args, &patch);
      if (!selector_status.ok()) {
        return absl::Status(
            selector_status.code(),
            absl::StrCat(
                selector_status.message(), "; selector object=", object_name,
                ", selector=", selector_name, ", args=[",
                absl::StrJoin(function_args, ", "), "], template_args=[",
                absl::StrJoin(template_args, ", "), "], descriptor=",
                desc_ptr->IsTensorDescriptor()
                    ? ToStringWithShape(
                          *static_cast<TensorDescriptor*>(desc_ptr))
                    : ""));
      }
      for (const auto& member_name : names) {
        const std::string new_name =
            absl::StrCat(kArgsPrefix, object_name, "_", member_name);
        ReplaceAllWords(member_name, new_name, &patch);
      }
      if (!linkable_patch.empty()) {
        patch = absl::StrCat("{\n", linkable_patch, patch, ";\n}");
      }
      code->replace(arg_pos, close_bracket_pos + 1 - arg_pos, patch);
      position = arg_pos + patch.size();
    } else {
      position = arg_pos + strlen(kArgsPrefix);
    }
    next_position = code->find(kArgsPrefix, position);
  }
  *code = write_fcns + *code;
  return absl::OkStatus();
}

// resolve constructions of type: ucl::func_name<template_args>(args)
// template_args optional and not like c++ template args, they are function
// specific and can represent any possible information that defined by function.
// Examples:
//   'ucl::convert_to<float>(value)' -> convert_float(value) (for OpenCL)
//   'ucl::init<float>(value)' -> (float)(value) (for OpenCL)
absl::Status ResolveSystemFunctionsPass(const GpuInfo& gpu_info,
                                        std::string* code,
                                        CodeInfo* code_info) {
  std::unordered_set<std::string> function_names;
  constexpr char kPrefix[] = "ucl::";
  size_t next_position = code->find(kPrefix);
  // Declare arg vectors here to avoid reallocating each loop.
  std::vector<std::string> template_args;
  std::vector<std::string> function_args;
  while (next_position != std::string::npos) {
    size_t start_pos = next_position;
    next_position += strlen(kPrefix);
    const std::string function_name =
        std::string(GetNextWord(*code, next_position));
    next_position += function_name.size();
    char next = (*code)[next_position];
    template_args.clear();
    if (next == '<') {
      size_t close_bracket_pos;
      ABSL_RETURN_IF_ERROR(ParseArguments(*code, next_position,
                                          &close_bracket_pos, &template_args));
      next_position = close_bracket_pos + 1;
      next = (*code)[next_position];
    }
    if (next != '(') {
      return absl::NotFoundError(
          absl::StrCat("Expected '(' after ucl::", function_name, " call."));
    }
    function_args.clear();
    size_t close_bracket_pos;
    ABSL_RETURN_IF_ERROR(ParseArguments(*code, next_position,
                                        &close_bracket_pos, &function_args));
    if (function_name == "GetGlobalId") {
      code_info->uses_global_id = true;
    } else if (function_name == "GetLocalId") {
      code_info->uses_local_id = true;
    } else if (function_name == "GetGroupId") {
      code_info->uses_group_id = true;
    } else if (function_name == "GetSubGroupLocalId") {
      code_info->uses_sub_group_local_id = true;
    } else if (function_name == "GetSubGroupId") {
      code_info->uses_sub_group_id = true;
    } else if (function_name == "GetSubGroupSize") {
      code_info->uses_sub_group_size = true;
    } else if (function_name == "GetGroupSize") {
      code_info->uses_group_size = true;
    }
    std::string patch;
    if (function_name == "U8ToVec4I2") {
      const std::string new_function_name = "Ucl" + function_name;
      code->replace(start_pos, strlen(kPrefix) + function_name.size(),
                    new_function_name);
      if (function_names.find(new_function_name) == function_names.end()) {
        std::string function_code = GetU8ToVec4I2Function(gpu_info);
        ABSL_RETURN_IF_ERROR(
            ResolveSystemFunctionsPass(gpu_info, &function_code, code_info));
        *code = function_code + *code;
        function_names.insert(new_function_name);
      }
    } else {
      ABSL_RETURN_IF_ERROR(PerformSystemFunction(
          gpu_info, function_name, function_args, template_args, &patch));
      code->replace(start_pos, close_bracket_pos + 1 - start_pos, patch);
    }
    if (function_name.rfind("ConvertFromBfloat", 0) == 0) {  // startswith
      ABSL_RETURN_IF_ERROR(AddConvertFromBfloat(gpu_info, template_args, code));
      start_pos = 0;
    } else if (function_name.rfind("ConvertToBfloat", 0) == 0) {  // startswith
      ABSL_RETURN_IF_ERROR(AddConvertToBfloat(gpu_info, template_args, code));
      start_pos = 0;
    }
    if (absl::StrContains(patch, "ReinterpretUvec2ToHalf4")) {
      ABSL_RETURN_IF_ERROR(AddGlslReinterpretUvec2ToHalf4(gpu_info, code));
      start_pos = 0;
    }
    if (absl::StrContains(patch, "ReinterpretHalf4ToUvec2")) {
      ABSL_RETURN_IF_ERROR(AddGlslReinterpretHalf4ToUvec2(gpu_info, code));
      start_pos = 0;
    }
    if (absl::StrContains(patch, "BitsTo") && absl::StrContains(patch, "vec")) {
      ABSL_RETURN_IF_ERROR(AddGlslBitsToVec(gpu_info, GetNextWord(patch, 0),
                                            template_args, code));
      start_pos = 0;
    }
    next_position = code->find(kPrefix, start_pos);
  }
  return absl::OkStatus();
}

}  // namespace

void GPUOperation::SetSrc(GpuSpatialTensor* ptr, size_t index) {
  if (index >= src_.size()) {
    src_.resize(index + 1, nullptr);
  }
  src_[index] = ptr;
}

void GPUOperation::SetDst(GpuSpatialTensor* ptr, size_t index) {
  if (index >= dst_.size()) {
    dst_.resize(index + 1, nullptr);
  }
  dst_[index] = ptr;
}

GPUOperation::GPUOperation(GPUOperation&& operation)
    : args_(std::move(operation.args_)),
      code_(std::move(operation.code_)),
      work_group_size_(operation.work_group_size_),
      compiler_options_(std::move(operation.compiler_options_)),
      tensor_to_grid_(operation.tensor_to_grid_),
      flops_(operation.flops_),
      const_args_size_(operation.const_args_size_),
      read_size_(operation.read_size_),
      write_size_(operation.write_size_),
      code_info_(operation.code_info_),
      src_(std::move(operation.src_)),
      dst_(std::move(operation.dst_)),
      grid_dimension_(operation.grid_dimension_),
      work_group_launch_order_(operation.work_group_launch_order_),
      grid_size_(operation.grid_size_),
      src_objects_names_(std::move(operation.src_objects_names_)),
      dst_objects_names_(std::move(operation.dst_objects_names_)),
      work_groups_count_(operation.work_groups_count_),
      elementwise_(operation.elementwise_),
      elementwise_inputs_(operation.elementwise_inputs_),
      second_elementwise_tensor_name_(
          operation.second_elementwise_tensor_name_),
      linkable_count_(operation.linkable_count_),
      elementwise_code_(std::move(operation.elementwise_code_)),
      reorder_op_(operation.reorder_op_),
      reorder_op_count_(operation.reorder_op_count_),
      reorder_code_(std::move(operation.reorder_code_)) {}

GPUOperation& GPUOperation::operator=(GPUOperation&& operation) {
  if (this != &operation) {
    args_ = std::move(operation.args_);
    code_ = std::move(operation.code_);
    std::swap(work_group_size_, operation.work_group_size_);
    compiler_options_ = std::move(operation.compiler_options_);
    tensor_to_grid_ = operation.tensor_to_grid_;
    flops_ = operation.flops_;
    const_args_size_ = operation.const_args_size_;
    read_size_ = operation.read_size_;
    write_size_ = operation.write_size_;
    code_info_ = operation.code_info_;
    src_ = std::move(operation.src_);
    dst_ = std::move(operation.dst_);
    std::swap(grid_dimension_, operation.grid_dimension_);
    std::swap(work_group_launch_order_, operation.work_group_launch_order_);
    std::swap(grid_size_, operation.grid_size_);
    src_objects_names_ = std::move(operation.src_objects_names_);
    dst_objects_names_ = std::move(operation.dst_objects_names_);
    std::swap(work_groups_count_, operation.work_groups_count_);
    elementwise_ = operation.elementwise_;
    std::swap(elementwise_inputs_, operation.elementwise_inputs_);
    std::swap(second_elementwise_tensor_name_,
              operation.second_elementwise_tensor_name_);
    std::swap(linkable_count_, operation.linkable_count_);
    elementwise_code_ = std::move(operation.elementwise_code_);
    std::swap(reorder_op_, operation.reorder_op_);
    std::swap(reorder_op_count_, operation.reorder_op_count_);
    reorder_code_ = std::move(operation.reorder_code_);
  }
  return *this;
}

absl::Status GPUOperation::SetOutputDescriptor(
    size_t index, const TensorDescriptor& new_tensor_desc) {
  if (index >= dst_objects_names_.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Index ", index, " is out of range [0, ",
                     dst_objects_names_.size(), ")"));
  }
  TensorDescriptor* dst_tensor_desc;
  ABSL_RETURN_IF_ERROR(
      GetTensorDescriptor(dst_objects_names_[index], &dst_tensor_desc));
  new_tensor_desc.CopyWithoutData(dst_tensor_desc);
  return absl::OkStatus();
}

absl::Status GPUOperation::AddOperation(const GpuInfo& gpu_info,
                                        GPUOperation* operation) {
  ABSL_RETURN_IF_ERROR(
      ResolveConstExprPass(gpu_info, operation->args_, &operation->code_));
  if (!const_expr_resolved_) {
    // if we fuse ops, dst desc can changes and as result, const expr can also
    // changes.
    ABSL_RETURN_IF_ERROR(ResolveConstExprPass(gpu_info, args_, &code_));
    const_expr_resolved_ = true;
  }
  TensorDescriptor* dst_tensor_desc;
  ABSL_RETURN_IF_ERROR(
      GetTensorDescriptor(dst_objects_names_[0], &dst_tensor_desc));
  TensorDescriptor* new_dst_tensor_desc;
  ABSL_RETURN_IF_ERROR(operation->GetTensorDescriptor(
      operation->dst_objects_names_[0], &new_dst_tensor_desc));
  const auto prev_type = dst_tensor_desc->GetDataType();
  new_dst_tensor_desc->CopyWithoutData(dst_tensor_desc);
  linkable_count_ += (operation->linkable_count_ + 1);
  std::string code = operation->elementwise_code_;
  std::string unique_postfix = absl::StrCat("_link", linkable_count_);
  code = absl::StrReplaceAll(
      code, {{"interm_value", "interm_value" + unique_postfix}});
  operation->args_.RenameArgs(unique_postfix, &code);
  operation->second_elementwise_tensor_name_ += unique_postfix;
  if (elementwise_code_.empty()) {
    elementwise_code_ = code;
    elementwise_inputs_ = operation->elementwise_inputs_;
    second_elementwise_tensor_name_ =
        operation->second_elementwise_tensor_name_;
  } else {
    if (operation->elementwise_inputs_ == 2) {
      if (elementwise_inputs_ == 2) {
        // if we have fusion of 2 2-input elementwise ops, we will get 3-input
        // elementwise, but currently we support only max 2-input elementwise.
        // So we will resolve one input here.
        ABSL_RETURN_IF_ERROR(ResolveSecondElementwiseInput(gpu_info));
      }
      second_elementwise_tensor_name_ =
          operation->second_elementwise_tensor_name_;
      elementwise_inputs_ = 2;
    }
    const std::string new_value_name = "interm_value" + unique_postfix;
    code = absl::StrReplaceAll(code, {{"in_value", new_value_name}});
    elementwise_code_ =
        absl::StrReplaceAll(elementwise_code_, {{"out_value", new_value_name}});
    const std::string type_decl = ToUclDataType(prev_type, 4);
    const std::string out_var_declaration =
        "\n " + type_decl + " " + new_value_name + ";\n";
    elementwise_code_ =
        absl::Substitute(elementwise_code_, out_var_declaration);
    elementwise_code_ = elementwise_code_ + "\n" + code;
  }
  ABSL_RETURN_IF_ERROR(args_.Merge(std::move(operation->args_), unique_postfix,
                                   {"src_tensor", "dst_tensor"}));
  for (size_t i = 1; i < operation->src_objects_names_.size(); ++i) {
    src_objects_names_.push_back(operation->src_objects_names_[i] +
                                 unique_postfix);
  }
  for (size_t i = 1; i < operation->dst_objects_names_.size(); ++i) {
    dst_objects_names_.push_back(operation->dst_objects_names_[i] +
                                 unique_postfix);
  }
  return absl::OkStatus();
}

absl::Status GPUOperation::AddReorderOperation(const BHWC& interm_shape,
                                               GPUOperation* operation) {
  TensorDescriptor* src_tensor_desc;
  ABSL_RETURN_IF_ERROR(
      GetTensorDescriptor(src_objects_names_[0], &src_tensor_desc));

  TensorDescriptor* new_src_tensor_desc;
  ABSL_RETURN_IF_ERROR(operation->GetTensorDescriptor(
      operation->src_objects_names_[0], &new_src_tensor_desc));

  reorder_op_count_ += (operation->reorder_op_count_ + 1);
  std::string unique_postfix = absl::StrCat("_link", reorder_op_count_);

  // declaring intermediate coords;
  const std::string interm_xc = "xc" + unique_postfix;
  const std::string interm_yc = "yc" + unique_postfix;
  const std::string interm_sc = "sc" + unique_postfix;
  const std::string interm_bc = "bc" + unique_postfix;
  std::string code = "  int " + interm_xc;
  code += ", " + interm_yc;
  code += ", " + interm_sc;
  if (src_tensor_desc->HasAxis(Axis::BATCH)) {
    code += ", " + interm_bc;
  }
  code += ";\n";
  code += "  {\n" +
          absl::StrReplaceAll(
              reorder_code_,
              {{"SRC_X", interm_xc},
               {"SRC_Y", interm_yc},
               {"SRC_S", interm_sc},
               {"SRC_B", interm_bc},
               {"SRC_WIDTH", std::to_string(interm_shape.w)},
               {"SRC_HEIGHT", std::to_string(interm_shape.h)},
               {"SRC_SLICES", std::to_string(DivideRoundUp(interm_shape.c, 4))},
               {"SRC_BATCH", std::to_string(interm_shape.b)}}) +
          "  }\n";

  code += absl::StrReplaceAll(operation->reorder_code_, {{"DST_X", interm_xc},
                                                         {"DST_Y", interm_yc},
                                                         {"DST_S", interm_sc},
                                                         {"DST_B", interm_bc}});

  new_src_tensor_desc->CopyWithoutData(src_tensor_desc);
  reorder_code_ = code;
  return absl::OkStatus();
}

void GPUOperation::ResolveReorderFinalShape(const BHWC& final_shape) {
  return ResolveReorderFinalShape(
      BHWDC(final_shape.b, final_shape.h, final_shape.w, 1, final_shape.c));
}

void GPUOperation::ResolveReorderFinalShape(const BHWDC& final_shape) {
  reorder_code_ = absl::StrReplaceAll(
      reorder_code_,
      {{"DST_WIDTH", std::to_string(final_shape.w)},
       {"DST_HEIGHT", std::to_string(final_shape.h)},
       {"DST_DEPTH", std::to_string(final_shape.d)},
       {"DST_SLICES", std::to_string(DivideRoundUp(final_shape.c, 4))},
       {"DST_BATCH", std::to_string(final_shape.b)}});
}

absl::Status GPUOperation::ResolveSecondElementwiseInput(
    const GpuInfo& gpu_info) {
  if (elementwise_inputs_ != 2) {
    return absl::FailedPreconditionError(
        "Can not apply ResolveSecondElementwiseInput for non 2 input "
        "elementwise");
  }
  TensorDescriptor* tensor_desc;
  ABSL_RETURN_IF_ERROR(
      GetTensorDescriptor(second_elementwise_tensor_name_, &tensor_desc));
  std::string coords = "X_COORD, Y_COORD";
  if (tensor_desc->HasAxis(Axis::DEPTH)) {
    coords += ", Z_COORD";
  }
  coords += ", S_COORD";
  if (tensor_desc->HasAxis(Axis::BATCH)) {
    coords += ", B_COORD";
  }
  const std::string type =
      absl::StrCat("args.", second_elementwise_tensor_name_, "::type");
  const std::string read_code =
      absl::StrCat(type, " second_value = args.",
                   second_elementwise_tensor_name_, ".Read(", coords, ");\n");
  elementwise_code_ = absl::StrReplaceAll(
      elementwise_code_,
      {{"in2_value", "second_value"}, {"READ_SECOND_VALUE", read_code}});
  elementwise_inputs_ = 1;
  return absl::OkStatus();
}

absl::Status GPUOperation::GetTensorDescriptor(const std::string& tensor_name,
                                               TensorDescriptor* result) const {
  GPUObjectDescriptor* desc_ptr;
  ABSL_RETURN_IF_ERROR(args_.GetDescriptor(tensor_name, &desc_ptr));
  TensorDescriptor* tensor_desc = AsTensorDescriptor(desc_ptr);
  if (tensor_desc != nullptr) {
    *result = *tensor_desc;
    return absl::OkStatus();
  } else {
    // Some buffer based tensors binded as buffers.
    BufferDescriptor* buf_desc = AsBufferDescriptor(desc_ptr);
    if (buf_desc != nullptr) {
      *result = TensorDescriptor(buf_desc->element_type,
                                 TensorStorageType::BUFFER, Layout::HWC);
      return absl::OkStatus();
    }
  }
  return absl::InternalError("requested object is not a tensor");
}

absl::Status GPUOperation::GetTensorDescriptor(const std::string& tensor_name,
                                               TensorDescriptor** result) {
  GPUObjectDescriptor* desc_ptr;
  ABSL_RETURN_IF_ERROR(args_.GetDescriptor(tensor_name, &desc_ptr));
  *result = static_cast<TensorDescriptor*>(desc_ptr);
  return absl::OkStatus();
}

void GPUOperation::AddSrcTensor(const std::string& tensor_name,
                                const TensorDescriptor& desc) {
  src_objects_names_.push_back(tensor_name);
  TensorDescriptor desc_copy;
  desc.CopyWithoutData(&desc_copy);
  auto desc_new = std::make_unique<TensorDescriptor>(std::move(desc_copy));
  args_.AddObjectRef(tensor_name, AccessType::READ, std::move(desc_new));
}

void GPUOperation::AddSrcBuffer(const std::string& buffer_name,
                                const BufferDescriptor& desc) {
  src_objects_names_.push_back(buffer_name);
  auto desc_new = std::make_unique<BufferDescriptor>(desc);
  args_.AddObjectRef(buffer_name, AccessType::READ, std::move(desc_new));
}

void GPUOperation::AddDstTensor(const std::string& tensor_name,
                                const TensorDescriptor& desc) {
  dst_objects_names_.push_back(tensor_name);
  auto desc_new = std::make_unique<TensorDescriptor>(desc);
  args_.AddObjectRef(tensor_name, AccessType::WRITE, std::move(desc_new));
}

void GPUOperation::AddDstBuffer(const std::string& buffer_name,
                                const BufferDescriptor& desc) {
  dst_objects_names_.push_back(buffer_name);
  auto desc_new = std::make_unique<BufferDescriptor>(desc);
  args_.AddObjectRef(buffer_name, AccessType::WRITE, std::move(desc_new));
}

absl::Status GPUOperation::AssembleCode(const GpuInfo& gpu_info) {
  ABSL_RETURN_IF_ERROR(
      ResolveWaveMemory(gpu_info, &code_, &args_, &compiler_options_));
  ABSL_RETURN_IF_ERROR(ResolveWaveMatrix(gpu_info, &code_));
  if (elementwise_inputs_ == 2) {
    ABSL_RETURN_IF_ERROR(ResolveSecondElementwiseInput(gpu_info));
  }
  if (elementwise_ || reorder_op_) {
    if (dst_objects_names_.empty()) {
      return absl::InvalidArgumentError(
          "Invalid operation, dst_objects_names_ empty.");
    }
    TensorDescriptor* dst_tensor_desc;
    ABSL_RETURN_IF_ERROR(
        GetTensorDescriptor(dst_objects_names_[0], &dst_tensor_desc));
    code_ = GetElementWiseCode(*dst_tensor_desc);
    if (reorder_op_) {
      const std::string patch = "  int r_s_x, r_s_y, r_s_s, r_s_b;\n";
      code_ = absl::StrReplaceAll(
          code_, {{"MAIN_FUNCTION($0) {", "MAIN_FUNCTION($0) {\n" + patch}});
    }
    if (const_expr_resolved_) {
      ABSL_RETURN_IF_ERROR(ResolveConstExprPass(gpu_info, args_, &code_));
    }
    elementwise_ = false;
    reorder_op_ = false;
  }
  std::map<std::string, LinkableContext, std::less<>> linkables;
  if (!reorder_code_.empty()) {
    if (src_objects_names_.empty() || dst_objects_names_.empty()) {
      return absl::InvalidArgumentError(
          "Invalid operation, src_objects_names_ or dst_objects_names_ empty.");
    }
    TensorDescriptor* src_tensor_desc;
    ABSL_RETURN_IF_ERROR(
        GetTensorDescriptor(src_objects_names_[0], &src_tensor_desc));
    if (src_tensor_desc->HasAxis(Axis::BATCH)) {
      src_tensor_desc->SetStateVar("batch_id", "B");
    }
    linkables[src_objects_names_[0]] = {reorder_code_, src_tensor_desc};
  }
  if (!elementwise_code_.empty()) {
    if (dst_objects_names_.empty()) {
      return absl::InvalidArgumentError(
          "Invalid operation, dst_objects_names_ empty.");
    }
    TensorDescriptor* dst_tensor_desc;
    ABSL_RETURN_IF_ERROR(
        GetTensorDescriptor(dst_objects_names_[0], &dst_tensor_desc));
    linkables[dst_objects_names_[0]] = {elementwise_code_, dst_tensor_desc};
  }
  if (!const_expr_resolved_) {
    ABSL_RETURN_IF_ERROR(ResolveConstExprPass(gpu_info, args_, &code_));
    const_expr_resolved_ = true;
  }
  ABSL_RETURN_IF_ERROR(
      ResolveSelectorsPass(gpu_info, linkables, args_, &code_));
  if (gpu_info.IsAdreno() &&
      gpu_info.adreno_info.IsGlDriverMajor615Minor88_97()) {
    args_.AddInt("one_gl_reserved", 1);
  }
  ABSL_RETURN_IF_ERROR(
      ResolveSystemFunctionsPass(gpu_info, &code_, &code_info_));
  code_ = GetStringWithoutComments(code_);
  ABSL_RETURN_IF_ERROR(args_.Compile(gpu_info, &code_));
  CalculateConstArgsSize();
  return absl::OkStatus();
}

void GPUOperation::RecalculateWorkGroupsCount() {
  work_groups_count_ = GetWorkGroupsCountInternal(
      grid_dimension_, grid_size_, work_group_size_, work_group_launch_order_);
}

void GPUOperation::CalculateConstArgsSize() {
  const_args_size_ = 0;
  for (const auto& obj : args_.GetObjects()) {
    const_args_size_ += obj.second->GetSizeInBytes();
  }
}

void GPUOperation::GetPossibleDispatches(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info,
    std::vector<DispatchInfo>* dispatches) const {
  const auto work_group_sizes =
      GetPossibleKernelWorkGroups(tuning_type, gpu_info, kernel_info);
  dispatches->resize(work_group_sizes.size());
  for (size_t i = 0; i < work_group_sizes.size(); ++i) {
    auto& dispatch_info = (*dispatches)[i];
    dispatch_info.work_group_size = work_group_sizes[i];
    dispatch_info.work_groups_count = GetWorkGroupsCountInternal(
        grid_dimension_, grid_size_, work_group_sizes[i],
        work_group_launch_order_);
  }
}

std::vector<int3> GPUOperation::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info) const {
  return GetPossibleWorkGroups(tuning_type, gpu_info, kernel_info, grid_size_);
}

int3 GPUOperation::GetGridSize() const {
  if (tensor_to_grid_ == TensorToGrid::kWBToX_HDToY_SToZ) {
    const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
    const int grid_y = dst_[0]->Height() * dst_[0]->Depth();
    const int grid_z = dst_[0]->Slices();
    return int3(grid_x, grid_y, grid_z);
  }
  if (tensor_to_grid_ == TensorToGrid::kWBToX_HDToY_ZIs1) {
    const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
    const int grid_y = dst_[0]->Height() * dst_[0]->Depth();
    const int grid_z = 1;
    return int3(grid_x, grid_y, grid_z);
  }
  if (tensor_to_grid_ == TensorToGrid::kWBToX_HToY_DToZ) {
    const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
    const int grid_y = dst_[0]->Height();
    const int grid_z = dst_[0]->Depth();
    return int3(grid_x, grid_y, grid_z);
  }
  if (tensor_to_grid_ == TensorToGrid::kBToX_YIs1_ZIs1) {
    const int grid_x = dst_[0]->Batch();
    const int grid_y = 1;
    const int grid_z = 1;
    return int3(grid_x, grid_y, grid_z);
  }
  return grid_size_;
}

GPUOperation CreateGpuOperation(const TensorDescriptor& src,
                                const TensorDescriptor& dst,
                                ElementwiseDescriptor&& descriptor) {
  OperationDef op_def;
  op_def.src_tensors.push_back(src);
  op_def.dst_tensors.push_back(dst);
  return CreateGpuOperation(op_def, std::move(descriptor));
}

GPUOperation CreateGpuOperation(const OperationDef& definition,
                                ElementwiseDescriptor&& descriptor) {
  const BHWDC second_shape(2, 2, 2, 2, 2);  // dummy non-broadcasted shape
  return CreateGpuOperation(definition, std::move(descriptor), second_shape,
                            second_shape);
}

GPUOperation CreateGpuOperation(const OperationDef& definition,
                                ElementwiseDescriptor&& descriptor,
                                const BHWDC& second_shape,
                                const BHWDC& dst_shape) {
  GPUOperation op;
  op.elementwise_code_ = std::move(descriptor.code);
  op.elementwise_ = true;
  if (definition.src_tensors.size() > 1 &&
      absl::StrContains(op.elementwise_code_, "in2_value")) {
    const auto second_tensor_def = definition.src_tensors[1];
    if (second_tensor_def.GetLayout() == Layout::LINEAR) {
      std::string s_coord = second_shape.c == 1 ? "0" : "S_COORD";
      std::string read_value_code = absl::StrCat(
          "args.src_tensor_1::type in2_value = args.src_tensor_1.Read(",
          s_coord, ");\n");
      if (second_shape.c == 1) {
        read_value_code += "  in2_value.y = in2_value.x;\n";
        read_value_code += "  in2_value.z = in2_value.x;\n";
        read_value_code += "  in2_value.w = in2_value.x;\n";
      }
      op.elementwise_code_ =
          "$0{" + read_value_code + op.elementwise_code_ + "}";
      op.elementwise_code_ = absl::StrReplaceAll(
          op.elementwise_code_, {{"in2_value", "second_value"}});
      op.elementwise_inputs_ = 1;
    } else if (NeedsBroadcast(second_tensor_def, second_shape, dst_shape)) {
      std::string x_coord = second_shape.w == 1 ? "0" : "X_COORD";
      if (second_shape.w != dst_shape.w && second_shape.w != 1) {
        x_coord = absl::StrCat("(X_COORD % ", second_shape.w, ")");
      }
      std::string y_coord = second_shape.h == 1 ? "0" : "Y_COORD";
      if (second_shape.h != dst_shape.h && second_shape.h != 1) {
        y_coord = absl::StrCat("(Y_COORD % ", second_shape.h, ")");
      }
      std::string s_coord = second_shape.c == 1 ? "0" : "S_COORD";
      // This will not work if second_shape.c % 4 != 0 and tile_size != 1.
      if (second_shape.c != dst_shape.c && second_shape.c != 1) {
        s_coord =
            absl::StrCat("(S_COORD % ", DivideRoundUp(second_shape.c, 4), ")");
      }
      std::string coords = absl::StrCat(x_coord, ", ", y_coord);
      if (second_tensor_def.HasAxis(Axis::DEPTH)) {
        std::string z_coord = second_shape.d == 1 ? "0" : "Z_COORD";
        if (second_shape.d != dst_shape.d && second_shape.d != 1) {
          z_coord = absl::StrCat("(Z_COORD % ", second_shape.d, ")");
        }
        coords += ", " + z_coord;
      }
      coords += ", " + s_coord;
      if (second_tensor_def.HasAxis(Axis::BATCH)) {
        const std::string b_coord = second_shape.b == 1 ? "0" : "B_COORD";
        coords += ", " + b_coord;
      }
      std::string read_value_code = absl::StrCat(
          "args.src_tensor_1::type in2_value = args.src_tensor_1.Read(", coords,
          ");\n");
      if (second_shape.c == 1) {
        read_value_code += "  in2_value.y = in2_value.x;\n";
        read_value_code += "  in2_value.z = in2_value.x;\n";
        read_value_code += "  in2_value.w = in2_value.x;\n";
      }
      op.elementwise_code_ =
          "$0{" + read_value_code + op.elementwise_code_ + "}";
      op.elementwise_code_ = absl::StrReplaceAll(
          op.elementwise_code_, {{"in2_value", "second_value"}});
      op.elementwise_inputs_ = 1;
    } else {
      op.elementwise_code_ =
          "$0{READ_SECOND_VALUE" + op.elementwise_code_ + "}";
      op.elementwise_inputs_ = 2;
      op.second_elementwise_tensor_name_ = "src_tensor_1";
    }
  } else {
    op.elementwise_code_ = "$0{" + op.elementwise_code_ + "}";
    op.elementwise_inputs_ = 1;
  }
  op.args_ = std::move(descriptor.args);
  op.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  for (size_t i = 1; i < definition.src_tensors.size(); ++i) {
    const std::string tensor_name = "src_tensor_" + std::to_string(i);
    op.AddSrcTensor(tensor_name, definition.src_tensors[i]);
  }
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

GPUOperation CreateGpuOperation(const OperationDef& definition,
                                ElementwiseDescriptor&& descriptor,
                                const BHWC& second_shape,
                                const BHWC& dst_shape) {
  return CreateGpuOperation(
      definition, std::move(descriptor),
      BHWDC(second_shape.b, second_shape.h, second_shape.w, 1, second_shape.c),
      BHWDC(dst_shape.b, dst_shape.h, dst_shape.w, 1, dst_shape.c));
}

GPUOperation CreateGpuOperation(const TensorDescriptor& in_out_descriptor,
                                const BufferDescriptor& src_buffer_descriptor,
                                ElementwiseDescriptor&& descriptor) {
  GPUOperation op;
  op.elementwise_code_ = "$0{" + descriptor.code + "}";
  op.elementwise_inputs_ = 1;
  op.args_ = std::move(descriptor.args);
  op.AddSrcTensor("src_tensor", in_out_descriptor);
  op.AddSrcBuffer("src_buffer", src_buffer_descriptor);
  op.AddDstTensor("dst_tensor", in_out_descriptor);
  op.elementwise_ = true;
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

GPUOperation CreateReorderGpuOperation(const OperationDef& definition,
                                       std::string&& code) {
  GPUOperation op;
  op.reorder_code_ = std::move(code);
  op.reorder_op_ = true;
  op.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

absl::Status FuseElemWithElemInternal(
    const GpuInfo& gpu_info, GPUOperation&& elem0, GPUOperation&& elem1,
    const std::vector<std::pair<std::string, std::string>>& replacements,
    GPUOperation* result) {
  const int linkable_count =
      std::max(elem0.linkable_count_, elem1.linkable_count_) + 1;

  const std::string unique_postfix = absl::StrCat("_link", linkable_count);
  elem1.args_.RenameArgs(unique_postfix, &elem1.elementwise_code_);

  TensorDescriptor* dst_tensor_desc;
  ABSL_RETURN_IF_ERROR(
      elem0.GetTensorDescriptor(elem0.dst_objects_names_[0], &dst_tensor_desc));
  const auto link_value_type = dst_tensor_desc->GetDataType();
  const std::string link_value_name = "interm_value" + unique_postfix;
  const std::string type_decl = ToUclDataType(link_value_type, 4);
  const std::string value_declaration =
      "\n " + type_decl + " " + link_value_name + ";\n";
  elem0.elementwise_code_ = absl::StrReplaceAll(
      elem0.elementwise_code_, {{"out_value", link_value_name}});
  elem0.elementwise_code_ =
      absl::Substitute(elem0.elementwise_code_, value_declaration);

  std::vector<std::pair<const absl::string_view, std::string>> replacements_new;
  for (size_t i = 0; i < replacements.size(); ++i) {
    if (replacements[i].second == "LINK_VALUE") {
      replacements_new.push_back(
          {absl::string_view(replacements[i].first), link_value_name});
    } else {
      replacements_new.push_back(
          {absl::string_view(replacements[i].first), replacements[i].second});
    }
  }
  elem1.elementwise_code_ =
      absl::StrReplaceAll(elem1.elementwise_code_,
                          {{"interm_value", "interm_value" + unique_postfix}});
  elem1.elementwise_code_ =
      absl::StrReplaceAll(elem1.elementwise_code_, replacements_new);

  *result = GPUOperation();
  result->elementwise_ = true;
  result->elementwise_inputs_ = 1;
  result->tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  result->elementwise_code_ =
      elem0.elementwise_code_ + "\n" + elem1.elementwise_code_;
  result->linkable_count_ = linkable_count;
  result->src_objects_names_ = {"src_tensor"};
  result->dst_objects_names_ = {"dst_tensor"};
  result->args_ = std::move(elem0.args_);
  {  // update dst_tensor descriptor
    TensorDescriptor* dst_tensor_desc;
    ABSL_RETURN_IF_ERROR(
        result->GetTensorDescriptor("dst_tensor", &dst_tensor_desc));
    TensorDescriptor* new_dst_tensor_desc;
    ABSL_RETURN_IF_ERROR(
        elem1.GetTensorDescriptor("dst_tensor", &new_dst_tensor_desc));
    new_dst_tensor_desc->CopyWithoutData(dst_tensor_desc);
  }
  ABSL_RETURN_IF_ERROR(result->args_.Merge(
      std::move(elem1.args_), unique_postfix,
      {"src_tensor", "dst_tensor", elem1.second_elementwise_tensor_name_}));
  return absl::OkStatus();
}

absl::Status FuseSimpleElemWithSimpleElem(const GpuInfo& gpu_info,
                                          GPUOperation&& elem0,
                                          GPUOperation&& elem1,
                                          GPUOperation* result) {
  return FuseElemWithElemInternal(gpu_info, std::move(elem0), std::move(elem1),
                                  {{"in_value", "LINK_VALUE"}}, result);
}

absl::Status Fuse2InputElemWithSimpleElemAsFirstInput(const GpuInfo& gpu_info,
                                                      GPUOperation&& elem0,
                                                      GPUOperation&& elem1,
                                                      GPUOperation* result) {
  return FuseElemWithElemInternal(gpu_info, std::move(elem0), std::move(elem1),
                                  {{"in_value", "LINK_VALUE"},
                                   {"READ_SECOND_VALUE", ""},
                                   {"in2_value", "in_value"}},
                                  result);
}

absl::Status Fuse2InputElemWithSimpleElemAsSecondInput(const GpuInfo& gpu_info,
                                                       GPUOperation&& elem0,
                                                       GPUOperation&& elem1,
                                                       GPUOperation* result) {
  return FuseElemWithElemInternal(
      gpu_info, std::move(elem0), std::move(elem1),
      {{"READ_SECOND_VALUE", ""}, {"in2_value", "LINK_VALUE"}}, result);
}

//      input                input           input
//     /    \               /    \             |
//  elem0  elem1           |    elem1          |
//     \    /      -->      \    /      -->  elem
//   elem_root              elem2              |
//       |                    |                |
//     output               output           output
absl::Status Fuse2InputElemWith2SimpleElem(const GpuInfo& gpu_info,
                                           GPUOperation&& elem0,
                                           GPUOperation&& elem1,
                                           GPUOperation&& elem_root,
                                           GPUOperation* result) {
  elem0.linkable_count_ =
      std::max(elem0.linkable_count_, elem1.linkable_count_);
  elem0.linkable_count_ =
      std::max(elem0.linkable_count_, elem_root.linkable_count_);
  GPUOperation elem2;
  ABSL_RETURN_IF_ERROR(
      FuseElemWithElemInternal(gpu_info, std::move(elem0), std::move(elem_root),
                               {{"in_value", "LINK_VALUE"}}, &elem2));
  return FuseElemWithElemInternal(
      gpu_info, std::move(elem1), std::move(elem2),
      {{"READ_SECOND_VALUE", ""}, {"in2_value", "LINK_VALUE"}}, result);
}

int ConvRuntimeCheckDesc::GetSlicesAlignment() const {
  return DivideRoundUp(kChannelsAlignment, 4);
}

std::string ConvRuntimeCheckDesc::GetRuntimeEndSlice(
    const std::string& channels, const std::string& max_slices) const {
  return absl::Substitute("min(((($0 + $1 - 1) / $1) * $1) / 4, $2)", channels,
                          kChannelsAlignment, max_slices);
}

}  // namespace ml_drift
