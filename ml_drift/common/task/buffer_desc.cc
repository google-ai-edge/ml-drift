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

#include "ml_drift/common/task/buffer_desc.h"

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/util.h"

namespace ml_drift {
namespace {
DataType DataTypeFromTemplateArg(const std::string& template_arg) {
  if (template_arg == "half") {
    return DataType::FLOAT16;
  } else if (template_arg == "float") {
    return DataType::FLOAT32;
  } else if (template_arg == "int") {
    return DataType::INT32;
  } else if (template_arg == "short") {
    return DataType::INT16;
  } else if (template_arg == "char") {
    return DataType::INT8;
  } else if (template_arg == "uint") {
    return DataType::UINT32;
  } else if (template_arg == "ushort") {
    return DataType::UINT16;
  } else if (template_arg == "uchar") {
    return DataType::UINT8;
  } else if (template_arg == "bool") {
    return DataType::BOOL;
  }
  return DataType::UNKNOWN;
}

std::string GetVec4FromVec16(const GpuInfo& gpu_info, int index,
                             const std::string& value_name) {
  if (gpu_info.IsApiMetal()) {
    const std::string coords[4] = {".v0", ".v1", ".v2", ".v3"};
    return absl::StrCat(value_name, coords[index]);
  } else if (gpu_info.IsApiOpenCl()) {
    const std::string coords[4] = {".s0123", ".s4567", ".s89ab", ".scdef"};
    return absl::StrCat(value_name, coords[index]);
  } else {
    return absl::StrCat(value_name, "[", index, "]");
  }
}
}  // namespace

GPUResources BufferDescriptor::GetGPUResources(const GpuInfo& gpu_info) const {
  GPUResources resources;
  GPUBufferDescriptor desc;
  desc.data_type = element_type;
  desc.access_type = access_type_;
  desc.element_size = element_size;
  desc.memory_type = memory_type;
  desc.attributes = attributes;
  if ((gpu_info.IsGlsl() || gpu_info.IsApiWebGpu()) &&
      memory_type == MemoryType::CONSTANT) {
    desc.attributes.push_back(
        std::to_string(size / (element_size * SizeOf(element_type))));
  }
  resources.buffers.push_back({"buffer", desc});
  return resources;
}

absl::Status BufferDescriptor::PerformConstExpr(const GpuInfo& gpu_info,
                                                absl::string_view const_expr,
                                                std::string* result) const {
  DataType data_type = element_type;
  if (const_expr == "type" || const_expr == "scalar_type") {
    const int vec_size = const_expr == "scalar_type" ? 1 : element_size;
    *result = ToUclDataType(data_type, vec_size);
    return absl::OkStatus();
  } else {
    return absl::UnimplementedError(
        absl::StrCat("Can not resolve constant expression - ", const_expr));
  }
}

absl::Status BufferDescriptor::PerformSelector(
    const GpuInfo& gpu_info, absl::string_view selector,
    const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
  if (selector == "Read") {
    return PerformReadSelector(gpu_info, args, template_args, result);
  } else if (selector == "ReadVec16AsVec4x4") {
    return PerformReadVec16AsVec4x4Selector(gpu_info, args, template_args,
                                            result);
  } else if (selector == "ReadAsU16") {
    return PerformReadAsU16Selector(gpu_info, args, result);
  } else if (selector == "ReadAsI16") {
    return PerformReadAsI16Selector(gpu_info, args, result);
  } else if (selector == "ReadAsU8") {
    return PerformReadAsU8Selector(gpu_info, args, result);
  } else if (selector == "ReadAsI8") {
    return PerformReadAsI8Selector(gpu_info, args, result);
  } else if (selector == "Write") {
    return PerformWriteSelector(gpu_info, args, result);
  } else if (selector == "GetPtr") {
    return PerformGetPtrSelector(gpu_info, args, template_args, result);
  } else {
    return absl::NotFoundError(absl::StrCat(
        "BufferDescriptor don't have selector with name - ", selector));
  }
}

absl::Status BufferDescriptor::PerformReadSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
  if (args.size() != 1) {
    return absl::NotFoundError(
        absl::StrCat("BufferDescriptor Read require one argument, but ",
                     args.size(), " was passed"));
  }
  if (gpu_info.IsGlsl()) {
    if (element_type == DataType::FLOAT16 &&
        !gpu_info.IsGlslSupportsExplicitFp16()) {
      if (memory_type == MemoryType::CONSTANT) {
        bool is_kernel_global_space = false;
        for (const auto& attribute : attributes) {
          if (attribute == "kernel_global_space") {
            is_kernel_global_space = true;
            break;
          }
        }
        if (is_kernel_global_space) {
          *result = absl::StrCat("buffer[", args[0], "]");
          return absl::OkStatus();
        }
        const std::string arg0 = "(" + args[0] + ")";
        *result =
            absl::StrCat("vec4(unpackHalf2x16(buffer[", arg0, " / 2][", arg0,
                         " % 2 == 0 ? 0 : 2]), unpackHalf2x16(buffer[", arg0,
                         " / 2][", arg0, " % 2 == 0 ? 1 : 3]))");
      } else {
        if (element_size == 4) {
          if (template_args.size() == 1) {
            DataType dst_type = DataTypeFromTemplateArg(template_args[0]);
            if (dst_type != DataType::UINT32) {
              return absl::InvalidArgumentError("Only Read<uint> possible.");
            }
            *result = absl::StrCat("buffer[", args[0], "]");
          } else {
            *result = absl::StrCat("ucl::Reinterpret<uint2, half4>(buffer[",
                                   args[0], "])");
          }
        } else if (element_size == 16) {
          const std::string vec0 = absl::Substitute(
              "vec4(unpackHalf2x16(buffer[$0].a.x), "
              "unpackHalf2x16(buffer[$0].a.y))",
              args[0]);
          const std::string vec1 = absl::Substitute(
              "vec4(unpackHalf2x16(buffer[$0].a.z), "
              "unpackHalf2x16(buffer[$0].a.w))",
              args[0]);
          const std::string vec2 = absl::Substitute(
              "vec4(unpackHalf2x16(buffer[$0].b.x), "
              "unpackHalf2x16(buffer[$0].b.y))",
              args[0]);
          const std::string vec3 = absl::Substitute(
              "vec4(unpackHalf2x16(buffer[$0].b.z), "
              "unpackHalf2x16(buffer[$0].b.w))",
              args[0]);
          *result = absl::Substitute("mat4x4($0, $1, $2, $3)", vec0, vec1, vec2,
                                     vec3);
        }
      }
    } else {
      *result = absl::StrCat("buffer[", args[0], "]");
    }
    return absl::OkStatus();
  } else if (gpu_info.IsApiWebGpu()) {
    bool is_kernel_global_space = false;
    for (const auto& attribute : attributes) {
      if (attribute == "kernel_global_space") {
        is_kernel_global_space = true;
        break;
      }
    }
    if (is_kernel_global_space) {
      *result = absl::StrCat("buffer[", args[0], "]");
      return absl::OkStatus();
    }
    if (element_type == DataType::FLOAT16 &&
        !gpu_info.webgpu_info.supports_fp16) {
      *result = absl::StrCat("Unpack4x16float(buffer.data[", args[0], "])");
      return absl::OkStatus();
    } else {
      *result = absl::StrCat("buffer.data[", args[0], "]");
      return absl::OkStatus();
    }
  } else {
    *result = absl::StrCat("buffer[", args[0], "]");
    return absl::OkStatus();
  }
}

absl::Status BufferDescriptor::PerformReadVec16AsVec4x4Selector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
  if (element_size != 16) {
    return absl::UnavailableError(
        "BufferDescriptor::ReadVec16AsVec4x4 available only for vec16 "
        "type.");
  }
  if (args.size() != 5) {
    return absl::NotFoundError(
        "BufferDescriptor::ReadVec16AsVec4x4 require 5 arguments. Example: "
        "args.buffer.ReadVec16AsVec4x4(v0, v1, v2, v3, index).");
  }
  DataType dst_type = element_type;
  if (template_args.size() == 1) {
    dst_type = DataTypeFromTemplateArg(template_args[0]);
  }
  if (dst_type == DataType::UNKNOWN) {
    return absl::UnavailableError(
        "Unsupported template_arg in BufferDescriptor::ReadVec16AsVec4x4.");
  }
  std::string c;
  c += "  {\n";
  if (gpu_info.IsGlsl()) {
    if (element_type == DataType::FLOAT16 &&
        !gpu_info.IsGlslSupportsExplicitFp16()) {
      c += "  uvec8 uw = buffer[" + args[4] + "];\n";
      c += "  mediump mat4x4 w;\n";
      c += "  w[0] = vec4(unpackHalf2x16(uw.a.x), unpackHalf2x16(uw.a.y));\n";
      c += "  w[1] = vec4(unpackHalf2x16(uw.a.z), unpackHalf2x16(uw.a.w));\n";
      c += "  w[2] = vec4(unpackHalf2x16(uw.b.x), unpackHalf2x16(uw.b.y));\n";
      c += "  w[3] = vec4(unpackHalf2x16(uw.b.z), unpackHalf2x16(uw.b.w));\n";
    } else {
      c += "  " + ToUclDataType(element_type, 16) + " w = buffer[" + args[4] +
           "];\n";
    }
  } else if (gpu_info.IsApiWebGpu()) {
    if (element_type == DataType::FLOAT16 &&
        !gpu_info.webgpu_info.supports_fp16) {
      c += "  var uw0 : vec4<u32> = buffer.data[(" + args[4] + ") * 2 + 0];\n";
      c += "  var uw1 : vec4<u32> = buffer.data[(" + args[4] + ") * 2 + 1];\n";
      c += "  var w : mat4x4<f32>;\n";
      c += "  w[0] = Unpack4x16float(uw0.xy);\n";
      c += "  w[1] = Unpack4x16float(uw0.zw);\n";
      c += "  w[2] = Unpack4x16float(uw1.xy);\n";
      c += "  w[3] = Unpack4x16float(uw1.zw);\n";
    } else {
      c += "  " + ToUclDataType(element_type, 16) + " w = buffer.data[" +
           args[4] + "];\n";
    }
  } else {
    c += "  " + ToUclDataType(element_type, 16) + " w = buffer[" + args[4] +
         "];\n";
  }
  const std::string type = ToUclDataType(dst_type, 4);
  c += "  " + args[0] + " = ucl::Convert<" + type + ">(" +
       GetVec4FromVec16(gpu_info, 0, "w") + ");\n";
  c += "  " + args[1] + " = ucl::Convert<" + type + ">(" +
       GetVec4FromVec16(gpu_info, 1, "w") + ");\n";
  c += "  " + args[2] + " = ucl::Convert<" + type + ">(" +
       GetVec4FromVec16(gpu_info, 2, "w") + ");\n";
  c += "  " + args[3] + " = ucl::Convert<" + type + ">(" +
       GetVec4FromVec16(gpu_info, 3, "w") + ");\n";
  c += "  }";
  *result = c;
  return absl::OkStatus();
}

absl::Status BufferDescriptor::PerformReadAsU16Selector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    std::string* result) const {
  if (element_size != 1 || element_type != DataType::UINT32) {
    return absl::UnavailableError(
        "BufferDescriptor::ReadAsU16 possible only for uint32x1 type.");
  }
  if (args.size() != 1) {
    return absl::NotFoundError(
        "BufferDescriptor::ReadAsU16 require 1 argument. Example: "
        "args.buffer.ReadAsU16(index).");
  }
  std::string read_expression;
  const std::string index = gpu_info.IsGlsl()
                                ? "ucl::Convert<uint>(" + args[0] + ") >> 1u"
                                : "(" + args[0] + ") >> 1";
  RETURN_IF_ERROR(PerformReadSelector(gpu_info, {index}, {}, &read_expression));
  *result = absl::StrCat("((", read_expression, " >> ((ucl::Convert<uint>(",
                         args[0], ") & 1u) * 16u)) & 65535u)");
  return absl::OkStatus();
}

absl::Status BufferDescriptor::PerformReadAsI16Selector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    std::string* result) const {
  if (element_size != 1 || element_type != DataType::INT32) {
    return absl::UnavailableError(
        "BufferDescriptor::ReadAsI16 possible only for int32x1 type.");
  }
  if (args.size() != 1) {
    return absl::NotFoundError(
        "BufferDescriptor::ReadAsI16 require 1 argument. Example: "
        "args.buffer.ReadAsI16(index).");
  }
  std::string read_expression;
  const std::string index = gpu_info.IsGlsl()
                                ? "ucl::Convert<uint>(" + args[0] + ") >> 1u"
                                : "(" + args[0] + ") >> 1";
  RETURN_IF_ERROR(PerformReadSelector(gpu_info, {index}, {}, &read_expression));
  *result = absl::StrCat("((", read_expression, " << ((ucl::Convert<uint>(",
                         args[0], ") ^ 1u) * 16u)) >> 16u)");
  return absl::OkStatus();
}

absl::Status BufferDescriptor::PerformReadAsU8Selector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    std::string* result) const {
  if (element_size != 1 ||
      (element_type != DataType::UINT32 && element_type != DataType::INT32)) {
    return absl::UnavailableError(
        "BufferDescriptor::ReadAsU8 possible only for uint32x1 or int32x1 "
        "type.");
  }
  if (args.size() != 1) {
    return absl::NotFoundError(
        "BufferDescriptor::ReadAsU8 require 1 argument. Example: "
        "args.buffer.ReadAsU8(index).");
  }
  std::string read_expression;
  const std::string index = gpu_info.IsGlsl()
                                ? "ucl::Convert<uint>(" + args[0] + ") >> 2u"
                                : "(" + args[0] + ") >> 2";
  RETURN_IF_ERROR(PerformReadSelector(gpu_info, {index}, {}, &read_expression));
  *result = absl::StrCat("((", read_expression, " >> ((ucl::Convert<uint>(",
                         args[0], ") & 3u) * 8u)) & 255u)");
  return absl::OkStatus();
}

absl::Status BufferDescriptor::PerformReadAsI8Selector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    std::string* result) const {
  if (element_size != 1 ||
      (element_type != DataType::UINT32 && element_type != DataType::INT32)) {
    return absl::UnavailableError(
        "BufferDescriptor::ReadAsI8 possible only for uint32x1 or int32x1 "
        "type.");
  }
  if (args.size() != 1) {
    return absl::NotFoundError(
        "BufferDescriptor::ReadAsI8 require 1 argument. Example: "
        "args.buffer.ReadAsI8(index).");
  }
  std::string read_expression;
  const std::string index = gpu_info.IsGlsl()
                                ? "ucl::Convert<uint>(" + args[0] + ") >> 2u"
                                : "(" + args[0] + ") >> 2";
  RETURN_IF_ERROR(PerformReadSelector(gpu_info, {index}, {}, &read_expression));
  // bitfieldExtract-like logic for sign extension
  *result = absl::StrCat("((ucl::Convert<int>(", read_expression,
                         ") << ((3u - (ucl::Convert<uint>(", args[0],
                         ") & 3u)) * 8u)) >> 24)");
  return absl::OkStatus();
}

absl::Status BufferDescriptor::PerformWriteSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    std::string* result) const {
  if (args.size() != 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        "BufferDescriptor Write require two arguments(value, index), but ",
        args.size(), " was passed"));
  }
  const std::string buffer_name =
      gpu_info.IsApiWebGpu() ? "buffer.data" : "buffer";
  *result = absl::StrCat(buffer_name, "[", args[1], "] = ", args[0]);
  return absl::OkStatus();
}

absl::Status BufferDescriptor::PerformGetPtrSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
  if (args.size() > 1) {
    return absl::NotFoundError(absl::StrCat(
        "BufferDescriptor GetPtr require one or zero arguments, but ",
        args.size(), " was passed"));
  }
  if (gpu_info.IsApiWebGpu()) {
    if (!template_args.empty()) {
      return absl::InvalidArgumentError(
          "BufferDescriptor::GetPtr do not support template argument (type "
          "cast) in WebGPU.");
    }
    if (!args.empty()) {
      return absl::InvalidArgumentError(
          "BufferDescriptor::GetPtr do not support offset argument in WebGPU.");
    }
    *result = "buffer.data";
    return absl::OkStatus();
  }
  if (template_args.size() > 1) {
    return absl::NotFoundError(
        absl::StrCat("BufferDescriptor GetPtr require one or zero teemplate "
                     "arguments, but ",
                     template_args.size(), " was passed"));
  }
  std::string conversion;
  if (template_args.size() == 1) {
    const std::string type_name = ToCLDataType(element_type, element_size);
    if (type_name != template_args[0]) {
      conversion = absl::StrCat("(", MemoryTypeToCLType(memory_type), " ",
                                template_args[0], "*)&");
    }
  }
  if (args.empty()) {
    *result = absl::StrCat(conversion, "buffer");
  } else if (conversion.empty()) {
    *result = absl::StrCat("(buffer + ", args[0], ")");
  } else {
    *result = absl::StrCat(conversion, "buffer[", args[0], "]");
  }
  return absl::OkStatus();
}

}  // namespace ml_drift
