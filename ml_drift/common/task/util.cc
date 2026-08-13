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

#include "ml_drift/common/task/util.h"

#include <stddef.h>

#include <cctype>
#include <cfloat>
#include <map>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace {
bool IsWordSymbol(char symbol) {
  return absl::ascii_isalnum(symbol) || symbol == '_';
}

absl::Status PerformInitSelector(const GpuInfo& gpu_info,
                                 const std::vector<std::string>& template_args,
                                 const std::vector<std::string>& args,
                                 std::string* result) {
  if (template_args.size() != 1) {
    return absl::NotFoundError(
        absl::StrCat("init requires one template argument, but ",
                     template_args.size(), " was passed"));
  }
  int vector_size = 0;
  DataType type;
  ABSL_RETURN_IF_ERROR(
      DataTypeFromTemplateArg(template_args[0], &type, &vector_size));
  if (args.size() != 1 && args.size() != vector_size) {
    return absl::NotFoundError(absl::StrCat(
        "init<", template_args[0], "> requires one or ", vector_size,
        ", arguments, but ", args.size(), " was passed"));
  }
  std::string args_str = args[0];
  if (args.size() != 1) {
    for (int i = 1; i < args.size(); ++i) {
      args_str += ", " + args[i];
    }
  }
  if (gpu_info.IsApiOpenCl()) {
    *result = "(" + ToCLDataType(type, vector_size) + ")(" + args_str + ")";
    return absl::OkStatus();
  } else if (gpu_info.IsApiMetal()) {
    *result = ToMetalDataType(type, vector_size) + "(" + args_str + ")";
    return absl::OkStatus();
  } else if (gpu_info.IsGlsl()) {
    *result = ToGlslShaderDataType(type, vector_size, false,
                                   gpu_info.IsGlslSupportsExplicitFp16()) +
              "(" + args_str + ")";
    return absl::OkStatus();
  } else if (gpu_info.IsApiWebGpu()) {
    *result =
        ToWebGpuType(type, vector_size, gpu_info.webgpu_info.supports_fp16) +
        "(" + args_str + ")";
    return absl::OkStatus();
  } else if (gpu_info.IsApiCuda()) {
    *result = "make_" + ToUclDataType(type, vector_size) + "(" + args_str + ")";
    return absl::OkStatus();
  } else {
    return absl::UnimplementedError("No support of this API for Init().");
  }
}

absl::Status PerformConvertSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& template_args,
    const std::vector<std::string>& args, std::string* result) {
  if (template_args.size() != 1) {
    return absl::NotFoundError(
        absl::StrCat("convert requires one template argument, but ",
                     template_args.size(), " was passed"));
  }
  if (args.size() != 1) {
    return absl::NotFoundError(absl::StrCat("convert<", template_args[0],
                                            "> requires one argument, but ",
                                            args.size(), " was passed"));
  }
  int vector_size = 0;
  DataType type;
  ABSL_RETURN_IF_ERROR(
      DataTypeFromTemplateArg(template_args[0], &type, &vector_size));
  if (gpu_info.IsApiOpenCl()) {
    if (type == DataType::BOOL) {
      // In OpenCL for bool4 we are using uchar4
      // From OpenCL specification for "Relational and Equality Operators":
      //   "These functions shall return a 0 if the specified relation is
      //   false and a -1 (i.e. all bits set) if the specified relation is
      //   true for vector argument types."
      // (convert_uchar4((value) != 0) & (uchar4)(1))
      const std::string vec_size =
          vector_size == 1 ? "" : std::to_string(vector_size);
      const std::string float_type = "float" + vec_size;
      const std::string uchar_type = "uchar" + vec_size;
      const std::string bool_float = "(convert_" + float_type + "(" + args[0] +
                                     ") != (" + float_type + ")(0.0f))";
      *result = "(convert_" + uchar_type + "(" + bool_float + ") & (" +
                uchar_type + ")(1))";
      return absl::OkStatus();
    } else {
      *result =
          "convert_" + ToCLDataType(type, vector_size) + "(" + args[0] + ")";
      return absl::OkStatus();
    }
  } else if (gpu_info.IsApiMetal()) {
    *result = ToMetalDataType(type, vector_size) + "(" + args[0] + ")";
    return absl::OkStatus();
  } else if (gpu_info.IsGlsl()) {
    *result = ToGlslShaderDataType(type, vector_size, false,
                                   gpu_info.IsGlslSupportsExplicitFp16()) +
              "(" + args[0] + ")";
    return absl::OkStatus();
  } else if (gpu_info.IsApiWebGpu()) {
    *result =
        ToWebGpuType(type, vector_size, gpu_info.webgpu_info.supports_fp16) +
        "(" + args[0] + ")";
    return absl::OkStatus();
  } else if (gpu_info.IsApiCuda()) {
    *result = "make_" + ToUclDataType(type, vector_size) + "(" + args[0] + ")";
    return absl::OkStatus();
  } else {
    return absl::UnimplementedError("No support of this API for Convert().");
  }
}

absl::Status PerformReinterpretSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& template_args,
    const std::vector<std::string>& args, std::string* result) {
  if (template_args.size() != 2) {
    return absl::NotFoundError(absl::StrCat(
        "ucl::Reinterpret<FromType, ToType>(value) requires two template "
        "arguments, but ",
        template_args.size(), " was passed"));
  }
  if (args.size() != 1) {
    return absl::NotFoundError(absl::StrCat(
        "ucl::Reinterpret<FromType, ToType>(value) requires one argument, but ",
        args.size(), " was passed"));
  }
  int from_vector_size = 0;
  DataType from_type;
  ABSL_RETURN_IF_ERROR(
      DataTypeFromTemplateArg(template_args[0], &from_type, &from_vector_size));
  int to_vector_size = 0;
  DataType to_type;
  ABSL_RETURN_IF_ERROR(
      DataTypeFromTemplateArg(template_args[1], &to_type, &to_vector_size));
  if (gpu_info.IsApiOpenCl()) {
    *result =
        "as_" + ToCLDataType(to_type, to_vector_size) + "(" + args[0] + ")";
    return absl::OkStatus();
  } else if (gpu_info.IsApiMetal()) {
    *result = "as_type<" + ToMetalDataType(to_type, to_vector_size) + ">(" +
              args[0] + ")";
    return absl::OkStatus();
  } else if (gpu_info.IsGlsl()) {
    if (from_type == DataType::UINT32 && from_vector_size == 2 &&
        to_type == DataType::FLOAT16 && to_vector_size == 4) {
      *result = "ReinterpretUvec2ToHalf4(" + args[0] + ")";
      return absl::OkStatus();
    }
    if (to_type == DataType::UINT32 && to_vector_size == 2 &&
        from_type == DataType::FLOAT16 && from_vector_size == 4) {
      *result = "ReinterpretHalf4ToUvec2(" + args[0] + ")";
      return absl::OkStatus();
    }
    if (from_vector_size != to_vector_size && from_vector_size != 1 &&
        to_vector_size != 1) {
      return absl::InvalidArgumentError(
          "Unsupported ucl::Reinterpret case for GLSL");
    }
    const std::string from_glsl_type =
        ToGlslShaderDataType(from_type, from_vector_size, false, false);
    std::string to_glsl_type =
        ToGlslShaderDataType(to_type, to_vector_size, false, false);
    if (from_glsl_type == "int" && to_glsl_type == "uint") {
      *result = "uint(" + args[0] + ")";
      return absl::OkStatus();
    } else if (from_glsl_type == "uint" && to_glsl_type == "int") {
      *result = "int(" + args[0] + ")";
      return absl::OkStatus();
    }
    to_glsl_type[0] = toupper(to_glsl_type[0]);
    *result = from_glsl_type + "BitsTo" + to_glsl_type + "(" + args[0] + ")";
    return absl::OkStatus();
  } else if (gpu_info.IsApiWebGpu()) {
    const bool is_src_char4 =
        from_type == DataType::INT8 && from_vector_size == 4;
    const bool is_src_uchar4 =
        from_type == DataType::UINT8 && from_vector_size == 4;
    if (is_src_char4 || is_src_uchar4) {
      std::string uint_val = is_src_char4 ? "pack4xI8" : "pack4xU8";
      uint_val += "(" + args[0] + ")";
      if (to_type == DataType::UINT32 && to_vector_size == 1) {
        *result = uint_val;
        return absl::OkStatus();
      } else if (SizeOf(to_type) == 4 && to_vector_size == 1) {
        *result = "bitcast<" +
                  ToWebGpuType(to_type, to_vector_size,
                               gpu_info.webgpu_info.supports_fp16) +
                  ">(" + uint_val + ")";
        return absl::OkStatus();
      } else {
        return absl::InvalidArgumentError(
            "Unsupported ucl::Reinterpret case in WebGPU.");
      }
    }
    const bool is_dst_char4 = to_type == DataType::INT8 && to_vector_size == 4;
    const bool is_dst_uchar4 =
        to_type == DataType::UINT8 && to_vector_size == 4;
    if (is_dst_char4 || is_dst_uchar4) {
      std::string unpack_fcn = is_src_char4 ? "unpack4xI8" : "unpack4xU8";
      if (from_type == DataType::UINT32 && from_vector_size == 1) {
        *result = unpack_fcn + "(" + args[0] + ")";
        return absl::OkStatus();
      } else if (SizeOf(from_type) == 4 && from_vector_size == 1) {
        *result = unpack_fcn + "(bitcast<" +
                  ToWebGpuType(DataType::UINT32, from_vector_size,
                               gpu_info.webgpu_info.supports_fp16) +
                  ">(" + args[0] + "))";
        return absl::OkStatus();
      } else {
        return absl::InvalidArgumentError(
            "Unsupported ucl::Reinterpret case in WebGPU.");
      }
    }
    *result = "bitcast<" +
              ToWebGpuType(to_type, to_vector_size,
                           gpu_info.webgpu_info.supports_fp16) +
              ">(" + args[0] + ")";
    return absl::OkStatus();
  } else {
    return absl::UnimplementedError(
        "No support of this API for Reinterpret().");
  }
}

absl::Status PerformGetGlobalIdSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& template_args,
    const std::vector<std::string>& args, std::string* result) {
  if (template_args.size() != 1 || !args.empty()) {
    return absl::NotFoundError(
        "GetGlobalId must have only one template argument. GetGlobalId<0>()");
  }
  int id_dimension = -1;
  if (template_args[0] == "0") {
    id_dimension = 0;
  } else if (template_args[0] == "1") {
    id_dimension = 1;
  } else if (template_args[0] == "2") {
    id_dimension = 2;
  }
  if (id_dimension == -1) {
    return absl::InvalidArgumentError(
        "Template argument for GetGlobalId must be 0, 1 or 2");
  }
  const std::string vec_coords[3] = {"x", "y", "z"};
  if (gpu_info.IsApiOpenCl()) {
    *result = "get_global_id(" + std::to_string(id_dimension) + ")";
    return absl::OkStatus();
  } else if (gpu_info.IsApiMetal()) {
    *result = "static_cast<int>(reserved_gid." + vec_coords[id_dimension] + ")";
    return absl::OkStatus();
  } else if (gpu_info.IsGlsl()) {
    *result = "int(gl_GlobalInvocationID." + vec_coords[id_dimension] + ")";
    if (gpu_info.IsAdreno() &&
        gpu_info.adreno_info.IsGlDriverMajor615Minor88_97()) {
      *result = "(" + *result + " * args.one_gl_reserved)";
    }
    return absl::OkStatus();
  } else if (gpu_info.IsApiWebGpu()) {
    *result = "i32(reserved_gid." + vec_coords[id_dimension] + ")";
    return absl::OkStatus();
  } else if (gpu_info.IsApiCuda()) {
    const std::string coord = vec_coords[id_dimension];
    *result = "static_cast<int>(blockDim." + coord + " * blockIdx." + coord +
              " + threadIdx." + coord + ")";
    return absl::OkStatus();
  } else {
    return absl::UnimplementedError("No support of GetGlobalId in this API.");
  }
}

absl::Status PerformGetLocalIdSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& template_args,
    const std::vector<std::string>& args, std::string* result) {
  if (template_args.size() != 1 || !args.empty()) {
    return absl::NotFoundError(
        "GetLocalId must have only one template argument. GetLocalId<0>()");
  }
  int id_dimension = -1;
  if (template_args[0] == "0") {
    id_dimension = 0;
  } else if (template_args[0] == "1") {
    id_dimension = 1;
  } else if (template_args[0] == "2") {
    id_dimension = 2;
  }
  if (id_dimension == -1) {
    return absl::InvalidArgumentError(
        "Template argument for GetLocalId must be 0, 1 or 2");
  }
  const std::string vec_coords[3] = {"x", "y", "z"};
  if (gpu_info.IsApiOpenCl()) {
    *result = "get_local_id(" + std::to_string(id_dimension) + ")";
    return absl::OkStatus();
  } else if (gpu_info.IsApiMetal()) {
    *result = "static_cast<int>(reserved_lid." + vec_coords[id_dimension] + ")";
    return absl::OkStatus();
  } else if (gpu_info.IsGlsl()) {
    *result = "int(gl_LocalInvocationID." + vec_coords[id_dimension] + ")";
    return absl::OkStatus();
  } else if (gpu_info.IsApiWebGpu()) {
    *result = "i32(reserved_lid." + vec_coords[id_dimension] + ")";
    return absl::OkStatus();
  } else if (gpu_info.IsApiCuda()) {
    *result = "static_cast<int>(threadIdx." + vec_coords[id_dimension] + ")";
    return absl::OkStatus();
  } else {
    return absl::UnimplementedError("No support of GetLocalId in this API.");
  }
}

absl::Status PerformGetGroupIdSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& template_args,
    const std::vector<std::string>& args, std::string* result) {
  if (template_args.size() != 1 || !args.empty()) {
    return absl::NotFoundError(
        "GetGroupId must have only one template argument. GetGroupId<0>()");
  }
  int id_dimension = -1;
  if (template_args[0] == "0") {
    id_dimension = 0;
  } else if (template_args[0] == "1") {
    id_dimension = 1;
  } else if (template_args[0] == "2") {
    id_dimension = 2;
  }
  if (id_dimension == -1) {
    return absl::InvalidArgumentError(
        "Template argument for GetGroupId must be 0, 1 or 2");
  }
  const std::string vec_coords[3] = {"x", "y", "z"};
  if (gpu_info.IsApiOpenCl()) {
    *result = "get_group_id(" + std::to_string(id_dimension) + ")";
    return absl::OkStatus();
  } else if (gpu_info.IsApiMetal()) {
    *result =
        "static_cast<int>(reserved_group_id." + vec_coords[id_dimension] + ")";
    return absl::OkStatus();
  } else if (gpu_info.IsGlsl()) {
    *result = "int(gl_WorkGroupID." + vec_coords[id_dimension] + ")";
    if (gpu_info.IsAdreno() &&
        gpu_info.adreno_info.IsGlDriverMajor615Minor88_97()) {
      *result = "(" + *result + " * args.one_gl_reserved)";
    }
    return absl::OkStatus();
  } else if (gpu_info.IsApiWebGpu()) {
    *result = "i32(reserved_group_id." + vec_coords[id_dimension] + ")";
    return absl::OkStatus();
  } else if (gpu_info.IsApiCuda()) {
    *result = "static_cast<int>(blockIdx." + vec_coords[id_dimension] + ")";
    return absl::OkStatus();
  } else {
    return absl::UnimplementedError("No support of GetGroupId in this API.");
  }
}

absl::Status PerformGetGroupSizeSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& template_args,
    const std::vector<std::string>& args, std::string* result) {
  if (template_args.size() != 1 || !args.empty()) {
    return absl::NotFoundError(
        "GetGroupSize must have only one template argument. GetGroupSize<0>()");
  }
  int id_dimension = -1;
  if (template_args[0] == "0") {
    id_dimension = 0;
  } else if (template_args[0] == "1") {
    id_dimension = 1;
  } else if (template_args[0] == "2") {
    id_dimension = 2;
  }
  if (id_dimension == -1) {
    return absl::InvalidArgumentError(
        "Template argument for GetGroupSize must be 0, 1 or 2");
  }
  const std::string vec_coords[3] = {"x", "y", "z"};
  if (gpu_info.IsApiOpenCl()) {
    *result = "get_local_size(" + std::to_string(id_dimension) + ")";
    return absl::OkStatus();
  } else if (gpu_info.IsApiMetal()) {
    *result = "static_cast<int>(reserved_group_size." +
              vec_coords[id_dimension] + ")";
    return absl::OkStatus();
  } else if (gpu_info.IsGlsl()) {
    *result = "int(gl_WorkGroupSize." + vec_coords[id_dimension] + ")";
    return absl::OkStatus();
  } else if (gpu_info.IsApiWebGpu()) {
    *result = "reserved_group_size." + vec_coords[id_dimension];
    return absl::OkStatus();
  } else if (gpu_info.IsApiCuda()) {
    *result = "static_cast<int>(blockDim." + vec_coords[id_dimension] + ")";
    return absl::OkStatus();
  } else {
    return absl::UnimplementedError("No support of GetGroupSize in this API.");
  }
}

absl::Status PerformGetSubGroupLocalIdSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& template_args,
    const std::vector<std::string>& args, std::string* result) {
  if (!template_args.empty() || !args.empty()) {
    return absl::NotFoundError("GetSubGroupLocalId do not support arguments");
  }
  if (gpu_info.IsApiOpenCl()) {
    *result = "get_sub_group_local_id()";
    return absl::OkStatus();
  } else if (gpu_info.IsApiMetal()) {
    *result = "static_cast<int>(reserved_simd_local_id)";
    return absl::OkStatus();
  } else if (gpu_info.IsGlsl()) {
    *result = "int(gl_SubgroupInvocationID)";
    return absl::OkStatus();
  } else if (gpu_info.IsApiWebGpu()) {
    *result = "i32(reserved_subgroup_local_id)";
    return absl::OkStatus();
  } else {
    return absl::UnimplementedError(
        "No support of GetSubGroupLocalId in this API.");
  }
}

absl::Status PerformGetSubGroupIdSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& template_args,
    const std::vector<std::string>& args, std::string* result) {
  if (!template_args.empty() || !args.empty()) {
    return absl::NotFoundError("GetSubGroupId do not support arguments");
  }
  if (gpu_info.IsApiOpenCl()) {
    *result = "get_sub_group_id()";
    return absl::OkStatus();
  } else if (gpu_info.IsApiMetal()) {
    *result = "static_cast<int>(reserved_simd_id)";
    return absl::OkStatus();
  } else if (gpu_info.IsGlsl()) {
    *result = "int(gl_SubgroupID)";
    return absl::OkStatus();
  } else if (gpu_info.IsApiWebGpu()) {
    *result = "i32(reserved_subgroup_id)";
    return absl::OkStatus();
  } else {
    return absl::UnimplementedError("No support of GetSubGroupId in this API.");
  }
}

absl::Status PerformGetSubGroupSizeSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& template_args,
    const std::vector<std::string>& args, std::string* result) {
  if (!template_args.empty() || !args.empty()) {
    return absl::NotFoundError("GetSubGroupSize does not support arguments");
  }
  if (gpu_info.IsApiMetal()) {
    *result = "reserved_simd_size";
    return absl::OkStatus();
  } else if (gpu_info.IsApiWebGpu()) {
    *result = "reserved_subgroup_size";
    return absl::OkStatus();
  } else {
    return absl::UnimplementedError(
        "No support of GetSubGroupSize in this API.");
  }
}

absl::Status PerformSyncThreadsSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& template_args,
    const std::vector<std::string>& args, std::string* result) {
  if (template_args.size() != 2 || !args.empty()) {
    return absl::NotFoundError(
        "SyncThreads must have 2 template argument. SyncThreads<scope, "
        "mem_type>() "
        "where scope is WorkGroup or SubGroup and mem_type Local or None");
  }
  if (template_args[0] == "WorkGroup") {
    if (gpu_info.IsApiOpenCl()) {
      *result = "barrier(CLK_LOCAL_MEM_FENCE)";
      return absl::OkStatus();
    } else if (gpu_info.IsApiMetal()) {
      const std::string mem_flags = template_args[1] == "Local"
                                        ? "mem_flags::mem_threadgroup"
                                        : "mem_flags::mem_none";
      *result = "threadgroup_barrier(" + mem_flags + ")";
      return absl::OkStatus();
    } else if (gpu_info.IsGlsl()) {
      if (template_args[1] == "Local") {
        *result = "memoryBarrierShared(); ";
      }
      *result += "barrier()";
      return absl::OkStatus();
    } else if (gpu_info.IsApiWebGpu()) {
      *result = "workgroupBarrier()";
      return absl::OkStatus();
    } else if (gpu_info.IsApiCuda()) {
      *result = "__syncthreads()";
      return absl::OkStatus();
    } else {
      return absl::UnimplementedError(
          "No support of SyncThreads<WorkGroup> in this API.");
    }
  } else if (template_args[0] == "SubGroup") {
    if (gpu_info.IsApiOpenCl()) {
      *result = "sub_group_barrier(CLK_LOCAL_MEM_FENCE)";
      return absl::OkStatus();
    } else if (gpu_info.IsApiMetal()) {
      // simdgroup_barrier is supported since Metal shading language version 2.0
      const std::string simdgroup_barrier =
          gpu_info.metal_info.IsMslVersionEqualOrHigher(2, 0)
              ? "simdgroup_barrier"
              : "threadgroup_barrier";
      const std::string mem_flags = template_args[1] == "Local"
                                        ? "mem_flags::mem_threadgroup"
                                        : "mem_flags::mem_none";
      *result = simdgroup_barrier + "(" + mem_flags + ")";
      return absl::OkStatus();
    } else if (gpu_info.IsGlsl()) {
      if (template_args[1] == "Local") {
        *result = "subgroupMemoryBarrierShared(); ";
      }
      *result += "subgroupBarrier()";
      return absl::OkStatus();
    } else {
      return absl::UnimplementedError(
          "No support of SyncThreads<SubGroup> in this API.");
    }
  }
  return absl::UnimplementedError(
      "No support of SyncThreads with this parameters.");
}

absl::Status PerformSubGroupBroadcastSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& template_args,
    const std::vector<std::string>& args, std::string* result) {
  if (!template_args.empty() || args.size() != 2) {
    return absl::NotFoundError("SubGroupBroadcast must have 2 arguments");
  }
  if (gpu_info.IsApiOpenCl()) {
    *result = "sub_group_broadcast(" + args[0] + ", " + args[1] + ")";
    return absl::OkStatus();
  } else if (gpu_info.IsApiMetal()) {
    *result = "simd_broadcast(" + args[0] + ", " + args[1] + ")";
    return absl::OkStatus();
  } else if (gpu_info.IsGlsl()) {
    *result = "subgroupBroadcast(" + args[0] + ", " + args[1] + ")";
    return absl::OkStatus();
  } else if (gpu_info.IsApiWebGpu()) {
    *result = "subgroupBroadcast(" + args[0] + ", " + args[1] + ")";
    return absl::OkStatus();
  } else {
    return absl::UnimplementedError(
        "No support of SubGroupBroadcast in this API.");
  }
}

absl::Status PerformU32x4ToU8x16AsVec4x4(
    const GpuInfo& gpu_info, const std::vector<std::string>& template_args,
    const std::vector<std::string>& args, std::string* result) {
  if (template_args.size() != 1 || args.size() != 5) {
    return absl::NotFoundError(
        "U32x4ToU8x16AsVec4x4 must have 1 template argument and 5 "
        "arguments");
  }
  if (gpu_info.IsMaleoon()) {
    *result = R"(
  $2 = ucl::Convert<$04>(ucl::Reinterpret<uint, uchar4>($1.x));
  $3 = ucl::Convert<$04>(ucl::Reinterpret<uint, uchar4>($1.y));
  $4 = ucl::Convert<$04>(ucl::Reinterpret<uint, uchar4>($1.z));
  $5 = ucl::Convert<$04>(ucl::Reinterpret<uint, uchar4>($1.w));
)";
  } else {
    *result = R"(
  $2.x = ucl::Convert<$0>(($1.x) & 255u);
  $2.y = ucl::Convert<$0>(($1.x >>  8u) & 255u);
  $2.z = ucl::Convert<$0>(($1.x >> 16u) & 255u);
  $2.w = ucl::Convert<$0>(($1.x >> 24u) & 255u);
  $3.x = ucl::Convert<$0>(($1.y) & 255u);
  $3.y = ucl::Convert<$0>(($1.y >>  8u) & 255u);
  $3.z = ucl::Convert<$0>(($1.y >> 16u) & 255u);
  $3.w = ucl::Convert<$0>(($1.y >> 24u) & 255u);
  $4.x = ucl::Convert<$0>(($1.z) & 255u);
  $4.y = ucl::Convert<$0>(($1.z >>  8u) & 255u);
  $4.z = ucl::Convert<$0>(($1.z >> 16u) & 255u);
  $4.w = ucl::Convert<$0>(($1.z >> 24u) & 255u);
  $5.x = ucl::Convert<$0>(($1.w) & 255u);
  $5.y = ucl::Convert<$0>(($1.w >>  8u) & 255u);
  $5.z = ucl::Convert<$0>(($1.w >> 16u) & 255u);
  $5.w = ucl::Convert<$0>(($1.w >> 24u) & 255u);
)";
  }
  *result = absl::Substitute(*result, template_args[0], args[0], args[1],
                             args[2], args[3], args[4]);
  return absl::OkStatus();
}

absl::Status PerformInt32x4ToInt8x16AsVec4x4(
    const std::vector<std::string>& template_args,
    const std::vector<std::string>& args, std::string* result) {
  if (template_args.size() != 1 || args.size() != 5) {
    return absl::NotFoundError(
        "Int32x4ToInt8x16AsVec4x4 must have 1 template argument and 5 "
        "arguments");
  }
  *result = absl::Substitute(R"(
  $2.x = ucl::Convert<$0>(ucl::Convert<char>(($1.x) & 255));
  $2.y = ucl::Convert<$0>(ucl::Convert<char>(($1.x >>  8u) & 255));
  $2.z = ucl::Convert<$0>(ucl::Convert<char>(($1.x >> 16u) & 255));
  $2.w = ucl::Convert<$0>(ucl::Convert<char>(($1.x >> 24u) & 255));
  $3.x = ucl::Convert<$0>(ucl::Convert<char>(($1.y) & 255));
  $3.y = ucl::Convert<$0>(ucl::Convert<char>(($1.y >>  8u) & 255));
  $3.z = ucl::Convert<$0>(ucl::Convert<char>(($1.y >> 16u) & 255));
  $3.w = ucl::Convert<$0>(ucl::Convert<char>(($1.y >> 24u) & 255));
  $4.x = ucl::Convert<$0>(ucl::Convert<char>(($1.z) & 255));
  $4.y = ucl::Convert<$0>(ucl::Convert<char>(($1.z >>  8u) & 255));
  $4.z = ucl::Convert<$0>(ucl::Convert<char>(($1.z >> 16u) & 255));
  $4.w = ucl::Convert<$0>(ucl::Convert<char>(($1.z >> 24u) & 255));
  $5.x = ucl::Convert<$0>(ucl::Convert<char>(($1.w) & 255));
  $5.y = ucl::Convert<$0>(ucl::Convert<char>(($1.w >>  8u) & 255));
  $5.z = ucl::Convert<$0>(ucl::Convert<char>(($1.w >> 16u) & 255));
  $5.w = ucl::Convert<$0>(ucl::Convert<char>(($1.w >> 24u) & 255));
)",
                             template_args[0], args[0], args[1], args[2],
                             args[3], args[4]);
  return absl::OkStatus();
}

absl::Status PerformU32x2ToU4x16AsVec4x4(
    const GpuInfo& gpu_info, const std::vector<std::string>& template_args,
    const std::vector<std::string>& args, std::string* result) {
  if (template_args.size() != 1 || args.size() != 5) {
    return absl::NotFoundError(
        "U32x2ToU4x16AsVec4x4 must have 1 template argument and 5 "
        "arguments");
  }
  int type_size = 0;
  DataType type;
  ABSL_RETURN_IF_ERROR(
      DataTypeFromTemplateArg(template_args[0], &type, &type_size));
  const bool use_float_math_for_unpacking =
      type == DataType::FLOAT16 && gpu_info.IsApple();
  if (use_float_math_for_unpacking) {
    *result = "  {\n";
    const bool use_reinterpret =
        gpu_info.IsApiOpenCl() || gpu_info.IsApiMetal();
    if (use_reinterpret) {
      *result += R"(
  $04 wt0 = ucl::Convert<$04>(ucl::Reinterpret<uint, uchar4>($1.x)) * ucl::Init<$0>(0.0625f);
  $04 wt1 = ucl::Convert<$04>(ucl::Reinterpret<uint, uchar4>($1.y)) * ucl::Init<$0>(0.0625f);
)";
    } else {
      *result += R"(
  $04 wt0;
  $04 wt1;
  wt0.x = ucl::Convert<$0>(($1.x) & 255u) * ucl::Init<$0>(0.0625f);
  wt0.y = ucl::Convert<$0>(($1.x >>  8u) & 255u) * ucl::Init<$0>(0.0625f);
  wt0.z = ucl::Convert<$0>(($1.x >> 16u) & 255u) * ucl::Init<$0>(0.0625f);
  wt0.w = ucl::Convert<$0>(($1.x >> 24u) & 255u) * ucl::Init<$0>(0.0625f);
  wt1.x = ucl::Convert<$0>(($1.y) & 255u) * ucl::Init<$0>(0.0625f);
  wt1.y = ucl::Convert<$0>(($1.y >>  8u) & 255u) * ucl::Init<$0>(0.0625f);
  wt1.z = ucl::Convert<$0>(($1.y >> 16u) & 255u) * ucl::Init<$0>(0.0625f);
  wt1.w = ucl::Convert<$0>(($1.y >> 24u) & 255u) * ucl::Init<$0>(0.0625f);
)";
    }
    *result += R"(
  $2.y = floor(wt0.x);
  $2.w = floor(wt0.y);
  $2.x = (wt0.x - $2.y) * ucl::Init<$0>(16.0f);
  $2.z = (wt0.y - $2.w) * ucl::Init<$0>(16.0f);
  $3.y = floor(wt0.z);
  $3.w = floor(wt0.w);
  $3.x = (wt0.z - $3.y) * ucl::Init<$0>(16.0f);
  $3.z = (wt0.w - $3.w) * ucl::Init<$0>(16.0f);
  $4.y = floor(wt1.x);
  $4.w = floor(wt1.y);
  $4.x = (wt1.x - $4.y) * ucl::Init<$0>(16.0f);
  $4.z = (wt1.y - $4.w) * ucl::Init<$0>(16.0f);
  $5.y = floor(wt1.z);
  $5.w = floor(wt1.w);
  $5.x = (wt1.z - $5.y) * ucl::Init<$0>(16.0f);
  $5.z = (wt1.w - $5.w) * ucl::Init<$0>(16.0f);
  }
)";
  } else if (gpu_info.IsMaleoon()) {
    *result = R"(
  $2 = ucl::Convert<$04>(ucl::Init<ushort4>($1.x, $1.x >> 4u, $1.x >> 8u, $1.x >> 12u) & ucl::Init<ushort4>(15u));
  $3 = ucl::Convert<$04>(ucl::Init<ushort4>($1.x >> 16u, $1.x >> 20u, $1.x >> 24u, $1.x >> 28u) & ucl::Init<ushort4>(15u));
  $4 = ucl::Convert<$04>(ucl::Init<ushort4>($1.y, $1.y >> 4u, $1.y >> 8u, $1.y >> 12u) & ucl::Init<ushort4>(15u));
  $5 = ucl::Convert<$04>(ucl::Init<ushort4>($1.y >> 16u, $1.y >> 20u, $1.y >> 24u, $1.y >> 28u) & ucl::Init<ushort4>(15u));
)";
  } else if (/* DISABLES CODE */ (false) && type == DataType::FLOAT16) {
    // experimental
    *result = R"(
  {
    //w13w12w11w10w03w02w01w00
    //w13w03w11w01w12w02w10w00

    uint t = ($1.x ^ ($1.x >> 12u)) & 0x0000F0F0u;
    $1.x ^= t ^ (t << 12u);

    uint base = 0x64006400u;
    uint mask = 0x000F000Fu;

    uint w01w00 = base | ($1.x & mask);
    uint w11w10 = base | (($1.x >> 4u)  & mask);
    uint w03w02 = base | (($1.x >> 8u)  & mask);
    uint w13w12 = base | (($1.x >> 12u) & mask);
    $2.xy = ucl::Reinterpret<uint, half2>(w01w00) - ucl::Init<half2>(1024.0h);
    $2.zw = ucl::Reinterpret<uint, half2>(w03w02) - ucl::Init<half2>(1024.0h);
    $3.xy = ucl::Reinterpret<uint, half2>(w11w10) - ucl::Init<half2>(1024.0h);
    $3.zw = ucl::Reinterpret<uint, half2>(w13w12) - ucl::Init<half2>(1024.0h);

    t = ($1.y ^ ($1.y >> 12u)) & 0x0000F0F0u;
    $1.y ^= t ^ (t << 12u);

    w01w00 = base | ($1.y & mask);
    w11w10 = base | (($1.y >> 4u)  & mask);
    w03w02 = base | (($1.y >> 8u)  & mask);
    w13w12 = base | (($1.y >> 12u) & mask);
    $4.xy = ucl::Reinterpret<uint, half2>(w01w00) - ucl::Init<half2>(1024.0h);
    $4.zw = ucl::Reinterpret<uint, half2>(w03w02) - ucl::Init<half2>(1024.0h);
    $5.xy = ucl::Reinterpret<uint, half2>(w11w10) - ucl::Init<half2>(1024.0h);
    $5.zw = ucl::Reinterpret<uint, half2>(w13w12) - ucl::Init<half2>(1024.0h);
  }
)";
  } else {
    *result = R"(
  $2.x = ucl::Convert<$0>(($1.x) & 15u);
  $2.y = ucl::Convert<$0>(($1.x >>  4u) & 15u);
  $2.z = ucl::Convert<$0>(($1.x >>  8u) & 15u);
  $2.w = ucl::Convert<$0>(($1.x >> 12u) & 15u);
  $3.x = ucl::Convert<$0>(($1.x >> 16u) & 15u);
  $3.y = ucl::Convert<$0>(($1.x >> 20u) & 15u);
  $3.z = ucl::Convert<$0>(($1.x >> 24u) & 15u);
  $3.w = ucl::Convert<$0>(($1.x >> 28u) & 15u);
  $4.x = ucl::Convert<$0>(($1.y) & 15u);
  $4.y = ucl::Convert<$0>(($1.y >>  4u) & 15u);
  $4.z = ucl::Convert<$0>(($1.y >>  8u) & 15u);
  $4.w = ucl::Convert<$0>(($1.y >> 12u) & 15u);
  $5.x = ucl::Convert<$0>(($1.y >> 16u) & 15u);
  $5.y = ucl::Convert<$0>(($1.y >> 20u) & 15u);
  $5.z = ucl::Convert<$0>(($1.y >> 24u) & 15u);
  $5.w = ucl::Convert<$0>(($1.y >> 28u) & 15u);
)";
  }
  *result = absl::Substitute(*result, template_args[0], args[0], args[1],
                             args[2], args[3], args[4]);
  return absl::OkStatus();
}

absl::Status PerformU32x1ToU2x16AsVec4x4(
    const GpuInfo& gpu_info, const std::vector<std::string>& template_args,
    const std::vector<std::string>& args, std::string* result) {
  if (template_args.size() != 1 || args.size() != 5) {
    return absl::NotFoundError(
        "U32x1ToU2x16AsVec4x4 must have 1 template argument and 5 "
        "arguments");
  }
  int type_size = 0;
  DataType type;
  ABSL_RETURN_IF_ERROR(
      DataTypeFromTemplateArg(template_args[0], &type, &type_size));
  const bool use_float_math_for_unpacking =
      type == DataType::FLOAT16 && gpu_info.IsApple();
  const bool use_reinterpret_unpacking =
      gpu_info.IsApiOpenCl() && gpu_info.IsMali();
  if (use_float_math_for_unpacking) {
    *result = "  {\n";
    const bool use_reinterpret =
        gpu_info.IsApiOpenCl() || gpu_info.IsApiMetal();
    if (use_reinterpret) {
      *result += R"(
  $04 wt0 = ucl::Convert<$04>(ucl::Reinterpret<uint, uchar4>($1)) * ucl::Init<$0>(0.25f);
)";
    } else {
      *result += R"(
  $04 wt0;
  wt0.x = ucl::Convert<$0>(($1) & 255u) * ucl::Init<$0>(0.25f);
  wt0.y = ucl::Convert<$0>(($1 >>  8u) & 255u) * ucl::Init<$0>(0.25f);
  wt0.z = ucl::Convert<$0>(($1 >> 16u) & 255u) * ucl::Init<$0>(0.25f);
  wt0.w = ucl::Convert<$0>(($1 >> 24u) & 255u) * ucl::Init<$0>(0.25f);
)";
    }
    *result += R"(
  $04 wt1 = floor(wt0);
  $2.x = (wt0.x - wt1.x) * ucl::Init<$0>(4.0f);
  $3.x = (wt0.y - wt1.y) * ucl::Init<$0>(4.0f);
  $4.x = (wt0.z - wt1.z) * ucl::Init<$0>(4.0f);
  $5.x = (wt0.w - wt1.w) * ucl::Init<$0>(4.0f);
  wt0 = wt1 * ucl::Init<$0>(0.25f);
  wt1 = floor(wt0);
  $2.y = (wt0.x - wt1.x) * ucl::Init<$0>(4.0f);
  $3.y = (wt0.y - wt1.y) * ucl::Init<$0>(4.0f);
  $4.y = (wt0.z - wt1.z) * ucl::Init<$0>(4.0f);
  $5.y = (wt0.w - wt1.w) * ucl::Init<$0>(4.0f);
  wt0 = wt1 * ucl::Init<$0>(0.25f);
  wt1 = floor(wt0);
  $2.z = (wt0.x - wt1.x) * ucl::Init<$0>(4.0f);
  $3.z = (wt0.y - wt1.y) * ucl::Init<$0>(4.0f);
  $4.z = (wt0.z - wt1.z) * ucl::Init<$0>(4.0f);
  $5.z = (wt0.w - wt1.w) * ucl::Init<$0>(4.0f);
  $2.w = wt1.x;
  $3.w = wt1.y;
  $4.w = wt1.z;
  $5.w = wt1.w;
  }
)";
  } else if (/* DISABLES CODE */ (false) && type == DataType::FLOAT16) {
    // experimental
    *result = R"(
  {
    // 15  14  13  12  11  10   9   8   7   6   5   4   3   2   1   0
    //w33 w32 w31 w30 w23 w22 w21 w20 w13 w12 w11 w10 w03 w02 w01 w00
    //w33 w13 w31 w11 w23 w03 w21 w01 w32 w12 w30 w10 w22 w02 w20 w00
    uint t = ($1 ^ ($1 >> 14u)) & 0x0000CCCCu;
    $1 ^= t ^ (t << 14u);

    uint base = 0x64006400u;
    uint mask = 0x00030003u;

    uint w01w00 = base | ($1 & mask);
    uint w21w20 = base | (($1 >> 2u) & mask);
    uint w03w02 = base | (($1 >> 4u) & mask);
    uint w23w22 = base | (($1 >> 6u) & mask);
    uint w11w10 = base | (($1 >> 8u) & mask);
    uint w31w30 = base | (($1 >> 10u) & mask);
    uint w13w12 = base | (($1 >> 12u) & mask);
    uint w33w32 = base | (($1 >> 14u) & mask);
    $2.xy = ucl::Reinterpret<uint, half2>(w01w00) - ucl::Init<half2>(1024.0h);
    $2.zw = ucl::Reinterpret<uint, half2>(w03w02) - ucl::Init<half2>(1024.0h);
    $3.xy = ucl::Reinterpret<uint, half2>(w11w10) - ucl::Init<half2>(1024.0h);
    $3.zw = ucl::Reinterpret<uint, half2>(w13w12) - ucl::Init<half2>(1024.0h);
    $4.xy = ucl::Reinterpret<uint, half2>(w21w20) - ucl::Init<half2>(1024.0h);
    $4.zw = ucl::Reinterpret<uint, half2>(w23w22) - ucl::Init<half2>(1024.0h);
    $5.xy = ucl::Reinterpret<uint, half2>(w31w30) - ucl::Init<half2>(1024.0h);
    $5.zw = ucl::Reinterpret<uint, half2>(w33w32) - ucl::Init<half2>(1024.0h);
  }
)";
  } else if (use_reinterpret_unpacking) {
    *result = R"(
  {
    uchar4 u0 = ucl::Reinterpret<uint, uchar4>($1 & 50529027u);
    uchar4 u1 = ucl::Reinterpret<uint, uchar4>(($1 >> 2u) & 50529027u);
    uchar4 u2 = ucl::Reinterpret<uint, uchar4>(($1 >> 4u) & 50529027u);
    uchar4 u3 = ucl::Reinterpret<uint, uchar4>(($1 >> 6u) & 50529027u);
    $2.x = ucl::Convert<$0>(u0.x);
    $2.y = ucl::Convert<$0>(u1.x);
    $2.z = ucl::Convert<$0>(u2.x);
    $2.w = ucl::Convert<$0>(u3.x);
    $3.x = ucl::Convert<$0>(u0.y);
    $3.y = ucl::Convert<$0>(u1.y);
    $3.z = ucl::Convert<$0>(u2.y);
    $3.w = ucl::Convert<$0>(u3.y);
    $4.x = ucl::Convert<$0>(u0.z);
    $4.y = ucl::Convert<$0>(u1.z);
    $4.z = ucl::Convert<$0>(u2.z);
    $4.w = ucl::Convert<$0>(u3.z);
    $5.x = ucl::Convert<$0>(u0.w);
    $5.y = ucl::Convert<$0>(u1.w);
    $5.z = ucl::Convert<$0>(u2.w);
    $5.w = ucl::Convert<$0>(u3.w);
  }
)";
  } else if (gpu_info.IsMaleoon()) {
    *result = R"(
  $2 = ucl::Convert<$04>(ucl::Init<ushort4>($1, $1 >> 2u, $1 >> 4u, $1 >> 6u) & ucl::Init<ushort4>(3u));
  $3 = ucl::Convert<$04>(ucl::Init<ushort4>($1 >> 8u, $1 >> 10u, $1 >> 12u, $1 >> 14u) & ucl::Init<ushort4>(3u));
  $4 = ucl::Convert<$04>(ucl::Init<ushort4>($1 >> 16u, $1 >> 18u, $1 >> 20u, $1 >> 22u) & ucl::Init<ushort4>(3u));
  $5 = ucl::Convert<$04>(ucl::Init<ushort4>($1 >> 24u, $1 >> 26u, $1 >> 28u, $1 >> 30u) & ucl::Init<ushort4>(3u));
)";
  } else {
    *result = R"(
  $2.x = ucl::Convert<$0>(($1) & 3u);
  $2.y = ucl::Convert<$0>(($1 >>  2u) & 3u);
  $2.z = ucl::Convert<$0>(($1 >>  4u) & 3u);
  $2.w = ucl::Convert<$0>(($1 >>  6u) & 3u);
  $3.x = ucl::Convert<$0>(($1 >>  8u) & 3u);
  $3.y = ucl::Convert<$0>(($1 >> 10u) & 3u);
  $3.z = ucl::Convert<$0>(($1 >> 12u) & 3u);
  $3.w = ucl::Convert<$0>(($1 >> 14u) & 3u);
  $4.x = ucl::Convert<$0>(($1 >> 16u) & 3u);
  $4.y = ucl::Convert<$0>(($1 >> 18u) & 3u);
  $4.z = ucl::Convert<$0>(($1 >> 20u) & 3u);
  $4.w = ucl::Convert<$0>(($1 >> 22u) & 3u);
  $5.x = ucl::Convert<$0>(($1 >> 24u) & 3u);
  $5.y = ucl::Convert<$0>(($1 >> 26u) & 3u);
  $5.z = ucl::Convert<$0>(($1 >> 28u) & 3u);
  $5.w = ucl::Convert<$0>(($1 >> 30u) & 3u);
)";
  }
  *result = absl::Substitute(*result, template_args[0], args[0], args[1],
                             args[2], args[3], args[4]);
  return absl::OkStatus();
}

absl::Status PerformU16x4ToU4x16AsVec4x4(
    const GpuInfo& gpu_info, const std::vector<std::string>& template_args,
    const std::vector<std::string>& args, std::string* result) {
  if (template_args.size() != 1 || args.size() != 5) {
    return absl::NotFoundError(
        "U16x4ToU4x16AsVec4x4 must have 1 template argument and 5 "
        "arguments");
  }
  int type_size = 0;
  DataType type;
  ABSL_RETURN_IF_ERROR(
      DataTypeFromTemplateArg(template_args[0], &type, &type_size));
  if (gpu_info.IsPowerVR() && gpu_info.IsApiOpenCl() &&
      type == DataType::FLOAT16) {
    *result = R"(
  {
  $04 wt0 = ucl::Convert<$04>(ucl::Reinterpret<ushort2, uchar4>($1.xy)) * ucl::Init<$0>(0.0625f);
  $04 wt1 = ucl::Convert<$04>(ucl::Reinterpret<ushort2, uchar4>($1.zw)) * ucl::Init<$0>(0.0625f);
  $2.y = floor(wt0.x);
  $2.w = floor(wt0.y);
  $2.x = (wt0.x - $2.y) * ucl::Init<$0>(16.0f);
  $2.z = (wt0.y - $2.w) * ucl::Init<$0>(16.0f);
  $3.y = floor(wt0.z);
  $3.w = floor(wt0.w);
  $3.x = (wt0.z - $3.y) * ucl::Init<$0>(16.0f);
  $3.z = (wt0.w - $3.w) * ucl::Init<$0>(16.0f);
  $4.y = floor(wt1.x);
  $4.w = floor(wt1.y);
  $4.x = (wt1.x - $4.y) * ucl::Init<$0>(16.0f);
  $4.z = (wt1.y - $4.w) * ucl::Init<$0>(16.0f);
  $5.y = floor(wt1.z);
  $5.w = floor(wt1.w);
  $5.x = (wt1.z - $5.y) * ucl::Init<$0>(16.0f);
  $5.z = (wt1.w - $5.w) * ucl::Init<$0>(16.0f);
  }
)";
  } else if (gpu_info.IsIntel() || gpu_info.IsPowerVR() ||
             gpu_info.IsMaleoon()) {
    // vector form
    *result = R"(
  $2 = ucl::Convert<$04>(ucl::Init<ushort4>($1.x, $1.x >> 4u, $1.x >> 8u, $1.x >> 12u) & ucl::Init<ushort4>(15u));
  $3 = ucl::Convert<$04>(ucl::Init<ushort4>($1.y, $1.y >> 4u, $1.y >> 8u, $1.y >> 12u) & ucl::Init<ushort4>(15u));
  $4 = ucl::Convert<$04>(ucl::Init<ushort4>($1.z, $1.z >> 4u, $1.z >> 8u, $1.z >> 12u) & ucl::Init<ushort4>(15u));
  $5 = ucl::Convert<$04>(ucl::Init<ushort4>($1.w, $1.w >> 4u, $1.w >> 8u, $1.w >> 12u) & ucl::Init<ushort4>(15u));
)";
  } else if (/* DISABLES CODE */ (false) && type == DataType::FLOAT16) {
    // experimental
    *result = R"(
  {
    //w13w12w11w10w03w02w01w00
    //w13w03w11w01w12w02w10w00

    uint2 WT = ucl::Reinterpret<ushort4, uint2>($1);

    uint t = (WT.x ^ (WT.x >> 12u)) & 0x0000F0F0u;
    WT.x ^= t ^ (t << 12u);

    uint base = 0x64006400u;
    uint mask = 0x000F000Fu;

    uint w01w00 = base | (WT.x & mask);
    uint w11w10 = base | ((WT.x >> 4u)  & mask);
    uint w03w02 = base | ((WT.x >> 8u)  & mask);
    uint w13w12 = base | ((WT.x >> 12u) & mask);
    $2.xy = ucl::Reinterpret<uint, half2>(w01w00) - ucl::Init<half2>(1024.0h);
    $2.zw = ucl::Reinterpret<uint, half2>(w03w02) - ucl::Init<half2>(1024.0h);
    $3.xy = ucl::Reinterpret<uint, half2>(w11w10) - ucl::Init<half2>(1024.0h);
    $3.zw = ucl::Reinterpret<uint, half2>(w13w12) - ucl::Init<half2>(1024.0h);

    t = (WT.y ^ (WT.y >> 12u)) & 0x0000F0F0u;
    WT.y ^= t ^ (t << 12u);

    w01w00 = base | (WT.y & mask);
    w11w10 = base | ((WT.y >> 4u)  & mask);
    w03w02 = base | ((WT.y >> 8u)  & mask);
    w13w12 = base | ((WT.y >> 12u) & mask);
    $4.xy = ucl::Reinterpret<uint, half2>(w01w00) - ucl::Init<half2>(1024.0h);
    $4.zw = ucl::Reinterpret<uint, half2>(w03w02) - ucl::Init<half2>(1024.0h);
    $5.xy = ucl::Reinterpret<uint, half2>(w11w10) - ucl::Init<half2>(1024.0h);
    $5.zw = ucl::Reinterpret<uint, half2>(w13w12) - ucl::Init<half2>(1024.0h);
  }
)";
  } else {
    *result = R"(
  $2.x = ucl::Convert<$0>(($1.x) & 15u);
  $2.y = ucl::Convert<$0>(($1.x >>  4u) & 15u);
  $2.z = ucl::Convert<$0>(($1.x >>  8u) & 15u);
  $2.w = ucl::Convert<$0>(($1.x >> 12u) & 15u);
  $3.x = ucl::Convert<$0>(($1.y) & 15u);
  $3.y = ucl::Convert<$0>(($1.y >> 4u) & 15u);
  $3.z = ucl::Convert<$0>(($1.y >> 8u) & 15u);
  $3.w = ucl::Convert<$0>(($1.y >> 12u) & 15u);
  $4.x = ucl::Convert<$0>(($1.z) & 15u);
  $4.y = ucl::Convert<$0>(($1.z >>  4u) & 15u);
  $4.z = ucl::Convert<$0>(($1.z >>  8u) & 15u);
  $4.w = ucl::Convert<$0>(($1.z >> 12u) & 15u);
  $5.x = ucl::Convert<$0>(($1.w) & 15u);
  $5.y = ucl::Convert<$0>(($1.w >> 4u) & 15u);
  $5.z = ucl::Convert<$0>(($1.w >> 8u) & 15u);
  $5.w = ucl::Convert<$0>(($1.w >> 12u) & 15u);
)";
  }
  *result = absl::Substitute(*result, template_args[0], args[0], args[1],
                             args[2], args[3], args[4]);
  return absl::OkStatus();
}

absl::Status PerformU8x4ToU2x16AsVec4x4(
    const GpuInfo& gpu_info, const std::vector<std::string>& template_args,
    const std::vector<std::string>& args, std::string* result) {
  if (template_args.size() != 1 || args.size() != 5) {
    return absl::NotFoundError(
        "U8x4ToU2x16AsVec4x4 must have 1 template argument and 5 "
        "arguments");
  }
  int type_size = 0;
  DataType type;
  ABSL_RETURN_IF_ERROR(
      DataTypeFromTemplateArg(template_args[0], &type, &type_size));
  const bool use_float_math_for_unpacking =
      type == DataType::FLOAT16 && (gpu_info.IsApple() || gpu_info.IsPowerVR());
  if (use_float_math_for_unpacking) {
    *result = R"(
  {
  $04 wt0 = ucl::Convert<$04>($1) * ucl::Init<$0>(0.25f);
  $04 wt1 = floor(wt0);
  $2.x = (wt0.x - wt1.x) * ucl::Init<$0>(4.0f);
  $3.x = (wt0.y - wt1.y) * ucl::Init<$0>(4.0f);
  $4.x = (wt0.z - wt1.z) * ucl::Init<$0>(4.0f);
  $5.x = (wt0.w - wt1.w) * ucl::Init<$0>(4.0f);
  wt0 = wt1 * ucl::Init<$0>(0.25f);
  wt1 = floor(wt0);
  $2.y = (wt0.x - wt1.x) * ucl::Init<$0>(4.0f);
  $3.y = (wt0.y - wt1.y) * ucl::Init<$0>(4.0f);
  $4.y = (wt0.z - wt1.z) * ucl::Init<$0>(4.0f);
  $5.y = (wt0.w - wt1.w) * ucl::Init<$0>(4.0f);
  wt0 = wt1 * ucl::Init<$0>(0.25f);
  wt1 = floor(wt0);
  $2.z = (wt0.x - wt1.x) * ucl::Init<$0>(4.0f);
  $3.z = (wt0.y - wt1.y) * ucl::Init<$0>(4.0f);
  $4.z = (wt0.z - wt1.z) * ucl::Init<$0>(4.0f);
  $5.z = (wt0.w - wt1.w) * ucl::Init<$0>(4.0f);
  $2.w = wt1.x;
  $3.w = wt1.y;
  $4.w = wt1.z;
  $5.w = wt1.w;
  }
)";
  } else {
    if (gpu_info.IsApiCuda()) {
      *result = R"(
  $2 = ucl::Convert<$04>(ucl::Init<uint4>($1.x, $1.x >> 2u, $1.x >> 4u, $1.x >> 6u) & ucl::Init<uint4>(3u));
  $3 = ucl::Convert<$04>(ucl::Init<uint4>($1.y, $1.y >> 2u, $1.y >> 4u, $1.y >> 6u) & ucl::Init<uint4>(3u));
  $4 = ucl::Convert<$04>(ucl::Init<uint4>($1.z, $1.z >> 2u, $1.z >> 4u, $1.z >> 6u) & ucl::Init<uint4>(3u));
  $5 = ucl::Convert<$04>(ucl::Init<uint4>($1.w, $1.w >> 2u, $1.w >> 4u, $1.w >> 6u) & ucl::Init<uint4>(3u));
)";
    } else {
      // ushort4 shows the best performance on Intel, haven't seen difference on
      // other GPUs (vs uchar8)
      *result = R"(
  $2 = ucl::Convert<$04>(ucl::Init<ushort4>($1.x, $1.x >> 2u, $1.x >> 4u, $1.x >> 6u) & ucl::Init<ushort4>(3u));
  $3 = ucl::Convert<$04>(ucl::Init<ushort4>($1.y, $1.y >> 2u, $1.y >> 4u, $1.y >> 6u) & ucl::Init<ushort4>(3u));
  $4 = ucl::Convert<$04>(ucl::Init<ushort4>($1.z, $1.z >> 2u, $1.z >> 4u, $1.z >> 6u) & ucl::Init<ushort4>(3u));
  $5 = ucl::Convert<$04>(ucl::Init<ushort4>($1.w, $1.w >> 2u, $1.w >> 4u, $1.w >> 6u) & ucl::Init<ushort4>(3u));
)";
    }
  }
  *result = absl::Substitute(*result, template_args[0], args[0], args[1],
                             args[2], args[3], args[4]);
  return absl::OkStatus();
}

absl::Status PerformI16ToVec4I4(const GpuInfo& gpu_info,
                                const std::vector<std::string>& template_args,
                                const std::vector<std::string>& args,
                                std::string* result) {
  if (template_args.size() != 1 || args.size() != 1) {
    return absl::NotFoundError(
        "PerformI16ToVec4I4 must have 1 template argument and 1 argument");
  }
  int type_size = 0;
  DataType type;
  ABSL_RETURN_IF_ERROR(
      DataTypeFromTemplateArg(template_args[0], &type, &type_size));
  *result = R"(ucl::Init<$04>(
  ucl::Convert<$0>(($1 << 28u) >> 28u),
  ucl::Convert<$0>(ucl::Convert<short>(($1 << 24u) >> 28u)),
  ucl::Convert<$0>(ucl::Convert<short>(($1 << 20u) >> 28u)),
  ucl::Convert<$0>(ucl::Convert<short>(($1 << 16u) >> 28u))))";
  *result = absl::Substitute(*result, template_args[0], args[0]);
  return absl::OkStatus();
}

absl::Status PerformI8ToVec4I2(const GpuInfo& gpu_info,
                               const std::vector<std::string>& template_args,
                               const std::vector<std::string>& args,
                               std::string* result) {
  if (template_args.size() != 1 || args.size() != 1) {
    return absl::NotFoundError(
        "PerformI8ToVec4I2 must have 1 template argument and 1 argument");
  }
  int type_size = 0;
  DataType type;
  ABSL_RETURN_IF_ERROR(
      DataTypeFromTemplateArg(template_args[0], &type, &type_size));
  *result = R"(ucl::Init<$04>(
  ucl::Convert<$0>(($1 << 30u) >> 30u),
  ucl::Convert<$0>(($1 << 28u) >> 30u),
  ucl::Convert<$0>(($1 << 26u) >> 30u),
  ucl::Convert<$0>(($1 << 24u) >> 30u)))";
  *result = absl::Substitute(*result, template_args[0], args[0]);
  return absl::OkStatus();
}

absl::Status PerformU32Sparse2x4ToU4x16AsVec4x4(
    const GpuInfo& gpu_info, const std::vector<std::string>& template_args,
    const std::vector<std::string>& args, std::string* result) {
  if (template_args.size() != 1 || args.size() != 6) {
    return absl::NotFoundError(
        "U32Sparse2x4ToU4x16AsVec4x4 must have 1 template argument and 6 "
        "arguments");
  }
  if (gpu_info.IsApiMetal() && gpu_info.IsApple()) {
    *result = R"(
{
  uint pos_i0o0 = ($2 & 3u);
  uint pos_i0o1 = (($2 >> 2u) & 3u);
  uint pos_i0o2 = (($2 >> 4u) & 3u);
  uint pos_i0o3 = (($2 >> 6u) & 3u);
  uint pos_i1o0 = (($2 >> 8u) & 3u);
  uint pos_i1o1 = (($2 >> 10u) & 3u);
  uint pos_i1o2 = (($2 >> 12u) & 3u);
  uint pos_i1o3 = (($2 >> 14u) & 3u);
  $04 wt0 = ucl::Convert<$04>(ucl::Reinterpret<uint, uchar4>($1)) * ucl::Init<$0>(0.0625f);
  $0 w_i0o1 = floor(wt0.x);
  $0 w_i0o3 = floor(wt0.y);
  $0 w_i0o0 = (wt0.x - w_i0o1) * ucl::Init<$0>(16.0f);
  $0 w_i0o2 = (wt0.y - w_i0o3) * ucl::Init<$0>(16.0f);
  $0 w_i1o1 = floor(wt0.z);
  $0 w_i1o3 = floor(wt0.w);
  $0 w_i1o0 = (wt0.z - w_i1o1) * ucl::Init<$0>(16.0f);
  $0 w_i1o2 = (wt0.w - w_i1o3) * ucl::Init<$0>(16.0f);
  $04 w_i0123o0 = ucl::Init<$04>(8.0f);
  w_i0123o0[pos_i0o0] = w_i0o0;
  w_i0123o0[pos_i1o0] = w_i1o0;
  $04 w_i0123o1 = ucl::Init<$04>(8.0f);
  w_i0123o1[pos_i0o1] = w_i0o1;
  w_i0123o1[pos_i1o1] = w_i1o1;
  $04 w_i0123o2 = ucl::Init<$04>(8.0f);
  w_i0123o2[pos_i0o2] = w_i0o2;
  w_i0123o2[pos_i1o2] = w_i1o2;
  $04 w_i0123o3 = ucl::Init<$04>(8.0f);
  w_i0123o3[pos_i0o3] = w_i0o3;
  w_i0123o3[pos_i1o3] = w_i1o3;
  $3.x = w_i0123o0[0];
  $3.y = w_i0123o1[0];
  $3.z = w_i0123o2[0];
  $3.w = w_i0123o3[0];
  $4.x = w_i0123o0[1];
  $4.y = w_i0123o1[1];
  $4.z = w_i0123o2[1];
  $4.w = w_i0123o3[1];
  $5.x = w_i0123o0[2];
  $5.y = w_i0123o1[2];
  $5.z = w_i0123o2[2];
  $5.w = w_i0123o3[2];
  $6.x = w_i0123o0[3];
  $6.y = w_i0123o1[3];
  $6.z = w_i0123o2[3];
  $6.w = w_i0123o3[3];
}
)";
  } else {
    *result = R"(
{
  uint pos_i0o0 = ($2 & 3u) * 4u;
  uint pos_i0o1 = (($2 >> 2u) & 3u) * 4u;
  uint pos_i0o2 = (($2 >> 4u) & 3u) * 4u;
  uint pos_i0o3 = (($2 >> 6u) & 3u) * 4u;
  uint pos_i1o0 = (($2 >> 8u) & 3u) * 4u;
  uint pos_i1o1 = (($2 >> 10u) & 3u) * 4u;
  uint pos_i1o2 = (($2 >> 12u) & 3u) * 4u;
  uint pos_i1o3 = (($2 >> 14u) & 3u) * 4u;
  uint w_i0o0 = ($1 & 15u);
  uint w_i0o1 = (($1 >> 4u) & 15u);
  uint w_i0o2 = (($1 >> 8u) & 15u);
  uint w_i0o3 = (($1 >> 12u) & 15u);
  uint w_i1o0 = (($1 >> 16u) & 15u);
  uint w_i1o1 = (($1 >> 20u) & 15u);
  uint w_i1o2 = (($1 >> 24u) & 15u);
  uint w_i1o3 = (($1 >> 28u) & 15u);
  uint w_i0123o0 = 34952u; // 0x8888
  w_i0123o0 &= ~((8u << pos_i0o0) | (8u << pos_i1o0));
  w_i0123o0 |= w_i0o0 << pos_i0o0;
  w_i0123o0 |= w_i1o0 << pos_i1o0;
  uint w_i0123o1 = 34952u; // 0x8888
  w_i0123o1 &= ~((8u << pos_i0o1) | (8u << pos_i1o1));
  w_i0123o1 |= w_i0o1 << pos_i0o1;
  w_i0123o1 |= w_i1o1 << pos_i1o1;
  uint w_i0123o2 = 34952u; // 0x8888
  w_i0123o2 &= ~((8u << pos_i0o2) | (8u << pos_i1o2));
  w_i0123o2 |= w_i0o2 << pos_i0o2;
  w_i0123o2 |= w_i1o2 << pos_i1o2;
  uint w_i0123o3 = 34952u; // 0x8888
  w_i0123o3 &= ~((8u << pos_i0o3) | (8u << pos_i1o3));
  w_i0123o3 |= w_i0o3 << pos_i0o3;
  w_i0123o3 |= w_i1o3 << pos_i1o3;
  $3.x = ucl::Convert<$0>((w_i0123o0) & 15u);
  $3.y = ucl::Convert<$0>((w_i0123o1) & 15u);
  $3.z = ucl::Convert<$0>((w_i0123o2) & 15u);
  $3.w = ucl::Convert<$0>((w_i0123o3) & 15u);
  $4.x = ucl::Convert<$0>((w_i0123o0 >> 4u) & 15u);
  $4.y = ucl::Convert<$0>((w_i0123o1 >> 4u) & 15u);
  $4.z = ucl::Convert<$0>((w_i0123o2 >> 4u) & 15u);
  $4.w = ucl::Convert<$0>((w_i0123o3 >> 4u) & 15u);
  $5.x = ucl::Convert<$0>((w_i0123o0 >> 8u) & 15u);
  $5.y = ucl::Convert<$0>((w_i0123o1 >> 8u) & 15u);
  $5.z = ucl::Convert<$0>((w_i0123o2 >> 8u) & 15u);
  $5.w = ucl::Convert<$0>((w_i0123o3 >> 8u) & 15u);
  $6.x = ucl::Convert<$0>((w_i0123o0 >> 12u) & 15u);
  $6.y = ucl::Convert<$0>((w_i0123o1 >> 12u) & 15u);
  $6.z = ucl::Convert<$0>((w_i0123o2 >> 12u) & 15u);
  $6.w = ucl::Convert<$0>((w_i0123o3 >> 12u) & 15u);
}
)";
  }
  *result = absl::Substitute(*result, template_args[0], args[0], args[1],
                             args[2], args[3], args[4], args[5]);
  return absl::OkStatus();
}

absl::Status PerformExpSelector(const GpuInfo& gpu_info,
                                const std::vector<std::string>& template_args,
                                const std::vector<std::string>& args,
                                std::string* result) {
  if (template_args.size() != 1) {
    return absl::NotFoundError(
        "ucl::Exp<Type>(value) requires one template argument");
  }
  if (args.size() != 1) {
    return absl::NotFoundError("ucl::Exp<Type>(value) requires one argument");
  }
  if (gpu_info.IsApiOpenCl()) {
    int vector_size = 0;
    DataType type;
    ABSL_RETURN_IF_ERROR(
        DataTypeFromTemplateArg(template_args[0], &type, &vector_size));
    if (type == DataType::FLOAT16 &&
        (gpu_info.IsAdreno() || gpu_info.IsPowerVR() || gpu_info.IsMali())) {
      *result = "convert_" + ToCLDataType(type, vector_size) +
                "(native_exp(convert_" +
                ToCLDataType(DataType::FLOAT32, vector_size) + "(" + args[0] +
                ")))";
    } else {
      *result = "exp(" + args[0] + ")";
    }
    return absl::OkStatus();
  } else {
    *result = "exp(" + args[0] + ")";
    return absl::OkStatus();
  }
}

inline bool IsOpenBracket(char c) {
  return c == '(' || c == '[' || c == '{' || c == '<';
}
inline bool IsCloseBracket(char c) {
  return c == ')' || c == ']' || c == '}' || c == '>';
}
// To be used only for IsCloseBracket(close_bracket) = true
inline char GetOpenBracket(char close_bracket) {
  if (close_bracket == ')') {
    return '(';
  } else if (close_bracket == ']') {
    return '[';
  } else if (close_bracket == '}') {
    return '{';
  } else if (close_bracket == '>') {
    return '<';
  } else {
    return '.';
  }
}

std::string GetTypePostfix(DataType data_type) {
  if (IsFloatType(data_type)) {
    return ".f";
  } else {
    return IsSigned(data_type) ? "" : "u";
  }
}
}  // namespace

absl::Status DataTypeFromTemplateArg(absl::string_view type_str, DataType* type,
                                     int* vector_size) {
  int vec_size = 0;
  int power = 1;
  while (isdigit(type_str.back())) {
    const int digit = type_str.back() - '0';
    type_str.remove_suffix(1);
    vec_size += power * digit;
    power *= 10;
  }
  if (vec_size == 0) {
    vec_size = 1;
  }
  *vector_size = vec_size;
  static const auto& kTypes =
      *new absl::flat_hash_map<absl::string_view, DataType>{
          {"half", DataType::FLOAT16},  {"float", DataType::FLOAT32},
          {"int", DataType::INT32},     {"short", DataType::INT16},
          {"char", DataType::INT8},     {"uint", DataType::UINT32},
          {"ushort", DataType::UINT16}, {"uchar", DataType::UINT8},
          {"bool", DataType::BOOL},     {"bfloat", DataType::BFLOAT16}};
  const auto it = kTypes.find(type_str);
  if (it != kTypes.end()) {
    *type = it->second;
    return absl::OkStatus();
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported data type - ", type_str));
  }
}

std::string MemoryTypeToCLType(MemoryType type) {
  switch (type) {
    case MemoryType::GLOBAL:
      return "__global";
    case MemoryType::CONSTANT:
      return "__constant";
    case MemoryType::LOCAL:
      return "__local";
  }
  return "";
}

std::string MemoryTypeToMetalType(MemoryType type) {
  switch (type) {
    case MemoryType::GLOBAL:
      return "device";
    case MemoryType::CONSTANT:
      return "constant";
      break;
    case MemoryType::LOCAL:
      return "threadgroup";
  }
  return "";
}

float4 GetMaskForLastPlane(int channels) {
  float4 mask = float4(0.0f);
  const int remainder = channels % 4 == 0 ? 4 : channels % 4;
  for (int i = 0; i < remainder; ++i) {
    mask[i] = 1.0f;
  }
  return mask;
}

int GetRecommendedBlockSizeForConv(const GpuInfo& gpu_info,
                                   CalculationsPrecision precision,
                                   int task_size) {
  const float task_size_per_cu =
      task_size / static_cast<float>(gpu_info.GetComputeUnitsCount());
  int block_size = 1;
  float threshold_1 = FLT_MAX;
  float threshold_2 = FLT_MAX;
  float threshold_4 = FLT_MAX;
  if (!gpu_info.IsMali()) {
    return 1;
  }
  MaliInfo mali_info = gpu_info.mali_info;
  switch (precision) {
    case CalculationsPrecision::F16:
      if (mali_info.IsMidgard()) {
        threshold_1 = 256.0f * 4.0f;
        threshold_2 = 256.0f * 16.0f;
      } else if (mali_info.IsBifrostGen1()) {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 4.0f;
        threshold_4 = 256.0f * 8.0f;
      } else if (mali_info.IsBifrostGen2()) {
        threshold_1 = 256.0f * 2.0f;
        threshold_2 = 256.0f * 8.0f;
        threshold_4 = 256.0f * 16.0f;
      } else {
        // BifrostGen3 and later
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 6.0f;
        threshold_4 = 256.0f * 16.0f;
      }
      break;
    case CalculationsPrecision::F32_F16:
      if (mali_info.IsMidgard()) {
        threshold_1 = 256.0f * 4.0f;
      } else if (mali_info.IsBifrostGen1()) {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 3.0f;
        threshold_4 = 256.0f * 32.0f;
      } else if (mali_info.IsBifrostGen2()) {
        threshold_1 = 256.0f * 2.0f;
        threshold_2 = 256.0f * 8.0f;
      } else {
        // BifrostGen3 and later
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 8.0f;
      }
      break;
    case CalculationsPrecision::F32:
      if (mali_info.IsMidgard()) {
        threshold_1 = 256.0f * 16.0f;
      } else if (mali_info.IsBifrostGen1()) {
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 4.0f;
      } else if (mali_info.IsBifrostGen2()) {
        threshold_1 = 128.0f;
        threshold_2 = 256.0f * 4.0f;
      } else {
        // BifrostGen3 and later
        threshold_1 = 256.0f;
        threshold_2 = 256.0f * 12.0f;
      }
      break;
  }
  if (task_size_per_cu <= threshold_1) {
    block_size = 1;
  } else if (task_size_per_cu <= threshold_2) {
    block_size = 2;
  } else if (task_size_per_cu <= threshold_4) {
    block_size = 4;
  } else {
    block_size = 8;
  }
  return block_size;
}

int3 GetWorkGroupsCount(const int3& grid_size, const int3& work_group_size) {
  int3 work_groups_count;
  work_groups_count.x = DivideRoundUp(grid_size.x, work_group_size.x);
  work_groups_count.y = DivideRoundUp(grid_size.y, work_group_size.y);
  work_groups_count.z = DivideRoundUp(grid_size.z, work_group_size.z);
  return work_groups_count;
}

std::string GetTypeDeclaration(const GpuInfo& gpu_info, DataType data_type,
                               int vec_size) {
  if (gpu_info.IsApiOpenCl()) {
    return ToCLDataType(data_type, vec_size);
  } else if (gpu_info.IsApiMetal()) {
    return ToMetalDataType(data_type, vec_size);
  } else if (gpu_info.IsGlsl()) {
    return ToGlslShaderDataType(data_type, vec_size, true,
                                gpu_info.IsGlslSupportsExplicitFp16());
  } else if (gpu_info.IsApiWebGpu()) {
    return ToCLDataType(data_type, vec_size);
  } else if (gpu_info.IsApiCuda()) {
    return ToUclDataType(data_type, vec_size);
  } else {
    return "";
  }
}

std::string GetZeroValue(DataType data_type) {
  if (data_type == DataType::BOOL) {
    return "false";
  } else {
    return "0" + GetTypePostfix(data_type);
  }
}

std::string GetOneValue(DataType data_type) {
  if (data_type == DataType::BOOL) {
    return "true";
  } else {
    return "1" + GetTypePostfix(data_type);
  }
}

absl::string_view GetNextWord(absl::string_view code, size_t first_position) {
  size_t pos = first_position;
  while (pos < code.size() && IsWordSymbol(code[pos])) {
    pos++;
  }
  return code.substr(first_position, pos - first_position);
}

std::string GetNextWordStr(absl::string_view code, size_t first_position) {
  return std::string(GetNextWord(code, first_position));
}

size_t FindEnclosingBracket(const std::string& text, size_t first_pos,
                            char bracket) {
  const std::map<char, char> brackets = {
      {'(', ')'},
      {'{', '}'},
      {'[', ']'},
      {'<', '>'},
  };
  char b_open = bracket;
  auto it = brackets.find(b_open);
  if (it == brackets.end()) {
    return -1;
  }
  char b_close = it->second;
  size_t pos = first_pos;
  int opened = 1;
  int closed = 0;
  while (opened != closed && pos < text.size()) {
    if (text[pos] == b_open) {
      opened++;
    } else if (text[pos] == b_close) {
      closed++;
    }
    pos++;
  }
  if (opened == closed) {
    return pos;
  } else {
    return -1;
  }
}

absl::Status ParseArguments(absl::string_view text, size_t open_bracket_pos,
                            size_t* close_bracket_pos,
                            std::vector<std::string>* arguments) {
  std::stack<char> brackets;

  size_t text_size = text.size();
  const char open_bracket = text[open_bracket_pos];
  size_t arg_start = open_bracket_pos + 1;
  for (size_t i = arg_start; i < text_size; ++i) {
    char c = text[i];
    if (c == ',' && brackets.empty()) {
      absl::string_view arg =
          absl::StripAsciiWhitespace(text.substr(arg_start, i - arg_start));
      if (arg.empty()) {
        return absl::InvalidArgumentError("Empty argument");
      }
      arguments->emplace_back(arg);
      arg_start = i + 1;
    } else if (c == '<' && i + 1 < text_size && text[i + 1] == '<') {
      i += 1;
    } else if (c == '>' && i + 1 < text_size && text[i + 1] == '>') {
      i += 1;
    } else if (IsOpenBracket(c)) {
      brackets.push(c);
    } else if (IsCloseBracket(c)) {
      const char local_open_bracket = GetOpenBracket(c);
      if (brackets.empty()) {
        if (local_open_bracket == open_bracket) {
          absl::string_view arg =
              absl::StripAsciiWhitespace(text.substr(arg_start, i - arg_start));
          if (arg.empty()) {
            // do not return error for no args call: func() or func(  );
            if (!arguments->empty()) {
              return absl::InvalidArgumentError("Empty argument");
            }
          } else {
            arguments->emplace_back(arg);
          }
          *close_bracket_pos = i;
          return absl::OkStatus();
        } else {
          return absl::FailedPreconditionError(absl::StrCat(
              "Expected closing bracket for ", std::string(1, open_bracket),
              ", but found ", std::string(1, c)));
        }
      }
      if (brackets.top() != local_open_bracket) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Expected closing bracket for ", std::string(1, brackets.top()),
            ", but found ", std::string(1, c)));
      }

      brackets.pop();
    }
  }
  return absl::FailedPreconditionError(absl::StrCat(
      "Expected closing bracket for ", std::string(1, open_bracket),
      ", but reached end of string"));
}

absl::Status PerformSystemFunction(
    const GpuInfo& gpu_info, absl::string_view function_name,
    const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) {
  if (function_name == "Init") {
    return PerformInitSelector(gpu_info, template_args, args, result);
  } else if (function_name == "Convert") {
    return PerformConvertSelector(gpu_info, template_args, args, result);
  } else if (function_name == "Reinterpret") {
    return PerformReinterpretSelector(gpu_info, template_args, args, result);
  } else if (function_name == "GetGlobalId") {
    return PerformGetGlobalIdSelector(gpu_info, template_args, args, result);
  } else if (function_name == "GetLocalId") {
    return PerformGetLocalIdSelector(gpu_info, template_args, args, result);
  } else if (function_name == "GetGroupId") {
    return PerformGetGroupIdSelector(gpu_info, template_args, args, result);
  } else if (function_name == "GetGroupSize") {
    return PerformGetGroupSizeSelector(gpu_info, template_args, args, result);
  } else if (function_name == "GetSubGroupLocalId") {
    return PerformGetSubGroupLocalIdSelector(gpu_info, template_args, args,
                                             result);
  } else if (function_name == "GetSubGroupId") {
    return PerformGetSubGroupIdSelector(gpu_info, template_args, args, result);
  } else if (function_name == "GetSubGroupSize") {
    return PerformGetSubGroupSizeSelector(gpu_info, template_args, args,
                                          result);
  } else if (function_name == "SyncThreads") {
    return PerformSyncThreadsSelector(gpu_info, template_args, args, result);
  } else if (function_name == "SubGroupBroadcast") {
    return PerformSubGroupBroadcastSelector(gpu_info, template_args, args,
                                            result);
  } else if (function_name == "U32x4ToU8x16AsVec4x4") {
    return PerformU32x4ToU8x16AsVec4x4(gpu_info, template_args, args, result);
  } else if (function_name == "Int32x4ToInt8x16AsVec4x4") {
    return PerformInt32x4ToInt8x16AsVec4x4(template_args, args, result);
  } else if (function_name == "U32x2ToU4x16AsVec4x4") {
    return PerformU32x2ToU4x16AsVec4x4(gpu_info, template_args, args, result);
  } else if (function_name == "U32x1ToU2x16AsVec4x4") {
    return PerformU32x1ToU2x16AsVec4x4(gpu_info, template_args, args, result);
  } else if (function_name == "U16x4ToU4x16AsVec4x4") {
    return PerformU16x4ToU4x16AsVec4x4(gpu_info, template_args, args, result);
  } else if (function_name == "U8x4ToU2x16AsVec4x4") {
    return PerformU8x4ToU2x16AsVec4x4(gpu_info, template_args, args, result);
  } else if (function_name == "I16ToVec4I4") {
    return PerformI16ToVec4I4(gpu_info, template_args, args, result);
  } else if (function_name == "I8ToVec4I2") {
    return PerformI8ToVec4I2(gpu_info, template_args, args, result);
  } else if (function_name == "U32Sparse2x4ToU4x16AsVec4x4") {
    return PerformU32Sparse2x4ToU4x16AsVec4x4(gpu_info, template_args, args,
                                              result);
  } else if (function_name == "Exp") {
    return PerformExpSelector(gpu_info, template_args, args, result);
  } else if (function_name == "ConvertFromBfloat") {
    if (template_args.size() != 1 || args.size() != 1) {
      return absl::NotFoundError(
          "ConvertFromBfloat requires 1 arg and 1 template arg.");
    }
    *result = "ConvertFromBfloatTo" + template_args[0] + "(" + args[0] + ")";
    return absl::OkStatus();
  } else if (function_name == "ConvertToBfloat") {
    if (template_args.size() != 1 || args.size() != 1) {
      return absl::NotFoundError(
          "ConvertToBfloat requires 1 arg and 1 template arg.");
    }
    *result = "ConvertToBfloatFrom" + template_args[0] + "(" + args[0] + ")";
    return absl::OkStatus();
  } else {
    return absl::NotFoundError(
        absl::StrCat("No system function with name - ", function_name));
  }
}

}  // namespace ml_drift
