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

#include "ml_drift/common/task/wave_memory_util.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/compiler_options.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/util.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace {
absl::Status ResolveWaveMemoryToWaveBroadcast(int wave_size,
                                              const GpuInfo& gpu_info,
                                              std::string* code) {
  const std::string mem_qualifier = "__wave";
  const size_t mem_pos = code->find(mem_qualifier);
  std::string mem_name;
  int array_size = 0;
  DataType data_type;
  int vector_size;
  bool glsl_broadcast_half4_as_uint2 = false;
  if (mem_pos != std::string::npos) {
    size_t pos = mem_pos + mem_qualifier.size() + 1;
    const std::string mem_type = std::string(GetNextWord(*code, pos));
    pos += mem_type.size() + 1;
    mem_name = std::string(GetNextWord(*code, pos));
    pos += mem_name.size() + 1;
    array_size = std::stoi(std::string(GetNextWord(*code, pos)));
    ABSL_RETURN_IF_ERROR(
        DataTypeFromTemplateArg(mem_type, &data_type, &vector_size));
    glsl_broadcast_half4_as_uint2 = gpu_info.IsGlsl() &&
                                    !gpu_info.IsGlslSupportsExplicitFp16() &&
                                    data_type == DataType::kFloat16;
    const int vals_per_thread = DivideRoundUp(array_size, wave_size);
    std::string patch = ToUclDataType(data_type, vector_size) + " ";
    if (glsl_broadcast_half4_as_uint2) {
      patch = "uint2 ";
    }
    for (int i = 0; i < vals_per_thread; ++i) {
      if (i != 0) {
        patch += ", ";
      }
      patch += mem_name + "_wave_var" + std::to_string(i);
    }
    patch += ";\n";
    patch += "  int wave_memory_id = ucl::GetSubGroupLocalId();";
    const size_t end_line = code->find('\n', mem_pos);
    code->replace(mem_pos, end_line - mem_pos, patch);
  }
#if defined(__linux__)
  // TODO: WebGPUs Vulkan backend has a bug when broadcasting
  // f16, so cast to u32. This should be fixed since it's slower than f16.
  const bool webgpu_broadcast_half4_as_uint2 =
      gpu_info.IsApiWebGpu() && gpu_info.webgpu_info.supports_fp16 &&
      data_type == DataType::kFloat16 && gpu_info.IsIntel() &&
      // TODO: This may be able to be extended down to Gen10.
      !gpu_info.intel_info.IsGenerationOrNewer(IntelGeneration::kGen12);
#else
  const bool webgpu_broadcast_half4_as_uint2 = false;
#endif

  {  // resolve access to the wave memory
    // wave_memory[0] -> wave_memory_wave_var0
    std::vector<std::pair<std::string, std::string>> replacements;
    std::string broadcast_func_name = "sub_group_broadcast";
    if (gpu_info.IsGlsl() || gpu_info.IsApiWebGpu()) {
      broadcast_func_name = "subgroupBroadcast";
    } else if (gpu_info.IsApiMetal()) {
      broadcast_func_name = "simd_broadcast";
    }
    const bool opencl_broadcast_half4_as_long = gpu_info.IsPowerVR() &&
                                                gpu_info.IsApiOpenCl() &&
                                                data_type == DataType::kFloat16;
    const bool broadcast_vec4 =
        gpu_info.IsApiMetal() || gpu_info.IsApiWebGpu() || gpu_info.IsGlsl() ||
        (gpu_info.IsApiOpenCl() &&
         gpu_info.SupportsExtension("cl_khr_subgroup_extended_types"));
    for (int i = 0; i < array_size; ++i) {
      const std::string val_name = mem_name + "[" + std::to_string(i) + "]";
      const int wave_local_id = i % wave_size;
      const int var_index = i / wave_size;
      const std::string var_name =
          absl::StrCat(mem_name, "_wave_var", var_index);
      std::string new_val_name;
      if (opencl_broadcast_half4_as_long) {
        new_val_name = absl::StrCat("as_half4(sub_group_broadcast(as_long(",
                                    var_name, "), ", wave_local_id, "u))");
      } else if (glsl_broadcast_half4_as_uint2) {
        new_val_name =
            absl::StrCat("ucl::Reinterpret<uint2, half4>(subgroupBroadcast(",
                         var_name, ", ", wave_local_id, "u))");
      } else if (webgpu_broadcast_half4_as_uint2) {
        new_val_name = absl::StrCat(
            "bitcast<vec4<f16>>(subgroupBroadcast(bitcast<vec2<u32>>(",
            var_name, "), ", wave_local_id, "u))");
      } else if (broadcast_vec4) {
        new_val_name = absl::StrCat(broadcast_func_name, "(", var_name, ", ",
                                    wave_local_id, "u)");
      } else {
        const std::string val_x = absl::StrCat(
            broadcast_func_name, "(", var_name, ".x, ", wave_local_id, "u)");
        const std::string val_y = absl::StrCat(
            broadcast_func_name, "(", var_name, ".y, ", wave_local_id, "u)");
        const std::string val_z = absl::StrCat(
            broadcast_func_name, "(", var_name, ".z, ", wave_local_id, "u)");
        const std::string val_w = absl::StrCat(
            broadcast_func_name, "(", var_name, ".w, ", wave_local_id, "u)");
        new_val_name = absl::StrCat(
            "ucl::Init<", ToUclDataType(data_type, vector_size), ">(", val_x,
            ", ", val_y, ", ", val_z, ", ", val_w, ")");
      }
      replacements.push_back({val_name, new_val_name});
    }
    absl::StrReplaceAll(replacements, code);
  }

  const std::string upload_func_name = "ucl::WaveLoad";
  size_t upload_pos = code->find(upload_func_name);
  while (upload_pos != std::string::npos) {
    size_t pos = upload_pos + upload_func_name.size();
    std::vector<std::string> args;
    size_t close_bracket_pos;
    ABSL_RETURN_IF_ERROR(ParseArguments(*code, pos, &close_bracket_pos, &args));
    const int upload_count = std::stoi(args[3]);
    std::string patch;
    for (int i = 0; i * wave_size < upload_count; ++i) {
      std::string var_name = mem_name + "_wave_var" + std::to_string(i);
      std::string index =
          "(wave_memory_id + " + std::to_string(i * wave_size) + ")";
      const bool partial_load = (i + 1) * wave_size > upload_count;
      if (partial_load) {
        patch += "if (" + index + " < " + std::to_string(upload_count) + ") { ";
      }
      absl::StrAppend(&patch, var_name, " = ", args[1], "[", args[2], " + ",
                      index, "];");
      if (partial_load) {
        patch += '}';
      }
      patch += '\n';
    }
    code->replace(upload_pos, close_bracket_pos - upload_pos + 1, patch);
    upload_pos = code->find(upload_func_name, upload_pos);
  }
  const std::string sync_func_name = "ucl::SyncThreads<WaveLoad>()";
  size_t sync_pos = code->find(sync_func_name);
  while (sync_pos != std::string::npos) {
    const std::string patch = "";
    const size_t end_line = code->find('\n', sync_pos);
    code->replace(sync_pos, end_line - sync_pos, patch);
    sync_pos = code->find(sync_func_name, sync_pos);
  }
  return absl::OkStatus();
}

absl::Status ResolveWaveMemoryToWaveBroadcast(
    const GpuInfo& gpu_info, std::string* code,
    std::vector<CompilerOptions>* compiler_options) {
  const std::string ext_declaration =
      "#pragma OPENCL EXTENSION ucl_wave_memory: enable";
  const size_t ext_pos = code->find(ext_declaration);
  if (ext_pos != std::string::npos) {
    std::string patch;
    if (gpu_info.IsApiOpenCl()) {
      if (gpu_info.opencl_info.cl_version == OpenClVersion::kCl2_0 ||
          gpu_info.SupportsExtension("cl_khr_subgroups")) {
        patch = "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n";
      } else if (gpu_info.SupportsExtension("cl_intel_subgroups")) {
        patch = "#pragma OPENCL EXTENSION cl_intel_subgroups : enable\n";
      }
      if (gpu_info.SupportsExtension("cl_khr_subgroup_extended_types")) {
        patch +=
            "#pragma OPENCL EXTENSION cl_khr_subgroup_extended_types : "
            "enable\n";
      }
    } else if (gpu_info.IsGlsl()) {
      patch = "#extension GL_KHR_shader_subgroup_ballot : require\n";
      if (gpu_info.IsGlslSupportsExplicitFp16()) {
        patch +=
            "#extension GL_EXT_shader_subgroup_extended_types_float16 : "
            "require\n";
      }
    } else if (gpu_info.IsApiWebGpu()) {
      patch += "enable subgroups;\n";
      if (gpu_info.webgpu_info.supports_fp16) {
        patch += "enable f16;\n";
      }
    }
    code->replace(ext_pos, ext_declaration.size(), patch);
  } else {
    return absl::OkStatus();
  }
  std::vector<int> wave_sizes;
  {  // parse __attribute__((ucl_wave_memory_sizes(N0, N1, ...)))
    const std::string attribute_wave_sizes =
        "__attribute__((ucl_wave_memory_sizes";
    const size_t attribute_pos = code->find(attribute_wave_sizes);
    if (attribute_pos != std::string::npos) {
      size_t pos = attribute_pos + attribute_wave_sizes.size();
      std::vector<std::string> args;
      size_t close_bracket_pos;
      ABSL_RETURN_IF_ERROR(
          ParseArguments(*code, pos, &close_bracket_pos, &args));
      code->erase(attribute_pos, close_bracket_pos + 2 - attribute_pos + 1);
      for (int i = 0; i < args.size(); ++i) {
        wave_sizes.push_back(std::stoi(args[i]));
      }
    }
  }
  if (wave_sizes.size() > 1) {
    const std::string scope_end_hint = "__hint__((ucl_wave_memory_scope_end))";
    const size_t scope_end_hint_pos = code->find(scope_end_hint);
    size_t scope_start, scope_end;
    if (scope_end_hint_pos != std::string::npos) {
      code->erase(scope_end_hint_pos, scope_end_hint.size());
      scope_end = scope_end_hint_pos;
      scope_start = code->find("__wave");
    } else {
      scope_start = code->find("MAIN_FUNCTION");
      while (code->at(scope_start) != '{') {
        scope_start++;
      }
      scope_start += 1;
      scope_end = FindEnclosingBracket(*code, scope_start, '{');
    }
    std::string main_body =
        code->substr(scope_start, scope_end - scope_start - 1);
    std::string patch;
    for (int i = 0; i < wave_sizes.size(); ++i) {
      std::string main_body_copy = main_body;
      ABSL_RETURN_IF_ERROR(ResolveWaveMemoryToWaveBroadcast(
          wave_sizes[i], gpu_info, &main_body_copy));
      if (i == wave_sizes.size() - 1) {
        patch +=
            "  if (ucl::GetSubGroupSize() >= " + std::to_string(wave_sizes[i]) +
            "u) {\n";
      } else {
        patch +=
            "  if (ucl::GetSubGroupSize() == " + std::to_string(wave_sizes[i]) +
            "u) {\n";
      }
      patch += main_body_copy;
      patch += "  }\n";
    }
    code->replace(scope_start, scope_end - scope_start - 1, patch);
  } else {
    int wave_size = 8;
    if (wave_sizes.size() == 1) {
      wave_size = wave_sizes[0];
    } else {
      if (gpu_info.IsMali()) {
        wave_size = 16;
      }
      if (gpu_info.IsPowerVR()) {
        wave_size = 128;
      }
    }
    if (gpu_info.IsIntel() && gpu_info.IsApiOpenCl() &&
        gpu_info.opencl_info.IsCLVK()) {
      wave_size = wave_sizes.empty() ? 16 : wave_sizes[0];
    }
    if (gpu_info.IsIntel() && gpu_info.IsApiOpenCl() &&
        !gpu_info.opencl_info.IsCLVK()) {
      const std::string intel_wave_size = "intel_reqd_sub_group_size";
      size_t pos = code->find(intel_wave_size + "(");
      if (pos != std::string::npos) {
        pos += intel_wave_size.size() + 1;
        wave_size = std::stoi(std::string(GetNextWord(*code, pos)));
      } else {
        wave_size = wave_sizes.empty() ? 32 : wave_sizes[0];
        const size_t main_pos = code->find("MAIN_FUNCTION");
        code->insert(main_pos, "__attribute__((intel_reqd_sub_group_size(" +
                                   std::to_string(wave_size) + ")))\n");
        code->insert(
            0,
            "#pragma OPENCL EXTENSION cl_intel_required_subgroup_size: "
            "enable\n");
      }
    }
    ABSL_RETURN_IF_ERROR(
        ResolveWaveMemoryToWaveBroadcast(wave_size, gpu_info, code));
  }
  compiler_options->push_back(CompilerOptions::kCl20);
  return absl::OkStatus();
}

absl::Status ResolveWaveMemoryToWorkGroupMemory(const GpuInfo& gpu_info,
                                                std::string* code) {
  const std::string ext_declaration =
      "#pragma OPENCL EXTENSION ucl_wave_memory: enable";
  const size_t ext_pos = code->find(ext_declaration);
  if (ext_pos != std::string::npos) {
    code->erase(ext_pos, ext_declaration.size());
  } else {
    return absl::OkStatus();
  }
  const bool async_work_group_copy =
      gpu_info.IsPowerVR() && gpu_info.IsApiOpenCl();
  const std::string mem_qualifier = "__wave";
  const size_t mem_pos = code->find(mem_qualifier);
  DataType data_type;
  int vector_size;
  if (mem_pos != std::string::npos) {
    size_t pos = mem_pos + mem_qualifier.size() + 1;
    std::string mem_type = std::string(GetNextWord(*code, pos));
    pos += mem_type.size() + 1;
    std::string mem_name = std::string(GetNextWord(*code, pos));
    pos += mem_name.size() + 1;
    int array_size = std::stoi(std::string(GetNextWord(*code, pos)));
    ABSL_RETURN_IF_ERROR(
        DataTypeFromTemplateArg(mem_type, &data_type, &vector_size));
    std::string patch = "  __local " + mem_type + " " + mem_name + "[" +
                        std::to_string(array_size) + "];";
    if (!async_work_group_copy) {
      patch +=
          "  int wave_memory_id = (ucl::GetLocalId<2>() * "
          "ucl::GetGroupSize<1>() "
          "+ ucl::GetLocalId<1>()) * ucl::GetGroupSize<0>() + "
          "ucl::GetLocalId<0>();\n";
      patch +=
          "  int wave_size = ucl::GetGroupSize<0>() * "
          "ucl::GetGroupSize<1>() * ucl::GetGroupSize<2>();\n";
    }
    const size_t end_line = code->find('\n', mem_pos);
    code->replace(mem_pos, end_line - mem_pos, patch);
  }

  // ucl::WaveLoad(wave_mem, buffer, offset, count);
  const std::string upload_func_name = "ucl::WaveLoad";
  size_t upload_pos = code->find(upload_func_name);
  while (upload_pos != std::string::npos) {
    size_t pos = upload_pos + upload_func_name.size();
    std::vector<std::string> args;
    size_t close_bracket_pos;
    ABSL_RETURN_IF_ERROR(ParseArguments(*code, pos, &close_bracket_pos, &args));
    std::string patch;
    if (async_work_group_copy) {
      // async_work_group_copy(wave_mem, buffer + offset, count, 0);
      patch = absl::StrCat("async_work_group_copy(" + args[0] + ", ", args[1],
                           " + ", args[2], ", ", args[3], ", 0)");
    } else {
      std::string src_val =
          absl::StrCat(args[1], "[", args[2], " + tmp_offset]");
      if (data_type == DataType::kFloat16 && gpu_info.IsGlsl() &&
          !gpu_info.IsGlslSupportsExplicitFp16()) {
        src_val = absl::StrCat("ucl::Reinterpret<uint2, half4>(", src_val, ")");
      }
      patch += "  for (int i = 0; i < " + args[3] + "; i += wave_size) {\n";
      patch += "    int tmp_offset = i + wave_memory_id;\n";
      patch += "    if (tmp_offset < " + args[3] + ") {\n";
      patch += "      " + args[0] + "[tmp_offset] = " + src_val + ";\n";
      patch += "    }\n";
      patch += "  }\n";
    }
    code->replace(upload_pos, close_bracket_pos - upload_pos + 1, patch);
    upload_pos = code->find(upload_func_name, upload_pos);
  }
  const std::string sync_func_name = "ucl::SyncThreads<WaveLoad>()";
  size_t sync_pos = code->find(sync_func_name);
  while (sync_pos != std::string::npos) {
    const std::string patch = R"(ucl::SyncThreads<WorkGroup, Local>();)";
    const size_t end_line = code->find('\n', sync_pos);
    code->replace(sync_pos, end_line - sync_pos, patch);
    sync_pos = code->find(sync_func_name, sync_pos);
  }
  return absl::OkStatus();
}
}  // namespace

BufferDescriptor GetBufferDescForWaveMemoryUpload(const GpuInfo& gpu_info,
                                                  DataType data_type) {
  BufferDescriptor buffer_desc;
  buffer_desc.memory_type = MemoryType::kGlobal;
  buffer_desc.element_type = data_type;
  buffer_desc.element_size = 4;
  return buffer_desc;
}

BufferDescriptor GetBufferDescForWaveMemoryUpload(
    const GpuInfo& gpu_info, const WeightsDescription& weights_desc) {
  return GetBufferDescForWaveMemoryUpload(gpu_info, weights_desc.type);
}

BufferDescriptor GetBufferDescForWaveMemoryUpload(
    const GpuInfo& gpu_info, const WeightsDescription& weights_desc,
    const OHWI& weights_shape) {
  BufferDescriptor buffer_desc =
      GetBufferDescForWaveMemoryUpload(gpu_info, weights_desc);
  buffer_desc.size =
      GetTotalElementsCountForLayout(weights_desc, weights_shape) *
      SizeOf(weights_desc.type);
  buffer_desc.data.resize(buffer_desc.size);
  return buffer_desc;
}

absl::Status ResolveWaveMemory(const GpuInfo& gpu_info, std::string* code,
                               Arguments* args,
                               std::vector<CompilerOptions>* compiler_options) {
  if (!gpu_info.SupportsExtension("ucl_wave_memory")) {
    return absl::OkStatus();
  }
  bool use_work_group_memory =
      gpu_info.IsPowerVR() && !gpu_info.powervr_info.IsImgDxx();
  if (gpu_info.IsPowerVR() && gpu_info.powervr_info.IsImgCxx() &&
      gpu_info.SupportsExtension("cl_img_pixel_subgroup_dot") &&
      code->find("__builtin_PXL_dot") != std::string::npos) {
    use_work_group_memory = false;
  }
  if (use_work_group_memory) {
    return ResolveWaveMemoryToWorkGroupMemory(gpu_info, code);
  } else {
    return ResolveWaveMemoryToWaveBroadcast(gpu_info, code,
                                            compiler_options);
  }
}

}  // namespace ml_drift
