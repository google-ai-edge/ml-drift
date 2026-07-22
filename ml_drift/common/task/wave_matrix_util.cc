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

#include "ml_drift/common/task/wave_matrix_util.h"

#include <cstddef>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/ascii.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/util.h"

namespace ml_drift {
namespace {
bool IsWordSymbol(char symbol) {
  return absl::ascii_isalnum(symbol) || symbol == '_';
}

struct MatrixDesc {
  std::string matrix_type;
  DataType data_type;
  std::string rows;
  std::string cols;
};

std::string GetWebGpuSubgroupMatrixType(const MatrixDesc& matrix_desc) {
  std::string matrix_type;
  if (matrix_desc.matrix_type == "left") {
    matrix_type = "subgroup_matrix_left";
  } else if (matrix_desc.matrix_type == "right") {
    matrix_type = "subgroup_matrix_right";
  } else if (matrix_desc.matrix_type == "accum") {
    matrix_type = "subgroup_matrix_result";
  }
  const int type_size_in_bits = SizeInBitsOf(matrix_desc.data_type);
  const std::string type = IsFloatType(matrix_desc.data_type)
                               ? "f"
                               : (IsSigned(matrix_desc.data_type) ? "i" : "u");
  return matrix_type + "<" + type + std::to_string(type_size_in_bits) + ", " +
         matrix_desc.cols + ", " + matrix_desc.rows + ">";
}
}  // namespace

absl::Status ResolveWaveMatrix(const GpuInfo& gpu_info, std::string* code) {
  const std::string ext_declaration =
      "#pragma OPENCL EXTENSION ucl_wave_matrix: enable";
  const size_t ext_pos = code->find(ext_declaration);
  if (ext_pos != std::string::npos) {
    if (gpu_info.IsApiMetal()) {
      code->erase(ext_pos, ext_declaration.size());
    } else if (gpu_info.IsGlsl()) {
      if (gpu_info.SupportsExtension("VK_KHR_cooperative_matrix")) {
        const std::string patch =
            R"(#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_KHR_cooperative_matrix : require)";
        code->replace(ext_pos, ext_declaration.size(), patch);
      } else if (gpu_info.SupportsExtension("VK_NV_cooperative_matrix")) {
        const std::string patch =
            R"(#extension GL_KHR_memory_scope_semantics : require
#extension GL_NV_cooperative_matrix : require)";
        code->replace(ext_pos, ext_declaration.size(), patch);
      }
    } else if (gpu_info.IsApiWebGpu()) {
      const std::string patch =
          "enable chromium_experimental_subgroup_matrix;\n";
      code->replace(ext_pos, ext_declaration.size(), patch);
    }
  } else {
    return absl::OkStatus();
  }

  // ucl::wave_matrix<matrix_type, data_type, rows, cols>
  const std::string decl_name = "ucl::wave_matrix";
  size_t declaration_pos = code->find(decl_name);
  absl::flat_hash_map<std::string, MatrixDesc> matrix_descs;
  while (declaration_pos != std::string::npos) {
    size_t pos = declaration_pos + decl_name.size();
    std::vector<std::string> args;
    size_t close_bracket_pos;
    ABSL_RETURN_IF_ERROR(ParseArguments(*code, pos, &close_bracket_pos, &args));
    if (args.size() != 4) {
      return absl::InvalidArgumentError(
          "Expected 4 arguments in ucl::wave_matrix<matrix_type, data_type, "
          "rows, cols>. Matrix type can be {left, right, accum}.");
    }
    std::string patch;
    if (gpu_info.IsApiMetal()) {
      patch =
          "simdgroup_matrix<" + args[1] + ", " + args[2] + ", " + args[3] + ">";
    } else if (gpu_info.IsGlsl()) {
      DataType data_type;
      int vector_size;
      ABSL_RETURN_IF_ERROR(
          DataTypeFromTemplateArg(args[1], &data_type, &vector_size));
      if (gpu_info.SupportsExtension("VK_KHR_cooperative_matrix")) {
        std::string usage;
        if (args[0] == "left") {
          usage = "gl_MatrixUseA";
        } else if (args[0] == "right") {
          usage = "gl_MatrixUseB";
        } else if (args[0] == "accum") {
          usage = "gl_MatrixUseAccumulator";
        } else {
          return absl::InvalidArgumentError(
              "Expected matrix type to be {left, right, accum}.");
        }
        const std::string type = IsFloatType(data_type)
                                     ? "float"
                                     : (IsSigned(data_type) ? "int" : "uint");
        patch = "coopmat<" + type + std::to_string(SizeInBitsOf(data_type)) +
                "_t, gl_ScopeSubgroup, " + args[2] + ", " + args[3] + ", " +
                usage + ">";
      } else if (gpu_info.SupportsExtension("VK_NV_cooperative_matrix")) {
        const std::string type =
            IsFloatType(data_type) ? "f" : (IsSigned(data_type) ? "i" : "u");
        patch = type + "coopmatNV<" + std::to_string(SizeInBitsOf(data_type)) +
                ", gl_ScopeSubgroup, " + args[2] + ", " + args[3] + ">";
      }
    } else if (gpu_info.IsApiWebGpu()) {
      DataType data_type;
      int vector_size;
      ABSL_RETURN_IF_ERROR(
          DataTypeFromTemplateArg(args[1], &data_type, &vector_size));
      const MatrixDesc matrix_desc = {args[0], data_type, args[2], args[3]};
      patch = GetWebGpuSubgroupMatrixType(matrix_desc);
      size_t name_pos = close_bracket_pos + 2;
      if ((*code)[close_bracket_pos + 1] == ' ' &&
          IsWordSymbol((*code)[name_pos])) {
        size_t end = name_pos;
        for (; end < code->size() && IsWordSymbol((*code)[end]); ++end) {
        }
        const std::string matrix_name = code->substr(name_pos, end - name_pos);
        matrix_descs[matrix_name] = {args[0], data_type, args[2], args[3]};
        close_bracket_pos = end - 1;
        patch = "var " + matrix_name + ": " + patch;
      }
    }
    code->replace(declaration_pos, close_bracket_pos - declaration_pos + 1,
                  patch);
    declaration_pos = code->find(decl_name, declaration_pos);
  }

  // ucl::WaveMatrixLoad(matrix, ptr, offset, stride, row_major = true);
  // TODO(sorokin): change it to matrix.Load(ptr, offset, row_major = true);
  const std::string load_func_name = "ucl::WaveMatrixLoad";
  size_t load_pos = code->find(load_func_name);
  while (load_pos != std::string::npos) {
    size_t pos = load_pos + load_func_name.size();
    std::vector<std::string> args;
    size_t close_bracket_pos;
    ABSL_RETURN_IF_ERROR(ParseArguments(*code, pos, &close_bracket_pos, &args));
    if (args.size() != 4 && args.size() != 5) {
      return absl::InvalidArgumentError(
          "Expected 4 or 5 arguments in ucl::WaveMatrixLoad(dst_matrix, "
          "src_ptr, "
          "src_offset, stride, (optional)row_major = true).");
    }
    std::string patch;
    const bool row_major =
        args.size() == 5 && args[4] == "false" ? false : true;
    if (gpu_info.IsApiMetal()) {
      const std::string transpose_matrix = row_major ? "false" : "true";
      patch = "simdgroup_load(" + args[0] + ", " + args[1] + " + " + args[2] +
              ", " + args[3] + ", 0, " + transpose_matrix + ")";
    } else if (gpu_info.IsGlsl()) {
      if (gpu_info.SupportsExtension("VK_KHR_cooperative_matrix")) {
        const std::string layout =
            row_major ? "gl_CooperativeMatrixLayoutRowMajor"
                      : "gl_CooperativeMatrixLayoutColumnMajor";
        patch = "coopMatLoad(" + args[0] + ", " + args[1] + ", " + args[2] +
                ", " + args[3] + ", " + layout + ")";
      } else if (gpu_info.SupportsExtension("VK_NV_cooperative_matrix")) {
        const std::string column_major = row_major ? "false" : "true";
        patch = "coopMatLoadNV(" + args[0] + ", " + args[1] + ", " + args[2] +
                ", " + args[3] + ", " + column_major + ")";
      }
    } else if (gpu_info.IsApiWebGpu()) {
      const MatrixDesc& matrix_desc = matrix_descs[args[0]];
      const std::string column_major = row_major ? "false" : "true";
      patch = args[0] + " = subgroupMatrixLoad<" +
              GetWebGpuSubgroupMatrixType(matrix_desc) + ">(&" + args[1] +
              ", u32(" + args[2] + "), " + column_major + ", u32(" + args[3] +
              "))";
    }
    code->replace(load_pos, close_bracket_pos - load_pos + 1, patch);
    load_pos = code->find(load_func_name, load_pos);
  }

  // ucl::WaveMatrixMAC(C, A, B);
  // TODO(sorokin): change it to C.AccumulateProduct(A, B);
  const std::string mac_func_name = "ucl::WaveMatrixMAC";
  size_t mac_pos = code->find(mac_func_name);
  while (mac_pos != std::string::npos) {
    size_t pos = mac_pos + mac_func_name.size();
    std::vector<std::string> args;
    size_t close_bracket_pos;
    ABSL_RETURN_IF_ERROR(ParseArguments(*code, pos, &close_bracket_pos, &args));
    if (args.size() != 3) {
      return absl::InvalidArgumentError(
          "Expected 3 arguments in ucl::WaveMatrixMAC(C, A, B).");
    }
    std::string patch;
    if (gpu_info.IsApiMetal()) {
      patch = "simdgroup_multiply_accumulate(" + args[0] + ", " + args[1] +
              ", " + args[2] + ", " + args[0] + ")";
    } else if (gpu_info.IsGlsl()) {
      if (gpu_info.SupportsExtension("VK_KHR_cooperative_matrix")) {
        patch = args[0] + " = coopMatMulAdd(" + args[1] + ", " + args[2] +
                ", " + args[0] + ")";
      } else if (gpu_info.SupportsExtension("VK_NV_cooperative_matrix")) {
        patch = args[0] + " = coopMatMulAddNV(" + args[1] + ", " + args[2] +
                ", " + args[0] + ")";
      }
    } else if (gpu_info.IsApiWebGpu()) {
      patch = args[0] + " = subgroupMatrixMultiplyAccumulate(" + args[1] +
              ", " + args[2] + ", " + args[0] + ")";
    }
    code->replace(mac_pos, close_bracket_pos - mac_pos + 1, patch);
    mac_pos = code->find(mac_func_name, mac_pos);
  }

  // ucl::WaveMatrixStore(matrix, ptr, offset, stride);
  // TODO(sorokin): change it to matrix.Store(ptr, offset);
  const std::string store_func_name = "ucl::WaveMatrixStore";
  size_t store_pos = code->find(store_func_name);
  while (store_pos != std::string::npos) {
    size_t pos = store_pos + store_func_name.size();
    std::vector<std::string> args;
    size_t close_bracket_pos;
    ABSL_RETURN_IF_ERROR(ParseArguments(*code, pos, &close_bracket_pos, &args));
    if (args.size() != 4) {
      return absl::InvalidArgumentError(
          "Expected 4 arguments in ucl::WaveMatrixStore(src_matrix, dst_ptr, "
          "dst_offset, stride).");
    }
    std::string patch;
    if (gpu_info.IsApiMetal()) {
      patch = "simdgroup_store(" + args[0] + ", " + args[1] + " + " + args[2] +
              ", " + args[3] + ")";
    } else if (gpu_info.IsGlsl()) {
      if (gpu_info.SupportsExtension("VK_KHR_cooperative_matrix")) {
        patch = "coopMatStore(" + args[0] + ", " + args[1] + ", " + args[2] +
                ", " + args[3] + ", gl_CooperativeMatrixLayoutRowMajor)";
      } else if (gpu_info.SupportsExtension("VK_NV_cooperative_matrix")) {
        patch = "coopMatStoreNV(" + args[0] + ", " + args[1] + ", " + args[2] +
                ", " + args[3] + ", false)";
      }
    } else if (gpu_info.IsApiWebGpu()) {
      patch = "subgroupMatrixStore(&" + args[1] + ", u32(" + args[2] + "), " +
              args[0] + ", false, u32(" + args[3] + "))";
    }
    code->replace(store_pos, close_bracket_pos - store_pos + 1, patch);
    store_pos = code->find(store_func_name, store_pos);
  }

  return absl::OkStatus();
}

}  // namespace ml_drift
