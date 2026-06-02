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

#include "ml_drift/common/kernels/elementwise.h"

#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

namespace {
std::string ErfFunction(const GpuInfo& gpu_info) {
  if (gpu_info.IsApiOpenCl() &&
      !(gpu_info.IsAdreno() || gpu_info.IsMali() || gpu_info.IsPowerVR())) {
    // Adreno/Mali/PowerVR faster with 'manual' implementation that uses native
    // functions
    return "$1 = erf($0);\n";
  } else {
    // https://en.wikipedia.org/wiki/Error_function#Approximation_with_elementary_functions
    std::string r = R"(
  { // erf approximation
    Type sign_val = sign($0);
    Type abs_val = fabs($0);
    SType a1 = ucl::Init<SType>(0.254829592f);
    SType a2 = ucl::Init<SType>(-0.284496736f);
    SType a3 = ucl::Init<SType>(1.421413741f);
    SType a4 = ucl::Init<SType>(-1.453152027f);
    SType a5 = ucl::Init<SType>(1.061405429f);
    SType p  = ucl::Init<SType>(0.3275911f);
    Type t = ucl::Init<SType>(1.0f) / (ucl::Init<SType>(1.0f) + p * abs_val);
    Type y = ucl::Init<SType>(1.0f) - ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t*ucl::Exp<Type>(-abs_val*abs_val);
    $1 = y * sign_val;
  }
)";
    return r;
  }
}

std::string ClampedTanh(const GpuInfo& gpu_info, const std::string& src_name,
                        const std::string& dst_name) {
  // On Adreno non native functions can be very slow.
  // On Mali native not much faster, but they use less instructions than the
  // builtin. Using non native increase chance that kernel will use more regs
  // and will cause register spilling. On Mali more preferable in general to
  // use manual tanh calculation instead of builtin, no matter if it uses inside
  // exp or native_exp.
  const bool use_native_opencl_functions =
      gpu_info.IsApiOpenCl() && (gpu_info.IsAdreno() || gpu_info.IsMali());
  // Some GPUs overflow the builtin tanh() on large input values and return NaN.
  // This function clamps the input to [-10, 10] before computing tanh() to
  // avoid the overflow.
  std::string code =
      "float4 in_fp32 = clamp(src_value, ucl::Init<float4>(-10.0f),"
      " ucl::Init<float4>(10.0f));\n";
  if (use_native_opencl_functions) {
    code += "float4 exp_val = native_exp(2.0f * in_fp32);\n";
    code +=
        "dst_value = (exp_val - ucl::Init<float4>(1.0f)) /"
        " (exp_val + ucl::Init<float4>(1.0f));\n";
  } else {
    code += "dst_value = tanh(in_fp32);\n";
  }
  absl::StrReplaceAll({{"src_value", src_name}, {"dst_value", dst_name}},
                      &code);
  return code;
}

std::string PowUsingNativePowr(const std::string& x_name,
                               const std::string& y_name,
                               const std::string& dst_name) {
  std::string c;
  for (int i = 0; i < 4; ++i) {
    const std::string postfixes[4] = {".x", ".y", ".z", ".w"};
    const std::string r_var = dst_name + postfixes[i];
    const std::string x_var = x_name + postfixes[i];
    const std::string y_var = y_name + postfixes[i];
    c += absl::Substitute(R"(
    if ($0 >= 0.0f) { $2 = native_powr($0, $1); }
    else {
      int iy = convert_int($1);
      if (iy % 2 == 0) {
        $2 = native_powr(-$0, $1);
      } else {
        $2 = -native_powr(-$0, $1);
      }
    })",
                          x_var, y_var, r_var);
  }
  return c;
}

std::string GetOneInputCode(const GpuInfo& gpu_info,
                            const OperationType& op_type,
                            const DataType& data_type,
                            const std::string& input_value,
                            const std::string& output_value) {
  const bool use_native_opencl_functions = gpu_info.IsApiOpenCl() &&
                                           data_type != DataType::FLOAT32 &&
                                           gpu_info.IsAdreno();
  const bool float_type = IsFloatType(data_type);
  std::string result;
  switch (op_type) {
    case OperationType::ABS:
      if (data_type == DataType::FLOAT16 || data_type == DataType::FLOAT32) {
        result = "$0 = fabs($1);";
      } else {
        result =
            "$0 = ucl::Convert<" + ToUclDataType(data_type, 4) + ">(abs($1));";
      }
      break;
    case OperationType::CEIL:
      result = "$0 = ceil($1);";
      break;
    case OperationType::COS:
      if (use_native_opencl_functions && float_type) {
        result =
            "$0 = ucl::Convert<Type>(native_cos(ucl::Convert<float4>($1)));";
      } else if (!float_type) {
        result =
            "$0 = ucl::Convert<Type>(round(cos(ucl::Convert<float4>($1))));";
      } else {
        result = "$0 = cos($1);";
      }
      break;
    case OperationType::COPY:
      result = "$0 = $1;";
      break;
    case OperationType::ELU:
      if (gpu_info.IsApiOpenCl()) {
        result = R"(
$0.x = $1.x < ucl::Init<SType>(0.0f) ? expm1($1.x) : $1.x;
$0.y = $1.y < ucl::Init<SType>(0.0f) ? expm1($1.y) : $1.y;
$0.z = $1.z < ucl::Init<SType>(0.0f) ? expm1($1.z) : $1.z;
$0.w = $1.w < ucl::Init<SType>(0.0f) ? expm1($1.w) : $1.w;)";
      } else {
        result = R"(
$0.x = $1.x < ucl::Init<SType>(0.0f) ? exp($1.x) - ucl::Init<SType>(1.0f) : $1.x;
$0.y = $1.y < ucl::Init<SType>(0.0f) ? exp($1.y) - ucl::Init<SType>(1.0f) : $1.y;
$0.z = $1.z < ucl::Init<SType>(0.0f) ? exp($1.z) - ucl::Init<SType>(1.0f) : $1.z;
$0.w = $1.w < ucl::Init<SType>(0.0f) ? exp($1.w) - ucl::Init<SType>(1.0f) : $1.w;)";
      }
      break;
    case OperationType::EXP:
      result = "$0 = ucl::Exp<Type>($1);";
      break;
    case OperationType::FLOOR:
      result = "$0 = floor($1);";
      break;
    case OperationType::GELU:
      // gelu(x) = 0.5 * x * (1 + erf(x/sqrt(2)))
      result = "  Type erf_in = $1 * ucl::Init<Type>(0.707106781186548f);\n";
      result += "  Type erf_out;\n";
      result += absl::Substitute(ErfFunction(gpu_info), "erf_in", "erf_out");
      result +=
          "  $0 = ucl::Init<Type>(0.5f) * $1 * (ucl::Init<Type>(1.0f) + "
          "erf_out);\n";
      break;
    case OperationType::GELU_TANH_APPROX:
      // Approximated GELU using tanh, per https://arxiv.org/abs/1606.08415v5
      // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
      // The real version caused NaNs on Apple GPUs. This was likely due to exp
      // overflowing for large x when computing tanh = (exp(x) - 1) / (exp(x) +
      // 1).
      // For similar reasons, exp(x) is handled separately for positive and
      // negative values.
      // See b/383769042 for details.
      result += "float4 src_f32 = ucl::Convert<float4>($1);\n";
      result += "float4 tanh_val;\n";
      result +=
          "float4 tanh_in = 0.7978845608f * (src_f32 + 0.044715f * src_f32 * "
          "src_f32 * src_f32);\n";
      result += ClampedTanh(gpu_info, "tanh_in", "tanh_val");

      result += "$0 = ucl::Convert<Type>(0.5f * src_f32 * (1.0f + tanh_val));";
      break;
    case OperationType::HARD_SWISH:
      result =
          "$0 = $1 * clamp($1 * ucl::Init<Type>(0.16666667f) + "
          "ucl::Init<Type>(0.5f), "
          "ucl::Init<Type>(0.0f), "
          "ucl::Init<Type>(1.0f));";
      break;
    case OperationType::LOG:
      if (use_native_opencl_functions) {
        result =
            "$0 = ucl::Convert<Type>(native_log(ucl::Convert<float4>($1)));";
      } else {
        result = "$0 = log($1);";
      }
      break;
    case OperationType::LOGICAL_NOT:
      if (gpu_info.IsApiOpenCl() && data_type == DataType::BOOL) {
        result = "$0.x = $1.x ? 0u : 1u;\n";
        result += "$0.y = $1.y ? 0u : 1u;\n";
        result += "$0.z = $1.z ? 0u : 1u;\n";
        result += "$0.w = $1.w ? 0u : 1u;\n";
      } else if (data_type == DataType::INT8 || data_type == DataType::INT16 ||
                 data_type == DataType::INT32) {
        result = "$0.x = ~$1.x;\n";
        result += "$0.y = ~$1.y;\n";
        result += "$0.z = ~$1.z;\n";
        result += "$0.w = ~$1.w;\n";
      } else {
        result = "$0.x = !$1.x;\n";
        result += "$0.y = !$1.y;\n";
        result += "$0.z = !$1.z;\n";
        result += "$0.w = !$1.w;\n";
      }
      break;
    case OperationType::NEG:
      result = "$0 = -($1);";
      break;
    case OperationType::ROUND:
      if (gpu_info.IsApiOpenCl() || gpu_info.IsApiMetal()) {
        result = "$0 = rint($1);";
      } else {
        result = "$0 = round($1);";
      }
      break;
    case OperationType::RSQRT:
      if (use_native_opencl_functions) {
        result =
            "$0 = ucl::Convert<Type>(native_rsqrt(ucl::Convert<float4>($1)));";
      } else {
        result = "$0 = rsqrt($1);";
      }
      break;
    case OperationType::SIGMOID:
      if (use_native_opencl_functions) {
        result =
            "$0 = ucl::Convert<Type>(native_recip(1.0f + "
            "native_exp(ucl::Convert<float4>(-$1))));";
      } else {
        result =
            "$0 = ucl::Init<Type>(1.0f) / (ucl::Init<Type>(1.0f) + "
            "exp(-($1)));";
      }
      break;
    case OperationType::SIGN:
      if (data_type == DataType::FLOAT16 || data_type == DataType::FLOAT32) {
        result = "$0 = sign($1);";
      } else {  // int
        if (gpu_info.IsApiWebGpu()) {
          result = "$0.x = clamp($1.x, -1, 1);\n";
          result += "$0.y = clamp($1.y, -1, 1);\n";
          result += "$0.z = clamp($1.z, -1, 1);\n";
          result += "$0.w = clamp($1.w, -1, 1);\n";
        } else {
          result = "$0 = clamp($1, -1, 1);";
        }
      }
      break;
    case OperationType::SIN:
      if (use_native_opencl_functions && float_type) {
        result =
            "$0 = ucl::Convert<Type>(native_sin(ucl::Convert<float4>($1)));";
      } else if (!float_type) {
        result =
            "$0 = ucl::Convert<Type>(round(sin(ucl::Convert<float4>($1))));";
      } else {
        result = "$0 = sin($1);";
      }
      break;
    case OperationType::SQRT:
      if (use_native_opencl_functions) {
        result =
            "$0 = ucl::Convert<Type>(native_sqrt(ucl::Convert<float4>($1)));";
      } else {
        result = "$0 = sqrt($1);";
      }
      break;
    case OperationType::SQUARE:
      result = "$0 = $1 * $1;";
      break;
    case OperationType::TANH:
      result = "float4 tanh_out;\n";
      result += "float4 tanh_in = ucl::Convert<float4>($1);\n";
      result += ClampedTanh(gpu_info, "tanh_in", "tanh_out");
      result += "$0 = ucl::Convert<Type>(tanh_out);\n";
      break;
    case OperationType::MISH:
      // Mish: x -> x tanh(log(1+exp(x)))
      // The exp may overflow, so the approximation mish(x) ~= x is used for
      // large x.

      // Outside of these bounds, the approximation error is within 2.5 ULP
      // which is what Vulkan and OpenCL guarantees for divisions.
      if (data_type == DataType::FLOAT32) {
        result = "SType large_boundary = ucl::Init<SType>(8);";
      } else {
        result = "SType large_boundary = ucl::Init<SType>(3.7);";
      }
      // We can cancel the log and the exp in the definition of tanh and
      // massage the expression a bit to get something that handles underflow
      // (the 1+exp(x) = 1 case) well.
      result += R"(
          Type expx = ucl::Exp<Type>($1);
          Type a = ucl::Init<Type>(1.0f) + expx;
          Type mish = $1 * (expx * (ucl::Init<Type>(1.0f) + a)) / (ucl::Init<Type>(1.0f) + a * a);
          $0.x = $1.x > large_boundary ? $1.x : mish.x;
          $0.y = $1.y > large_boundary ? $1.y : mish.y;
          $0.z = $1.z > large_boundary ? $1.z : mish.z;
          $0.w = $1.w > large_boundary ? $1.w : mish.w;
          )";
      break;
    default:
      return "Unknown operation type;";
  }
  absl::StrReplaceAll({{"SType", ToUclDataType(data_type, 1)},
                       {"Type", ToUclDataType(data_type, 4)}},
                      &result);
  return absl::Substitute(result, output_value, input_value);
}

std::string GetTwoInputCode(const GpuInfo& gpu_info,
                            const OperationType& op_type,
                            const DataType& data_type,
                            const std::string& result_var,
                            const std::string& input0,
                            const std::string& input1,
                            bool swap_inputs = false) {
  const bool use_native_opencl_functions = gpu_info.IsApiOpenCl() &&
                                           data_type != DataType::FLOAT32 &&
                                           gpu_info.IsAdreno();
  std::string result;
  switch (op_type) {
    case OperationType::ADD:
      result += "$0 = $1 + $2;";
      break;
    case OperationType::ATAN2:
      if (IsFloatType(data_type)) {
        result += "$0 = atan2($1, $2);";
      } else {
        result +=
            "$0 = ucl::Convert<" + ToUclDataType(data_type, 4) + ">(round(" +
            "atan2(ucl::Convert<float4>($1), ucl::Convert<float4>($2))));";
      }

      break;
    case OperationType::DIV:
      result += "$0 = $1 / $2;";
      break;
    case OperationType::FLOOR_DIV:
      if (data_type == DataType::FLOAT16 || data_type == DataType::FLOAT32 ||
          data_type == DataType::FLOAT64 || data_type == DataType::BFLOAT16) {
        result = "$0 = floor($1 / $2);";
      } else {
        const std::string postfixes[4] = {".x", ".y", ".z", ".w"};
        for (int i = 0; i < 4; ++i) {
          std::string code = "C = A / B;\n";
          // C * B != A is equivalent to A % B != 0, but on nvidia in glsl for
          // negative numbers % doesn't work properly?.
          code += "if (C * B != A && ((A < 0) != (B < 0))) { C -= 1;}\n";
          result += absl::StrReplaceAll(code, {{"A", "$1" + postfixes[i]},
                                               {"B", "$2" + postfixes[i]},
                                               {"C", "$0" + postfixes[i]}});
        }
      }
      break;
    case OperationType::FLOOR_MOD:
      if (data_type == DataType::FLOAT16 || data_type == DataType::FLOAT32 ||
          data_type == DataType::FLOAT64 || data_type == DataType::BFLOAT16) {
        result = "$0 = $1 - floor($1 / $2) * $2;";
      } else {
        const std::string postfixes[4] = {".x", ".y", ".z", ".w"};
        for (int i = 0; i < 4; ++i) {
          // on nvidia in glsl for negative numbers % doesn't work properly?.
          std::string code = "C = A - B * (A / B);\n";
          code += "if ((C != 0) && ((A < 0) != (B < 0))) { C += B;}\n";
          result += absl::StrReplaceAll(code, {{"A", "$1" + postfixes[i]},
                                               {"B", "$2" + postfixes[i]},
                                               {"C", "$0" + postfixes[i]}});
        }
      }
      break;
    case OperationType::MAXIMUM:
      result += "$0 = max($1, $2);";
      break;
    case OperationType::MINIMUM:
      result += "$0 = min($1, $2);";
      break;
    case OperationType::MOD:
      result += "$0 = $1 % $2;";
      break;
    case OperationType::MUL:
      result += "$0 = $1 * $2;";
      break;
    case OperationType::POW:
      if (use_native_opencl_functions) {
        result = "float4 out_f32;\n";
        result += "float4 in_f32_x = ucl::Convert<float4>($1);\n";
        result += "float4 in_f32_y = ucl::Convert<float4>($2);\n";
        result += PowUsingNativePowr("in_f32_x", "in_f32_y", "out_f32");
        result += "$0 = ucl::Convert<" + ToUclDataType(data_type, 4) +
                  ">(out_f32);\n";
      } else {
        result += "$0 = pow($1, $2);";
      }
      break;
    case OperationType::SQUARED_DIFF:
      result += "$0 = ($1 - $2) * ($1 - $2);";
      break;
    case OperationType::SUB:
      result += "$0 = $1 - $2;";
      break;
    // Comparison operators
    case OperationType::LESS:
      result = "$0.x = $1.x < $2.x;\n";
      result += "$0.y = $1.y < $2.y;\n";
      result += "$0.z = $1.z < $2.z;\n";
      result += "$0.w = $1.w < $2.w;";
      break;
    case OperationType::LESS_EQUAL:
      result = "$0.x = $1.x <= $2.x;\n";
      result += "$0.y = $1.y <= $2.y;\n";
      result += "$0.z = $1.z <= $2.z;\n";
      result += "$0.w = $1.w <= $2.w;";
      break;
    case OperationType::LOGICAL_AND:
      if (data_type == DataType::BOOL) {
        result = "$0.x = $1.x && $2.x;\n";
        result += "$0.y = $1.y && $2.y;\n";
        result += "$0.z = $1.z && $2.z;\n";
        result += "$0.w = $1.w && $2.w;\n";
      } else {
        result = "$0.x = $1.x & $2.x;\n";
        result += "$0.y = $1.y & $2.y;\n";
        result += "$0.z = $1.z & $2.z;\n";
        result += "$0.w = $1.w & $2.w;\n";
      }

      break;
    case OperationType::LOGICAL_OR:
      if (data_type == DataType::BOOL) {
        result = "$0.x = $1.x || $2.x;\n";
        result += "$0.y = $1.y || $2.y;\n";
        result += "$0.z = $1.z || $2.z;\n";
        result += "$0.w = $1.w || $2.w;\n";
      } else {
        result = "$0.x = $1.x | $2.x;\n";
        result += "$0.y = $1.y | $2.y;\n";
        result += "$0.z = $1.z | $2.z;\n";
        result += "$0.w = $1.w | $2.w;\n";
      }
      break;
    case OperationType::LOGICAL_XOR:
      if (gpu_info.IsApiOpenCl() ||
          (!gpu_info.IsApiOpenCl() && data_type != DataType::BOOL)) {
        result = "$0.x = $1.x ^ $2.x;\n";
        result += "$0.y = $1.y ^ $2.y;\n";
        result += "$0.z = $1.z ^ $2.z;\n";
        result += "$0.w = $1.w ^ $2.w;\n";
      } else {
        result = "$0.x = ($1.x || $2.x) && !($1.x && $2.x);\n";
        result += "$0.y = ($1.y || $2.y) && !($1.y && $2.y);\n";
        result += "$0.z = ($1.z || $2.z) && !($1.z && $2.z);\n";
        result += "$0.w = ($1.w || $2.w) && !($1.w && $2.w);\n";
      }
      break;
    case OperationType::GREATER:
      result = "$0.x = $1.x > $2.x;\n";
      result += "$0.y = $1.y > $2.y;\n";
      result += "$0.z = $1.z > $2.z;\n";
      result += "$0.w = $1.w > $2.w;";
      break;
    case OperationType::GREATER_EQUAL:
      result = "$0.x = $1.x >= $2.x;\n";
      result += "$0.y = $1.y >= $2.y;\n";
      result += "$0.z = $1.z >= $2.z;\n";
      result += "$0.w = $1.w >= $2.w;";
      break;
    case OperationType::EQUAL:
      result = "$0.x = $1.x == $2.x;\n";
      result += "$0.y = $1.y == $2.y;\n";
      result += "$0.z = $1.z == $2.z;\n";
      result += "$0.w = $1.w == $2.w;";
      break;
    case OperationType::NOT_EQUAL:
      result = "$0.x = $1.x != $2.x;\n";
      result += "$0.y = $1.y != $2.y;\n";
      result += "$0.z = $1.z != $2.z;\n";
      result += "$0.w = $1.w != $2.w;";
      break;
    case OperationType::SHIFT_LEFT:
      if (gpu_info.IsApiWebGpu()) {
        result = "$0 = $1 << vec4<u32>($2);";
      } else {
        result = "$0 = $1 << $2;";
      }
      break;
    case OperationType::SHIFT_RIGHT:
      if (gpu_info.IsApiWebGpu()) {
        result = "$0 = $1 >> vec4<u32>($2);";
      } else {
        result = "$0 = $1 >> $2;";
      }
      break;
    default:
      return "Unknown operation type;";
  }
  if (swap_inputs) {
    return absl::Substitute(result, result_var, input1, input0);
  } else {
    return absl::Substitute(result, result_var, input0, input1);
  }
}

// Creates simple two input (first input is runtime tensor and second input is
// scalar argument) operation, for example sub, div, pow, etc.
ElementwiseDescriptor CreateElementwiseOneRuntimeOneScalar(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type, const ScalarValue& scalar, bool swap_inputs) {
  ElementwiseDescriptor op_desc;
  DataType scalar_type = DataType::UNKNOWN;
  if (std::holds_alternative<float>(scalar)) {
    scalar_type = DataType::FLOAT32;
    const float* value = std::get_if<float>(&scalar);
    if (op_type == OperationType::POW && *value == 3.0f) {
      op_desc.code = "out_value = in_value * in_value * in_value;";
      return op_desc;
    }
    if (op_type == OperationType::POW && *value == 1.5f) {
      op_desc.code = "out_value = in_value * sqrt(in_value);";
      return op_desc;
    }
    op_desc.args.AddFloat("scalar", *value);
  } else if (std::holds_alternative<int>(scalar)) {
    scalar_type = DataType::INT32;
    const int* value = std::get_if<int>(&scalar);
    op_desc.args.AddInt("scalar", *value);
  } else if (std::holds_alternative<unsigned int>(scalar)) {
    scalar_type = DataType::UINT32;
    const unsigned int* value = std::get_if<unsigned int>(&scalar);
    op_desc.args.AddUint("scalar", *value);
  }
  const DataType src_type_raw = definition.src_tensors[0].GetDataType();
  bool convert_bf16 =
      src_type_raw == DataType::BFLOAT16 &&
      !(gpu_info.IsApiMetal() && gpu_info.metal_info.IsNativeBfloatSupported());
  const DataType src_type = convert_bf16 ? DataType::FLOAT32 : src_type_raw;
  const std::string src_type_str = ToUclDataType(src_type, 4);
  std::string scalar_value_str = "args.scalar";
  if (src_type != scalar_type) {
    const std::string src_scalar_type_str = ToUclDataType(src_type, 1);
    scalar_value_str = "ucl::Convert<" + src_scalar_type_str + ">(args.scalar)";
  }
  op_desc.code = src_type_str + " second_val = ucl::Init<" + src_type_str +
                 ">(" + scalar_value_str + ");\n";
  op_desc.code += GetTwoInputCode(gpu_info, op_type, src_type, "out_value",
                                  "in_value", "second_val", swap_inputs);
  return op_desc;
}

// Creates simple two input(first input is runtime tensor and second input is
// constant linear tensor) operation, for example sub, div and etc.
ElementwiseDescriptor CreateElementwiseTwoInput(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type,
    const Tensor<Linear, DataType::FLOAT32>& constant_tensor,
    bool swap_inputs) {
  TensorDescriptor const_tensor_desc = CreateConstantLinearTensorDescriptor(
      gpu_info, definition.src_tensors[0].GetDataType(), constant_tensor);
  ElementwiseDescriptor op_desc;
  op_desc.args.AddObject("second_tensor", std::make_unique<TensorDescriptor>(
                                              std::move(const_tensor_desc)));
  const std::string s_coord = constant_tensor.shape.v == 1 ? "0" : "S_COORD";
  op_desc.code = absl::StrCat(
      "args.second_tensor::type second_val = args.second_tensor.Read(", s_coord,
      ");\n");
  if (constant_tensor.shape.v == 1) {
    op_desc.code += "  second_val.y = second_val.x;\n";
    op_desc.code += "  second_val.z = second_val.x;\n";
    op_desc.code += "  second_val.w = second_val.x;\n";
  }
  op_desc.code += GetTwoInputCode(
      gpu_info, op_type, definition.src_tensors[0].GetDataType(), "out_value",
      "in_value", "second_val", swap_inputs);
  return op_desc;
}

// Creates simple two input(first input is runtime tensor and second input is
// constant BHWC tensor) operation, for example sub, div and etc.
absl::StatusOr<ElementwiseDescriptor> CreateElementwiseTwoInput(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type,
    const Tensor<BHWC, DataType::FLOAT32>& constant_tensor, bool swap_inputs) {
  TensorDescriptor const_tensor_desc = definition.src_tensors[0];
  RETURN_IF_ERROR(const_tensor_desc.UpdateToSupportedStorageType(
      gpu_info, constant_tensor.shape));
  const_tensor_desc.UploadData(constant_tensor);

  ElementwiseDescriptor op_desc;
  op_desc.args.AddObject("second_tensor", std::make_unique<TensorDescriptor>(
                                              std::move(const_tensor_desc)));
  const std::string b_coord = constant_tensor.shape.b == 1 ? "0" : "B_COORD";
  const std::string x_coord = constant_tensor.shape.w == 1 ? "0" : "X_COORD";
  const std::string y_coord = constant_tensor.shape.h == 1 ? "0" : "Y_COORD";
  const std::string s_coord = constant_tensor.shape.c == 1 ? "0" : "S_COORD";
  op_desc.code = absl::StrCat(
      "args.second_tensor::type second_val = args.second_tensor.Read(", x_coord,
      ", ", y_coord, ", ", s_coord, ", ", b_coord, ");\n");
  if (constant_tensor.shape.c == 1) {
    op_desc.code += "  second_val.y = second_val.x;\n";
    op_desc.code += "  second_val.z = second_val.x;\n";
    op_desc.code += "  second_val.w = second_val.x;\n";
  }
  op_desc.code += GetTwoInputCode(
      gpu_info, op_type, definition.src_tensors[0].GetDataType(), "out_value",
      "in_value", "second_val", swap_inputs);

  return op_desc;
}

absl::StatusOr<ElementwiseDescriptor> CreateElementwiseTwoInput(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type,
    const Tensor<BHWDC, DataType::FLOAT32>& constant_tensor, bool swap_inputs) {
  TensorDescriptor const_tensor_desc = definition.src_tensors[0];
  const_tensor_desc.SetBHWDCShape(constant_tensor.shape);
  const_tensor_desc.SetLayout(Layout::BHWDC);
  RETURN_IF_ERROR(const_tensor_desc.UpdateToSupportedStorageType(
      gpu_info, constant_tensor.shape));
  const_tensor_desc.UploadData(constant_tensor);

  ElementwiseDescriptor op_desc;
  op_desc.args.AddObject("second_tensor", std::make_unique<TensorDescriptor>(
                                              std::move(const_tensor_desc)));
  std::string b_coord = constant_tensor.shape.b == 1 ? "0" : "B_COORD";
  std::string x_coord = constant_tensor.shape.w == 1 ? "0" : "X_COORD";
  std::string y_coord = constant_tensor.shape.h == 1 ? "0" : "Y_COORD";
  std::string z_coord = constant_tensor.shape.d == 1 ? "0" : "Z_COORD";
  std::string s_coord = constant_tensor.shape.c == 1 ? "0" : "S_COORD";

  op_desc.code = absl::StrCat(
      "args.second_tensor::type second_val = args.second_tensor.Read(", x_coord,
      ", ", y_coord, ", ", z_coord, ", ", s_coord, ", ", b_coord, ");\n");
  if (constant_tensor.shape.c == 1) {
    op_desc.code += "  second_val.y = second_val.x;\n";
    op_desc.code += "  second_val.z = second_val.x;\n";
    op_desc.code += "  second_val.w = second_val.x;\n";
  }
  op_desc.code += GetTwoInputCode(
      gpu_info, op_type, definition.src_tensors[0].GetDataType(), "out_value",
      "in_value", "second_val", swap_inputs);

  return op_desc;
}

absl::StatusOr<ElementwiseDescriptor> CreateElementwiseDesc(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type, const ElementwiseAttributes& attr) {
  return std::visit(
      [&](auto&& arg) {
        absl::StatusOr<ElementwiseDescriptor> result;
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, ScalarValue>) {
          result = CreateElementwiseOneRuntimeOneScalar(
              gpu_info, definition, op_type, arg,
              attr.runtime_tensor_is_second);
        } else if constexpr (std::is_same_v<
                                 T, Tensor<Linear, DataType::FLOAT32>>) {
          result = CreateElementwiseTwoInput(gpu_info, definition, op_type, arg,
                                             attr.runtime_tensor_is_second);
        } else if constexpr (std::is_same_v<T,
                                            Tensor<BHWC, DataType::FLOAT32>>) {
          result = CreateElementwiseTwoInput(gpu_info, definition, op_type, arg,
                                             attr.runtime_tensor_is_second);
        } else if constexpr (std::is_same_v<T,
                                            Tensor<BHWDC, DataType::FLOAT32>>) {
          result = CreateElementwiseTwoInput(gpu_info, definition, op_type, arg,
                                             attr.runtime_tensor_is_second);
        } else if constexpr (std::is_same_v<T, std::monostate>) {
          result = absl::InternalError("Received unsupported tensor.");
        }
        return result;
      },
      attr.param);
}

}  // namespace

ElementwiseDescriptor CreateElementwiseOneInput(const GpuInfo& gpu_info,
                                                const OperationType& op_type,
                                                const DataType& data_type) {
  ElementwiseDescriptor op_desc;
  op_desc.code =
      GetOneInputCode(gpu_info, op_type, data_type, "in_value", "out_value");
  return op_desc;
}

GPUOperation CreateElementwiseOneInput(const GpuInfo& gpu_info,
                                       const OperationDef& definition,
                                       const OperationType& op_type) {
  return CreateGpuOperation(
      definition,
      CreateElementwiseOneInput(gpu_info, op_type,
                                definition.src_tensors[0].GetDataType()));
}

GPUOperation CreateElementwiseOneInput(const GpuInfo& gpu_info,
                                       const TensorDescriptor& src,
                                       const TensorDescriptor& dst,
                                       const OperationType& op_type) {
  return CreateGpuOperation(
      src, dst,
      CreateElementwiseOneInput(gpu_info, op_type, src.GetDataType()));
}

GPUOperation CreateElementwise(const GpuInfo& gpu_info,
                               const OperationDef& definition,
                               const OperationType& op_type,
                               const ElementwiseAttributes& attr) {
  auto result = CreateElementwiseDesc(gpu_info, definition, op_type, attr);
  if (!result.ok()) {
    ABSL_LOG(ERROR) << result.status().message();
    return GPUOperation();
  } else {
    return CreateGpuOperation(definition, std::move(result.value()));
  }
}

GPUOperation CreateElementwiseTwoInput(const GpuInfo& gpu_info,
                                       const OperationDef& definition,
                                       const OperationType& op_type,
                                       const BHWDC& second_shape,
                                       const BHWDC& dst_shape) {
  ElementwiseDescriptor op_desc;
  op_desc.code = GetTwoInputCode(
      gpu_info, op_type, definition.src_tensors[0].GetDataType(), "out_value",
      "in_value", "in2_value", false);
  return CreateGpuOperation(definition, std::move(op_desc), second_shape,
                            dst_shape);
}

namespace {
std::string GetKernelBodyCode(const OperationDef& definition) {
  const TensorDescriptor& dst_desc = definition.dst_tensors[0];
  bool has_depth = dst_desc.HasAxis(Axis::DEPTH);
  for (const auto& src : definition.src_tensors) {
    if (src.HasAxis(Axis::DEPTH)) {
      has_depth = true;
    }
  }

  std::string c;
  c += "MAIN_FUNCTION($$0) {\n";
  if (dst_desc.HasAxis(Axis::BATCH)) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = ucl::GetGlobalId<0>();\n";
  }

  std::string coords = "X, Y";
  if (has_depth) {
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

  c += "  args.dst_tensor::type result;\n";
  c += "  $0\n";
  c += "  args.dst_tensor.Write(result, " + coords + ");\n";
  c += "} \n";
  return c;
}
std::string GetReadBroadcastedValueCode(const BHWDC& src_shape,
                                        const TensorDescriptor& src_desc,
                                        const BHWDC& dst_shape) {
  const std::string x_coord = src_shape.w != dst_shape.w ? "0" : "X";
  const std::string y_coord = src_shape.h != dst_shape.h ? "0" : "Y";
  const std::string s_coord = src_shape.c != dst_shape.c ? "0" : "S";
  const std::string z_coord =
      (src_shape.d != dst_shape.d || dst_shape.d == 1) ? "0" : "Z";
  const std::string b_coord =
      (src_shape.b != dst_shape.b || dst_shape.b == 1) ? "0" : "B";

  std::string coords;
  if (src_desc.HasAxis(Axis::DEPTH)) {
    coords = absl::StrCat(x_coord, ", ", y_coord, ", ", z_coord, ", ", s_coord);
    if (src_desc.HasAxis(Axis::BATCH)) {
      coords += ", " + b_coord;
    }
  } else {
    coords = absl::StrCat(x_coord, ", ", y_coord, ", ", s_coord);
    if (src_desc.HasAxis(Axis::BATCH)) {
      coords += ", " + b_coord;
    }
  }
  std::string read_value_code =
      absl::StrCat("args.$0::type $1 = args.$0.Read(", coords, ");\n");
  if (src_shape.c != dst_shape.c) {
    read_value_code += "  $1.y = $1.x;\n";
    read_value_code += "  $1.z = $1.x;\n";
    read_value_code += "  $1.w = $1.x;\n";
  }
  return read_value_code;
}
}  // namespace

GPUOperation CreateElementwiseOneInputWithBroadcast(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type, const BHWDC& input_shape,
    const BHWDC& output_shape) {
  GPUOperation op;
  op.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  std::string c;
  c += "  " + absl::Substitute(
                  GetReadBroadcastedValueCode(
                      input_shape, definition.src_tensors[0], output_shape),
                  "src_tensor", "first_value");
  c += "  " + GetOneInputCode(gpu_info, op_type,
                              definition.src_tensors[0].GetDataType(),
                              "first_value", "result");
  op.code_ = absl::Substitute(GetKernelBodyCode(definition), c);
  return op;
}

GPUOperation CreateElementwiseWithBroadcast(const GpuInfo& gpu_info,
                                            const OperationDef& definition,
                                            const OperationType& op_type,
                                            const ElementwiseAttributes& attr,
                                            const BHWDC& input_shape,
                                            const BHWDC& output_shape) {
  auto result = CreateElementwiseDesc(gpu_info, definition, op_type, attr);
  if (!result.ok()) {
    ABSL_LOG(ERROR) << result.status().message();
    return GPUOperation();
  }
  ElementwiseDescriptor& op_desc = result.value();
  GPUOperation op;
  op.args_ = std::move(op_desc.args);
  op.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  std::string c;
  c += "  " + absl::Substitute(
                  GetReadBroadcastedValueCode(
                      input_shape, definition.src_tensors[0], output_shape),
                  "src_tensor", "first_value");
  c += "  " + absl::StrReplaceAll(op_desc.code, {{"in_value", "first_value"},
                                                 {"out_value", "result"},
                                                 {"X_COORD", "X"},
                                                 {"Y_COORD", "Y"},
                                                 {"Z_COORD", "Z"},
                                                 {"S_COORD", "S"},
                                                 {"B_COORD", "B"}});
  op.code_ = absl::Substitute(GetKernelBodyCode(definition), c);
  return op;
}

GPUOperation CreateElementwiseTwoInputWithBroadcast(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type, const BHWDC& first_input_shape,
    const BHWDC& second_input_shape, const BHWDC& output_shape,
    const ElementwiseAttributes& attr) {
  GPUOperation op;
  op.AddSrcTensor("src0_tensor", definition.src_tensors[0]);
  op.AddSrcTensor("src1_tensor", definition.src_tensors[1]);
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  std::string c;
  c += "  " + absl::Substitute(GetReadBroadcastedValueCode(
                                   first_input_shape, definition.src_tensors[0],
                                   output_shape),
                               "src0_tensor", "first_value");
  c += "  " + absl::Substitute(GetReadBroadcastedValueCode(
                                   second_input_shape,
                                   definition.src_tensors[1], output_shape),
                               "src1_tensor", "second_value");
  c += "  " + GetTwoInputCode(gpu_info, op_type,
                              definition.src_tensors[0].GetDataType(), "result",
                              "first_value", "second_value",
                              attr.runtime_tensor_is_second);
  op.code_ = absl::Substitute(GetKernelBodyCode(definition), c);
  return op;
}

}  // namespace ml_drift
