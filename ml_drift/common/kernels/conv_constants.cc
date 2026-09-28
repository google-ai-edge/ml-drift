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

#include "ml_drift/common/kernels/conv_constants.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/compiler_options.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/task/work_group_picking.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

namespace {
int GetOptimalMaxConstantSize(const GpuInfo& gpu_info) {
  if (gpu_info.IsAdreno()) {
    // Adreno can provide up to ~3-4KB of constant memory, but in some cases
    // even 3KB can have very bad performance.
    const auto& adreno_info = gpu_info.adreno_info;
    if (adreno_info.IsAdreno3xx() || adreno_info.IsAdreno4xx() ||
        adreno_info.IsAdreno5xx()) {
      return 256 * 10;  // 2.5KB
    } else {
      return 256 * 14;  // 3.5KB
    }
  } else if (gpu_info.IsAMD()) {
    return 4096;
  } else if (gpu_info.IsMali()) {
    if (gpu_info.mali_info.generation >= MaliInfo::Gen::kValhallV4) {
      return 1024;
    }
    return 0;
  } else if (gpu_info.IsPowerVR()) {
    return 1024;
  } else {
    return 1024;
  }
}

void AppendConditionally(const std::string& value, const std::string& delimiter,
                         std::string* result) {
  if (!result->empty()) {
    *result += delimiter;
  }
  *result += value;
}

std::string GenerateConvolutionConstantCode(const GpuInfo& gpu_info,
                                            const OperationDef& op_def,
                                            CalculationsPrecision precision,
                                            const OHWI& weights_shape,
                                            bool x_oob_reads, bool y_oob_reads,
                                            bool src_local_memory_caching,
                                            bool has_bias) {
  auto src_desc = op_def.src_tensors[0];

  const int src_slices = DivideRoundUp(weights_shape.i, 4);
  const int dst_slices = DivideRoundUp(weights_shape.o, 4);

  std::string c;
  int ch_counter = 0;
  auto GetWeightValueX4 = [&]() {
    if (ch_counter % 4 == 0) {
      c += "  w_value = args.weights.Read(" + std::to_string(ch_counter / 4) +
           ");\n";
    }
    ch_counter += 4;
    return "w_value";
  };
  auto GetWeightValueX1 = [&]() {
    if (ch_counter % 4 == 0) {
      c += "  w_value = args.weights.Read(" + std::to_string(ch_counter / 4) +
           ");\n";
    }
    const std::string postfixes[] = {"x", "y", "z", "w"};
    ch_counter += 1;
    return "w_value." + postfixes[(ch_counter - 1) % 4];
  };

  c += "MAIN_FUNCTION($0) {\n";
  if (src_desc.HasAxis(Axis::kBatch)) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = ucl::GetGlobalId<0>();\n";
  }
  c += "  int Y = ucl::GetGlobalId<1>();\n";
  c += "  Type w_value;\n";
  if (src_local_memory_caching) {
    c += R"(
  __local Type src_cache[LOCAL_MEM_SIZE_Y][LOCAL_MEM_SIZE_X];
  int local_x = ucl::GetLocalId<0>();
  int local_y = ucl::GetLocalId<1>();
  int local_id = local_y * WG_SIZE_X + local_x;
  for (int tile_id = 0; tile_id < TILES_COUNT; ++tile_id) {
    int interm_linear_id = local_id + tile_id * WG_SIZE_Y * WG_SIZE_X;
    if (interm_linear_id < LOCAL_MEM_SIZE_Y * LOCAL_MEM_SIZE_X) {
      int interm_local_y = interm_linear_id / LOCAL_MEM_SIZE_X;
      int interm_local_x = interm_linear_id - interm_local_y * LOCAL_MEM_SIZE_X;
      int interm_x = WG_SIZE_X * ucl::GetGroupId<0>() + interm_local_x + args.padding_x;
      int interm_y = WG_SIZE_Y * ucl::GetGroupId<1>() + interm_local_y + args.padding_y;
      if (interm_x >= 0 && interm_x < args.src_tensor.Width() && interm_y >= 0 && interm_y < args.src_tensor.Height()) {
        src_cache[interm_local_y][interm_local_x] = args.src_tensor.Read(interm_x, interm_y, 0);
      } else {
        src_cache[interm_local_y][interm_local_x] = ucl::Init<Type>(0.0f);
      }
    }
  }
  ucl::SyncThreads<WorkGroup, Local>();
)";
  }
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height()) "
       "return;\n";
  c += "  int start_x = X * args.stride_x + args.padding_x;\n";
  c += "  int start_y = Y * args.stride_y + args.padding_y;\n";
  for (int dst_s = 0; dst_s < dst_slices; ++dst_s) {
    c += "  AccType r" + std::to_string(dst_s) +
         " = ucl::Init<AccType>(0.0f);\n";
  }
  std::string check;
  if (y_oob_reads && !src_desc.SupportsZeroClamp(Axis::kHeight, gpu_info)) {
    AppendConditionally("inside_y", " && ", &check);
  }
  if (x_oob_reads && !src_desc.SupportsZeroClamp(Axis::kWidth, gpu_info)) {
    AppendConditionally("inside_x", " && ", &check);
  }
  for (int src_s = 0; src_s < src_slices; ++src_s) {
    for (int ky = 0; ky < weights_shape.h; ++ky) {
      std::string s_y = absl::StrCat("(start_y + ", ky, " * args.dilation_y)");
      c += "  {\n";
      if (src_local_memory_caching) {
        c += "    int y_c = local_y + args.padding_y + " + std::to_string(ky) +
             ";\n";
      } else {
        c += "    int y_c = start_y + " + std::to_string(ky) +
             " * args.dilation_y;\n";
        if (y_oob_reads &&
            !src_desc.SupportsZeroClamp(Axis::kHeight, gpu_info)) {
          c += "    bool inside_y = y_c >= 0 && y_c < "
               "args.src_tensor.Height();\n";
          c += "    y_c = clamp(y_c, 0, args.src_tensor.Height() - 1);\n";
        }
      }
      for (int kx = 0; kx < weights_shape.w; ++kx) {
        c += "    {\n";
        if (src_local_memory_caching) {
          c += "      int x_c = local_x + args.padding_x + " +
               std::to_string(kx) + ";\n";
          c += "      Type src = src_cache[y_c][x_c];\n";
        } else {
          c += "      int x_c = start_x + " + std::to_string(kx) +
               " * args.dilation_x;\n";
          if (x_oob_reads &&
              !src_desc.SupportsZeroClamp(Axis::kWidth, gpu_info)) {
            c += "      bool inside_x = x_c >= 0 && x_c < "
                 "args.src_tensor.Width();\n";
            c += "      x_c = clamp(x_c, 0, args.src_tensor.Width() - 1);\n";
          }
          c += "      Type src = args.src_tensor.Read(x_c, y_c, " +
               std::to_string(src_s) + ");\n";
          if (!check.empty()) {
            c += "      src *= ucl::Convert<ScalarType>(" + check + ");\n";
          }
        }
        for (int dst_s = 0; dst_s < dst_slices; ++dst_s) {
          const int src_ch_count = std::min(4, weights_shape.i - src_s * 4);
          for (int src_ch = 0; src_ch < src_ch_count; ++src_ch) {
            const bool use_fma = gpu_info.IsApiOpenCl() && gpu_info.IsAMD();
            const std::string postfixes[] = {".x", ".y", ".z", ".w"};
            const std::string src_name = "src" + postfixes[src_ch];
            if (weights_shape.o % 4 == 0 &&
                precision != CalculationsPrecision::kF32F16) {
              const std::string w_name = GetWeightValueX4();
              if (use_fma) {
                c += "      r" + std::to_string(dst_s) + " = fma(" + src_name +
                     ", " + w_name + ", r" + std::to_string(dst_s) + ");\n";
              } else {
                c += "      r" + std::to_string(dst_s) + " += " + src_name +
                     " * " + w_name + ";\n";
              }
            } else {
              const int dst_ch_count = std::min(4, weights_shape.o - dst_s * 4);
              for (int dst_ch = 0; dst_ch < dst_ch_count; ++dst_ch) {
                const std::string w_name = GetWeightValueX1();
                const std::string dst_name =
                    "r" + std::to_string(dst_s) + postfixes[dst_ch];
                if (precision == CalculationsPrecision::kF32F16) {
                  c += "      " + dst_name +
                       " += ucl::Convert<AccScalarType>(" + src_name + " * " +
                       w_name + ");\n";
                } else if (use_fma) {
                  c += "      " + dst_name + " = fma(" + src_name + ", " +
                       w_name + ", " + dst_name + ");\n";
                } else {
                  c += "      " + dst_name + " += " + src_name + " * " +
                       w_name + ";\n";
                }
              }
            }
          }
        }
        c += "    }\n";
      }
      c += "  }\n";
    }
  }
  for (int dst_s = 0; dst_s < dst_slices; ++dst_s) {
    std::string s_i = std::to_string(dst_s);
    c += "  {\n";
    c += "    Type res = ucl::Convert<Type>(r" + s_i + ");\n";
    if (has_bias) {
      c += "  res = res + args.biases.Read(" + s_i + ");\n";
    }
    c += "    args.dst_tensor.Write(res, X, Y, " + s_i + ");\n";
    c += "  }\n";
  }
  c += "}\n";
  const DataType acc_type = precision == CalculationsPrecision::kF16
                                ? DataType::kFloat16
                                : DataType::kFloat32;
  const DataType type = op_def.src_tensors[0].GetDataType();
  absl::StrReplaceAll({{"ScalarType", ToUclDataType(type, 1)},
                       {"Type", ToUclDataType(type, 4)},
                       {"AccScalarType", ToUclDataType(acc_type, 1)},
                       {"AccType", ToUclDataType(acc_type, 4)}},
                      &c);
  return c;
}
}  // namespace

bool IsConvConstantsSupported(const GpuInfo& gpu_info,
                              CalculationsPrecision precision,
                              const Convolution2DAttributes& attr) {
  if (gpu_info.IsApiOpenCl() && gpu_info.IsAdreno()) {
    const std::string kBadDriver =
        "OpenCL 2.0 QUALCOMM build: commit #7ff4f54 changeid #I4460aa6217 "
        "Date: 12/30/18";
    if (absl::StrContains(gpu_info.opencl_info.platform_version, kBadDriver)) {
      return false;
    }
  }

  if (gpu_info.IsApiOpenCl() && gpu_info.IsAMD() &&
      precision == CalculationsPrecision::kF32F16) {
    return false;
  }

  if (attr.groups != 1) {
    return false;
  }

  const auto& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  const int filters_count = AlignByN(weights_shape.DimensionsProduct(), 4);
  const int float_size =
      precision == CalculationsPrecision::kF32 ? sizeof(float) : sizeof(half);
  const int filters_buffer_size = filters_count * float_size;
  const int kConstantMaxSize = GetOptimalMaxConstantSize(gpu_info);
  const int flt4_registers = DivideRoundUp(weights_shape.o, 4);
  return filters_buffer_size <= kConstantMaxSize && flt4_registers <= 8;
}

ConvConstants::ConvConstants(const GpuInfo& gpu_info,
                             const OperationDef& definition,
                             CalculationsPrecision precision,
                             const Convolution2DAttributes& attr,
                             bool has_bias) {
  weights_data_type_ = DeduceDataTypeFromPrecision(precision);
  weights_shape_ =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  // intentionally disabled version
  src_local_memory_caching_ =
      gpu_info.IsAMD() && gpu_info.IsNvidia() && weights_shape_.w == 3 &&
      weights_shape_.h == 3 && weights_shape_.o == 4 && weights_shape_.i == 4 &&
      attr.strides.w == 1 && attr.strides.h == 1 &&
      attr.padding.prepended.w == 1 && attr.padding.prepended.h == 1 &&
      attr.padding.appended.w == 1 && attr.padding.appended.h == 1 &&
      attr.dilations.w == 1 && attr.dilations.h == 1;
  args_.AddInt("stride_x", attr.strides.w);
  args_.AddInt("stride_y", attr.strides.h);
  args_.AddInt("padding_x", -attr.padding.prepended.w);
  args_.AddInt("padding_y", -attr.padding.prepended.h);
  args_.AddInt("dilation_x", attr.dilations.w);
  args_.AddInt("dilation_y", attr.dilations.h);
  AddSrcTensor("src_tensor", definition.src_tensors[0]);
  AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  bool x_oob_reads =
      attr.padding.appended.w != 0 || attr.padding.prepended.w != 0;
  bool y_oob_reads =
      attr.padding.appended.h != 0 || attr.padding.prepended.h != 0;
  if (src_local_memory_caching_) {
    work_group_size_ = int3(16, 8, 1);
  }
  code_ = GenerateConvolutionConstantCode(
      gpu_info, definition, precision, weights_shape_, x_oob_reads, y_oob_reads,
      src_local_memory_caching_, has_bias);
  if (src_local_memory_caching_) {
    const int local_mem_size_x = work_group_size_.x + 2;
    const int local_mem_size_y = work_group_size_.y + 2;
    const int tiles_count =
        DivideRoundUp(local_mem_size_x * local_mem_size_y,
                      work_group_size_.x * work_group_size_.y);
    code_ = absl::StrReplaceAll(
        code_,
        {{"LOCAL_MEM_SIZE_X", std::to_string(local_mem_size_x)},
         {"LOCAL_MEM_SIZE_Y", std::to_string(local_mem_size_y)},
         {"WG_SIZE_X", std::to_string(work_group_size_.x)},
         {"WG_SIZE_Y", std::to_string(work_group_size_.y)},
         {"TILES_COUNT", std::to_string(tiles_count)},
         {"Type", ToUclDataType(definition.src_tensors[0].GetDataType(), 4)}});
  }
  if (precision == CalculationsPrecision::kF16 && gpu_info.IsAdreno() &&
      gpu_info.adreno_info.IsAdreno3xx()) {
    compiler_options_.push_back(CompilerOptions::kAdrenoFullSimd);
  }
  if (gpu_info.IsMali() || gpu_info.IsPowerVR()) {
    compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
}

std::vector<int3> ConvConstants::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info) const {
  if (src_local_memory_caching_) {
    return {work_group_size_};
  }
  return GetPossibleWorkGroups(tuning_type, gpu_info, kernel_info, grid_size_);
}

int3 ConvConstants::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height() * dst_[0]->Depth();
  const int grid_z = 1;
  return int3(grid_x, grid_y, grid_z);
}

BufferDescriptor GetBufferDescriptor(const GpuInfo& gpu_info,
                                     const WeightsDescription& weights_desc) {
  BufferDescriptor buffer_desc;
  buffer_desc.element_type = weights_desc.type;
  buffer_desc.element_size = 4;
  if (gpu_info.IsApiOpenCl() || gpu_info.IsApiMetal()) {
    buffer_desc.memory_type = MemoryType::kConstant;
  } else {
    buffer_desc.memory_type = MemoryType::kGlobal;
  }
  return buffer_desc;
}

void ConvConstants::UploadWeights(
    const GpuInfo& gpu_info, const Tensor<OHWI, DataType::kFloat32>& weights) {
  WeightsDescription weights_desc = GetWeightsDescription();
  BufferDescriptor buffer_desc = GetBufferDescriptor(gpu_info, weights_desc);
  const int elements_count =
      GetTotalElementsCountForLayout(weights_desc, weights.shape);
  buffer_desc.size = SizeOf(weights_desc.type) * AlignByN(elements_count, 4);
  buffer_desc.data.resize(buffer_desc.size);

  RearrangeWeights(weights, weights_desc, absl::MakeSpan(buffer_desc.data));

  args_.AddObject("weights",
                  std::make_unique<BufferDescriptor>(std::move(buffer_desc)));
}

ConvConstants CreateConvConstants(const GpuInfo& gpu_info,
                                  const OperationDef& definition,
                                  CalculationsPrecision precision,
                                  const Convolution2DAttributes& attr) {
  const bool has_bias = !attr.bias.data.empty();
  ConvConstants conv(gpu_info, definition, precision, attr, has_bias);
  conv.UploadWeights(gpu_info, GetFloatWeights(attr));
  if (has_bias) {
    TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
        gpu_info, definition.src_tensors[0].GetDataType(), attr.bias);
    conv.args_.AddObject("biases", std::make_unique<TensorDescriptor>(
                                       std::move(bias_tensor_desc)));
  }
  return conv;
}

ConvConstants CreateConvConstantsExternalWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision, const Convolution2DAttributes& attr,
    const TensorDescriptor* bias) {
  const bool has_bias = bias != nullptr;
  ConvConstants conv(gpu_info, definition, precision, attr, has_bias);
  {  // Add src buffer for external weights.
    WeightsDescription weights_desc = conv.GetWeightsDescription();
    BufferDescriptor buffer_desc = GetBufferDescriptor(gpu_info, weights_desc);
    conv.AddSrcBuffer("weights", buffer_desc);
  }
  if (has_bias) {
    conv.AddSrcTensor("biases", *bias);
  }
  return conv;
}

}  // namespace ml_drift
