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

#include "ml_drift/common/kernels/depthwise_conv_wave_memory.h"

#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/work_group_picking.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace {

std::string GenerateDepthwiseConvCode(const GpuInfo& gpu_info,
                                      const OperationDef& op_def,
                                      CalculationsPrecision precision,
                                      const int2& kernel_size,
                                      int weights_cache_size_flt4,
                                      bool has_bias) {
  const auto src_tensor_type = op_def.src_tensors[0].GetStorageType();
  const bool is_src_image_buffer =
      src_tensor_type == TensorStorageType::IMAGE_BUFFER;
  const bool manual_clamp =
      src_tensor_type == TensorStorageType::BUFFER || is_src_image_buffer;
  const bool late_xy_check = !gpu_info.IsAdreno();

  std::string c;
  c += "#pragma OPENCL EXTENSION ucl_wave_memory: enable\n";
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = ucl::GetGlobalId<0>();\n";
  }
  c += "  int Y = ucl::GetGlobalId<1>();\n";
  c += "  int Z = ucl::GetGlobalId<2>();\n";
  c += "  int x_src = mad24(X, args.stride_x, args.padding_x);\n";
  c += "  int y_src = mad24(Y, args.stride_y, args.padding_y);\n";
  for (int i = 0; i < kernel_size.x; ++i) {
    c += "  int gidx" + std::to_string(i) + " = x_src + " + std::to_string(i) +
         " * args.dilation_x;\n";
  }
  for (int i = 0; i < kernel_size.y; ++i) {
    c += "  int gidy" + std::to_string(i) + " = y_src + " + std::to_string(i) +
         " * args.dilation_y;\n";
  }
  if (!late_xy_check) {
    c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height()) "
         "return;\n";
  }
  c += "  if (Z >= args.dst_tensor.Slices()) return;\n";
  c += "\n";
  c += "  AccType r0 = ucl::Init<AccType>(0.0f);\n";
  c += "\n";
  c += "  int f_offset = Z * args.filter_offset;\n";
  c += "  __wave Type weights_cache[" +
       std::to_string(weights_cache_size_flt4) + "];\n";
  if (manual_clamp) {
    for (int i = 0; i < kernel_size.x; ++i) {
      c += "  bool x" + std::to_string(i) + "_in = gidx" + std::to_string(i) +
           " >= 0 && gidx" + std::to_string(i) +
           " < args.src_tensor.Width();\n";
    }
    for (int i = 0; i < kernel_size.y; ++i) {
      c += "  bool y" + std::to_string(i) + "_in = gidy" + std::to_string(i) +
           " >= 0 && gidy" + std::to_string(i) +
           " < args.src_tensor.Height();\n";
    }
    for (int i = 0; i < kernel_size.x; ++i) {
      c += "  gidx" + std::to_string(i) + " = clamp(gidx" + std::to_string(i) +
           ", 0, args.src_tensor.Width() - 1);\n";
    }
    for (int i = 0; i < kernel_size.y; ++i) {
      c += "  gidy" + std::to_string(i) + " = clamp(gidy" + std::to_string(i) +
           ", 0, args.src_tensor.Height() - 1);\n";
    }
  }
  if (is_src_image_buffer) {
    c += "  int linear_addr;\n";
  }
  auto read_src = [&](int x, int y) {
    const std::string condition =
        manual_clamp
            ? "(x" + std::to_string(x) + "_in && y" + std::to_string(y) + "_in)"
            : std::string("");
    const std::string channels[] = {"x", "y", "z"};
    const std::string xc = "gidx" + std::to_string(x);
    const std::string yc = "gidy" + std::to_string(y);
    if (is_src_image_buffer) {
      c += "  linear_addr = args.src_tensor.GetAddress(" + xc + ", " + yc +
           ", Z);\n";
      c += "  linear_addr = select(-1, linear_addr, " + condition + ");\n";
      return std::string("args.src_tensor.Read(linear_addr)");
    } else {
      const std::string multiplier =
          manual_clamp ? " * ucl::Convert<SType>" + condition : std::string("");
      return "args.src_tensor.Read(" + xc + ", " + yc + ", Z)" + multiplier;
    }
  };
  auto maybe_load_weights = [&](int weight_index) {
    if (weight_index % weights_cache_size_flt4 == 0) {
      if (weight_index != 0) {
        c += "  ucl::SyncThreads<WaveLoad>();\n";
      }
      c += "    ucl::WaveLoad(weights_cache, args.weights.GetPtr(), "
           "f_offset, " +
           std::to_string(weights_cache_size_flt4) + ");\n";
      c += "  f_offset += " + std::to_string(weights_cache_size_flt4) + ";\n";
      c += "  ucl::SyncThreads<WaveLoad>();\n";
    }
  };
  auto get_filter = [&](int weight_index) {
    const int cache_index = weight_index % weights_cache_size_flt4;
    return "weights_cache[" + std::to_string(cache_index) + "]";
  };
  int batch_size = 4;
  if (kernel_size.x * kernel_size.y <= weights_cache_size_flt4 &&
      kernel_size.x == 3) {
    batch_size = 3;
  }
  for (int j = 0; j < batch_size; ++j) {
    c += "  Type s" + std::to_string(j) + ";\n";
  }
  for (int i = 0; i < kernel_size.x * kernel_size.y; i += batch_size) {
    maybe_load_weights(i);
    // dummy check to stabilize register usage
    c += "  if (Z >= -" + std::to_string(i + 1) + ") {\n";
    for (int j = 0; j < batch_size; ++j) {
      int index = i + j;
      if (index >= kernel_size.x * kernel_size.y) {
        break;
      }
      int x = index % kernel_size.x;
      int y = index / kernel_size.x;
      c += "    s" + std::to_string(j) + " = " + read_src(x, y) + ";\n";
    }
    c += "  }\n";
    for (int j = 0; j < batch_size; ++j) {
      int index = i + j;
      if (index >= kernel_size.x * kernel_size.y) {
        break;
      }
      c += "  r0 += ucl::Convert<AccType>(s" + std::to_string(j) + " * " +
           get_filter(index) + ");\n";
    }
  }
  if (late_xy_check) {
    c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height()) "
         "return;\n";
  }
  c += "  Type res0 = ucl::Convert<Type>(r0);\n";
  if (has_bias) {
    c += "  res0 += args.biases.Read(Z);\n";
  }
  c += "  args.dst_tensor.Write(res0, X, Y, Z);\n";
  c += "}\n";
  const DataType acc_type = precision == CalculationsPrecision::F16
                                ? DataType::FLOAT16
                                : DataType::FLOAT32;
  const DataType type = op_def.src_tensors[0].GetDataType();
  absl::StrReplaceAll({{"SType", ToUclDataType(type, 1)},
                       {"Type", ToUclDataType(type, 4)},
                       {"AccType", ToUclDataType(acc_type, 4)}},
                      &c);
  return c;
}

}  // namespace

DepthwiseConvWaveMemory::DepthwiseConvWaveMemory(
    const OperationDef& definition, CalculationsPrecision precision,
    const DepthwiseConvolution2DAttributes& attr, const GpuInfo& gpu_info) {
  const auto& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  int2 kernel_size = int2(weights_shape.w, weights_shape.h);
  if (gpu_info.IsAdreno()) {
    wave_size_ = 128;
    work_group_size_ = int3(16, 8, 1);
  } else if (gpu_info.IsPowerVR()) {
    wave_size_ = 128;
    work_group_size_ = int3(16, 8, 1);
  } else {
    wave_size_ = 32;
    work_group_size_ = int3(8, 4, 1);
  }
  weights_cache_size_flt4_ = 8;
  if (precision != CalculationsPrecision::F32) {
    weights_cache_size_flt4_ *= 2;
  }
  if (!gpu_info.IsAdreno()) {
    weights_cache_size_flt4_ = wave_size_;
  }
  if (kernel_size.x * kernel_size.y <= weights_cache_size_flt4_) {
    weights_cache_size_flt4_ = AlignByN(kernel_size.x * kernel_size.y, 4);
  }
  args_.AddInt("stride_x", attr.strides.w);
  args_.AddInt("stride_y", attr.strides.h);
  args_.AddInt("padding_x", -attr.padding.prepended.w);
  args_.AddInt("padding_y", -attr.padding.prepended.h);
  args_.AddInt("dilation_x", attr.dilations.w);
  args_.AddInt("dilation_y", attr.dilations.h);
  const int flt4_per_layer =
      AlignByN(kernel_size.x * kernel_size.y, weights_cache_size_flt4_);
  args_.AddInt("filter_offset", flt4_per_layer);
  AddSrcTensor("src_tensor", definition.src_tensors[0]);
  AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  code_ = GenerateDepthwiseConvCode(gpu_info, definition, precision,
                                    kernel_size, weights_cache_size_flt4_,
                                    !attr.bias.data.empty());
}

int3 DepthwiseConvWaveMemory::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

std::vector<int3> DepthwiseConvWaveMemory::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info) const {
  switch (tuning_type) {
    case TuningType::kExhaustive:
      return GetWorkGroupsXYMultipleOf(wave_size_, gpu_info, kernel_info,
                                       grid_size_);
    case TuningType::kFast:
    default:
      return {GetSimpleWorkGroupXYisMultipleK(wave_size_, grid_size_)};
  }
}

bool HardwareSupportsDepthwiseConvWaveMemory(const GpuInfo& gpu_info) {
  if (gpu_info.IsAdreno() && gpu_info.SupportsExtension("ucl_wave_memory")) {
    return true;
  }
  if (gpu_info.IsIntel() && gpu_info.SupportsExtension("ucl_wave_memory")) {
    return true;
  }
  if (gpu_info.IsPowerVR() && gpu_info.SupportsExtension("ucl_wave_memory")) {
    return true;
  }
  return false;
}

bool IsDepthwiseConvWaveMemorySupported(
    const GpuInfo& gpu_info, const DepthwiseConvolution2DAttributes& attr) {
  return std::visit([](const auto& w) { return w.shape.o; }, attr.weights) ==
             1 &&
         HardwareSupportsDepthwiseConvWaveMemory(gpu_info);
}

DepthwiseConvWaveMemory CreateDepthwiseConvWaveMemory(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const DepthwiseConvolution2DAttributes& attr) {
  DepthwiseConvWaveMemory result(definition, precision, attr, gpu_info);
  const DataType weights_data_type = precision == CalculationsPrecision::F32
                                         ? DataType::FLOAT32
                                         : DataType::FLOAT16;
  result.UploadWeights(gpu_info, GetFloatWeights(attr), weights_data_type);
  if (!attr.bias.data.empty()) {
    TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
        gpu_info, definition.src_tensors[0].GetDataType(), attr.bias);
    result.args_.AddObject("biases", std::make_unique<TensorDescriptor>(
                                         std::move(bias_tensor_desc)));
  }
  return result;
}

}  // namespace ml_drift
