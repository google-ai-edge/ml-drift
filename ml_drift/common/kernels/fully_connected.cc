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

#include "ml_drift/common/kernels/fully_connected.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernels/fully_connected_util.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/compiler_options.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

namespace {
inline bool UseBufferForWeights(const GpuInfo& gpu_info,
                                const OHWI& weights_shape) {
  if (gpu_info.IsAdreno() || gpu_info.IsAMD() || gpu_info.IsMali() ||
      gpu_info.IsBroadcom() || gpu_info.IsApple() || gpu_info.IsApiWebGpu()) {
    return true;
  }
  if (gpu_info.IsIntel() && gpu_info.IsApiOpenCl()) {
    return true;
  }
  WeightsDescription texture_weights_desc;
  texture_weights_desc.layout =
      WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4;
  texture_weights_desc.output_group_size = 1;
  const uint2 tex_size = Get2dResourceSize(texture_weights_desc, weights_shape);
  const bool can_use_textures = gpu_info.SupportsImages() &&
                                tex_size.x <= gpu_info.GetMaxImage2DWidth() &&
                                tex_size.y <= gpu_info.GetMaxImage2DHeight();
  return !can_use_textures;
}

inline bool UseBufferForIntWeights(const GpuInfo& gpu_info, int int_bit_size,
                                   const OHWI& weights_shape,
                                   bool prefer_textures) {
  if (gpu_info.IsIntel() && gpu_info.IsApiOpenCl()) {
    return true;
  }
  if (int_bit_size == 2 && gpu_info.IsApiOpenCl() && gpu_info.IsMali()) {
    return true;
  }
  WeightsDescription weights_desc;
  weights_desc.type = DataType::UINT8;
  weights_desc.layout = WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;
  weights_desc.output_group_size = DivideRoundUp(weights_shape.o, 4);
  uint2 tex_size = Get2dResourceSize(weights_desc, weights_shape);
  tex_size.x /= 4;  // because we store 4 uint2/4/8 as one uint8/16/32
  const bool can_use_textures = gpu_info.SupportsImages() &&
                                tex_size.x <= gpu_info.GetMaxImage2DWidth() &&
                                tex_size.y <= gpu_info.GetMaxImage2DHeight();
  const bool is_textures_on_adreno_recommended =
      gpu_info.IsAdreno() &&
      gpu_info.adreno_info.adreno_gpu != AdrenoGpu::kAdreno740 &&
      gpu_info.adreno_info.adreno_gpu != AdrenoGpu::kAdreno750 &&
      !(gpu_info.adreno_info.generation >= AdrenoInfo::Generation::kGen8);
  const bool is_textures_recommended =
      (gpu_info.IsApiWebGpu() && prefer_textures) ||
      (!gpu_info.IsApiWebGpu() &&
       (is_textures_on_adreno_recommended || gpu_info.IsAMD() ||
        gpu_info.IsMali() || gpu_info.IsBroadcom() || gpu_info.IsIntel() ||
        gpu_info.IsPowerVR() || gpu_info.IsMaleoon()));
  return !(can_use_textures && is_textures_recommended);
}

int3 GetWorkGroupSize(const FullyConnected::ConvParams& params,
                      const GpuInfo& gpu_info, DataType acc_type,
                      const OHWI& weights_shape) {
  const bool is_quantized = fc::IsQuantized(params.weights_type);
  const int dst_slices = DivideRoundUp(weights_shape.o, 4);
  if (gpu_info.IsApple() && gpu_info.IsApiMetal() &&
      gpu_info.apple_info.IsMSeries()) {
    int total_task_size = dst_slices;
    if (params.batched_weights) {
      if (params.runtime_batch_ids) {
        total_task_size *= params.runtime_batch_ids;
      } else {
        total_task_size *= weights_shape.h;
      }
    }
    const int cu_count = gpu_info.GetComputeUnitsCount();
    double task_size_per_cu = static_cast<double>(total_task_size) / cu_count;
    float multiplier = 1.0;
    if (SizeInBitsOf(params.weights_type) <= 8) {
      multiplier = 16.0 / SizeInBitsOf(params.weights_type);
    }
    int x_size = 1;
    int y_size = 1;
    if (task_size_per_cu <= 2 * multiplier) {
      y_size = 128;
    } else if (task_size_per_cu <= 4 * multiplier) {
      y_size = 64;
    } else if (task_size_per_cu <= 8 * multiplier) {
      y_size = 32;
    } else if (task_size_per_cu <= 16 * multiplier) {
      y_size = 16;
    } else if (task_size_per_cu <= 32 * multiplier) {
      y_size = 8;
    } else if (task_size_per_cu <= 64 * multiplier) {
      y_size = 4;
    } else if (task_size_per_cu <= 128 * multiplier) {
      y_size = 2;
    } else {
      y_size = 1;
    }
    if (dst_slices >= 1024 * 16) {
      y_size = 1;
    }
    int wg_total_size = cu_count >= 16 ? 256 : 128;
    y_size = std::min(y_size, wg_total_size);
    x_size = wg_total_size / y_size;
    return int3(x_size, y_size, 1);
  }
  if (gpu_info.IsIntel() && gpu_info.IsApiOpenCl()) {
    int total_task_size = dst_slices;
    if (params.batched_weights) {
      if (params.runtime_batch_ids) {
        total_task_size *= params.runtime_batch_ids;
      } else {
        total_task_size *= weights_shape.h;
      }
    }
    const int cu_count = gpu_info.GetComputeUnitsCount();
    double task_size_per_cu = static_cast<double>(total_task_size) / cu_count;
    float multiplier = 1.0;
    if (SizeInBitsOf(params.weights_type) <= 8) {
      multiplier = 16.0 / SizeInBitsOf(params.weights_type);
    }
    int x_size = 1;
    int y_size = 1;
    if (task_size_per_cu <= 1 * multiplier) {
      y_size = 64;
    } else if (task_size_per_cu <= 2 * multiplier) {
      y_size = 32;
    } else if (task_size_per_cu <= 4 * multiplier) {
      y_size = 16;
    } else if (task_size_per_cu <= 8 * multiplier) {
      y_size = 8;
    } else if (task_size_per_cu <= 16 * multiplier) {
      y_size = 4;
    } else if (task_size_per_cu <= 32 * multiplier) {
      y_size = 2;
    } else {
      y_size = 2;
    }
    int wg_total_size = cu_count >= 64 ? 256 : 128;
    y_size = std::min(y_size, wg_total_size);
    x_size = wg_total_size / y_size;
    return int3(x_size, y_size, 1);
  }
  const int block_spatial =
      params.block_size.b * params.block_size.w * params.block_size.h;
  int wg_total_size = 32;
  float y_size_multiplier = 1;
  if (gpu_info.IsMali()) {
    if (gpu_info.mali_info.gpu_version == MaliGpu::kG925) {
      wg_total_size = 128;
      y_size_multiplier = 2;
    } else {
      wg_total_size = 64;
    }
  }
  if (gpu_info.IsPowerVR()) {
    wg_total_size = 128;
  }
  if (gpu_info.IsApple()) {
    const int cu_count = gpu_info.GetComputeUnitsCount();
    if (cu_count >= 24) {
      wg_total_size = 256;
      y_size_multiplier = 4;
    } else if (cu_count >= 12) {
      wg_total_size = 128;
      y_size_multiplier = 2;
    } else {
      wg_total_size = 64;
    }
  }
  if (gpu_info.IsAMD()) {
    wg_total_size = 128;
    const int cu_count = gpu_info.GetComputeUnitsCount();
    if (cu_count >= 32) {
      y_size_multiplier = 4;
    } else if (cu_count >= 16) {
      y_size_multiplier = 2;
    } else {
      y_size_multiplier = 1;
    }
  }
  if (gpu_info.IsIntel()) {
    wg_total_size = 128;
    y_size_multiplier = block_spatial > 1 ? 2 : 4;
  }
  if (gpu_info.IsAdreno()) {
    if (gpu_info.adreno_info.IsAdreno3xx() ||
        gpu_info.adreno_info.IsAdreno4xx() ||
        gpu_info.adreno_info.IsAdreno5xx()) {
      wg_total_size = 64;
    } else {
      wg_total_size = 128;
    }
    if (is_quantized &&
        gpu_info.adreno_info.generation >= AdrenoInfo::Generation::kGen7) {
      wg_total_size = 256;
    }
    if (gpu_info.adreno_info.IsLowEnd()) {
      wg_total_size = std::min(wg_total_size, 32);
    }
  }
  if (gpu_info.IsNvidia()) {
    const int cu_count = gpu_info.GetComputeUnitsCount();
    if (cu_count >= 32) {
      wg_total_size = 256;
      y_size_multiplier = 4;
    } else if (cu_count >= 16) {
      wg_total_size = 128;
      y_size_multiplier = 2;
    } else {
      wg_total_size = 128;
    }
  }
  int y_size = 8;
  if (is_quantized) {
    if (params.weights_type == DataType::INT2) {
      y_size = 32;
    } else {
      if (gpu_info.IsPowerVR() && params.weights_type == DataType::INT4) {
        y_size = 32;
      } else {
        y_size = 16;
      }
    }
  }
  if (dst_slices >= 512) {
    if (is_quantized) {
      if (params.weights_type == DataType::INT4 ||
          params.weights_type == DataType::INT2) {
        y_size = 16;
      } else {
        y_size = 8;
      }
    } else {
      y_size = 4;
    }
  }
  if (dst_slices >= 1024) {
    y_size = is_quantized ? 4 : 2;
  }
  if (dst_slices >= 1024 * 2) {
    if (is_quantized) {
      if (params.weights_type == DataType::INT4 ||
          params.weights_type == DataType::INT2) {
        y_size = 4;
      } else {
        y_size = 2;
      }
    } else {
      y_size = 1;
    }
  }
  if (dst_slices >= 1024 * 4) {
    y_size = std::min(2, y_size);
    if (is_quantized && params.weights_type == DataType::INT2) {
      y_size = 4;
    }
  }
  if (dst_slices >= 1024 * 8) {
    y_size = 1;
  }
  y_size *= y_size_multiplier;
  int x_size = wg_total_size / y_size;
  if (dst_slices <= 64 && x_size >= 8) {
    if (is_quantized) {
      x_size /= 2;
      y_size *= 2;
    } else {
      x_size = 2;
      y_size = std::min(32, wg_total_size / x_size);
    }
  }
  if (dst_slices <= 64 && gpu_info.IsApple()) {
    x_size = 1;
    y_size = 64;
  }
  return int3(x_size, y_size, 1);
}

int3 GetBlockSpatialCoords(int linear_spatial, const BHWC& shape) {
  int b_coord = linear_spatial % shape.b;
  linear_spatial /= shape.b;
  int x_coord = linear_spatial % shape.w;
  linear_spatial /= shape.w;
  int y_coord = linear_spatial % shape.h;
  return int3(b_coord, y_coord, x_coord);
}

void AddWeightsParams(const GpuInfo& gpu_info,
                      const Tensor<OHWI, DataType::FLOAT32>& weights_scale,
                      const Tensor<OHWI, DataType::FLOAT32>& weights_zero_point,
                      DataType dst_data_type, Arguments* args) {
  ABSL_CHECK(!weights_scale.data.empty());
  auto weights_scale_td =
      ScaleOrZeroPointToTensorDesc(gpu_info, weights_scale, dst_data_type);
  args->AddObject("weights_scale", std::make_unique<TensorDescriptor>(
                                       std::move(weights_scale_td)));
  if (!weights_zero_point.data.empty()) {
    auto weights_zero_point_td = ScaleOrZeroPointToTensorDesc(
        gpu_info, weights_zero_point, dst_data_type);
    args->AddObject(
        "weights_zero_point",
        std::make_unique<TensorDescriptor>(std::move(weights_zero_point_td)));
  }
}

DataType GetDataTypeForWeights(DataType weights_type) {
  if (weights_type == DataType::UINT8) {
    return DataType::INT8;
  }
  if (weights_type == DataType::UINT4) {
    return DataType::INT4;
  }
  if (weights_type == DataType::UINT2) {
    return DataType::INT2;
  }
  return weights_type;
}

}  // namespace

FullyConnected::FullyConnected(const TensorDescriptor& src,
                               const TensorDescriptor& dst,
                               CalculationsPrecision precision,
                               const GpuInfo& gpu_info,
                               const OHWI& weights_shape,
                               const WeightsDescription& weights_desc,
                               const ConvParams& conv_params) {
  conv_params_ = conv_params;
  AddSrcTensor("src_tensor", src);
  AddDstTensor("dst_tensor", dst);
  args_.AddInt("o_group_size", weights_desc.GetOutputGroupSize());
  const int dst_slices = DivideRoundUp(weights_shape.o, 4);
  args_.AddInt("o_groups",
               DivideRoundUp(dst_slices, weights_desc.GetOutputGroupSize()));
  if (conv_params.scale_zp_shape.i != 1) {
    const int src_slices = DivideRoundUp(weights_shape.i, 4);
    const int src_groups = conv_params.scale_zp_shape.i;
    args_.AddInt("src_groups", src_groups);
    args_.AddInt("src_group_size", src_slices / src_groups);
  }
  const DataType acc_type = precision == CalculationsPrecision::F16
                                ? DataType::FLOAT16
                                : DataType::FLOAT32;

  work_group_size_ =
      conv_params_.wg_size.x != 0
          ? conv_params_.wg_size
          : GetWorkGroupSize(conv_params_, gpu_info, acc_type, weights_shape);
  wg_reduction_ = work_group_size_.y != 1;
  const int scale_zp_group_size =
      DivideRoundUp(weights_shape.i / conv_params.scale_zp_shape.i, 4);
  code_ = GetFullyConnectedKernelCode(src, precision, gpu_info, weights_desc,
                                      scale_zp_group_size);

  const int wg_total_size = work_group_size_.x * work_group_size_.y;
  const std::string scope = gpu_info.IsApiMetal() && wg_total_size == 32 &&
                                    gpu_info.IsWaveSizeEqualTo32()
                                ? "SubGroup"
                                : "WorkGroup";
  absl::StrReplaceAll(
      {{"SType", ToUclDataType(dst.GetDataType(), 1)},
       {"Type", ToUclDataType(dst.GetDataType(), 4)},
       {"AccSType", ToUclDataType(acc_type, 1)},
       {"AccType", ToUclDataType(acc_type, 4)},
       {"WG_SIZE_X", std::to_string(work_group_size_.x)},
       {"WG_SIZE_Y", std::to_string(work_group_size_.y)},
       {"LOCAL_MEM_BARRIER", "ucl::SyncThreads<" + scope + ", Local>()"}},
      &code_);
  if (gpu_info.IsMali()) {
    compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
  if (gpu_info.IsAdreno()) {
    compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
  if (precision == CalculationsPrecision::F16 && gpu_info.IsIntel()) {
    compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
  if (gpu_info.IsMaleoon()) {
    compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
  if (gpu_info.IsPowerVR()) {
    compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
  if (src.GetDataType() == DataType::INT8) {
    if (gpu_info.SupportsExtension("cl_qcom_dot_product8")) {
      compiler_options_.push_back(CompilerOptions::kCl20);
    } else if (gpu_info.SupportsExtension("cl_khr_integer_dot_product")) {
      compiler_options_.push_back(CompilerOptions::kCl30);
    }
  }
}

bool UseFMA(const GpuInfo& gpu_info) {
  return gpu_info.IsApiWebGpu() ||
         (gpu_info.IsAMD() && gpu_info.IsApiOpenCl()) ||
         (gpu_info.IsMaleoon() && gpu_info.IsApiOpenCl());
}

std::string ReadWeightsAsFloat(const FullyConnected::ConvParams& conv_params,
                               const WeightsDescription& weights_desc) {
  std::string c;
  if (conv_params.runtime_check.ring_o_offset_index.has_value()) {
    // kOSpatialIOGroupO4I4 -> kBIOI4
    c += R"(
    int o0 = (dst_s * 4 + ring_o_offset) % ring_size;
    int o1 = (dst_s * 4 + 1 + ring_o_offset) % ring_size;
    int o2 = (dst_s * 4 + 2 + ring_o_offset) % ring_size;
    int o3 = (dst_s * 4 + 3 + ring_o_offset) % ring_size;
    int a0 = src_s * ring_size + o0;
    int a1 = src_s * ring_size + o1;
    int a2 = src_s * ring_size + o2;
    int a3 = src_s * ring_size + o3;
)";
    if (conv_params.batched_weights) {
      c += R"(
    a0 += weights_batch_id * ring_size * args.src_tensor.Slices();
    a1 += weights_batch_id * ring_size * args.src_tensor.Slices();
    a2 += weights_batch_id * ring_size * args.src_tensor.Slices();
    a3 += weights_batch_id * ring_size * args.src_tensor.Slices();
)";
    }
    c += R"(
    w0 = args.weights.Read(a0);
    w1 = args.weights.Read(a1);
    w2 = args.weights.Read(a2);
    w3 = args.weights.Read(a3);
)";
  } else if (conv_params.runtime_check.ring_i_offset_index.has_value()) {
    // kOSpatialIOGroupI4O4 -> kBIOI4O4
    c += R"(
    int i0 = (src_s * 4 + ring_i_offset) % ring_size;
    int i1 = (src_s * 4 + 1 + ring_i_offset) % ring_size;
    int i2 = (src_s * 4 + 2 + ring_i_offset) % ring_size;
    int i3 = (src_s * 4 + 3 + ring_i_offset) % ring_size;
    int a0 = ((i0 / 4) * args.dst_tensor.Slices() + dst_s) * 4 + i0 % 4;
    int a1 = ((i1 / 4) * args.dst_tensor.Slices() + dst_s) * 4 + i1 % 4;
    int a2 = ((i2 / 4) * args.dst_tensor.Slices() + dst_s) * 4 + i2 % 4;
    int a3 = ((i3 / 4) * args.dst_tensor.Slices() + dst_s) * 4 + i3 % 4;
)";
    if (conv_params.batched_weights) {
      c += R"(
    a0 += weights_batch_id * ring_size * args.dst_tensor.Slices();
    a1 += weights_batch_id * ring_size * args.dst_tensor.Slices();
    a2 += weights_batch_id * ring_size * args.dst_tensor.Slices();
    a3 += weights_batch_id * ring_size * args.dst_tensor.Slices();
)";
    }
    c += R"(
    w0 = args.weights.Read(a0);
    w1 = args.weights.Read(a1);
    w2 = args.weights.Read(a2);
    w3 = args.weights.Read(a3);
)";
  } else if (weights_desc.layout == WeightsLayout::kUnknown) {
    const std::string h_coord =
        conv_params.batched_weights ? "weights_batch_id" : "0";
    c += "    w0 = args.weights.Read<SType>(0, " + h_coord +
         ", src_s, dst_s * 4 + 0);\n";
    c += "    w1 = args.weights.Read<SType>(0, " + h_coord +
         ", src_s, dst_s * 4 + 1);\n";
    c += "    w2 = args.weights.Read<SType>(0, " + h_coord +
         ", src_s, dst_s * 4 + 2);\n";
    c += "    w3 = args.weights.Read<SType>(0, " + h_coord +
         ", src_s, dst_s * 4 + 3);\n";
  } else if (weights_desc.IsLinearLayout()) {
    c += "    int linear_i4o4 = src_s * args.dst_tensor.Slices() + dst_s;\n";
    if (conv_params.batched_weights) {
      c += "    linear_i4o4 += weights_batch_id * args.src_tensor.Slices() * "
           "args.dst_tensor.Slices();\n";
    }
    if (conv_params.sparse_2x4) {
      c += "    uint weights = args.weights.Read(linear_i4o4);\n";
      c += "    uint indexes = args.weights_indices.ReadAsU16(linear_i4o4);\n";
      c += "    ucl::U32Sparse2x4ToU4x16AsVec4x4<SType>(weights, indexes, "
           "w0, w1, w2, w3);\n";
    } else if (fc::IsQuantized(conv_params.weights_type)) {
      if (conv_params.weights_type == DataType::INT2) {
        c += "    uint w = args.weights.Read(linear_i4o4);\n";
        c += "    ucl::U32x1ToU2x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
      } else if (conv_params.weights_type == DataType::INT4) {
        c += "    uint2 w = args.weights.Read(linear_i4o4);\n";
        c += "    ucl::U32x2ToU4x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
      } else {
        c += "    uint4 w = args.weights.Read(linear_i4o4);\n";
        c += "    ucl::U32x4ToU8x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
      }
    } else {
      c += "    args.weights.ReadVec16AsVec4x4(w0, w1, w2, w3, "
           "linear_i4o4);\n";
    }
  } else {
    std::string x_c = "dst_s";
    std::string y_c = "src_s";
    if (conv_params.batched_weights) {
      y_c = "weights_batch_id * args.src_tensor.Slices() + src_s";
    }
    if (weights_desc.layout ==
        WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4) {
      c += "    w0 = args.weights0.Read<SType>(" + x_c + ", " + y_c + ");\n";
      c += "    w1 = args.weights1.Read<SType>(" + x_c + ", " + y_c + ");\n";
      c += "    w2 = args.weights2.Read<SType>(" + x_c + ", " + y_c + ");\n";
      c += "    w3 = args.weights3.Read<SType>(" + x_c + ", " + y_c + ");\n";
    } else if (conv_params.weights_type == DataType::INT8) {
      c += "    uint4 w = args.weights.Read(" + x_c + ", " + y_c + ");\n";
      c += "    ucl::U32x4ToU8x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
    } else if (conv_params.weights_type == DataType::INT4) {
      c += "    ushort4 w = args.weights.Read(" + x_c + ", " + y_c + ");\n";
      c += "    ucl::U16x4ToU4x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
    } else if (conv_params.weights_type == DataType::INT2) {
      c += "    uchar4 w = args.weights.Read(" + x_c + ", " + y_c + ");\n";
      c += "    ucl::U8x4ToU2x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
    }
  }
  return c;
}

std::string ReadWeightsAs4Uint8x4(const FullyConnected::ConvParams& conv_params,
                                  const WeightsDescription& weights_desc) {
  std::string c;
  std::string coords_2d;
  if (weights_desc.IsLinearLayout()) {
    c += "    int linear_i4o4 = src_s * args.dst_tensor.Slices() + dst_s;\n";
    if (conv_params.batched_weights) {
      c += "    linear_i4o4 += weights_batch_id * args.src_tensor.Slices() * "
           "args.dst_tensor.Slices();\n";
    }
  } else {
    std::string x_c = "dst_s";
    std::string y_c = "src_s";
    if (conv_params.batched_weights) {
      y_c = "weights_batch_id * args.src_tensor.Slices() + src_s";
    }
    coords_2d = x_c + ", " + y_c;
  }
  if (fc::IsQuantized(conv_params.weights_type)) {
    if (conv_params.weights_type == DataType::INT2) {
      if (weights_desc.IsLinearLayout()) {
        c += "    uint w = args.weights.Read(linear_i4o4);\n";
      } else {
        c += "    uchar4 wt = args.weights.Read(" + coords_2d + ");\n";
        c += "    uint w = ucl::Reinterpret<uchar4, uint>(wt);\n";
      }
      c += "    w0 = w & 50529027u;\n";
      c += "    w1 = (w >> 2u) & 50529027u;\n";
      c += "    w2 = (w >> 4u) & 50529027u;\n";
      c += "    w3 = (w >> 6u) & 50529027u;\n";
    } else if (conv_params.weights_type == DataType::INT4) {
      if (weights_desc.IsLinearLayout()) {
        c += "    uint2 w = args.weights.Read(linear_i4o4);\n";
      } else {
        c += "    ushort4 wt = args.weights.Read(" + coords_2d + ");\n";
        c += "    uint2 w = ucl::Reinterpret<ushort4, uint2>(wt);\n";
      }
      c += R"(
  uint t = (w.y ^ (w.x >> 16u)) & 0x0000FFFFu;

  uint t0 = w.x ^ (t << 16u);
  uint t1 = w.y ^ t;

  t = (t1 ^ (t0 >> 8u)) & 0x00FF00FFu;
  t1 ^= t;
  t0 ^= (t << 8u);

  w0 = t0 & 0x0F0F0F0Fu;
  w1 = (t0 >> 4u) & 0x0F0F0F0Fu;
  w2 = t1 & 0x0F0F0F0Fu;
  w3 = (t1 >> 4u) & 0x0F0F0F0Fu;
)";
    } else if (conv_params.weights_type == DataType::INT8) {
      if (weights_desc.IsLinearLayout()) {
        c += "    uint4 w = args.weights.Read(linear_i4o4);\n";
      } else {
        c += "    uint4 w = args.weights.Read(" + coords_2d + ");\n";
      }
      c += R"(
  uint t = (w.z ^ (w.x >> 16u)) & 0x0000FFFFu;
  w2 = w.z ^ t;
  w0 = w.x ^ (t << 16u);

  t = (w.w ^ (w.y >> 16u)) & 0x0000FFFFu;
  w3 = w.w ^ t;
  w1 = w.y ^ (t << 16u);

  t = (w1 ^ (w0 >> 8u)) & 0x00FF00FFu;
  w1 ^= t;
  w0 ^= (t << 8u);

  t = (w3 ^ (w2 >> 8u)) & 0x00FF00FFu;
  w3 ^= t;
  w2 ^= (t << 8u);
)";
    }
  }
  return c;
}

std::string FullyConnected::GetFullyConnectedKernelCode(
    const TensorDescriptor& src, CalculationsPrecision precision,
    const GpuInfo& gpu_info, const WeightsDescription& weights_desc,
    int scale_zp_group_size) {
  const int block_spatial = conv_params_.block_size.b *
                            conv_params_.block_size.w *
                            conv_params_.block_size.h;
  const bool int8_math = src.GetDataType() == DataType::INT8;
  const bool is_quantized = fc::IsQuantized(conv_params_.weights_type);

  std::string c;
  if (int8_math) {
    if (gpu_info.SupportsExtension("cl_qcom_dot_product8")) {
      c += "#pragma OPENCL EXTENSION cl_qcom_dot_product8 : enable\n";
    } else if (gpu_info.SupportsExtension("cl_khr_integer_dot_product")) {
      c += "#pragma OPENCL EXTENSION "
           "cl_khr_integer_dot_product : enable\n";
    }
  }
  c += "MAIN_FUNCTION($0) {\n";
  c += "  int dst_s = ucl::GetGlobalId<0>();\n";
  if (conv_params_.runtime_check.dst_end_ch_index.has_value()) {
    c += "  int dst_end_slice = " +
         conv_params_.runtime_check.GetRuntimeEndSlice(
             "args.params.Read(args.dst_end_ch_index)",
             "args.dst_tensor.Slices()") +
         ";\n";
  } else {
    c += "  int dst_end_slice = args.dst_tensor.Slices();\n";
  }
  if (conv_params_.runtime_check.ring_o_offset_index.has_value()) {
    c +=
        "  int ring_o_offset = args.params.Read(" +
        std::to_string(conv_params_.runtime_check.ring_o_offset_index.value()) +
        ");\n";
    c += "  int ring_size = " +
         std::to_string(conv_params_.runtime_check.ring_size.value()) + ";\n";
  }
  if (conv_params_.runtime_check.ring_i_offset_index.has_value()) {
    c +=
        "  int ring_i_offset = args.params.Read(" +
        std::to_string(conv_params_.runtime_check.ring_i_offset_index.value()) +
        ");\n";
    c += "  int ring_size = " +
         std::to_string(conv_params_.runtime_check.ring_size.value()) + ";\n";
  }
  if (wg_reduction_) {
    // We require uniform execution for the entire workgroup since we use a
    // memory barrier below.
    c += "  int dst_s_wg_offset = ucl::GetGroupId<0>() * "
         "ucl::GetGroupSize<0>();\n";
    c += "  if (dst_s_wg_offset >= dst_end_slice) return;\n";
  } else {
    c += "  if (dst_s >= dst_end_slice) return;\n";
  }
  if (conv_params_.batched_weights) {
    if (conv_params_.runtime_batch_ids) {
      c += "  int dst_h = ucl::GetGroupId<2>();\n";
      c += "  int weights_batch_id;\n";
      c += "  args.batch_ids.ReadPerChannel<int>(weights_batch_id, 0, 0, "
           "dst_h);\n";
    } else {
      c += "  int weights_batch_id = ucl::GetGroupId<2>();\n";
    }
  }
  if (conv_params_.runtime_check.packed_groups.has_value()) {
    c += "  int dst_w = ucl::GetGroupId<1>();\n";
    c += "  int w_group_size = args.params.Read(args.packed_params_offset + "
         "weights_batch_id);\n";
    c += "  int w_group_offset = args.params.Read(args.packed_params_offset + "
         "weights_batch_id + " +
         std::to_string(conv_params_.runtime_check.packed_groups->num_groups) +
         ");\n";
    c += "  int wg_first_w = dst_w * " +
         std::to_string(conv_params_.block_size.w) + ";\n";
    c += "  if (wg_first_w >= w_group_size) return;\n";
    c += "  dst_w = w_group_offset + dst_w * " +
         std::to_string(conv_params_.block_size.w) + ";\n";
  }
  for (int sp_id = 0; sp_id < block_spatial; ++sp_id) {
    const std::string r_name = "r_sp" + std::to_string(sp_id);
    if (int8_math) {
      c += "  int4 " + r_name + " = ucl::Init<int4>(0);\n";
      c += "  int " + r_name + "_sum = 0;\n";
    } else {
      c += "  AccType " + r_name + " = ucl::Init<AccType>(0.0f);\n";
    }
  }
  if (conv_params_.softmax_input_activation) {
    for (int i = 0; i < block_spatial; ++i) {
      const int3 bhw = GetBlockSpatialCoords(i, conv_params_.block_size);
      std::string y_coord = std::to_string(bhw.y);
      if (conv_params_.batched_weights) {
        y_coord = "weights_batch_id";
        if (conv_params_.runtime_batch_ids) {
          y_coord = "0";
        }
      }
      c += "  SType2 exp_val" + std::to_string(i) + " = args.src_exp.Read(" +
           std::to_string(bhw.z) + ", " + y_coord + ", 0, " +
           std::to_string(bhw.x) + ").xy;\n";
    }
  }
  std::string src_end_slices = "args.src_tensor.Slices()";
  if (conv_params_.runtime_check.src_end_ch_index.has_value()) {
    c += "  int src_end_slices = "
         "(args.params.Read(args.src_end_ch_index) + 3) / 4;\n";
    src_end_slices = "src_end_slices";
  }
  if (wg_reduction_) {
    c += "  int2 tid;\n";
    c += "  tid.x = ucl::GetLocalId<0>();\n";
    c += "  tid.y = ucl::GetLocalId<1>();\n";
    c += "  if (dst_s < args.dst_tensor.Slices()) {\n";
  }
  const bool is_block_quantized = fc::IsBlockQuantized(
      conv_params_.weights_type, conv_params_.scale_zp_shape);
  if (is_quantized && !int8_math) {
    if (is_block_quantized) {
      // block-wise quantization handled later
      c += "  Type w_scale;\n";
      c += "  Type w_bias;\n";
    } else if (fc::IsLinearQuantized(conv_params_.weights_type,
                                     conv_params_.scale_zp_shape)) {
      c += fc::ReadScaleZeroPointLinear(conv_params_.scale_zp_shape,
                                        conv_params_.has_zero_point,
                                        conv_params_.weights_type);
    } else if (fc::IsScalarQuantized(conv_params_.weights_type,
                                     conv_params_.scale_zp_shape)) {
      c += fc::ReadScaleZeroPointScalar(conv_params_.has_zero_point,
                                        conv_params_.weights_type);
    }
  }
  const std::string start_slice = wg_reduction_ ? "tid.y" : "0";
  const std::string slice_stride = wg_reduction_ ? "WG_SIZE_Y" : "1";
  if (is_block_quantized) {
    c += "  for (int src_group = " + start_slice +
         "; src_group < args.src_groups; src_group += " + slice_stride +
         ") {\n";
    c += fc::ReadScaleZeroPointBlock(conv_params_.scale_zp_shape,
                                     conv_params_.has_zero_point,
                                     conv_params_.weights_type);
    c += "  for (int src_sub_id = 0; src_sub_id < args.src_group_size; "
         "src_sub_id += 1) {\n";
    c += "    int src_s = src_group * args.src_group_size + src_sub_id;\n";
  } else {
    c += "  for (int src_s = " + start_slice + "; src_s < " + src_end_slices +
         "; src_s += " + slice_stride + ") {\n";
  }
  for (int i = 0; i < block_spatial; ++i) {
    const int3 bhw = GetBlockSpatialCoords(i, conv_params_.block_size);
    const std::string val_name = "v" + std::to_string(i);
    std::string y_coord = std::to_string(bhw.y);
    if (conv_params_.batched_weights) {
      y_coord = "weights_batch_id";
      if (conv_params_.runtime_batch_ids) {
        y_coord = "min(dst_h, args.src_tensor.Height() - 1)";
      }
    }
    std::string x_coord = std::to_string(bhw.z);
    if (conv_params_.runtime_check.packed_groups.has_value()) {
      x_coord = "dst_w + " + x_coord;
      if (!src.CanReadOutOfBorder(Axis::WIDTH, gpu_info)) {
        x_coord = "min(" + x_coord + ", args.src_tensor.Width() - 1)";
      }
      y_coord = "0";
    }
    const std::string read_expression = "args.src_tensor.Read(" + x_coord +
                                        ", " + y_coord + ", src_s, " +
                                        std::to_string(bhw.x) + ")";
    if (int8_math) {
      c += "    uint " + val_name + ";\n";
      c += "    { char4 t = " + read_expression + "; " + val_name +
           " = ucl::Reinterpret<char4, uint>(t);}\n";
    } else {
      c += "    Type " + val_name + " = " + read_expression + ";\n";
    }
    if (conv_params_.softmax_input_activation) {
      const std::string id = std::to_string(i);
      c += "  " + val_name + " = ucl::Exp<Type>(" + val_name + " - exp_val" +
           id + ".y) * exp_val" + id + ".x;\n";
    }
  }

  if (!int8_math) {
    c += "    Type w0, w1, w2, w3;\n";
    c += ReadWeightsAsFloat(conv_params_, weights_desc);
    const bool isI4O4 = weights_desc.IsI4O4() || conv_params_.sparse_2x4;
    const bool use_fma = UseFMA(gpu_info);
    if (is_quantized) {
      const std::string w_scale = "w_scale";
      const std::string w_bias = "w_bias";
      c += fc::WeightsScaleAddBias(w_scale, w_bias, isI4O4, use_fma);
    }
    for (int i = 0; i < block_spatial; ++i) {
      const std::string r_name = "r_sp" + std::to_string(i);
      const std::string src_name = "v" + std::to_string(i);
      c += fc::AccumulateFloat(r_name, src_name, precision, isI4O4, use_fma);
    }
  } else {
    c += "    uint w0, w1, w2, w3;\n";
    c += ReadWeightsAs4Uint8x4(conv_params_, weights_desc);
    for (int i = 0; i < block_spatial; ++i) {
      const std::string acc_name = "r_sp" + std::to_string(i);
      const std::string src_name = "v" + std::to_string(i);
      c += fc::AccumulateUint(acc_name, src_name, gpu_info);
    }
  }
  if (gpu_info.IsApple() && gpu_info.IsApiMetal() &&
      gpu_info.apple_info.IsMSeries() && is_quantized) {
    // In some cases, on Metal with Apple GPUs, kernel with big dst channels
    // size shows unexpected slowdown. This workaround helps to restore
    // performance.
    const int dst_slices = DivideRoundUp(conv_params_.scale_zp_shape.o, 4);
    if (dst_slices >= 1024 * 16 && dst_slices % work_group_size_.x == 0 &&
        work_group_size_.y == 1) {
      c += "    ucl::SyncThreads<WorkGroup, None>();\n";
    }
  }
  if (is_block_quantized) {
    c += "  } // end for block loop\n";
  }
  c += "  } // end for loop\n";
  if (int8_math) {
    for (int i = 0; i < block_spatial; ++i) {
      const std::string r_name = "r_sp" + std::to_string(i);
      c += fc::AdjustUintSum(r_name, conv_params_.weights_type);
    }
  }
  if (wg_reduction_) {
    c += "  } // end if condition\n";
    int local_patch_size = block_spatial;
    int data_type_size =
        precision == CalculationsPrecision::F16 && gpu_info.SupportsFP16() ? 2
                                                                           : 4;
    int workgroup_storage_size = work_group_size_.x * work_group_size_.y *
                                 local_patch_size * data_type_size * 4;
    if (local_patch_size > 8 ||
        (gpu_info.IsApiWebGpu() &&
         workgroup_storage_size >
             gpu_info.webgpu_info.max_compute_workgroup_storage_size)) {
      local_patch_size = 1;
    }
    if (gpu_info.IsAdreno() && gpu_info.adreno_info.IsLowEnd()) {
      local_patch_size = 1;
    }
    if (int8_math) {
      c += "  __local int4 temp";
    } else {
      c += "  __local AccType temp";
    }
    if (local_patch_size != 1) {
      c += "[" + std::to_string(local_patch_size) + "]";
    }
    c += "[" + std::to_string(work_group_size_.x * work_group_size_.y) + "];\n";

    const int upload_groups = DivideRoundUp(block_spatial, local_patch_size);
    for (int group = 0; group < upload_groups; ++group) {
      const int first = group * local_patch_size;
      const int last =
          std::min(block_spatial - 1, first + local_patch_size - 1);
      for (int sp_id = first; sp_id <= last; ++sp_id) {
        const std::string local_mem =
            local_patch_size == 1 ? "temp"
                                  : "temp[" + std::to_string(sp_id) + "]";
        c += "  " + local_mem + "[tid.x * WG_SIZE_Y + tid.y] = r_sp" +
             std::to_string(sp_id) + ";\n";
      }
      c += "  for (int ystride = WG_SIZE_Y / 2; ystride > 0; ystride /= 2) {\n";
      c += "    LOCAL_MEM_BARRIER;\n";
      c += "    if (tid.y < ystride) {\n";
      for (int sp_id = first; sp_id <= last; ++sp_id) {
        const std::string local_mem =
            local_patch_size == 1 ? "temp"
                                  : "temp[" + std::to_string(sp_id) + "]";
        c += "      r_sp" + std::to_string(sp_id) + " += " + local_mem +
             "[tid.x * WG_SIZE_Y + tid.y + ystride];\n";
        c += "      " + local_mem + "[tid.x * WG_SIZE_Y + tid.y] = r_sp" +
             std::to_string(sp_id) + ";\n";
      }
      c += "    }\n";
      c += "  }\n";
      if (group != upload_groups - 1) {
        c += "  LOCAL_MEM_BARRIER;\n";
      }
    }
    c += "  if (dst_s >= args.dst_tensor.Slices()) return;\n";
    c += "  if (tid.y != 0) return;\n";
  }
  c += "  {\n";
  if (conv_params_.has_bias) {
    c += "  Type bias_value = args.biases.Read(dst_s);\n";
  }
  for (int sp_id = 0; sp_id < block_spatial; ++sp_id) {
    const std::string r_name = "r_sp" + std::to_string(sp_id);
    const int3 bhw = GetBlockSpatialCoords(sp_id, conv_params_.block_size);
    std::string y_coord = std::to_string(bhw.y);
    if (conv_params_.batched_weights) {
      y_coord = "weights_batch_id";
      if (conv_params_.runtime_batch_ids) {
        y_coord = "dst_h";
      }
    }
    std::string x_coord = std::to_string(bhw.z);
    if (conv_params_.runtime_check.packed_groups.has_value()) {
      x_coord = "dst_w + " + x_coord;
      y_coord = "0";
      c += "  if (" + x_coord + " < w_group_offset + w_group_size) {\n";
    } else if (block_spatial != 1) {
      c +=
          "  if (" + std::to_string(-sp_id) + " < args.dst_tensor.Width()) {\n";
    }
    c += "  args.dst_tensor::type res_value = "
         "ucl::Convert<args.dst_tensor::type>(" +
         r_name + ");\n";
    if (conv_params_.has_bias) {
      c += "  res_value += bias_value;\n";
    }
    c += "  args.dst_tensor.Write(res_value, " + x_coord + ", " + y_coord +
         ", dst_s, " + std::to_string(bhw.x) + ");\n";
    if (conv_params_.runtime_check.packed_groups.has_value() ||
        block_spatial != 1) {
      c += "  }\n";
    }
  }
  c += "  }\n";
  c += "}\n";
  return c;
}

int3 FullyConnected::GetGridSize() const {
  int w_batch_size = conv_params_.batched_weights ? dst_[0]->Height() : 1;
  int w_groups = 1;
  if (conv_params_.runtime_check.packed_groups.has_value()) {
    w_groups =
        DivideRoundUp(conv_params_.runtime_check.packed_groups->max_group_size,
                      conv_params_.block_size.w);
    w_batch_size = conv_params_.runtime_check.packed_groups->num_groups;
  }
  return int3(dst_[0]->Slices(), work_group_size_.y * w_groups, w_batch_size);
}

int GetRecommendedMaxTotalSpatialSize(const GpuInfo& gpu_info,
                                      CalculationsPrecision precision) {
  int base_max_size = 4;
  if (precision == CalculationsPrecision::F16) {
    base_max_size *= 2;
  }
  if (!gpu_info.IsMali()) {
    base_max_size *= 2;
  }
  if (gpu_info.IsAdreno() &&
      gpu_info.adreno_info.generation >= AdrenoInfo::Generation::kGen7) {
    base_max_size *= 2;
  }
  base_max_size = std::min(base_max_size, 16);
  return base_max_size;
}

BHWC GetBlockSize(const BHWC* dst_shape_ptr, bool batched_weights) {
  if (!dst_shape_ptr) {
    return BHWC(1, 1, 1, 1);
  }
  BHWC block_size = *dst_shape_ptr;
  block_size.c = 1;
  if (batched_weights) {
    block_size.h = 1;
  }
  return block_size;
}

FullyConnected CreateFullyConnected(const GpuInfo& gpu_info,
                                    const OperationDef& definition,
                                    CalculationsPrecision precision,
                                    const FullyConnectedAttributes& attr,
                                    const BHWC* dst_shape_ptr,
                                    const int3* wg_size) {
  WeightsDescription weights_desc;
  weights_desc.type = DeduceDataTypeFromPrecision(precision);
  if (UseBufferForWeights(gpu_info, attr.weights.shape)) {
    weights_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
    weights_desc.output_group_size = DivideRoundUp(attr.weights.shape.o, 4);
  } else {
    weights_desc.layout = WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4;
    weights_desc.output_group_size = 1;
  }
  FullyConnected::ConvParams conv_params;
  conv_params.weights_type = weights_desc.type;
  if (wg_size) {
    conv_params.wg_size = *wg_size;
  }
  conv_params.has_bias = !attr.bias.data.empty();
  conv_params.block_size =
      GetBlockSize(dst_shape_ptr, conv_params.batched_weights);
  FullyConnected result(definition.src_tensors[0], definition.dst_tensors[0],
                        precision, gpu_info, attr.weights.shape, weights_desc,
                        conv_params);

  result.UploadWeights(attr.weights, weights_desc);
  if (conv_params.has_bias) {
    TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
        gpu_info, definition.src_tensors[0].GetDataType(), attr.bias);
    result.args_.AddObject("biases", std::make_unique<TensorDescriptor>(
                                         std::move(bias_tensor_desc)));
  }

  return result;
}

absl::StatusOr<FullyConnected> CreateFullyConnectedWeightsAreSpatialTensor(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision, const OHWI& weights_shape,
    const TensorDescriptor* bias, const BHWC* dst_shape_ptr,
    const int3* wg_size, const ConvRuntimeCheckDesc& runtime_check) {
  if (!IsFullyConnectedWeightsAreSpatialTensorSupported(weights_shape)) {
    return absl::InvalidArgumentError("Unsupported weights shape.");
  }
  FullyConnected::ConvParams conv_params;
  conv_params.weights_type = definition.src_tensors[1].GetDataType();
  if (wg_size) {
    conv_params.wg_size = *wg_size;
  }
  conv_params.batched_weights = weights_shape.h != 1;
  conv_params.has_bias = bias != nullptr;
  conv_params.runtime_check = runtime_check;
  conv_params.block_size =
      GetBlockSize(dst_shape_ptr, conv_params.batched_weights);
  WeightsDescription weights_desc;
  weights_desc.type = definition.src_tensors[1].GetDataType();
  weights_desc.layout = WeightsLayout::kUnknown;  // Using Spatial tensor as
                                                  // weights.
  FullyConnected result(definition.src_tensors[0], definition.dst_tensors[0],
                        precision, gpu_info, weights_shape, weights_desc,
                        conv_params);
  result.AddSrcTensor("weights", definition.src_tensors[1]);
  if (bias) {
    result.AddSrcTensor("biases", *bias);
  }
  fc::AddRuntimeParam(runtime_check, &result);

  return result;
}

WeightsDescription GetFullyConnectedWeightsDesc(DataType weights_type,
                                                const OHWI& weights_shape) {
  const int dst_depth = DivideRoundUp(weights_shape.o, 4);
  WeightsDescription weights_desc;
  weights_desc.type = weights_type;
  weights_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
  weights_desc.output_group_size = dst_depth;
  return weights_desc;
}

WeightsDescription GetFullyConnectedWeightsDesc(const GpuInfo& gpu_info,
                                                DataType weights_type,
                                                const OHWI& weights_shape) {
  WeightsDescription weights_desc;
  weights_desc.type = weights_type;
  weights_desc.output_group_size = DivideRoundUp(weights_shape.o, 4);
  if (UseBufferForWeights(gpu_info, weights_shape)) {
    weights_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
  } else {
    weights_desc.layout = WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4;
  }
  return weights_desc;
}

absl::StatusOr<FullyConnected> CreateFullyConnectedExternalWeights(
    const GpuInfo& gpu_info, CalculationsPrecision precision,
    const TensorDescriptor& src, const TensorDescriptor& dst,
    const ExternalWeights& weights, const TensorDescriptor* bias,
    const BHWC* dst_shape_ptr, const TensorDescriptor* src_exp,
    const ConvRuntimeCheckDesc& runtime_check, const int3* wg_size) {
  const auto& weights_desc = weights.desc;
  const auto& weights_shape = weights.shape;

  if (weights_desc.type == DataType::FLOAT32 ||
      weights_desc.type == DataType::FLOAT16) {
    if (weights_desc.type != DeduceDataTypeFromPrecision(precision)) {
      return absl::InvalidArgumentError("Unsupported WeightsDescription type.");
    }
    if (weights_desc.layout == WeightsLayout::kOSpatialIOGroupI4O4 ||
        weights_desc.layout == WeightsLayout::kOSpatialIOGroupO4I4 ||
        weights_desc.layout ==
            WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4) {
      if (weights_desc.output_group_size != DivideRoundUp(weights_shape.o, 4)) {
        return absl::InvalidArgumentError(
            "Unsupported WeightsDescription output_group_size.");
      }
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported WeightsDescription layout: ",
                       ToString(weights_desc.layout)));
    }
  }

  FullyConnected::ConvParams conv_params;
  conv_params.weights_type = GetDataTypeForWeights(weights_desc.type);
  if (weights.scale) {
    conv_params.scale_zp_shape = weights.scale_zp_shape;
  }
  conv_params.batched_weights = weights_shape.h != 1;
  conv_params.softmax_input_activation = src_exp != nullptr;
  conv_params.has_bias = bias != nullptr;
  conv_params.has_zero_point =
      weights.zero_point != nullptr || weights.scalar_zero_point.has_value();
  conv_params.runtime_check = runtime_check;
  conv_params.block_size =
      GetBlockSize(dst_shape_ptr, conv_params.batched_weights);
  if (runtime_check.packed_groups.has_value()) {
    conv_params.block_size = BHWC(1, 1, 1, 1);
    conv_params.block_size.w = 4;
    if (dst_shape_ptr) {
      const int average_task_size = DivideRoundUp(
          dst_shape_ptr->w, runtime_check.packed_groups->num_groups);
      const int max_tile =
          GetRecommendedMaxTotalSpatialSize(gpu_info, precision);
      conv_params.block_size.w = std::min(max_tile, average_task_size);
    }
    conv_params.wg_size = int3(64, 1, 1);
  }
  if (wg_size) {
    conv_params.wg_size = *wg_size;
  }
  FullyConnected result(src, dst, precision, gpu_info, weights_shape,
                        weights_desc, conv_params);

  const bool vec4_elements =
      conv_params.runtime_check.ring_o_offset_index.has_value() ||
      conv_params.runtime_check.ring_i_offset_index.has_value();
  const int vec_size = vec4_elements ? 4 : 16;
  fc::AddWeightsArguments(weights, vec_size, &result);

  if (bias) {
    result.AddSrcTensor("biases", *bias);
  }

  if (src_exp) {
    result.AddSrcTensor("src_exp", *src_exp);
  }

  fc::AddRuntimeParam(runtime_check, &result);
  return result;
}

absl::StatusOr<FullyConnected> CreateFullyConnectedWeightsBatchIds(
    const GpuInfo& gpu_info, CalculationsPrecision precision,
    const TensorDescriptor& src, const TensorDescriptor& batch_ids,
    const TensorDescriptor& dst, const ExternalWeights& weights,
    const TensorDescriptor* bias, const BHWC* dst_shape_ptr) {
  const auto& weights_desc = weights.desc;
  const auto& weights_shape = weights.shape;

  FullyConnected::ConvParams conv_params;
  conv_params.weights_type = GetDataTypeForWeights(weights_desc.type);
  if (weights.scale) {
    conv_params.scale_zp_shape = weights.scale_zp_shape;
  }
  conv_params.batched_weights = weights_shape.h != 1;
  conv_params.has_bias = bias != nullptr;
  conv_params.has_zero_point =
      weights.zero_point != nullptr || weights.scalar_zero_point.has_value();
  conv_params.runtime_batch_ids = dst_shape_ptr ? dst_shape_ptr->h : 1;
  conv_params.block_size =
      GetBlockSize(dst_shape_ptr, conv_params.batched_weights);

  FullyConnected result(src, dst, precision, gpu_info, weights_shape,
                        weights_desc, conv_params);

  result.AddSrcTensor("batch_ids", batch_ids);
  const bool vec4_elements =
      conv_params.runtime_check.ring_o_offset_index.has_value() ||
      conv_params.runtime_check.ring_i_offset_index.has_value();
  const int vec_size = vec4_elements ? 4 : 16;
  fc::AddWeightsArguments(weights, vec_size, &result);

  if (bias) {
    result.AddSrcTensor("biases", *bias);
  }

  return result;
}

WeightsDescription GetFullyConnectedInt8WeightsDesc(const GpuInfo& gpu_info,
                                                    const OHWI& weights_shape,
                                                    bool prefer_textures) {
  WeightsDescription weights_desc;
  weights_desc.type = DataType::UINT8;
  if (UseBufferForIntWeights(gpu_info, /*int_bit_size=*/8, weights_shape,
                             prefer_textures)) {
    weights_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
  } else {
    weights_desc.layout = WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;
  }
  weights_desc.output_group_size = DivideRoundUp(weights_shape.o, 4);
  return weights_desc;
}

WeightsDescription GetFullyConnectedInt4WeightsDesc(const GpuInfo& gpu_info,
                                                    const OHWI& weights_shape,
                                                    bool prefer_textures) {
  WeightsDescription weights_desc;
  weights_desc.type = DataType::UINT4;
  if (UseBufferForIntWeights(gpu_info, /*int_bit_size=*/4, weights_shape,
                             prefer_textures)) {
    weights_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
  } else {
    weights_desc.layout = WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;
  }
  weights_desc.output_group_size = DivideRoundUp(weights_shape.o, 4);
  return weights_desc;
}

WeightsDescription GetFullyConnectedInt2WeightsDesc(const GpuInfo& gpu_info,
                                                    const OHWI& weights_shape,
                                                    bool prefer_textures) {
  WeightsDescription weights_desc;
  weights_desc.type = DataType::UINT2;
  if (UseBufferForIntWeights(gpu_info, /*int_bit_size=*/2, weights_shape,
                             prefer_textures)) {
    weights_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
  } else {
    weights_desc.layout = WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;
  }
  weights_desc.output_group_size = DivideRoundUp(weights_shape.o, 4);
  return weights_desc;
}

bool SupportsFullyConnectedUint8Math(const GpuInfo& gpu_info) {
  return gpu_info.SupportsExtension("cl_qcom_dot_product8") ||
         gpu_info.SupportsExtension("cl_khr_integer_dot_product");
}

FullyConnected CreateFullyConnectedInt4Sparse2x4(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const Tensor<OHWI, DataType::INT8>& weights,
    const Tensor<OHWI, DataType::UINT8>& weights_indices,
    const Tensor<OHWI, DataType::FLOAT32>& weights_scale,
    const Tensor<OHWI, DataType::FLOAT32>& weights_zero_point,
    const Tensor<Linear, DataType::FLOAT32>& biases, const BHWC* dst_shape_ptr,
    const int3* wg_size) {
  OHWI weights_shape_dense = weights.shape;
  weights_shape_dense.i *= 2;
  WeightsDescription weights_desc;
  weights_desc.type = DataType::UINT4;
  weights_desc.layout = WeightsLayout::kCustomGroups;
  weights_desc.group_sizes = {
      {Axis::OUTPUT_CHANNELS, 4},
      {Axis::INPUT_CHANNELS, 2},
      {Axis::OUTPUT_CHANNELS, DivideRoundUp(weights_shape_dense.o, 4)},
      {Axis::INPUT_CHANNELS, 0},
      {Axis::OUTPUT_CHANNELS, 0},
  };
  FullyConnected::ConvParams conv_params;
  conv_params.weights_type = DataType::INT4;
  conv_params.scale_zp_shape = weights_scale.shape;
  if (weights_scale.shape.i != 1) {
    conv_params.has_zero_point = false;
  }
  if (wg_size) {
    conv_params.wg_size = *wg_size;
  }
  conv_params.has_bias = !biases.data.empty();
  conv_params.sparse_2x4 = true;
  conv_params.block_size =
      GetBlockSize(dst_shape_ptr, conv_params.batched_weights);
  FullyConnected result(definition.src_tensors[0], definition.dst_tensors[0],
                        precision, gpu_info, weights_shape_dense, weights_desc,
                        conv_params);
  {
    const int elements_count =
        GetTotalElementsCountForLayout(weights_desc, weights.shape);
    {
      std::vector<uint8_t> weights_data(elements_count / 2);
      RearrangeWeightsInt8AsUint4(weights, weights_desc,
                                  absl::MakeSpan(weights_data),
                                  /*shift_value=*/8, /*pad_value=*/8u);

      BufferDescriptor buffer_desc;
      buffer_desc.element_type = DataType::UINT32;
      buffer_desc.element_size = 1;
      buffer_desc.size = weights_data.size();
      buffer_desc.data = std::move(weights_data);
      result.args_.AddObject("weights", std::make_unique<BufferDescriptor>(
                                            std::move(buffer_desc)));
    }
    {
      weights_desc.type = DataType::UINT2;
      std::vector<uint8_t> weights_indices_data(elements_count / 4);
      RearrangeWeightsUint2(weights_indices, weights_desc,
                            absl::MakeSpan(weights_indices_data));

      BufferDescriptor buffer_desc;
      buffer_desc.element_type = DataType::UINT32;
      buffer_desc.element_size = 1;
      buffer_desc.size = weights_indices_data.size();
      buffer_desc.data = std::move(weights_indices_data);
      result.args_.AddObject(
          "weights_indices",
          std::make_unique<BufferDescriptor>(std::move(buffer_desc)));
    }
  }

  {
    const DataType type = definition.dst_tensors[0].GetDataType();
    AddWeightsParams(gpu_info, weights_scale, weights_zero_point, type,
                     &result.args_);
    if (conv_params.has_bias) {
      TensorDescriptor bias_tensor_desc =
          CreateConstantLinearTensorDescriptor(gpu_info, type, biases);
      result.args_.AddObject("biases", std::make_unique<TensorDescriptor>(
                                           std::move(bias_tensor_desc)));
    }
  }

  return result;
}

}  // namespace ml_drift
