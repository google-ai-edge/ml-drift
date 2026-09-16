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

#include "ml_drift/common/kernels/fully_connected_util.h"

#include <string>

#include "absl/strings/substitute.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/types.h"

namespace ml_drift {
namespace fc {
namespace {
inline int GetRangeShift(DataType type) {
  if (type == DataType::INT8) {
    return 128;
  } else if (type == DataType::INT4) {
    return 8;
  } else if (type == DataType::INT2) {
    return 2;
  } else {
    return 0;
  }
}
}  // namespace

void AddRuntimeParam(const ConvRuntimeCheckDesc& runtime_check,
                     GPUOperation* op) {
  bool has_runtime_check = false;
  if (runtime_check.src_end_ch_index.has_value()) {
    op->args_.AddInt("src_end_ch_index", *runtime_check.src_end_ch_index);
    has_runtime_check = true;
  }
  if (runtime_check.dst_end_ch_index.has_value()) {
    op->args_.AddInt("dst_end_ch_index", *runtime_check.dst_end_ch_index);
    has_runtime_check = true;
  }
  if (runtime_check.packed_groups.has_value()) {
    op->args_.AddInt("packed_params_offset",
                     runtime_check.packed_groups->params_offset);
    has_runtime_check = true;
  }
  if (runtime_check.ring_o_offset_index.has_value()) {
    op->args_.AddInt("ring_o_offset_index", *runtime_check.ring_o_offset_index);
    has_runtime_check = true;
  }
  if (runtime_check.ring_i_offset_index.has_value()) {
    op->args_.AddInt("ring_i_offset_index", *runtime_check.ring_i_offset_index);
    has_runtime_check = true;
  }
  if (has_runtime_check) {
    BufferDescriptor desc;
    desc.element_type = DataType::INT32;
    desc.element_size = 1;
    op->AddSrcBuffer("params", desc);
  }
}

std::string ReadScaleZeroPointBlock(const std::string& o_slice,
                                    const OHWI& scale_zp_shape,
                                    bool has_zero_point,
                                    DataType weights_type) {
  std::string c;
  const int range_shift = GetRangeShift(weights_type);
  std::string coords = o_slice + ", 0, src_group_id";
  if (scale_zp_shape.h != 1) {
    coords = o_slice + ", w_batch_id, src_group_id";
  }
  c += "    w_scale = args.weights_scale.Read(" + coords + ");\n";
  if (has_zero_point) {
    c += "    Type w_zp = args.weights_zero_point.Read(" + coords + ");\n";
    c += "    w_bias = -w_scale * (ucl::Init<Type>(" +
         std::to_string(range_shift) + ") + w_zp);\n";
  } else {
    c += "    w_bias = -w_scale * ucl::Init<Type>(" +
         std::to_string(range_shift) + ".0f);\n";
  }
  return c;
}

std::string ReadScaleZeroPointLinear(const std::string& o_slice,
                                     const OHWI& scale_zp_shape,
                                     bool has_zero_point,
                                     DataType weights_type) {
  std::string c;
  const int range_shift = GetRangeShift(weights_type);
  std::string coords = o_slice;
  if (scale_zp_shape.h != 1) {
    coords += ", w_batch_id, 0";
  }
  c += "  w_scale = args.weights_scale.Read(" + coords + ");\n";
  if (has_zero_point) {
    c += "  Type w_zp = args.weights_zero_point.Read(" + coords + ");\n";
    c += "  w_bias = -w_scale * (ucl::Init<Type>(" +
         std::to_string(range_shift) + ") + w_zp);\n";
  } else {
    c += "  w_bias = -w_scale * (ucl::Init<Type>(" +
         std::to_string(range_shift) + ".0f));\n";
  }
  return c;
}

std::string ReadScaleZeroPointScalar(bool has_zero_point,
                                     DataType weights_type) {
  std::string c;
  const int range_shift = GetRangeShift(weights_type);
  c += "  w_scale = "
       "ucl::Convert<Type>(ucl::Init<float4>(args.scale));\n";
  if (has_zero_point) {
    c += "  Type w_zp = "
         "ucl::Convert<Type>(ucl::Init<float4>(args.zero_point));\n";
    c += "  w_bias = -w_scale * (ucl::Init<Type>(" +
         std::to_string(range_shift) + ") + w_zp);\n";
  } else {
    c += "  w_bias = -w_scale * (ucl::Init<Type>(" +
         std::to_string(range_shift) + ".0f));\n";
  }
  return c;
}

std::string WeightsScaleAddBias(const std::string& w_scale,
                                const std::string& w_bias, bool isI4O4,
                                bool use_fma) {
  std::string c;
  if (isI4O4) {
    if (use_fma) {
      c += "    w0 = fma(w0, " + w_scale + ", " + w_bias + ");\n";
      c += "    w1 = fma(w1, " + w_scale + ", " + w_bias + ");\n";
      c += "    w2 = fma(w2, " + w_scale + ", " + w_bias + ");\n";
      c += "    w3 = fma(w3, " + w_scale + ", " + w_bias + ");\n";
    } else {
      c += "    w0 = w0 * " + w_scale + " + " + w_bias + ";\n";
      c += "    w1 = w1 * " + w_scale + " + " + w_bias + ";\n";
      c += "    w2 = w2 * " + w_scale + " + " + w_bias + ";\n";
      c += "    w3 = w3 * " + w_scale + " + " + w_bias + ";\n";
    }
  } else {
    c += "    w0 = w0 * " + w_scale + ".x + " + w_bias + ".x;\n";
    c += "    w1 = w1 * " + w_scale + ".y + " + w_bias + ".y;\n";
    c += "    w2 = w2 * " + w_scale + ".z + " + w_bias + ".z;\n";
    c += "    w3 = w3 * " + w_scale + ".w + " + w_bias + ".w;\n";
  }
  return c;
}

std::string AccumulateFloat(const std::string& r_name,
                            const std::string& src_name,
                            CalculationsPrecision precision, bool isI4O4,
                            bool use_fma) {
  std::string c;
  if (precision != CalculationsPrecision::F32_F16) {
    if (isI4O4) {
      if (use_fma) {
        c += "    $0 = fma(ucl::Init<Type>($1.x), w0, $0);\n";
        c += "    $0 = fma(ucl::Init<Type>($1.y), w1, $0);\n";
        c += "    $0 = fma(ucl::Init<Type>($1.z), w2, $0);\n";
        c += "    $0 = fma(ucl::Init<Type>($1.w), w3, $0);\n";
      } else {
        c += "    $0 += $1.x * w0;\n";
        c += "    $0 += $1.y * w1;\n";
        c += "    $0 += $1.z * w2;\n";
        c += "    $0 += $1.w * w3;\n";
      }
    } else {
      c += "    $0.x += dot($1, w0);\n";
      c += "    $0.y += dot($1, w1);\n";
      c += "    $0.z += dot($1, w2);\n";
      c += "    $0.w += dot($1, w3);\n";
    }
  } else {
    if (isI4O4) {
      c += "    $0 += ucl::Convert<AccType>($1.x * w0 + $1.y * w1 + $1.z * w2 "
           "+ $1.w * w3);\n";
    } else {
      c += "    $0.x += ucl::Convert<AccSType>(dot($1, w0));\n";
      c += "    $0.y += ucl::Convert<AccSType>(dot($1, w1));\n";
      c += "    $0.z += ucl::Convert<AccSType>(dot($1, w2));\n";
      c += "    $0.w += ucl::Convert<AccSType>(dot($1, w3));\n";
    }
  }
  return absl::Substitute(c, r_name, src_name);
}

std::string AccumulateUint(const std::string& r_name,
                           const std::string& src_name,
                           const GpuInfo& gpu_info) {
  std::string c = R"(
  $0.x = $2($1, w0, $0.x);
  $0.y = $2($1, w1, $0.y);
  $0.z = $2($1, w2, $0.z);
  $0.w = $2($1, w3, $0.w);
  $3 = $2($1, 16843009u, $3);
  )";
  std::string acc_func;
  if (gpu_info.SupportsExtension("cl_qcom_dot_product8")) {
    acc_func = "qcom_dot8_acc";
  } else if (gpu_info.SupportsExtension("cl_khr_integer_dot_product")) {
    acc_func = "dot_acc_sat_4x8packed_su_int";
  }
  const std::string sum_name = r_name + "_sum";
  return absl::Substitute(c, r_name, src_name, acc_func, sum_name);
}

std::string AdjustUintSum(const std::string& r_name, DataType weights_type) {
  return absl::Substitute(
      "  $0 -= $0_sum * " + std::to_string(GetRangeShift(weights_type)) + ";\n",
      r_name);
}

std::string GetActiveDstSlices(const ConvRuntimeCheckDesc& runtime_check) {
  std::string c;
  if (runtime_check.dst_end_ch_index.has_value()) {
    c += "  int dst_end_slice = " +
         runtime_check.GetRuntimeEndSlice(
             "args.params.Read(args.dst_end_ch_index)",
             "args.dst_tensor.Slices()") +
         ";\n";
  } else {
    c += "  int dst_end_slice = args.dst_tensor.Slices();\n";
  }
  return c;
}

std::string GetRingOOffset(const ConvRuntimeCheckDesc& runtime_check) {
  std::string c;
  c += "  int ring_o_offset = args.params.Read(" +
       std::to_string(runtime_check.ring_o_offset_index.value()) + ");\n";
  c += "  int ring_size = " + std::to_string(runtime_check.ring_size.value()) +
       ";\n";
  return c;
}

std::string GetRingIOffset(const ConvRuntimeCheckDesc& runtime_check) {
  std::string c;
  c += "  int ring_i_offset = args.params.Read(" +
       std::to_string(runtime_check.ring_i_offset_index.value()) + ");\n";
  c += "  int ring_size = " + std::to_string(runtime_check.ring_size.value()) +
       ";\n";
  return c;
}

std::string GetPackedGroupsParams(const ConvRuntimeCheckDesc& runtime_check,
                                  int dim_id, int block_size) {
  std::string c;
  c += "  int dst_w = ucl::GetGroupId<" + std::to_string(dim_id) + ">();\n";
  c += "  int w_group_size = args.params.Read(args.packed_params_offset + "
       "w_batch_id);\n";
  c += "  int w_group_offset = args.params.Read(args.packed_params_offset + "
       "w_batch_id + " +
       std::to_string(runtime_check.packed_groups->num_groups) + ");\n";
  c += "  int wg_first_w = dst_w * " + std::to_string(block_size) + ";\n";
  c += "  if (wg_first_w >= w_group_size) return;\n";
  c += "  dst_w = w_group_offset + dst_w * " + std::to_string(block_size) +
       ";\n";
  return c;
}

std::string GetWeightsBatchId(int runtime_batch_ids) {
  std::string c;
  if (runtime_batch_ids) {
    c += "  int dst_h = ucl::GetGroupId<2>();\n";
    c += "  int w_batch_id;\n";
    c += "  args.batch_ids.ReadPerChannel<int>(w_batch_id, 0, 0, "
         "dst_h);\n";
  } else {
    c += "  int w_batch_id = ucl::GetGroupId<2>();\n";
  }
  return c;
}

std::string GenerateDstWrite(const BHWC& block_size,
                             const ConvRuntimeCheckDesc& runtime_check,
                             bool has_bias, bool batched_weights,
                             int runtime_batch_ids) {
  std::string c;
  if (has_bias) {
    c += "  Type bias_value = args.biases.Read(dst_s);\n";
  }
  c += "  args.dst_tensor::type res_value;\n";
  const int block_spatial = block_size.b * block_size.w * block_size.h;
  for (int sp_id = 0; sp_id < block_spatial; ++sp_id) {
    const int3 bhw = GetBlockSpatialCoords(sp_id, block_size);
    std::string y_coord = std::to_string(bhw.y);
    if (batched_weights) {
      y_coord = "w_batch_id";
      if (runtime_batch_ids) {
        y_coord = "dst_h";
      }
    }
    std::string x_coord = std::to_string(bhw.z);
    if (runtime_check.packed_groups.has_value()) {
      x_coord = "dst_w + " + x_coord;
      y_coord = "0";
      c += "  if (" + x_coord + " < w_group_offset + w_group_size) {\n";
    }
    const std::string r_name = "r_sp" + std::to_string(sp_id);
    c += "  res_value = "
         "ucl::Convert<args.dst_tensor::type>(" +
         r_name + ");\n";
    if (has_bias) {
      c += "  res_value += bias_value;\n";
    }
    c += "  args.dst_tensor.Write(res_value, " + x_coord + ", " + y_coord +
         ", dst_s, " + std::to_string(bhw.x) + ");\n";
    if (runtime_check.packed_groups.has_value()) {
      c += "  }\n";
    }
  }
  return c;
}

void AddWeightsArguments(const ExternalWeights& weights, int vec_size,
                         GPUOperation* op) {
  if (weights.desc.type == DataType::FLOAT32 ||
      weights.desc.type == DataType::FLOAT16) {
    if (weights.desc.IsLinearLayout()) {
      BufferDescriptor desc;
      desc.element_type = weights.desc.type;
      desc.element_size = vec_size;
      op->AddSrcBuffer("weights", desc);
    } else {
      // Texture based weights, stored as 4 separate 2D textures.
      TensorDescriptor desc{weights.desc.type, TensorStorageType::TEXTURE_2D,
                            Layout::HW};
      for (int i = 0; i < 4; ++i) {
        const std::string name = "weights" + std::to_string(i);
        op->AddSrcTensor(name, desc);
      }
    }
  } else if (weights.desc.type == DataType::UINT8) {
    if (weights.desc.IsLinearLayout()) {
      BufferDescriptor desc;
      desc.element_type = DataType::UINT32;
      desc.element_size = vec_size / 4;
      op->AddSrcBuffer("weights", desc);
    } else {
      TensorDescriptor desc = TensorDescriptor(
          DataType::UINT32, TensorStorageType::TEXTURE_2D, Layout::HW);
      op->AddSrcTensor("weights", desc);
    }
  } else if (weights.desc.type == DataType::UINT4) {
    if (weights.desc.IsLinearLayout()) {
      BufferDescriptor desc;
      desc.element_type = DataType::UINT32;
      desc.element_size = vec_size / 8;
      op->AddSrcBuffer("weights", desc);
    } else {
      DataType texture_type = DataType::UINT16;
      if (vec_size == 32) {
        texture_type = DataType::UINT32;
      }
      TensorDescriptor desc = TensorDescriptor(
          texture_type, TensorStorageType::TEXTURE_2D, Layout::HW);
      op->AddSrcTensor("weights", desc);
    }
  } else if (weights.desc.type == DataType::UINT2) {
    if (weights.desc.IsLinearLayout()) {
      BufferDescriptor desc;
      desc.element_type = DataType::UINT32;
      desc.element_size = vec_size / 16;
      op->AddSrcBuffer("weights", desc);
    } else {
      DataType texture_type = DataType::UINT8;
      if (vec_size == 32) {
        texture_type = DataType::UINT16;
      } else if (vec_size == 64) {
        texture_type = DataType::UINT32;
      }
      TensorDescriptor desc = TensorDescriptor(
          texture_type, TensorStorageType::TEXTURE_2D, Layout::HW);
      op->AddSrcTensor("weights", desc);
    }
  }

  if (weights.scale) {
    op->AddSrcTensor("weights_scale", *weights.scale);
  } else if (weights.scalar_scale.has_value()) {
    op->args_.AddFloat("scale", *weights.scalar_scale);
  }
  if (weights.zero_point) {
    op->AddSrcTensor("weights_zero_point", *weights.zero_point);
  } else if (weights.scalar_zero_point.has_value()) {
    op->args_.AddFloat("zero_point", *weights.scalar_zero_point);
  }
}

void AddSparseWeightsArguments(const ExternalWeights& weights, int vec_size,
                               GPUOperation* op) {
  BufferDescriptor desc;
  desc.element_type = DataType::UINT32;
  desc.element_size = vec_size / 16;
  op->AddSrcBuffer("weights", desc);

  BufferDescriptor desc_indices;
  desc_indices.element_type = DataType::UINT32;
  desc_indices.element_size = 1;
  op->AddSrcBuffer("weights_indices", desc_indices);

  if (weights.scale) {
    op->AddSrcTensor("weights_scale", *weights.scale);
  } else if (weights.scalar_scale.has_value()) {
    op->args_.AddFloat("scale", *weights.scalar_scale);
  }
  if (weights.zero_point) {
    op->AddSrcTensor("weights_zero_point", *weights.zero_point);
  } else if (weights.scalar_zero_point.has_value()) {
    op->args_.AddFloat("zero_point", *weights.scalar_zero_point);
  }
}

bool IsQuantized(DataType weights_type) {
  return weights_type != DataType::FLOAT32 && weights_type != DataType::FLOAT16;
}

bool IsScalarQuantized(DataType weights_type, const OHWI& scale_zp_shape) {
  return IsQuantized(weights_type) && scale_zp_shape.DimensionsProduct() == 1;
}

bool IsLinearQuantized(DataType weights_type, const OHWI& scale_zp_shape) {
  return IsQuantized(weights_type) && scale_zp_shape.o != 1 &&
         scale_zp_shape.i == 1;
}

bool IsBlockQuantized(DataType weights_type, const OHWI& scale_zp_shape) {
  return IsQuantized(weights_type) && scale_zp_shape.o != 1 &&
         scale_zp_shape.i != 1;
}

int3 GetBlockSpatialCoords(int linear_spatial, const BHWC& shape) {
  int b_coord = linear_spatial % shape.b;
  linear_spatial /= shape.b;
  int x_coord = linear_spatial % shape.w;
  linear_spatial /= shape.w;
  int y_coord = linear_spatial % shape.h;
  return int3(b_coord, y_coord, x_coord);
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

bool UseFMA(const GpuInfo& gpu_info) {
  return gpu_info.IsApiWebGpu() ||
         (gpu_info.IsAMD() && gpu_info.IsApiOpenCl()) ||
         (gpu_info.IsMaleoon() && gpu_info.IsApiOpenCl());
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

}  // namespace fc
}  // namespace ml_drift
