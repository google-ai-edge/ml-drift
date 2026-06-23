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

#include "ml_drift/common/kernels/conv_apple_mpp.h"

#include <string>

#include "absl/log/absl_check.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace {
void AddRuntimeParams(GPUOperation& op,
                      const ConvRuntimeCheckDesc& runtime_check) {
  bool has_runtime_check = false;
  if (runtime_check.src_end_ch_index.has_value()) {
    op.args_.AddInt("src_end_ch_index", *runtime_check.src_end_ch_index);
    has_runtime_check = true;
  }
  if (runtime_check.dst_end_ch_index.has_value()) {
    op.args_.AddInt("dst_end_ch_index", *runtime_check.dst_end_ch_index);
    has_runtime_check = true;
  }
  if (runtime_check.packed_groups.has_value()) {
    op.args_.AddInt("packed_params_offset",
                    runtime_check.packed_groups->params_offset);
    has_runtime_check = true;
  }
  if (has_runtime_check) {
    BufferDescriptor buffer_desc;
    buffer_desc.element_type = DataType::INT32;
    buffer_desc.element_size = 1;
    buffer_desc.memory_type = MemoryType::CONSTANT;
    op.AddSrcBuffer("params", buffer_desc);
  }
}

inline int GetRangeShift(DataType type) {
  return 1u << (SizeInBitsOf(type) - 1);
}

std::string ReadFloatWeights(
    const ConvAppleMPP::ExternalWeightsParams& params) {
  const bool weights_conversion =
      params.weights_desc.layout == WeightsLayout::kOSpatialIOGroupI4O4;
  const bool quantized_weights =
      weights_conversion && SizeInBitsOf(params.weights_desc.type) <= 8;
  std::string c;
  if (quantized_weights && params.scale_zp_shape.i != 1) {
    // grouped quantization
    c += "    int src_group_id = (k + sub_i) / " +
         std::to_string(params.src_group_slices) + +";\n";
    c += "    if (last_src_group_id != src_group_id) {\n";
    c += "      last_src_group_id = src_group_id;\n";
    // do not use different_weights_for_height here because with batched weights
    // we can have non batched scale/zp.
    const std::string scale_zp_batch_id =
        params.scale_zp_shape.h != 1 ? "w_batch_id" : "0";
    std::string coords = "w_o_slice, " + scale_zp_batch_id + ", src_group_id";
    c += "      w_scale = args.weights_scale.Read(" + coords + ");\n";
    if (params.has_zero_point) {
      c += "      half4 w_zp = args.weights_zero_point.Read(" + coords + ");\n";
    } else {
      c += "      half4 w_zp = ucl::Init<half4>(0.0f);\n";
    }
    c += "      w_bias = -w_scale * ucl::Init<half4>(" +
         std::to_string(GetRangeShift(params.weights_desc.type)) +
         ") + w_zp;\n";
    c += "    }\n";
  }
  if (params.weights_desc.type == DataType::UINT8) {
    c += "    uint4 u8_i4o4 = args.weights.Read(w_wg_offset);\n";
    c += "    ucl::U32x4ToU8x16AsVec4x4<half>(u8_i4o4, w0, w1, w2, w3);\n";
  } else if (params.weights_desc.type == DataType::UINT4) {
    c += "    uint2 u4_i4o4 = args.weights.Read(w_wg_offset);\n";
    c += "    ucl::U32x2ToU4x16AsVec4x4<half>(u4_i4o4, w0, w1, w2, w3);\n";
  } else if (params.weights_desc.type == DataType::UINT2) {
    c += "    uint u2_i4o4 = args.weights.Read(w_wg_offset);\n";
    c += "    ucl::U32x1ToU2x16AsVec4x4<half>(u2_i4o4, w0, w1, w2, w3);\n";
  } else {
    c += "    args.weights.ReadVec16AsVec4x4(w0, w1, w2, w3, w_wg_offset);\n";
  }
  if (quantized_weights) {
    c += R"(
    w0 = w0 * w_scale + w_bias;
    w1 = w1 * w_scale + w_bias;
    w2 = w2 * w_scale + w_bias;
    w3 = w3 * w_scale + w_bias;
)";
  }
  return c;
}
}  // namespace

std::string ConvAppleMPP::GetKernelCode(bool has_batch, bool has_bias) const {
  const bool weights_conversion =
      external_weights_params_.weights_desc.layout ==
      WeightsLayout::kOSpatialIOGroupI4O4;
  const bool quantized_weights =
      weights_conversion &&
      SizeInBitsOf(external_weights_params_.weights_desc.type) <= 8;
  const bool manual_src_reading =
      softmax_input_activation_ ||
      src_desc_.GetStorageType() != TensorStorageType::BUFFER ||
      !src_desc_.IsCC4Layout();
  const bool manual_k_tiling = runtime_check_.src_end_ch_index.has_value() ||
                               manual_src_reading || weights_conversion;
  const int k_tile = manual_k_tiling ? 32 : AlignByN(weights_shape_.i, 4);
  std::string c;
  c += "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n";
  if (weights_conversion && weights_data_type_ != DataType::FLOAT16) {
    c += R"(
uint32_t expand_int4_to_int8(uint32_t val) {
  uint32_t x = val & 0xFFFF;
  x = (x | (x << 8)) & 0x00FF00FF;
  x = (x | (x << 4)) & 0x0F0F0F0F;
  uint32_t signs = x & 0x08080808;
  uint32_t sign_extensions = signs * 0x1E;
  return x | sign_extensions;
}
)";
  }
  c += R"(
MAIN_FUNCTION($0) {
  int slice_tile_id = ucl::GetGroupId<0>();
  int spatial_tile_id = ucl::GetGroupId<1>();
  int sub_spatial_id = ucl::GetLocalId<0>() % M_TILE;
  int sub_slice_id = ucl::GetLocalId<0>() / M_TILE;
  int slices_per_wg = WG_SIZE / M_TILE;
  int spatial_id = spatial_tile_id * M_TILE + sub_spatial_id;
)";
  if (has_batch) {
    c += "  int dst_b = spatial_id % args.dst.Batch();\n";
    c += "  spatial_id = spatial_id / args.dst.Batch();\n";
  }
  if (batched_weights_) {
    c += "  int dst_w = spatial_id;\n";
    c += "  int dst_h = ucl::GetGroupId<2>();\n";
  } else {
    c += "  int dst_w = spatial_id % args.dst.Width();\n";
    c += "  int dst_h = spatial_id / args.dst.Width();\n";
  }
  if (runtime_check_.packed_groups.has_value()) {
    c += "  int w_batch_id = dst_h;\n";
    c += "  dst_h = 0;\n";
    c += "  int w_group_size = args.params.Read(args.packed_params_offset + "
         "w_batch_id);\n";
    c += "  int w_group_offset = args.params.Read(args.packed_params_offset + "
         "w_batch_id + " +
         std::to_string(runtime_check_.packed_groups->num_groups) + ");\n";
    c += "  int wg_first_w = spatial_tile_id * M_TILE;\n";
    c += "  if (wg_first_w >= w_group_size) return;\n";
    c += "  dst_w = w_group_offset + dst_w;\n";
  } else if (batched_weights_) {
    c += "  int w_batch_id = dst_h;\n";
  }
  if (runtime_check_.dst_end_ch_index.has_value()) {
    c += "  int dst_end_slice_runtime = " +
         runtime_check_.GetRuntimeEndSlice(
             "args.params.Read(args.dst_end_ch_index)", "args.dst.Slices()") +
         ";\n";
    c += "  if (slice_tile_id * N_TILE_SLICES >= dst_end_slice_runtime) "
         "return;\n";
  }
  if (weights_conversion) {
    ABSL_CHECK(n_tile_ == 64);
    ABSL_CHECK(k_tile == 32);
    ABSL_CHECK(simdgroups_ == 4);
    const std::string w_loc_type =
        weights_data_type_ == DataType::FLOAT16 ? "half" : "int8_t";
    const std::string w_loc_x4_type =
        weights_data_type_ == DataType::FLOAT16 ? "half4" : "uint";
    c += "  threadgroup " + w_loc_type + " w_loc[K_TILE * N_TILE];\n";
    c += "  threadgroup " + w_loc_x4_type + "* w_loc_x4 = (threadgroup " +
         w_loc_x4_type + "*)(w_loc);\n";
    c += R"(
  auto b_tile = tensor(w_loc, dextents<int, 2>(N_TILE, K_TILE));

  int loc_id = ucl::GetLocalId<0>();
  int sub_i = loc_id / 16;
  int sub_o = loc_id % 16;

  int w_o_slice = min(slice_tile_id * N_TILE_SLICES + sub_o, args.dst.Slices() - 1);
  int w_stride = args.dst.Slices() * 8;
)";
    const std::string batch_part =
        batched_weights_ ? "w_batch_id * args.src.Slices()" : "0";
    c += "  int w_wg_offset = (" + batch_part +
         " + sub_i) * args.dst.Slices() + w_o_slice;\n";
    if (quantized_weights && weights_data_type_ == DataType::FLOAT16) {
      c += "  half4 w_scale, w_bias;\n";
      if (external_weights_params_.scale_zp_shape.i != 1) {
        // grouped quantization
        c += "  int last_src_group_id = -1;\n";
      } else {
        // linear quantization
        // do not use different_weights_for_height here because with batched
        // weights we can have non batched scale/zp.
        const std::string coords =
            external_weights_params_.scale_zp_shape.h != 1
                ? "w_o_slice, w_batch_id, 0"
                : "w_o_slice";
        c += "  w_scale = args.weights_scale.Read(" + coords + ");\n";
        if (external_weights_params_.has_zero_point) {
          c += "  half4 w_zp = args.weights_zero_point.Read(" + coords + ");\n";
        } else {
          c += "  half4 w_zp = ucl::Init<half4>(0.0h);\n";
        }
        c += "  w_bias = -w_scale * ucl::Init<half4>(" +
             std::to_string(
                 GetRangeShift(external_weights_params_.weights_desc.type)) +
             ") + w_zp;\n";
      }
    }
  } else {
    c += R"(
  device TYPE* w_ptr = reinterpret_cast<device TYPE*>(args.weights.GetPtr());
  int b_rows = args.src.Slices() * 4;
  int b_cols = args.dst.Slices() * 4;
)";
    if (batched_weights_) {
      c += "  w_ptr += w_batch_id * b_rows * b_cols;\n";
    }
    c += "  auto b_tensor = tensor(w_ptr, dextents<int, 2>(b_cols, b_rows));\n";
    c += "  auto b_tile = b_tensor.slice(slice_tile_id * N_TILE, 0);\n";
  }
  if (manual_src_reading) {
    c += R"(
  threadgroup half a_loc[K_TILE * M_TILE];
  threadgroup half4* a_loc_x4 = (threadgroup half4*)(a_loc);
  auto a_tile = tensor(a_loc, dextents<int, 2>(K_TILE, M_TILE));
  int src_w = min(dst_w, args.src.Width() - 1);
  int src_h = min(dst_h, args.src.Height() - 1);
)";
    if (softmax_input_activation_) {
      c += "  half2 exp_val = args.src_exp.Read(src_w, src_h, 0).xy;\n";
    }
  } else {
    c += R"(
  device TYPE* s_ptr = reinterpret_cast<device TYPE*>(args.src.GetHandle());
  int a_rows = SPATIAL_SIZE;
  int a_cols = args.src.Slices() * 4;
)";
    if (runtime_check_.packed_groups.has_value()) {
      c += "  s_ptr += w_group_offset * a_cols;\n";
    } else {
      if (batched_weights_) {
        c += "  s_ptr += w_batch_id * a_rows * a_cols;\n";
      }
    }
    c += "  auto a_tensor = tensor(s_ptr, dextents<int, 2>(a_cols, a_rows));\n";
    c += "  auto a_tile = a_tensor.slice(0, spatial_tile_id * M_TILE);\n";
  }
  c += R"(
  constexpr auto matmul_desc = mpp::tensor_ops::matmul2d_descriptor(M_TILE, N_TILE, K_TILE, false, false, false, MAT_MUL_MODE);
  mpp::tensor_ops::matmul2d<matmul_desc, execution_simdgroups<SIMDGROUPS>> matmul_op;

  auto c_tile = matmul_op.get_destination_cooperative_tensor<decltype(a_tile), decltype(b_tile), args.dst::scalar_type>();
  #pragma unroll_full
  for (uint16_t i = 0; i < c_tile.get_capacity(); ++i) {
    if(c_tile.is_valid_element(i)) c_tile[i] = 0;
  }
)";
  std::string src_end_slice = "args.src.Slices()";
  if (runtime_check_.src_end_ch_index.has_value()) {
    c += "  int src_slices_dynamic = " +
         runtime_check_.GetRuntimeEndSlice(
             "args.params.Read(args.src_end_ch_index)", "args.src.Slices()") +
         ";\n";
    src_end_slice = "src_slices_dynamic";
  }
  if (manual_k_tiling) {
    c += "  for (int k = 0; k < " + src_end_slice + "; k += K_TILE_SLICES) {\n";
    if (weights_conversion) {
      if (weights_data_type_ == DataType::FLOAT16) {
        c += "    half4 w0, w1, w2, w3;\n";
        c += ReadFloatWeights(external_weights_params_);
      } else {
        c += R"(
    uint2 u4_i4o4 = args.weights.Read(w_wg_offset);
    u4_i4o4.x ^= 0x88888888u;
    u4_i4o4.y ^= 0x88888888u;
    uint w0 = expand_int4_to_int8(u4_i4o4.x);
    uint w1 = expand_int4_to_int8(u4_i4o4.x >> 16);
    uint w2 = expand_int4_to_int8(u4_i4o4.y);
    uint w3 = expand_int4_to_int8(u4_i4o4.y >> 16);
)";
      }
      c += "    w_wg_offset += w_stride;\n";
      c += R"(
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // store it in i32o64 or half4 in i32o16
    w_loc_x4[(sub_i * 4 + 0) * 16 + sub_o] = w0;
    w_loc_x4[(sub_i * 4 + 1) * 16 + sub_o] = w1;
    w_loc_x4[(sub_i * 4 + 2) * 16 + sub_o] = w2;
    w_loc_x4[(sub_i * 4 + 3) * 16 + sub_o] = w3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
)";
    } else {
      c += "    b_tile = b_tensor.slice(slice_tile_id * N_TILE, k * 4);\n";
    }
    if (manual_src_reading) {
      c += R"(
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll_full
    for (uint16_t i = 0; i < K_TILE_SLICES / slices_per_wg; ++i) {
      int tile_k_id = i * slices_per_wg + sub_slice_id;
      int slice_id = min(k + tile_k_id, args.src.Slices() - 1);
      half4 src = args.src.Read(src_w, src_h, slice_id);
)";
      if (softmax_input_activation_) {
        c += "      src = exp(src - exp_val.y) * exp_val.x;\n";
      }
      c += R"(
      a_loc_x4[sub_spatial_id * K_TILE_SLICES + tile_k_id] = src;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
)";
    } else {
      c += "    a_tile = a_tensor.slice(k * 4, spatial_tile_id * M_TILE);\n";
    }
    c += "    matmul_op.run(a_tile, b_tile, c_tile);\n";
    c += "  }\n";
  } else {
    c += "  matmul_op.run(a_tile, b_tile, c_tile);\n";
  }
  c += R"(
  threadgroup args.dst::scalar_type tmp[N_TILE * M_TILE];
  auto c_local = tensor(tmp, dextents<int, 2>(N_TILE, M_TILE));
  c_tile.store(c_local);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup args.dst::type* tmp_x4 = (threadgroup args.dst::type*)(tmp);
)";
  const std::string oob_check = batched_weights_ ? "dst_w >= args.dst.Width()"
                                                 : "dst_h >= args.dst.Height()";
  c += "  if (" + oob_check + ") return;\n";
  if (runtime_check_.packed_groups.has_value()) {
    c += "  if (dst_w >= w_group_offset + w_group_size) return;\n";
  }
  c += R"(
  for (int i = 0; i < N_TILE_SLICES / slices_per_wg; ++i) {
    int tile_n_id = i * slices_per_wg + sub_slice_id;
    int slice_id = slice_tile_id * N_TILE_SLICES + tile_n_id;
)";
  c += "    if (slice_id < args.dst.Slices()) {\n";
  c += "      args.dst::type res_value = tmp_x4[sub_spatial_id * N_TILE_SLICES "
       "+ "
       "tile_n_id];\n";
  if (has_bias) {
    c += "      res_value += args.biases.Read(slice_id);\n";
  }
  std::string coords = "dst_w, dst_h, slice_id";
  if (has_batch) {
    coords += ", dst_b";
  }
  c += "      args.dst.Write(res_value, " + coords + ");\n";
  c += "    }\n";
  c += "  }\n";
  c += "}\n";
  const std::string type =
      weights_data_type_ == DataType::FLOAT16 ? "half" : "int8_t";
  std::string spatial_size = "args.dst.Width()";
  if (!batched_weights_) {
    spatial_size += " * args.dst.Height()";
  }
  if (has_batch) {
    spatial_size += " * args.dst.Batch()";
  }
  if (runtime_check_.packed_groups.has_value()) {
    spatial_size = "w_group_size";
  }
  const std::string mat_mul_mode =
      manual_k_tiling
          ? "mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate"
          : "mpp::tensor_ops::matmul2d_descriptor::mode::multiply";
  absl::StrReplaceAll(
      {
          {"M_TILE", std::to_string(m_tile_)},
          {"N_TILE", std::to_string(n_tile_)},
          {"N_TILE_SLICES", std::to_string(n_tile_ / 4)},
          {"K_TILE", std::to_string(k_tile)},
          {"K_TILE_SLICES", std::to_string(k_tile / 4)},
          {"SIMDGROUPS", std::to_string(simdgroups_)},
          {"WG_SIZE", std::to_string(32 * simdgroups_)},
          {"TYPE", type},
          {"SPATIAL_SIZE", spatial_size},
          {"MAT_MUL_MODE", mat_mul_mode},
      },
      &c);
  return c;
}

// Tile size M(64)xN(128) shows better and more stable performance on big convs.
// This include:
//   square size (src_ch == dst_ch = spatial) (2048x2048 -> 2048x2048)
//   longer src_ch (src_ch > dst_ch) (1024x(1536 * 8) -> 1024x1536)
//   longer dst_ch (dst_ch > src_ch) (1024x1536 -> 1024x(1536 * 8))
ConvAppleMPP::ConvAppleMPP(const TensorDescriptor& src,
                           const OHWI& weights_shape,
                           DataType weights_data_type,
                           bool different_weights_for_height,
                           bool softmax_input_activation,
                           const ConvRuntimeCheckDesc& runtime_check)
    : src_desc_(src),
      m_tile_(64),
      n_tile_(128),
      simdgroups_(4),
      weights_shape_(weights_shape),
      weights_data_type_(weights_data_type),
      batched_weights_(different_weights_for_height),
      softmax_input_activation_(softmax_input_activation),
      runtime_check_(runtime_check) {
  work_group_size_ = int3(32 * simdgroups_, 1, 1);
  if (weights_shape.o % 128 != 0) {
    n_tile_ = 64;
  }
  external_weights_params_.weights_desc.layout = WeightsLayout::kUnknown;
}

int3 ConvAppleMPP::GetGridSize() const {
  const int groups_x = DivideRoundUp(dst_[0]->Channels(), n_tile_);
  int spatial_size = dst_[0]->Width() * dst_[0]->Batch();
  int groups_z = 1;
  if (batched_weights_) {
    groups_z = dst_[0]->Height();
  } else {
    spatial_size *= dst_[0]->Height();
  }
  if (runtime_check_.packed_groups.has_value()) {
    spatial_size = runtime_check_.packed_groups->max_group_size;
    groups_z = runtime_check_.packed_groups->num_groups;
  }
  const int groups_y = DivideRoundUp(spatial_size, m_tile_);
  return int3(groups_x * work_group_size_.x, groups_y, groups_z);
}

bool SupportsConvAppleMPP(const GpuInfo& gpu_info) {
  return gpu_info.IsApiMetal() && gpu_info.IsApple() &&
         gpu_info.apple_info.gpu_family >= AppleInfo::Family::kApple10 &&
         gpu_info.metal_info.IsMslVersionEqualOrHigher(4, 0);
}

bool SupportsConvAppleMPP(const GpuInfo& gpu_info,
                          const ExternalWeights& weights) {
  if (!SupportsConvAppleMPP(gpu_info)) {
    return false;
  }
  const bool supported_type = weights.desc.type == DataType::FLOAT32 ||
                              weights.desc.type == DataType::FLOAT16 ||
                              weights.desc.type == DataType::UINT8 ||
                              weights.desc.type == DataType::UINT4 ||
                              weights.desc.type == DataType::UINT2;
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  if (weights.desc.layout != WeightsLayout::kOSpatialIOGroupI4O4 ||
      weights.desc.output_group_size != dst_slices || !supported_type) {
    return false;
  }
  return true;
}

ConvAppleMPP CreateConvAppleMPP(const TensorDescriptor& src,
                                const TensorDescriptor& dst,
                                const Tensor<OHWI, DataType::FLOAT32>& weights,
                                const Tensor<Linear, DataType::FLOAT32>& bias) {
  ConvAppleMPP conv(src, weights.shape, DataType::FLOAT16);
  conv.code_ = conv.GetKernelCode(dst.HasAxis(Axis::BATCH), !bias.data.empty());
  conv.AddSrcTensor("src", src);
  conv.AddDstTensor("dst", dst);

  conv.UploadWeights(weights);
  if (!bias.data.empty()) {
    conv.UploadBias(bias);
  }
  return conv;
}

ConvAppleMPP CreateConvAppleMPPExternalWeights(
    const TensorDescriptor& src, const TensorDescriptor& dst,
    const OHWI& weights_shape, const TensorDescriptor* bias,
    const TensorDescriptor* src_exp, bool different_weights_for_height,
    const ConvRuntimeCheckDesc& runtime_check) {
  ConvAppleMPP conv(src, weights_shape, DataType::FLOAT16,
                    different_weights_for_height, src_exp != nullptr,
                    runtime_check);
  conv.code_ = conv.GetKernelCode(dst.HasAxis(Axis::BATCH), bias != nullptr);
  conv.AddSrcTensor("src", src);
  conv.AddDstTensor("dst", dst);

  BufferDescriptor weights_desc;
  weights_desc.element_type = DataType::FLOAT16;
  weights_desc.element_size = 1;
  conv.AddSrcBuffer("weights", weights_desc);

  if (bias) {
    conv.AddSrcTensor("biases", *bias);
  }

  if (src_exp) {
    conv.AddSrcTensor("src_exp", *src_exp);
  }

  AddRuntimeParams(conv, runtime_check);
  return conv;
}

ConvAppleMPP CreateConvAppleMPPExternalWeights(
    const TensorDescriptor& src, const TensorDescriptor& dst,
    const ExternalWeights& weights, const TensorDescriptor* bias,
    const TensorDescriptor* src_exp, bool different_weights_for_height,
    const ConvRuntimeCheckDesc& runtime_check) {
  ConvAppleMPP conv(src, weights.shape, DataType::FLOAT16,
                    different_weights_for_height, src_exp != nullptr,
                    runtime_check);
  ConvAppleMPP::ExternalWeightsParams params;
  params.weights_desc = weights.desc;
  params.scale_zp_shape = weights.scale_zp_shape;
  params.has_zero_point = weights.zero_point != nullptr;
  params.src_group_slices =
      DivideRoundUp(weights.shape.i, 4) / weights.scale_zp_shape.i;
  conv.SetExternalWeightsParams(params);
  conv.SetNTile(64);
  conv.code_ = conv.GetKernelCode(dst.HasAxis(Axis::BATCH), bias != nullptr);
  conv.AddSrcTensor("src", src);
  conv.AddDstTensor("dst", dst);

  BufferDescriptor buffer_desc;
  if (SizeInBitsOf(weights.desc.type) >= 16) {
    // float weights
    buffer_desc.element_type = weights.desc.type;
    buffer_desc.element_size = 16;
  } else {
    // quantized weights
    buffer_desc.element_type = DataType::UINT32;
    buffer_desc.element_size = SizeInBitsOf(weights.desc.type) / 2;
  }
  buffer_desc.memory_type = MemoryType::GLOBAL;
  conv.AddSrcBuffer("weights", buffer_desc);

  if (weights.scale) {
    conv.AddSrcTensor("weights_scale", *weights.scale);
  }
  if (weights.zero_point) {
    conv.AddSrcTensor("weights_zero_point", *weights.zero_point);
  }

  if (bias) {
    conv.AddSrcTensor("biases", *bias);
  }

  if (src_exp) {
    conv.AddSrcTensor("src_exp", *src_exp);
  }

  AddRuntimeParams(conv, runtime_check);
  return conv;
}

ConvAppleMPP CreateConvAppleMPPInt8(
    const TensorDescriptor& src, const TensorDescriptor& dst,
    const Tensor<OHWI, DataType::INT8>& weights) {
  ConvAppleMPP conv(src, weights.shape, DataType::INT8);
  conv.code_ = conv.GetKernelCode(dst.HasAxis(Axis::BATCH));
  conv.AddSrcTensor("src", src);
  conv.AddDstTensor("dst", dst);
  conv.UploadWeights(weights);
  return conv;
}

ConvAppleMPP CreateConvAppleMPPInt8(const TensorDescriptor& src,
                                    const TensorDescriptor& dst,
                                    const OHWI& weights_shape) {
  ConvAppleMPP conv(src, weights_shape, DataType::INT8, weights_shape.h != 1);
  conv.code_ = conv.GetKernelCode(dst.HasAxis(Axis::BATCH));
  conv.AddSrcTensor("src", src);
  conv.AddDstTensor("dst", dst);

  BufferDescriptor weights_desc;
  weights_desc.element_type = DataType::INT8;
  weights_desc.element_size = 1;
  conv.AddSrcBuffer("weights", weights_desc);
  return conv;
}

// Creates an Apple MPP convolution operation with INT8 external weights.
ConvAppleMPP CreateConvAppleMPPInt8(const TensorDescriptor& src,
                                    const TensorDescriptor& dst,
                                    const ExternalWeights& weights) {
  ConvAppleMPP conv(src, weights.shape, DataType::INT8, weights.shape.h != 1);
  ConvAppleMPP::ExternalWeightsParams params;
  params.weights_desc = weights.desc;
  conv.SetExternalWeightsParams(params);
  conv.SetNTile(64);
  conv.code_ = conv.GetKernelCode(dst.HasAxis(Axis::BATCH));
  conv.AddSrcTensor("src", src);
  conv.AddDstTensor("dst", dst);

  BufferDescriptor weights_desc;
  weights_desc.element_type = DataType::UINT32;
  weights_desc.element_size = 2;
  weights_desc.memory_type = MemoryType::GLOBAL;
  conv.AddSrcBuffer("weights", weights_desc);
  return conv;
}

}  // namespace ml_drift
