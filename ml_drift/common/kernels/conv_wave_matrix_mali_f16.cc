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

#include "ml_drift/common/kernels/conv_wave_matrix_mali_f16.h"

#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/compiler_options.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/work_group_picking.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace {
std::string GetCode(int x_block_size, int s_block_size, bool slices_first) {
  auto GetGlobalID = [&](int id) {
    int3 wg_order = slices_first ? int3(1, 0, 2) : int3(0, 1, 2);
    int3 launch_remap;
    launch_remap[wg_order.x] = 0;
    launch_remap[wg_order.y] = 1;
    launch_remap[wg_order.z] = 2;
    std::string result;
    const std::string sid = std::to_string(id);
    if (wg_order[id] == id) {
      return "ucl::GetGlobalId<" + sid + ">()";
    } else {
      return "(ucl::GetGroupId<" + std::to_string(launch_remap[id]) +
             ">() * ucl::GetGroupSize<" + sid + ">() + ucl::GetLocalId<" + sid +
             ">())";
    }
  };
  std::string c = R"(
half4 arm_simd16_mmul8x4x8(half2 a, half2 b, half4 acc) {
  acc.xy = arm_matrix_multiply_af0(a, b, acc.xy);
  acc.zw = arm_matrix_multiply_af1(a, b, acc.zw);
  return acc;
}

MAIN_FUNCTION($0) {
  // WG must be [M*16, N, K]
)";
  c += "  int subgroup_id = " + GetGlobalID(0) + " / 16;\n";
  c += "  int subgroup_local_id = " + GetGlobalID(0) + " % 16;\n";
  c += "  int slices_sub_id = subgroup_local_id % 2;\n";
  c += "  int x_sub_id = subgroup_local_id / 2;\n";
  c += "  int SRC_X = subgroup_id * " + std::to_string(x_block_size * 8) +
       " + x_sub_id;\n";
  c += "  int warp_dst_s_base = " + GetGlobalID(1) + " * " +
       std::to_string(s_block_size * 4) + ";\n";
  c += "  int y = " + GetGlobalID(2) + ";\n";
  c += R"(

  if (SRC_X >= args.dst.Width() || y >= args.dst.Height() || warp_dst_s_base >= args.dst.Slices()) {
    return;
  }

  __global half8* w_base = (__global half8*)(args.weights.GetPtr() + warp_dst_s_base * 4 * args.src.Slices());

  )";
  for (int s = 0; s < s_block_size; s++) {
    for (int x = 0; x < x_block_size; x++) {
      std::string patch = R"(
  half4 r_w$0_s$1 = (half4)(0.0f);
  half4 r_w$0_s$2 = (half4)(0.0f);
)";
      c += absl::Substitute(patch, x, s * 2 + 0, s * 2 + 1);
    }
  }
  c += R"(

  for (int src_slice = 0; src_slice < args.src.Slices(); src_slice += 2) {
)";
  for (int x = 0; x < x_block_size; x++) {
    std::string patch = R"(
    half4 s$0 = args.src.Read(SRC_X + $1, y, src_slice + slices_sub_id);
)";
    c += absl::Substitute(patch, x, x * 8);
  }
  for (int s = 0; s < s_block_size; s++) {
    c += R"(
    {
    half8 W = w_base[subgroup_local_id];
    w_base += 16;
)";
    for (int x = 0; x < x_block_size; x++) {
      std::string patch = R"(
    r_w$0_s$1 = arm_simd16_mmul8x4x8(s$0.xy, W.s01, r_w$0_s$1);
    r_w$0_s$1 = arm_simd16_mmul8x4x8(s$0.zw, W.s23, r_w$0_s$1);
    r_w$0_s$2 = arm_simd16_mmul8x4x8(s$0.xy, W.s45, r_w$0_s$2);
    r_w$0_s$2 = arm_simd16_mmul8x4x8(s$0.zw, W.s67, r_w$0_s$2);
)";
      c += absl::Substitute(patch, x, s * 2 + 0, s * 2 + 1);
    }
    c += "    }";
  }
  c += R"(
  }

  int DST_S = warp_dst_s_base + subgroup_local_id % 4;
)";
  c += "  int DST_X = subgroup_id * " + std::to_string(x_block_size * 8) +
       " + x_sub_id / 2 * 2;\n";
  for (int s = 0; s < s_block_size; s++) {
    for (int x = 0; x < x_block_size; x++) {
      std::string patch = R"(
  {
  half4 rh_w0 = (half4)(r_w$0_s$1.x, r_w$0_s$1.y, r_w$0_s$2.x, r_w$0_s$2.y);
  half4 rh_w1 = (half4)(r_w$0_s$1.z, r_w$0_s$1.w, r_w$0_s$2.z, r_w$0_s$2.w);

  args.dst.Write(rh_w0, DST_X + $0 * 8 + 0, y, DST_S + $3);
  args.dst.Write(rh_w1, DST_X + $0 * 8 + 1, y, DST_S + $3);
  }
)";
      c += absl::Substitute(patch, x, s * 2 + 0, s * 2 + 1, s * 4);
    }
  }
  c += "}\n";
  return c;
}

static std::vector<float> reorder_weights(const std::vector<float>& w,
                                          int src_channels, int dst_channels,
                                          int s_block_size) {
  std::vector<float> out(w.size(), 0.0f);
  const int src_groups = src_channels / 8;
  const int dst_groups = dst_channels / 16 / s_block_size;
  for (int dst_group = 0; dst_group < dst_groups; dst_group++) {
    for (int src_group = 0; src_group < src_groups; src_group++) {
      for (int dst_block = 0; dst_block < s_block_size; dst_block++) {
        std::vector<float> w128(128, 0.0f);
        int counter = 0;
        int i_offset = src_group * 8;
        int o_offset = (dst_group * s_block_size + dst_block) * 16;
        // OSpatialII2I2gr0O8I2gr1O2
        for (int i2 = 0; i2 < 2; i2++) {
          for (int i1 = 0; i1 < 2; i1++) {
            for (int o1 = 0; o1 < 8; o1++) {
              for (int i0 = 0; i0 < 2; i0++) {
                for (int o0 = 0; o0 < 2; o0++) {
                  int i_block = (i2 * 2 + i0) * 2 + i1;
                  int o_block = o1 * 2 + o0;
                  int i_ch = i_offset + i_block;
                  int o_ch = o_offset + o_block;
                  w128[counter++] = w[o_ch * src_channels + i_ch];
                }
              }
            }
          }
        }
        const int block_offset =
            (dst_group * src_groups + src_group) * s_block_size + dst_block;
        for (int i = 0; i < 128; i++) {
          out[block_offset * 128 + i] = w128[i];
        }
      }
    }
  }
  return out;
}

}  // namespace

ConvWaveMatrixMaliF16::ConvWaveMatrixMaliF16(
    const TensorDescriptor& src_desc, const TensorDescriptor& dst_desc,
    const ml_drift::Tensor<OHWI, DataType::kFloat32>& weights) {
  x_block_size_ = 2;
  s_block_size_ = 2;
  slices_first_ = false;
  work_group_size_ = int3(16, 1, 1);
  code_ = GetCode(x_block_size_, s_block_size_, slices_first_);
  work_group_launch_order_ = slices_first_ ? int3(1, 0, 2) : int3(0, 1, 2);
  compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);

  AddSrcTensor("src", src_desc);
  AddDstTensor("dst", dst_desc);

  std::vector<float> reordered_weights = reorder_weights(
      weights.data, weights.shape.i, weights.shape.o, s_block_size_);
  std::vector<half> reordered_weights_half(reordered_weights.size());
  for (int i = 0; i < reordered_weights.size(); i++) {
    reordered_weights_half[i] = half(reordered_weights[i]);
  }
  BufferDescriptor buffer_desc;
  buffer_desc.element_type = DataType::kFloat16;
  buffer_desc.element_size = 4;
  buffer_desc.memory_type = MemoryType::kGlobal;
  buffer_desc.size = reordered_weights.size() * SizeOf(DataType::kFloat16);
  buffer_desc.data.resize(buffer_desc.size);
  std::memcpy(buffer_desc.data.data(), reordered_weights_half.data(),
              buffer_desc.size);
  args_.AddObject("weights",
                  std::make_unique<BufferDescriptor>(std::move(buffer_desc)));
}

std::vector<int3> ConvWaveMatrixMaliF16::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info) const {
  switch (tuning_type) {
    case TuningType::kExhaustive:
      return GetWorkGroupsXMultipleOf(/*wave_size=*/16, gpu_info, kernel_info,
                                      grid_size_);
    case TuningType::kFast:
    default:
      return {{32, 2, 1}};
  }
}

int3 ConvWaveMatrixMaliF16::GetGridSize() const {
  int x_tiles = DivideRoundUp(dst_[0]->Width(), 8 * x_block_size_);
  const int grid_x = x_tiles * /*wave_size=*/16;
  const int grid_y = DivideRoundUp(dst_[0]->Slices(), 4 * s_block_size_);
  const int grid_z = dst_[0]->Height();
  return int3(grid_x, grid_y, grid_z);
}

}  // namespace ml_drift
