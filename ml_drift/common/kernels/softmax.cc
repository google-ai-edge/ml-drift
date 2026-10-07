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

#include "ml_drift/common/kernels/softmax.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/work_group_picking.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace {
std::string GetSoftmaxReducedExp(const OperationDef& op_def,
                                 bool use_wg_reduction, const int3& wg_size) {
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  std::string coords = "X, Y";
  if (op_def.dst_tensors[0].HasAxis(Axis::kBatch)) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
  } else {
    c += "  int X = ucl::GetGlobalId<0>();\n";
  }
  c += "  int Y = ucl::GetGlobalId<1>();\n";
  if (use_wg_reduction) {
    c += "  __local float2 loc_mem[WG_SIZE_Z][" +
         std::to_string(wg_size.x * wg_size.y) + "];\n";
  }

  if (op_def.dst_tensors[0].HasAxis(Axis::kDepth)) {
    coords += ", Z";
    if (!use_wg_reduction) {
      c += "  if (X >= args.dst_tensor.Width() || Y >= "
           "args.dst_tensor.Height() * args.dst_tensor.Depth()) "
           "return; \n";
    }
    c += "  int Z = Y % args.dst_tensor.Depth();\n";
    c += "  Y = Y / args.dst_tensor.Depth();\n";
  } else {
    if (!use_wg_reduction) {
      c += "  if (X >= args.dst_tensor.Width() || Y >= "
           "args.dst_tensor.Height()) "
           "return; \n";
    }
  }

  std::string coords_s0 = coords + ", 0";
  std::string coords_s = coords + ", d";
  if (op_def.dst_tensors[0].HasAxis(Axis::kBatch)) {
    coords_s0 += ", B";
    coords_s += ", B";
  }

  c += "  float sum = 0.0f;\n";
  c += "  int end_channel = END_CHANNEL;\n";
  c += "  int end_slice = (end_channel + 3) / 4;\n";
  c += "  bool need_per_channels_check = end_channel % 4 != 0;\n";
  c += "  float maximum;\n";
  c += "  args.src_tensor.ReadPerChannel<float>(maximum, " + coords_s0 + ");\n";
  if (use_wg_reduction) {
    c += "  for (int d = ucl::GetLocalId<2>(); d < end_slice; d "
         "+= WG_SIZE_Z) "
         "{\n";
  } else {
    c += "  for (int d = 0; d < end_slice; d += 1) {\n";
  }
  c += "    float4 mask_dot = ucl::Init<float4>(1.0f);\n";
  c += "    float4 src = args.src_tensor.Read<float>(" + coords_s + ");\n";
  c += "    if (need_per_channels_check && (d == end_slice - 1)) {\n";
  c += "      if (d * 4 + 0 >= end_channel) {\n";
  c += "        mask_dot.x = 0.0f;\n";
  c += "        src.x = maximum;\n";
  c += "      }\n";
  c += "      if (d * 4 + 1 >= end_channel) {\n";
  c += "        mask_dot.y = 0.0f;\n";
  c += "        src.y = maximum;\n";
  c += "      }\n";
  c += "      if (d * 4 + 2 >= end_channel) {\n";
  c += "        mask_dot.z = 0.0f;\n";
  c += "        src.z = maximum;\n";
  c += "      }\n";
  c += "      if (d * 4 + 3 >= end_channel) {\n";
  c += "        mask_dot.w = 0.0f;\n";
  c += "        src.w = maximum;\n";
  c += "      }\n";
  c += "    }\n";
  c += "    float new_max = max(src.x, src.y);\n";
  c += "    new_max = max(new_max, src.z);\n";
  c += "    new_max = max(new_max, src.w);\n";
  c += "    new_max = max(new_max, maximum);\n";
  c += "    float scale = EXP_FUNC(maximum - new_max);\n";
  c += "    maximum = new_max;\n";
  c += "    sum *= scale;\n";
  c += "    float4 exp_res = EXP_FUNC(src - ucl::Init<float4>(maximum));\n";
  c += "    sum += dot(mask_dot, exp_res);\n";
  c += "  }\n";
  if (use_wg_reduction) {
    c += "  float2 value;\n";
    c += "  value.x = sum;\n";
    c += "  value.y = maximum;\n";
    c += "  int local_xy = ucl::GetLocalId<1>() * WG_SIZE_X + "
         "ucl::GetLocalId<0>();\n";
    c += "  loc_mem[ucl::GetLocalId<2>()][local_xy] = value;\n";
    c += "  ucl::SyncThreads<WorkGroup, Local>();\n";
    c += "  if (ucl::GetLocalId<2>() == 0) {\n";
    for (int i = 1; i < wg_size.z; ++i) {
      c += "    {\n";
      c += "    float2 new_value = loc_mem[" + std::to_string(i) +
           "][local_xy];\n";
      c += "    float new_max = max(value.y, new_value.y);\n";
      c += "    new_value.x *= EXP_FUNC(new_value.y - new_max);\n";
      c += "    value.x *= EXP_FUNC(value.y - new_max);\n";
      c += "    value.x = value.x + new_value.x;\n";
      c += "    value.y = new_max;\n";
      c += "    }\n";
    }
    c += "    sum = value.x;\n";
    c += "    maximum = value.y;\n";
    c += "  }\n";
  }
  c += "  float inv_sum = 1.0f / sum;\n";
  return c;
}

// spatial_size must be pow of 2
std::pair<int, int2> GetMinGroupsForSpatialSize(const BHWC& shape,
                                                int spatial_size) {
  int min_groups = DivideRoundUp(shape.w * shape.b, spatial_size) * shape.h;
  int2 group_size{spatial_size, 1};
  for (int i = 2; i <= spatial_size; i *= 2) {
    int wg_y = i;
    int wg_x = spatial_size / wg_y;
    int groups =
        DivideRoundUp(shape.w * shape.b, wg_x) * DivideRoundUp(shape.h, wg_y);
    if (groups < min_groups) {
      min_groups = groups;
      group_size = {wg_x, wg_y};
    }
  }
  return {min_groups, group_size};
}

int3 GetWorkGroupSize(const GpuInfo& gpu_info, const BHWC& shape) {
  std::vector<int> spatial_sizes = {32, 16, 8, 4};
  float wg_per_cu_threshold = 4.0f;
  if (gpu_info.IsMali()) {
    spatial_sizes = {64, 32, 16, 8, 4};
    wg_per_cu_threshold = 6.0f;
  }
  for (int spatial_size : spatial_sizes) {
    auto min_group = GetMinGroupsForSpatialSize(shape, spatial_size);
    float wg_per_cu =
        static_cast<float>(min_group.first) / gpu_info.GetComputeUnitsCount();
    if (wg_per_cu >= wg_per_cu_threshold) {
      int total_wg_size = 128;
      int3 wg_size;
      wg_size.x = min_group.second.x;
      wg_size.y = min_group.second.y;
      wg_size.z = total_wg_size / spatial_size;
      return wg_size;
    }
  }
  return {4, 1, 32};
}

Softmax CreateSoftmaxImpl(const OperationDef& definition,
                          const GpuInfo& gpu_info, const BHWC& shape,
                          bool reduce_only,
                          const SoftmaxRuntimeCheckDesc& runtime_check) {
  auto softmax =
      Softmax(definition, gpu_info, shape, reduce_only, runtime_check);
  if (runtime_check.end_ch_index.has_value()) {
    softmax.args_.AddInt("end_ch_index", *runtime_check.end_ch_index);
    BufferDescriptor buffer_desc;
    buffer_desc.element_type = DataType::kInt32;
    buffer_desc.element_size = 1;
    softmax.AddSrcBuffer("params", buffer_desc);
  }
  return softmax;
}

}  // namespace

Softmax::Softmax(const OperationDef& definition, const GpuInfo& gpu_info,
                 const BHWC& shape, bool reduce_only,
                 const SoftmaxRuntimeCheckDesc& runtime_check) {
  AddSrcTensor("src_tensor", definition.src_tensors[0]);
  AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  const int reduction_size = DivideRoundUp(shape.c, 4);
  const int total_spatial_size = shape.b * shape.h * shape.w;
  const float spatial_per_cu =
      static_cast<float>(total_spatial_size) / gpu_info.GetComputeUnitsCount();
  const float threshold_per_cu = gpu_info.IsPowerVR() ? 1024.0f : 256.0f;
  if ((reduction_size >= 64 && spatial_per_cu < threshold_per_cu) ||
      (reduction_size >= 16 && spatial_per_cu < threshold_per_cu * 0.25f)) {
    use_wg_reduction_ = true;
    work_group_size_ = GetWorkGroupSize(gpu_info, shape);
  }
  std::string c = GetSoftmaxReducedExp(definition, use_wg_reduction_,
                                       work_group_size_);

  std::string coords = "X, Y";
  if (definition.dst_tensors[0].HasAxis(Axis::kDepth)) {
    coords += ", Z";
  }
  std::string coords_s0 = coords + ", 0";
  std::string coords_s = coords + ", d";
  if (definition.dst_tensors[0].HasAxis(Axis::kBatch)) {
    coords_s0 += ", B";
    coords_s += ", B";
  }

  if (reduce_only) {
    // only reduction step, final step must be done as a separate kernel.
    if (use_wg_reduction_) {
      c += "  if (ucl::GetLocalId<2>() != 0) return;\n";
      if (definition.dst_tensors[0].HasAxis(Axis::kDepth)) {
        c += "  if (X >= args.dst_tensor.Width() || Y >= "
             "args.dst_tensor.Height() || Z >= "
             "args.dst_tensor.Depth()) "
             "return; \n";
      } else {
        c += "  if (X >= args.dst_tensor.Width() || Y >= "
             "args.dst_tensor.Height()) "
             "return; \n";
      }
    }
    c += "  args.dst_tensor::type result;\n";
    c += "  result.x = ucl::Convert<args.dst_tensor::scalar_type>(inv_sum);\n";
    c += "  result.y = ucl::Convert<args.dst_tensor::scalar_type>(maximum);\n";
    c += "  args.dst_tensor.Write(result, " + coords_s0 + ");\n";
    c += "}\n";
  } else {
    // full softmax in one step
    if (use_wg_reduction_) {
      c += "  ucl::SyncThreads<WorkGroup, Local>();\n";
      c += "  if (ucl::GetLocalId<2>() == 0) {\n";
      c += "    float2 value_to_share;\n";
      c += "    value_to_share.x = inv_sum;\n";
      c += "    value_to_share.y = maximum;\n";
      c += "    loc_mem[0][local_xy] = value_to_share;\n";
      c += "  }\n";
      c += "  ucl::SyncThreads<WorkGroup, Local>();\n";
      c += "  value = loc_mem[0][local_xy];\n";
      c += "  inv_sum = value.x;\n";
      c += "  maximum = value.y;\n";
      c += "  if (X >= args.dst_tensor.Width() || Y >= "
           "args.dst_tensor.Height()) "
           "return; \n";
    }
    if (use_wg_reduction_) {
      c += "  for (int d = ucl::GetLocalId<2>(); d < end_slice; "
           "d += WG_SIZE_Z) {\n";
    } else {
      c += "  for (int d = 0; d < end_slice; d += 1) {\n";
    }
    c += "    float4 src = args.src_tensor.Read<float>(" + coords_s + ");\n";
    c += "    if (need_per_channels_check && (d ==  end_slice - 1)) {\n";
    c += "      if (d * 4 + 0 >= end_channel) {\n";
    c += "        src.x = maximum;\n";
    c += "      }\n";
    c += "      if (d * 4 + 1 >= end_channel) {\n";
    c += "        src.y = maximum;\n";
    c += "      }\n";
    c += "      if (d * 4 + 2 >= end_channel) {\n";
    c += "        src.z = maximum;\n";
    c += "      }\n";
    c += "      if (d * 4 + 3 >= end_channel) {\n";
    c += "        src.w = maximum;\n";
    c += "      }\n";
    c += "    }\n";
    c += "    float4 t = EXP_FUNC(src - ucl::Init<float4>(maximum)) * "
         "inv_sum;\n";
    c += "    args.dst_tensor::type result = "
         "ucl::Convert<args.dst_tensor::type>(t);\n";
    c += "    args.dst_tensor.Write(result, " + coords_s + ");\n";
    c += "  }\n";
    c += "}\n";
  }
  const std::string end_channel =
      runtime_check.end_ch_index.has_value()
          ? "min(args.params.Read(args.end_ch_index), "
            "args.src_tensor.Channels())"
          : "args.src_tensor.Channels()";
  const std::string exp_func = gpu_info.IsApiOpenCl() ? "native_exp" : "exp";
  absl::StrReplaceAll({{"EXP_FUNC", exp_func},
                       {"END_CHANNEL", end_channel},
                       {"WG_SIZE_X", std::to_string(work_group_size_.x)},
                       {"WG_SIZE_Y", std::to_string(work_group_size_.y)},
                       {"WG_SIZE_Z", std::to_string(work_group_size_.z)}},
                      &c);
  code_ = c;
  tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_ZIs1;
}

std::vector<int3> Softmax::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info) const {
  if (use_wg_reduction_) {
    return {work_group_size_};
  } else {
    return GetPossibleWorkGroups(tuning_type, gpu_info, kernel_info,
                                 grid_size_);
  }
}

Softmax CreateSoftmax(const OperationDef& definition, const GpuInfo& gpu_info,
                      const BHWC& shape,
                      const SoftmaxRuntimeCheckDesc& runtime_check) {
  return CreateSoftmaxImpl(definition, gpu_info, shape, /*reduce_only=*/false,
                           runtime_check);
}

Softmax CreateSoftmaxReduce(const OperationDef& definition,
                            const GpuInfo& gpu_info, const BHWC& shape,
                            const SoftmaxRuntimeCheckDesc& runtime_check) {
  return CreateSoftmaxImpl(definition, gpu_info, shape, /*reduce_only=*/true,
                           runtime_check);
}

GPUOperation CreateSoftmaxFinal(const OperationDef& definition, int channels) {
  ElementwiseDescriptor op_desc;
  std::string coords = "X_COORD, Y_COORD";
  if (definition.src_tensors[1].HasAxis(Axis::kDepth)) {
    coords += ", Z_COORD";
  }
  coords += ", 0";
  if (definition.src_tensors[1].HasAxis(Axis::kBatch)) {
    coords += ", B_COORD";
  }
  op_desc.code = "  args.src_tensor_1::type exp_val = args.src_tensor_1.Read(" +
                 coords + ");\n";
  if (channels > 0 && channels % 4 != 0) {
    op_desc.args.AddInt("ch_count", channels);
    op_desc.code += R"(
  if (S_COORD * 4 + 1 >= args.ch_count) {
    in_value.y = exp_val.y;
  }
  if (S_COORD * 4 + 2 >= args.ch_count) {
    in_value.z = exp_val.y;
  }
  if (S_COORD * 4 + 3 >= args.ch_count) {
    in_value.w = exp_val.y;
  }
)";
  }
  op_desc.code += "  out_value = exp(in_value - exp_val.y) * exp_val.x;\n";
  return CreateGpuOperation(definition, std::move(op_desc));
}

}  // namespace ml_drift
