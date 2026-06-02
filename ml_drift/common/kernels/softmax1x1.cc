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

#include "ml_drift/common/kernels/softmax1x1.h"

#include <string>
#include <vector>

#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace {
std::string GetReduceCode(int group_reduction_size) {
  std::vector<int> stages;
  if (group_reduction_size == 1024) {
    stages = {8, 8, 4, 4};
  } else if (group_reduction_size == 512) {
    stages = {8, 8, 8};
  } else if (group_reduction_size == 256) {
    stages = {8, 8, 4};
  } else if (group_reduction_size == 128) {
    stages = {8, 4, 4};
  } else if (group_reduction_size == 64) {
    stages = {8, 8};
  } else if (group_reduction_size == 32) {
    stages = {8, 4};
  } else if (group_reduction_size == 16) {
    stages = {4, 4};
  } else if (group_reduction_size <= 8) {
    stages = {group_reduction_size};
  }
  std::string c;
  c += "  loc_mem[tid] = value;\n";
  int stride = 1;
  for (int i = 0; i < stages.size(); ++i) {
    const bool last_stage = i == stages.size() - 1;
    const std::string condition =
        last_stage ? "tid == 0"
                   : "tid % " + std::to_string(stride * stages[i]) + " == 0";
    const std::string location = last_stage ? "loc_mem[0]" : "loc_mem[tid]";
    c += "  ucl::SyncThreads<WorkGroup, Local>();\n";
    c += "  if (" + condition + ") {\n";
    for (int j = 1; j < stages[i]; ++j) {
      c += "    {\n";
      c += "    float2 new_value = loc_mem[tid + " +
           std::to_string(stride * j) + "];\n";
      c += "    float new_max = max(value.y, new_value.y);\n";
      c += "    new_value.x *= EXP_FUNC(new_value.y - new_max);\n";
      c += "    value.x *= EXP_FUNC(value.y - new_max);\n";
      c += "    value.x = value.x + new_value.x;\n";
      c += "    value.y = new_max;\n";
      c += "    }\n";
    }
    c += "    " + location + " = value;\n";
    c += "  }\n";
    stride *= stages[i];
  }
  c += "  ucl::SyncThreads<WorkGroup, Local>();\n";
  c += "  value = loc_mem[0];\n";
  return c;
}

Softmax1x1 CreateSoftmax1x1Impl(const OperationDef& definition,
                                const GpuInfo& gpu_info, const BHWC& shape,
                                bool reduce_only,
                                const SoftmaxRuntimeCheckDesc& runtime_check) {
  auto softmax1x1 =
      Softmax1x1(definition, gpu_info, shape, reduce_only, runtime_check);
  if (runtime_check.end_ch_index.has_value()) {
    softmax1x1.args_.AddInt("end_ch_index", *runtime_check.end_ch_index);
    BufferDescriptor buffer_desc;
    buffer_desc.element_type = DataType::INT32;
    buffer_desc.element_size = 1;
    softmax1x1.AddSrcBuffer("params", buffer_desc);
  }
  return softmax1x1;
}

}  // namespace

Softmax1x1::Softmax1x1(const OperationDef& definition, const GpuInfo& gpu_info,
                       const BHWC& shape, bool reduce_only,
                       const SoftmaxRuntimeCheckDesc& runtime_check)
    : reduce_only_(reduce_only), runtime_check_(runtime_check) {
  // work_group_size_.x must be power of 2
  if (gpu_info.IsAdreno()) {
    if (gpu_info.adreno_info.IsAdreno3xx()) {
      work_group_size_ = int3(32, 1, 1);
    } else if (gpu_info.adreno_info.IsAdreno4xx()) {
      work_group_size_ = int3(64, 1, 1);
    } else if (gpu_info.adreno_info.IsAdreno5xx()) {
      work_group_size_ = int3(128, 1, 1);
    } else {
      // Adreno 6xx and later
      work_group_size_ = int3(512, 1, 1);
    }
  } else if (gpu_info.IsMali()) {
    if (gpu_info.mali_info.IsMidgard()) {
      work_group_size_ = int3(128, 1, 1);
    } else {
      work_group_size_ = int3(1024, 1, 1);
    }
  } else {
    work_group_size_ = int3(256, 1, 1);
  }
  const int end_slice = DivideRoundUp(shape.c, 4);
  while (work_group_size_.x >= end_slice * 2) {
    work_group_size_.x /= 2;
  }
  if (gpu_info.IsAdreno()) {
    while (work_group_size_.x >= gpu_info.GetMaxWorkGroupSizeForX()) {
      work_group_size_.x /= 2;
    }
  } else {
    while (work_group_size_.x > gpu_info.GetMaxWorkGroupSizeForX()) {
      work_group_size_.x /= 2;
    }
  }
  code_ = GetSoftmaxKernelCode(definition, gpu_info);
}

std::string Softmax1x1::GetSoftmaxKernelCode(const OperationDef& op_def,
                                             const GpuInfo& gpu_info) {
  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = ucl::GetGroupId<1>();\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  if (B >= args.dst_tensor.Batch()) return;\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = ucl::GetGroupId<1>();\n";
  }
  std::string coords = "X, Y";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int linear_y = ucl::GetGroupId<2>();\n";
    c += "  int Y = linear_y / args.dst_tensor.Depth();\n";
    c += "  int Z = linear_y % args.dst_tensor.Depth();\n";
    coords += ", Z";
    c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() "
         "|| Z >= args.dst_tensor.Depth()) return;\n";
  } else {
    c += "  int Y = ucl::GetGroupId<2>();\n";
    c += "  if (X >= args.dst_tensor.Width()) return;\n";
    c += "  if (Y >= args.dst_tensor.Height()) return;\n";
  }
  c += "  int tid = ucl::GetLocalId<0>();\n";

  std::string coords_s0 = coords + ", 0";
  std::string coords_s = coords + ", s";
  std::string coords_dst_s = coords + ", dst_s";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    coords_s0 += ", B";
    coords_s += ", B";
    coords_dst_s += ", B";
  }

  c += "  int end_channel = END_CHANNEL;\n";
  c += "  int end_slice = (end_channel + 3) / 4;\n";
  c += "  float sum = 0.0f;\n";
  c += "  bool need_per_channels_check = end_channel "
       "% 4 != 0;\n";
  c += "  float maximum;\n";
  c += "  args.src_tensor.ReadPerChannel<float>(maximum, " + coords_s0 + ");\n";
  c += "  for (int s = tid; s < end_slice; s += "
       "GROUP_REDUCTION_SIZE) {\n";
  c += "    float4 mask_dot = ucl::Init<float4>(1.0f);\n";
  c += "    float4 src = args.src_tensor.Read<float>(" + coords_s + ");\n";
  c += "    if (need_per_channels_check && (s == end_slice - 1)) {\n";
  c += "      if (s * 4 + 0 >= end_channel) {\n";
  c += "        mask_dot.x = 0.0f;\n";
  c += "        src.x = maximum;\n";
  c += "      }\n";
  c += "      if (s * 4 + 1 >= end_channel) {\n";
  c += "        mask_dot.y = 0.0f;\n";
  c += "        src.y = maximum;\n";
  c += "      }\n";
  c += "      if (s * 4 + 2 >= end_channel) {\n";
  c += "        mask_dot.z = 0.0f;\n";
  c += "        src.z = maximum;\n";
  c += "      }\n";
  c += "      if (s * 4 + 3 >= end_channel) {\n";
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
  c += "  float2 value;\n";
  c += "  value.x = sum;\n";
  c += "  value.y = maximum;\n";
  c += "  __local float2 loc_mem[GROUP_REDUCTION_SIZE];\n";
  c += GetReduceCode(work_group_size_.x);
  c += "  float inv_sum = 1.0f / value.x;\n";
  c += "  maximum = value.y;\n";
  if (reduce_only_) {
    c += "  if (tid != 0) return;\n";
    c += "  float4 result;\n";
    c += "  result.x = inv_sum;\n";
    c += "  result.y = maximum;\n";
    c += "  args.dst_tensor.Write(ucl::Convert<args.dst_tensor::type>(result), "
         + coords_s0 + ");\n";
    c += "}\n";
  } else {
    c += "  int dst_s = ucl::GetGlobalId<0>();\n";
    c += "  if (dst_s < end_slice) {\n";
    c +=
        "    float4 src = args.src_tensor.Read<float>(" + coords_dst_s + ");\n";
    c += "    if (need_per_channels_check && (dst_s == end_slice - 1)) {\n";
    c += "      if (dst_s * 4 + 0 >= end_channel) {\n";
    c += "        src.x = maximum;\n";
    c += "      }\n";
    c += "      if (dst_s * 4 + 1 >= end_channel) {\n";
    c += "        src.y = maximum;\n";
    c += "      }\n";
    c += "      if (dst_s * 4 + 2 >= end_channel) {\n";
    c += "        src.z = maximum;\n";
    c += "      }\n";
    c += "      if (dst_s * 4 + 3 >= end_channel) {\n";
    c += "        src.w = maximum;\n";
    c += "      }\n";
    c += "    }\n";
    c += "    float4 t = EXP_FUNC(src - ucl::Init<float4>(maximum)) * "
         "inv_sum;\n";
    c += "    args.dst_tensor::type result = "
         "ucl::Convert<args.dst_tensor::type>(t);\n";
    c += "    args.dst_tensor.Write(result, " + coords_dst_s + ");\n";
    c += "  }\n";
    c += "}\n";
  }
  const std::string exp_func = gpu_info.IsApiOpenCl() ? "native_exp" : "exp";
  const std::string end_channel = runtime_check_.end_ch_index.has_value()
                                      ? "args.params.Read(args.end_ch_index)"
                                      : "args.src_tensor.Channels()";
  absl::StrReplaceAll(
      {{"EXP_FUNC", exp_func},
       {"END_CHANNEL", end_channel},
       {"GROUP_REDUCTION_SIZE", std::to_string(work_group_size_.x)}},
      &c);
  return c;
}

int3 Softmax1x1::GetGridSize() const {
  const int grid_x = reduce_only_ ? 1 : dst_[0]->Slices();
  return int3(grid_x, dst_[0]->Width() * dst_[0]->Batch(),
              dst_[0]->Height() * dst_[0]->Depth());
}

Softmax1x1 CreateSoftmax1x1(const OperationDef& definition,
                            const GpuInfo& gpu_info, const BHWC& shape,
                            const SoftmaxRuntimeCheckDesc& runtime_check) {
  return CreateSoftmax1x1Impl(definition, gpu_info, shape,
                              /*reduce_only=*/false, runtime_check);
}

Softmax1x1 CreateSoftmax1x1Reduce(
    const OperationDef& definition, const GpuInfo& gpu_info, const BHWC& shape,
    const SoftmaxRuntimeCheckDesc& runtime_check) {
  return CreateSoftmax1x1Impl(definition, gpu_info, shape, /*reduce_only=*/true,
                              runtime_check);
}

}  // namespace ml_drift
