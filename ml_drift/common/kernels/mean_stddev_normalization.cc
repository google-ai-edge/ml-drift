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

#include "ml_drift/common/kernels/mean_stddev_normalization.h"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

namespace {
std::string GetReduceCode(const std::string& value, int3 work_group_size,
                          bool two_step) {
  int reduction_size = work_group_size.z;
  std::string mem_name =
      work_group_size.x * work_group_size.y != 1
          ? "shared_mem[ucl::GetLocalId<1>()][ucl::GetLocalId<0>()]"
          : "shared_mem";
  if (reduction_size <= 8) {
    std::string result;
    result += "  {  // reduction\n";
    result += "    " + mem_name + "[local_id] = " + value + ";\n";
    result += "    ucl::SyncThreads<WorkGroup, Local>();\n";
    result += "    if (ucl::GetLocalId<2>() == 0) {\n";
    result += "      " + value + " = " + mem_name + "[0];\n";
    for (int i = 1; i < reduction_size; ++i) {
      result += "      " + value + " += " + mem_name + "[" + std::to_string(i) +
                "];\n";
    }
    result += "      " + mem_name + "[0] = " + value + ";\n";
    result += "    }\n";
    result += "    ucl::SyncThreads<WorkGroup, Local>();\n";
    result += "    " + value + " = " + mem_name + "[0];\n";
    if (two_step) {
      result += "    ucl::SyncThreads<WorkGroup, Local>();\n";
    }
    result += "  }\n";
    return result;
  } else {
    // In the reduction step add upper half of the still-to-be-summed vector to
    // the lower half, while taking care of odd sizes and rounding. E.g.:
    // Number of items still to be summed before: 5
    // Local memory before: [a, b, c, d, e];
    // Local memory after: [a+d, b+e, c, d, e];
    // Threads doing work: id < 2 = floor(5/2)
    // Offset to the added items: 3 = ceil(5/2)
    // Number of items still to be summed after: 3 = ceil(5/2)
    return absl::Substitute(R"(
  {  // reduction, all threads inside workgroup must execute this code
    $2[local_id] = $1;
    ucl::SyncThreads<WorkGroup, Local>();
    // The number of items still need to be summed
    int reduction_size = $0;
    while (reduction_size > 1) {
      int active_thread_limit = reduction_size / 2;
      int offset = (reduction_size + 1) / 2;
      if (local_id < active_thread_limit) {
        $1 += $2[local_id + offset];
        $2[local_id] = $1;
      }
      ucl::SyncThreads<WorkGroup, Local>();
      reduction_size = offset;
    }
    $1 = $2[0];
  }
)",
                            reduction_size, value, mem_name);
  }
}

std::string ZeroClampVec4Code(const std::string& slice_name,
                              const std::string& channels_name,
                              const std::string& value_name) {
  return absl::Substitute(R"(
    // no need to check first element, always valid
    if ($0 * 4 + 1 >= $1) { $2.y = 0.0f; }
    if ($0 * 4 + 2 >= $1) { $2.z = 0.0f; }
    if ($0 * 4 + 3 >= $1) { $2.w = 0.0f; }
)",
                          slice_name, channels_name, value_name);
}

bool UseWorkGroupReduction(const GpuInfo& gpu_info, const BHWC& shape) {
  const int tensor_slices = DivideRoundUp(shape.c, 4);
  if (gpu_info.IsAdreno() && tensor_slices <= 32 &&
      shape.w * shape.h * shape.b >= 128) {
    return false;
  } else {
    return true;
  }
}

int3 GetRecommendedWorkGroupSize(const GpuInfo& gpu_info, const BHWC& shape) {
  const int tensor_slices = DivideRoundUp(shape.c, 4);
  int desired_work_group_size = gpu_info.GetMaxWorkGroupSizeForZ();
  const int total_spatial_size = shape.b * shape.h * shape.w;
  const float spatial_per_cu =
      static_cast<float>(total_spatial_size) / gpu_info.GetComputeUnitsCount();
  if (gpu_info.IsMali()) {
    // Don't use more than 64 work items per work group on ARM Mali. They
    // implement local memory using the global memory, larger workgroups have
    // severe performance penalty.
    desired_work_group_size = 64;
  }
  if (gpu_info.IsAdreno()) {
    AdrenoInfo info = gpu_info.adreno_info;
    desired_work_group_size = 256;
    if (info.IsAdreno3xx()) {
      if (info.adreno_gpu == AdrenoGpu::kAdreno320 ||
          info.adreno_gpu == AdrenoGpu::kAdreno330) {
        desired_work_group_size = 128;
      } else {
        desired_work_group_size = 64;
      }
    } else if (info.IsAdreno4xx()) {
      if (info.adreno_gpu == AdrenoGpu::kAdreno430) {
        desired_work_group_size = 256;
      } else {
        desired_work_group_size = 128;
      }
    } else if (info.IsAdreno5xx()) {
      if (info.adreno_gpu == AdrenoGpu::kAdreno530 ||
          info.adreno_gpu == AdrenoGpu::kAdreno540) {
        desired_work_group_size = 256;
      } else {
        desired_work_group_size = 128;
      }
    }
  }
  if (gpu_info.IsPowerVR()) {
    desired_work_group_size = 64;
  }
  if (gpu_info.IsApple()) {
    desired_work_group_size = 64;
  }
  if (gpu_info.IsAMD()) {
    if (spatial_per_cu > 16) {
      desired_work_group_size = 512;
    } else if (spatial_per_cu > 8) {
      desired_work_group_size = 256;
    } else {
      desired_work_group_size = 128;
    }
  }
  if (gpu_info.IsIntel()) {
    desired_work_group_size = 128;
  }
  int3 work_group_size(1, 1, 1);
  if (spatial_per_cu < 2.0f) {
    desired_work_group_size = gpu_info.GetMaxWorkGroupSizeForZ();
    while (desired_work_group_size >= tensor_slices * 2) {
      desired_work_group_size /= 2;
    }
    work_group_size.x = 1;
    work_group_size.y = 1;
    work_group_size.z = desired_work_group_size;
  } else {
    if (spatial_per_cu < 8 && tensor_slices % 64 == 0) {
      work_group_size.z = 32;
    } else if (spatial_per_cu < 32 && tensor_slices % 32 == 0) {
      work_group_size.z = 16;
    } else if (tensor_slices >= 16) {
      work_group_size.z = 8;
    } else if (tensor_slices >= 10) {
      work_group_size.z = 4;
    } else {
      std::map<int, int> slices_to_group_size = {
          {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 3},
          {6, 3}, {7, 4}, {8, 4}, {9, 3},
      };
      work_group_size.z = slices_to_group_size[tensor_slices];
    }
    desired_work_group_size =
        std::min(desired_work_group_size, gpu_info.GetMaxWorkGroupTotalSize());
    work_group_size.x =
        desired_work_group_size / AlignByN(work_group_size.z, 4);
    work_group_size.y = 1;
    while (work_group_size.x >= (shape.b * shape.w) * 2) {
      work_group_size.x /= 2;
      work_group_size.y *= 2;
    }
    while (work_group_size.y >= shape.h * 2) {
      work_group_size.y /= 2;
    }
  }
  return work_group_size;
}

bool IsBadAdrenoDriver(const GpuInfo& gpu_info) {
  return gpu_info.IsApiOpenCl() && gpu_info.IsAdreno() &&
         gpu_info.adreno_info.IsAdreno8xx() &&
         gpu_info.adreno_info.cl_compiler_version.major == 47 &&
         gpu_info.adreno_info.cl_compiler_version.minor == 12;
}

std::string GetVarianceCalculationCode(const GpuInfo& gpu_info,
                                       bool work_group_reduction,
                                       const int3& work_group_size,
                                       bool has_batch, bool channels_x4,
                                       bool two_step, bool has_depth) {
  std::string c;
  if (work_group_reduction && gpu_info.IsApiOpenCl()) {
    c += "__attribute__((reqd_work_group_size(" +
         std::to_string(work_group_size.x) + ", " +
         std::to_string(work_group_size.y) + ", " +
         std::to_string(work_group_size.z) + ")))\n";
  }
  c += "MAIN_FUNCTION($0) {\n";
  if (work_group_reduction) {
    std::string accum_type = two_step ? "float" : "float2";
    if (work_group_size.x * work_group_size.y == 1) {
      c += "__local " + accum_type + " shared_mem[" +
           std::to_string(work_group_size.z) + "];\n";
    } else {
      c += "__local " + accum_type + " shared_mem[" +
           std::to_string(work_group_size.y) + "][" +
           std::to_string(work_group_size.x) + "][" +
           std::to_string(work_group_size.z) + "];\n";
    }
  }
  if (has_batch) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = ucl::GetGlobalId<0>();\n";
  }
  if (has_depth) {
    c += "  int linear_y = ucl::GetGlobalId<1>();\n";
    c += "  int Y = linear_y / args.dst_tensor.Depth();\n";
    c += "  int D = linear_y % args.dst_tensor.Depth();\n";
  } else {
    c += "  int Y = ucl::GetGlobalId<1>();\n";
  }
  if (!work_group_reduction) {
    c += "  if (X >= args.dst_tensor.Width()) { return; }\n";
    c += "  if (Y >= args.dst_tensor.Height()) { return; }\n";
  }
  if (!two_step) {
    c += "  float4 private_sum4_sq = ucl::Init<float4>(0.0f);\n";
  }
  if (work_group_reduction) {
    c += "  int local_id = ucl::GetLocalId<2>();\n";
    c += "  int reduction_group_size = ucl::GetGroupSize<2>();\n";
  } else {
    c += "  int local_id = 0;\n";
    c += "  int reduction_group_size = 1;\n";
  }
  c += R"(
  float4 private_sum4 = ucl::Init<float4>(0.0f);
  for (int S = local_id; S < args.src_tensor.Slices(); S += reduction_group_size) {
    int x_clamped = min(X, args.src_tensor.Width() - 1);
    int y_clamped = min(Y, args.src_tensor.Height() - 1);
)";
  if (has_depth) {
    c += "    int d_clamped = min(D, args.src_tensor.Depth() - 1);\n";
    c += "    float4 t = args.src_tensor.Read<float>(x_clamped, y_clamped, "
         "d_clamped, S);\n";
  } else {
    c += "    float4 t = args.src_tensor.Read<float>(x_clamped, y_clamped, "
         "S);\n";
  }
  if (!channels_x4) {
    c += ZeroClampVec4Code("S", "args.src_tensor.Channels()", "t");
  }
  if (two_step) {
    c += "    private_sum4 += t;\n";
    c += "  }\n";
    c += "  float sum = dot(private_sum4, ucl::Init<float4>(1.0f));\n";
  } else {
    c += "    private_sum4 += t;\n";
    c += "    private_sum4_sq += t * t;\n";
    c += "  }\n";
    c += "  float2 sum;\n";
    c += "  sum.x = dot(private_sum4, ucl::Init<float4>(1.0f));\n";
    c += "  sum.y = dot(private_sum4_sq, ucl::Init<float4>(1.0f));\n";
  }
  if (work_group_reduction) {
    c += GetReduceCode("sum", work_group_size, two_step);
  }
  if (two_step) {
    if (IsBadAdrenoDriver(gpu_info)) {
      c += R"(
  float mean = sum;
  if (X != -1) {
    mean *= args.inv_ch_count;
  }
    )";
    } else {
      c += "  float mean = sum * args.inv_ch_count;\n";
    }
    c += R"(
  // Calculate the squared sum of the difference from the mean.
  float4 private_sum_diff_sq4 = ucl::Init<float4>(0.0f);
  for (int S = local_id; S < args.src_tensor.Slices(); S += reduction_group_size) {
    int x_clamped = min(X, args.src_tensor.Width() - 1);
    int y_clamped = min(Y, args.src_tensor.Height() - 1);
)";
    if (has_depth) {
      c += "    int d_clamped = min(D, args.src_tensor.Depth() - 1);\n";
      c += "    float4 t = args.src_tensor.Read<float>(x_clamped, y_clamped, "
           "d_clamped, S);\n";
    } else {
      c += "    float4 t = args.src_tensor.Read<float>(x_clamped, y_clamped, "
           "S);\n";
    }
    c += R"(
    float4 diff = t - mean;)";
    if (!channels_x4) {
      c += ZeroClampVec4Code("S", "args.src_tensor.Channels()", "diff");
    }
    c += R"(
    private_sum_diff_sq4 += diff * diff;
  }
  // Reduce
  float sum_diff_sq = dot(private_sum_diff_sq4, ucl::Init<float4>(1.0f));
)";
    if (work_group_reduction) {
      c += GetReduceCode("sum_diff_sq", work_group_size, two_step);
    }
    if (IsBadAdrenoDriver(gpu_info)) {
      c += R"(
  float variance = sum_diff_sq;
  if (X != -1) {
    variance *= args.inv_ch_count;
  }
)";
    } else {
      c += "  float variance = sum_diff_sq * args.inv_ch_count;\n";
    }
  } else {
    if (IsBadAdrenoDriver(gpu_info)) {
      c += R"(
  float mean = sum.x;
  float mean_sq = sum.y;
  if (X != -1) {
    mean *= args.inv_ch_count;
    mean_sq *= args.inv_ch_count;
  }
)";
    } else {
      c += "  float mean = sum.x * args.inv_ch_count;\n";
      c += "  float mean_sq = sum.y * args.inv_ch_count;\n";
    }
    c += "  float variance = max(0.0f, mean_sq - mean * mean);\n";
  }
  if (work_group_reduction) {
    c += "  // no more shared memory usage, 'useless' threads can exit now\n";
    c += "  if (X >= args.dst_tensor.Width()) { return; }\n";
    c += "  if (Y >= args.dst_tensor.Height()) { return; }\n";
  }
  return c;
}

std::string GetHWCNormalizationCodeGroupSize1(const GpuInfo& gpu_info,
                                              const OperationDef& op_def,
                                              const int3& work_group_size) {
  std::string c;
  if (gpu_info.IsApiOpenCl()) {
    c += "__attribute__((reqd_work_group_size(WG_X, WG_Y, WG_Z)))\n";
  }
  c += "MAIN_FUNCTION($0) {\n";
  c += "__local float4 shared_mem0[WG_Z][WG_SPATIAL];\n";
  c += "__local float4 shared_mem1[WG_Z][WG_SPATIAL];\n";
  if (op_def.src_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int B = linear_id / WG_X;\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  }
  c += R"(
  int S = ucl::GetGlobalId<2>();
  int local_id = ucl::GetLocalId<1>() * WG_X + ucl::GetLocalId<0>();
  int local_s = ucl::GetLocalId<2>();
  float4 sum = ucl::Init<float4>(0.0f);
  float4 sum_squares = ucl::Init<float4>(0.0f);
)";
  std::string coords = "X, Y";
  if (op_def.src_tensors[0].HasAxis(Axis::DEPTH)) {
    coords += ", D";
    c += "  for (int linear_y = ucl::GetLocalId<1>(); linear_y < "
         "args.src_tensor.Height() * args.src_tensor.Depth(); linear_y += "
         "WG_Y) {\n";
    c += "    int Y = linear_y / args.src_tensor.Depth();\n";
    c += "    int D = linear_y % args.src_tensor.Depth();\n";
  } else {
    c += "  for (int Y = ucl::GetLocalId<1>(); Y < args.src_tensor.Height(); Y "
         "+= WG_Y) {\n";
  }
  coords += ", S";

  c += "    for (int X = ucl::GetLocalId<0>(); X < args.src_tensor.Width(); X "
       "+= WG_X) {\n";
  c += "      float4 t = args.src_tensor.Read<float>(" + coords + ");\n";
  c += R"(
      sum += t;
      sum_squares += t * t;
    }
  }
  float4 reduced_sum = sum;
  float4 reduced_sum_squares = sum_squares;
  {  // reduction, all threads inside workgroup must execute this code
    shared_mem0[local_s][local_id] = reduced_sum;
    shared_mem1[local_s][local_id] = reduced_sum_squares;
    ucl::SyncThreads<WorkGroup, Local>();
    // The number of items still need to be summed
    int reduction_size = WG_Y * WG_X;
    while (reduction_size > 1) {
      int active_thread_limit = reduction_size / 2;
      int offset = (reduction_size + 1) / 2;
      if (local_id < active_thread_limit) {
        reduced_sum += shared_mem0[local_s][local_id + offset];
        reduced_sum_squares += shared_mem1[local_s][local_id + offset];
        shared_mem0[local_s][local_id] = reduced_sum;
        shared_mem1[local_s][local_id] = reduced_sum_squares;
      }
      ucl::SyncThreads<WorkGroup, Local>();
      reduction_size = offset;
    }
    reduced_sum = shared_mem0[local_s][0];
    reduced_sum_squares = shared_mem1[local_s][0];
  }
  float4 mean = reduced_sum * args.inv_elements_count;
  float4 mean_sq = reduced_sum_squares * args.inv_elements_count;
  float4 variance = mean_sq - mean * mean;
  float4 stddev_inv = rsqrt(variance + args.variance_bias);
  // Calculate (t-mean)/stddev for each element
)";
  if (op_def.src_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  for (int linear_y = ucl::GetLocalId<1>(); linear_y < "
         "args.src_tensor.Height() * args.src_tensor.Depth(); linear_y += "
         "WG_Y) {\n";
    c += "    int Y = linear_y / args.src_tensor.Depth();\n";
    c += "    int D = linear_y % args.src_tensor.Depth();\n";
  } else {
    c += "  for (int Y = ucl::GetLocalId<1>(); Y < args.src_tensor.Height(); Y "
         "+= WG_Y) {\n";
  }
  c += "    for (int X = ucl::GetLocalId<0>(); X < args.src_tensor.Width(); X "
       "+= WG_X) {\n";
  c += "      float4 t = args.src_tensor.Read<float>(" + coords + ");\n";
  c += R"(
      float4 t_normalized = (t - mean) * stddev_inv;
      t_normalized *= args.gamma.Read<float>(S);
      t_normalized += args.beta.Read<float>(S);
      args.dst_tensor::type result = ucl::Convert<args.dst_tensor::type>(t_normalized);
)";
  c += "      args.dst_tensor.Write(result, " + coords + ");\n";
  c += "    }\n  }\n}\n";

  return absl::StrReplaceAll(
      c,
      {{"WG_X", std::to_string(work_group_size.x)},
       {"WG_Y", std::to_string(work_group_size.y)},
       {"WG_Z", std::to_string(work_group_size.z)},
       {"WG_SPATIAL", std::to_string(work_group_size.x * work_group_size.y)}});
}

std::string GetHWCNormalizationCode(const GpuInfo& gpu_info,
                                    const OperationDef& op_def,
                                    const int3& work_group_size,
                                    int group_size) {
  std::string c;
  if (gpu_info.IsApiOpenCl()) {
    c += "__attribute__((reqd_work_group_size(" +
         std::to_string(work_group_size.x) + ", " +
         std::to_string(work_group_size.y) + ", " +
         std::to_string(work_group_size.z) + ")))\n";
  }
  const int wg_total_size =
      work_group_size.x * work_group_size.y * work_group_size.z;
  c += "MAIN_FUNCTION($0) {\n";
  if (group_size % 4 == 0) {
    c += "__local float2 shared_mem[" + std::to_string(wg_total_size) + "];\n";
  } else {
    c += "__local float4 shared_mem[" + std::to_string(wg_total_size) + "];\n";
  }
  if (op_def.src_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int B = linear_id / ucl::GetGroupSize<0>();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  }
  c += R"(
  int local_id = (ucl::GetLocalId<2>() * ucl::GetGroupSize<1>() + ucl::GetLocalId<1>()) * ucl::GetGroupSize<0>() + ucl::GetLocalId<0>();
  int group_start = ucl::GetGroupId<2>() * args.group_size + ucl::GetLocalId<2>();
  int group_end = (ucl::GetGroupId<2>() + 1) * args.group_size;
  group_end = min(group_end, args.src_tensor.Slices());
  float2 sum0;
  sum0.x = 0.0f;
  sum0.y = 0.0f;
)";
  if (group_size % 4 != 0) {
    c += "  float2 sum1;\n";
    c += "  sum1.x = 0.0f;\n";
    c += "  sum1.y = 0.0f;\n";
  }
  if (group_size % 4 == 0) {
    c += R"(
  for (int S = group_start; S < group_end; S += ucl::GetGroupSize<2>()) {
    for (int Y = ucl::GetLocalId<1>(); Y < args.src_tensor.Height(); Y += ucl::GetGroupSize<1>()) {
      for (int X = ucl::GetLocalId<0>(); X < args.src_tensor.Width(); X += ucl::GetGroupSize<0>()) {
        float4 t = args.src_tensor.Read<float>(X, Y, S);
        sum0.x += dot(ucl::Init<float4>(1.0f), t);
        sum0.y += dot(ucl::Init<float4>(1.0f), t * t);
      }
    }
  }
  float2 reduce_sum;
  reduce_sum.x = sum0.x;
  reduce_sum.y = sum0.y;
)";
  } else {
    c += R"(
  for (int S = group_start; S < group_end; S += ucl::GetGroupSize<2>()) {
    int gr0_start = group_start * 4;
    int gr1_start = gr0_start + args.group_size_ch;
    int gr2_start = gr1_start + args.group_size_ch;
    float4 mask0;
    mask0.x = S * 4 + 0 >= gr0_start && S * 4 + 0 < gr1_start ? 1.0f : 0.0f;
    mask0.y = S * 4 + 1 >= gr0_start && S * 4 + 1 < gr1_start ? 1.0f : 0.0f;
    mask0.z = S * 4 + 2 >= gr0_start && S * 4 + 2 < gr1_start ? 1.0f : 0.0f;
    mask0.w = S * 4 + 3 >= gr0_start && S * 4 + 3 < gr1_start ? 1.0f : 0.0f;
    float4 mask1;
    mask1.x = S * 4 + 0 >= gr1_start && S * 4 + 0 < gr2_start ? 1.0f : 0.0f;
    mask1.y = S * 4 + 1 >= gr1_start && S * 4 + 1 < gr2_start ? 1.0f : 0.0f;
    mask1.z = S * 4 + 2 >= gr1_start && S * 4 + 2 < gr2_start ? 1.0f : 0.0f;
    mask1.w = S * 4 + 3 >= gr1_start && S * 4 + 3 < gr2_start ? 1.0f : 0.0f;
    for (int Y = ucl::GetLocalId<1>(); Y < args.src_tensor.Height(); Y += ucl::GetGroupSize<1>()) {
      for (int X = ucl::GetLocalId<0>(); X < args.src_tensor.Width(); X += ucl::GetGroupSize<0>()) {
        float4 t = args.src_tensor.Read<float>(X, Y, S);
        sum0.x += dot(mask0, t);
        sum0.y += dot(mask0, t * t);
        sum1.x += dot(mask1, t);
        sum1.y += dot(mask1, t * t);
      }
    }
  }
  float4 reduce_sum;
  reduce_sum.x = sum0.x;
  reduce_sum.y = sum0.y;
  reduce_sum.z = sum1.x;
  reduce_sum.w = sum1.y;
)";
  }
  if (IsBadAdrenoDriver(gpu_info)) {
    c += R"(
  shared_mem[local_id] = reduce_sum;
  ucl::SyncThreads<WorkGroup, Local>();
  reduce_sum = shared_mem[0];
)";
    for (int i = 1; i < wg_total_size; i += 1) {
      c += "  reduce_sum += shared_mem[" + std::to_string(i) + "];\n";
    }
  } else {
    c += absl::Substitute(R"(
    {  // reduction, all threads inside workgroup must execute this code
      $2[local_id] = $1;
      ucl::SyncThreads<WorkGroup, Local>();
      // The number of items still need to be summed
      int reduction_size = $0;
      while (reduction_size > 1) {
        int active_thread_limit = reduction_size / 2;
        int offset = (reduction_size + 1) / 2;
        if (local_id < active_thread_limit) {
          $1 += $2[local_id + offset];
          $2[local_id] = $1;
        }
        ucl::SyncThreads<WorkGroup, Local>();
        reduction_size = offset;
      }
      $1 = $2[0];
      ucl::SyncThreads<WorkGroup, Local>();
    }
  )",
                          wg_total_size, "reduce_sum", "shared_mem");
  }

  if (group_size % 4 == 0) {
    c += R"(
  float mean = reduce_sum.x * args.inv_elements_count;
  float mean_sq = reduce_sum.y * args.inv_elements_count;
  float variance = mean_sq - mean * mean;
  float stddev_inv = rsqrt(variance + args.variance_bias);
  // Calculate (t-mean)/stddev for each element
  for (int S = group_start; S < group_end; S += ucl::GetGroupSize<2>()) {
)";
  } else {
    c += R"(
  float mean0 = reduce_sum.x * args.inv_elements_count;
  float mean_sq0 = reduce_sum.y * args.inv_elements_count;
  float variance0 = mean_sq0 - mean0 * mean0;
  float stddev_inv0 = rsqrt(variance0 + args.variance_bias);
  float mean1 = reduce_sum.z * args.inv_elements_count;
  float mean_sq1 = reduce_sum.w * args.inv_elements_count;
  float variance1 = mean_sq1 - mean1 * mean1;
  float stddev_inv1 = rsqrt(variance1 + args.variance_bias);
  // Calculate (t-mean)/stddev for each element
  for (int S = group_start; S < group_end; S += ucl::GetGroupSize<2>()) {
    int gr0_start = group_start * 4;
    int gr1_start = gr0_start + args.group_size_ch;
    int gr2_start = gr1_start + args.group_size_ch;
    float4 mask0;
    mask0.x = S * 4 + 0 >= gr0_start && S * 4 + 0 < gr1_start ? 1.0f : 0.0f;
    mask0.y = S * 4 + 1 >= gr0_start && S * 4 + 1 < gr1_start ? 1.0f : 0.0f;
    mask0.z = S * 4 + 2 >= gr0_start && S * 4 + 2 < gr1_start ? 1.0f : 0.0f;
    mask0.w = S * 4 + 3 >= gr0_start && S * 4 + 3 < gr1_start ? 1.0f : 0.0f;
    float4 mask1;
    mask1.x = S * 4 + 0 >= gr1_start && S * 4 + 0 < gr2_start ? 1.0f : 0.0f;
    mask1.y = S * 4 + 1 >= gr1_start && S * 4 + 1 < gr2_start ? 1.0f : 0.0f;
    mask1.z = S * 4 + 2 >= gr1_start && S * 4 + 2 < gr2_start ? 1.0f : 0.0f;
    mask1.w = S * 4 + 3 >= gr1_start && S * 4 + 3 < gr2_start ? 1.0f : 0.0f;
    float4 mean = mask0 * mean0 + mask1 * mean1;
    float4 stddev_inv = mask0 * stddev_inv0 + mask1 * stddev_inv1;
)";
  }
  c += R"(
    for (int Y = ucl::GetLocalId<1>(); Y < args.src_tensor.Height(); Y += ucl::GetGroupSize<1>()) {
      for (int X = ucl::GetLocalId<0>(); X < args.src_tensor.Width(); X += ucl::GetGroupSize<0>()) {
        float4 t = args.src_tensor.Read<float>(X, Y, S);
        float4 t_normalized = (t - mean) * stddev_inv;
        t_normalized *= args.gamma.Read<float>(S);
        t_normalized += args.beta.Read<float>(S);
        args.dst_tensor::type result = ucl::Convert<args.dst_tensor::type>(t_normalized);
        args.dst_tensor.Write(result, X, Y, S);
      }
    }
  }
})";
  return absl::StrReplaceAll(
      c, {{"ucl::GetGroupSize<0>()", std::to_string(work_group_size.x)},
          {"ucl::GetGroupSize<1>()", std::to_string(work_group_size.y)},
          {"ucl::GetGroupSize<2>()", std::to_string(work_group_size.z)}});
}
}  // namespace

MeanStdDevNormalization::MeanStdDevNormalization(
    const OperationDef& definition, const GpuInfo& gpu_info, const BHWC& shape,
    float variance_bias, bool two_step,
    const Tensor<Linear, DataType::FLOAT32>* gamma,
    const Tensor<Linear, DataType::FLOAT32>* beta) {
  work_group_reduction_ = UseWorkGroupReduction(gpu_info, shape);
  if (work_group_reduction_) {
    work_group_size_ = GetRecommendedWorkGroupSize(gpu_info, shape);
  } else {
    work_group_size_ = int3(8, 8, 1);
  }
  args_.AddFloat("variance_bias", variance_bias);
  args_.AddFloat("inv_ch_count", 1.0f / shape.c);
  has_depth_ = definition.src_tensors[0].HasAxis(Axis::DEPTH);
  AddSrcTensor("src_tensor", definition.src_tensors[0]);
  AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  if (gamma) {
    TensorDescriptor gamma_tensor_desc = CreateConstantLinearTensorDescriptor(
        gpu_info, definition.src_tensors[0].GetDataType(), *gamma);
    args_.AddObject("gamma", std::make_unique<TensorDescriptor>(
                                 std::move(gamma_tensor_desc)));
  }
  if (beta) {
    TensorDescriptor beta_tensor_desc = CreateConstantLinearTensorDescriptor(
        gpu_info, definition.src_tensors[0].GetDataType(), *beta);
    args_.AddObject("beta", std::make_unique<TensorDescriptor>(
                                std::move(beta_tensor_desc)));
  }
  code_ = GetNormalizationCode(gpu_info,
                               definition.dst_tensors[0].HasAxis(Axis::BATCH),
                               shape.c % 4 == 0, two_step, gamma != nullptr,
                               beta != nullptr, has_depth_);
}

std::string MeanStdDevNormalization::GetNormalizationCode(
    const GpuInfo& gpu_info, bool has_batch, bool channels_x4, bool two_step,
    bool gamma, bool beta, bool has_depth) {
  std::string c = GetVarianceCalculationCode(gpu_info, work_group_reduction_,
                                             work_group_size_, has_batch,
                                             channels_x4, two_step, has_depth);
  if (IsBadAdrenoDriver(gpu_info)) {
    c += R"(
  if (X != -1) {
    variance += args.variance_bias;
  }
  float stddev_inv = rsqrt(variance);
)";
  } else {
    c += "  float stddev_inv = rsqrt(variance + args.variance_bias);\n";
  }
  c += R"(
  // Calculate (t-mean)/stddev for each element
  for (int S = local_id; S < args.src_tensor.Slices(); S += reduction_group_size) {
  )";
  const std::string coords = has_depth ? "X, Y, D, S" : "X, Y, S";
  c += "    float4 t = args.src_tensor.Read<float>(" + coords + ");\n";
  c += R"(
    // Known to be flaky on some devices / backends (e.g. non-OpenCL on Mali)
    // when t == mean and stddev_inv is large. Possible compiler / driver bug.
    float4 t_normalized = (t - mean) * stddev_inv;
  )";
  if (gamma) {
    c += "    t_normalized *= args.gamma.Read<float>(S);\n";
  }
  if (beta) {
    c += "    t_normalized += args.beta.Read<float>(S);\n";
  }
  c += R"(
    args.dst_tensor::type result = ucl::Convert<args.dst_tensor::type>(t_normalized);
  )";
  c += "    args.dst_tensor.Write(result, " + coords + ");\n";
  c += R"(
  }
  })";
  return c;
}

std::string GetRMSNormalizationCode(const GpuInfo& gpu_info, bool has_batch,
                                    bool channels_x4, bool work_group_reduction,
                                    const int3& work_group_size,
                                    bool has_depth) {
  std::string c = GetVarianceCalculationCode(
      gpu_info, work_group_reduction, work_group_size, has_batch, channels_x4,
      /*two_step=*/false, has_depth);
  // Current version of the kernel(Jun 2025) cause compilation error on some
  // Adreno devices.
  // Adreno(TM) 830, Compiler E031.47.12.01: ENCODE FATAL ERROR
  // (vendor/qcom/proprietary/graphics/adreno200/shadercompiler/llvm/lib/Target/Oxili/QGPUMachineEncoder.cpp:1083:
  // address register can't be src operand in direct addressing
  // Adreno(TM) 830, Compiler E031.47.18.30: compiled without error.
  if (IsBadAdrenoDriver(gpu_info)) {
    c += R"(
      if (X != -1) {
        mean_sq += args.variance_bias;
      }
      float stddev_inv = rsqrt(mean_sq);
)";
  } else {
    c += "  float stddev_inv = rsqrt(mean_sq + args.variance_bias);\n";
  }
  c += R"(
  for (int S = local_id; S < args.src_tensor.Slices(); S += reduction_group_size) {
)";
  if (has_depth) {
    c += "    float4 t = args.src_tensor.Read<float>(X, Y, D, S);\n";
  } else {
    c += "    float4 t = args.src_tensor.Read<float>(X, Y, S);\n";
  }
  c += R"(
    float4 t_normalized = t * stddev_inv;
)";
  c += R"(
    args.dst_tensor::type result = ucl::Convert<args.dst_tensor::type>(t_normalized);
)";
  if (has_depth) {
    c += "    args.dst_tensor.Write(result, X, Y, D, S);\n";
  } else {
    c += "    args.dst_tensor.Write(result, X, Y, S);\n";
  }
  c += R"(
  }
})";
  return c;
}

std::string GetStatisticalTopKCode(const GpuInfo& gpu_info, bool has_batch,
                                   bool channels_x4, bool work_group_reduction,
                                   const int3& work_group_size,
                                   bool has_depth) {
  std::string c = GetVarianceCalculationCode(
      gpu_info, work_group_reduction, work_group_size, has_batch, channels_x4,
      /*two_step=*/false, has_depth);
  c += R"(
  float cutoff = mean + args.stddev_multiplier * sqrt(variance);
  for (int S = local_id; S < args.src_tensor.Slices(); S += reduction_group_size) {
)";
  if (has_depth) {
    c += "    float4 t = args.src_tensor.Read<float>(X, Y, D, S);\n";
  } else {
    c += "    float4 t = args.src_tensor.Read<float>(X, Y, S);\n";
  }
  c += R"(
    float4 t_cut = max(t, ucl::Init<float4>(cutoff)) - ucl::Init<float4>(cutoff);
    args.dst_tensor::type result = ucl::Convert<args.dst_tensor::type>(t_cut);
)";
  if (has_depth) {
    c += "    args.dst_tensor.Write(result, X, Y, D, S);\n";
  } else {
    c += "    args.dst_tensor.Write(result, X, Y, S);\n";
  }
  c += R"(
  }
})";
  return c;
}

int3 MeanStdDevNormalization::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = work_group_reduction_ ? work_group_size_.z : 1;
  return int3(grid_x, grid_y, grid_z);
}

MeanStdDevNormalization CreateMeanStdDevNormalization(
    const OperationDef& definition, const GpuInfo& gpu_info, const BHWC& shape,
    float variance_bias, bool two_step) {
  return MeanStdDevNormalization(definition, gpu_info, shape, variance_bias,
                                 two_step, nullptr, nullptr);
}

MeanStdDevNormalization CreateMeanStdDevNormalization(
    const OperationDef& definition, const GpuInfo& gpu_info, const BHWC& shape,
    float variance_bias, const Tensor<Linear, DataType::FLOAT32>& gamma,
    const Tensor<Linear, DataType::FLOAT32>& beta, bool two_step) {
  return MeanStdDevNormalization(definition, gpu_info, shape, variance_bias,
                                 two_step, &gamma, &beta);
}

MeanStdDevNormalization CreateRMSNormalization(const OperationDef& definition,
                                               const GpuInfo& gpu_info,
                                               const BHWC& shape,
                                               float variance_bias) {
  MeanStdDevNormalization norm;
  norm.work_group_reduction_ = UseWorkGroupReduction(gpu_info, shape);
  if (norm.work_group_reduction_) {
    norm.work_group_size_ = GetRecommendedWorkGroupSize(gpu_info, shape);
  } else {
    norm.work_group_size_ = int3(8, 8, 1);
  }
  norm.args_.AddFloat("variance_bias", variance_bias);
  norm.args_.AddFloat("inv_ch_count", 1.0f / shape.c);
  norm.has_depth_ = definition.src_tensors[0].HasAxis(Axis::DEPTH);
  norm.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  norm.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  norm.code_ = GetRMSNormalizationCode(
      gpu_info, definition.dst_tensors[0].HasAxis(Axis::BATCH),
      shape.c % 4 == 0, norm.work_group_reduction_, norm.work_group_size_,
      norm.has_depth_);
  return norm;
}

std::unique_ptr<GPUOperation> CreateStatisticalTopK(
    const OperationDef& definition, const GpuInfo& gpu_info, const BHWC& shape,
    float stddev_multiplier) {
  MeanStdDevNormalization norm;
  norm.work_group_reduction_ = UseWorkGroupReduction(gpu_info, shape);
  if (norm.work_group_reduction_) {
    norm.work_group_size_ = GetRecommendedWorkGroupSize(gpu_info, shape);
  } else {
    norm.work_group_size_ = int3(8, 8, 1);
  }
  norm.args_.AddFloat("inv_ch_count", 1.0f / shape.c);
  norm.has_depth_ = definition.src_tensors[0].HasAxis(Axis::DEPTH);
  norm.args_.AddFloat("stddev_multiplier", stddev_multiplier);
  norm.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  norm.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  norm.code_ = GetStatisticalTopKCode(
      gpu_info, definition.dst_tensors[0].HasAxis(Axis::BATCH),
      shape.c % 4 == 0, norm.work_group_reduction_, norm.work_group_size_,
      norm.has_depth_);
  return std::make_unique<MeanStdDevNormalization>(std::move(norm));
}

HWCGroupNormalization::HWCGroupNormalization(
    const OperationDef& definition, const GpuInfo& gpu_info, const BHWC& shape,
    int groups, float variance_bias,
    const Tensor<Linear, DataType::FLOAT32>& gamma,
    const Tensor<Linear, DataType::FLOAT32>& beta)
    : groups_(groups) {
  const int group_size = shape.c / groups;
  int max_total_size = 128;
  if (gpu_info.IsAdreno()) {
    max_total_size = 512;
  } else if (gpu_info.IsMali() &&
             gpu_info.mali_info.generation >= MaliInfo::Gen::kValhallV1) {
    max_total_size = 512;
  } else if (gpu_info.IsPowerVR() &&
             gpu_info.powervr_info.gpu_version >= PowerVRGpu::kCXT) {
    max_total_size = 512;
  }
  max_total_size =
      std::min(max_total_size, gpu_info.GetMaxWorkGroupTotalSize());
  int remaining_size = max_total_size;
  work_group_size_ = int3(1, 1, 1);
  while (remaining_size != 1) {
    if (work_group_size_.y < work_group_size_.x &&
        work_group_size_.y * 2 <= gpu_info.GetMaxWorkGroupSizeForY() &&
        work_group_size_.y * 2 <= shape.h) {
      work_group_size_.y *= 2;
      remaining_size /= 2;
    } else if (work_group_size_.x * 2 <= gpu_info.GetMaxWorkGroupSizeForX() &&
               work_group_size_.x * 2 <= shape.w) {
      work_group_size_.x *= 2;
      remaining_size /= 2;
    } else {
      break;
    }
  }
  if (group_size % 4 == 0) {
    const int group_slices = DivideRoundUp(group_size, 4);
    while (remaining_size != 1 &&
           work_group_size_.z * 2 <= gpu_info.GetMaxWorkGroupSizeForZ() &&
           work_group_size_.z * 2 <= group_slices) {
      remaining_size /= 2;
      work_group_size_.z *= 2;
    }
  }
  args_.AddFloat("variance_bias", variance_bias);
  args_.AddInt("group_size_ch", group_size);
  args_.AddFloat("inv_elements_count", 1.0f / (shape.h * shape.w * group_size));
  if (group_size % 4 == 0) {
    args_.AddInt("group_size", DivideRoundUp(group_size, 4));
  } else if (group_size % 2 == 0) {
    args_.AddInt("group_size", DivideRoundUp(group_size * 2, 4));
    groups_ = DivideRoundUp(groups_, 2);
  } else if (group_size == 1) {
    groups_ = DivideRoundUp(groups_, 4);
  }
  AddSrcTensor("src_tensor", definition.src_tensors[0]);
  AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  TensorDescriptor gamma_tensor_desc = CreateConstantLinearTensorDescriptor(
      gpu_info, definition.src_tensors[0].GetDataType(), gamma);
  args_.AddObject("gamma", std::make_unique<TensorDescriptor>(
                               std::move(gamma_tensor_desc)));
  TensorDescriptor beta_tensor_desc = CreateConstantLinearTensorDescriptor(
      gpu_info, definition.src_tensors[0].GetDataType(), beta);
  args_.AddObject(
      "beta", std::make_unique<TensorDescriptor>(std::move(beta_tensor_desc)));
  if (group_size == 1) {
    code_ = GetHWCNormalizationCodeGroupSize1(gpu_info, definition,
                                              work_group_size_);
  } else {
    code_ = GetHWCNormalizationCode(gpu_info, definition, work_group_size_,
                                    group_size);
  }
}

int3 HWCGroupNormalization::GetGridSize() const {
  const int grid_x = work_group_size_.x * dst_[0]->Batch();
  const int grid_y = work_group_size_.y;
  const int grid_z = work_group_size_.z * groups_;
  return int3(grid_x, grid_y, grid_z);
}
}  // namespace ml_drift
