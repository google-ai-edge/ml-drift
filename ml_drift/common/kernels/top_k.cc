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

#include "ml_drift/common/kernels/top_k.h"

#include <limits>
#include <string>

#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/task/compiler_options.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/types.h"

namespace ml_drift {
std::string GetAddValueToTopKCode(int top_k_size, bool skip_top) {
  bool kPreserveOrder = true;
  std::string add_value_step;
  if (kPreserveOrder) {
    add_value_step = R"(  if (value_to_add >= $0) {
    int tmp_ind = $1;
    if ($0 == value_to_add) {
      tmp_ind = max(index_to_add, $1);
      index_to_add = min(index_to_add, $1);
    }
    ScalarType tmp_max = $0;
    $0 = value_to_add;
    $1 = index_to_add;
    value_to_add = tmp_max;
    index_to_add = tmp_ind;
  }
)";
  } else {
    add_value_step = R"(  if (value_to_add > $0) {
    ScalarType tmp_max = $0;
    int tmp_ind = $1;
    $0 = value_to_add;
    $1 = index_to_add;
    value_to_add = tmp_max;
    index_to_add = tmp_ind;
  }
)";
  }
  if (skip_top) {
    add_value_step =
        "  if (value_to_add < skip_max || (value_to_add == skip_max && "
        "index_to_add > skip_ind)) {\n" +
        add_value_step + "  }\n";
  }
  std::string add_value_code;
  const std::string kPostfixes[] = {".x", ".y", ".z", ".w"};
  for (int k = 0; k < top_k_size; ++k) {
    const std::string top_k_max =
        "top_k_max_" + std::to_string(k / 4) + kPostfixes[k % 4];
    const std::string top_k_ind =
        "top_k_ind_" + std::to_string(k / 4) + kPostfixes[k % 4];
    add_value_code += absl::Substitute(add_value_step, top_k_max, top_k_ind);
  }
  return add_value_code;
}

std::string GetPerThreadReduceCode(bool has_index_tensor, bool skip_top) {
  std::string code = R"(
  int H = ucl::GetGroupId<1>();
  int local_id = ucl::GetLocalId<0>();
  int start_position = ucl::GetGroupId<0>() * args.reduction_per_group;
  int end_position = min(start_position + args.reduction_per_group, args.src.Width());
)";
  if (skip_top) {
    code += R"(
  ScalarType skip_max = args.top_k_max.Read(0, H, args.k_offset - 1).w;
  int skip_ind = args.top_k_ind.Read(0, H, args.k_offset - 1).w;
)";
  }
  code += R"(
  int4 top_k_ind_0 = ucl::Init<int4>(-1);
  Type top_k_max_0 = ucl::Init<Type>(args.min_value);
  for (int w = start_position + local_id; w < end_position; w += ucl::GetGroupSize<0>()) {
    Type new_value = args.src.Read(w, H, 0);
    int4 new_inds = GET_INDEXES;
    ScalarType value_to_add = new_value.x;
    int index_to_add = new_inds.x;
    ADD_VALUE_TO_TOP_K
    value_to_add = new_value.y;
    index_to_add = new_inds.y;
    ADD_VALUE_TO_TOP_K
    value_to_add = new_value.z;
    index_to_add = new_inds.z;
    ADD_VALUE_TO_TOP_K
    value_to_add = new_value.w;
    index_to_add = new_inds.w;
    ADD_VALUE_TO_TOP_K
  }
)";
  std::string get_indexes_code =
      "ucl::Init<int4>(w * 4 + 0, w * 4 + 1, w * 4 + 2, w * 4 + 3)";
  if (has_index_tensor) {
    get_indexes_code = "args.src_ind.Read(w, H, 0)";
  }
  absl::StrReplaceAll({{"GET_INDEXES", get_indexes_code}}, &code);
  return code;
}

std::string GetTopKCode(const OperationDef& op_def, int top_k_size,
                        bool wg_reduction, bool skip_top) {
  std::string code = "MAIN_FUNCTION($0) {\n";
  const bool has_index_tensor = op_def.src_tensors.size() == 2;
  code += GetPerThreadReduceCode(has_index_tensor, skip_top);
  absl::StrReplaceAll(
      {{"ADD_VALUE_TO_TOP_K", GetAddValueToTopKCode(top_k_size, skip_top)}},
      &code);
  if (!wg_reduction) {
    code += R"(
  args.dst_max.Write(top_k_max_0, ucl::GetGlobalId<0>(), H, 0);
  args.dst_ind.Write(top_k_ind_0, ucl::GetGlobalId<0>(), H, 0);
})";
    return code;
  }
  code += R"(
  __local int4 local_mem_int[WG_SIZE];
  __local Type local_mem_flt[WG_SIZE];
  local_mem_int[local_id] = top_k_ind_0;
  local_mem_flt[local_id] = top_k_max_0;
  ucl::SyncThreads<WorkGroup, Local>();

  int steps[4];
  steps[0] = 4;
  steps[1] = 4;
  steps[2] = 4;
  steps[3] = 4;
  int reduction_size = WG_SIZE;
  for (int i = 0; i < args.steps_count; i++) {
    int step_size = steps[i];
    int active_threads = reduction_size / step_size;
    if (local_id < active_threads) {
      for (int s = local_id + active_threads; s < reduction_size; s += active_threads) {
        int4 new_inds = local_mem_int[s];
        Type new_value = local_mem_flt[s];
        ScalarType value_to_add = new_value.x;
        int index_to_add = new_inds.x;
        ADD_VALUE_TO_TOP_K
        value_to_add = new_value.y;
        index_to_add = new_inds.y;
        ADD_VALUE_TO_TOP_K
        value_to_add = new_value.z;
        index_to_add = new_inds.z;
        ADD_VALUE_TO_TOP_K
        value_to_add = new_value.w;
        index_to_add = new_inds.w;
        ADD_VALUE_TO_TOP_K
      }
    }
    ucl::SyncThreads<WorkGroup, Local>();
    if (local_id < active_threads) {
      local_mem_int[local_id] = top_k_ind_0;
      local_mem_flt[local_id] = top_k_max_0;
    }
    ucl::SyncThreads<WorkGroup, Local>();
    reduction_size = active_threads;
  }

  if (local_id == 0) {
    args.dst_max.Write(top_k_max_0, ucl::GetGroupId<0>(), H, args.k_offset);
    args.dst_ind.Write(top_k_ind_0, ucl::GetGroupId<0>(), H, args.k_offset);
  }
})";
  absl::StrReplaceAll(
      {{"ADD_VALUE_TO_TOP_K", GetAddValueToTopKCode(top_k_size, false)}},
      &code);
  return code;
}

TopKOp CreateTopK(const GpuInfo& gpu_info, const OperationDef& op_def,
                  bool use_wg_reduction, int k_offset) {
  const int kTopKSize = 4;
  int3 kWgSize = int3(256, 1, 1);
  if (gpu_info.IsMali() && gpu_info.mali_info.IsMidgard()) {
    kWgSize = int3(128, 1, 1);
  }
  if (gpu_info.IsAdreno() && gpu_info.adreno_info.IsLowEnd()) {
    kWgSize = int3(64, 1, 1);
  }
  const bool skip_top = k_offset != 0 && (op_def.src_tensors.size() == 1 ||
                                          op_def.src_tensors.size() == 3);
  std::string code = GetTopKCode(op_def, kTopKSize, use_wg_reduction, skip_top);
  const DataType type = op_def.src_tensors[0].GetDataType();
  absl::StrReplaceAll({{"ScalarType", ToUclDataType(type, 1)},
                       {"Type", ToUclDataType(type, 4)},
                       {"WG_SIZE", std::to_string(kWgSize.x)}},
                      &code);

  TopKOp op(kWgSize, use_wg_reduction);
  op.AddSrcTensor("src", op_def.src_tensors[0]);
  if (op_def.src_tensors.size() == 2) {
    op.AddSrcTensor("src_ind", op_def.src_tensors[1]);
  }
  if (skip_top && op_def.src_tensors.size() == 1) {
    // We must reuse destination tensors for our sources
    op.AddSrcTensor("top_k_max", op_def.dst_tensors[0]);
    op.AddSrcTensor("top_k_ind", op_def.dst_tensors[1]);
  } else if (skip_top) {
    // We've been given separate source and destination tensors
    op.AddSrcTensor("top_k_max", op_def.src_tensors[1]);
    op.AddSrcTensor("top_k_ind", op_def.src_tensors[2]);
  }
  op.AddDstTensor("dst_max", op_def.dst_tensors[0]);
  op.AddDstTensor("dst_ind", op_def.dst_tensors[1]);
  const float min_float =
      type == DataType::kFloat32 ? std::numeric_limits<float>::max() : kMaxHalf;
  op.args_.AddFloat("min_value", -min_float, type);
  op.args_.AddInt("steps_count", 4);
  op.args_.AddInt("k_offset", k_offset / 4);
  op.args_.AddInt("reduction_per_group");
  if (gpu_info.IsPowerVR()) {
    // without this, the kernel outputs nans in some cases.
    op.compiler_options_.push_back(CompilerOptions::kClDisableOptimizations);
  }
  op.code_ = code;

  return op;
}
}  // namespace ml_drift
