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

#include "ml_drift/common/kernels/special/experts_remap.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class ExpertsRemapOp : public GPUOperation {
 public:
  ExpertsRemapOp() = default;

  int3 GetGridSize() const override {
    return int3(work_group_size_.x, dst_[0]->Height(), 1);
  }

  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }

  // Move only
  ExpertsRemapOp(ExpertsRemapOp&& operation) = default;
  ExpertsRemapOp& operator=(ExpertsRemapOp&& operation) = default;
  ExpertsRemapOp(const ExpertsRemapOp&) = delete;
  ExpertsRemapOp& operator=(const ExpertsRemapOp&) = delete;
};

std::unique_ptr<GPUOperation> CreateExpertsRemapOp(
    const TensorDescriptor& src, const TensorDescriptor& dst) {
  ExpertsRemapOp op;
  op.AddSrcTensor("src", src);
  op.AddDstTensor("dst", dst);
  BufferDescriptor dst_count;
  dst_count.element_type = DataType::kInt32;
  dst_count.element_size = 1;
  op.AddDstBuffer("dst_count", dst_count);
  op.work_group_size_ = {256, 1, 1};

  op.code_ = R"(
MAIN_FUNCTION($0) {
  int local_id = ucl::GetLocalId<0>();
  int expert_id = ucl::GetGroupId<1>();
  if (expert_id >= args.dst.Height()) {
    return;
  }
  int seq_size = args.src.Width();
  int w_groups = (seq_size + WG_SIZE - 1) / WG_SIZE;
  __local int offsets[WG_SIZE];
  int wg_offset = 0;
  for (int w_group = 0; w_group < w_groups; w_group += 1) {
    int expert_index = -1;
    int w = w_group * WG_SIZE + local_id;
    if (w < seq_size) {
    for (int s = 0; s < args.src.Slices(); ++s) {
      int4 expert_ids = args.src.Read(w, 0, s);
      if (s * 4 < args.src.Channels() && expert_ids.x == expert_id) {
        expert_index = s * 4;
        break;
      }
      if (s * 4 + 1 < args.src.Channels() && expert_ids.y == expert_id) {
        expert_index = s * 4 + 1;
        break;
      }
      if (s * 4 + 2 < args.src.Channels() && expert_ids.z == expert_id) {
        expert_index = s * 4 + 2;
        break;
      }
      if (s * 4 + 3 < args.src.Channels() && expert_ids.w == expert_id) {
        expert_index = s * 4 + 3;
        break;
      }
    }
    }
    // parallel prefix sum to compute offsets
    offsets[local_id] = expert_index == -1 ? 0 : 1;
    ucl::SyncThreads<WorkGroup, Local>();
    for (int offset = 1; offset < WG_SIZE; offset *= 2) {
      int new_offset = offsets[local_id];
      if (local_id >= offset) {
        new_offset += offsets[local_id - offset];
      }
      ucl::SyncThreads<WorkGroup, Local>();
      offsets[local_id] = new_offset;
      ucl::SyncThreads<WorkGroup, Local>();
    }
    if (expert_index != -1) {
      int4 res_value;
      res_value.x = expert_index;
      res_value.y = w;
      int dst_w_offset = wg_offset + offsets[local_id] - 1;
      args.dst.Write(res_value, dst_w_offset, expert_id, 0);
    }
    wg_offset += offsets[WG_SIZE - 1];
  }
  if (local_id == 0) {
    args.dst_count.Write(wg_offset, expert_id);
  }
}
)";
  op.code_ = absl::StrReplaceAll(
      op.code_, {{"WG_SIZE", absl::StrCat(op.work_group_size_.x)}});
  return std::make_unique<ExpertsRemapOp>(std::move(op));
}

class OffsetsOp : public GPUOperation {
 public:
  OffsetsOp() = default;

  int3 GetGridSize() const override { return int3(work_group_size_.x, 1, 1); }

  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }

  absl::Status BindArguments(ArgumentsBinder* args) override {
    ABSL_RETURN_IF_ERROR(args->SetInt("size", dst_[0]->Channels()));
    return absl::OkStatus();
  }

  // Move only
  OffsetsOp(OffsetsOp&& operation) = default;
  OffsetsOp& operator=(OffsetsOp&& operation) = default;
  OffsetsOp(const OffsetsOp&) = delete;
  OffsetsOp& operator=(const OffsetsOp&) = delete;
};

std::unique_ptr<GPUOperation> CreateOffsetsOp() {
  OffsetsOp op;
  BufferDescriptor src_count;
  src_count.element_type = DataType::kInt32;
  src_count.element_size = 1;
  op.AddSrcBuffer("src_count", src_count);
  BufferDescriptor dst_offset;
  dst_offset.element_type = DataType::kInt32;
  dst_offset.element_size = 1;
  op.AddDstBuffer("dst_offset", dst_offset);
  op.args_.AddInt("size", 0);
  op.work_group_size_ = {256, 1, 1};

  op.code_ = R"(
MAIN_FUNCTION($0) {
  int local_id = ucl::GetLocalId<0>();
  int groups = (args.size + WG_SIZE - 1) / WG_SIZE;
  __local int offsets[WG_SIZE];
  int wg_offset = 0;
  for (int group = 0; group < groups; group += 1) {
    int index = group * WG_SIZE + local_id;
    // parallel prefix sum to compute offsets
    int value = 0;
    if (index < args.size) {
      value = args.src_count.Read(index);
    }
    offsets[local_id] = value;
    ucl::SyncThreads<WorkGroup, Local>();
    for (int offset = 1; offset < WG_SIZE; offset *= 2) {
      int new_offset = offsets[local_id];
      if (local_id >= offset) {
        new_offset += offsets[local_id - offset];
      }
      ucl::SyncThreads<WorkGroup, Local>();
      offsets[local_id] = new_offset;
      ucl::SyncThreads<WorkGroup, Local>();
    }
    if (index < args.size) {
      int offset = wg_offset + offsets[local_id] - value;
      args.dst_offset.Write(offset, index);
    }
    wg_offset += offsets[WG_SIZE - 1];
  }
}
)";
  op.code_ = absl::StrReplaceAll(
      op.code_, {{"WG_SIZE", absl::StrCat(op.work_group_size_.x)}});
  return std::make_unique<OffsetsOp>(std::move(op));
}

class LinearizeMapOp : public GPUOperation {
 public:
  LinearizeMapOp() = default;

  int3 GetGridSize() const override {
    return int3(work_group_size_.x, src_[0]->Height(), 1);
  }

  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }

  // Move only
  LinearizeMapOp(LinearizeMapOp&& operation) = default;
  LinearizeMapOp& operator=(LinearizeMapOp&& operation) = default;
  LinearizeMapOp(const LinearizeMapOp&) = delete;
  LinearizeMapOp& operator=(const LinearizeMapOp&) = delete;
};

std::unique_ptr<GPUOperation> CreateLinearizeMapOp(
    const TensorDescriptor& src_map, const TensorDescriptor& dst_map) {
  LinearizeMapOp op;
  op.AddSrcTensor("src", src_map);
  BufferDescriptor dst_count;
  dst_count.element_type = DataType::kInt32;
  dst_count.element_size = 1;
  op.AddSrcBuffer("counts", dst_count);
  op.AddSrcBuffer("offsets", dst_count);
  op.AddDstTensor("dst", dst_map);
  op.work_group_size_ = {256, 1, 1};

  op.code_ = R"(
MAIN_FUNCTION($0) {
  int local_id = ucl::GetLocalId<0>();
  int expert_id = ucl::GetGroupId<1>();
  int experts_count = args.counts.Read(expert_id);
  int experts_offset = args.offsets.Read(expert_id);
  int groups = (experts_count + WG_SIZE - 1) / WG_SIZE;
  for (int group = 0; group < groups; group += 1) {
    int index = group * WG_SIZE + local_id;
    if (index < experts_count) {
      int4 value = args.src.Read(index, expert_id, 0);
      args.dst.Write(value, experts_offset + index, 0, 0);
    }
  }
}
)";
  op.code_ = absl::StrReplaceAll(
      op.code_, {{"WG_SIZE", absl::StrCat(op.work_group_size_.x)}});
  return std::make_unique<LinearizeMapOp>(std::move(op));
}

std::unique_ptr<GPUOperation> CreateExpertsRemapToOp(
    const TensorDescriptor& src, const TensorDescriptor& experts_packed_map,
    const TensorDescriptor& dst) {
  GPUOperation op;
  op.AddSrcTensor("src", src);
  op.AddSrcTensor("experts_packed_map", experts_packed_map);
  op.AddDstTensor("dst", dst);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  op.code_ = R"(
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.dst.Width() || Y >= args.dst.Height() || S >= args.dst.Slices()) {
    return;
  }

  int4 coords = args.experts_packed_map.Read(X, 0, 0);
  int seq_id = coords.y;
  int expert_id = min(coords.x, args.src.Height() - 1);
  args.src::type val = args.src.Read(seq_id, expert_id, S);
  args.dst.Write(val, X, 0, S);
}
)";
  return std::make_unique<GPUOperation>(std::move(op));
}

std::unique_ptr<GPUOperation> CreateExpertsRemapFromOp(
    const TensorDescriptor& src, const TensorDescriptor& experts_packed_map,
    const TensorDescriptor& dst) {
  class RemapFromOp : public GPUOperation {
   public:
    RemapFromOp() = default;
    int3 GetGridSize() const override {
      return int3(src_[0]->Width(), src_[0]->Height(), src_[0]->Slices());
    }

    // Move only
    RemapFromOp(RemapFromOp&& operation) = default;
    RemapFromOp& operator=(RemapFromOp&& operation) = default;
    RemapFromOp(const RemapFromOp&) = delete;
    RemapFromOp& operator=(const RemapFromOp&) = delete;
  };

  RemapFromOp op;
  op.AddSrcTensor("src", src);
  op.AddSrcTensor("experts_packed_map", experts_packed_map);
  op.AddDstTensor("dst", dst);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  op.code_ = R"(
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.src.Width() || Y >= args.src.Height() || S >= args.src.Slices()) {
    return;
  }

  int4 coords = args.experts_packed_map.Read(X, 0, 0);
  int seq_id = coords.y;
  int expert_id = coords.x;
  args.src::type val = args.src.Read(X, 0, S);
  args.dst.Write(val, seq_id, expert_id, S);
}
)";
  return std::make_unique<RemapFromOp>(std::move(op));
}

}  // namespace ml_drift
