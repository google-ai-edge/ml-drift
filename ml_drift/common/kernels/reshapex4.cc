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

#include "ml_drift/common/kernels/reshapex4.h"

#include <string>
#include <utility>
#include <vector>

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/work_group_picking.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

std::vector<int3> Reshapex4::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info) const {
  if (tuning_type == TuningType::kFast && src_[0]->Batch() != 1 &&
      src_[0]->Height() == 1 && dst_[0]->Batch() == 1 &&
      dst_[0]->Height() == src_[0]->Batch() && dst_[0]->Width() >= 1024) {
    // B-H swap, B is fastest changing dimension in all physical layouts, it
    // means that we should keep wg_size.y bigger (wg_size.y 'responsible' to
    // write output H or read input B) for better cached reads.
    int3 wg_size = int3(1, 1, 1);
    if (src_[0]->Batch() % 16 == 0 || src_[0]->Batch() >= 32) {
      wg_size.y = 16;
    } else if (src_[0]->Batch() % 8 == 0 || src_[0]->Batch() >= 16) {
      wg_size.y = 8;
    } else if (src_[0]->Batch() % 4 == 0 || src_[0]->Batch() >= 8) {
      wg_size.y = 4;
    } else {
      wg_size.y = 2;
    }
    wg_size.x = 128 / wg_size.y;
    return {wg_size};
  }
  return GetPossibleWorkGroups(tuning_type, gpu_info, kernel_info, grid_size_);
}

Reshapex4 CreateReshapex4(const OperationDef& definition) {
  std::string code;
  if (definition.dst_tensors[0].HasAxis(Axis::BATCH)) {
    code += "  int out_linear = DST_B;\n";
  } else {
    code += "  int out_linear = 0;\n";
  }
  code += R"(
  out_linear = ((out_linear * DST_HEIGHT + DST_Y) * DST_WIDTH + DST_X) * DST_SLICES + DST_S;
  SRC_S = out_linear % SRC_SLICES;
  out_linear = out_linear / SRC_SLICES;
  SRC_X = out_linear % SRC_WIDTH;
  out_linear = out_linear / SRC_WIDTH;
  SRC_Y = out_linear % SRC_HEIGHT;
)";
  if (definition.src_tensors[0].HasAxis(Axis::BATCH)) {
    code += "SRC_B = out_linear / SRC_HEIGHT;\n";
  }

  Reshapex4 op;
  op.SetReorderCode(std::move(code));
  op.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  return op;
}

}  // namespace ml_drift
