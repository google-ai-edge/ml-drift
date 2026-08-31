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

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {
namespace fc {

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

}  // namespace fc
}  // namespace ml_drift
