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

#include "ml_drift/common/kernels/dynamic_update_slice.h"

#include <string>
#include <vector>

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {

namespace {

std::string GetDynamicUpdateSliceCode(const OperationDef& op_def) {
  std::string c = R"(MAIN_FUNCTION($0) {
  int linear_xb = ucl::GetGlobalId<0>();
  int DST_X = linear_xb / args.dst_tensor.Batch();
  int DST_B = linear_xb % args.dst_tensor.Batch();
  int DST_Y = ucl::GetGlobalId<1>();
  int DST_S = ucl::GetGlobalId<2>();

  if (DST_X >= args.dst_tensor.Width() ||
      DST_Y >= args.dst_tensor.Height() ||
      DST_S >= args.dst_tensor.Slices()) {
    return;
  }
  int4 start_position = ucl::Convert<int4>(args.start_indices.Read(0, 0, 0));
  // 4d: offset -> bhwc, start_position -> bhwc - {x,y,z,w}
  int offset_b = start_position.x;
  int offset_h = start_position.y;
  int offset_w = start_position.z;
  int offset_c = start_position.w;
  if (args.start_indices.Channels() == 3) {
    // 3d: offset -> 0hwc, start_position -> hwc - {x,y,z}
    offset_b = 0;
    offset_h = start_position.x;
    offset_w = start_position.y;
    offset_c = start_position.z;
  }
  if (args.start_indices.Channels() == 2) {
    // 2d: offset -> 00wc, start_position -> wc - {x,y}
    offset_b = 0;
    offset_h = 0;
    offset_w = start_position.x;
    offset_c = start_position.y;
  }
  if (args.start_indices.Channels() == 1) {
    if (args.start_indices.Width() == 4) {
      offset_b = ucl::Convert<int>(args.start_indices.Read(0, 0, 0).x);
      offset_h = ucl::Convert<int>(args.start_indices.Read(1, 0, 0).x);
      offset_w = ucl::Convert<int>(args.start_indices.Read(2, 0, 0).x);
      offset_c = ucl::Convert<int>(args.start_indices.Read(3, 0, 0).x);
    } else {
      // 1d: offset -> 000c, start_position -> c - {x}
      offset_b = 0;
      offset_h = 0;
      offset_w = 0;
      offset_c = start_position.x;
    }
  }

  if (offset_b + args.update_slice.Batch() > args.dst_tensor.Batch()) {
    offset_b = args.dst_tensor.Batch() - args.update_slice.Batch();
  }
  if (offset_w + args.update_slice.Width() > args.dst_tensor.Width()) {
    offset_w = args.dst_tensor.Width() - args.update_slice.Width();
  }
  if (offset_h + args.update_slice.Height() > args.dst_tensor.Height()) {
    offset_h = args.dst_tensor.Height() - args.update_slice.Height();
  }
  if (offset_c + args.update_slice.Channels() > args.dst_tensor.Channels()) {
    offset_c = args.dst_tensor.Channels() - args.update_slice.Channels();
  }
  // Slice offset to indicate which slice to update.
  int offset_s = offset_c / 4;

  args.src_tensor::type src_val = args.src_tensor.Read(DST_X, DST_Y, DST_S, DST_B);
  args.update_slice::scalar_type update_vals_array[4];
  update_vals_array[0] = ucl::Convert<args.update_slice::scalar_type>(src_val.x);
  update_vals_array[1] = ucl::Convert<args.update_slice::scalar_type>(src_val.y);
  update_vals_array[2] = ucl::Convert<args.update_slice::scalar_type>(src_val.z);
  update_vals_array[3] = ucl::Convert<args.update_slice::scalar_type>(src_val.w);

  int update_b = DST_B - offset_b;
  int update_x = DST_X - offset_w;
  int update_y = DST_Y - offset_h;
  if (update_b >= 0 && update_b < args.update_slice.Batch() &&
      update_x >= 0 && update_x < args.update_slice.Width() &&
      update_y >= 0 && update_y < args.update_slice.Height()) {
    for (int i = 0; i < 4; i++) {
      int dst_ch = DST_S * 4 + i;
      int update_ch = dst_ch - offset_c;
      if (update_ch >= 0 && update_ch < args.update_slice.Channels()) {
        args.update_slice.ReadPerChannel(update_vals_array[i],
                                         update_x, update_y, update_ch, update_b);
      }
    }
  }
  args.dst_tensor::type update_val;
  update_val.x = ucl::Convert<args.dst_tensor::scalar_type>(update_vals_array[0]);
  update_val.y = ucl::Convert<args.dst_tensor::scalar_type>(update_vals_array[1]);
  update_val.z = ucl::Convert<args.dst_tensor::scalar_type>(update_vals_array[2]);
  update_val.w = ucl::Convert<args.dst_tensor::scalar_type>(update_vals_array[3]);
  args.dst_tensor.Write(update_val, DST_X, DST_Y, DST_S, DST_B);
}
)";

  return c;
}

}  // namespace

GPUOperation CreateDynamicUpdateSlice(const OperationDef& op_def,
                                      const GpuInfo& gpu_info) {
  GPUOperation op;
  op.code_ = GetDynamicUpdateSliceCode(op_def);
  op.AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  op.AddSrcTensor("update_slice", op_def.src_tensors[1]);
  op.AddSrcTensor("start_indices", op_def.src_tensors[2]);
  op.AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace ml_drift
