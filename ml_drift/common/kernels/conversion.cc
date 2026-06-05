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

#include "ml_drift/common/kernels/conversion.h"

#include <string>

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"

namespace ml_drift {

TensorToTensor CreateTensorToTensorOp(const GpuInfo& gpu_info,
                                      const TensorDescriptor& src_desc,
                                      const TensorDescriptor& dst_desc) {
  TensorToTensor op;
  op.AddSrcTensor("src_tensor", src_desc);
  op.AddDstTensor("dst_tensor", dst_desc);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  op.code_ +=
      R"(MAIN_FUNCTION($0) {
  int linear_id = ucl::GetGlobalId<0>();
  int x = linear_id / args.dst_tensor.Batch();
  int b = linear_id % args.dst_tensor.Batch();
  int y = ucl::GetGlobalId<1>();
  int d = ucl::GetGlobalId<2>();
  if (x >= args.dst_tensor.Width() || y >= args.dst_tensor.Height() || d >= args.dst_tensor.Slices()) return;
  args.src_tensor::type in_value = args.src_tensor.Read(x, y, d, b);
  args.dst_tensor::type out_value = ucl::Convert<args.dst_tensor::type>(in_value);
  args.dst_tensor.Write(out_value, x, y, d, b);
}
)";
  return op;
}

TensorToBhwcBuffer CreateTensorToBhwcBufferOp(
    const GpuInfo& gpu_info, const TensorDescriptor& src_desc,
    const BufferDescriptor& dst_desc) {
  TensorToBhwcBuffer op;
  op.AddSrcTensor("tensor", src_desc);
  op.AddDstBuffer("buffer", dst_desc);

  op.code_ += R"(MAIN_FUNCTION($0) {
  int linear_id = ucl::GetGlobalId<0>();
  int x = linear_id / args.tensor.Batch();
  int b = linear_id % args.tensor.Batch();
  int y = ucl::GetGlobalId<1>();
  int d = ucl::GetGlobalId<2>();
  if (x >= args.tensor.Width() || y >= args.tensor.Height() || d >= args.tensor.Slices()) return;
  args.tensor::type in_value = args.tensor.Read(x, y, d, b);
  args.buffer::scalar_type out_x = ucl::Convert<args.buffer::scalar_type>(in_value.x);
  args.buffer::scalar_type out_y = ucl::Convert<args.buffer::scalar_type>(in_value.y);
  args.buffer::scalar_type out_z = ucl::Convert<args.buffer::scalar_type>(in_value.z);
  args.buffer::scalar_type out_w = ucl::Convert<args.buffer::scalar_type>(in_value.w);
  int c = d * 4;
  int index = ((b * args.tensor.Height() + y) * args.tensor.Width() + x) * args.tensor.Channels() + c;

  args.buffer.Write(out_x, index);
  if (c + 1 < args.tensor.Channels()) {
    args.buffer.Write(out_y, index + 1);
  }
  if (c + 2 < args.tensor.Channels()) {
    args.buffer.Write(out_z, index + 2);
  }
  if (c + 3 < args.tensor.Channels()) {
    args.buffer.Write(out_w, index + 3);
  }
})";
  return op;
}

TensorToBhwcBuffer CreateTensorToBhwcBufferAlignedOp(
    const GpuInfo& gpu_info, const TensorDescriptor& src_desc,
    const BufferDescriptor& dst_desc) {
  TensorToBhwcBuffer op;
  op.is_aligned_ = true;
  op.AddSrcTensor("tensor", src_desc);
  TensorDescriptor dst_tensor_desc(dst_desc.element_type,
                                   TensorStorageType::BUFFER, Layout::LINEAR);
  op.AddDstTensor("buffer", dst_tensor_desc);

  op.code_ += R"(MAIN_FUNCTION($0) {
  int linear_id = ucl::GetGlobalId<0>();
  int total_size = args.tensor.Batch() * args.tensor.Height() * args.tensor.Width() * args.tensor.Channels();
  int total_size_x4 = (total_size + 3) / 4;
  if (linear_id >= total_size_x4) {
    return;
  }
  args.tensor::type result = args.tensor::zero_value;
  {
    int bhwc = linear_id * 4 + 0;
    int c = bhwc % args.tensor.Channels();
    int rem = bhwc / args.tensor.Channels();
    int w = rem % args.tensor.Width();
    rem = rem / args.tensor.Width();
    int h = rem % args.tensor.Height();
    int b = rem / args.tensor.Height();
    if (w < args.tensor.Width() && h < args.tensor.Height() && b < args.tensor.Batch() && c < args.tensor.Channels()) {
      args.tensor.ReadPerChannel(result.x, w, h, c, b);
    }
  }
  {
    int bhwc = linear_id * 4 + 1;
    int c = bhwc % args.tensor.Channels();
    int rem = bhwc / args.tensor.Channels();
    int w = rem % args.tensor.Width();
    rem = rem / args.tensor.Width();
    int h = rem % args.tensor.Height();
    int b = rem / args.tensor.Height();
    if (w < args.tensor.Width() && h < args.tensor.Height() && b < args.tensor.Batch() && c < args.tensor.Channels()) {
      args.tensor.ReadPerChannel(result.y, w, h, c, b);
    }
  }
  {
    int bhwc = linear_id * 4 + 2;
    int c = bhwc % args.tensor.Channels();
    int rem = bhwc / args.tensor.Channels();
    int w = rem % args.tensor.Width();
    rem = rem / args.tensor.Width();
    int h = rem % args.tensor.Height();
    int b = rem / args.tensor.Height();
    if (w < args.tensor.Width() && h < args.tensor.Height() && b < args.tensor.Batch() && c < args.tensor.Channels()) {
      args.tensor.ReadPerChannel(result.z, w, h, c, b);
    }
  }
  {
    int bhwc = linear_id * 4 + 3;
    int c = bhwc % args.tensor.Channels();
    int rem = bhwc / args.tensor.Channels();
    int w = rem % args.tensor.Width();
    rem = rem / args.tensor.Width();
    int h = rem % args.tensor.Height();
    int b = rem / args.tensor.Height();
    if (w < args.tensor.Width() && h < args.tensor.Height() && b < args.tensor.Batch() && c < args.tensor.Channels()) {
      args.tensor.ReadPerChannel(result.w, w, h, c, b);
    }
  }
  args.buffer.WriteLinear(ucl::Convert<args.buffer::type>(result), linear_id);
})";
  return op;
}

BhwcBufferToTensor CreateBhwcBufferToTensorOp(
    const GpuInfo& gpu_info, const BufferDescriptor& src_desc,
    const TensorDescriptor& dst_desc) {
  BhwcBufferToTensor op;
  op.AddSrcBuffer("buffer", src_desc);
  op.AddDstTensor("tensor", dst_desc);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  op.code_ += R"(MAIN_FUNCTION($0) {
  int linear_id = ucl::GetGlobalId<0>();
  int x = linear_id / args.tensor.Batch();
  int b = linear_id % args.tensor.Batch();
  int y = ucl::GetGlobalId<1>();
  int d = ucl::GetGlobalId<2>();

  if (x >= args.tensor.Width() || y >= args.tensor.Height() || d >= args.tensor.Slices()) return;
  int c = d * 4;
  int index = ((b * args.tensor.Height() + y) * args.tensor.Width() + x) * args.tensor.Channels() + c;
  args.tensor::type result = args.tensor::zero_value;
  result.x = ucl::Convert<args.tensor::scalar_type>(args.buffer.Read(index));
  if (c + 1 < args.tensor.Channels()) {
    result.y = ucl::Convert<args.tensor::scalar_type>(args.buffer.Read(index + 1));
  }
  if (c + 2 < args.tensor.Channels()) {
    result.z = ucl::Convert<args.tensor::scalar_type>(args.buffer.Read(index + 2));
  }
  if (c + 3 < args.tensor.Channels()) {
    result.w = ucl::Convert<args.tensor::scalar_type>(args.buffer.Read(index + 3));
  }
  args.tensor.Write(result, x, y, d, b);
})";
  return op;
}

BhwcBufferToTensor CreateBhwcBufferAlignedToTensorOp(
    const GpuInfo& gpu_info, const BufferDescriptor& src_desc,
    const TensorDescriptor& dst_desc) {
  BhwcBufferToTensor op;
  TensorDescriptor src_tensor_desc(src_desc.element_type,
                                   TensorStorageType::BUFFER, Layout::LINEAR);
  op.AddSrcTensor("buffer", src_tensor_desc);
  op.AddDstTensor("tensor", dst_desc);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  op.code_ += R"(MAIN_FUNCTION($0) {
  int linear_id = ucl::GetGlobalId<0>();
  int x = linear_id / args.tensor.Batch();
  int b = linear_id % args.tensor.Batch();
  int y = ucl::GetGlobalId<1>();
  int d = ucl::GetGlobalId<2>();

  if (x >= args.tensor.Width() || y >= args.tensor.Height() || d >= args.tensor.Slices()) return;
  int c = d * 4;
  int index = ((b * args.tensor.Height() + y) * args.tensor.Width() + x) * args.tensor.Channels() + c;
  args.buffer::type result = args.buffer::zero_value;
  args.buffer.ReadPerChannel(result.x, index);
  if (c + 1 < args.tensor.Channels()) {
    args.buffer.ReadPerChannel(result.y, index + 1);
  }
  if (c + 2 < args.tensor.Channels()) {
    args.buffer.ReadPerChannel(result.z, index + 2);
  }
  if (c + 3 < args.tensor.Channels()) {
    args.buffer.ReadPerChannel(result.w, index + 3);
  }
  args.tensor.Write(ucl::Convert<args.tensor::type>(result), x, y, d, b);
})";
  return op;
}

}  // namespace ml_drift
