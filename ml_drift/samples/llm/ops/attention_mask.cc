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

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "ml_drift/common/op_api.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor_handle.h"

namespace ml_drift {

class AttentionMask : public OpBase<AttentionMask> {
 public:
  // clang-format off
  MLD_OP_ATTRS(
      MLD_OP_ATTR(int, window_size, 0);
      MLD_OP_ATTR(int, sliding_local_cache, 0);
      MLD_OP_ATTR_NO_BIND(int, mask_width, 0);
      MLD_OP_ATTR_NO_BIND(int, token_offset_index, 0);
      MLD_OP_ATTR_NO_BIND(int, batch_size, 1);
      MLD_OP_ATTR_NO_BIND(int, sequence_size, 1);
  )
  // clang-format on

  absl::Status Build(GpuModelBuilder& graph,
                     const std::vector<TensorHandle>& inputs, Attrs& attrs,
                     std::vector<TensorHandle>& outputs) const {
    if (inputs.size() != 1) {
      return absl::InvalidArgumentError("AttentionMask expects 1 input");
    }
    if (attrs.batch_size <= 0 || attrs.sequence_size <= 0 ||
        attrs.mask_width <= 0) {
      return absl::InvalidArgumentError(
          "AttentionMask requires batch_size > 0, sequence_size > 0, and "
          "mask_width > 0");
    }
    if (attrs.window_size < 0) {
      return absl::InvalidArgumentError(
          "AttentionMask requires window_size >= 0");
    }
    if (attrs.sliding_local_cache != 0 && attrs.sliding_local_cache != 1) {
      return absl::InvalidArgumentError(
          "AttentionMask requires sliding_local_cache to be 0 or 1");
    }
    if (attrs.sliding_local_cache == 1 && attrs.window_size <= 0) {
      return absl::InvalidArgumentError(
          "AttentionMask requires window_size > 0 when sliding_local_cache == "
          "1");
    }
    if (attrs.token_offset_index < 0 ||
        attrs.token_offset_index >= inputs[0].tensor_desc.GetBHWCShape().c) {
      return absl::InvalidArgumentError(
          "AttentionMask token_offset_index is out of range for params_tensor");
    }

    outputs = {graph.AddTensor(
        BHWC(attrs.batch_size, 1, attrs.sequence_size, attrs.mask_width),
        DataType::kBool)};

    auto* op = AppendNewOp(graph, attrs);
    AddSrcTensor(graph, op, "params_tensor", inputs[0]);
    AddDstTensor(graph, op, "dst", outputs[0]);

    const auto& dst_shape = outputs[0].tensor_desc.GetBHWCShape();
    op->SetGridSize(
        int3(dst_shape.w * dst_shape.b, dst_shape.h, (dst_shape.c + 3) / 4));
    op->flops_ = dst_shape.DimensionsProduct() * 2;

    op->code_ = absl::Substitute(R"(MAIN_FUNCTION($$0) {
      int linear_xb = ucl::GetGlobalId<0>();
      int X = linear_xb / args.dst.Batch();
      int B = linear_xb % args.dst.Batch();
      args.dst.SetBatchRef(B);
      int Y = ucl::GetGlobalId<1>();
      int S = ucl::GetGlobalId<2>();
      if (X >= args.dst.Width() || Y >= args.dst.Height() || S >= args.dst.Slices()) return;
      int token_offset;
      args.params_tensor.ReadPerChannel(token_offset, 0, 0, $0);
      int effective_offset = args.sliding_local_cache == 1
                                 ? min(token_offset, args.window_size - 1)
                                 : token_offset;

      args.dst::type mask_value;
      int k;
      bool visible;

      // x
      k = S * 4 + 0;
      {
        int token_index = k - effective_offset;
        visible = (token_index <= X);
        if (args.window_size > 0) visible &= (token_index > X - args.window_size);
        if (args.sliding_local_cache == 1) {
          visible &= (token_offset + token_index >= 0);
        }
      }
      mask_value.x = !visible;

      // y
      k = S * 4 + 1;
      {
        int token_index = k - effective_offset;
        visible = (token_index <= X);
        if (args.window_size > 0) visible &= (token_index > X - args.window_size);
        if (args.sliding_local_cache == 1) {
          visible &= (token_offset + token_index >= 0);
        }
      }
      mask_value.y = !visible;

      // z
      k = S * 4 + 2;
      {
        int token_index = k - effective_offset;
        visible = (token_index <= X);
        if (args.window_size > 0) visible &= (token_index > X - args.window_size);
        if (args.sliding_local_cache == 1) {
          visible &= (token_offset + token_index >= 0);
        }
      }
      mask_value.z = !visible;

      // w
      k = S * 4 + 3;
      {
        int token_index = k - effective_offset;
        visible = (token_index <= X);
        if (args.window_size > 0) visible &= (token_index > X - args.window_size);
        if (args.sliding_local_cache == 1) {
          visible &= (token_offset + token_index >= 0);
        }
      }
      mask_value.w = !visible;

      args.dst.Write(mask_value, X, Y, S);
    })",
                                 attrs.token_offset_index);
    return absl::OkStatus();
  }
};

MLD_REGISTER_OP(AttentionMask);

}  // namespace ml_drift
