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

#include "ml_drift/common/kernels/dot_general.h"

#include <map>
#include <string>
#include <vector>

#include "absl/log/absl_log.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {

namespace {
std::string GetDotGeneralCode(const OperationDef& op_def,
                              const DotGeneralAttributes& attr) {
  // Our first goal is to get the ReadPerChannel coords for each of the two
  // src tensors. First define some maps
  std::vector<std::string> lhs_read_coords = {"0", "0", "0", "0"};
  std::vector<std::string> rhs_read_coords = {"0", "0", "0", "0"};
  std::map<Axis, int> axis_to_idx = {
      {Axis::BATCH, 0},
      {Axis::HEIGHT, 1},
      {Axis::WIDTH, 2},
      {Axis::CHANNELS, 3},
  };
  std::map<int, std::string> idx_to_coord_str = {
      {0, "Batch"},
      {1, "Height"},
      {2, "Width"},
      {3, "Channels"},
  };
  // Note that lhs_batch_axes.size() == rhs_batch_axes.size()
  const int num_output_dims = attr.lhs_batch_axes.size() +
                              attr.lhs_resulting_axes.size() +
                              attr.rhs_resulting_axes.size();
  // Given some number of output dimensions, we need to be able to map each
  // output dimension to an axis. But, according to ExtractTensorShape, these
  // are not placed in succession. For example, if we have 2 output dims, the
  // shape will be (B, 1, 1, C) instead of (B, H, 1, 1).
  std::map<int, std::string> idx_to_output_coord;
  switch (num_output_dims) {
    case 1:
      idx_to_output_coord[0] = "B";
      break;
    case 2:
      idx_to_output_coord[0] = "B";
      // C + i for computing all channels w/i slice
      idx_to_output_coord[1] = "C + i";
      break;
    case 3:
      idx_to_output_coord[0] = "B";
      idx_to_output_coord[1] = "X";
      idx_to_output_coord[2] = "C + i";
      break;
    case 4:
      idx_to_output_coord[0] = "B";
      idx_to_output_coord[1] = "Y";
      idx_to_output_coord[2] = "X";
      idx_to_output_coord[3] = "C + i";
      break;
    default:
      ABSL_LOG(ERROR) << "Unsupported number of output dimensions: "
                      << num_output_dims;
      break;
  }
  // Output sh = batch_axes + lhs_resulting_axes + rhs_resulting_axes
  // Therefore we can step through the output dimensions and assign the
  // corresponding coords to the lhs and rhs read coords.
  int dim_counter = 0;  // Represents the output dimension
  for (int idx = 0; idx < attr.lhs_batch_axes.size(); ++idx) {
    int lhs_read_idx = axis_to_idx[attr.lhs_batch_axes[idx]];
    int rhs_read_idx = axis_to_idx[attr.rhs_batch_axes[idx]];
    lhs_read_coords[lhs_read_idx] = idx_to_output_coord[dim_counter];
    rhs_read_coords[rhs_read_idx] = idx_to_output_coord[dim_counter];
    ++dim_counter;
  }
  for (int idx = 0; idx < attr.lhs_resulting_axes.size(); ++idx) {
    int lhs_read_idx = axis_to_idx[attr.lhs_resulting_axes[idx]];
    lhs_read_coords[lhs_read_idx] = idx_to_output_coord[dim_counter];
    ++dim_counter;
  }
  for (int idx = 0; idx < attr.rhs_resulting_axes.size(); ++idx) {
    int rhs_read_idx = axis_to_idx[attr.rhs_resulting_axes[idx]];
    rhs_read_coords[rhs_read_idx] = idx_to_output_coord[dim_counter];
    ++dim_counter;
  }
  // For each contracting axis, we need to add a loop over the contracting
  // index. We also need to add the contracting index to the read coords.
  std::vector<int> lhs_contracting_indices;
  for (int idx = 0; idx < attr.lhs_contracting_axes.size(); ++idx) {
    int lhs_read_idx = axis_to_idx[attr.lhs_contracting_axes[idx]];
    int rhs_read_idx = axis_to_idx[attr.rhs_contracting_axes[idx]];
    lhs_read_coords[lhs_read_idx] = "contract" + std::to_string(idx);
    rhs_read_coords[rhs_read_idx] = "contract" + std::to_string(idx);
    // Arbitrarily use lhs to keep track of contracting indices
    lhs_contracting_indices.push_back(lhs_read_idx);
  }

  // Read is order WHCB. Therefore we combine w/ order [2, 1, 3, 0]
  std::string lhs_read_str = "";
  std::string rhs_read_str = "";
  std::vector<int> read_order = {2, 1, 3, 0};
  bool increment_i = false;  // Whether or not to increment i in the dot loop
  for (int idx = 0; idx < 4; ++idx) {
    lhs_read_str += lhs_read_coords[read_order[idx]];
    rhs_read_str += rhs_read_coords[read_order[idx]];
    if (lhs_read_coords[idx] == "C + i" || rhs_read_coords[idx] == "C + i") {
      increment_i = true;
    }
    if (idx < 3) {
      lhs_read_str += ", ";
      rhs_read_str += ", ";
    }
  }

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  c += "  int linear_id = ucl::GetGlobalId<0>();\n";
  c += "  int X = linear_id / args.dst_tensor.Batch();\n";
  c += "  int B = linear_id % args.dst_tensor.Batch();\n";
  c += "  args.lhs.SetBatchRef(B);\n";
  c += "  args.rhs.SetBatchRef(B);\n";
  c += "  args.dst_tensor.SetBatchRef(B);\n";

  c += "  int Y = ucl::GetGlobalId<1>();\n";
  c += "  int S = ucl::GetGlobalId<2>();\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  int C = S * 4;\n";
  c += "  args.lhs::scalar_type lhs;\n";
  c += "  args.rhs::scalar_type rhs;\n";
  c += "  args.lhs::type accum = ucl::Init<args.lhs::type>(0);\n";
  // Set up for loops over contracting indices
  for (int i = 0; i < lhs_contracting_indices.size(); ++i) {
    const std::string ctr_idx_str = "contract" + std::to_string(i);
    c += "  for (int " + ctr_idx_str + " = 0; " + ctr_idx_str + " < args.lhs." +
         idx_to_coord_str[lhs_contracting_indices[i]] + "(); ++" + ctr_idx_str +
         ") {\n";
  }

  // Dot product logic
  const std::vector<std::string> postfixes = {"x", "y", "z", "w"};
  if (increment_i) {
    c += "    int i = 0;\n";
  }
  for (int j = 0; j < 4; ++j) {  // Iterate over channels w/i a slice
    c += "    args.lhs.ReadPerChannel(lhs, " + lhs_read_str + ");\n";
    c += "    args.rhs.ReadPerChannel(rhs, " + rhs_read_str + ");\n";
    c += "    accum." + postfixes[j] + " += lhs * rhs;\n";
    if (!increment_i) {  // Output channel is 1.
      break;
    }
    if (j != 3) {
      c += "    i += 1;\n";
    }
  }

  // Add closing braces for contracting loops
  for (int i = 0; i < lhs_contracting_indices.size(); ++i) {
    c += "  }\n";
  }
  c += "  args.dst_tensor.Write(accum, X, Y, S, B);\n";
  c += "}\n";
  return c;
}
}  // namespace

GPUOperation CreateDotGeneral(const OperationDef& op_def,
                              const DotGeneralAttributes& attr) {
  GPUOperation op;
  op.AddSrcTensor("lhs", op_def.src_tensors[0]);
  op.AddSrcTensor("rhs", op_def.src_tensors[1]);
  op.AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  op.code_ = GetDotGeneralCode(op_def, attr);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace ml_drift
