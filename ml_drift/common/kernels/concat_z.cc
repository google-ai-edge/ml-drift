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

#include "ml_drift/common/kernels/concat_z.h"

#include <string>
#include <vector>

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {
namespace {

struct SliceInfo {
  struct ChannelInfo {
    int src_tensor_index = -1;
    int slice_offset = -1;
    int sub_channel = -1;
  };
  ChannelInfo channel_info[4];
  int first_dst_slice = -1;
  int last_dst_slice = -1;
};

bool EqualChannelInfo(const SliceInfo::ChannelInfo& a,
                      const SliceInfo::ChannelInfo& b) {
  return a.src_tensor_index == b.src_tensor_index &&
         a.sub_channel == b.sub_channel;
}

bool EqualChannelsInfo(const SliceInfo& a, const SliceInfo& b) {
  for (int i = 0; i < 4; ++i) {
    if (!EqualChannelInfo(a.channel_info[i], b.channel_info[i])) {
      return false;
    }
  }
  return true;
}

void AddSliceInfo(std::vector<SliceInfo>& slices_info,
                  const SliceInfo& slice_info, int dst_ch) {
  if (!slices_info.empty() &&
      EqualChannelsInfo(slices_info.back(), slice_info)) {
    slices_info.back().last_dst_slice = dst_ch / 4;
  } else {
    slices_info.push_back(slice_info);
    slices_info.back().first_dst_slice = dst_ch / 4;
    slices_info.back().last_dst_slice = dst_ch / 4;
  }
}

std::string GetConcatKernelCode(const OperationDef& op_def,
                                const std::vector<int>& channels) {
  std::vector<SliceInfo> slices_info;
  SliceInfo slice_info;
  int counter = 0;
  int dst_ch = 0;
  for (int t_id = 0; t_id < channels.size(); ++t_id) {
    for (int src_ch = 0; src_ch < channels[t_id]; ++src_ch, ++dst_ch) {
      slice_info.channel_info[counter++] = {t_id, src_ch / 4, src_ch % 4};
      if (counter == 4) {
        AddSliceInfo(slices_info, slice_info, dst_ch);
        slice_info = SliceInfo();
        counter = 0;
      }
    }
  }
  if (counter != 0) {
    AddSliceInfo(slices_info, slice_info, dst_ch);
  }

  std::vector<std::string> tensor_names(op_def.src_tensors.size());
  for (int i = 0; i < op_def.src_tensors.size(); ++i) {
    tensor_names[i] = "src_tensor_" + std::to_string(i);
  }

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    for (int i = 0; i < op_def.src_tensors.size(); ++i) {
      c += "  args." + tensor_names[i] + ".SetBatchRef(B);\n";
    }
  } else {
    c += "  int X = ucl::GetGlobalId<0>();\n";
  }
  std::string coords = "X, Y";
  if (op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    c += "  int linear_id_1 = ucl::GetGlobalId<1>();\n";
    c += "  int Y = linear_id_1 / args.dst_tensor.Depth();\n";
    c += "  int Z = linear_id_1 % args.dst_tensor.Depth();\n";
    coords = "X, Y, Z";
  } else {
    c += "  int Y = ucl::GetGlobalId<1>();\n";
  }
  c += "  int S = ucl::GetGlobalId<2>();\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) return;\n";

  c += "  args.dst_tensor::type result = args.dst_tensor::zero_value;\n";
  const std::string vec_coords[] = {".x", ".y", ".z", ".w"};
  for (const auto& slice_info : slices_info) {
    c += "  if (S >= " + std::to_string(slice_info.first_dst_slice) +
         " && S <= " + std::to_string(slice_info.last_dst_slice) + ") {\n";
    c += "    int base_s = S - " + std::to_string(slice_info.first_dst_slice) +
         ";\n";
    int prev_tensor_index = -1;
    int prev_slice_offset = -1;
    for (int i = 0; i < 4; ++i) {
      int tensor_index = slice_info.channel_info[i].src_tensor_index;
      int slice_offset = slice_info.channel_info[i].slice_offset;
      if (tensor_index != -1 && (tensor_index != prev_tensor_index ||
                                 slice_offset != prev_slice_offset)) {
        const std::string tensor_name =
            "args.src_tensor_" + std::to_string(tensor_index);
        const std::string var_name = "src_var_" + std::to_string(tensor_index) +
                                     "_" + std::to_string(slice_offset);
        c += "    args.dst_tensor::type " + var_name + " = " + tensor_name +
             ".Read(" + coords + ", base_s + " + std::to_string(slice_offset) +
             ");\n";
      }
      prev_tensor_index = tensor_index;
      prev_slice_offset = slice_offset;
    }
    for (int i = 0; i < 4; ++i) {
      int tensor_index = slice_info.channel_info[i].src_tensor_index;
      int slice_offset = slice_info.channel_info[i].slice_offset;
      if (tensor_index != -1) {
        const std::string var_name = "src_var_" + std::to_string(tensor_index) +
                                     "_" + std::to_string(slice_offset);
        c += "    result" + vec_coords[i] + " = " + var_name +
             vec_coords[slice_info.channel_info[i].sub_channel] + ";\n";
      }
    }
    c += "  }\n";
  }
  c += "  args.dst_tensor.Write(result, " + coords + ", S);\n";
  c += "}\n";
  return c;
}

}  // namespace

GPUOperation CreateConcatZ(const OperationDef& definition,
                           const std::vector<int>& channels,
                           const GpuInfo& gpu_info) {
  GPUOperation op;
  for (int i = 0; i < definition.src_tensors.size(); ++i) {
    const std::string name = "src_tensor_" + std::to_string(i);
    op.AddSrcTensor(name, definition.src_tensors[i]);
  }
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.code_ = GetConcatKernelCode(definition, channels);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace ml_drift
