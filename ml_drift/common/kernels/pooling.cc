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

#include "ml_drift/common/kernels/pooling.h"

#include <map>
#include <string>
#include <vector>

#include "absl/strings/substitute.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/util.h"

namespace ml_drift {

namespace {
std::string GetAveragePoolingKernelCode(const OperationDef& op_def,
                                        const GpuInfo& gpu_info,
                                        GPUOperation* op) {
  op->AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  op->AddDstTensor("dst_tensor", op_def.dst_tensors[0]);

  std::map<Axis, std::string> axis_to_src_coord = {
      {Axis::kWidth, "x_c"},  {Axis::kHeight, "y_c"}, {Axis::kDepth, "d_c"},
      {Axis::kChannels, "Z"}, {Axis::kBatch, "B"},
  };

  std::map<Axis, std::string> axis_to_dst_coord = {
      {Axis::kWidth, "X"},    {Axis::kHeight, "Y"}, {Axis::kDepth, "D"},
      {Axis::kChannels, "Z"}, {Axis::kBatch, "B"},
  };

  std::vector<std::string> src_coords;
  std::vector<std::string> dst_coords;
  for (auto axis :
       {Axis::kWidth, Axis::kHeight, Axis::kDepth, Axis::kChannels}) {
    if (op_def.dst_tensors[0].HasAxis(axis)) {
      dst_coords.push_back(axis_to_dst_coord[axis]);
    }
    if (op_def.src_tensors[0].HasAxis(axis)) {
      src_coords.push_back(axis_to_src_coord[axis]);
    }
  }
  std::string src_coord = src_coords[0];
  for (int i = 1; i < src_coords.size(); ++i) {
    src_coord += ", " + src_coords[i];
  }
  std::string dst_coord = dst_coords[0];
  for (int i = 1; i < dst_coords.size(); ++i) {
    dst_coord += ", " + dst_coords[i];
  }

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::kBatch)) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = ucl::GetGlobalId<0>();\n";
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::kDepth)) {
    c += "  int linear_id_1 = ucl::GetGlobalId<1>();\n";
    c += "  int Y = linear_id_1 / args.dst_tensor.Depth();\n";
    c += "  int D = linear_id_1 % args.dst_tensor.Depth();\n";
  } else {
    c += "  int Y = ucl::GetGlobalId<1>();\n";
  }
  c += "  int Z = ucl::GetGlobalId<2>();\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "Z >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  float4 r = ucl::Init<float4>(0.0f);\n";
  c += "  float window_size = 0.0;\n";
  c += "  int xs = X * args.stride_x + args.padding_x;\n";
  c += "  int ys = Y * args.stride_y + args.padding_y;\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::kDepth)) {
    c += "  int ds = D * args.stride_z + args.padding_z;\n";
    c += "  for (int kz = 0; kz < args.kernel_size_z; ++kz) {\n";
    c += "    int d_c = ds + kz;\n";
    c += "    if (d_c < 0 || d_c >= args.src_tensor.Depth()) continue;\n";
  }
  c += "  for (int ky = 0; ky < args.kernel_size_y; ++ky) {\n";
  c += "    int y_c = ys + ky;\n";
  c += "    bool outside_y = y_c < 0 || y_c >= args.src_tensor.Height();\n";
  c += "    for (int kx = 0; kx < args.kernel_size_x; ++kx) {\n";
  c += "      int x_c = xs + kx;\n";
  c += "      bool outside = outside_y || x_c < 0 || x_c >= "
       "args.src_tensor.Width();\n";
  if (op_def.src_tensors[0].SupportsZeroClamp(Axis::kWidth, gpu_info) &&
      op_def.src_tensors[0].SupportsZeroClamp(Axis::kHeight, gpu_info)) {
    c += "      r += args.src_tensor.Read<float>(" + src_coord + ");\n";
  } else {
    c += "     r += !outside ? args.src_tensor.Read<float>(" + src_coord +
         ") : "
         "ucl::Init<float4>(0.0f);\n";
  }
  c += "        window_size += !outside ? 1.0 : 0.0;\n";
  c += "    }\n";
  c += "  }\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::kDepth)) {
    c += "  }  // Depth\n";
  }
  // If window_size==0, window covered nothing. This situation is a sign of
  // incorrectly constructed operation. NaNs are expected as output.
  c += "  args.dst_tensor::type result = ucl::Convert<args.dst_tensor::type>(r "
       "/ window_size);\n";
  c += "  args.dst_tensor.Write(result, " + dst_coord + ");\n";
  c += "}\n";

  return c;
}

std::string GetMaxPoolingKernelCode(const OperationDef& op_def,
                                    const GpuInfo& gpu_info,
                                    bool output_indices, GPUOperation* op) {
  op->AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  op->AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  if (output_indices) {
    op->AddDstTensor("dst_indices", op_def.dst_tensors[1]);
  }

  std::map<Axis, std::string> axis_to_src_coord = {
      {Axis::kWidth, "x_c"},  {Axis::kHeight, "y_c"}, {Axis::kDepth, "d_c"},
      {Axis::kChannels, "Z"}, {Axis::kBatch, "B"},
  };

  std::map<Axis, std::string> axis_to_dst_coord = {
      {Axis::kWidth, "X"},    {Axis::kHeight, "Y"}, {Axis::kDepth, "D"},
      {Axis::kChannels, "Z"}, {Axis::kBatch, "B"},
  };

  std::vector<std::string> src_coords;
  std::vector<std::string> dst_coords;
  for (auto axis :
       {Axis::kWidth, Axis::kHeight, Axis::kDepth, Axis::kChannels}) {
    if (op_def.dst_tensors[0].HasAxis(axis)) {
      dst_coords.push_back(axis_to_dst_coord[axis]);
    }
    if (op_def.src_tensors[0].HasAxis(axis)) {
      src_coords.push_back(axis_to_src_coord[axis]);
    }
  }
  std::string src_coord = src_coords[0];
  for (int i = 1; i < src_coords.size(); ++i) {
    src_coord += ", " + src_coords[i];
  }
  std::string dst_coord = dst_coords[0];
  for (int i = 1; i < dst_coords.size(); ++i) {
    dst_coord += ", " + dst_coords[i];
  }

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::kBatch)) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
    if (output_indices) {
      c += "  args.dst_indices.SetBatchRef(B);\n";
    }
  } else {
    c += "  int X = ucl::GetGlobalId<0>();\n";
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::kDepth)) {
    c += "  int linear_id_1 = ucl::GetGlobalId<1>();\n";
    c += "  int Y = linear_id_1 / args.dst_tensor.Depth();\n";
    c += "  int D = linear_id_1 % args.dst_tensor.Depth();\n";
  } else {
    c += "  int Y = ucl::GetGlobalId<1>();\n";
  }
  c += "  int Z = ucl::GetGlobalId<2>();\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "Z >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  args.src_tensor::type maximum = "
       "ucl::Init<args.src_tensor::type>(-10000.0f);\n";
  if (output_indices) {
    c += "  int4 indexes = ucl::Init<int4>(0, 0, 0, 0);\n";
  }
  c += "  int xs = X * args.stride_x + args.padding_x;\n";
  c += "  int ys = Y * args.stride_y + args.padding_y;\n";
  c += "  for (int ky = 0; ky < args.kernel_size_y; ++ky) {\n";
  c += "    int y_c = ys + ky;\n";
  c += "    if (y_c < 0 || y_c >= args.src_tensor.Height()) continue;\n";
  c += "    for (int kx = 0; kx < args.kernel_size_x; ++kx) {\n";
  c += "      int x_c = xs + kx;\n";
  c += "      if (x_c < 0 || x_c >= args.src_tensor.Width()) continue;\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::kDepth)) {
    c += "    int ds = D * args.stride_z + args.padding_z;\n";
    c += "    for (int kz = 0; kz < args.kernel_size_z; ++kz) {\n";
    c += "    int d_c = ds + kz;\n";
    c += "      if (d_c < 0 || d_c >= args.src_tensor.Depth()) continue;\n";
  }
  c += "      args.src_tensor::type src = args.src_tensor.Read(" + src_coord +
       ");\n";
  if (output_indices) {
    if (op_def.dst_tensors[0].HasAxis(Axis::kDepth)) {
      c += "      int index_counter = (ky * args.kernel_size_x + kx) * "
           "args.kernel_size_z + kz;\n";
    } else {
      c += "      int index_counter = ky * args.kernel_size_x + kx;\n";
    }
    c += "      if (src.x > maximum.x) {\n";
    c += "        indexes.x = index_counter;\n";
    c += "        maximum.x = src.x;\n";
    c += "      }\n";
    c += "      if (src.y > maximum.y) {\n";
    c += "        indexes.y = index_counter;\n";
    c += "        maximum.y = src.y;\n";
    c += "      }\n";
    c += "      if (src.z > maximum.z) {\n";
    c += "        indexes.z = index_counter;\n";
    c += "        maximum.z = src.z;\n";
    c += "      }\n";
    c += "      if (src.w > maximum.w) {\n";
    c += "        indexes.w = index_counter;\n";
    c += "        maximum.w = src.w;\n";
    c += "      }\n";
  } else {
    c += "      maximum = max(src, maximum);\n";
  }
  if (op_def.dst_tensors[0].HasAxis(Axis::kDepth)) {
    c += "    }  // Depth\n";
  }
  c += "    }\n";
  c += "  }\n";
  c += "  args.dst_tensor.Write(maximum, " + dst_coord + ");\n";
  if (output_indices) {
    c += "  args.dst_indices::type indexes_final_type = "
         "ucl::Convert<args.dst_indices::type>(indexes);\n";
    c += "  args.dst_indices.Write(indexes_final_type, " + dst_coord + ");\n";
  }
  c += "}\n";

  return c;
}
}  // namespace

GPUOperation CreatePooling(const OperationDef& definition,
                           const GpuInfo& gpu_info,
                           const Pooling2DAttributes& attr) {
  GPUOperation op;
  op.args_.AddInt("kernel_size_x", attr.kernel.w);
  op.args_.AddInt("padding_x", -attr.padding.prepended.w);
  op.args_.AddInt("stride_x", attr.strides.w);
  op.args_.AddInt("kernel_size_y", attr.kernel.h);
  op.args_.AddInt("padding_y", -attr.padding.prepended.h);
  op.args_.AddInt("stride_y", attr.strides.h);
  if (attr.type == PoolingType::kAverage) {
    op.code_ = GetAveragePoolingKernelCode(definition, gpu_info, &op);
  } else if (attr.type == PoolingType::kMax) {
    op.code_ =
        GetMaxPoolingKernelCode(definition, gpu_info, attr.output_indices, &op);
  }
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

GPUOperation CreatePooling(const OperationDef& definition,
                           const GpuInfo& gpu_info,
                           const Pooling3DAttributes& attr) {
  GPUOperation op;
  op.args_.AddInt("kernel_size_x", attr.kernel.w);
  op.args_.AddInt("padding_x", -attr.padding.prepended.w);
  op.args_.AddInt("stride_x", attr.strides.w);
  op.args_.AddInt("kernel_size_y", attr.kernel.h);
  op.args_.AddInt("padding_y", -attr.padding.prepended.h);
  op.args_.AddInt("stride_y", attr.strides.h);
  op.args_.AddInt("kernel_size_z", attr.kernel.d);
  op.args_.AddInt("padding_z", -attr.padding.prepended.d);
  op.args_.AddInt("stride_z", attr.strides.d);
  if (attr.type == PoolingType::kAverage) {
    op.code_ = GetAveragePoolingKernelCode(definition, gpu_info, &op);
  } else if (attr.type == PoolingType::kMax) {
    op.code_ =
        GetMaxPoolingKernelCode(definition, gpu_info, attr.output_indices, &op);
  }
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

}  // namespace ml_drift
