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

#include "ml_drift/common/kernels/strided_slice.h"

#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

namespace {
bool Is4Aligned(const Slice3DAttributes& attr) {
  return attr.strides.c == 1 && attr.starts.c % 4 == 0;
}

int4 GetOffset(const Slice3DAttributes& attr, int src_width, int src_height,
               int src_channels, int src_batch) {
  int4 offset;
  if (attr.strides.w > 0) {
    offset.x = attr.starts.w;
  } else {
    if (attr.ends.w > 0) {
      offset.x = attr.ends.w;
    } else {
      offset.x = src_width + attr.ends.w;
    }
  }
  if (attr.strides.h > 0) {
    offset.y = attr.starts.h;
  } else {
    if (attr.ends.h > 0) {
      offset.y = attr.ends.h;
    } else {
      offset.y = src_height + attr.ends.h;
    }
  }
  if (attr.strides.c > 0) {
    offset.z = attr.starts.c;
  } else {
    if (attr.ends.c > 0) {
      offset.z = attr.ends.c;
    } else {
      offset.z = src_channels + attr.ends.c;
    }
  }
  if (Is4Aligned(attr)) {
    offset.z /= 4;
  }
  if (attr.strides.b > 0) {
    offset.w = attr.starts.b;
  } else {
    if (attr.ends.b > 0) {
      offset.w = attr.ends.b;
    } else {
      offset.w = src_batch + attr.ends.b;
    }
  }
  return offset;
}

Slice3DAttributes ConvertTo3D(const SliceAttributes& attr) {
  Slice3DAttributes attr3d;
  attr3d.starts =
      BHWDC(attr.starts.b, attr.starts.h, attr.starts.w, 0, attr.starts.c);
  attr3d.ends = BHWDC(attr.ends.b, attr.ends.h, attr.ends.w, 1, attr.ends.c);
  attr3d.strides =
      BHWDC(attr.strides.b, attr.strides.h, attr.strides.w, 1, attr.strides.c);
  return attr3d;
}

}  // namespace

StridedSlice::StridedSlice(const OperationDef& definition,
                           const SliceAttributes& attr)
    : attributes_(ConvertTo3D(attr)) {
  work_group_size_ = int3(8, 4, 1);
  code_ = GetStridedSliceCode(definition, Is4Aligned(attributes_));
  tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
}

StridedSlice::StridedSlice(const OperationDef& definition,
                           const Slice3DAttributes& attr)
    : attributes_(attr) {
  work_group_size_ = int3(8, 4, 1);
  code_ = GetStridedSliceCode(definition, Is4Aligned(attributes_));
  tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
}

StridedSlice::StridedSlice(StridedSlice&& operation)
    : GPUOperation(std::move(operation)), attributes_(operation.attributes_) {}

StridedSlice& StridedSlice::operator=(StridedSlice&& operation) {
  if (this != &operation) {
    attributes_ = operation.attributes_;
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

std::string StridedSlice::GetStridedSliceCode(const OperationDef& op_def,
                                              bool alignedx4) {
  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  args_.AddInt("offset_x");
  args_.AddInt("offset_y");
  args_.AddInt("offset_z");
  args_.AddInt("offset_d");
  args_.AddInt("offset_b");
  args_.AddInt("stride_x");
  args_.AddInt("stride_y");
  args_.AddInt("stride_z");
  args_.AddInt("stride_d");
  args_.AddInt("stride_b");

  const std::string batch_id =
      op_def.dst_tensors[0].HasAxis(Axis::kBatch) ? "B" : "0";
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::kBatch)) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = ucl::GetGlobalId<0>();\n";
  }

  std::string coords = "X, Y";
  if (op_def.dst_tensors[0].HasAxis(Axis::kDepth)) {
    c += "  int linear_y = ucl::GetGlobalId<1>();\n";
    c += "  int Y = linear_y / args.dst_tensor.Depth();\n";
    c += "  int Z = linear_y % args.dst_tensor.Depth();\n";
    coords += ", Z";
  } else {
    c += "  int Y = ucl::GetGlobalId<1>();\n";
  }
  c += "  int S = ucl::GetGlobalId<2>();\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  coords += ", S";

  c += "  int s_x = X * args.stride_x + args.offset_x;\n";
  c += "  int s_y = Y * args.stride_y + args.offset_y;\n";

  std::string read_coords = "s_x, s_y";
  if (op_def.src_tensors[0].HasAxis(Axis::kDepth)) {
    if (op_def.dst_tensors[0].HasAxis(Axis::kDepth)) {
      c += "  int s_d = Z * args.stride_d + args.offset_d;\n";
    } else {
      c += "  int s_d = args.offset_d;\n";
    }
    read_coords += ", s_d";
  }

  if (op_def.src_tensors[0].HasAxis(Axis::kBatch)) {
    c += "  int s_b = " + batch_id + " * args.stride_b + args.offset_b;\n";
    c += "  args.src_tensor.SetBatchRef(s_b);\n";
  }

  if (alignedx4) {
    c += "  int s_z = S + args.offset_z;\n";
    read_coords += ", s_z";
    c += "  args.dst_tensor::type result = args.src_tensor.Read(" +
         read_coords + ");\n";
  } else {
    c += "  args.dst_tensor::type result;\n";
    const std::string postfixes[] = {"x", "y", "z", "w"};
    for (int i = 0; i < 4; ++i) {
      c += "  {\n";
      const std::string channel = "(S * 4 + " + std::to_string(i) + ")";
      c += "    int s_ch = min(" + channel +
           " * args.stride_z + args.offset_z, args.src_tensor.Channels() - "
           "1);\n";
      c += "    args.src_tensor.ReadPerChannel(result." + postfixes[i] + ", " +
           read_coords + ", s_ch);\n";
      c += "  }\n";
    }
  }
  c += "  args.dst_tensor.Write(result, " + coords + ");\n";
  c += "}\n";
  return c;
}

absl::Status StridedSlice::BindArguments(ArgumentsBinder* args) {
  int4 offset = GetOffset(attributes_, src_[0]->Width(), src_[0]->Height(),
                          src_[0]->Channels(), src_[0]->Batch());
  int offset_d = 0;
  if (attributes_.strides.d > 0) {
    offset_d = attributes_.starts.d;
  } else {
    if (attributes_.ends.d > 0) {
      offset_d = attributes_.ends.d;
    } else {
      offset_d = src_[0]->Depth() + attributes_.ends.d;
    }
  }

  ABSL_RETURN_IF_ERROR(args->SetInt("offset_x", offset.x));
  ABSL_RETURN_IF_ERROR(args->SetInt("offset_y", offset.y));
  ABSL_RETURN_IF_ERROR(args->SetInt("offset_z", offset.z));
  ABSL_RETURN_IF_ERROR(args->SetInt("offset_d", offset_d));
  ABSL_RETURN_IF_ERROR(args->SetInt("offset_b", offset.w));
  ABSL_RETURN_IF_ERROR(args->SetInt("stride_x", attributes_.strides.w));
  ABSL_RETURN_IF_ERROR(args->SetInt("stride_y", attributes_.strides.h));
  ABSL_RETURN_IF_ERROR(args->SetInt("stride_z", attributes_.strides.c));
  ABSL_RETURN_IF_ERROR(args->SetInt("stride_d", attributes_.strides.d));
  ABSL_RETURN_IF_ERROR(args->SetInt("stride_b", attributes_.strides.b));
  return absl::OkStatus();
}

int3 StridedSlice::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height() * dst_[0]->Depth();
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

StridedSlice CreateStridedSlice(const OperationDef& definition,
                                const SliceAttributes& attr) {
  return StridedSlice(definition, attr);
}

StridedSlice CreateStridedSlice(const OperationDef& definition,
                                const Slice3DAttributes& attr) {
  return StridedSlice(definition, attr);
}

}  // namespace ml_drift
