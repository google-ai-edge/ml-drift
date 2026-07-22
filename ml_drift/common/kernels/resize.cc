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

#include "ml_drift/common/kernels/resize.h"

#include <string>
#include <utility>

#include "absl/strings/substitute.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

Resize::Resize(const OperationDef& definition, const Resize2DAttributes& attr)
    : attr_(attr) {
  code_ = GetResizeCode(definition, attr_);
}

Resize::Resize(Resize&& operation)
    : GPUOperation(std::move(operation)), attr_(operation.attr_) {}

Resize& Resize::operator=(Resize&& operation) {
  if (this != &operation) {
    attr_ = operation.attr_;
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

std::string GetResize2dCode(const Resize2DAttributes& attr,
                            const std::string& src_tensor_name,
                            const std::string& x_coord,
                            const std::string& y_coord,
                            const std::string& s_coord,
                            const std::string& result_name) {
  std::string c;
  c += "  {\n";
  c += "  float f_coords_x = ucl::Convert<float>($1);\n";
  c += "  float f_coords_y = ucl::Convert<float>($2);\n";
  if (attr.half_pixel_centers) {
    c += "  f_coords_x += 0.5f;\n";
    c += "  f_coords_y += 0.5f;\n";
  }
  c += "  f_coords_x *= args.scale_factor_x;\n";
  c += "  f_coords_y *= args.scale_factor_y;\n";
  if (attr.type == SamplingType::NEAREST) {
    if (attr.align_corners) {
      c += "  f_coords_x += 0.5f;\n";
      c += "  f_coords_y += 0.5f;\n";
    }
    c += "  args.$0.ReadNearest($4, f_coords_x, f_coords_y, $3);\n";
  } else {
    if (attr.half_pixel_centers) {
      c += "  f_coords_x -= 0.5f;\n";
      c += "  f_coords_y -= 0.5f;\n";
    }
    c += "  args.$0.ReadBilinear($4, f_coords_x, f_coords_y, $3);\n";
  }
  c += "  }\n";
  return absl::Substitute(c, src_tensor_name, x_coord, y_coord, s_coord,
                          result_name);
}

std::string Resize::GetResizeCode(const OperationDef& op_def,
                                  const Resize2DAttributes& attr) {
  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  args_.AddFloat("scale_factor_x");
  args_.AddFloat("scale_factor_y");

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = ucl::GetGlobalId<0>();\n";
  }
  c += "  int Y = ucl::GetGlobalId<1>();\n";
  c += "  int Z = ucl::GetGlobalId<2>();\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() "
       "|| Z >= args.dst_tensor.Slices()) return;\n";
  c += "  args.dst_tensor::type r0;\n";
  c += GetResize2dCode(attr, "src_tensor", "X", "Y", "Z", "r0");
  c += "  args.dst_tensor.Write(r0, X, Y, Z);\n";
  c += "}\n";
  return c;
}

absl::Status Resize::BindArguments(ArgumentsBinder* args) {
  ABSL_RETURN_IF_ERROR(args->SetFloat(
      "scale_factor_x",
      CalculateResizeScale(src_[0]->Width(), dst_[0]->Width(), attr_)));
  ABSL_RETURN_IF_ERROR(args->SetFloat(
      "scale_factor_y",
      CalculateResizeScale(src_[0]->Height(), dst_[0]->Height(), attr_)));
  return absl::OkStatus();
}

int3 Resize::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

Resize CreateResize(const OperationDef& definition,
                    const Resize2DAttributes& attr) {
  return Resize(definition, attr);
}

Resize3D::Resize3D(const OperationDef& definition,
                   const Resize3DAttributes& attr)
    : attr_(attr) {
  code_ = GetResize3DCode(definition, attr_);
}

Resize3D::Resize3D(Resize3D&& operation)
    : GPUOperation(std::move(operation)), attr_(operation.attr_) {}

Resize3D& Resize3D::operator=(Resize3D&& operation) {
  if (this != &operation) {
    attr_ = operation.attr_;
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

std::string Resize3D::GetResize3DCode(const OperationDef& op_def,
                                      const Resize3DAttributes& attr) {
  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  args_.AddFloat("scale_factor_x");
  args_.AddFloat("scale_factor_y");
  args_.AddFloat("scale_factor_z");

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = ucl::GetGlobalId<0>();\n";
  }
  c += "  int Y = ucl::GetGlobalId<1>();\n";
  c += "  int linear_id_z = ucl::GetGlobalId<2>();\n";
  c += "  int S = linear_id_z % args.dst_tensor.Slices();\n";
  c += "  int Z = linear_id_z / args.dst_tensor.Slices();\n";
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() "
       "|| Z >= args.dst_tensor.Depth()) return;\n";
  if (attr.half_pixel_centers) {
    c += "  float f_coords_x = (ucl::Convert<float>(X) + 0.5f) * "
         "args.scale_factor_x;\n";
    c += "  float f_coords_y = (ucl::Convert<float>(Y) + 0.5f) * "
         "args.scale_factor_y;\n";
    c += "  float f_coords_z = (ucl::Convert<float>(Z) + 0.5f) * "
         "args.scale_factor_z;\n";
  } else {
    c += "  float f_coords_x = ucl::Convert<float>(X) * args.scale_factor_x;\n";
    c += "  float f_coords_y = ucl::Convert<float>(Y) * args.scale_factor_y;\n";
    c += "  float f_coords_z = ucl::Convert<float>(Z) * args.scale_factor_z;\n";
  }
  c += "  args.dst_tensor::type r0;\n";
  if (attr.type == SamplingType::NEAREST) {
    if (attr.align_corners) {
      c += "  f_coords_x += 0.5f;";
      c += "  f_coords_y += 0.5f;";
      c += "  f_coords_z += 0.5f;";
    }
    c += "  args.src_tensor.ReadNearest(r0, f_coords_x, f_coords_y, "
         "f_coords_z, S);\n";
  } else {
    if (attr.half_pixel_centers) {
      c += "  f_coords_x -= 0.5f;";
      c += "  f_coords_y -= 0.5f;";
      c += "  f_coords_z -= 0.5f;";
    }
    c += "  args.src_tensor.ReadBilinear(r0, f_coords_x, f_coords_y, "
         "f_coords_z, S);\n";
  }
  c += "  args.dst_tensor.Write(r0, X, Y, Z, S);\n";
  c += "}\n";
  return c;
}

absl::Status Resize3D::BindArguments(ArgumentsBinder* args) {
  ABSL_RETURN_IF_ERROR(args->SetFloat(
      "scale_factor_x",
      CalculateResizeScale(src_[0]->Width(), dst_[0]->Width(), attr_)));
  ABSL_RETURN_IF_ERROR(args->SetFloat(
      "scale_factor_y",
      CalculateResizeScale(src_[0]->Height(), dst_[0]->Height(), attr_)));
  ABSL_RETURN_IF_ERROR(args->SetFloat(
      "scale_factor_z",
      CalculateResizeScale(src_[0]->Depth(), dst_[0]->Depth(), attr_)));
  return absl::OkStatus();
}

int3 Resize3D::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height();
  const int grid_z = dst_[0]->Slices() * dst_[0]->Depth();
  return int3(grid_x, grid_y, grid_z);
}

Resize3D CreateResize3D(const OperationDef& definition,
                        const Resize3DAttributes& attr) {
  return Resize3D(definition, attr);
}

}  // namespace ml_drift
