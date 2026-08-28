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

#include "ml_drift/common/kernels/tests/resize_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/resize.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_ref_ops.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace {
absl::Status ResizeTest(TestExecutionEnvironment& exec_env,
                        const Resize2DAttributes& attr,
                        const TensorFloat32& src_tensor,
                        const OperationDef& op_def) {
  TensorFloat32 dst_ref_tensor = ResizeReference(attr, src_tensor);

  Resize operation = CreateResize(op_def, attr);

  TensorFloat32 dst_tensor;
  ABSL_EXPECT_OK(exec_env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Resize>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  const float eps =
      op_def.src_tensors[0].GetDataType() == DataType::FLOAT32 ? 1e-5f : 0.02f;
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status Resize3DTest(TestExecutionEnvironment& exec_env,
                          const Resize3DAttributes& attr,
                          const Tensor5DFloat32& src_tensor,
                          const OperationDef& op_def) {
  Tensor5DFloat32 dst_ref_tensor = ResizeReference(attr, src_tensor);

  Resize3D operation = CreateResize3D(op_def, attr);

  Tensor5DFloat32 dst_tensor;
  ABSL_EXPECT_OK(exec_env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Resize3D>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  const float eps =
      op_def.src_tensors[0].GetDataType() == DataType::FLOAT32 ? 1e-5f : 0.02f;
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}
}  // namespace

absl::Status ResizeBilinearAlignedTest(TestExecutionEnvironment& env,
                                       DataType data_type,
                                       TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 3, 1);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  Resize2DAttributes attr;
  attr.type = SamplingType::BILINEAR;
  attr.new_shape = HW(4, 4);
  attr.align_corners = true;

  const float eps = data_type == DataType::FLOAT32 ? 1e-5f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  Resize operation = CreateResize(op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Resize>(std::move(operation)),
      BHWC(1, 4, 4, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps),
                        {0.0f, 0.666667f, 1.33333f, 2.0f, 1.0f, 1.66667f,
                         2.33333f, 3.0f, 2.0f, 2.66667f, 3.33333f, 4.0f, 3.0f,
                         3.66667f, 4.33333f, 5.0f}));
  return absl::OkStatus();
}

absl::Status ResizeBilinearNonAlignedTest(TestExecutionEnvironment& env,
                                          DataType data_type,
                                          TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 3, 1);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  Resize2DAttributes attr;
  attr.type = SamplingType::BILINEAR;
  attr.new_shape = HW(4, 4);
  attr.align_corners = false;

  const float eps = data_type == DataType::FLOAT32 ? 1e-5f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  Resize operation = CreateResize(op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Resize>(std::move(operation)),
      BHWC(1, 4, 4, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps),
                        {0.0f, 0.75f, 1.5f, 2.0f, 1.5f, 2.25f, 3.0f, 3.5f, 3.0f,
                         3.75f, 4.5f, 5.0f, 3.0f, 3.75f, 4.5f, 5.0f}));
  return absl::OkStatus();
}

absl::Status ResizeBilinearWithoutHalfPixelTest(TestExecutionEnvironment& env,
                                                DataType data_type,
                                                TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 1);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f};

  Resize2DAttributes attr;
  attr.type = SamplingType::BILINEAR;
  attr.new_shape = HW(3, 3);
  attr.align_corners = false;
  attr.half_pixel_centers = false;

  const float eps = data_type == DataType::FLOAT32 ? 1e-5f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  Resize operation = CreateResize(op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Resize>(std::move(operation)),
      BHWC(1, 3, 3, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {1.0f, 1.666666f, 2.0f, 2.333333f, 3.0f,
                                         3.333333f, 3.0f, 3.666666f, 4.0f}));
  return absl::OkStatus();
}

absl::Status ResizeBilinearWithHalfPixelTest(TestExecutionEnvironment& env,
                                             DataType data_type,
                                             TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 1);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f};

  Resize2DAttributes attr;
  attr.type = SamplingType::BILINEAR;
  attr.new_shape = HW(3, 3);
  attr.align_corners = false;
  attr.half_pixel_centers = true;

  const float eps = data_type == DataType::FLOAT32 ? 1e-5f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  Resize operation = CreateResize(op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Resize>(std::move(operation)),
      BHWC(1, 3, 3, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {1.0f, 1.5f, 2.0f, 2.0f, 2.5f, 3.0f,
                                         3.0f, 3.5f, 4.0f}));
  return absl::OkStatus();
}

absl::Status ResizeNearestTest(TestExecutionEnvironment& env,
                               DataType data_type, TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 2, 1);
  src_tensor.data = {1.0f, 2.0f};

  Resize2DAttributes attr;
  attr.align_corners = false;
  attr.half_pixel_centers = false;
  attr.new_shape = HW(2, 4);
  attr.type = SamplingType::NEAREST;

  const float eps = data_type == DataType::FLOAT32 ? 1e-5f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  Resize operation = CreateResize(op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Resize>(std::move(operation)),
      BHWC(1, 2, 4, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps),
                        {1.0f, 1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f}));
  return absl::OkStatus();
}

absl::Status ResizeNearestAlignCornersTest(TestExecutionEnvironment& env,
                                           DataType data_type,
                                           TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 1);
  src_tensor.data = {3.0f, 6.0f, 9.0f, 12.0f};

  Resize2DAttributes attr;
  attr.align_corners = true;
  attr.half_pixel_centers = false;
  attr.new_shape = HW(3, 3);
  attr.type = SamplingType::NEAREST;

  const float eps = data_type == DataType::FLOAT32 ? 1e-5f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  Resize operation = CreateResize(op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Resize>(std::move(operation)),
      BHWC(1, 3, 3, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {3.0f, 6.0f, 6.0f, 9.0f, 12.0f, 12.0f,
                                         9.0f, 12.0f, 12.0f}));
  return absl::OkStatus();
}

absl::Status ResizeNearestHalfPixelCentersTest(TestExecutionEnvironment& env,
                                               DataType data_type,
                                               TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 1);
  src_tensor.data = {3.0f, 6.0f, 9.0f, 12.0f};

  Resize2DAttributes attr;
  attr.align_corners = false;
  attr.half_pixel_centers = true;
  attr.new_shape = HW(3, 3);
  attr.type = SamplingType::NEAREST;

  const float eps = data_type == DataType::FLOAT32 ? 1e-5f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  Resize operation = CreateResize(op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Resize>(std::move(operation)),
      BHWC(1, 3, 3, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {3.0f, 6.0f, 6.0f, 9.0f, 12.0f, 12.0f,
                                         9.0f, 12.0f, 12.0f}));
  return absl::OkStatus();
}

absl::Status ResizeBilinearAlignedBigTest(TestExecutionEnvironment& env,
                                          DataType data_type,
                                          TensorStorageType storage) {
  Resize2DAttributes attr;
  attr.type = SamplingType::BILINEAR;
  attr.new_shape = HW(7, 9);
  attr.align_corners = true;

  auto src_shape = BHWC(1, 5, 7, 5);
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return ResizeTest(env, attr, src_tensor, op_def);
}

absl::Status ResizeBilinearNonAlignedBigTest(TestExecutionEnvironment& env,
                                             DataType data_type,
                                             TensorStorageType storage) {
  Resize2DAttributes attr;
  attr.type = SamplingType::BILINEAR;
  attr.new_shape = HW(25, 17);
  attr.align_corners = false;

  auto src_shape = BHWC(1, 11, 13, 5);
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return ResizeTest(env, attr, src_tensor, op_def);
}

absl::Status ResizeBilinearAlignedBatchedBigTest(TestExecutionEnvironment& env,
                                                 DataType data_type,
                                                 TensorStorageType storage) {
  Resize2DAttributes attr;
  attr.type = SamplingType::BILINEAR;
  attr.new_shape = HW(7, 9);
  attr.align_corners = true;

  auto src_shape = BHWC(2, 5, 7, 5);
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  return ResizeTest(env, attr, src_tensor, op_def);
}

absl::Status ResizeBilinearNonAlignedBatchedBigTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage) {
  Resize2DAttributes attr;
  attr.type = SamplingType::BILINEAR;
  attr.new_shape = HW(25, 17);
  attr.align_corners = false;

  auto src_shape = BHWC(3, 11, 13, 5);
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  return ResizeTest(env, attr, src_tensor, op_def);
}

absl::Status ResizeNearestBigTest(TestExecutionEnvironment& env,
                                  DataType data_type,
                                  TensorStorageType storage) {
  Resize2DAttributes attr;
  attr.type = SamplingType::NEAREST;
  attr.new_shape = HW(25, 17);
  attr.align_corners = false;

  auto src_shape = BHWC(1, 11, 13, 5);
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return ResizeTest(env, attr, src_tensor, op_def);
}

absl::Status ResizeNearestBatchedBigTest(TestExecutionEnvironment& env,
                                         DataType data_type,
                                         TensorStorageType storage) {
  Resize2DAttributes attr;
  attr.type = SamplingType::NEAREST;
  attr.new_shape = HW(7, 9);
  attr.align_corners = true;

  auto src_shape = BHWC(2, 5, 7, 5);
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  return ResizeTest(env, attr, src_tensor, op_def);
}

absl::Status ResizeBilinear3DAlignedBigTest(TestExecutionEnvironment& env,
                                            DataType data_type,
                                            TensorStorageType storage) {
  Resize3DAttributes attr;
  attr.type = SamplingType::BILINEAR;
  attr.new_shape = HWD(7, 9, 8);
  attr.align_corners = true;

  auto src_shape = BHWDC(1, 5, 7, 5, 5);
  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWDC});
  return Resize3DTest(env, attr, src_tensor, op_def);
}

absl::Status ResizeBilinear3DNonAlignedBigTest(TestExecutionEnvironment& env,
                                               DataType data_type,
                                               TensorStorageType storage) {
  Resize3DAttributes attr;
  attr.type = SamplingType::BILINEAR;
  attr.new_shape = HWD(25, 17, 22);
  attr.align_corners = false;

  auto src_shape = BHWDC(1, 11, 13, 7, 5);
  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWDC});
  return Resize3DTest(env, attr, src_tensor, op_def);
}

absl::Status ResizeBilinear3DAlignedBatchedBigTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage) {
  Resize3DAttributes attr;
  attr.type = SamplingType::BILINEAR;
  attr.new_shape = HWD(7, 9, 10);
  attr.align_corners = true;

  auto src_shape = BHWDC(2, 5, 7, 7, 5);
  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWDC});
  return Resize3DTest(env, attr, src_tensor, op_def);
}

absl::Status ResizeBilinear3DNonAlignedBatchedBigTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage) {
  Resize3DAttributes attr;
  attr.type = SamplingType::BILINEAR;
  attr.new_shape = HWD(25, 17, 19);
  attr.align_corners = false;

  auto src_shape = BHWDC(3, 11, 13, 15, 5);
  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWDC});
  return Resize3DTest(env, attr, src_tensor, op_def);
}

absl::Status ResizeNearest3DBigTest(TestExecutionEnvironment& env,
                                    DataType data_type,
                                    TensorStorageType storage) {
  Resize3DAttributes attr;
  attr.type = SamplingType::NEAREST;
  attr.new_shape = HWD(25, 17, 22);
  attr.align_corners = false;

  auto src_shape = BHWDC(1, 11, 13, 7, 5);
  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWDC});
  return Resize3DTest(env, attr, src_tensor, op_def);
}

absl::Status ResizeNearest3DBatchedBigTest(TestExecutionEnvironment& env,
                                           DataType data_type,
                                           TensorStorageType storage) {
  Resize3DAttributes attr;
  attr.type = SamplingType::NEAREST;
  attr.new_shape = HWD(7, 9, 10);
  attr.align_corners = true;

  auto src_shape = BHWDC(2, 5, 7, 7, 5);
  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWDC});
  return Resize3DTest(env, attr, src_tensor, op_def);
}

}  // namespace ml_drift
