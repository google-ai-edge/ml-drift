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

#include "ml_drift/common/kernels/tests/padding_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/padding.h"
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

absl::Status PaddingAppendWidthTest(TestExecutionEnvironment& env,
                                    DataType data_type,
                                    TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  PadAttributes attr;
  attr.prepended = BHWC(0, 0, 0, 0);
  attr.appended = BHWC(0, 0, 1, 0);

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreatePadding(env.GetGpuInfo(), op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 2, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps),
                        {0.0f, 1.0f, 0.0f, 0.0f, 2.0f, 3.0f, 0.0f, 0.0f}));
  return absl::OkStatus();
}

absl::Status PaddingAppendWidthConstValuesTest(TestExecutionEnvironment& env,
                                               DataType data_type,
                                               TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  PadAttributes attr;
  attr.prepended = BHWC(0, 0, 0, 0);
  attr.appended = BHWC(0, 0, 1, 0);
  attr.constant_values = 5;

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreatePadding(env.GetGpuInfo(), op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 2, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps),
                        {0.0f, 1.0f, 5.0f, 5.0f, 2.0f, 3.0f, 5.0f, 5.0f}));
  return absl::OkStatus();
}

absl::Status PaddingPrependWidthTest(TestExecutionEnvironment& env,
                                     DataType data_type,
                                     TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  PadAttributes attr;
  attr.prepended = BHWC(0, 0, 1, 0);
  attr.appended = BHWC(0, 0, 0, 0);

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreatePadding(env.GetGpuInfo(), op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 2, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps),
                        {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 2.0f, 3.0f}));
  return absl::OkStatus();
}

absl::Status PaddingAppendHeightTest(TestExecutionEnvironment& env,
                                     DataType data_type,
                                     TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  PadAttributes attr;
  attr.prepended = BHWC(0, 0, 0, 0);
  attr.appended = BHWC(0, 1, 0, 0);

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreatePadding(env.GetGpuInfo(), op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 3, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 0.0f}));
  return absl::OkStatus();
}

absl::Status PaddingPrependHeightTest(TestExecutionEnvironment& env,
                                      DataType data_type,
                                      TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  PadAttributes attr;
  attr.prepended = BHWC(0, 1, 0, 0);
  attr.appended = BHWC(0, 0, 0, 0);

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreatePadding(env.GetGpuInfo(), op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 3, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f}));
  return absl::OkStatus();
}

absl::Status PaddingAppendChannelsTest(TestExecutionEnvironment& env,
                                       DataType data_type,
                                       TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  PadAttributes attr;
  attr.prepended = BHWC(0, 0, 0, 0);
  attr.appended = BHWC(0, 0, 0, 1);

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreatePadding(env.GetGpuInfo(), op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 3), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {0.0f, 1.0f, 0.0f, 2.0f, 3.0f, 0.0f}));
  return absl::OkStatus();
}

absl::Status PaddingPrependChannelsTest(TestExecutionEnvironment& env,
                                        DataType data_type,
                                        TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  PadAttributes attr;
  attr.prepended = BHWC(0, 0, 0, 1);
  attr.appended = BHWC(0, 0, 0, 0);

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreatePadding(env.GetGpuInfo(), op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 3), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {0.0f, 0.0f, 1.0f, 0.0f, 2.0f, 3.0f}));
  return absl::OkStatus();
}

absl::Status PaddingPrependChannelsX4Test(TestExecutionEnvironment& env,
                                          DataType data_type,
                                          TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 2);
  src_tensor.data = {1.0f, 2.0f};

  PadAttributes attr;
  attr.prepended = BHWC(0, 0, 0, 4);
  attr.appended = BHWC(0, 0, 0, 0);

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreatePadding(env.GetGpuInfo(), op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 1, 1, 6), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f}));
  return absl::OkStatus();
}

absl::Status PaddingComplexTest(TestExecutionEnvironment& env,
                                DataType data_type, TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  PadAttributes attr;
  attr.prepended = BHWC(0, 0, 1, 1);
  attr.appended = BHWC(0, 1, 1, 0);

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreatePadding(env.GetGpuInfo(), op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 3, 3, 3), &dst_tensor));
  EXPECT_THAT(
      dst_tensor.data,
      Pointwise(FloatNear(eps),
                {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
  return absl::OkStatus();
}

absl::Status PaddingReflectWidthTest(TestExecutionEnvironment& env,
                                     DataType data_type,
                                     TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 3, 1);
  src_tensor.data = {1.0f, 2.0f, 3.0f};

  PadAttributes attr;
  attr.prepended = BHWC(0, 0, 2, 0);
  attr.appended = BHWC(0, 0, 2, 0);
  attr.type = PaddingContentType::REFLECT;

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreatePadding(env.GetGpuInfo(), op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 1, 7, 1), &dst_tensor));
  EXPECT_THAT(
      dst_tensor.data,
      Pointwise(FloatNear(eps), {3.0f, 2.0f, 1.0f, 2.0f, 3.0f, 2.0f, 1.0f}));
  return absl::OkStatus();
}

absl::Status PaddingReflectChannelsTest(TestExecutionEnvironment& env,
                                        DataType data_type,
                                        TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 3);
  src_tensor.data = {1.0f, 2.0f, 3.0f};

  PadAttributes attr;
  attr.prepended = BHWC(0, 0, 0, 2);
  attr.appended = BHWC(0, 0, 0, 2);
  attr.type = PaddingContentType::REFLECT;

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreatePadding(env.GetGpuInfo(), op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 1, 1, 7), &dst_tensor));
  EXPECT_THAT(
      dst_tensor.data,
      Pointwise(FloatNear(eps), {3.0f, 2.0f, 1.0f, 2.0f, 3.0f, 2.0f, 1.0f}));
  return absl::OkStatus();
}

namespace {

absl::Status PaddingTest(TestExecutionEnvironment& exec_env,
                         const PadAttributes& attr,
                         const TensorFloat32& src_tensor,
                         const OperationDef& op_def) {
  TensorFloat32 dst_ref_tensor = PaddingReference(attr, src_tensor);

  GPUOperation operation = CreatePadding(exec_env.GetGpuInfo(), op_def, attr);

  TensorFloat32 dst_tensor;
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.001f), dst_ref_tensor.data));
  return absl::OkStatus();
}
}  // namespace

absl::Status PaddingBigTest(TestExecutionEnvironment& env, DataType data_type,
                            TensorStorageType storage) {
  PadAttributes attr;
  attr.prepended = BHWC(0, 3, 2, 1);
  attr.appended = BHWC(0, 5, 4, 3);

  auto src_shape = BHWC(1, 7, 3, 5);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return PaddingTest(env, attr, src_tensor, op_def);
}

absl::Status PaddingBatchedBigTest(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage) {
  PadAttributes attr;
  attr.prepended = BHWC(4, 3, 2, 1);
  attr.appended = BHWC(0, 5, 4, 3);

  auto src_shape = BHWC(5, 7, 3, 5);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  return PaddingTest(env, attr, src_tensor, op_def);
}

}  // namespace ml_drift
