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

#include "ml_drift/common/kernels/tests/transpose_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/transpose.h"
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

template <DataType T>
absl::Status TransposeIntTest(TestExecutionEnvironment& env,
                              TensorStorageType storage) {
  Tensor<BHWC, T> src;
  src.shape = BHWC(1, 1, 2, 3);
  src.data = {1, 2, -3, -4, 3, 6};

  TransposeAttributes attr;
  attr.perm = BHWC(0, 1, 3, 2);

  Tensor<BHWC, T> ref_tensor;
  ref_tensor.shape = BHWC(1, 1, 3, 2);
  ref_tensor.data = {1, -4, 2, 3, -3, 6};

  OperationDef op_def;
  op_def.src_tensors.push_back({T, storage, Layout::HWC});
  op_def.dst_tensors.push_back({T, storage, Layout::HWC});
  TensorDescriptor src_0, dst;
  src_0 = op_def.src_tensors[0];
  src_0.UploadData(src);
  dst.SetBHWCShape(BHWC(1, 1, 3, 2));
  GPUOperation operation = CreateTranspose(op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0}, {&dst}, std::make_unique<GPUOperation>(std::move(operation))));
  Tensor<BHWC, T> dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

template absl::Status TransposeIntTest<DataType::INT32>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status TransposeIntTest<DataType::INT16>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status TransposeIntTest<DataType::INT8>(
    TestExecutionEnvironment& env, TensorStorageType storage);

template <DataType T>
absl::Status TransposeUintTest(TestExecutionEnvironment& env,
                               TensorStorageType storage) {
  Tensor<BHWC, T> src;
  src.shape = BHWC(1, 1, 2, 3);
  src.data = {1, 2, 3, 4, 5, 6};

  TransposeAttributes attr;
  attr.perm = BHWC(0, 1, 3, 2);

  Tensor<BHWC, T> ref_tensor;
  ref_tensor.shape = BHWC(1, 1, 3, 2);
  ref_tensor.data = {1, 4, 2, 5, 3, 6};

  OperationDef op_def;
  op_def.src_tensors.push_back({T, storage, Layout::HWC});
  op_def.dst_tensors.push_back({T, storage, Layout::HWC});
  TensorDescriptor src_0, dst;
  src_0 = op_def.src_tensors[0];
  src_0.UploadData(src);
  dst.SetBHWCShape(BHWC(1, 1, 3, 2));
  GPUOperation operation = CreateTranspose(op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0}, {&dst}, std::make_unique<GPUOperation>(std::move(operation))));
  Tensor<BHWC, T> dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

template absl::Status TransposeUintTest<DataType::UINT32>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status TransposeUintTest<DataType::UINT16>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status TransposeUintTest<DataType::UINT8>(
    TestExecutionEnvironment& env, TensorStorageType storage);

absl::Status TransposeTest(TestExecutionEnvironment& env, DataType data_type,
                           TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 2, 3);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  TransposeAttributes attr;
  attr.perm = BHWC(0, 1, 3, 2);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateTranspose(op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 1, 3, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.0f), {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f}));
  return absl::OkStatus();
}

absl::Status TransposeTest(TestExecutionEnvironment& exec_env,
                           const TransposeAttributes& attr,
                           const BHWC& src_shape, DataType data_type,
                           TensorStorageType storage) {
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  TensorFloat32 dst_ref_tensor = TransposeReference(attr, src_tensor);

  OperationDef op_def;
  const Layout src_layout = src_shape.b > 1 ? Layout::BHWC : Layout::HWC;
  const Layout dst_layout =
      dst_ref_tensor.shape.b > 1 ? Layout::BHWC : Layout::HWC;
  op_def.src_tensors.push_back({data_type, storage, src_layout});
  op_def.dst_tensors.push_back({data_type, storage, dst_layout});
  GPUOperation operation = CreateTranspose(op_def, attr);

  TensorFloat32 dst_tensor;
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.001f), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status Transpose3DTest(TestExecutionEnvironment& exec_env,
                             const Transpose3DAttributes& attr,
                             const BHWDC& src_shape, DataType data_type,
                             TensorStorageType storage) {
  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  Tensor5DFloat32 dst_ref_tensor = TransposeReference(attr, src_tensor);

  OperationDef op_def;
  const Layout src_layout = src_shape.b > 1 ? Layout::BHWDC : Layout::HWDC;
  const Layout dst_layout =
      dst_ref_tensor.shape.b > 1 ? Layout::BHWDC : Layout::HWDC;
  op_def.src_tensors.push_back({data_type, storage, src_layout});
  op_def.dst_tensors.push_back({data_type, storage, dst_layout});
  GPUOperation operation = CreateTranspose(op_def, attr);

  Tensor5DFloat32 dst_tensor;
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.001f), dst_ref_tensor.data));
  return absl::OkStatus();
}

}  // namespace ml_drift
