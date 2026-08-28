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

#include "ml_drift/common/kernels/tests/gather_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/gather.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

template <DataType T>
absl::Status GatherWidthIntTest(TestExecutionEnvironment& env,
                                TensorStorageType storage) {
  Tensor<BHWC, T> src_tensor;
  src_tensor.shape = BHWC(1, 1, 5, 1);
  src_tensor.data = {0, 1, 2, 3, 4};
  Tensor<BHWC, T> src_indices;
  src_indices.shape = BHWC(1, 1, 1, 9);
  src_indices.data = {1, 2, 3, 0, 1, 4, 2, 3, 1};
  GatherAttributes attr;
  attr.axis = Axis::WIDTH;

  OperationDef op_def;
  op_def.src_tensors.push_back({T, storage, Layout::HWC});
  op_def.src_tensors.push_back({T, storage, Layout::HWC});
  op_def.dst_tensors.push_back({T, storage, Layout::HWC});
  TensorDescriptor src_0, src_1, dst;
  src_0 = op_def.src_tensors[0];
  src_1 = op_def.src_tensors[1];
  src_0.UploadData(src_tensor);
  src_1.UploadData(src_indices);
  dst.SetBHWDCShape(BHWDC(1, 1, 9, 1, 1));
  GPUOperation operation = CreateGather(op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0, &src_1}, {&dst},
      std::make_unique<GPUOperation>(std::move(operation))));
  Tensor<BHWC, T> dst_tensor;
  dst.DownloadData(&dst_tensor);
  Tensor<BHWC, T> exp_tensor;
  exp_tensor.data = {1, 2, 3, 0, 1, 4, 2, 3, 1};
  EXPECT_EQ(dst_tensor.data, exp_tensor.data);
  return absl::OkStatus();
}

template absl::Status GatherWidthIntTest<DataType::INT8>(
  TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status GatherWidthIntTest<DataType::INT16>(
  TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status GatherWidthIntTest<DataType::INT32>(
  TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status GatherWidthIntTest<DataType::UINT8>(
  TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status GatherWidthIntTest<DataType::UINT16>(
  TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status GatherWidthIntTest<DataType::UINT32>(
  TestExecutionEnvironment& env, TensorStorageType storage);

absl::Status GatherTest(TestExecutionEnvironment& env, DataType data_type,
                        TensorStorageType storage, Axis axis) {
  // 2^-9 should be the maximum error for values in range [0.0, 8.0]
  constexpr float kEps = 1.0f / (1 << 9);

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 1);
  src_tensor.shape.set(axis, 5);
  BHWC dst_shape = BHWC(1, 1, 1, 1);
  dst_shape.set(axis, 9);
  src_tensor.data = {1.5f, 2.4f, 3.3f, 4.2f, 5.1f};
  TensorFloat32 src_indices;
  src_indices.shape = BHWC(1, 1, 1, 9);
  src_indices.data = {1.1f, 2.1f, 3.1f, 0.1f, 1.1f, 4.1f, 2.1f, 3.1f, 1.1f};
  GatherAttributes attr;
  attr.axis = axis;

  const Layout src_layout =
      src_tensor.shape.b != 1 ? Layout::BHWC : Layout::HWC;
  const Layout dst_layout = dst_shape.b != 1 ? Layout::BHWC : Layout::HWC;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, src_layout});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, dst_layout});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateGather(op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor, src_indices},
      std::make_unique<GPUOperation>(std::move(operation)), dst_shape,
      &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps), {2.4f, 3.3f, 4.2f, 1.5f, 2.4f, 5.1f,
                                          3.3f, 4.2f, 2.4f}));
  return absl::OkStatus();
}

}  // namespace ml_drift
