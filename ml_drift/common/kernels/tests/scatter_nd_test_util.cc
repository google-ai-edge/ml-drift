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

#include "ml_drift/common/kernels/tests/scatter_nd_test_util.h"

#include <memory>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/scatter_nd.h"
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
absl::Status ScatterNdIntTest(TestExecutionEnvironment& env,
                              TensorStorageType storage) {
  // Two index vectors (index depth 3) addressing distinct output cells.
  // Coordinate order is (batch, height, width): vector 0 -> (b0, h0, w1),
  // vector 1 -> (b0, h1, w0).
  Tensor<BHWC, T> src_indices;
  src_indices.shape = BHWC(1, 1, 2, 3);
  src_indices.data = {0, 0, 1, 0, 1, 0};
  Tensor<BHWC, T> src_updates;
  src_updates.shape = BHWC(1, 1, 2, 1);
  src_updates.data = {5, 7};

  OperationDef op_def;
  op_def.src_tensors.push_back({T, storage, Layout::HWC});  // indices
  op_def.src_tensors.push_back({T, storage, Layout::HWC});  // updates
  op_def.dst_tensors.push_back({T, storage, Layout::HWC});
  TensorDescriptor indices_desc, updates_desc, dst;
  indices_desc = op_def.src_tensors[0];
  updates_desc = op_def.src_tensors[1];
  indices_desc.UploadData(src_indices);
  updates_desc.UploadData(src_updates);
  dst.SetBHWDCShape(BHWDC(1, 2, 2, 1, 1));
  GPUOperation operation = CreateScatterNd(op_def);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&indices_desc, &updates_desc}, {&dst},
      std::make_unique<GPUOperation>(std::move(operation))));
  Tensor<BHWC, T> dst_tensor;
  dst.DownloadData(&dst_tensor);
  // Output layout is (b, h, w, c): {h0w0, h0w1, h1w0, h1w1}.
  Tensor<BHWC, T> exp_tensor;
  exp_tensor.data = {0, 5, 7, 0};
  EXPECT_EQ(dst_tensor.data, exp_tensor.data);
  return absl::OkStatus();
}

template absl::Status ScatterNdIntTest<DataType::INT32>(
    TestExecutionEnvironment& env, TensorStorageType storage);

template <DataType T>
absl::Status ScatterNdDuplicateIntTest(TestExecutionEnvironment& env,
                                       TensorStorageType storage) {
  // Two index vectors addressing the SAME output cell (b0, h0, w1). Integer
  // updates are assigned, so the last matching index (iteration order) wins.
  Tensor<BHWC, T> src_indices;
  src_indices.shape = BHWC(1, 1, 2, 3);
  src_indices.data = {0, 0, 1, 0, 0, 1};
  Tensor<BHWC, T> src_updates;
  src_updates.shape = BHWC(1, 1, 2, 1);
  src_updates.data = {3, 9};

  OperationDef op_def;
  op_def.src_tensors.push_back({T, storage, Layout::HWC});  // indices
  op_def.src_tensors.push_back({T, storage, Layout::HWC});  // updates
  op_def.dst_tensors.push_back({T, storage, Layout::HWC});
  TensorDescriptor indices_desc, updates_desc, dst;
  indices_desc = op_def.src_tensors[0];
  updates_desc = op_def.src_tensors[1];
  indices_desc.UploadData(src_indices);
  updates_desc.UploadData(src_updates);
  dst.SetBHWDCShape(BHWDC(1, 1, 2, 1, 1));
  GPUOperation operation = CreateScatterNd(op_def);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&indices_desc, &updates_desc}, {&dst},
      std::make_unique<GPUOperation>(std::move(operation))));
  Tensor<BHWC, T> dst_tensor;
  dst.DownloadData(&dst_tensor);
  Tensor<BHWC, T> exp_tensor;
  exp_tensor.data = {0, 9};
  EXPECT_EQ(dst_tensor.data, exp_tensor.data);
  return absl::OkStatus();
}

template absl::Status ScatterNdDuplicateIntTest<DataType::INT32>(
    TestExecutionEnvironment& env, TensorStorageType storage);

absl::Status ScatterNdSumFloatTest(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage) {
  constexpr float kEps = 1.0f / (1 << 9);

  // Two index vectors addressing the SAME output cell (b0, h0, w1). Float
  // updates accumulate (2.5 + 4.0 = 6.5).
  TensorFloat32 src_indices;
  src_indices.shape = BHWC(1, 1, 2, 3);
  src_indices.data = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
  TensorFloat32 src_updates;
  src_updates.shape = BHWC(1, 1, 2, 1);
  src_updates.data = {2.5f, 4.0f};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});  // indices
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});  // updates
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateScatterNd(op_def);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_indices, src_updates},
      std::make_unique<GPUOperation>(std::move(operation)), BHWC(1, 1, 2, 1),
      &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(kEps), {0.0f, 6.5f}));
  return absl::OkStatus();
}

}  // namespace ml_drift
