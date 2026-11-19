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

#include "ml_drift/common/kernels/tests/one_hot_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/one_hot.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

absl::Status OneHotTest(TestExecutionEnvironment& env, DataType data_type,
                        TensorStorageType storage) {
  Tensor<BHWC, DataType::kInt32> src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 1);
  src_tensor.data = {3};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::kInt32, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});
  TensorDescriptor src = op_def.src_tensors[0];
  TensorDescriptor dst = op_def.dst_tensors[0];
  OneHotAttributes attr;
  GPUOperation operation = CreateOneHot(op_def, attr);
  src.UploadData(src_tensor);
  dst.SetBHWCShape(BHWC(1, 1, 1, 8));
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src}, {&dst}, std::make_unique<GPUOperation>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst.DownloadData(&dst_tensor);
  TensorFloat32 exp_tensor;
  exp_tensor.data = {0, 0, 0, 1.0, 0, 0, 0, 0};
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(0.0f), exp_tensor.data));
  return absl::OkStatus();
}

absl::Status OneHotBatchTest(TestExecutionEnvironment& env, DataType data_type,
                             TensorStorageType storage) {
  Tensor<BHWC, DataType::kInt32> src_tensor;
  src_tensor.shape = BHWC(10, 1, 1, 1);
  src_tensor.data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::kInt32, storage, Layout::kBHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kBHWC});
  TensorDescriptor src = op_def.src_tensors[0];
  TensorDescriptor dst = op_def.dst_tensors[0];
  OneHotAttributes attr = {/*on_value=*/2.0, /*off_value=*/-2.0};
  GPUOperation operation = CreateOneHot(op_def, attr);
  src.UploadData(src_tensor);
  dst.SetBHWCShape(BHWC(10, 1, 1, 10));
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src}, {&dst}, std::make_unique<GPUOperation>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst.DownloadData(&dst_tensor);
  std::vector<float> expected(100, attr.off_value);
  for (int i = 0; i < 10; i++) {
    expected[i * 10 + i] = attr.on_value;
  }
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(0.0f), expected));
  return absl::OkStatus();
}

absl::Status OneHot2DTest(TestExecutionEnvironment& env, DataType data_type,
                          TensorStorageType storage) {
  Tensor<BHWC, DataType::kInt32> src_tensor;
  src_tensor.shape = BHWC(2, 1, 1, 3);
  src_tensor.data = {0, 1, 2, 2, 1, 0};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::kInt32, storage, Layout::kBHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kBHWC});
  TensorDescriptor src = op_def.src_tensors[0];
  TensorDescriptor dst = op_def.dst_tensors[0];
  OneHotAttributes attr = {/*on_value=*/1.0, /*off_value=*/0.0};
  GPUOperation operation = CreateOneHot(op_def, attr);
  src.UploadData(src_tensor);
  dst.SetBHWCShape(BHWC(2, 1, 3, 3));
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src}, {&dst}, std::make_unique<GPUOperation>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst.DownloadData(&dst_tensor);

  std::vector<float> expected = {
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0,

      0.0, 0.0, 1.0,
      0.0, 1.0, 0.0,
      1.0, 0.0, 0.0
  };
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(0.0f), expected));
  return absl::OkStatus();
}

}  // namespace ml_drift
