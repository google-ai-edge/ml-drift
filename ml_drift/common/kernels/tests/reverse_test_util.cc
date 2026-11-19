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

#include "ml_drift/common/kernels/tests/reverse_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/reverse.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

absl::Status ReverseHWCTest(TestExecutionEnvironment& env, DataType data_type,
                            TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 3, 2);
  src_tensor.data = {1, 2, 3, 4, 5, 6};
  ReverseAttributes attr;
  attr.axes = {Axis::kChannels};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::kFloat32, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({DataType::kFloat32, storage, Layout::kHWC});
  TensorDescriptor src, dst;
  src = op_def.src_tensors[0];
  src.UploadData(src_tensor);
  dst.SetBHWCShape(BHWC(1, 1, 3, 2));
  GPUOperation operation = CreateReverse(op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src}, {&dst}, std::make_unique<GPUOperation>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(0.0f), {2, 1, 4, 3, 6, 5}));
  return absl::OkStatus();
}

absl::Status ReverseBHWCTest(TestExecutionEnvironment& env, DataType data_type,
                             TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(3, 1, 1, 2);
  src_tensor.data = {1, 2, 3, 4, 5, 6};
  ReverseAttributes attr;
  attr.axes = {Axis::kBatch};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::kFloat32, storage, Layout::kBHWC});
  op_def.dst_tensors.push_back({DataType::kFloat32, storage, Layout::kBHWC});
  TensorDescriptor src, dst;
  src = op_def.src_tensors[0];
  src.UploadData(src_tensor);
  dst.SetBHWCShape(BHWC(3, 1, 1, 2));
  GPUOperation operation = CreateReverse(op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src}, {&dst}, std::make_unique<GPUOperation>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(0.0f), {5, 6, 3, 4, 1, 2}));
  return absl::OkStatus();
}
}  // namespace ml_drift
