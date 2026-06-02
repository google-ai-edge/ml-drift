// Copyright 2024 The ML Drift Authors.
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

#include "ml_drift/common/kernels/tests/relu_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/relu.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

absl::Status ReLUNoClipNoAlphaTest(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {-0.5f, 0.8f, -0.6f, 3.2f};

  ReLUAttributes attr;
  attr.alpha = 0.0f;
  attr.activation_max = 0.0f;

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateReLU(op_def, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {0.0f, 0.8f, 0.0f, 3.2f}));
  return absl::OkStatus();
}

absl::Status ReLUClipTest(TestExecutionEnvironment& env, DataType data_type,
                          TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {-0.5f, 0.8f, -0.6f, 3.2f};

  ReLUAttributes attr;
  attr.alpha = 0.0f;
  attr.activation_max = 0.9f;

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateReLU(op_def, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {0.0f, 0.8f, 0.0f, 0.9f}));
  return absl::OkStatus();
}

absl::Status ReLUAlphaTest(TestExecutionEnvironment& env, DataType data_type,
                           TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {-0.5f, 0.8f, -0.6f, 3.2f};

  ReLUAttributes attr;
  attr.alpha = 0.5f;
  attr.activation_max = 0.0f;

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateReLU(op_def, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {-0.25f, 0.8f, -0.3f, 3.2f}));
  return absl::OkStatus();
}

absl::Status ReLUAlphaClipTest(TestExecutionEnvironment& env,
                               DataType data_type, TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {-0.5f, 0.8f, -0.6f, 3.2f};

  ReLUAttributes attr;
  attr.alpha = 0.5f;
  attr.activation_max = 0.5f;

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateReLU(op_def, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {-0.25f, 0.5f, -0.3f, 0.5f}));
  return absl::OkStatus();
}

}  // namespace ml_drift
