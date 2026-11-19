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

#include "ml_drift/common/kernels/tests/prelu_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/prelu.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

absl::Status PReLUAlphaTest(TestExecutionEnvironment& env, DataType data_type,
                            TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, -1.0f, -2.0f, 3.0f};

  PReLUAttributes attr;
  Tensor<Linear, DataType::kFloat32> parameters;
  parameters.shape = Linear(2);
  parameters.data = {0.5f, -2.0f};
  attr.alpha = parameters;

  const float eps = data_type == DataType::kFloat32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreatePReLU(env.GetGpuInfo(), op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {0.0f, 2.0f, -1.0f, 3.0f}));
  return absl::OkStatus();
}

absl::Status PReLUHWCAlphaTest(TestExecutionEnvironment& env,
                               DataType data_type, TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, -1.0f, -2.0f, 3.0f};

  PReLUAttributes attr;
  Tensor<HWC, DataType::kFloat32> hwc_tensor;
  hwc_tensor.shape = HWC(2, 1, 2);
  hwc_tensor.data = {0.5f, -2.0f, 0.7f, 4.7f};
  attr.alpha = hwc_tensor;

  const float eps = data_type == DataType::kFloat32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreatePReLU(env.GetGpuInfo(), op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {0.0f, 2.0f, -1.4f, 3.0f}));
  return absl::OkStatus();
}

}  // namespace ml_drift
