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

#include "ml_drift/common/kernels/tests/conversion_test_util.h"

#include <memory>
#include <utility>

#include "gmock/gmock.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/conversion.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

// 2^-9 should be the maximum error for values in range [0.0, 8.0]
constexpr float kEps = 1.0f / (1 << 9);

absl::Status TensorToTensorTest(TestExecutionEnvironment& env,
                                DataType data_type, TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 3, 4);
  src_tensor.data = {
      1.0f, 2.0f, 3.0f, 4.0f,    // (0,0) channels 0-3
      5.0f, 6.0f, 7.0f, 0.5f,    // (0,1) channels 0-3
      -1.0f, -2.0f, 0.0f, 1.5f,  // (0,2) channels 0-3
      0.1f, 0.2f, 0.3f, 0.4f,    // (1,0) channels 0-3
      2.1f, 3.1f, 4.1f, 5.1f,    // (1,1) channels 0-3
      6.1f, 7.1f, 0.9f, -0.5f,   // (1,2) channels 0-3
  };

  TensorDescriptor src_desc{data_type, storage, Layout::kHWC};
  TensorDescriptor dst_desc{data_type, storage, Layout::kHWC};
  TensorToTensor operation =
      CreateTensorToTensorOp(env.GetGpuInfo(), src_desc, dst_desc);

  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<TensorToTensor>(std::move(operation)),
      src_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(kEps), src_tensor.data));
  return absl::OkStatus();
}

absl::Status TensorToTensorNonAlignedChannelsTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 3);
  src_tensor.data = {
      1.0f, 2.0f, 3.0f,    // (0,0) channels 0-2
      4.0f, 5.0f, 6.0f,    // (0,1) channels 0-2
      7.0f, 0.5f, -1.0f,   // (1,0) channels 0-2
      -2.0f, 0.0f, 1.5f,   // (1,1) channels 0-2
  };

  TensorDescriptor src_desc{data_type, storage, Layout::kHWC};
  TensorDescriptor dst_desc{data_type, storage, Layout::kHWC};
  TensorToTensor operation =
      CreateTensorToTensorOp(env.GetGpuInfo(), src_desc, dst_desc);

  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<TensorToTensor>(std::move(operation)),
      src_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(kEps), src_tensor.data));
  return absl::OkStatus();
}

absl::Status TensorToTensorBatchedTest(TestExecutionEnvironment& env,
                                       DataType data_type,
                                       TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(2, 1, 2, 4);
  src_tensor.data = {
      // Batch 0
      1.0f, 2.0f, 3.0f, 4.0f,    // (0,0) channels 0-3
      5.0f, 6.0f, 7.0f, 0.5f,    // (0,1) channels 0-3
      // Batch 1
      -1.0f, -2.0f, 0.0f, 1.5f,  // (0,0) channels 0-3
      0.1f, 0.2f, 0.3f, 0.4f,    // (0,1) channels 0-3
  };

  TensorDescriptor src_desc{data_type, storage, Layout::kBHWC};
  TensorDescriptor dst_desc{data_type, storage, Layout::kBHWC};
  TensorToTensor operation =
      CreateTensorToTensorOp(env.GetGpuInfo(), src_desc, dst_desc);

  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<TensorToTensor>(std::move(operation)),
      src_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(kEps), src_tensor.data));
  return absl::OkStatus();
}

}  // namespace ml_drift
