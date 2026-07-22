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

#include "ml_drift/common/kernels/tests/split_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/split.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace ml_drift {

// 2^-7 should be the maximum error for values in range [0.0, 32.0]
constexpr float kEps = 1.0f / (1 << 7);

absl::Status SplitChannelsTest(TestExecutionEnvironment& env,
                               DataType data_type, TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 3, 2, 5);
  src_tensor.data = {0.1f,  0.2f,  0.3f,  0.4f,  0.5f,  1.1f,  1.2f,  1.3f,
                     1.4f,  1.5f,  10.1f, 10.2f, 10.3f, 10.4f, 10.5f, 11.1f,
                     11.2f, 11.3f, 11.4f, 11.5f, 20.1f, 20.2f, 20.3f, 20.4f,
                     20.5f, 21.1f, 21.2f, 21.3f, 21.4f, 21.5f};

  SplitAttributes attr;
  attr.axis = Axis::CHANNELS;

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor0, dst_tensor1;
  Split operation = CreateSplit(env.GetGpuInfo(), op_def, attr, {2, 3});
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor}, std::make_unique<Split>(std::move(operation)),
      {BHWC(1, 3, 2, 2), BHWC(1, 3, 2, 3)}, {&dst_tensor0, &dst_tensor1}));
  EXPECT_THAT(
      dst_tensor0.data,
      Pointwise(FloatNear(kEps), {0.1f, 0.2f, 1.1f, 1.2f, 10.1f, 10.2f, 11.1f,
                                  11.2f, 20.1f, 20.2f, 21.1f, 21.2f}));
  EXPECT_THAT(
      dst_tensor1.data,
      Pointwise(FloatNear(kEps),
                {0.3f, 0.4f, 0.5f, 1.3f, 1.4f, 1.5f, 10.3f, 10.4f, 10.5f, 11.3f,
                 11.4f, 11.5f, 20.3f, 20.4f, 20.5f, 21.3f, 21.4f, 21.5f}));
  return absl::OkStatus();
}

absl::Status SplitChannelsX4Test(TestExecutionEnvironment& env,
                                 DataType data_type,
                                 TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 8);
  src_tensor.data = {0.1f,  0.2f,  0.3f,  0.4f,  1.1f,  1.2f,  1.3f,  1.4f,
                     10.1f, 10.2f, 10.3f, 10.4f, 11.1f, 11.2f, 11.3f, 11.4f,
                     20.1f, 20.2f, 20.3f, 20.4f, 21.1f, 21.2f, 21.3f, 21.4f,
                     30.1f, 30.2f, 30.3f, 30.4f, 31.1f, 31.2f, 31.3f, 31.4f};

  SplitAttributes attr;
  attr.axis = Axis::CHANNELS;

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor0, dst_tensor1;
  Split operation = CreateSplit(env.GetGpuInfo(), op_def, attr, {4, 4});
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor}, std::make_unique<Split>(std::move(operation)),
      {BHWC(1, 2, 2, 4), BHWC(1, 2, 2, 4)}, {&dst_tensor0, &dst_tensor1}));
  EXPECT_THAT(dst_tensor0.data,
              Pointwise(FloatNear(kEps), {0.1f, 0.2f, 0.3f, 0.4f, 10.1f, 10.2f,
                                          10.3f, 10.4f, 20.1f, 20.2f, 20.3f,
                                          20.4f, 30.1f, 30.2f, 30.3f, 30.4f}));
  EXPECT_THAT(dst_tensor1.data,
              Pointwise(FloatNear(kEps), {1.1f, 1.2f, 1.3f, 1.4f, 11.1f, 11.2f,
                                          11.3f, 11.4f, 21.1f, 21.2f, 21.3f,
                                          21.4f, 31.1f, 31.2f, 31.3f, 31.4f}));
  return absl::OkStatus();
}

absl::Status SplitWidthTest(TestExecutionEnvironment& env, DataType data_type,
                            TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 6, 5, 1);
  src_tensor.data = {0.1f,  0.2f,  0.3f,  0.4f,  0.5f,  1.1f,  1.2f,  1.3f,
                     1.4f,  1.5f,  10.1f, 10.2f, 10.3f, 10.4f, 10.5f, 11.1f,
                     11.2f, 11.3f, 11.4f, 11.5f, 20.1f, 20.2f, 20.3f, 20.4f,
                     20.5f, 21.1f, 21.2f, 21.3f, 21.4f, 21.5f};

  SplitAttributes attr;
  attr.axis = Axis::WIDTH;

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor0, dst_tensor1;
  Split operation = CreateSplit(env.GetGpuInfo(), op_def, attr, {1, 1});
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor}, std::make_unique<Split>(std::move(operation)),
      {BHWC(1, 6, 2, 1), BHWC(1, 6, 3, 1)}, {&dst_tensor0, &dst_tensor1}));
  EXPECT_THAT(
      dst_tensor0.data,
      Pointwise(FloatNear(kEps), {0.1f, 0.2f, 1.1f, 1.2f, 10.1f, 10.2f, 11.1f,
                                  11.2f, 20.1f, 20.2f, 21.1f, 21.2f}));
  EXPECT_THAT(
      dst_tensor1.data,
      Pointwise(FloatNear(kEps),
                {0.3f, 0.4f, 0.5f, 1.3f, 1.4f, 1.5f, 10.3f, 10.4f, 10.5f, 11.3f,
                 11.4f, 11.5f, 20.3f, 20.4f, 20.5f, 21.3f, 21.4f, 21.5f}));
  return absl::OkStatus();
}

absl::Status SplitHeightTest(TestExecutionEnvironment& env, DataType data_type,
                             TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 6, 5, 1);
  src_tensor.data = {0.1f,  0.2f,  0.3f,  0.4f,  0.5f,  1.1f,  1.2f,  1.3f,
                     1.4f,  1.5f,  10.1f, 10.2f, 10.3f, 10.4f, 10.5f, 11.1f,
                     11.2f, 11.3f, 11.4f, 11.5f, 20.1f, 20.2f, 20.3f, 20.4f,
                     20.5f, 21.1f, 21.2f, 21.3f, 21.4f, 21.5f};

  SplitAttributes attr;
  attr.axis = Axis::HEIGHT;

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor0, dst_tensor1;
  Split operation = CreateSplit(env.GetGpuInfo(), op_def, attr, {1, 1});
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor}, std::make_unique<Split>(std::move(operation)),
      {BHWC(1, 2, 5, 1), BHWC(1, 4, 5, 1)}, {&dst_tensor0, &dst_tensor1}));
  EXPECT_THAT(dst_tensor0.data,
              Pointwise(FloatNear(kEps), {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 1.1f,
                                          1.2f, 1.3f, 1.4f, 1.5f}));
  EXPECT_THAT(dst_tensor1.data,
              Pointwise(FloatNear(kEps),
                        {10.1f, 10.2f, 10.3f, 10.4f, 10.5f, 11.1f, 11.2f,
                         11.3f, 11.4f, 11.5f, 20.1f, 20.2f, 20.3f, 20.4f,
                         20.5f, 21.1f, 21.2f, 21.3f, 21.4f, 21.5f}));
  return absl::OkStatus();
}

absl::Status SplitBatchTest(TestExecutionEnvironment& env, DataType data_type,
                            TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(6, 1, 5, 1);
  src_tensor.data = {0.1f,  0.2f,  0.3f,  0.4f,  0.5f,  1.1f,  1.2f,  1.3f,
                     1.4f,  1.5f,  10.1f, 10.2f, 10.3f, 10.4f, 10.5f, 11.1f,
                     11.2f, 11.3f, 11.4f, 11.5f, 20.1f, 20.2f, 20.3f, 20.4f,
                     20.5f, 21.1f, 21.2f, 21.3f, 21.4f, 21.5f};

  SplitAttributes attr;
  attr.axis = Axis::BATCH;

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  TensorFloat32 dst_tensor0, dst_tensor1;
  Split operation = CreateSplit(env.GetGpuInfo(), op_def, attr, {1, 1});
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor}, std::make_unique<Split>(std::move(operation)),
      {BHWC(1, 1, 5, 1), BHWC(5, 1, 5, 1)}, {&dst_tensor0, &dst_tensor1}));
  EXPECT_THAT(dst_tensor0.data,
              Pointwise(FloatNear(kEps), {0.1f, 0.2f, 0.3f, 0.4f, 0.5f}));
  EXPECT_THAT(
      dst_tensor1.data,
      Pointwise(FloatNear(kEps),
                {1.1f,  1.2f,  1.3f,  1.4f,  1.5f,  10.1f, 10.2f, 10.3f, 10.4f,
                 10.5f, 11.1f, 11.2f, 11.3f, 11.4f, 11.5f, 20.1f, 20.2f, 20.3f,
                 20.4f, 20.5f, 21.1f, 21.2f, 21.3f, 21.4f, 21.5f}));
  return absl::OkStatus();
}

absl::Status SplitDepthTest(TestExecutionEnvironment& env, DataType data_type,
                            TensorStorageType storage) {
  Tensor5DFloat32 src_tensor;
  src_tensor.shape = BHWDC(1, 6, 1, 5, 1);
  src_tensor.data = {0.1f,  0.2f,  0.3f,  0.4f,  0.5f,  1.1f,  1.2f,  1.3f,
                     1.4f,  1.5f,  10.1f, 10.2f, 10.3f, 10.4f, 10.5f, 11.1f,
                     11.2f, 11.3f, 11.4f, 11.5f, 20.1f, 20.2f, 20.3f, 20.4f,
                     20.5f, 21.1f, 21.2f, 21.3f, 21.4f, 21.5f};

  SplitAttributes attr;
  attr.axis = Axis::DEPTH;

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWDC});
  Tensor5DFloat32 dst_tensor0, dst_tensor1;
  Split operation = CreateSplit(env.GetGpuInfo(), op_def, attr, {1, 1});
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor}, std::make_unique<Split>(std::move(operation)),
      {BHWDC(1, 6, 1, 2, 1), BHWDC(1, 6, 1, 3, 1)},
      {&dst_tensor0, &dst_tensor1}));
  EXPECT_THAT(
      dst_tensor0.data,
      Pointwise(FloatNear(kEps), {0.1f, 0.2f, 1.1f, 1.2f, 10.1f, 10.2f, 11.1f,
                                  11.2f, 20.1f, 20.2f, 21.1f, 21.2f}));
  EXPECT_THAT(
      dst_tensor1.data,
      Pointwise(FloatNear(kEps),
                {0.3f, 0.4f, 0.5f, 1.3f, 1.4f, 1.5f, 10.3f, 10.4f, 10.5f, 11.3f,
                 11.4f, 11.5f, 20.3f, 20.4f, 20.5f, 21.3f, 21.4f, 21.5f}));
  return absl::OkStatus();
}

absl::Status Split5DTest(TestExecutionEnvironment& env, DataType data_type,
                         TensorStorageType storage) {
  Tensor5DFloat32 src_tensor;
  src_tensor.shape = BHWDC(2, 3, 1, 4, 1);
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = static_cast<float>(i + 1);
  }

  SplitAttributes attr;
  attr.axis = Axis::DEPTH;

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWDC});
  Tensor5DFloat32 dst_tensor0, dst_tensor1;
  Split operation = CreateSplit(env.GetGpuInfo(), op_def, attr, {2, 2});
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor}, std::make_unique<Split>(std::move(operation)),
      {BHWDC(2, 3, 1, 2, 1), BHWDC(2, 3, 1, 2, 1)},
      {&dst_tensor0, &dst_tensor1}));

  EXPECT_THAT(
      dst_tensor0.data,
      Pointwise(FloatNear(0.0f), {
                                     1.f, 2.f, 5.f, 6.f, 9.f, 10.f,      // B=0
                                     13.f, 14.f, 17.f, 18.f, 21.f, 22.f  // B=1
                                 }));
  EXPECT_THAT(
      dst_tensor1.data,
      Pointwise(FloatNear(0.0f), {
                                     3.f, 4.f, 7.f, 8.f, 11.f, 12.f,     // B=0
                                     15.f, 16.f, 19.f, 20.f, 23.f, 24.f  // B=1
                                 }));
  return absl::OkStatus();
}

}  // namespace ml_drift
