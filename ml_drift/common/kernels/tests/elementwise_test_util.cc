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

#include "ml_drift/common/kernels/tests/elementwise_test_util.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/elementwise.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace ml_drift {

absl::Status AbsTest(TestExecutionEnvironment& env,
                     DataType data_type,
                     TensorStorageType storage) {
  // 2^-12 should be the maximum error for values in range [0.0, 1.0]
  constexpr float kEps = 1.0f / (1 << 12);

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, -1.0f, 0.05f, 0.045f};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwiseOneInput(env.GetGpuInfo(), op_def, OperationType::ABS);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps), {0.0f, 1.0f, 0.05f, 0.045f}));
  return absl::OkStatus();
}

absl::Status CosTest(TestExecutionEnvironment& env,
                     DataType data_type,
                     TensorStorageType storage) {
  // 2^-12 should be the maximum error for values in range [0.0, 1.0],
  // but due to loss at beginning, cos, loss at the end, the error is larger.
  constexpr float kEps = 1.5f / (1 << 11);

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, -1.0f, -0.05f, 0.045f};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwiseOneInput(env.GetGpuInfo(), op_def, OperationType::COS);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps), {std::cos(0.0f), std::cos(-1.0f),
                                          std::cos(-0.05f), std::cos(0.045f)}));
  return absl::OkStatus();
}

absl::Status CosIntTest(TestExecutionEnvironment& env,
                        TensorStorageType storage) {
  TensorInt32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0, -10, 20, 5};

  const DataType data_type = DataType::INT32;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

  GPUOperation operation =
      CreateElementwiseOneInput(env.GetGpuInfo(), op_def, OperationType::COS);
  TensorDescriptor src_desc, dst_desc;
  src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src_tensor);
  dst_desc.SetBHWCShape(src_tensor.shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  TensorInt32 dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  TensorInt32 ref_tensor;
  ref_tensor.data = {1, -1, 0, 0};
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

absl::Status CopyTest(TestExecutionEnvironment& env,
                      DataType data_type,
                      TensorStorageType storage) {
  // 2^-12 should be the maximum error for values in range [0.0, 1.0]
  constexpr float kEps = 1.0f / (1 << 12);

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, 1.0f, 0.05f, 0.045f};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwiseOneInput(env.GetGpuInfo(), op_def, OperationType::COPY);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(kEps), src_tensor.data));
  return absl::OkStatus();
}

absl::Status EluTest(TestExecutionEnvironment& env, DataType data_type,
                     TensorStorageType storage) {
  // 2^-5 should be the maximum error for values in range [0.0, 100.0]
  constexpr float kEps = 1.0f / (1 << 5);

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 7);
  src_tensor.data = {0.0f, 1.0f, -1.0f, 100.0f, -100.0f, 0.01f, -0.01f};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwiseOneInput(env.GetGpuInfo(), op_def, OperationType::ELU);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 1, 1, 7), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps), {0.0f, 1.0f, std::exp(-1.0f) - 1.0f,
                                          100.0f, std::exp(-100.0f) - 1.0f,
                                          0.01f, std::exp(-0.01f) - 1.0f}));
  return absl::OkStatus();
}

absl::Status ExpTest(TestExecutionEnvironment& env, DataType data_type,
                     TensorStorageType storage) {
  // 2^-8 should be the maximum error for values in range [0.0, 16.0],
  // but due to loss at beginning, exp, loss at the end, the error is larger.
  constexpr float kEps = 5.0f / (1 << 8);

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 7);
  src_tensor.data = {0.0f, 1.0f, -1.0f, 2.5f, -1.7f, 0.01f, -0.01f};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwiseOneInput(env.GetGpuInfo(), op_def, OperationType::EXP);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 1, 1, 7), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps),
                        {std::exp(0.0f), std::exp(1.0f), std::exp(-1.0f),
                         std::exp(2.5f), std::exp(-1.7f), std::exp(0.01f),
                         std::exp(-0.01f)}));
  return absl::OkStatus();
}

absl::Status FloorTest(TestExecutionEnvironment& env, DataType data_type,
                       TensorStorageType storage) {
  // 2^-9 should be the maximum error for values in range [0.0, 8.0]
  constexpr float kEps = 1.0f / (1 << 9);

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 7);
  src_tensor.data = {-4.5f, -3.0f, -1.5f, 0.0f, 1.5f, 3.0f, 4.5f};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwiseOneInput(env.GetGpuInfo(), op_def, OperationType::FLOOR);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      src_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps),
                        {-5.0f, -3.0f, -2.0f, 0.0f, 1.0f, 3.0f, 4.0f}));
  return absl::OkStatus();
}

absl::Status FloorDivTest(TestExecutionEnvironment& env, DataType data_type,
                          TensorStorageType storage) {
  // 2^-9 should be the maximum error for values in range [0.0, 8.0]
  constexpr float kEps = 1.0f / (1 << 9);

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 7);
  src_tensor.data = {-4.5f, -3.0f, -1.5f, 0.0f, 1.5f, 3.0f, 4.5f};

  float scalar = 2.7f;
  ElementwiseAttributes attr;
  attr.param = scalar;

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateElementwise(env.GetGpuInfo(), op_def,
                                             OperationType::FLOOR_DIV, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      src_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps),
                        {std::floor(-4.5f / scalar), std::floor(-3.0f / scalar),
                         std::floor(-1.5f / scalar), std::floor(0.0f / scalar),
                         std::floor(1.5f / scalar), std::floor(3.0f / scalar),
                         std::floor(4.5f / scalar)}));
  return absl::OkStatus();
}

absl::Status FloorDivIntTest(TestExecutionEnvironment& env,
                             TensorStorageType storage) {
  TensorInt32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 1, 1, 7);
  src_tensor_0.data = {-50, -7, -3, 0, 5, 17, 99};
  TensorInt32 src_tensor_1;
  src_tensor_1.shape = BHWC(1, 1, 1, 7);
  src_tensor_1.data = {-4, 3, 1, -4, 5, -4, -100};

  auto data_type = DataType::INT32;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

  GPUOperation operation = CreateElementwiseTwoInput(
      env.GetGpuInfo(), op_def, OperationType::FLOOR_DIV, src_tensor_1.shape,
      src_tensor_0.shape);

  TensorDescriptor src_desc0, src_desc1, dst_desc;
  src_desc0 = op_def.src_tensors[0];
  src_desc0.UploadData(src_tensor_0);
  src_desc1 = op_def.src_tensors[1];
  src_desc1.UploadData(src_tensor_1);
  dst_desc.SetBHWCShape(src_tensor_0.shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc0, &src_desc1}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  TensorInt32 dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  TensorInt32 ref_tensor;
  ref_tensor.data = {12, -3, -3, 0, 1, -5, -1};
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

absl::Status FloorModTest(TestExecutionEnvironment& env, DataType data_type,
                          TensorStorageType storage) {
  // 2^-9 should be the maximum error for values in range [0.0, 8.0]
  constexpr float kEps = 1.0f / (1 << 9);

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 7);
  src_tensor.data = {-4.5f, -3.0f, -1.5f, 0.0f, 1.5f, 3.0f, 4.5f};

  float scalar = 2.7f;
  ElementwiseAttributes attr;
  attr.param = scalar;

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateElementwise(env.GetGpuInfo(), op_def,
                                             OperationType::FLOOR_MOD, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      src_tensor.shape, &dst_tensor));
  EXPECT_THAT(
      dst_tensor.data,
      Pointwise(FloatNear(kEps), {-4.5f - std::floor(-4.5f / scalar) * scalar,
                                  -3.0f - std::floor(-3.0f / scalar) * scalar,
                                  -1.5f - std::floor(-1.5f / scalar) * scalar,
                                  0.0f - std::floor(0.0f / scalar) * scalar,
                                  1.5f - std::floor(1.5f / scalar) * scalar,
                                  3.0f - std::floor(3.0f / scalar) * scalar,
                                  4.5f - std::floor(4.5f / scalar) * scalar}));
  return absl::OkStatus();
}

absl::Status FloorModIntTest(TestExecutionEnvironment& env,
                             TensorStorageType storage) {
  TensorInt32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 1, 1, 7);
  src_tensor_0.data = {-50, -7, -3, 0, 5, 17, 99};
  TensorInt32 src_tensor_1;
  src_tensor_1.shape = BHWC(1, 1, 1, 7);
  src_tensor_1.data = {-4, 3, 1, -4, 5, -4, -100};

  auto data_type = DataType::INT32;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

  GPUOperation operation = CreateElementwiseTwoInput(
      env.GetGpuInfo(), op_def, OperationType::FLOOR_MOD, src_tensor_1.shape,
      src_tensor_0.shape);

  TensorDescriptor src_desc0, src_desc1, dst_desc;
  src_desc0 = op_def.src_tensors[0];
  src_desc0.UploadData(src_tensor_0);
  src_desc1 = op_def.src_tensors[1];
  src_desc1.UploadData(src_tensor_1);
  dst_desc.SetBHWCShape(src_tensor_0.shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc0, &src_desc1}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  TensorInt32 dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  TensorInt32 ref_tensor;
  ref_tensor.data = {-2, 2, 0, 0, 0, -3, -1};
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

absl::Status GeluTest(TestExecutionEnvironment& env, DataType data_type,
                      TensorStorageType storage) {
  // 2^-9 should be the maximum error for values in range [0.0, 8.0]
  constexpr float kEps = 1.0f / (1 << 9);

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 6);
  src_tensor.data = {0.0f, 1.0f, 3.0f, 1.0f, -1.0f, -2.0f};
  std::vector<float> ref_out(src_tensor.data.size());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    float x = src_tensor.data[i];
    ref_out[i] = 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
  }

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwiseOneInput(env.GetGpuInfo(), op_def, OperationType::GELU);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 1, 1, 6), &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(kEps), ref_out));
  return absl::OkStatus();
}

absl::Status HardSwishTest(TestExecutionEnvironment& env,
                           DataType data_type,
                           TensorStorageType storage) {
  // 2^-9 should be the maximum error for values in range [0.0, 8.0]
  constexpr float kEps = 1.0f / (1 << 9);

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 7);
  src_tensor.data = {-4.5f, -3.0f, -1.5f, 0.0f, 1.5f, 3.0f, 4.5f};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateElementwiseOneInput(env.GetGpuInfo(), op_def,
                                                     OperationType::HARD_SWISH);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      src_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps),
                        {0.0f, 0.0f, -0.375f, 0.0f, 1.125f, 3.f, 4.5f}));
  return absl::OkStatus();
}

absl::Status LogTest(TestExecutionEnvironment& env, DataType data_type,
                     TensorStorageType storage) {
  // 2^-11 should be the maximum error for values in range [0.0, 2.0],
  // but due to loss at beginning, log, loss at the end, the error is larger.
  constexpr float kEps = 2.0f / (1 << 11);

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwiseOneInput(env.GetGpuInfo(), op_def, OperationType::LOG);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps), {std::log(1.0f), std::log(2.0f),
                                          std::log(3.0f), std::log(4.0f)}));

  return absl::OkStatus();
}

absl::Status NegTest(TestExecutionEnvironment& env,
                     DataType data_type,
                     TensorStorageType storage) {
  // 2^-10 should be the maximum error for values in range [0.0, 4.0]
  constexpr float kEps = 1.0f / (1 << 10);

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {1.0f, -2.0f, 0.0f, 4.0f};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwiseOneInput(env.GetGpuInfo(), op_def, OperationType::NEG);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps), {-1.0f, 2.0f, 0.0f, -4.0f}));
  return absl::OkStatus();
}

absl::Status RoundTest(TestExecutionEnvironment& env,
                       DataType data_type,
                       TensorStorageType storage) {
  // 2^-10 should be the maximum error for values in range [0.0, 4.0]
  constexpr float kEps = 1.0f / (1 << 10);

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.9f, 2.5f, 3.5f, 4.1f};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwiseOneInput(env.GetGpuInfo(), op_def, OperationType::ROUND);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps), {1.0f, 2.0f, 4.0f, 4.0f}));
  return absl::OkStatus();
}

absl::Status RsqrtTest(TestExecutionEnvironment& env,
                       DataType data_type,
                       TensorStorageType storage) {
  // 2^-12 should be the maximum error for values in range [0.0, 1.0]
  constexpr float kEps = 1.0f / (1 << 12);

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwiseOneInput(env.GetGpuInfo(), op_def, OperationType::RSQRT);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps),
                        {1.0f / std::sqrt(1.0f), 1.0f / std::sqrt(2.0f),
                         1.0f / std::sqrt(3.0f), 1.0f / std::sqrt(4.0f)}));
  return absl::OkStatus();
}

absl::Status SigmoidTest(TestExecutionEnvironment& env,
                         DataType data_type,
                         TensorStorageType storage) {
  // 2^-13 should be the maximum error for values in range [0.0, 0.5],
  // but due to loss at beginning, log, loss at the end, the error is larger.
  constexpr float kEps = 1.5f / (1 << 13);

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {-std::log(1.0f), -std::log(2.0f), -std::log(3.0f),
                     -std::log(4.0f)};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateElementwiseOneInput(env.GetGpuInfo(), op_def,
                                                     OperationType::SIGMOID);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps), {0.5f, 1.0f / 3.0f, 0.25f, 0.2f}));
  return absl::OkStatus();
}

absl::Status SignTest(TestExecutionEnvironment& env,
                      DataType data_type,
                      TensorStorageType storage) {
  // 2^-3 should be the maximum error for values in range [0.0, 512.0]
  constexpr float kEps = 1.0f / (1 << 3);

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {-0.75f, 0.54f, 1.54f, -345.0f};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwiseOneInput(env.GetGpuInfo(), op_def, OperationType::SIGN);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps), {-1.0f, 1.0f, 1.0f, -1.0f}));
  return absl::OkStatus();
}

absl::Status SignInt8Test(TestExecutionEnvironment& env,
                          TensorStorageType storage) {
  Tensor<BHWC, DataType::INT8> src_tensor_si8;
  src_tensor_si8.shape = BHWC(1, 2, 1, 2);
  src_tensor_si8.data = {-75, 54, 120, 0};
  BHWC dst_shape_si8 = BHWC(1, 2, 1, 2);

  Tensor<BHWC, DataType::INT8> ref_tensor_si8;
  ref_tensor_si8.shape = BHWC(1, 2, 1, 2);
  ref_tensor_si8.data = {-1, 1, 1, 0};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::INT8, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::INT8, storage, Layout::HWC});

  TensorDescriptor src_desc, dst_desc;
  src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src_tensor_si8);
  dst_desc.SetBHWCShape(ref_tensor_si8.shape);
  GPUOperation operation =
      CreateElementwiseOneInput(env.GetGpuInfo(), op_def, OperationType::SIGN);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::INT8> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor_si8.data);

  return absl::OkStatus();
}

absl::Status SinTest(TestExecutionEnvironment& env,
                     DataType data_type,
                     TensorStorageType storage) {
  // 2^-11 should be the maximum error for values in range [-1.0, 1.0],
  // but due to loss at beginning, sin, loss at the end, the error is larger.
  constexpr float kEps = 2.0f / (1 << 11);

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, -1.0f, -0.05f, 0.045f};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwiseOneInput(env.GetGpuInfo(), op_def, OperationType::SIN);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps), {std::sin(0.0f), std::sin(-1.0f),
                                          std::sin(-0.05f), std::sin(0.045f)}));

  return absl::OkStatus();
}

absl::Status SqrtTest(TestExecutionEnvironment& env,
                      DataType data_type,
                      TensorStorageType storage) {
  // 2^-11 should be the maximum error for values in range [0.0, 2.0],
  // but due to loss at beginning, sqrt, loss at the end, the error is larger.
  constexpr float kEps = 1.5f / (1 << 11);

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwiseOneInput(env.GetGpuInfo(), op_def, OperationType::SQRT);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps), {std::sqrt(1.0f), std::sqrt(2.0f),
                                          std::sqrt(3.0f), std::sqrt(4.0f)}));
  return absl::OkStatus();
}

absl::Status SquareTest(TestExecutionEnvironment& env,
                        DataType data_type,
                        TensorStorageType storage) {
  // 2^-8 should be the maximum error for values in range [0.0, 16.0]
  constexpr float kEps = 1.0f / (1 << 8);

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {1.0f, -2.0f, 3.0f, 4.0f};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateElementwiseOneInput(env.GetGpuInfo(), op_def,
                                                     OperationType::SQUARE);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps), {1.0f, 4.0f, 9.0f, 16.0f}));
  return absl::OkStatus();
}

absl::Status TanhTest(TestExecutionEnvironment& env,
                      DataType data_type,
                      TensorStorageType storage) {
  // 2^-12 should be the maximum error for values in range [0.0, 1.0],
  // but due to loss at beginning, tanh, loss at the end, the error is larger.
  constexpr float kEps = 2.f / (1 << 12);

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 3);
  src_tensor.data = {-4.0f, -0.1f, 0.1f, 2.0f, 6.0f, -6.0f};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwiseOneInput(env.GetGpuInfo(), op_def, OperationType::TANH);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 3), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps),
                        {std::tanh(-4.0f), std::tanh(-0.1f), std::tanh(0.1f),
                         std::tanh(2.0f), std::tanh(6.0f), std::tanh(-6.0f)}));
  return absl::OkStatus();
}

absl::Status SubTest(TestExecutionEnvironment& env,
                     DataType data_type,
                     TensorStorageType storage) {
  // 2^-12 should be the maximum error for values in range [0.0, 1.0]
  constexpr float kEps = 1.0f / (1 << 12);

  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {1.0f, 2.0f, 3.0f, 4.0f};
  src_tensor_1.data = {0.5f, 1.0f, 3.0f, 3.5f};
  BHWC dst_shape = BHWC(1, 2, 1, 2);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwiseTwoInput(env.GetGpuInfo(), op_def, OperationType::SUB,
                                src_tensor_1.shape, dst_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor_0, src_tensor_1},
      std::make_unique<GPUOperation>(std::move(operation)), dst_shape,
      &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps), {0.5f, 1.0f, 0.0f, 0.5f}));
  return absl::OkStatus();
}

absl::Status ShiftLeftTest(TestExecutionEnvironment& env,
                           TensorStorageType storage) {
  Tensor<BHWC, DataType::INT32> src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {-1, 0, 1, 10};
  src_tensor_1.data = {1, 2, 3, 1};
  BHWC dst_shape = BHWC(1, 2, 1, 2);

  Tensor<BHWC, DataType::INT32> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {-2, 0, 8, 20};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  op_def.src_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::INT32, storage, Layout::HWC});

  TensorDescriptor src_desc0, src_desc1, dst_desc;
  src_desc0 = op_def.src_tensors[0];
  src_desc0.UploadData(src_tensor_0);
  src_desc1 = op_def.src_tensors[1];
  src_desc1.UploadData(src_tensor_1);
  dst_desc.SetBHWCShape(dst_shape);
  GPUOperation operation = CreateElementwiseTwoInput(
      env.GetGpuInfo(), op_def, OperationType::SHIFT_LEFT, src_tensor_1.shape,
      dst_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc0, &src_desc1}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::INT32> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

absl::Status ShiftRightTest(TestExecutionEnvironment& env,
                            TensorStorageType storage) {
  Tensor<BHWC, DataType::INT32> src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {-1, 0, 8, 10};
  src_tensor_1.data = {1, 2, 3, 1};
  BHWC dst_shape = BHWC(1, 2, 1, 2);

  Tensor<BHWC, DataType::INT32> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {-1, 0, 1, 5};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  op_def.src_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::INT32, storage, Layout::HWC});

  TensorDescriptor src_desc0, src_desc1, dst_desc;
  src_desc0 = op_def.src_tensors[0];
  src_desc0.UploadData(src_tensor_0);
  src_desc1 = op_def.src_tensors[1];
  src_desc1.UploadData(src_tensor_1);
  dst_desc.SetBHWCShape(dst_shape);
  GPUOperation operation = CreateElementwiseTwoInput(
      env.GetGpuInfo(), op_def, OperationType::SHIFT_RIGHT, src_tensor_1.shape,
      dst_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc0, &src_desc1}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::INT32> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

absl::Status SquaredDiffTest(TestExecutionEnvironment& env,
                             DataType data_type,
                             TensorStorageType storage) {
  // 2^-12 should be the maximum error for values in range [0.0, 1.0]
  constexpr float kEps = 1.0f / (1 << 12);

  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {1.0f, 2.0f, 3.0f, 4.0f};
  src_tensor_1.data = {0.5f, 1.0f, 3.0f, 3.5f};
  BHWC dst_shape = BHWC(1, 2, 1, 2);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateElementwiseTwoInput(
      env.GetGpuInfo(), op_def, OperationType::SQUARED_DIFF, src_tensor_1.shape,
      dst_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor_0, src_tensor_1},
      std::make_unique<GPUOperation>(std::move(operation)), dst_shape,
      &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps), {0.25f, 1.0f, 0.0f, 0.25f}));
  return absl::OkStatus();
}

absl::Status DivTest(TestExecutionEnvironment& env,
                     DataType data_type,
                     TensorStorageType storage) {
  // 2^-10 should be the maximum error for values in range [0.0, 4.0]
  constexpr float kEps = 1.0f / (1 << 10);

  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {1.0f, 2.0f, 3.0f, 4.5f};
  src_tensor_1.data = {0.5f, 1.0f, 3.0f, 1.5f};
  BHWC dst_shape = BHWC(1, 2, 1, 2);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwiseTwoInput(env.GetGpuInfo(), op_def, OperationType::DIV,
                                src_tensor_1.shape, dst_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor_0, src_tensor_1},
      std::make_unique<GPUOperation>(std::move(operation)), dst_shape,
      &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps), {2.0f, 2.0f, 1.0f, 3.0f}));
  return absl::OkStatus();
}

absl::Status ModInt3Test(TestExecutionEnvironment& env,
                         TensorStorageType storage) {
  Tensor<BHWC, DataType::INT32> src;
  src.shape = BHWC(1, 2, 1, 2);
  src.data = {3, 4, 5, 6};
  Tensor<BHWC, DataType::INT32> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {0, 1, 2, 0};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  TensorDescriptor src_td, dst_td;
  src_td = op_def.src_tensors[0];
  src_td.UploadData(src);
  dst_td.SetBHWCShape(BHWC(1, 2, 1, 2));
  ElementwiseAttributes attr;
  attr.param = 3;
  GPUOperation operation =
      CreateElementwise(env.GetGpuInfo(), op_def, OperationType::MOD, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_td}, {&dst_td},
      std::make_unique<GPUOperation>(std::move(operation))));
  Tensor<BHWC, DataType::INT32> dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);

  return absl::OkStatus();
}

absl::Status ModUint5Test(TestExecutionEnvironment& env,
                          TensorStorageType storage) {
  Tensor<BHWC, DataType::UINT32> src;
  src.shape = BHWC(1, 2, 1, 2);
  src.data = {3, 4, 5, 6};
  Tensor<BHWC, DataType::UINT32> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {3, 4, 0, 1};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::UINT32, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::UINT32, storage, Layout::HWC});
  TensorDescriptor src_td, dst_td;
  src_td = op_def.src_tensors[0];
  src_td.UploadData(src);
  dst_td.SetBHWCShape(BHWC(1, 2, 1, 2));
  ElementwiseAttributes attr;
  attr.param = 5u;
  GPUOperation operation =
      CreateElementwise(env.GetGpuInfo(), op_def, OperationType::MOD, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_td}, {&dst_td},
      std::make_unique<GPUOperation>(std::move(operation))));
  Tensor<BHWC, DataType::UINT32> dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);

  return absl::OkStatus();
}

absl::Status PowTest(TestExecutionEnvironment& env,
                     DataType data_type,
                     TensorStorageType storage) {
  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {6.0f, 7.0f, 4.0f, 2.0f};
  src_tensor_1.data = {0.0f, 1.0f, 2.0f, 3.0f};
  BHWC dst_shape = BHWC(1, 2, 1, 2);

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwiseTwoInput(env.GetGpuInfo(), op_def, OperationType::POW,
                                src_tensor_1.shape, dst_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor_0, src_tensor_1},
      std::make_unique<GPUOperation>(std::move(operation)), dst_shape,
      &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {1.0f, 7.0f, 16.0f, 8.0f}));
  return absl::OkStatus();
}

absl::Status PowWithScalarTest(TestExecutionEnvironment& env,
                               DataType data_type, TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {-2.0f, -3.0f, 2.0f, 0.0f};
  BHWC dst_shape = BHWC(1, 2, 1, 2);

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  ElementwiseAttributes attr;
  attr.param = 2.0f;
  GPUOperation operation =
      CreateElementwise(env.GetGpuInfo(), op_def, OperationType::POW, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor}, std::make_unique<GPUOperation>(std::move(operation)),
      dst_shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {4.0f, 9.0f, 4.0f, 0.0f}));
  return absl::OkStatus();
}

absl::Status AddTest(TestExecutionEnvironment& env,
                     DataType data_type,
                     TensorStorageType storage) {
  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {1.0f, 2.0f, 3.0f, 4.5f};
  src_tensor_1.data = {0.5f, 1.0f, 3.0f, 1.5f};
  BHWC dst_shape = BHWC(1, 2, 1, 2);

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwiseTwoInput(env.GetGpuInfo(), op_def, OperationType::ADD,
                                src_tensor_1.shape, dst_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor_0, src_tensor_1},
      std::make_unique<GPUOperation>(std::move(operation)), dst_shape,
      &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {1.5f, 3.0f, 6.0f, 6.0f}));
  return absl::OkStatus();
}

absl::Status AddWithConstantBHWCTensorTest(TestExecutionEnvironment& env,
                                           DataType data_type,
                                           TensorStorageType storage) {
  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {1.0f, 2.0f, 3.0f, 4.5f};

  TensorFloat32 const_tensor;
  const_tensor.shape = BHWC(1, 2, 1, 2);
  const_tensor.data = {0.5f, 1.0f, 3.0f, 1.5f};
  ElementwiseAttributes attr;
  attr.param = const_tensor;

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwise(env.GetGpuInfo(), op_def, OperationType::ADD, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor_0, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {1.5f, 3.0f, 6.0f, 6.0f}));
  return absl::OkStatus();
}

absl::Status AddTiledTest(TestExecutionEnvironment& env,
                          DataType data_type,
                          TensorStorageType storage) {
  TensorFloat32 src_tensor_0, src_tensor_1;

  for (auto axis : {Axis::WIDTH, Axis::HEIGHT, Axis::CHANNELS}) {
    src_tensor_0.shape = BHWC(1, 1, 1, 1);
    src_tensor_1.shape = BHWC(1, 1, 1, 1);
    src_tensor_0.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    src_tensor_1.data = {0.5f, 1.0f, 1.5f, 2.0f};

    src_tensor_0.shape.set(axis, 8);
    src_tensor_1.shape.set(axis, 4);
    BHWC dst_shape = src_tensor_0.shape;
    const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
    OperationDef op_def;
    op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
    op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
    op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
    TensorFloat32 dst_tensor;
    GPUOperation operation =
        CreateElementwiseTwoInput(env.GetGpuInfo(), op_def, OperationType::ADD,
                                  src_tensor_1.shape, dst_shape);
    ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
        {src_tensor_0, src_tensor_1},
        std::make_unique<GPUOperation>(std::move(operation)), dst_shape,
        &dst_tensor));
    EXPECT_THAT(dst_tensor.data,
                Pointwise(FloatNear(eps),
                          {1.5f, 3.0f, 4.5f, 6.0f, 5.5f, 7.0f, 8.5f, 10.0f}));
  }

  return absl::OkStatus();
}

absl::Status Atan2Test(TestExecutionEnvironment& env,
                       DataType data_type,
                       TensorStorageType storage) {
  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {1.0f, 2.0f, 3.0f, 4.0f};
  src_tensor_1.data = {0.5f, 1.0f, 3.0f, 3.5f};
  BHWC dst_shape = BHWC(1, 2, 1, 2);

  const float eps = data_type == DataType::FLOAT32 ? 1e-5f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwiseTwoInput(env.GetGpuInfo(), op_def, OperationType::ATAN2,
                                src_tensor_1.shape, dst_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor_0, src_tensor_1},
      std::make_unique<GPUOperation>(std::move(operation)), dst_shape,
      &dst_tensor));
  std::vector<float> expected(src_tensor_0.data.size());
  for (int i = 0; i < src_tensor_0.data.size(); ++i) {
    expected[i] = atan2(src_tensor_0.data[i], src_tensor_1.data[i]);
  }
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), expected));
  return absl::OkStatus();
}

absl::Status Atan2IntTest(TestExecutionEnvironment& env,
                          TensorStorageType storage) {
  TensorInt32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {1, 2, 3, 4};
  src_tensor_1.data = {1, 1, 3, 4};

  auto data_type = DataType::INT32;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

  GPUOperation operation =
      CreateElementwiseTwoInput(env.GetGpuInfo(), op_def, OperationType::ATAN2,
                                src_tensor_1.shape, src_tensor_0.shape);

  TensorDescriptor src_desc0, src_desc1, dst_desc;
  src_desc0 = op_def.src_tensors[0];
  src_desc0.UploadData(src_tensor_0);
  src_desc1 = op_def.src_tensors[1];
  src_desc1.UploadData(src_tensor_1);
  dst_desc.SetBHWCShape(src_tensor_0.shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc0, &src_desc1}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  TensorInt32 dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  std::vector<int32_t> expected(src_tensor_0.data.size());
  for (int i = 0; i < src_tensor_0.data.size(); ++i) {
    expected[i] = round(atan2(src_tensor_0.data[i], src_tensor_1.data[i]));
  }
  EXPECT_EQ(dst_tensor.data, expected);

  return absl::OkStatus();
}

absl::Status MaximumTest(TestExecutionEnvironment& env,
                         DataType data_type,
                         TensorStorageType storage) {
  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {0.0f, -6.2f, 2.0f, -3.0f};
  src_tensor_1.data = {1.0f, 2.0f, 3.0f, -2.0f};
  BHWC dst_shape = BHWC(1, 2, 1, 2);

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateElementwiseTwoInput(
      env.GetGpuInfo(), op_def, OperationType::MAXIMUM, src_tensor_1.shape,
      dst_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor_0, src_tensor_1},
      std::make_unique<GPUOperation>(std::move(operation)), dst_shape,
      &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {1.0f, 2.0f, 3.0f, -2.0f}));
  return absl::OkStatus();
}

absl::Status MaximumInt8Test(TestExecutionEnvironment& env,
                             TensorStorageType storage) {
  Tensor<BHWC, DataType::INT8> src_tensor_0_si8, src_tensor_1_si8;
  src_tensor_0_si8.shape = BHWC(1, 2, 1, 2);
  src_tensor_1_si8.shape = BHWC(1, 2, 1, 2);
  src_tensor_0_si8.data = {0, 1, 8, -9};
  src_tensor_1_si8.data = {-128, -1, 8, -9};
  BHWC dst_shape_si8 = BHWC(1, 2, 1, 2);

  Tensor<BHWC, DataType::INT8> ref_tensor_si8;
  ref_tensor_si8.shape = BHWC(1, 2, 1, 2);
  ref_tensor_si8.data = {0, 1, 8, -9};
  BHWC dst_shape = BHWC(1, 2, 1, 2);

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::INT8, storage, Layout::HWC});
  op_def.src_tensors.push_back({DataType::INT8, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::INT8, storage, Layout::HWC});

  TensorDescriptor src_desc0, src_desc1, dst_desc;
  src_desc0 = op_def.src_tensors[0];
  src_desc0.UploadData(src_tensor_0_si8);
  src_desc1 = op_def.src_tensors[1];
  src_desc1.UploadData(src_tensor_1_si8);
  dst_desc.SetBHWCShape(dst_shape);
  GPUOperation operation = CreateElementwiseTwoInput(
      env.GetGpuInfo(), op_def, OperationType::MAXIMUM, src_tensor_1_si8.shape,
      dst_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc0, &src_desc1}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::INT8> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor_si8.data);

  return absl::OkStatus();
}

absl::Status MaximumWithScalarTest(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage) {
  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 4, 1, 1);
  src_tensor_0.data = {0.0f, -6.2f, 2.0f, -3.0f};

  ElementwiseAttributes attr;
  attr.param = -1.0f;

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwise(env.GetGpuInfo(), op_def, OperationType::MAXIMUM, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor_0, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 4, 1, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {0.0f, -1.0f, 2.0f, -1.0f}));
  return absl::OkStatus();
}

absl::Status MaximumWithIntScalarTest(TestExecutionEnvironment& env,
                                      TensorStorageType storage) {
  Tensor<BHWC, DataType::INT32> src_tensor_0;
  src_tensor_0.shape = BHWC(1, 4, 1, 1);
  src_tensor_0.data = {3, -4, 5, -6};

  Tensor<BHWC, DataType::INT32> ref_tensor;
  ref_tensor.shape = BHWC(1, 4, 1, 1);
  ref_tensor.data = {3, -1, 5, -1};

  ElementwiseAttributes attr;
  const int value = -1;
  attr.param = value;

  const auto data_type = DataType::INT32;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

  TensorDescriptor src_0, dst;
  src_0 = op_def.src_tensors[0];
  src_0.UploadData(src_tensor_0);
  dst.SetBHWCShape(BHWC(1, 4, 1, 1));

  GPUOperation operation =
      CreateElementwise(env.GetGpuInfo(), op_def, OperationType::MAXIMUM, attr);

  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0}, {&dst}, std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::INT32> dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);

  return absl::OkStatus();
}

absl::Status MaximumWithUintScalarTest(TestExecutionEnvironment& env,
                                       TensorStorageType storage) {
  Tensor<BHWC, DataType::UINT32> src_tensor_0;
  src_tensor_0.shape = BHWC(1, 4, 1, 1);
  src_tensor_0.data = {3, 40, 5, 60};

  Tensor<BHWC, DataType::UINT32> ref_tensor;
  ref_tensor.shape = BHWC(1, 4, 1, 1);
  ref_tensor.data = {8, 40, 8, 60};

  ElementwiseAttributes attr;
  const uint value = 8;
  attr.param = value;

  const auto data_type = DataType::UINT32;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

  TensorDescriptor src_0, dst;
  src_0 = op_def.src_tensors[0];
  src_0.UploadData(src_tensor_0);
  dst.SetBHWCShape(BHWC(1, 4, 1, 1));

  GPUOperation operation =
      CreateElementwise(env.GetGpuInfo(), op_def, OperationType::MAXIMUM, attr);

  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0}, {&dst}, std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::UINT32> dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);

  return absl::OkStatus();
}

absl::Status MaximumWithConstantLinearTensorTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage) {
  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {1.0f, -6.2f, -2.0f, 3.0f};

  Tensor<Linear, DataType::FLOAT32> linear_tensor;
  linear_tensor.shape = Linear(2);
  linear_tensor.data = {0.5f, 2.0f};
  ElementwiseAttributes attr;
  attr.param = linear_tensor;

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwise(env.GetGpuInfo(), op_def, OperationType::MAXIMUM, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor_0, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {1.0f, 2.0f, 0.5f, 3.0f}));

  return absl::OkStatus();
}

absl::Status MaximumWithConstantBHWCTensorTest(TestExecutionEnvironment& env,
                                               DataType data_type,
                                               TensorStorageType storage) {
  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {1.0f, -6.2f, -2.0f, 3.0f};

  TensorFloat32 const_tensor;
  const_tensor.shape = BHWC(1, 2, 1, 2);
  const_tensor.data = {0.5f, 2.0f, 0.7f, 4.7f};
  ElementwiseAttributes attr;
  attr.param = const_tensor;

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwise(env.GetGpuInfo(), op_def, OperationType::MAXIMUM, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor_0, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {1.0f, 2.0f, 0.7f, 4.7f}));
  return absl::OkStatus();
}
absl::Status MaximumWithConstantBHWCTensorBroadcastChannelsTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage) {
  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {1.0f, -6.2f, -2.0f, 3.0f};

  TensorFloat32 const_tensor;
  const_tensor.shape = BHWC(1, 2, 1, 1);
  const_tensor.data = {0.5f, 2.0f};
  ElementwiseAttributes attr;
  attr.param = const_tensor;

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwise(env.GetGpuInfo(), op_def, OperationType::MAXIMUM, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor_0, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {1.0f, 0.5f, 2.0f, 3.0f}));
  return absl::OkStatus();
}

absl::Status MinimumTest(TestExecutionEnvironment& env,
                         DataType data_type,
                         TensorStorageType storage) {
  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {0.0f, -6.2f, 2.0f, -3.0f};
  src_tensor_1.data = {1.0f, 2.0f, 3.0f, -2.0f};
  BHWC dst_shape = BHWC(1, 2, 1, 2);

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateElementwiseTwoInput(
      env.GetGpuInfo(), op_def, OperationType::MINIMUM, src_tensor_1.shape,
      dst_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor_0, src_tensor_1},
      std::make_unique<GPUOperation>(std::move(operation)), dst_shape,
      &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {0.0f, -6.2f, 2.0f, -3.0f}));
  return absl::OkStatus();
}

absl::Status MinimumInt8Test(TestExecutionEnvironment& env,
                             TensorStorageType storage) {
  Tensor<BHWC, DataType::INT8> src_tensor_0_si8, src_tensor_1_si8;
  src_tensor_0_si8.shape = BHWC(1, 2, 1, 2);
  src_tensor_1_si8.shape = BHWC(1, 2, 1, 2);
  src_tensor_0_si8.data = {0, 1, 8, -9};
  src_tensor_1_si8.data = {-128, -1, 8, -9};
  BHWC dst_shape_si8 = BHWC(1, 2, 1, 2);

  Tensor<BHWC, DataType::INT8> ref_tensor_si8;
  ref_tensor_si8.shape = BHWC(1, 2, 1, 2);
  ref_tensor_si8.data = {-128, -1, 8, -9};
  BHWC dst_shape = BHWC(1, 2, 1, 2);

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::INT8, storage, Layout::HWC});
  op_def.src_tensors.push_back({DataType::INT8, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::INT8, storage, Layout::HWC});

  TensorDescriptor src_desc0, src_desc1, dst_desc;
  src_desc0 = op_def.src_tensors[0];
  src_desc0.UploadData(src_tensor_0_si8);
  src_desc1 = op_def.src_tensors[1];
  src_desc1.UploadData(src_tensor_1_si8);
  dst_desc.SetBHWCShape(dst_shape);
  GPUOperation operation = CreateElementwiseTwoInput(
      env.GetGpuInfo(), op_def, OperationType::MINIMUM, src_tensor_1_si8.shape,
      dst_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc0, &src_desc1}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::INT8> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor_si8.data);

  return absl::OkStatus();
}

absl::Status MinimumWithScalarTest(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage) {
  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 4, 1, 1);
  src_tensor_0.data = {0.0f, -6.2f, 2.0f, -3.0f};

  ElementwiseAttributes attr;
  attr.param = -1.0f;

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwise(env.GetGpuInfo(), op_def, OperationType::MINIMUM, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor_0, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 4, 1, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {-1.0f, -6.2f, -1.0f, -3.0f}));

  return absl::OkStatus();
}

absl::Status MulTest(TestExecutionEnvironment& env,
                     DataType data_type,
                     TensorStorageType storage) {
  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {1.0f, 2.0f, 3.0f, 4.5f};
  src_tensor_1.data = {0.5f, 1.0f, 3.0f, 1.5f};
  BHWC dst_shape = BHWC(1, 2, 1, 2);

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwiseTwoInput(env.GetGpuInfo(), op_def, OperationType::MUL,
                                src_tensor_1.shape, dst_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor_0, src_tensor_1},
      std::make_unique<GPUOperation>(std::move(operation)), dst_shape,
      &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {0.5f, 2.0f, 9.0f, 6.75f}));

  return absl::OkStatus();
}

absl::Status MulBroadcastHWTest(TestExecutionEnvironment& env,
                                DataType data_type,
                                TensorStorageType storage) {
  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 1, 1, 2);
  src_tensor_0.data = {1.0f, 2.0f, 3.0f, 4.5f};
  src_tensor_1.data = {0.5f, 3.0f};
  BHWC dst_shape = BHWC(1, 2, 1, 2);

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwiseTwoInput(env.GetGpuInfo(), op_def, OperationType::MUL,
                                src_tensor_1.shape, dst_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor_0, src_tensor_1},
      std::make_unique<GPUOperation>(std::move(operation)), dst_shape,
      &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {0.5f, 6.0f, 1.5f, 13.5f}));
  return absl::OkStatus();
}

absl::Status MulBroadcastChannelsTest(TestExecutionEnvironment& env,
                                      DataType data_type,
                                      TensorStorageType storage) {
  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 1);
  src_tensor_0.data = {1.0f, 2.0f, 3.0f, 4.5f};
  src_tensor_1.data = {0.5f, 3.0f};
  BHWC dst_shape = BHWC(1, 2, 1, 2);

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwiseTwoInput(env.GetGpuInfo(), op_def, OperationType::MUL,
                                src_tensor_1.shape, dst_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor_0, src_tensor_1},
      std::make_unique<GPUOperation>(std::move(operation)), dst_shape,
      &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {0.5f, 1.0f, 9.0f, 13.5f}));
  return absl::OkStatus();
}

absl::Status SubWithScalarAtFirstPositionTest(TestExecutionEnvironment& env,
                                              DataType data_type,
                                              TensorStorageType storage) {
  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 4, 1, 1);
  src_tensor_0.data = {0.0f, -6.2f, 2.0f, -3.0f};

  ElementwiseAttributes attr;
  attr.param = 4.0f;
  attr.runtime_tensor_is_second = true;

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwise(env.GetGpuInfo(), op_def, OperationType::SUB, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor_0, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 4, 1, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {4.0f, 10.2f, 2.0f, 7.0f}));
  return absl::OkStatus();
}

absl::Status LessTest(TestExecutionEnvironment& env,
                      TensorStorageType storage) {
  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {0.0f, 1.0f, 2.0f, 3.0f};
  src_tensor_1.data = {1.0f, 0.0f, 2.0f, -4.0f};
  BHWC dst_shape = BHWC(1, 2, 1, 2);

  Tensor<BHWC, DataType::BOOL> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {true, false, false, false};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::FLOAT32, storage, Layout::HWC});
  op_def.src_tensors.push_back({DataType::FLOAT32, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::BOOL, storage, Layout::HWC});

  TensorDescriptor src_desc0, src_desc1, dst_desc;
  src_desc0 = op_def.src_tensors[0];
  src_desc0.UploadData(src_tensor_0);
  src_desc1 = op_def.src_tensors[1];
  src_desc1.UploadData(src_tensor_1);
  dst_desc.SetBHWCShape(dst_shape);
  GPUOperation operation =
      CreateElementwiseTwoInput(env.GetGpuInfo(), op_def, OperationType::LESS,
                                src_tensor_1.shape, dst_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc0, &src_desc1}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::BOOL> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);

  return absl::OkStatus();
}

absl::Status LessEqualTest(TestExecutionEnvironment& env,
                           TensorStorageType storage) {
  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {0.0f, 1.0f, 2.0f, 3.0f};

  ml_drift::Tensor<BHWC, DataType::BOOL> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {true, true, true, false};

  ElementwiseAttributes attr;
  attr.param = 2.0f;

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::FLOAT32, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::BOOL, storage, Layout::HWC});
  TensorDescriptor src_desc, dst_desc;
  src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src_tensor_0);
  dst_desc.SetBHWCShape(BHWC(1, 2, 1, 2));
  GPUOperation operation = CreateElementwise(env.GetGpuInfo(), op_def,
                                             OperationType::LESS_EQUAL, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  ml_drift::Tensor<BHWC, DataType::BOOL> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);

  return absl::OkStatus();
}

absl::Status GreaterTest(TestExecutionEnvironment& env,
                         TensorStorageType storage) {
  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {0.0f, 1.0f, 2.0f, 3.0f};

  ml_drift::Tensor<BHWC, DataType::BOOL> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {false, false, false, true};

  ElementwiseAttributes attr;
  attr.param = 2.0f;

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::FLOAT32, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::BOOL, storage, Layout::HWC});
  TensorDescriptor src_desc, dst_desc;
  src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src_tensor_0);
  dst_desc.SetBHWCShape(BHWC(1, 2, 1, 2));
  GPUOperation operation =
      CreateElementwise(env.GetGpuInfo(), op_def, OperationType::GREATER, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  ml_drift::Tensor<BHWC, DataType::BOOL> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);

  return absl::OkStatus();
}

absl::Status GreaterEqualTest(TestExecutionEnvironment& env,
                              TensorStorageType storage) {
  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {0.0f, 1.0f, 2.0f, 3.0f};

  ml_drift::Tensor<BHWC, DataType::BOOL> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {false, false, true, true};

  ElementwiseAttributes attr;
  attr.param = 2.0f;

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::FLOAT32, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::BOOL, storage, Layout::HWC});
  TensorDescriptor src_desc, dst_desc;
  src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src_tensor_0);
  dst_desc.SetBHWCShape(BHWC(1, 2, 1, 2));
  GPUOperation operation = CreateElementwise(
      env.GetGpuInfo(), op_def, OperationType::GREATER_EQUAL, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  ml_drift::Tensor<BHWC, DataType::BOOL> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);

  return absl::OkStatus();
}

absl::Status EqualTest(TestExecutionEnvironment& env,
                       TensorStorageType storage) {
  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {0.0f, 1.0f, 2.0f, 3.0f};

  ml_drift::Tensor<BHWC, DataType::BOOL> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {false, false, true, false};

  ElementwiseAttributes attr;
  attr.param = 2.0f;

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::FLOAT32, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::BOOL, storage, Layout::HWC});
  TensorDescriptor src_desc, dst_desc;
  src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src_tensor_0);
  dst_desc.SetBHWCShape(BHWC(1, 2, 1, 2));
  GPUOperation operation =
      CreateElementwise(env.GetGpuInfo(), op_def, OperationType::EQUAL, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  ml_drift::Tensor<BHWC, DataType::BOOL> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);

  return absl::OkStatus();
}

absl::Status NotEqualTest(TestExecutionEnvironment& env,
                          TensorStorageType storage) {
  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {0.0f, 1.0f, 2.0f, 3.0f};

  ml_drift::Tensor<BHWC, DataType::BOOL> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {true, true, false, true};

  ElementwiseAttributes attr;
  attr.param = 2.0f;

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::FLOAT32, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::BOOL, storage, Layout::HWC});
  TensorDescriptor src_desc, dst_desc;
  src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src_tensor_0);
  dst_desc.SetBHWCShape(BHWC(1, 2, 1, 2));
  GPUOperation operation = CreateElementwise(env.GetGpuInfo(), op_def,
                                             OperationType::NOT_EQUAL, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  ml_drift::Tensor<BHWC, DataType::BOOL> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);

  return absl::OkStatus();
}

absl::Status CosBroadcastTest(TestExecutionEnvironment& env,
                              DataType data_type,
                              TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 1);
  src_tensor.data = {0.7f, -1.5f};

  const float eps = data_type == DataType::FLOAT32 ? 5e-5f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  BHWC output_shape(1, 2, 1, 2);
  GPUOperation operation = CreateElementwiseOneInputWithBroadcast(
      env.GetGpuInfo(), op_def, OperationType::COS, src_tensor.shape,
      output_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      output_shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {std::cos(0.7f), std::cos(0.7f),
                                         std::cos(-1.5f), std::cos(-1.5f)}));
  return absl::OkStatus();
}

absl::Status MaximumScalarBroadcastInputTest(TestExecutionEnvironment& env,
                                             DataType data_type,
                                             TensorStorageType storage) {
  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 1);
  src_tensor_0.data = {2.0f, -3.0f};

  ElementwiseAttributes attr;
  attr.param = -2.0f;

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  BHWC output_shape(1, 2, 1, 2);
  GPUOperation operation = CreateElementwiseWithBroadcast(
      env.GetGpuInfo(), op_def, OperationType::MAXIMUM, attr,
      src_tensor_0.shape, output_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor_0, std::make_unique<GPUOperation>(std::move(operation)),
      output_shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {2.0f, 2.0f, -2.0f, -2.0f}));

  return absl::OkStatus();
}

absl::Status MulLinearBroadcastInputTest(TestExecutionEnvironment& env,
                                         DataType data_type,
                                         TensorStorageType storage) {
  TensorFloat32 src_tensor_0;
  src_tensor_0.shape = BHWC(1, 2, 1, 1);
  src_tensor_0.data = {2.0f, -3.0f};

  Tensor<Linear, DataType::FLOAT32> linear_tensor;
  linear_tensor.shape = Linear(2);
  linear_tensor.data = {0.5f, 2.0f};
  ElementwiseAttributes attr;
  attr.param = linear_tensor;

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  BHWC output_shape(1, 2, 1, 2);
  GPUOperation operation = CreateElementwiseWithBroadcast(
      env.GetGpuInfo(), op_def, OperationType::MUL, attr, src_tensor_0.shape,
      output_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor_0, std::make_unique<GPUOperation>(std::move(operation)),
      output_shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {1.0f, 4.0f, -1.5f, -6.0f}));

  return absl::OkStatus();
}

absl::Status MulBroadcastBothInputsTest(TestExecutionEnvironment& env,
                                        DataType data_type,
                                        TensorStorageType storage) {
  TensorFloat32 src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 1, 2, 1);
  src_tensor_1.shape = BHWC(1, 1, 1, 2);
  src_tensor_0.data = {1.0f, 2.0f};
  src_tensor_1.data = {3.0f, 4.0f};
  ElementwiseAttributes attr;

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  BHWC output_shape(1, 1, 2, 2);
  GPUOperation operation = CreateElementwiseTwoInputWithBroadcast(
      env.GetGpuInfo(), op_def, OperationType::MUL, src_tensor_0.shape,
      src_tensor_1.shape, output_shape, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor_0, src_tensor_1},
      std::make_unique<GPUOperation>(std::move(operation)), output_shape,
      &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {3.0f, 4.0f, 6.0f, 8.0f}));

  return absl::OkStatus();
}

absl::Status MishTest(TestExecutionEnvironment& env,
                      DataType data_type,
                      TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 2);
  src_tensor.data = {
      -4.0f, -0.1f, 0.1f, 2.0f, -15.0f, -8.0f, 4.0f, 9.0f,
  };

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 5e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateElementwiseOneInput(env.GetGpuInfo(), op_def, OperationType::MISH);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 2, 2), &dst_tensor));

  auto mish = [](double x) -> float {
    return x * std::tanh(std::log(std::exp(x) + 1));
  };
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), {
                                                             mish(-4.0f),
                                                             mish(-0.1f),
                                                             mish(0.1f),
                                                             mish(2.0f),
                                                             mish(-15.0f),
                                                             mish(-8.0f),
                                                             mish(4.0f),
                                                             mish(9.0f),
                                                         }));
  return absl::OkStatus();
}

absl::Status LogicalAndTest(TestExecutionEnvironment& env,
                            TensorStorageType storage) {
  Tensor<BHWC, DataType::BOOL> src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {true, true, false, false};
  src_tensor_1.data = {true, false, true, false};
  BHWC dst_shape = BHWC(1, 2, 1, 2);

  Tensor<BHWC, DataType::BOOL> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {true, false, false, false};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::BOOL, storage, Layout::HWC});
  op_def.src_tensors.push_back({DataType::BOOL, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::BOOL, storage, Layout::HWC});

  TensorDescriptor src_desc0, src_desc1, dst_desc;
  src_desc0 = op_def.src_tensors[0];
  src_desc0.UploadData(src_tensor_0);
  src_desc1 = op_def.src_tensors[1];
  src_desc1.UploadData(src_tensor_1);
  dst_desc.SetBHWCShape(dst_shape);
  GPUOperation operation = CreateElementwiseTwoInput(
      env.GetGpuInfo(), op_def, OperationType::LOGICAL_AND, src_tensor_1.shape,
      dst_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc0, &src_desc1}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::BOOL> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

absl::Status LogicalAndInt8Test(TestExecutionEnvironment& env,
                                TensorStorageType storage) {
  Tensor<BHWC, DataType::INT8> src_tensor_0_si8, src_tensor_1_si8;
  src_tensor_0_si8.shape = BHWC(1, 2, 1, 2);
  src_tensor_1_si8.shape = BHWC(1, 2, 1, 2);
  src_tensor_0_si8.data = {127, -128, -128, 7};
  src_tensor_1_si8.data = {0, 127, -128, 15};
  BHWC dst_shape_si8 = BHWC(1, 2, 1, 2);

  Tensor<BHWC, DataType::INT8> ref_tensor_si8;
  ref_tensor_si8.shape = BHWC(1, 2, 1, 2);
  ref_tensor_si8.data = {0, 0, -128, 7};
  BHWC dst_shape = BHWC(1, 2, 1, 2);

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::INT8, storage, Layout::HWC});
  op_def.src_tensors.push_back({DataType::INT8, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::INT8, storage, Layout::HWC});

  TensorDescriptor src_desc0, src_desc1, dst_desc;
  src_desc0 = op_def.src_tensors[0];
  src_desc0.UploadData(src_tensor_0_si8);
  src_desc1 = op_def.src_tensors[1];
  src_desc1.UploadData(src_tensor_1_si8);
  dst_desc.SetBHWCShape(dst_shape);
  GPUOperation operation = CreateElementwiseTwoInput(
      env.GetGpuInfo(), op_def, OperationType::LOGICAL_AND,
      src_tensor_1_si8.shape, dst_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc0, &src_desc1}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::INT8> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor_si8.data);

  return absl::OkStatus();
}

absl::Status LogicalOrTest(TestExecutionEnvironment& env,
                           TensorStorageType storage) {
  Tensor<BHWC, DataType::BOOL> src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {true, true, false, false};
  src_tensor_1.data = {true, false, true, false};
  BHWC dst_shape = BHWC(1, 2, 1, 2);

  Tensor<BHWC, DataType::BOOL> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {true, true, true, false};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::BOOL, storage, Layout::HWC});
  op_def.src_tensors.push_back({DataType::BOOL, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::BOOL, storage, Layout::HWC});

  TensorDescriptor src_desc0, src_desc1, dst_desc;
  src_desc0 = op_def.src_tensors[0];
  src_desc0.UploadData(src_tensor_0);
  src_desc1 = op_def.src_tensors[1];
  src_desc1.UploadData(src_tensor_1);
  dst_desc.SetBHWCShape(dst_shape);
  GPUOperation operation = CreateElementwiseTwoInput(
      env.GetGpuInfo(), op_def, OperationType::LOGICAL_OR, src_tensor_1.shape,
      dst_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc0, &src_desc1}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::BOOL> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

absl::Status LogicalOrInt8Test(TestExecutionEnvironment& env,
                               TensorStorageType storage) {
  Tensor<BHWC, DataType::INT8> src_tensor_0_si8, src_tensor_1_si8;
  src_tensor_0_si8.shape = BHWC(1, 2, 1, 2);
  src_tensor_1_si8.shape = BHWC(1, 2, 1, 2);
  src_tensor_0_si8.data = {127, -128, -128, 7};
  src_tensor_1_si8.data = {0, 127, -128, 15};
  BHWC dst_shape_si8 = BHWC(1, 2, 1, 2);

  Tensor<BHWC, DataType::INT8> ref_tensor_si8;
  ref_tensor_si8.shape = BHWC(1, 2, 1, 2);
  ref_tensor_si8.data = {127, -1, -128, 15};
  BHWC dst_shape = BHWC(1, 2, 1, 2);

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::INT8, storage, Layout::HWC});
  op_def.src_tensors.push_back({DataType::INT8, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::INT8, storage, Layout::HWC});

  TensorDescriptor src_desc0, src_desc1, dst_desc;
  src_desc0 = op_def.src_tensors[0];
  src_desc0.UploadData(src_tensor_0_si8);
  src_desc1 = op_def.src_tensors[1];
  src_desc1.UploadData(src_tensor_1_si8);
  dst_desc.SetBHWCShape(dst_shape);
  GPUOperation operation = CreateElementwiseTwoInput(
      env.GetGpuInfo(), op_def, OperationType::LOGICAL_OR,
      src_tensor_1_si8.shape, dst_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc0, &src_desc1}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::INT8> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor_si8.data);

  return absl::OkStatus();
}

absl::Status LogicalNotTest(TestExecutionEnvironment& env,
                            TensorStorageType storage) {
  Tensor<BHWC, DataType::BOOL> src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {true, true, false, false};
  BHWC dst_shape = BHWC(1, 2, 1, 2);

  Tensor<BHWC, DataType::BOOL> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {false, false, true, true};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::BOOL, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::BOOL, storage, Layout::HWC});

  TensorDescriptor src_desc, dst_desc;
  src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src_tensor);
  dst_desc.SetBHWCShape(dst_shape);
  GPUOperation operation = CreateElementwiseOneInput(
      env.GetGpuInfo(), op_def, OperationType::LOGICAL_NOT);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::BOOL> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

absl::Status LogicalNotInt8Test(TestExecutionEnvironment& env,
                                TensorStorageType storage) {
  Tensor<BHWC, DataType::INT8> src_tensor_si8;
  src_tensor_si8.shape = BHWC(1, 2, 1, 2);
  src_tensor_si8.data = {127, -128, 0, 7};
  BHWC dst_shape_si8 = BHWC(1, 2, 1, 2);

  Tensor<BHWC, DataType::INT8> ref_tensor_si8;
  ref_tensor_si8.shape = BHWC(1, 2, 1, 2);
  ref_tensor_si8.data = {-128, 127, -1, -8};
  BHWC dst_shape = BHWC(1, 2, 1, 2);

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::INT8, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::INT8, storage, Layout::HWC});

  TensorDescriptor src_desc, dst_desc;
  src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src_tensor_si8);
  dst_desc.SetBHWCShape(dst_shape);
  GPUOperation operation = CreateElementwiseOneInput(
      env.GetGpuInfo(), op_def, OperationType::LOGICAL_NOT);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::INT8> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor_si8.data);

  return absl::OkStatus();
}

absl::Status LogicalXorTest(TestExecutionEnvironment& env,
                            TensorStorageType storage) {
  Tensor<BHWC, DataType::BOOL> src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWC(1, 2, 1, 2);
  src_tensor_1.shape = BHWC(1, 2, 1, 2);
  src_tensor_0.data = {true, true, false, false};
  src_tensor_1.data = {true, false, true, false};
  BHWC dst_shape = BHWC(1, 2, 1, 2);

  Tensor<BHWC, DataType::BOOL> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {false, true, true, false};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::BOOL, storage, Layout::HWC});
  op_def.src_tensors.push_back({DataType::BOOL, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::BOOL, storage, Layout::HWC});

  TensorDescriptor src_desc0, src_desc1, dst_desc;
  src_desc0 = op_def.src_tensors[0];
  src_desc0.UploadData(src_tensor_0);
  src_desc1 = op_def.src_tensors[1];
  src_desc1.UploadData(src_tensor_1);
  dst_desc.SetBHWCShape(dst_shape);
  GPUOperation operation = CreateElementwiseTwoInput(
      env.GetGpuInfo(), op_def, OperationType::LOGICAL_XOR, src_tensor_1.shape,
      dst_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc0, &src_desc1}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::BOOL> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

absl::Status LogicalXorInt8Test(TestExecutionEnvironment& env,
                                TensorStorageType storage) {
  Tensor<BHWC, DataType::INT8> src_tensor_0_si8, src_tensor_1_si8;
  src_tensor_0_si8.shape = BHWC(1, 2, 1, 2);
  src_tensor_1_si8.shape = BHWC(1, 2, 1, 2);
  src_tensor_0_si8.data = {127, -128, -128, 7};
  src_tensor_1_si8.data = {0, 127, -128, 15};
  BHWC dst_shape_si8 = BHWC(1, 2, 1, 2);

  Tensor<BHWC, DataType::INT8> ref_tensor_si8;
  ref_tensor_si8.shape = BHWC(1, 2, 1, 2);
  ref_tensor_si8.data = {127, -1, 0, 8};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::INT8, storage, Layout::HWC});
  op_def.src_tensors.push_back({DataType::INT8, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::INT8, storage, Layout::HWC});

  TensorDescriptor src_desc0, src_desc1, dst_desc;
  src_desc0 = op_def.src_tensors[0];
  src_desc0.UploadData(src_tensor_0_si8);
  src_desc1 = op_def.src_tensors[1];
  src_desc1.UploadData(src_tensor_1_si8);
  dst_desc.SetBHWCShape(dst_shape_si8);
  GPUOperation operation = CreateElementwiseTwoInput(
      env.GetGpuInfo(), op_def, OperationType::LOGICAL_XOR,
      src_tensor_1_si8.shape, dst_shape_si8);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc0, &src_desc1}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::INT8> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor_si8.data);

  return absl::OkStatus();
}

absl::Status Add5DTest(TestExecutionEnvironment& env, DataType data_type,
                       TensorStorageType storage) {
  Tensor<BHWDC, DataType::FLOAT32> src_tensor_0, src_tensor_1;
  src_tensor_0.shape = BHWDC(1, 2, 1, 2, 2);
  src_tensor_1.shape = BHWDC(1, 2, 1, 2, 2);
  src_tensor_0.data = {1.0f, 2.0f, 3.0f, 4.5f, 1.0f, 2.0f, 3.0f, 4.5f};
  src_tensor_1.data = {0.5f, 1.0f, 3.0f, 1.5f, 0.5f, 1.0f, 3.0f, 1.5f};
  BHWDC dst_shape = BHWDC(1, 2, 1, 2, 2);

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWDC});
  Tensor<BHWDC, DataType::FLOAT32> dst_tensor;
  GPUOperation operation =
      CreateElementwiseTwoInput(env.GetGpuInfo(), op_def, OperationType::ADD,
                                src_tensor_1.shape, dst_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor_0, src_tensor_1},
      std::make_unique<GPUOperation>(std::move(operation)), dst_shape,
      &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps),
                        {1.5f, 3.0f, 6.0f, 6.0f, 1.5f, 3.0f, 6.0f, 6.0f}));
  return absl::OkStatus();
}

absl::Status OneInputWithBroadcast5DTest(TestExecutionEnvironment& env,
                                         DataType data_type,
                                         TensorStorageType storage) {
  Tensor<BHWDC, DataType::FLOAT32> src_tensor;
  src_tensor.shape = BHWDC(1, 1, 1, 1, 1);
  src_tensor.data = {2.0f};

  BHWDC dst_shape = BHWDC(1, 2, 1, 2, 2);

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWDC});
  Tensor<BHWDC, DataType::FLOAT32> dst_tensor;
  GPUOperation operation = CreateElementwiseOneInputWithBroadcast(
      env.GetGpuInfo(), op_def, OperationType::COS, src_tensor.shape,
      dst_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      dst_shape, &dst_tensor));
  EXPECT_THAT(
      dst_tensor.data,
      Pointwise(FloatNear(eps), {std::cos(2.0f), std::cos(2.0f), std::cos(2.0f),
                                 std::cos(2.0f), std::cos(2.0f), std::cos(2.0f),
                                 std::cos(2.0f), std::cos(2.0f)}));
  return absl::OkStatus();
}

absl::Status WithBroadcast5DTest(TestExecutionEnvironment& env,
                                 DataType data_type,
                                 TensorStorageType storage) {
  Tensor<BHWDC, DataType::FLOAT32> src_tensor;
  src_tensor.shape = BHWDC(1, 2, 1, 2, 2);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  Tensor<BHWDC, DataType::FLOAT32> const_tensor;
  const_tensor.shape = BHWDC(1, 1, 1, 1, 2);
  const_tensor.data = {0.5f, 2.0f};
  ElementwiseAttributes attr;
  attr.param = const_tensor;

  BHWDC dst_shape = BHWDC(1, 2, 1, 2, 2);

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWDC});
  Tensor<BHWDC, DataType::FLOAT32> dst_tensor;
  GPUOperation operation = CreateElementwiseWithBroadcast(
      env.GetGpuInfo(), op_def, OperationType::MUL, attr, src_tensor.shape,
      dst_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      dst_shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps),
                        {0.5f, 4.0f, 1.5f, 8.0f, 2.5f, 12.0f, 3.5f, 16.0f}));
  return absl::OkStatus();
}

absl::Status WithBroadcast5DPaddedGridTest(TestExecutionEnvironment& env,
                                           DataType data_type,
                                           TensorStorageType storage) {
  Tensor<BHWDC, DataType::FLOAT32> src_tensor;
  src_tensor.shape = BHWDC(1, 7, 1, 3, 1);
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = static_cast<float>(i + 1);
  }

  Tensor<BHWDC, DataType::FLOAT32> const_tensor;
  const_tensor.shape = BHWDC(1, 1, 1, 3, 1);
  const_tensor.data = {0.5f, 2.0f, 3.5f};
  ElementwiseAttributes attr;
  attr.param = const_tensor;

  BHWDC dst_shape = BHWDC(1, 7, 1, 3, 1);

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWDC});
  Tensor<BHWDC, DataType::FLOAT32> dst_tensor;
  GPUOperation operation = CreateElementwiseWithBroadcast(
      env.GetGpuInfo(), op_def, OperationType::ADD, attr, src_tensor.shape,
      dst_shape);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      dst_shape, &dst_tensor));

  std::vector<float> expected_data(dst_shape.DimensionsProduct());
  for (int y = 0; y < dst_shape.h; ++y) {
    for (int z = 0; z < dst_shape.d; ++z) {
      expected_data[y * dst_shape.d + z] =
          src_tensor.data[y * dst_shape.d + z] + const_tensor.data[z];
    }
  }

  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), expected_data));
  return absl::OkStatus();
}

absl::Status TwoInputWithBroadcast5DTest(TestExecutionEnvironment& env,
                                         DataType data_type,
                                         TensorStorageType storage) {
  Tensor<BHWDC, DataType::FLOAT32> src_tensor_0;
  src_tensor_0.shape = BHWDC(1, 2, 1, 2, 2);
  src_tensor_0.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  Tensor<BHWDC, DataType::FLOAT32> src_tensor_1;
  src_tensor_1.shape = BHWDC(1, 1, 1, 1, 2);
  src_tensor_1.data = {0.5f, 2.0f};

  BHWDC dst_shape = BHWDC(1, 2, 1, 2, 2);

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWDC});
  Tensor<BHWDC, DataType::FLOAT32> dst_tensor;
  ElementwiseAttributes attr;
  GPUOperation operation = CreateElementwiseTwoInputWithBroadcast(
      env.GetGpuInfo(), op_def, OperationType::MUL, src_tensor_0.shape,
      src_tensor_1.shape, dst_shape, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor_0, src_tensor_1},
      std::make_unique<GPUOperation>(std::move(operation)), dst_shape,
      &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps),
                        {0.5f, 4.0f, 1.5f, 8.0f, 2.5f, 12.0f, 3.5f, 16.0f}));
  return absl::OkStatus();
}

}  // namespace ml_drift
