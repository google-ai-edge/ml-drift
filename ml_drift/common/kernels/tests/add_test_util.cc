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

#include "ml_drift/common/kernels/tests/add_test_util.h"

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/add.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_ref_ops.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace {

absl::Status AddTest(TestExecutionEnvironment& exec_env,
                     const std::vector<TensorFloat32>& src_tensors,
                     const OperationDef& op_def) {
  std::vector<int> channels;
  for (size_t i = 0; i < src_tensors.size(); ++i) {
    channels.push_back(src_tensors[i].shape.c);
  }
  TensorFloat32 dst_ref_tensor = AddTableReference(src_tensors);

  GPUOperation operation = CreateAdd(op_def, channels, channels[0]);
  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_tensors, std::make_unique<GPUOperation>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.004f), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status AddNotEqualTest(TestExecutionEnvironment& exec_env,
                             const PadAttributes& attr,
                             const TensorFloat32& src_tensor0,
                             const TensorFloat32& src_tensor1,
                             const OperationDef& op_def) {
  auto src_shape0 = src_tensor0.shape;
  auto src_shape1 = src_tensor1.shape;
  std::vector<int> channels = {src_shape0.c, src_shape1.c};

  TensorFloat32 tensor1 = PaddingReference(attr, src_tensor1);
  std::vector<TensorFloat32> src_tensors(2);
  src_tensors[0] = src_tensor0;
  src_tensors[1] = tensor1;
  TensorFloat32 dst_ref_tensor = AddTableReference({src_tensor0, tensor1});

  GPUOperation operation = CreateAdd(op_def, channels, src_shape0.c);
  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      {src_tensor0, src_tensor1},
      std::make_unique<GPUOperation>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.004f), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status AddNotEqualFirstTensorTest(TestExecutionEnvironment& exec_env,
                                        const PadAttributes& attr,
                                        const TensorFloat32& src_tensor0,
                                        const TensorFloat32& src_tensor1,
                                        const OperationDef& op_def) {
  auto src_shape0 = src_tensor0.shape;
  auto src_shape1 = src_tensor1.shape;
  std::vector<int> channels = {src_shape0.c, src_shape1.c};

  TensorFloat32 tensor0 = PaddingReference(attr, src_tensor0);
  TensorFloat32 dst_ref_tensor = AddTableReference({tensor0, src_tensor1});

  GPUOperation operation = CreateAdd(op_def, channels, src_shape1.c);
  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      {src_tensor0, src_tensor1},
      std::make_unique<GPUOperation>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.004f), dst_ref_tensor.data));
  return absl::OkStatus();
}

}  // namespace

template <DataType data_type>
absl::Status AddTwoEqualIntTensorsTest(TestExecutionEnvironment& env,
                                       TensorStorageType storage) {
  Tensor<BHWC, data_type> src0, src1;
  src0.shape = BHWC(1, 2, 1, 2);
  src0.data = {3, 4, 5, 6};
  src1.shape = BHWC(1, 2, 1, 2);
  src1.data = {-4, 12, 17, -2};
  std::vector<int> channels = {2, 2};
  Tensor<BHWC, data_type> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {-1, 16, 22, 4};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorDescriptor src_0, src_1, dst;
  src_0 = op_def.src_tensors[0];
  src_1 = op_def.src_tensors[1];
  src_0.UploadData(src0);
  src_1.UploadData(src1);
  dst.SetBHWCShape(BHWC(1, 2, 1, 2));
  GPUOperation operation = CreateAdd(op_def, channels, channels[0]);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0, &src_1}, {&dst},
      std::make_unique<GPUOperation>(std::move(operation))));
  Tensor<BHWC, data_type> dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

template absl::Status AddTwoEqualIntTensorsTest<DataType::INT8>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status AddTwoEqualIntTensorsTest<DataType::INT16>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status AddTwoEqualIntTensorsTest<DataType::INT32>(
    TestExecutionEnvironment& env, TensorStorageType storage);

template <DataType data_type>
absl::Status AddTwoEqualUintTensorsTest(TestExecutionEnvironment& env,
                                        TensorStorageType storage) {
  Tensor<BHWC, data_type> src0, src1;
  src0.shape = BHWC(1, 2, 1, 2);
  src0.data = {3, 4, 5, 6};
  src1.shape = BHWC(1, 2, 1, 2);
  src1.data = {4, 12, 17, 2};
  std::vector<int> channels = {2, 2};
  Tensor<BHWC, data_type> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {7, 16, 22, 8};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorDescriptor src_0, src_1, dst;
  src_0 = op_def.src_tensors[0];
  src_1 = op_def.src_tensors[1];
  src_0.UploadData(src0);
  src_1.UploadData(src1);
  dst.SetBHWCShape(BHWC(1, 2, 1, 2));
  GPUOperation operation = CreateAdd(op_def, channels, channels[0]);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0, &src_1}, {&dst},
      std::make_unique<GPUOperation>(std::move(operation))));
  Tensor<BHWC, data_type> dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

template absl::Status AddTwoEqualUintTensorsTest<DataType::UINT8>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status AddTwoEqualUintTensorsTest<DataType::UINT16>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status AddTwoEqualUintTensorsTest<DataType::UINT32>(
    TestExecutionEnvironment& env, TensorStorageType storage);

absl::Status AddTwoEqualTensorsBFloatTest(TestExecutionEnvironment& env,
                                          TensorStorageType storage) {
  Tensor<BHWC, DataType::BFLOAT16> src0, src1;
  src0.shape = BHWC(1, 2, 1, 2);
  src0.data = FloatToBFloat({1.0f, 2.0f, 16.0625f, 16.1875f});
  src1.shape = BHWC(1, 2, 1, 2);
  src1.data = FloatToBFloat({-20.0f, -4.0f, 0.0f, 0.0f});

  std::vector<int> channels = {2, 2};
  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::BFLOAT16, storage, Layout::HWC});
  op_def.src_tensors.push_back({DataType::BFLOAT16, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::BFLOAT16, storage, Layout::HWC});
  TensorDescriptor src_0, src_1, dst;
  src_0 = op_def.src_tensors[0];
  src_1 = op_def.src_tensors[1];
  src_0.UploadData(src0);
  src_1.UploadData(src1);
  dst.SetBHWCShape(BHWC(1, 2, 1, 2));
  GPUOperation operation = CreateAdd(op_def, channels, channels[0]);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0, &src_1}, {&dst},
      std::make_unique<GPUOperation>(std::move(operation))));
  Tensor<BHWC, DataType::BFLOAT16> dst_tensor;
  dst.DownloadData(&dst_tensor);
  // Not exact values to check for correct rounding. (Float->Bfloat rte)
  EXPECT_THAT(BFloatToFloat(dst_tensor.data),
              Pointwise(FloatNear(1e-3f), {-19.0f, -2.0f, 16.0f, 16.25f}));
  return absl::OkStatus();
}

absl::Status AddTwoEqualTensorsTest(TestExecutionEnvironment& env,
                                    DataType data_type,
                                    TensorStorageType storage) {
  TensorFloat32 src0, src1;
  src0.shape = BHWC(1, 2, 1, 2);
  src0.data = {0.0f, -1.0f, -0.05f, 0.045f};
  src1.shape = BHWC(1, 2, 1, 2);
  src1.data = {0.0f, 1.0f, -0.05f, -0.045f};
  std::vector<int> channels = {2, 2};

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateAdd(op_def, channels, channels[0]);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src0, src1}, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {0.0f, 0.0f, -0.1f, 0.0f}));
  return absl::OkStatus();
}

absl::Status AddFirstTensorHasMoreChannelsThanSecondTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage) {
  TensorFloat32 src0, src1;
  src0.shape = BHWC(1, 2, 1, 6);
  src0.data = {0.0f,   -1.0f,  -0.05f, 0.045f, 1.0f,   -2.0f,
               -1.05f, 1.045f, 2.0f,   -3.0f,  -2.05f, 2.045f};
  src1.shape = BHWC(1, 2, 1, 2);
  src1.data = {0.0f, 1.0f, -0.05f, -0.045f};
  std::vector<int> channels = {6, 2};
  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateAdd(op_def, channels, channels[0]);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src0, src1}, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 6), &dst_tensor));
  EXPECT_THAT(
      dst_tensor.data,
      Pointwise(FloatNear(eps), {0.0f, 0.0f, -0.05f, 0.045f, 1.0f, -2.0f, -1.1f,
                                 1.0f, 2.0f, -3.0f, -2.05f, 2.045f}));
  return absl::OkStatus();
}

absl::Status AddFirstTensorHasLessChannelsThanSecondTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage) {
  TensorFloat32 src0, src1;
  src1.shape = BHWC(1, 2, 1, 6);
  src1.data = {0.0f,   -1.0f,  -0.05f, 0.045f, 1.0f,   -2.0f,
               -1.05f, 1.045f, 2.0f,   -3.0f,  -2.05f, 2.045f};
  src0.shape = BHWC(1, 2, 1, 2);
  src0.data = {0.0f, 1.0f, -0.05f, -0.045f};
  std::vector<int> channels = {2, 6};
  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateAdd(op_def, channels, 6);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src0, src1}, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 6), &dst_tensor));
  EXPECT_THAT(
      dst_tensor.data,
      Pointwise(FloatNear(eps), {0.0f, 0.0f, -0.05f, 0.045f, 1.0f, -2.0f, -1.1f,
                                 1.0f, 2.0f, -3.0f, -2.05f, 2.045f}));
  return absl::OkStatus();
}

absl::Status AddBigTest(TestExecutionEnvironment& env, DataType data_type,
                        TensorStorageType storage) {
  auto src_shape = BHWC(1, 8, 13, 17);
  std::vector<int> channels = {src_shape.c, src_shape.c, src_shape.c};
  std::vector<TensorFloat32> src_tensors = {MakeSyntheticTensor(src_shape),
                                            MakeSyntheticTensor(src_shape),
                                            MakeSyntheticTensor(src_shape)};

  OperationDef op_def;
  for (size_t i = 0; i < src_tensors.size(); ++i) {
    op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  }
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return AddTest(env, src_tensors, op_def);
}

absl::Status AddBatchedBigTest(TestExecutionEnvironment& env,
                               DataType data_type, TensorStorageType storage) {
  auto src_shape = BHWC(3, 8, 6, 5);
  std::vector<int> channels = {src_shape.c, src_shape.c, src_shape.c};
  std::vector<TensorFloat32> src_tensors = {MakeSyntheticTensor(src_shape),
                                            MakeSyntheticTensor(src_shape),
                                            MakeSyntheticTensor(src_shape)};

  OperationDef op_def;
  for (size_t i = 0; i < src_tensors.size(); ++i) {
    op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  }
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  return AddTest(env, src_tensors, op_def);
}

absl::Status AddNotEqualBigTest(TestExecutionEnvironment& env,
                                DataType data_type, TensorStorageType storage) {
  PadAttributes attr;
  attr.prepended = BHWC(0, 0, 0, 0);
  attr.appended = BHWC(0, 0, 0, 4);
  auto src_shape0 = BHWC(1, 8, 6, 12);
  TensorFloat32 src_tensor0 = MakeSyntheticTensor(src_shape0);
  auto src_shape1 = BHWC(1, 8, 6, 8);
  TensorFloat32 src_tensor1 = MakeSyntheticTensor(src_shape1);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return AddNotEqualTest(env, attr, src_tensor0, src_tensor1, op_def);
}

absl::Status AddNotEqualBatchedBigTest(TestExecutionEnvironment& env,
                                       DataType data_type,
                                       TensorStorageType storage) {
  PadAttributes attr;
  attr.prepended = BHWC(0, 0, 0, 0);
  attr.appended = BHWC(0, 0, 0, 4);
  auto src_shape0 = BHWC(7, 8, 6, 12);
  TensorFloat32 src_tensor0 = MakeSyntheticTensor(src_shape0);
  auto src_shape1 = BHWC(7, 8, 6, 8);
  TensorFloat32 src_tensor1 = MakeSyntheticTensor(src_shape1);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  return AddNotEqualTest(env, attr, src_tensor0, src_tensor1, op_def);
}

absl::Status AddNotEqualFirstTensorBigTest(TestExecutionEnvironment& env,
                                           DataType data_type,
                                           TensorStorageType storage) {
  PadAttributes attr;
  attr.prepended = BHWC(0, 0, 0, 0);
  attr.appended = BHWC(0, 0, 0, 4);
  auto src_shape0 = BHWC(1, 8, 6, 8);
  auto src_shape1 = BHWC(1, 8, 6, 12);
  TensorFloat32 src_tensor0 = MakeSyntheticTensor(src_shape0);
  TensorFloat32 src_tensor1 = MakeSyntheticTensor(src_shape1);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return AddNotEqualFirstTensorTest(env, attr, src_tensor0, src_tensor1,
                                    op_def);
}

absl::Status AddNotEqualFirstTensorBatchedBigTest(TestExecutionEnvironment& env,
                                                  DataType data_type,
                                                  TensorStorageType storage) {
  PadAttributes attr;
  attr.prepended = BHWC(0, 0, 0, 0);
  attr.appended = BHWC(0, 0, 0, 4);
  auto src_shape0 = BHWC(6, 8, 6, 8);
  auto src_shape1 = BHWC(6, 8, 6, 12);
  TensorFloat32 src_tensor0 = MakeSyntheticTensor(src_shape0);
  TensorFloat32 src_tensor1 = MakeSyntheticTensor(src_shape1);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  return AddNotEqualFirstTensorTest(env, attr, src_tensor0, src_tensor1,
                                    op_def);
}

absl::Status AddBroadcast5DTest(TestExecutionEnvironment& env,
                                DataType data_type, TensorStorageType storage) {
  Tensor<BHWDC, DataType::FLOAT32> src0;
  src0.shape = BHWDC(1, 2, 1, 2, 2);
  src0.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  Tensor<BHWDC, DataType::FLOAT32> src1;
  src1.shape = BHWDC(1, 2, 1, 2, 4);
  src1.data = {0.5f, 2.0f, 1.0f, 1.5f, 2.0f, 3.0f, 1.0f, 1.0f,
               0.5f, 2.0f, 1.0f, 1.5f, 2.0f, 3.0f, 1.0f, 1.0f};

  Tensor<BHWDC, DataType::FLOAT32> ref_tensor;
  ref_tensor.shape = BHWDC(1, 2, 1, 2, 4);
  ref_tensor.data = {1.5f, 4.0f,  1.0f, 1.5f,   // h=0, d=0  (1.0+0.5, 2.0+2.0)
                     5.0f, 7.0f,  1.0f, 1.0f,   // h=0, d=1  (3.0+2.0, 4.0+3.0)
                     5.5f, 8.0f,  1.0f, 1.5f,   // h=1, d=0  (5.0+0.5, 6.0+2.0)
                     9.0f, 11.0f, 1.0f, 1.0f};  // h=1, d=1 (7.0+2.0, 8.0+3.0)

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWDC});

  std::vector<int> channels = {src0.shape.c, src1.shape.c};
  GPUOperation operation = CreateAdd(op_def, channels, src1.shape.c);

  TensorDescriptor src_0, src_1, dst;
  src_0 = op_def.src_tensors[0];
  src_1 = op_def.src_tensors[1];
  src_0.UploadData(src0);
  src_1.UploadData(src1);
  dst.SetBHWDCShape(BHWDC(1, 2, 1, 2, 4));

  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0, &src_1}, {&dst},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWDC, DataType::FLOAT32> dst_tensor;
  dst.DownloadData(&dst_tensor);

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), ref_tensor.data));
  return absl::OkStatus();
}

}  // namespace ml_drift
