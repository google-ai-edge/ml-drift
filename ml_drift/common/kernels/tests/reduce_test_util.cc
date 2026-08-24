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

#include "ml_drift/common/kernels/tests/reduce_test_util.h"

#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/reduce.h"
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

absl::Status ReduceTest(TestExecutionEnvironment& exec_env,
                        const std::set<Axis>& axis_to_reduce,
                        OperationType op_type, const TensorFloat32& src_tensor,
                        const OperationDef& op_def) {
  TensorFloat32 dst_ref_tensor =
      ReduceReference(axis_to_reduce, op_type, src_tensor);

  Reduce operation = CreateReduce(axis_to_reduce, src_tensor.shape, op_type,
                                  op_def, exec_env.GetGpuInfo());
  float eps =
      GetEpsilon(op_def.src_tensors[0].GetDataType(), exec_env.GetGpuInfo());
  if (op_type == OperationType::REDUCE_SUM) {
    const double total_src_elements = 1.0 * src_tensor.shape.b *
                                      src_tensor.shape.w * src_tensor.shape.h *
                                      src_tensor.shape.c;
    const double total_dst_elements =
        1.0 * dst_ref_tensor.shape.b * dst_ref_tensor.shape.w *
        dst_ref_tensor.shape.h * dst_ref_tensor.shape.c;
    const double reduction_size = total_src_elements / total_dst_elements;
    eps *= reduction_size;
  }
  TensorFloat32 dst_tensor;
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Reduce>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ReduceTest(TestExecutionEnvironment& exec_env,
                        const std::set<Axis>& axis_to_reduce,
                        OperationType op_type,
                        const Tensor5DFloat32& src_tensor,
                        const OperationDef& op_def) {
  Tensor5DFloat32 dst_ref_tensor =
      ReduceReference(axis_to_reduce, op_type, src_tensor);

  Reduce operation = CreateReduce(axis_to_reduce, src_tensor.shape, op_type,
                                  op_def, exec_env.GetGpuInfo());
  float eps =
      GetEpsilon(op_def.src_tensors[0].GetDataType(), exec_env.GetGpuInfo());
  if (op_type == OperationType::REDUCE_SUM) {
    const double total_src_elements = 1.0 * src_tensor.shape.b *
                                      src_tensor.shape.w * src_tensor.shape.h *
                                      src_tensor.shape.d * src_tensor.shape.c;
    const double total_dst_elements =
        1.0 * dst_ref_tensor.shape.b * dst_ref_tensor.shape.w *
        dst_ref_tensor.shape.h * dst_ref_tensor.shape.d *
        dst_ref_tensor.shape.c;
    const double reduction_size = total_src_elements / total_dst_elements;
    eps *= reduction_size;
  }
  Tensor5DFloat32 dst_tensor;
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Reduce>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}
}  // namespace

template <DataType T>
absl::Status ReduceSumChannelsIntTest(TestExecutionEnvironment& env,
                                      TensorStorageType storage) {
  Tensor<BHWC, T> src;
  src.shape = BHWC(1, 2, 1, 5);
  src.data = {1, 2, -5, -2, 1, 3, 4, -2, 1, 4};

  const std::set<Axis> axis{Axis::CHANNELS};

  Tensor<BHWC, T> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 1);
  ref_tensor.data = {-3, 10};

  OperationDef op_def;
  op_def.src_tensors.push_back({T, storage, Layout::HWC});
  op_def.dst_tensors.push_back({T, storage, Layout::HWC});
  TensorDescriptor src_0, dst;
  src_0 = op_def.src_tensors[0];
  src_0.UploadData(src);
  dst.SetBHWCShape(BHWC(1, 2, 1, 1));
  Reduce operation = CreateReduce(axis, src.shape, OperationType::REDUCE_SUM,
                                  op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0}, {&dst}, std::make_unique<Reduce>(std::move(operation))));
  Tensor<BHWC, T> dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

template absl::Status ReduceSumChannelsIntTest<DataType::INT32>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status ReduceSumChannelsIntTest<DataType::INT16>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status ReduceSumChannelsIntTest<DataType::INT8>(
    TestExecutionEnvironment& env, TensorStorageType storage);

template <DataType T>
absl::Status ReduceProductChannelsUIntTest(TestExecutionEnvironment& env,
                                           TensorStorageType storage) {
  Tensor<BHWC, T> src;
  src.shape = BHWC(1, 3, 1, 2);
  src.data = {1, 2, 3, 4, 0, 7};
  const std::set<Axis> axis{Axis::CHANNELS};

  Tensor<BHWC, T> ref_tensor;
  ref_tensor.shape = BHWC(1, 3, 1, 1);
  ref_tensor.data = {2, 12, 0};

  OperationDef op_def;
  op_def.src_tensors.push_back({T, storage, Layout::HWC});
  op_def.dst_tensors.push_back({T, storage, Layout::HWC});
  TensorDescriptor src_0, dst;
  src_0 = op_def.src_tensors[0];
  src_0.UploadData(src);
  dst.SetBHWCShape(BHWC(1, 3, 1, 1));
  Reduce operation = CreateReduce(
      axis, src.shape, OperationType::REDUCE_PRODUCT, op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0}, {&dst}, std::make_unique<Reduce>(std::move(operation))));
  Tensor<BHWC, T> dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

template absl::Status ReduceProductChannelsUIntTest<DataType::UINT8>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status ReduceProductChannelsUIntTest<DataType::UINT16>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status ReduceProductChannelsUIntTest<DataType::UINT32>(
    TestExecutionEnvironment& env, TensorStorageType storage);

absl::Status ReduceAllTest(TestExecutionEnvironment& env,
                           TensorStorageType storage) {
  TensorBool src;
  src.shape = BHWC(1, 2, 1, 5);
  src.data = {true, true, true, true, true, true, true, true, true, false};

  const std::set<Axis> axis{Axis::CHANNELS};

  TensorBool ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 1);
  ref_tensor.data = {true, false};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::BOOL, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::BOOL, storage, Layout::HWC});
  TensorDescriptor src_0, dst;
  src_0 = op_def.src_tensors[0];
  src_0.UploadData(src);
  dst.SetBHWCShape(BHWC(1, 2, 1, 1));
  Reduce operation = CreateReduce(axis, src.shape, OperationType::REDUCE_ALL,
                                  op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0}, {&dst}, std::make_unique<Reduce>(std::move(operation))));
  TensorBool dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

absl::Status ReduceAnyTest(TestExecutionEnvironment& env,
                           TensorStorageType storage) {
  TensorBool src;
  src.shape = BHWC(1, 2, 1, 5);
  src.data = {true, true, false, true, true, false, false, false, false, false};

  const std::set<Axis> axis{Axis::CHANNELS};

  TensorBool ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 1);
  ref_tensor.data = {true, false};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::BOOL, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::BOOL, storage, Layout::HWC});
  TensorDescriptor src_0, dst;
  src_0 = op_def.src_tensors[0];
  src_0.UploadData(src);
  dst.SetBHWCShape(BHWC(1, 2, 1, 1));
  Reduce operation = CreateReduce(axis, src.shape, OperationType::REDUCE_ANY,
                                  op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0}, {&dst}, std::make_unique<Reduce>(std::move(operation))));
  TensorBool dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

absl::Status MeanHWTest(TestExecutionEnvironment& env, DataType data_type,
                        TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 1);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::set<Axis> axis{Axis::HEIGHT, Axis::WIDTH};

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  Reduce operation =
      CreateReduce(axis, src_tensor.shape, OperationType::MEAN, op_def,
                    env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Reduce>(std::move(operation)),
      BHWC(1, 1, 1, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), {2.5f}));
  return absl::OkStatus();
}

absl::Status ReduceSumChannelsTest(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 5);
  src_tensor.data = {1.1, 2.1, 0.7, 0.3, 1.2, 3.1, 4.1, 0.0, 1.0, 4.4};
  const std::set<Axis> axis{Axis::CHANNELS};

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  Reduce operation =
      CreateReduce(axis, src_tensor.shape, OperationType::REDUCE_SUM,
                    op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Reduce>(std::move(operation)),
      BHWC(1, 2, 1, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), {5.4f, 12.6f}));
  return absl::OkStatus();
}

absl::Status ReduceProductChannelsTest(TestExecutionEnvironment& env,
                                       DataType data_type,
                                       TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {1.1, 2.0, 3.1, 4.0};
  const std::set<Axis> axis{Axis::CHANNELS};

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  Reduce operation =
      CreateReduce(axis, src_tensor.shape, OperationType::REDUCE_PRODUCT,
                    op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Reduce>(std::move(operation)),
      BHWC(1, 2, 1, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), {2.2f, 12.4f}));
  return absl::OkStatus();
}

absl::Status ReduceMaxChannelsTest(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage) {
  TensorFloat32 src_tensor;
  const std::vector<int> num_channels = {6, 257};

  for (int channels : num_channels) {
    src_tensor.shape = BHWC(1, 2, 1, channels);
    src_tensor.data.resize(2 * channels, -1000.0f);
    // Move the custom values to the end of the tensor to ensure masking works
    // correctly.
    std::vector<float> channel1 = {1.1, 2.0, -0.3, -100.0, 32.6, 1.1};
    std::vector<float> channel2 = {-3.1, -4.0, -5.0, -7.0, -2.0, -100.0};
    src_tensor.data.insert(src_tensor.data.begin() + (channels - 6),
                           channel1.begin(), channel1.end());
    src_tensor.data.insert(src_tensor.data.begin() + (2 * channels - 6),
                           channel2.begin(), channel2.end());

    const std::set<Axis> axis{Axis::CHANNELS};

    const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
    OperationDef op_def;
    op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
    op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
    TensorFloat32 dst_tensor;
    Reduce operation =
        CreateReduce(axis, src_tensor.shape, OperationType::REDUCE_MAXIMUM,
                      op_def, env.GetGpuInfo());
    ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
        src_tensor, std::make_unique<Reduce>(std::move(operation)),
        BHWC(1, 2, 1, 1), &dst_tensor));
    EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), {32.6f, -2.0f}));
  }
  return absl::OkStatus();
}

absl::Status ReduceMinChannelsTest(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 6);
  src_tensor.data = {1.1,  2.0,  -0.3, -100.0, 32.6, 1.1,
                     -3.1, -4.0, -5.0, -7.0,   -2.0, 100.0};
  const std::set<Axis> axis{Axis::CHANNELS};

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  Reduce operation =
      CreateReduce(axis, src_tensor.shape, OperationType::REDUCE_MINIMUM,
                    op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Reduce>(std::move(operation)),
      BHWC(1, 2, 1, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), {-100.0f, -7.0f}));
  return absl::OkStatus();
}

absl::Status ReduceMaxIndChannelsTest(TestExecutionEnvironment& env,
                                      DataType data_type,
                                      TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 6);
  src_tensor.data = {1.1,  2.0,  -0.3,  -100.0, 32.6, 1.1,
                     -3.1, -4.0, 100.0, -7.0,   -2.0, 100.0};
  const std::set<Axis> axis{Axis::CHANNELS};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  TensorDescriptor src, dst;
  src = op_def.src_tensors[0];
  src.UploadData(src_tensor);
  dst.SetBHWCShape(BHWC(1, 2, 1, 1));
  Reduce operation =
      CreateReduce(axis, src_tensor.shape, Reduce::Type::kMaximumIndex, op_def,
                   env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src}, {&dst}, std::make_unique<Reduce>(std::move(operation))));
  TensorInt32 dst_tensor;
  dst.DownloadData(&dst_tensor);
  TensorInt32 ref_tensor;
  ref_tensor.data = {4, 2};
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

absl::Status ReduceMaxIndHeightTest(TestExecutionEnvironment& env,
                                    DataType data_type,
                                    TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 6);
  src_tensor.data = {1.1,  2.0,  -0.3,  -100.0, 32.6, 1.1,
                     -3.1, -4.0, 100.0, -7.0,   -2.0, 100.0};
  const std::set<Axis> axis{Axis::HEIGHT};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  TensorDescriptor src, dst;
  src = op_def.src_tensors[0];
  src.UploadData(src_tensor);
  dst.SetBHWCShape(BHWC(1, 1, 1, 6));
  Reduce operation =
      CreateReduce(axis, src_tensor.shape, Reduce::Type::kMaximumIndex, op_def,
                   env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src}, {&dst}, std::make_unique<Reduce>(std::move(operation))));
  TensorInt32 dst_tensor;
  dst.DownloadData(&dst_tensor);
  TensorInt32 ref_tensor;
  ref_tensor.data = {0, 0, 1, 1, 0, 1};
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

absl::Status ReduceHWBigTest(TestExecutionEnvironment& env, DataType data_type,
                             TensorStorageType storage, OperationType op_type) {
  auto src_shape = BHWC(1, 43, 57, 7);

  const std::set<Axis> axis{Axis::HEIGHT, Axis::WIDTH};

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return ReduceTest(env, axis, op_type, src_tensor, op_def);
}

absl::Status ReduceHWBatchedBigTest(TestExecutionEnvironment& env,
                                    DataType data_type,
                                    TensorStorageType storage,
                                    OperationType op_type) {
  auto src_shape = BHWC(5, 21, 13, 7);

  const std::set<Axis> axis{Axis::HEIGHT, Axis::WIDTH};

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  return ReduceTest(env, axis, op_type, src_tensor, op_def);
}

absl::Status ReduceHBigTest(TestExecutionEnvironment& env, DataType data_type,
                            TensorStorageType storage, OperationType op_type) {
  auto src_shape = BHWC(1, 4, 5, 5);

  const std::set<Axis> axis{Axis::HEIGHT};

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return ReduceTest(env, axis, op_type, src_tensor, op_def);
}

absl::Status ReduceBHBigTest(TestExecutionEnvironment& env, DataType data_type,
                             TensorStorageType storage, OperationType op_type) {
  auto src_shape = BHWC(6, 4, 5, 5);

  const std::set<Axis> axis{Axis::HEIGHT, Axis::BATCH};

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return ReduceTest(env, axis, op_type, src_tensor, op_def);
}

absl::Status ReduceCBigTest(TestExecutionEnvironment& env, DataType data_type,
                            TensorStorageType storage, OperationType op_type) {
  auto src_shape = BHWC(1, 4, 5, 5);

  const std::set<Axis> axis{Axis::CHANNELS};

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return ReduceTest(env, axis, op_type, src_tensor, op_def);
}

absl::Status ReduceCx4BigTest(TestExecutionEnvironment& env, DataType data_type,
                              TensorStorageType storage,
                              OperationType op_type) {
  auto src_shape = BHWC(1, 4, 5, 44);

  const std::set<Axis> axis{Axis::CHANNELS};

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return ReduceTest(env, axis, op_type, src_tensor, op_def);
}

absl::Status ReduceHWCBigTest(TestExecutionEnvironment& env, DataType data_type,
                              TensorStorageType storage,
                              OperationType op_type) {
  auto src_shape = BHWC(1, 4, 5, 13);

  const std::set<Axis> axis{Axis::HEIGHT, Axis::WIDTH, Axis::CHANNELS};

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return ReduceTest(env, axis, op_type, src_tensor, op_def);
}

absl::Status ReduceBHDBigTest(TestExecutionEnvironment& env, DataType data_type,
                              TensorStorageType storage,
                              OperationType op_type) {
  auto src_shape = BHWDC(13, 6, 7, 11, 7);

  const std::set<Axis> axis{Axis::BATCH, Axis::HEIGHT, Axis::DEPTH};

  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWDC});
  return ReduceTest(env, axis, op_type, src_tensor, op_def);
}

absl::Status ReduceBHWDBigTest(TestExecutionEnvironment& env,
                               DataType data_type, TensorStorageType storage,
                               OperationType op_type) {
  auto src_shape = BHWDC(13, 6, 7, 11, 7);

  const std::set<Axis> axis{Axis::BATCH, Axis::HEIGHT, Axis::WIDTH,
                            Axis::DEPTH};

  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWDC});
  return ReduceTest(env, axis, op_type, src_tensor, op_def);
}

}  // namespace ml_drift
