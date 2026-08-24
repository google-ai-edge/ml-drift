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

#include "ml_drift/common/kernels/tests/dynamic_update_slice_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/dynamic_update_slice.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

absl::Status DynamicUpdateSliceBoolTest(TestExecutionEnvironment& env,
                                        TensorStorageType storage) {
  TensorBool array_to_update;
  array_to_update.shape = BHWC(1, 1, 1, 8);
  array_to_update.data = {true, true, true, true, true, true, true, true};

  TensorBool updated_slice;
  updated_slice.shape = BHWC(1, 1, 1, 3);
  updated_slice.data = {false, false, false};
  Tensor<BHWC, DataType::INT32> start_indices;
  start_indices.shape = BHWC(1, 1, 1, 4);
  start_indices.data = {0, 0, 0, 4};

  TensorBool ref_tensor;
  ref_tensor.data = {true, true, true, true, false, false, false, true};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::BOOL, storage, Layout::HWC});
  op_def.src_tensors.push_back({DataType::BOOL, storage, Layout::HWC});
  op_def.src_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::BOOL, storage, Layout::HWC});
  TensorDescriptor src_0, src_1, src_2, dst;
  src_0 = op_def.src_tensors[0];
  src_1 = op_def.src_tensors[1];
  src_2 = op_def.src_tensors[2];
  src_0.UploadData(array_to_update);
  src_1.UploadData(updated_slice);
  src_2.UploadData(start_indices);
  dst.SetBHWCShape(BHWC(1, 1, 1, 8));
  GPUOperation operation = CreateDynamicUpdateSlice(op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0, &src_1, &src_2}, {&dst},
      std::make_unique<GPUOperation>(std::move(operation))));
  TensorBool dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

template <DataType T>
absl::Status DynamicUpdateSliceIntTest(TestExecutionEnvironment& env,
                                       TensorStorageType storage) {
  Tensor<BHWC, T> array_to_update;
  array_to_update.shape = BHWC(1, 1, 1, 16);
  array_to_update.data = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

  Tensor<BHWC, T> updated_slice;
  updated_slice.shape = BHWC(1, 1, 1, 9);
  updated_slice.data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Tensor<BHWC, T> start_indices;
  start_indices.shape = BHWC(1, 1, 1, 4);
  start_indices.data = {0, 0, 0, 5};

  Tensor<BHWC, T> ref_tensor;
  ref_tensor.data = {2, 2, 2, 2, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 2};

  OperationDef op_def;
  op_def.src_tensors.push_back({T, storage, Layout::HWC});
  op_def.src_tensors.push_back({T, storage, Layout::HWC});
  op_def.src_tensors.push_back({T, storage, Layout::HWC});
  op_def.dst_tensors.push_back({T, storage, Layout::HWC});
  TensorDescriptor src_0, src_1, src_2, dst;
  src_0 = op_def.src_tensors[0];
  src_1 = op_def.src_tensors[1];
  src_2 = op_def.src_tensors[2];
  src_0.UploadData(array_to_update);
  src_1.UploadData(updated_slice);
  src_2.UploadData(start_indices);
  dst.SetBHWCShape(BHWC(1, 1, 1, 16));
  GPUOperation operation = CreateDynamicUpdateSlice(op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0, &src_1, &src_2}, {&dst},
      std::make_unique<GPUOperation>(std::move(operation))));
  Tensor<BHWC, T> dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

template absl::Status DynamicUpdateSliceIntTest<DataType::INT32>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status DynamicUpdateSliceIntTest<DataType::INT16>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status DynamicUpdateSliceIntTest<DataType::INT8>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status DynamicUpdateSliceIntTest<DataType::UINT32>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status DynamicUpdateSliceIntTest<DataType::UINT16>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status DynamicUpdateSliceIntTest<DataType::UINT8>(
    TestExecutionEnvironment& env, TensorStorageType storage);

absl::Status DynamicUpdateSliceTest(TestExecutionEnvironment& env,
                                    DataType data_type,
                                    TensorStorageType storage) {
  TensorFloat32 array_to_update;
  array_to_update.shape = BHWC(1, 1, 1, 16);
  array_to_update.data = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

  TensorFloat32 updated_slice;
  updated_slice.shape = BHWC(1, 1, 1, 9);
  updated_slice.data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  TensorFloat32 start_indices;
  start_indices.shape = BHWC(1, 1, 1, 4);
  start_indices.data = {0, 0, 0, 5};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorDescriptor src_0, src_1, src_2, dst;
  src_0 = op_def.src_tensors[0];
  src_1 = op_def.src_tensors[1];
  src_2 = op_def.src_tensors[2];
  src_0.UploadData(array_to_update);
  src_1.UploadData(updated_slice);
  src_2.UploadData(start_indices);
  dst.SetBHWCShape(BHWC(1, 1, 1, 16));
  GPUOperation operation = CreateDynamicUpdateSlice(op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0, &src_1, &src_2}, {&dst},
      std::make_unique<GPUOperation>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.0f),
                        {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                         5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 2.0f, 2.0f}));
  return absl::OkStatus();
}

absl::Status DynamicUpdateSliceTwoDimensionSliceTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage) {
  TensorFloat32 array_to_update;
  array_to_update.shape = BHWC(1, 4, 1, 5);
  array_to_update.data = {5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                          5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
  TensorFloat32 updated_slice;
  updated_slice.shape = BHWC(1, 2, 1, 2);
  updated_slice.data = {1, 2, 3, 4};
  TensorFloat32 start_indices;
  start_indices.shape = BHWC(1, 1, 1, 4);
  start_indices.data = {0, 2, 0, 2};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorDescriptor src_0, src_1, src_2, dst;
  src_0 = op_def.src_tensors[0];
  src_1 = op_def.src_tensors[1];
  src_2 = op_def.src_tensors[2];
  src_0.UploadData(array_to_update);
  src_1.UploadData(updated_slice);
  src_2.UploadData(start_indices);
  dst.SetBHWCShape(BHWC(1, 4, 1, 5));
  GPUOperation operation = CreateDynamicUpdateSlice(op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0, &src_1, &src_2}, {&dst},
      std::make_unique<GPUOperation>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_THAT(
      dst_tensor.data,
      Pointwise(FloatNear(0.0f),
                {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
                 5.0f, 5.0f, 1.0f, 2.0f, 5.0f, 5.0f, 5.0f, 3.0f, 4.0f, 5.0f}));
  return absl::OkStatus();
}

absl::Status DynamicUpdateSliceThreeDimensionSliceTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage) {
  TensorFloat32 array_to_update;
  array_to_update.shape = BHWC(1, 3, 3, 3);
  array_to_update.data = {9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                          9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};
  TensorFloat32 updated_slice;
  updated_slice.shape = BHWC(1, 2, 2, 2);
  updated_slice.data = {1, 2, 3, 4, 5, 6, 7, 8};
  TensorFloat32 start_indices;
  start_indices.shape = BHWC(1, 1, 1, 4);
  start_indices.data = {0, 1, 1, 1};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorDescriptor src_0, src_1, src_2, dst;
  src_0 = op_def.src_tensors[0];
  src_1 = op_def.src_tensors[1];
  src_2 = op_def.src_tensors[2];
  src_0.UploadData(array_to_update);
  src_1.UploadData(updated_slice);
  src_2.UploadData(start_indices);
  dst.SetBHWCShape(BHWC(1, 3, 3, 3));
  GPUOperation operation = CreateDynamicUpdateSlice(op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0, &src_1, &src_2}, {&dst},
      std::make_unique<GPUOperation>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_THAT(
      dst_tensor.data,
      Pointwise(FloatNear(0.0f),
                {9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f,
                 9.0f, 9.0f, 9.0f, 9.0f, 1.0f, 2.0f, 9.0f, 3.0f, 4.0f,
                 9.0f, 9.0f, 9.0f, 9.0f, 5.0f, 6.0f, 9.0f, 7.0f, 8.0f}));
  return absl::OkStatus();
}

absl::Status DynamicUpdateSliceStartIndicesThreeValuesSliceTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage) {
  TensorFloat32 array_to_update;
  array_to_update.shape = BHWC(1, 3, 3, 3);
  array_to_update.data = {9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                          9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};
  TensorFloat32 updated_slice;
  updated_slice.shape = BHWC(1, 2, 2, 2);
  updated_slice.data = {1, 2, 3, 4, 5, 6, 7, 8};
  TensorFloat32 start_indices;
  start_indices.shape = BHWC(1, 1, 1, 3);
  start_indices.data = {1, 1, 1};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorDescriptor src_0, src_1, src_2, dst;
  src_0 = op_def.src_tensors[0];
  src_1 = op_def.src_tensors[1];
  src_2 = op_def.src_tensors[2];
  src_0.UploadData(array_to_update);
  src_1.UploadData(updated_slice);
  src_2.UploadData(start_indices);
  dst.SetBHWCShape(BHWC(1, 3, 3, 3));
  GPUOperation operation = CreateDynamicUpdateSlice(op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0, &src_1, &src_2}, {&dst},
      std::make_unique<GPUOperation>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_THAT(
      dst_tensor.data,
      Pointwise(FloatNear(0.0f),
                {9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f,
                 9.0f, 9.0f, 9.0f, 9.0f, 1.0f, 2.0f, 9.0f, 3.0f, 4.0f,
                 9.0f, 9.0f, 9.0f, 9.0f, 5.0f, 6.0f, 9.0f, 7.0f, 8.0f}));
  return absl::OkStatus();
}

absl::Status DynamicUpdateSliceStartIndicesTwoValuesSliceTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage) {
  TensorFloat32 array_to_update;
  array_to_update.shape = BHWC(1, 3, 3, 3);
  // clang-format off
  array_to_update.data = {9, 9, 9,
                          9, 9, 9,
                          9, 9, 9,

                          9, 9, 9,
                          9, 9, 9,
                          9, 9, 9,

                          9, 9, 9,
                          9, 9, 9,
                          9, 9, 9};
  // clang-format on
  TensorFloat32 updated_slice;
  updated_slice.shape = BHWC(1, 2, 2, 2);
  // clang-format off
  updated_slice.data = {1, 2,
                        3, 4,

                        5, 6,
                        7, 8};
  // clang-format on
  TensorFloat32 start_indices;
  start_indices.shape = BHWC(1, 1, 1, 2);
  start_indices.data = {1, 1};
  // equivalent to:
  // start_indices.shape = BHWC(1, 1, 1, 3);
  // start_indices.data = {0, 1, 1};
  // clang-format off
  std::vector<float> ref_output = {9, 9, 9,
                                   9, 1, 2,
                                   9, 3, 4,

                                   9, 9, 9,
                                   9, 5, 6,
                                   9, 7, 8,

                                   9, 9, 9,
                                   9, 9, 9,
                                   9, 9, 9};
  // clang-format on

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorDescriptor src_0, src_1, src_2, dst;
  src_0 = op_def.src_tensors[0];
  src_1 = op_def.src_tensors[1];
  src_2 = op_def.src_tensors[2];
  src_0.UploadData(array_to_update);
  src_1.UploadData(updated_slice);
  src_2.UploadData(start_indices);
  dst.SetBHWCShape(BHWC(1, 3, 3, 3));
  GPUOperation operation = CreateDynamicUpdateSlice(op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0, &src_1, &src_2}, {&dst},
      std::make_unique<GPUOperation>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(0.0f), ref_output));
  return absl::OkStatus();
}

absl::Status DynamicUpdateSliceClampTest(TestExecutionEnvironment& env,
                                         DataType data_type,
                                         TensorStorageType storage) {
  TensorFloat32 array_to_update;
  array_to_update.shape = BHWC(1, 3, 1, 4);
  array_to_update.data = {5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
  TensorFloat32 updated_slice;
  updated_slice.shape = BHWC(1, 2, 1, 2);
  updated_slice.data = {1, 2, 3, 4};
  TensorFloat32 start_indices;
  start_indices.shape = BHWC(1, 1, 1, 4);
  start_indices.data = {0, 2, 0, 2};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorDescriptor src_0, src_1, src_2, dst;
  src_0 = op_def.src_tensors[0];
  src_1 = op_def.src_tensors[1];
  src_2 = op_def.src_tensors[2];
  src_0.UploadData(array_to_update);
  src_1.UploadData(updated_slice);
  src_2.UploadData(start_indices);
  dst.SetBHWCShape(BHWC(1, 3, 1, 4));
  GPUOperation operation = CreateDynamicUpdateSlice(op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0, &src_1, &src_2}, {&dst},
      std::make_unique<GPUOperation>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.0f), {5.0f, 5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
                                          1.0f, 2.0f, 5.0f, 5.0f, 3.0f, 4.0f}));
  return absl::OkStatus();
}

absl::Status DynamicUpdateSliceConversionTest(TestExecutionEnvironment& env,
                                              DataType src_type,
                                              DataType dst_type,
                                              TensorStorageType storage) {
  TensorFloat32 array_to_update;
  array_to_update.shape = BHWC(1, 1, 1, 16);
  array_to_update.data = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

  TensorFloat32 updated_slice;
  updated_slice.shape = BHWC(1, 1, 1, 9);
  updated_slice.data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Tensor<BHWC, DataType::INT32> start_indices;
  start_indices.shape = BHWC(1, 1, 1, 4);
  start_indices.data = {0, 0, 0, 5};

  OperationDef op_def;
  op_def.src_tensors.push_back({dst_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({src_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  op_def.dst_tensors.push_back({dst_type, storage, Layout::HWC});
  TensorDescriptor src_0, src_1, src_2, dst;
  src_0 = op_def.src_tensors[0];
  src_1 = op_def.src_tensors[1];
  src_2 = op_def.src_tensors[2];
  src_0.UploadData(array_to_update);
  src_1.UploadData(updated_slice);
  src_2.UploadData(start_indices);
  dst.SetBHWCShape(BHWC(1, 1, 1, 16));
  GPUOperation operation = CreateDynamicUpdateSlice(op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0, &src_1, &src_2}, {&dst},
      std::make_unique<GPUOperation>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.0f),
                        {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                         5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 2.0f, 2.0f}));
  return absl::OkStatus();
}

}  // namespace ml_drift
