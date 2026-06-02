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

#include "ml_drift/common/kernels/tests/tile_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tile.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

absl::Status TileChannelsTest(TestExecutionEnvironment& env, DataType data_type,
                              TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 3);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  TensorDescriptor src_desc(data_type, storage, Layout::HWC);
  TensorDescriptor dst_desc(data_type, storage, Layout::HWC);
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateTile(src_desc, dst_desc, src_tensor.shape.c % 4 == 0);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 6), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.0f), {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                          4.0f, 5.0f, 6.0f, 4.0f, 5.0f, 6.0f}));
  return absl::OkStatus();
}

absl::Status TileChannelsX4Test(TestExecutionEnvironment& env,
                                DataType data_type, TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 4);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 7.0f, 4.0f, 5.0f, 6.0f, 8.0f};

  TensorDescriptor src_desc(data_type, storage, Layout::HWC);
  TensorDescriptor dst_desc(data_type, storage, Layout::HWC);
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateTile(src_desc, dst_desc, src_tensor.shape.c % 4 == 0);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 8), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.0f),
                        {1.0f, 2.0f, 3.0f, 7.0f, 1.0f, 2.0f, 3.0f, 7.0f, 4.0f,
                         5.0f, 6.0f, 8.0f, 4.0f, 5.0f, 6.0f, 8.0f}));
  return absl::OkStatus();
}

absl::Status TileWidthTest(TestExecutionEnvironment& env, DataType data_type,
                           TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 2, 3);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  TensorDescriptor src_desc(data_type, storage, Layout::HWC);
  TensorDescriptor dst_desc(data_type, storage, Layout::HWC);
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateTile(src_desc, dst_desc, src_tensor.shape.c % 4 == 0);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 1, 4, 3), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.0f), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                          1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  return absl::OkStatus();
}

absl::Status TileHeightTest(TestExecutionEnvironment& env, DataType data_type,
                            TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 3);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  TensorDescriptor src_desc(data_type, storage, Layout::HWC);
  TensorDescriptor dst_desc(data_type, storage, Layout::HWC);
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateTile(src_desc, dst_desc, src_tensor.shape.c % 4 == 0);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 4, 1, 3), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.0f), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                          1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  return absl::OkStatus();
}

absl::Status TileHWCTest(TestExecutionEnvironment& env, DataType data_type,
                         TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 3);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                     7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  TensorDescriptor src_desc(data_type, storage, Layout::HWC);
  TensorDescriptor dst_desc(data_type, storage, Layout::HWC);
  TensorFloat32 dst_tensor;
  GPUOperation operation =
      CreateTile(src_desc, dst_desc, src_tensor.shape.c % 4 == 0);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 4, 4, 6), &dst_tensor));
  EXPECT_THAT(
      dst_tensor.data,
      Pointwise(
          FloatNear(0.0f),
          {1.0f,  2.0f,  3.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  4.0f,
           5.0f,  6.0f,  1.0f,  2.0f,  3.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,
           6.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  7.0f,  8.0f,  9.0f,
           10.0f, 11.0f, 12.0f, 10.0f, 11.0f, 12.0f, 7.0f,  8.0f,  9.0f,  7.0f,
           8.0f,  9.0f,  10.0f, 11.0f, 12.0f, 10.0f, 11.0f, 12.0f, 1.0f,  2.0f,
           3.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  4.0f,  5.0f,  6.0f,
           1.0f,  2.0f,  3.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  4.0f,
           5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f,
           12.0f, 10.0f, 11.0f, 12.0f, 7.0f,  8.0f,  9.0f,  7.0f,  8.0f,  9.0f,
           10.0f, 11.0f, 12.0f, 10.0f, 11.0f, 12.0f}));
  return absl::OkStatus();
}

// Non-float tests.

template <DataType T>
absl::Status TileChannelsIntTest(TestExecutionEnvironment& env,
                                 TensorStorageType storage) {
  Tensor<BHWC, T> src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 3);
  src_tensor.data = {1, 2, 3, 4, 5, 6};

  Tensor<BHWC, T> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 6);
  ref_tensor.data = {1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6};

  TensorDescriptor src_desc(T, storage, Layout::HWC);
  src_desc.UploadData(src_tensor);
  TensorDescriptor dst_desc(T, storage, Layout::HWC);
  dst_desc.SetBHWCShape(BHWC(1, 2, 1, 6));

  GPUOperation operation =
      CreateTile(src_desc, dst_desc, src_tensor.shape.c % 4 == 0);
  Tensor<BHWC, T> dst_tensor;
  dst_tensor.shape = BHWC(1, 2, 1, 6);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));
  dst_desc.DownloadData(&dst_tensor);

  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

template absl::Status TileChannelsIntTest<DataType::INT8>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status TileChannelsIntTest<DataType::INT16>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status TileChannelsIntTest<DataType::INT32>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status TileChannelsIntTest<DataType::UINT8>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status TileChannelsIntTest<DataType::UINT16>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status TileChannelsIntTest<DataType::UINT32>(
    TestExecutionEnvironment& env, TensorStorageType storage);

template <DataType T>
absl::Status TileChannelsX4IntTest(TestExecutionEnvironment& env,
                                   TensorStorageType storage) {
  Tensor<BHWC, T> src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 4);
  src_tensor.data = {1, 2, 3, 7, 4, 5, 6, 8};

  Tensor<BHWC, T> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 8);
  ref_tensor.data = {1, 2, 3, 7, 1, 2, 3, 7, 4, 5, 6, 8, 4, 5, 6, 8};

  TensorDescriptor src_desc(T, storage, Layout::HWC);
  src_desc.UploadData(src_tensor);
  TensorDescriptor dst_desc(T, storage, Layout::HWC);
  dst_desc.SetBHWCShape(BHWC(1, 2, 1, 8));

  GPUOperation operation =
      CreateTile(src_desc, dst_desc, src_tensor.shape.c % 4 == 0);
  Tensor<BHWC, T> dst_tensor;
  dst_tensor.shape = BHWC(1, 2, 1, 8);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));
  dst_desc.DownloadData(&dst_tensor);

  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

template absl::Status TileChannelsX4IntTest<DataType::INT8>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status TileChannelsX4IntTest<DataType::INT16>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status TileChannelsX4IntTest<DataType::INT32>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status TileChannelsX4IntTest<DataType::UINT8>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status TileChannelsX4IntTest<DataType::UINT16>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status TileChannelsX4IntTest<DataType::UINT32>(
    TestExecutionEnvironment& env, TensorStorageType storage);

absl::Status Tile5DTest(TestExecutionEnvironment& env, DataType data_type,
                        TensorStorageType storage) {
  Tensor5DFloat32 src_tensor;
  src_tensor.shape = BHWDC(2, 2, 1, 2, 1);
  src_tensor.data = {1, 2, 3, 4, 5, 6, 7, 8};

  TensorDescriptor src_desc(data_type, storage, Layout::BHWDC);
  TensorDescriptor dst_desc(data_type, storage, Layout::BHWDC);
  Tensor5DFloat32 dst_tensor;
  GPUOperation operation = CreateTile(src_desc, dst_desc, true);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWDC(2, 4, 1, 2, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.0f),
                        {1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8}));
  return absl::OkStatus();
}

}  // namespace ml_drift
