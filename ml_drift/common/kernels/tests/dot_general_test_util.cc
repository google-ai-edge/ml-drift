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

#include "ml_drift/common/kernels/tests/dot_general_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/dot_general.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

template <DataType T>
absl::Status DotGeneral2DIntTest(TestExecutionEnvironment& env,
                                 TensorStorageType storage) {
  Tensor<BHWC, T> lhs, rhs, ref;
  lhs.shape = BHWC(2, 1, 1, 3);
  lhs.data = {1, 2, 3, 4, 5, 6};
  rhs.shape = BHWC(2, 1, 1, 3);
  rhs.data = {1, 1, 1, 2, 2, 2};
  ref.shape = BHWC(2, 1, 1, 2);
  ref.data = {6, 12, 15, 30};

  DotGeneralAttributes attr;
  attr.lhs_contracting_axes = {Axis::CHANNELS};
  attr.rhs_contracting_axes = {Axis::CHANNELS};
  attr.lhs_resulting_axes = {Axis::BATCH};
  attr.rhs_resulting_axes = {Axis::BATCH};
  OperationDef op_def;
  op_def.src_tensors.push_back({T, storage, Layout::BHWC});
  op_def.src_tensors.push_back({T, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({T, storage, Layout::BHWC});
  TensorDescriptor lhs_td, rhs_td, dst;
  lhs_td = op_def.src_tensors[0];
  rhs_td = op_def.src_tensors[1];
  lhs_td.UploadData(lhs);
  rhs_td.UploadData(rhs);
  dst.SetBHWCShape(ref.shape);
  GPUOperation operation = CreateDotGeneral(op_def, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&lhs_td, &rhs_td}, {&dst},
      std::make_unique<GPUOperation>(std::move(operation))));
  Tensor<BHWC, T> dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref.data);
  return absl::OkStatus();
}

template absl::Status DotGeneral2DIntTest<DataType::INT8>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status DotGeneral2DIntTest<DataType::INT16>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status DotGeneral2DIntTest<DataType::INT32>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status DotGeneral2DIntTest<DataType::UINT8>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status DotGeneral2DIntTest<DataType::UINT16>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status DotGeneral2DIntTest<DataType::UINT32>(
    TestExecutionEnvironment& env, TensorStorageType storage);

absl::Status DotGeneral2DBfloatTest(TestExecutionEnvironment& env,
                                    TensorStorageType storage) {
  Tensor<BHWC, DataType::BFLOAT16> lhs, rhs;
  lhs.shape = BHWC(2, 1, 1, 3);
  lhs.data = FloatToBFloat({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  rhs.shape = BHWC(2, 1, 1, 3);
  rhs.data = FloatToBFloat({1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f});

  DotGeneralAttributes attr;
  attr.lhs_contracting_axes = {Axis::CHANNELS};
  attr.rhs_contracting_axes = {Axis::CHANNELS};
  attr.lhs_resulting_axes = {Axis::BATCH};
  attr.rhs_resulting_axes = {Axis::BATCH};
  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::BFLOAT16, storage, Layout::BHWC});
  op_def.src_tensors.push_back({DataType::BFLOAT16, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({DataType::BFLOAT16, storage, Layout::BHWC});
  TensorDescriptor lhs_td, rhs_td, dst;
  lhs_td = op_def.src_tensors[0];
  rhs_td = op_def.src_tensors[1];
  lhs_td.UploadData(lhs);
  rhs_td.UploadData(rhs);
  dst.SetBHWCShape(BHWC(2, 1, 1, 2));
  GPUOperation operation = CreateDotGeneral(op_def, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&lhs_td, &rhs_td}, {&dst},
      std::make_unique<GPUOperation>(std::move(operation))));
  Tensor<BHWC, DataType::BFLOAT16> dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_THAT(BFloatToFloat(dst_tensor.data),
              Pointwise(FloatNear(1e-3f), {6.0f, 12.0f, 15.0f, 30.0f}));
  return absl::OkStatus();
}

absl::Status DotGeneral1DTest(TestExecutionEnvironment& env, DataType data_type,
                              TensorStorageType storage) {
  TensorFloat32 lhs;
  lhs.shape = BHWC(2, 1, 1, 2);
  lhs.data = {1.0f, 2.0f, 3.0f, 4.0f};
  TensorFloat32 rhs;
  rhs.shape = BHWC(1, 1, 1, 2);
  rhs.data = {1.0f, 2.0f};
  DotGeneralAttributes attr;
  attr.lhs_resulting_axes = {Axis::BATCH};
  attr.lhs_contracting_axes = {Axis::CHANNELS};
  attr.rhs_contracting_axes = {Axis::CHANNELS};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  TensorDescriptor lhs_td, rhs_td, dst;
  lhs_td = op_def.src_tensors[0];
  rhs_td = op_def.src_tensors[1];
  lhs_td.UploadData(lhs);
  rhs_td.UploadData(rhs);
  dst.SetBHWCShape(BHWC(2, 1, 1, 1));
  GPUOperation operation = CreateDotGeneral(op_def, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&lhs_td, &rhs_td}, {&dst},
      std::make_unique<GPUOperation>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(0.0f), {5.0f, 11.0f}));
  return absl::OkStatus();
}

absl::Status DotGeneral2DTest(TestExecutionEnvironment& env, DataType data_type,
                              TensorStorageType storage) {
  TensorFloat32 lhs;
  lhs.shape = BHWC(2, 1, 1, 3);
  lhs.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  TensorFloat32 rhs;
  rhs.shape = BHWC(2, 1, 1, 3);
  rhs.data = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};
  DotGeneralAttributes attr;
  attr.lhs_contracting_axes = {Axis::CHANNELS};
  attr.rhs_contracting_axes = {Axis::CHANNELS};
  attr.lhs_resulting_axes = {Axis::BATCH};
  attr.rhs_resulting_axes = {Axis::BATCH};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  TensorDescriptor lhs_td, rhs_td, dst;
  lhs_td = op_def.src_tensors[0];
  rhs_td = op_def.src_tensors[1];
  lhs_td.UploadData(lhs);
  rhs_td.UploadData(rhs);
  dst.SetBHWCShape(BHWC(2, 1, 1, 2));
  GPUOperation operation = CreateDotGeneral(op_def, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&lhs_td, &rhs_td}, {&dst},
      std::make_unique<GPUOperation>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.0f), {6.0f, 12.0f, 15.0f, 30.0f}));
  return absl::OkStatus();
}

absl::Status DotGeneral3DBatchTest(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage) {
  TensorFloat32 lhs;
  lhs.shape = BHWC(2, 1, 2, 2);
  lhs.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  TensorFloat32 rhs;
  rhs.shape = BHWC(2, 1, 2, 2);
  rhs.data = {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
  DotGeneralAttributes attr;
  attr.lhs_contracting_axes = {Axis::CHANNELS};
  attr.rhs_contracting_axes = {Axis::WIDTH};
  attr.lhs_batch_axes = {Axis::BATCH};
  attr.rhs_batch_axes = {Axis::BATCH};
  attr.lhs_resulting_axes = {Axis::WIDTH};
  attr.rhs_resulting_axes = {Axis::CHANNELS};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  TensorDescriptor lhs_td, rhs_td, dst;
  lhs_td = op_def.src_tensors[0];
  rhs_td = op_def.src_tensors[1];
  lhs_td.UploadData(lhs);
  rhs_td.UploadData(rhs);
  dst.SetBHWCShape(BHWC(2, 1, 2, 2));
  GPUOperation operation = CreateDotGeneral(op_def, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&lhs_td, &rhs_td}, {&dst},
      std::make_unique<GPUOperation>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.0f),
                        {0.0f, 3.0f, 0.0f, 7.0f, 0.0f, 11.0f, 0.0f, 15.0f}));
  return absl::OkStatus();
}

absl::Status DotGeneral4DTest(TestExecutionEnvironment& env, DataType data_type,
                              TensorStorageType storage) {
  TensorFloat32 lhs;
  lhs.shape = BHWC(2, 2, 2, 2);
  lhs.data = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
              9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  TensorFloat32 rhs;
  rhs.shape = BHWC(2, 2, 2, 2);
  rhs.data = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
              1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f};
  DotGeneralAttributes attr;
  attr.lhs_contracting_axes = {Axis::BATCH, Axis::HEIGHT};
  attr.rhs_contracting_axes = {Axis::BATCH, Axis::HEIGHT};
  attr.lhs_resulting_axes = {Axis::WIDTH, Axis::CHANNELS};
  attr.rhs_resulting_axes = {Axis::WIDTH, Axis::CHANNELS};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  TensorDescriptor lhs_td, rhs_td, dst;
  lhs_td = op_def.src_tensors[0];
  rhs_td = op_def.src_tensors[1];
  lhs_td.UploadData(lhs);
  rhs_td.UploadData(rhs);
  dst.SetBHWCShape(BHWC(2, 2, 2, 2));
  GPUOperation operation = CreateDotGeneral(op_def, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&lhs_td, &rhs_td}, {&dst},
      std::make_unique<GPUOperation>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.0f),
                        {28.0f, 0.0f, 0.0f, 28.0f, 32.0f, 0.0f, 0.0f, 32.0f,
                         36.0f, 0.0f, 0.0f, 36.0f, 40.0f, 0.0f, 0.0f, 40.0f}));
  return absl::OkStatus();
}

}  // namespace ml_drift
