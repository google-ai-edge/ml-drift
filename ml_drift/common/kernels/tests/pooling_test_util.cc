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

#include "ml_drift/common/kernels/tests/pooling_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/pooling.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_ref_ops.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

absl::Status AveragePoolingTest(TestExecutionEnvironment& env,
                                DataType data_type, TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  Pooling2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(2, 2);
  attr.kernel = HW(2, 2);
  attr.type = PoolingType::AVERAGE;

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreatePooling(op_def, env.GetGpuInfo(), attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 1, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), {3.0f, 4.0f}));
  return absl::OkStatus();
}

absl::Status AveragePoolingNonEmptyPaddingTest(TestExecutionEnvironment& env,
                                               DataType data_type,
                                               TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 1);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  Pooling2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(1, 1);
  attr.strides = HW(1, 1);
  attr.kernel = HW(2, 2);
  attr.type = PoolingType::AVERAGE;

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreatePooling(op_def, env.GetGpuInfo(), attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 2, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {1.5f, 2.0f, 2.5f, 3.0f}));
  return absl::OkStatus();
}

absl::Status MaxPoolingTest(TestExecutionEnvironment& env, DataType data_type,
                            TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 2);
  src_tensor.data = {8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  Pooling2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(2, 2);
  attr.kernel = HW(2, 2);
  attr.type = PoolingType::MAX;

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreatePooling(op_def, env.GetGpuInfo(), attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 1, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), {8.0f, 7.0f}));
  return absl::OkStatus();
}

absl::Status MaxPoolingIndicesTest(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 2);
  src_tensor.data = {8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  Pooling2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(2, 2);
  attr.kernel = HW(2, 2);
  attr.type = PoolingType::MAX;
  attr.output_indices = true;

  {
    const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
    OperationDef op_def;
    op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
    op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
    op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
    TensorFloat32 dst_tensor;
    TensorFloat32 dst_tensor_ind;
    GPUOperation operation = CreatePooling(op_def, env.GetGpuInfo(), attr);
    RETURN_IF_ERROR(env.ExecuteGPUOperation(
        {src_tensor}, std::make_unique<GPUOperation>(std::move(operation)),
        {BHWC(1, 1, 1, 2), BHWC(1, 1, 1, 2)}, {&dst_tensor, &dst_tensor_ind}));
    EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), {8.0f, 7.0f}));
    for (auto& v : dst_tensor_ind.data) {
      v = static_cast<int>(v);
    }
    EXPECT_THAT(dst_tensor_ind.data, Pointwise(FloatNear(eps), {0.0f, 3.0f}));
  }

  // Testing writing of indices in int tensor
  {
    OperationDef op_def;
    op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
    op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
    op_def.dst_tensors.push_back({DataType::INT32, storage, Layout::HWC});

    TensorDescriptor src_0, dst_0, dst_1;
    src_0 = op_def.src_tensors[0];
    src_0.UploadData(src_tensor);
    dst_0.SetBHWCShape(BHWC(1, 1, 1, 2));
    dst_1.SetBHWCShape(BHWC(1, 1, 1, 2));

    GPUOperation operation = CreatePooling(op_def, env.GetGpuInfo(), attr);
    RETURN_IF_ERROR(env.ExecuteGPUOperation(
        {&src_0}, {&dst_0, &dst_1},
        std::make_unique<GPUOperation>(std::move(operation))));

    TensorFloat32 dst_tensor;
    dst_0.DownloadData(&dst_tensor);
    Tensor<BHWC, DataType::INT32> dst_tensor_ind;
    dst_1.DownloadData(&dst_tensor_ind);
    const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
    EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), {8.0f, 7.0f}));
    Tensor<BHWC, DataType::INT32> ref_tensor;
    ref_tensor.shape = BHWC(1, 1, 1, 2);
    ref_tensor.data = {0, 3};
    EXPECT_EQ(dst_tensor_ind.data, ref_tensor.data);
  }

  return absl::OkStatus();
}

namespace {

absl::Status AveragePoolingTest(TestExecutionEnvironment& exec_env,
                                const Pooling2DAttributes& attr,
                                const TensorFloat32& src_tensor,
                                const OperationDef& op_def) {
  TensorFloat32 dst_ref_tensor = AveragePoolingReference(attr, src_tensor);

  auto operation = CreatePooling(op_def, exec_env.GetGpuInfo(), attr);

  TensorFloat32 dst_tensor;
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.001f), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status AveragePooling3DTest(TestExecutionEnvironment& exec_env,
                                  const Pooling3DAttributes& attr,
                                  const Tensor5DFloat32& src_tensor,
                                  const OperationDef& op_def) {
  Tensor5DFloat32 dst_ref_tensor = AveragePoolingReference(attr, src_tensor);

  auto operation = CreatePooling(op_def, exec_env.GetGpuInfo(), attr);

  Tensor5DFloat32 dst_tensor;
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.001f), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status MaxPoolingTest(TestExecutionEnvironment& exec_env,
                            const Pooling2DAttributes& attr,
                            const TensorFloat32& src_tensor,
                            const OperationDef& op_def) {
  TensorFloat32 dst_ref_tensor = MaxPoolingReference(attr, src_tensor)[0];

  auto operation = CreatePooling(op_def, exec_env.GetGpuInfo(), attr);

  TensorFloat32 dst_tensor;
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.001f), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status MaxPooling3DTest(TestExecutionEnvironment& exec_env,
                              const Pooling3DAttributes& attr,
                              const Tensor5DFloat32& src_tensor,
                              const OperationDef& op_def) {
  Tensor5DFloat32 dst_ref_tensor = MaxPoolingReference(attr, src_tensor)[0];

  auto operation = CreatePooling(op_def, exec_env.GetGpuInfo(), attr);

  Tensor5DFloat32 dst_tensor;
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.001f), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status MaxPoolingIndicesTest(TestExecutionEnvironment& exec_env,
                                   const Pooling2DAttributes& attr,
                                   const TensorFloat32& src_tensor,
                                   const OperationDef& op_def) {
  auto dst_ref_tensors = MaxPoolingReference(attr, src_tensor);

  auto operation = CreatePooling(op_def, exec_env.GetGpuInfo(), attr);

  std::vector<TensorFloat32> dst_tensors(2);
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      {src_tensor}, std::make_unique<GPUOperation>(std::move(operation)),
      {dst_ref_tensors[0].shape, dst_ref_tensors[1].shape},
      {&dst_tensors[0], &dst_tensors[1]}));
  EXPECT_THAT(dst_tensors[0].data,
              Pointwise(FloatNear(0.001f), dst_ref_tensors[0].data));
  EXPECT_THAT(dst_tensors[1].data,
              Pointwise(FloatNear(0.001f), dst_ref_tensors[1].data));
  return absl::OkStatus();
}

absl::Status MaxPoolingIndices3DTest(TestExecutionEnvironment& exec_env,
                                     const Pooling3DAttributes& attr,
                                     const Tensor5DFloat32& src_tensor,
                                     const OperationDef& op_def) {
  auto dst_ref_tensors = MaxPoolingReference(attr, src_tensor);

  auto operation = CreatePooling(op_def, exec_env.GetGpuInfo(), attr);

  std::vector<Tensor5DFloat32> dst_tensors(2);
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      {src_tensor}, std::make_unique<GPUOperation>(std::move(operation)),
      {dst_ref_tensors[0].shape, dst_ref_tensors[1].shape},
      {&dst_tensors[0], &dst_tensors[1]}));
  EXPECT_THAT(dst_tensors[0].data,
              Pointwise(FloatNear(0.001f), dst_ref_tensors[0].data));
  EXPECT_THAT(dst_tensors[1].data,
              Pointwise(FloatNear(0.001f), dst_ref_tensors[1].data));
  return absl::OkStatus();
}
}  // namespace

absl::Status AveragePoolingBigTest(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage) {
  Pooling2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(2, 2);
  attr.kernel = HW(2, 2);
  attr.type = PoolingType::AVERAGE;

  auto src_shape = BHWC(1, 8, 6, 5);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return AveragePoolingTest(env, attr, src_tensor, op_def);
}

absl::Status AveragePoolingBatchedBigTest(TestExecutionEnvironment& env,
                                          DataType data_type,
                                          TensorStorageType storage) {
  Pooling2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(2, 2);
  attr.kernel = HW(2, 2);
  attr.type = PoolingType::AVERAGE;

  auto src_shape = BHWC(3, 8, 6, 5);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  return AveragePoolingTest(env, attr, src_tensor, op_def);
}

absl::Status AveragePoolingNonEmptyPaddingBigTest(TestExecutionEnvironment& env,
                                                  DataType data_type,
                                                  TensorStorageType storage) {
  Pooling2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(1, 1);
  attr.strides = HW(1, 1);
  attr.kernel = HW(2, 2);
  attr.type = PoolingType::AVERAGE;

  auto src_shape = BHWC(1, 4, 4, 1);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return AveragePoolingTest(env, attr, src_tensor, op_def);
}

absl::Status AveragePooling3DBigTest(TestExecutionEnvironment& env,
                                     DataType data_type,
                                     TensorStorageType storage) {
  Pooling3DAttributes attr;
  attr.padding.prepended = HWD(0, 2, 1);
  attr.padding.appended = HWD(1, 1, 2);
  attr.strides = HWD(1, 1, 2);
  attr.kernel = HWD(2, 3, 2);
  attr.type = PoolingType::AVERAGE;

  auto src_shape = BHWDC(1, 4, 7, 6, 5);

  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWDC});
  return AveragePooling3DTest(env, attr, src_tensor, op_def);
}

absl::Status AveragePooling3DBatchedBigTest(TestExecutionEnvironment& env,
                                            DataType data_type,
                                            TensorStorageType storage) {
  Pooling3DAttributes attr;
  attr.padding.prepended = HWD(0, 2, 1);
  attr.padding.appended = HWD(1, 1, 2);
  attr.strides = HWD(1, 1, 2);
  attr.kernel = HWD(2, 3, 2);
  attr.type = PoolingType::AVERAGE;

  auto src_shape = BHWDC(7, 4, 7, 6, 5);

  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWDC});
  return AveragePooling3DTest(env, attr, src_tensor, op_def);
}

absl::Status MaxPoolingBigTest(TestExecutionEnvironment& env,
                               DataType data_type, TensorStorageType storage) {
  Pooling2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(2, 2);
  attr.kernel = HW(2, 2);
  attr.type = PoolingType::MAX;

  auto src_shape = BHWC(1, 8, 6, 5);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return MaxPoolingTest(env, attr, src_tensor, op_def);
}

absl::Status MaxPoolingBatchedBigTest(TestExecutionEnvironment& env,
                                      DataType data_type,
                                      TensorStorageType storage) {
  Pooling2DAttributes attr;
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(0, 2);
  attr.strides = HW(2, 2);
  attr.kernel = HW(2, 3);
  attr.type = PoolingType::MAX;

  auto src_shape = BHWC(7, 8, 6, 5);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  return MaxPoolingTest(env, attr, src_tensor, op_def);
}

absl::Status MaxPooling3DBigTest(TestExecutionEnvironment& env,
                                 DataType data_type,
                                 TensorStorageType storage) {
  Pooling3DAttributes attr;
  attr.padding.prepended = HWD(0, 0, 0);
  attr.padding.appended = HWD(0, 0, 0);
  attr.strides = HWD(2, 2, 2);
  attr.kernel = HWD(2, 2, 2);
  attr.type = PoolingType::MAX;

  auto src_shape = BHWDC(1, 8, 6, 3, 5);

  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWDC});
  return MaxPooling3DTest(env, attr, src_tensor, op_def);
}

absl::Status MaxPooling3DBatchedBigTest(TestExecutionEnvironment& env,
                                        DataType data_type,
                                        TensorStorageType storage) {
  Pooling3DAttributes attr;
  attr.padding.prepended = HWD(1, 1, 0);
  attr.padding.appended = HWD(0, 2, 1);
  attr.strides = HWD(2, 2, 2);
  attr.kernel = HWD(2, 3, 2);
  attr.type = PoolingType::MAX;

  auto src_shape = BHWDC(7, 8, 6, 7, 5);

  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWDC});
  return MaxPooling3DTest(env, attr, src_tensor, op_def);
}

absl::Status MaxPoolingIndicesBigTest(TestExecutionEnvironment& env,
                                      DataType data_type,
                                      TensorStorageType storage) {
  Pooling2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(2, 2);
  attr.kernel = HW(2, 2);
  attr.type = PoolingType::MAX;
  attr.output_indices = true;

  auto src_shape = BHWC(1, 8, 6, 5);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    float t = src_tensor.data[i];
    int quantized = t * 100.0f;
    src_tensor.data[i] = static_cast<float>(quantized) / 100.0f;
  }

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return MaxPoolingIndicesTest(env, attr, src_tensor, op_def);
}

absl::Status MaxPoolingIndicesBatchedBigTest(TestExecutionEnvironment& env,
                                             DataType data_type,
                                             TensorStorageType storage) {
  Pooling2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(2, 2);
  attr.kernel = HW(2, 2);
  attr.type = PoolingType::MAX;
  attr.output_indices = true;

  auto src_shape = BHWC(5, 8, 6, 5);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    float t = src_tensor.data[i];
    int quantized = t * 100.0f;
    src_tensor.data[i] = static_cast<float>(quantized) / 100.0f;
  }

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  return MaxPoolingIndicesTest(env, attr, src_tensor, op_def);
}

absl::Status MaxPoolingIndices3DBigTest(TestExecutionEnvironment& env,
                                        DataType data_type,
                                        TensorStorageType storage) {
  Pooling3DAttributes attr;
  attr.padding.prepended = HWD(0, 0, 0);
  attr.padding.appended = HWD(0, 0, 0);
  attr.strides = HWD(2, 2, 2);
  attr.kernel = HWD(2, 2, 2);
  attr.type = PoolingType::MAX;
  attr.output_indices = true;

  auto src_shape = BHWDC(1, 8, 6, 5, 5);

  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    float t = src_tensor.data[i];
    int quantized = t * 100.0f;
    src_tensor.data[i] = static_cast<float>(quantized) / 100.0f;
  }

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWDC});
  return MaxPoolingIndices3DTest(env, attr, src_tensor, op_def);
}

absl::Status MaxPoolingIndices3DBatchedBigTest(TestExecutionEnvironment& env,
                                               DataType data_type,
                                               TensorStorageType storage) {
  Pooling3DAttributes attr;
  attr.padding.prepended = HWD(0, 0, 0);
  attr.padding.appended = HWD(0, 0, 0);
  attr.strides = HWD(2, 2, 2);
  attr.kernel = HWD(2, 2, 2);
  attr.type = PoolingType::MAX;
  attr.output_indices = true;

  auto src_shape = BHWDC(5, 8, 6, 7, 5);

  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    float t = src_tensor.data[i];
    int quantized = t * 100.0f;
    src_tensor.data[i] = static_cast<float>(quantized) / 100.0f;
  }

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWDC});
  return MaxPoolingIndices3DTest(env, attr, src_tensor, op_def);
}

}  // namespace ml_drift
