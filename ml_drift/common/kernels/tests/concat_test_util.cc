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

#include "ml_drift/common/kernels/tests/concat_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/concat_xy.h"
#include "ml_drift/common/kernels/concat_z.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_ref_ops.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace {

absl::Status ConcatTest(TestExecutionEnvironment& exec_env,
                        const ConcatAttributes& attr,
                        const std::vector<TensorFloat32>& src_tensors,
                        const OperationDef& op_def) {
  TensorFloat32 dst_ref_tensor = ConcatReference(attr, src_tensors);
  GPUOperation operation = CreateConcatXY(op_def, attr);
  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_tensors, std::make_unique<GPUOperation>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.001f), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConcatTest(TestExecutionEnvironment& exec_env,
                        const ConcatAttributes& attr,
                        const std::vector<Tensor5DFloat32>& src_tensors,
                        const OperationDef& op_def) {
  Tensor5DFloat32 dst_ref_tensor = ConcatReference(attr, src_tensors);

  GPUOperation operation = CreateConcatXY(op_def, attr);
  Tensor5DFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_tensors, std::make_unique<GPUOperation>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.001f), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConcatChannelsTest(TestExecutionEnvironment& exec_env,
                                const ConcatAttributes& attr,
                                const std::vector<TensorFloat32>& src_tensors,
                                const OperationDef& op_def) {
  const int tensors_count = src_tensors.size();
  std::vector<int> channels(tensors_count);
  for (int i = 0; i < tensors_count; ++i) {
    channels[i] = src_tensors[i].shape.c;
  }
  TensorFloat32 dst_ref_tensor = ConcatReference(attr, src_tensors);

  GPUOperation operation =
      CreateConcatZ(op_def, channels, exec_env.GetGpuInfo());
  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_tensors, std::make_unique<GPUOperation>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.001f), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status ConcatChannelsTest(TestExecutionEnvironment& exec_env,
                                const ConcatAttributes& attr,
                                const std::vector<Tensor5DFloat32>& src_tensors,
                                const OperationDef& op_def) {
  Tensor5DFloat32 dst_ref_tensor = ConcatReference(attr, src_tensors);

  const int tensors_count = src_tensors.size();
  std::vector<int> channels(tensors_count);
  for (int i = 0; i < tensors_count; ++i) {
    channels[i] = src_tensors[i].shape.c;
  }

  GPUOperation operation =
      CreateConcatZ(op_def, channels, exec_env.GetGpuInfo());
  Tensor5DFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_tensors, std::make_unique<GPUOperation>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.001f), dst_ref_tensor.data));
  return absl::OkStatus();
}

}  // namespace

absl::Status ConcatChannelsBoolTest(TestExecutionEnvironment& env,
                                    TensorStorageType storage) {
  TensorBool src0, src1, src2;
  src0.shape = BHWC(1, 2, 1, 1);
  src0.data = {true, false};
  src1.shape = BHWC(1, 2, 1, 2);
  src1.data = {true, true, true, false};
  src2.shape = BHWC(1, 2, 1, 3);
  src2.data = {true, true, false, false, true, true};

  ConcatAttributes attr;
  attr.axis = Axis::kChannels;

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::kBool, storage, Layout::kHWC});
  op_def.src_tensors.push_back({DataType::kBool, storage, Layout::kHWC});
  op_def.src_tensors.push_back({DataType::kBool, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({DataType::kBool, storage, Layout::kHWC});

  TensorDescriptor src_0, src_1, src_2, dst;
  src_0 = op_def.src_tensors[0];
  src_1 = op_def.src_tensors[1];
  src_2 = op_def.src_tensors[2];
  src_0.UploadData(src0);
  src_1.UploadData(src1);
  src_2.UploadData(src2);
  dst.SetBHWCShape(BHWC(1, 2, 1, 6));

  GPUOperation operation = CreateConcatZ(op_def, {1, 2, 3}, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0, &src_1, &src_2}, {&dst},
      std::make_unique<GPUOperation>(std::move(operation))));
  std::vector<unsigned char> ref_data = {true,  true, true,  true,  true, false,
                                         false, true, false, false, true, true};
  TensorBool dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_data);
  return absl::OkStatus();
}

template <DataType data_type>
absl::Status ConcatIntTest(TestExecutionEnvironment& env,
                           TensorStorageType storage) {
  Tensor<BHWC, data_type> src0, src1;
  src0.shape = BHWC(1, 2, 1, 2);
  src0.data = {0, 1, 2, 3};
  src1.shape = BHWC(1, 2, 2, 2);
  src1.data = {4, 5, 6, 7, 8, 9, 10, 11};

  ConcatAttributes attr;
  attr.axis = Axis::kWidth;

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});
  TensorDescriptor src_0, src_1, dst;
  src_0 = op_def.src_tensors[0];
  src_1 = op_def.src_tensors[1];
  src_0.UploadData(src0);
  src_1.UploadData(src1);
  dst.SetBHWCShape(BHWC(1, 2, 3, 2));

  GPUOperation operation = CreateConcatXY(op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0, &src_1}, {&dst},
      std::make_unique<GPUOperation>(std::move(operation))));
  Tensor<BHWC, data_type> dst_tensor;
  dst.DownloadData(&dst_tensor);
  Tensor<BHWC, data_type> ref_tensor;
  ref_tensor.data = {0, 1, 4, 5, 6, 7, 2, 3, 8, 9, 10, 11};
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

template absl::Status ConcatIntTest<DataType::kInt8>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status ConcatIntTest<DataType::kInt16>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status ConcatIntTest<DataType::kInt32>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status ConcatIntTest<DataType::kUint8>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status ConcatIntTest<DataType::kUint16>(
    TestExecutionEnvironment& env, TensorStorageType storage);
template absl::Status ConcatIntTest<DataType::kUint32>(
    TestExecutionEnvironment& env, TensorStorageType storage);

absl::Status ConcatWidthTest(TestExecutionEnvironment& env, DataType data_type,
                             TensorStorageType storage) {
  TensorFloat32 src0, src1;
  src0.shape = BHWC(1, 2, 1, 2);
  src0.data = {0.0f, -1.0f, -0.05f, 0.045f};
  src1.shape = BHWC(1, 2, 2, 2);
  src1.data = {1.0f, -1.2f, -0.45f, 1.045f, 1.1f, -1.3f, -0.55f, 2.045f};

  ConcatAttributes attr;
  attr.axis = Axis::kWidth;

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateConcatXY(op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src0, src1}, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 3, 2), &dst_tensor));
  const float eps = data_type == DataType::kFloat32 ? 0.0f : 1e-3f;
  EXPECT_THAT(
      dst_tensor.data,
      Pointwise(FloatNear(eps), {0.0f, -1.0f, 1.0f, -1.2f, -0.45f, 1.045f,
                                 -0.05f, 0.045f, 1.1f, -1.3f, -0.55f, 2.045f}));
  return absl::OkStatus();
}

absl::Status ConcatHeightTest(TestExecutionEnvironment& env, DataType data_type,
                              TensorStorageType storage) {
  TensorFloat32 src0, src1;
  src0.shape = BHWC(1, 2, 1, 2);
  src0.data = {0.0, -1.0, -0.05, 0.045};
  src1.shape = BHWC(1, 1, 1, 2);
  src1.data = {1.0, -1.2};

  ConcatAttributes attr;
  attr.axis = Axis::kHeight;

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateConcatXY(op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src0, src1}, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 3, 1, 2), &dst_tensor));
  const float eps = data_type == DataType::kFloat32 ? 0.0f : 1e-3f;
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {0.0, -1.0, -0.05, 0.045, 1.0, -1.2}));
  return absl::OkStatus();
}

absl::Status ConcatChannelsTest(TestExecutionEnvironment& env,
                                DataType data_type, TensorStorageType storage) {
  TensorFloat32 src0, src1, src2;
  src0.shape = BHWC(1, 2, 1, 1);
  src0.data = {0.0, -1.0};
  src1.shape = BHWC(1, 2, 1, 2);
  src1.data = {1.0, 2.0, 3.0, 4.0};
  src2.shape = BHWC(1, 2, 1, 3);
  src2.data = {5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

  ConcatAttributes attr;
  attr.axis = Axis::kChannels;

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateConcatZ(op_def, {1, 2, 3}, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src0, src1, src2}, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 6), &dst_tensor));
  const float eps = data_type == DataType::kFloat32 ? 0.0f : 1e-3f;
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {0.0, 1.0, 2.0, 5.0, 6.0, 7.0, -1.0,
                                         3.0, 4.0, 8.0, 9.0, 10.0}));
  return absl::OkStatus();
}

absl::Status ConcatChannelsAlignedx4Test(TestExecutionEnvironment& env,
                                         DataType data_type,
                                         TensorStorageType storage) {
  TensorFloat32 src0, src1;
  src0.shape = BHWC(1, 2, 1, 4);
  src0.data = {-1.0, -2.0, -3.0, -4.0, 1.0, 2.0, 3.0, 4.0};
  src1.shape = BHWC(1, 2, 1, 4);
  src1.data = {5.0, 6.0, 7.0, 8.0, -5.0, -6.0, -7.0, -8.0};

  ConcatAttributes attr;
  attr.axis = Axis::kChannels;

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateConcatZ(op_def, {4, 4}, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src0, src1}, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 1, 8), &dst_tensor));
  const float eps = data_type == DataType::kFloat32 ? 0.0f : 1e-3f;
  EXPECT_THAT(
      dst_tensor.data,
      Pointwise(FloatNear(eps), {-1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0,
                                 1.0, 2.0, 3.0, 4.0, -5.0, -6.0, -7.0, -8.0}));
  return absl::OkStatus();
}

absl::Status ConcatWidthBigTest(TestExecutionEnvironment& env,
                                DataType data_type, TensorStorageType storage) {
  const int tensors_count = 7;
  const int height = 5;
  const int channels = 7;
  std::vector<TensorFloat32> src_tensors(tensors_count);
  for (int i = 0; i < tensors_count; ++i) {
    int width = i + 2;
    const auto src_shape = BHWC(1, height, width, channels);
    src_tensors[i] = MakeSyntheticTensor(src_shape);
  }

  ConcatAttributes attr;
  attr.axis = Axis::kWidth;

  OperationDef op_def;
  for (int i = 0; i < src_tensors.size(); ++i) {
    op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  }
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});
  return ConcatTest(env, attr, src_tensors, op_def);
}

absl::Status ConcatWidthBatchedBigTest(TestExecutionEnvironment& env,
                                       DataType data_type,
                                       TensorStorageType storage) {
  const int tensors_count = 7;
  const int height = 5;
  const int channels = 7;
  const int batch_size = 6;
  std::vector<TensorFloat32> src_tensors(tensors_count);
  for (int i = 0; i < tensors_count; ++i) {
    int width = i + 2;
    const auto src_shape = BHWC(batch_size, height, width, channels);
    src_tensors[i] = MakeSyntheticTensor(src_shape);
  }

  ConcatAttributes attr;
  attr.axis = Axis::kWidth;

  OperationDef op_def;
  for (int i = 0; i < src_tensors.size(); ++i) {
    op_def.src_tensors.push_back({data_type, storage, Layout::kBHWC});
  }
  op_def.dst_tensors.push_back({data_type, storage, Layout::kBHWC});
  return ConcatTest(env, attr, src_tensors, op_def);
}

absl::Status ConcatHeightBigTest(TestExecutionEnvironment& env,
                                 DataType data_type,
                                 TensorStorageType storage) {
  const int tensors_count = 7;
  const int width = 11;
  const int channels = 7;
  std::vector<TensorFloat32> src_tensors(tensors_count);
  for (int i = 0; i < tensors_count; ++i) {
    int height = i + 2;
    const auto src_shape = BHWC(1, height, width, channels);
    src_tensors[i] = MakeSyntheticTensor(src_shape);
  }

  ConcatAttributes attr;
  attr.axis = Axis::kHeight;

  OperationDef op_def;
  for (int i = 0; i < src_tensors.size(); ++i) {
    op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  }
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});
  return ConcatTest(env, attr, src_tensors, op_def);
}

absl::Status ConcatHeightBatchedBigTest(TestExecutionEnvironment& env,
                                        DataType data_type,
                                        TensorStorageType storage) {
  const int tensors_count = 7;
  const int width = 11;
  const int channels = 7;
  const int batch_size = 2;
  std::vector<TensorFloat32> src_tensors(tensors_count);
  for (int i = 0; i < tensors_count; ++i) {
    int height = i + 2;
    const auto src_shape = BHWC(batch_size, height, width, channels);
    src_tensors[i] = MakeSyntheticTensor(src_shape);
  }

  ConcatAttributes attr;
  attr.axis = Axis::kHeight;

  OperationDef op_def;
  for (int i = 0; i < src_tensors.size(); ++i) {
    op_def.src_tensors.push_back({data_type, storage, Layout::kBHWC});
  }
  op_def.dst_tensors.push_back({data_type, storage, Layout::kBHWC});
  return ConcatTest(env, attr, src_tensors, op_def);
}

absl::Status ConcatBatchBigTest(TestExecutionEnvironment& env,
                                DataType data_type, TensorStorageType storage) {
  const int tensors_count = 7;
  const int width = 4;
  const int height = 5;
  const int channels = 5;
  std::vector<TensorFloat32> src_tensors(tensors_count);
  for (int i = 0; i < tensors_count; ++i) {
    int batch = i + 2;
    const auto src_shape = BHWC(batch, height, width, channels);
    src_tensors[i] = MakeSyntheticTensor(src_shape);
  }

  ConcatAttributes attr;
  attr.axis = Axis::kBatch;

  OperationDef op_def;
  for (int i = 0; i < src_tensors.size(); ++i) {
    op_def.src_tensors.push_back({data_type, storage, Layout::kBHWC});
  }
  op_def.dst_tensors.push_back({data_type, storage, Layout::kBHWC});
  return ConcatTest(env, attr, src_tensors, op_def);
}

absl::Status ConcatDepthBigTest(TestExecutionEnvironment& env,
                                DataType data_type, TensorStorageType storage) {
  const int tensors_count = 7;
  const int width = 4;
  const int height = 5;
  const int channels = 5;
  std::vector<Tensor5DFloat32> src_tensors(tensors_count);
  for (int i = 0; i < tensors_count; ++i) {
    int depth = i + 2;
    const auto src_shape = BHWDC(1, height, width, depth, channels);
    src_tensors[i] = MakeSyntheticTensor(src_shape);
  }

  ConcatAttributes attr;
  attr.axis = Axis::kDepth;

  OperationDef op_def;
  for (int i = 0; i < src_tensors.size(); ++i) {
    op_def.src_tensors.push_back({data_type, storage, Layout::kHWDC});
  }
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWDC});
  return ConcatTest(env, attr, src_tensors, op_def);
}

absl::Status ConcatDepthBatchedBigTest(TestExecutionEnvironment& env,
                                       DataType data_type,
                                       TensorStorageType storage) {
  const int tensors_count = 7;
  const int width = 4;
  const int height = 5;
  const int channels = 5;
  std::vector<Tensor5DFloat32> src_tensors(tensors_count);
  for (int i = 0; i < tensors_count; ++i) {
    int depth = i + 2;
    const auto src_shape = BHWDC(3, height, width, depth, channels);
    src_tensors[i] = MakeSyntheticTensor(src_shape);
  }

  ConcatAttributes attr;
  attr.axis = Axis::kDepth;

  OperationDef op_def;
  for (int i = 0; i < src_tensors.size(); ++i) {
    op_def.src_tensors.push_back({data_type, storage, Layout::kBHWDC});
  }
  op_def.dst_tensors.push_back({data_type, storage, Layout::kBHWDC});
  return ConcatTest(env, attr, src_tensors, op_def);
}

absl::Status ConcatChannelsBigTest(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage) {
  const int tensors_count = 7;
  const int width = 7;
  const int height = 4;
  std::vector<TensorFloat32> src_tensors(tensors_count);
  for (int i = 0; i < tensors_count; ++i) {
    const auto src_shape = BHWC(1, height, width, i + 1);
    src_tensors[i] = MakeSyntheticTensor(src_shape);
  }

  ConcatAttributes attr;
  attr.axis = Axis::kChannels;

  OperationDef op_def;
  for (int i = 0; i < src_tensors.size(); ++i) {
    op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  }
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});
  return ConcatChannelsTest(env, attr, src_tensors, op_def);
}

absl::Status ConcatChannelsBatchedBigTest(TestExecutionEnvironment& env,
                                          DataType data_type,
                                          TensorStorageType storage) {
  const int tensors_count = 7;
  const int width = 7;
  const int height = 4;
  const int batch_size = 7;
  std::vector<TensorFloat32> src_tensors(tensors_count);
  for (int i = 0; i < tensors_count; ++i) {
    const auto src_shape = BHWC(batch_size, height, width, i + 1);
    src_tensors[i] = MakeSyntheticTensor(src_shape);
  }

  ConcatAttributes attr;
  attr.axis = Axis::kChannels;

  OperationDef op_def;
  for (int i = 0; i < src_tensors.size(); ++i) {
    op_def.src_tensors.push_back({data_type, storage, Layout::kBHWC});
  }
  op_def.dst_tensors.push_back({data_type, storage, Layout::kBHWC});
  return ConcatChannelsTest(env, attr, src_tensors, op_def);
}

absl::Status ConcatChannelsx4BigTest(TestExecutionEnvironment& env,
                                     DataType data_type,
                                     TensorStorageType storage) {
  const int tensors_count = 3;
  const int width = 7;
  const int height = 4;
  std::vector<TensorFloat32> src_tensors(tensors_count);
  const std::vector<int> channels{12, 8, 16};
  for (int i = 0; i < tensors_count; ++i) {
    const auto src_shape = BHWC(1, height, width, channels[i]);
    src_tensors[i] = MakeSyntheticTensor(src_shape);
  }

  ConcatAttributes attr;
  attr.axis = Axis::kChannels;

  OperationDef op_def;
  for (int i = 0; i < src_tensors.size(); ++i) {
    op_def.src_tensors.push_back({data_type, storage, Layout::kHWC});
  }
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});
  return ConcatChannelsTest(env, attr, src_tensors, op_def);
}

absl::Status ConcatChannelsx4BatchedBigTest(TestExecutionEnvironment& env,
                                            DataType data_type,
                                            TensorStorageType storage) {
  const int tensors_count = 3;
  const int width = 7;
  const int height = 4;
  const int batch_size = 9;
  std::vector<TensorFloat32> src_tensors(tensors_count);
  const std::vector<int> channels{12, 8, 16};
  for (int i = 0; i < tensors_count; ++i) {
    const auto src_shape = BHWC(batch_size, height, width, channels[i]);
    src_tensors[i] = MakeSyntheticTensor(src_shape);
  }

  ConcatAttributes attr;
  attr.axis = Axis::kChannels;

  OperationDef op_def;
  for (int i = 0; i < src_tensors.size(); ++i) {
    op_def.src_tensors.push_back({data_type, storage, Layout::kBHWC});
  }
  op_def.dst_tensors.push_back({data_type, storage, Layout::kBHWC});
  return ConcatChannelsTest(env, attr, src_tensors, op_def);
}

absl::Status ConcatChannelsBHWDCBigTest(TestExecutionEnvironment& env,
                                        DataType data_type,
                                        TensorStorageType storage) {
  const int tensors_count = 7;
  const int width = 7;
  const int height = 4;
  const int depth = 3;
  const int batch_size = 7;
  std::vector<Tensor5DFloat32> src_tensors(tensors_count);
  for (int i = 0; i < tensors_count; ++i) {
    const auto src_shape = BHWDC(batch_size, height, width, depth, i + 1);
    src_tensors[i] = MakeSyntheticTensor(src_shape);
  }

  ConcatAttributes attr;
  attr.axis = Axis::kChannels;

  OperationDef op_def;
  for (int i = 0; i < src_tensors.size(); ++i) {
    op_def.src_tensors.push_back({data_type, storage, Layout::kBHWDC});
  }
  op_def.dst_tensors.push_back({data_type, storage, Layout::kBHWDC});
  return ConcatChannelsTest(env, attr, src_tensors, op_def);
}

}  // namespace ml_drift
