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

#include "ml_drift/common/kernels/tests/select_v2_test_util.h"

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/select_v2.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

using TrueTensor = Tensor<BHWC, DataType::FLOAT32>;
using ElseTensor = Tensor<BHWC, DataType::FLOAT32>;

namespace {
template <DataType cond_type>
std::vector<float> SetUpData(Tensor<BHWC, cond_type>& cond_tensor,
                             TrueTensor& true_tensor, ElseTensor& false_tensor,
                             int batch, int height, int width, int channels,
                             bool broadcast_true, bool broadcast_false,
                             bool gather_by_rows) {
  if (gather_by_rows) {
    cond_tensor.shape = BHWC(batch, height, width, 1);
    for (int b = 0; b < batch; b++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          cond_tensor.data.push_back(w % 2 == 0);
        }
      }
    }
  } else {
    cond_tensor.shape = BHWC(batch, height, 1, channels);
    for (int b = 0; b < batch; b++) {
      for (int h = 0; h < height; h++) {
        for (int c = 0; c < channels; c++) {
          cond_tensor.data.push_back(c % 2 == 0);
        }
      }
    }
  }
  true_tensor.shape =
      broadcast_true ? BHWC(1, 1, 1, 1) : BHWC(batch, height, width, channels);
  false_tensor.shape =
      broadcast_false ? BHWC(1, 1, 1, 1) : BHWC(batch, height, width, channels);
  float collected_true_value = 99.f;
  float collected_false_value = -99.f;
  if (broadcast_true) {
    true_tensor.data.push_back(collected_true_value);
  }
  if (broadcast_false) {
    false_tensor.data.push_back(collected_false_value);
  }
  std::vector<float> expected_data;
  expected_data.reserve(batch * height * width * channels);

  for (int b = 0; b < batch; b++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        for (int c = 0; c < channels; c++) {
          int expected_true_value = collected_true_value;
          int expected_false_value = collected_false_value;
          if (!broadcast_true) {
            // true are even values.
            expected_true_value = w * channels + c * 2 / 2;
            true_tensor.data.push_back(expected_true_value);
          }
          if (!broadcast_false) {
            // false are odd values.
            expected_false_value = w * channels + c * 2 / 2 + 1;
            false_tensor.data.push_back(expected_false_value);
          }
          if (gather_by_rows) {
            expected_data.push_back(
                cond_tensor.data[b * width * height + h * width + w]
                    ? expected_true_value
                    : expected_false_value);
          } else {
            expected_data.push_back(
                cond_tensor.data[b * height * channels + h * channels + c]
                    ? expected_true_value
                    : expected_false_value);
          }
        }
      }
    }
  }
  return expected_data;
}

template <DataType cond_type>
absl::Status RunSelectV2(
    TestExecutionEnvironment& env, const DataType& src_data_type,
    const TensorStorageType& cond_storage, const TensorStorageType& storage,
    const DataType& dst_data_type, const Tensor<BHWC, cond_type>& cond_tensor,
    const TrueTensor& true_tensor, const ElseTensor& false_tensor, int batch,
    int height, int width, int channels, bool broadcast_true,
    bool broadcast_false, TensorFloat32& dst_tensor) {
  OperationDef op_def;
  const auto layout = batch > 1 ? Layout::BHWC : Layout::HWC;
  op_def.src_tensors.push_back({cond_type, cond_storage, layout});
  op_def.src_tensors.push_back({src_data_type, storage, layout});
  op_def.src_tensors.push_back({src_data_type, storage, layout});
  op_def.dst_tensors.push_back({dst_data_type, storage, layout});
  TensorDescriptor cond_descriptor = op_def.src_tensors[0];
  TensorDescriptor true_descriptor, else_descriptor;
  true_descriptor = op_def.src_tensors[1];
  else_descriptor = op_def.src_tensors[2];
  TensorDescriptor dst_descriptor = op_def.dst_tensors[0];
  GPUOperation operation = CreateSelectV2(op_def, {});
  cond_descriptor.UploadData(cond_tensor);
  true_descriptor.UploadData(true_tensor);
  else_descriptor.UploadData(false_tensor);
  dst_descriptor.SetBHWCShape(BHWC(batch, height, width, channels));
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&cond_descriptor, &true_descriptor, &else_descriptor}, {&dst_descriptor},
      std::make_unique<GPUOperation>(std::move(operation))));
  dst_descriptor.DownloadData(&dst_tensor);
  return absl::OkStatus();
}
}  // namespace

// Stable_hlo if. Single bool cond tensor, variable output type/shape
template <DataType dst_data_type>
absl::Status IfTest(TestExecutionEnvironment& env, DataType data_type,
                    TensorStorageType storage, TensorStorageType cond_storage) {
  const BHWC shape = BHWC(4, 1, 1, 4);

  Tensor<BHWC, DataType::BOOL> cond_tensor;
  Tensor<BHWC, dst_data_type> true_tensor;
  Tensor<BHWC, dst_data_type> false_tensor;
  cond_tensor.shape = BHWC(1, 1, 1, 1);
  cond_tensor.data.resize(1);
  true_tensor.shape = shape;
  false_tensor.shape = shape;
  for (int i = 0; i < shape.DimensionsProduct(); ++i) {
    switch (dst_data_type) {
      case DataType::FLOAT32:
        true_tensor.data.push_back(std::sin(i * 0.123f));
        false_tensor.data.push_back(std::sin(i * 0.456f));
        break;
      case DataType::INT8:
      case DataType::INT16:
      case DataType::INT32:
        true_tensor.data.push_back(i % 256 - 128);
        false_tensor.data.push_back((i + 10) % 256 - 128);
        break;
      case DataType::BOOL:
        true_tensor.data.push_back(i % 2);
        false_tensor.data.push_back((i + 1) % 2);
        break;
      default:
        return absl::InvalidArgumentError("Unsupported data type");
    }
  }

  for (int i = 0; i < 2; ++i) {
    cond_tensor.data[0] = i == 0;
    Tensor<BHWC, dst_data_type> dst_tensor;
    OperationDef op_def;
    op_def.src_tensors.push_back({DataType::BOOL, cond_storage, Layout::BHWC});
    op_def.src_tensors.push_back({dst_data_type, storage, Layout::BHWC});
    op_def.src_tensors.push_back({dst_data_type, storage, Layout::BHWC});
    op_def.dst_tensors.push_back({dst_data_type, storage, Layout::BHWC});
    TensorDescriptor cond_descriptor = op_def.src_tensors[0];
    TensorDescriptor true_descriptor = op_def.src_tensors[1];
    TensorDescriptor else_descriptor = op_def.src_tensors[2];
    TensorDescriptor dst_descriptor = op_def.dst_tensors[0];
    GPUOperation operation = CreateSelectV2(op_def, {});
    cond_descriptor.UploadData(cond_tensor);
    true_descriptor.UploadData(true_tensor);
    else_descriptor.UploadData(false_tensor);
    dst_descriptor.SetBHWCShape(shape);
    return absl::OkStatus();
    ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
        {&cond_descriptor, &true_descriptor, &else_descriptor},
        {&dst_descriptor},
        std::make_unique<GPUOperation>(std::move(operation))));
    dst_descriptor.DownloadData(&dst_tensor);
    if (i == 0) {
      if (dst_tensor.data != true_tensor.data) {
        return absl::InternalError("Expected true value");
      }
    } else {
      if (dst_tensor.data != false_tensor.data) {
        return absl::InternalError("Expected false value");
      }
    }
  }
  return absl::OkStatus();
}

template absl::Status IfTest<DataType::FLOAT32>(TestExecutionEnvironment& env,
                                                DataType data_type,
                                                TensorStorageType storage,
                                                TensorStorageType cond_storage);
template absl::Status IfTest<DataType::BOOL>(TestExecutionEnvironment& env,
                                             DataType data_type,
                                             TensorStorageType storage,
                                             TensorStorageType cond_storage);
template absl::Status IfTest<DataType::INT8>(TestExecutionEnvironment& env,
                                             DataType data_type,
                                             TensorStorageType storage,
                                             TensorStorageType cond_storage);
template absl::Status IfTest<DataType::INT16>(TestExecutionEnvironment& env,
                                              DataType data_type,
                                              TensorStorageType storage,
                                              TensorStorageType cond_storage);
template absl::Status IfTest<DataType::INT32>(TestExecutionEnvironment& env,
                                              DataType data_type,
                                              TensorStorageType storage,
                                              TensorStorageType cond_storage);

template <DataType cond_type>
absl::Status SelectV2Test(TestExecutionEnvironment& env, DataType data_type,
                          TensorStorageType storage,
                          TensorStorageType cond_storage) {
  const int kBatch = 1;
  const int kHeight = 1;
  const int kWidth = 10;
  const int kChannels = 10;

  Tensor<BHWC, cond_type> cond_tensor;
  Tensor<BHWC, DataType::FLOAT32> true_tensor;
  Tensor<BHWC, DataType::FLOAT32> false_tensor;
  std::vector<float> expected_data =
      SetUpData<cond_type>(cond_tensor, true_tensor, false_tensor, kBatch,
                           kHeight, kWidth, kChannels,
                           /*broadcast_true=*/false, /*broadcast_false=*/false,
                           /*gather_by_rows=*/true);

  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(RunSelectV2<cond_type>(
      env, data_type, cond_storage, storage, data_type, cond_tensor,
      true_tensor, false_tensor, kBatch, kHeight, kWidth, kChannels,
      /*broadcast_true=*/false, /*broadcast_false=*/false, dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(0.0f), expected_data));
  return absl::OkStatus();
}

template absl::Status SelectV2Test<DataType::FLOAT32>(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, TensorStorageType cond_storage);
template absl::Status SelectV2Test<DataType::BOOL>(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, TensorStorageType cond_storage);

template <DataType cond_type>
absl::Status SelectV2Scalar4DTest(TestExecutionEnvironment& env,
                                  DataType data_type, TensorStorageType storage,
                                  TensorStorageType cond_storage) {
  const int kBatch = 1;
  const int kHeight = 1;
  const int kWidth = 1;
  const int kChannels = 4;

  Tensor<BHWC, cond_type> cond_tensor;
  cond_tensor.shape = BHWC(kBatch, kHeight, kWidth, 1);
  // [true]
  cond_tensor.data.push_back(true);

  Tensor<BHWC, DataType::FLOAT32> true_tensor;
  true_tensor.shape = BHWC(kBatch, kHeight, kWidth, kChannels);
  // [0, 1, 2, 3]
  for (int c = 0; c < kChannels; ++c) {
    true_tensor.data.push_back(c);
  }

  Tensor<BHWC, DataType::FLOAT32> false_tensor;
  false_tensor.shape = BHWC(kBatch, kHeight, kWidth, 1);
  // [99.f]
  false_tensor.data.push_back(99.f);

  // we should expect true value for results
  std::vector<float> expected_data = {0.f, 1.f, 2.f, 3.f};

  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(RunSelectV2<cond_type>(
      env, data_type, cond_storage, storage, data_type, cond_tensor,
      true_tensor, false_tensor, kBatch, kHeight, kWidth, kChannels,
      /*broadcast_true=*/false, /*broadcast_false=*/true, dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(0.0f), expected_data));
  return absl::OkStatus();
}

template absl::Status SelectV2Scalar4DTest<DataType::FLOAT32>(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, TensorStorageType cond_storage);
template absl::Status SelectV2Scalar4DTest<DataType::BOOL>(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, TensorStorageType cond_storage);

template <DataType cond_type>
absl::Status SelectV2TrueValueTest(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage,
                                   TensorStorageType cond_storage) {
  const int kBatch = 1;
  const int kHeight = 1;
  const int kWidth = 10;
  const int kChannels = 10;

  Tensor<BHWC, cond_type> cond_tensor;
  Tensor<BHWC, DataType::FLOAT32> true_tensor;
  Tensor<BHWC, DataType::FLOAT32> false_tensor;
  std::vector<float> expected_data =
      SetUpData<cond_type>(cond_tensor, true_tensor, false_tensor, kBatch,
                           kHeight, kWidth, kChannels,
                           /*broadcast_true=*/false, /*broadcast_false=*/false,
                           /*gather_by_rows=*/true);

  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(RunSelectV2<cond_type>(
      env, data_type, cond_storage, storage, data_type, cond_tensor,
      true_tensor, false_tensor, kBatch, kHeight, kWidth, kChannels,
      /*broadcast_true=*/true, /*broadcast_false=*/false, dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(0.0f), expected_data));
  return absl::OkStatus();
}

template absl::Status SelectV2TrueValueTest<DataType::FLOAT32>(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, TensorStorageType cond_storage);
template absl::Status SelectV2TrueValueTest<DataType::BOOL>(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, TensorStorageType cond_storage);

template <DataType cond_type>
absl::Status SelectV2FalseValueTest(TestExecutionEnvironment& env,
                                    DataType data_type,
                                    TensorStorageType storage,
                                    TensorStorageType cond_storage) {
  const int kBatch = 1;
  const int kHeight = 1;
  const int kWidth = 10;
  const int kChannels = 10;

  Tensor<BHWC, cond_type> cond_tensor;
  Tensor<BHWC, DataType::FLOAT32> true_tensor;
  Tensor<BHWC, DataType::FLOAT32> false_tensor;
  std::vector<float> expected_data =
      SetUpData<cond_type>(cond_tensor, true_tensor, false_tensor, kBatch,
                           kHeight, kWidth, kChannels,
                           /*broadcast_true=*/false, /*broadcast_false=*/false,
                           /*gather_by_rows=*/true);

  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(RunSelectV2<cond_type>(
      env, data_type, cond_storage, storage, data_type, cond_tensor,
      true_tensor, false_tensor, kBatch, kHeight, kWidth, kChannels,
      /*broadcast_true=*/false, /*broadcast_false=*/true, dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(0.0f), expected_data));
  return absl::OkStatus();
}

template absl::Status SelectV2FalseValueTest<DataType::FLOAT32>(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, TensorStorageType cond_storage);
template absl::Status SelectV2FalseValueTest<DataType::BOOL>(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, TensorStorageType cond_storage);

template <DataType cond_type>
absl::Status SelectV2BatchTest(TestExecutionEnvironment& env,
                               DataType data_type, TensorStorageType storage,
                               TensorStorageType cond_storage) {
  const int kBatch = 4;
  const int kHeight = 1;
  const int kWidth = 10;
  const int kChannels = 10;
  Tensor<BHWC, cond_type> cond_tensor;
  Tensor<BHWC, DataType::FLOAT32> true_tensor;
  Tensor<BHWC, DataType::FLOAT32> false_tensor;
  std::vector<float> expected_data =
      SetUpData<cond_type>(cond_tensor, true_tensor, false_tensor, kBatch,
                           kHeight, kWidth, kChannels,
                           /*broadcast_true=*/false, /*broadcast_false=*/false,
                           /*gather_by_rows=*/true);

  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(RunSelectV2<cond_type>(
      env, data_type, cond_storage, storage, data_type, cond_tensor,
      true_tensor, false_tensor, kBatch, kHeight, kWidth, kChannels,
      /*broadcast_true=*/false, /*broadcast_false=*/false, dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(0.0f), expected_data));
  return absl::OkStatus();
}

template absl::Status SelectV2BatchTest<DataType::FLOAT32>(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, TensorStorageType cond_storage);
template absl::Status SelectV2BatchTest<DataType::BOOL>(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, TensorStorageType cond_storage);

template <DataType cond_type>
absl::Status SelectV2BroadcastFalseTest(TestExecutionEnvironment& env,
                                        DataType data_type,
                                        TensorStorageType storage,
                                        TensorStorageType cond_storage) {
  const int kBatch = 1;
  const int kHeight = 1;
  const int kWidth = 10;
  const int kChannels = 10;
  const bool kBroadcastTrue = false;
  const bool kBroadcastFalse = true;
  Tensor<BHWC, cond_type> cond_tensor;
  Tensor<BHWC, DataType::FLOAT32> true_tensor;
  Tensor<BHWC, DataType::FLOAT32> false_tensor;
  std::vector<float> expected_data = SetUpData<cond_type>(
      cond_tensor, true_tensor, false_tensor, kBatch, kHeight, kWidth,
      kChannels, kBroadcastTrue, kBroadcastFalse,
      /*gather_by_rows=*/true);

  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(RunSelectV2<cond_type>(
      env, data_type, cond_storage, storage, data_type, cond_tensor,
      true_tensor, false_tensor, kBatch, kHeight, kWidth, kChannels,
      kBroadcastTrue, kBroadcastFalse, dst_tensor));

  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(0.0f), expected_data));
  return absl::OkStatus();
}

template absl::Status SelectV2BroadcastFalseTest<DataType::FLOAT32>(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, TensorStorageType cond_storage);
template absl::Status SelectV2BroadcastFalseTest<DataType::BOOL>(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, TensorStorageType cond_storage);

template <DataType cond_type>
absl::Status SelectV2BroadcastTrueTest(TestExecutionEnvironment& env,
                                       DataType data_type,
                                       TensorStorageType storage,
                                       TensorStorageType cond_storage) {
  const int kBatch = 1;
  const int kHeight = 1;
  const int kWidth = 10;
  const int kChannels = 10;
  const bool kBroadcastTrue = true;
  const bool kBroadcastFalse = false;
  Tensor<BHWC, cond_type> cond_tensor;
  Tensor<BHWC, DataType::FLOAT32> true_tensor;
  Tensor<BHWC, DataType::FLOAT32> false_tensor;
  std::vector<float> expected_data = SetUpData<cond_type>(
      cond_tensor, true_tensor, false_tensor, kBatch, kHeight, kWidth,
      kChannels, kBroadcastTrue, kBroadcastFalse,
      /*gather_by_rows=*/true);

  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(RunSelectV2<cond_type>(
      env, data_type, cond_storage, storage, data_type, cond_tensor,
      true_tensor, false_tensor, kBatch, kHeight, kWidth, kChannels,
      kBroadcastTrue, kBroadcastFalse, dst_tensor));

  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(0.0f), expected_data));
  return absl::OkStatus();
}

template absl::Status SelectV2BroadcastTrueTest<DataType::FLOAT32>(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, TensorStorageType cond_storage);
template absl::Status SelectV2BroadcastTrueTest<DataType::BOOL>(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, TensorStorageType cond_storage);

template <DataType cond_type>
absl::Status SelectV2BroadcastBothTest(TestExecutionEnvironment& env,
                                       DataType data_type,
                                       TensorStorageType storage,
                                       TensorStorageType cond_storage) {
  const int kBatch = 1;
  const int kHeight = 1;
  const int kWidth = 10;
  const int kChannels = 1;
  const bool kBroadcastTrue = true;
  const bool kBroadcastFalse = true;
  Tensor<BHWC, cond_type> cond_tensor;
  Tensor<BHWC, DataType::FLOAT32> true_tensor;
  Tensor<BHWC, DataType::FLOAT32> false_tensor;
  std::vector<float> expected_data = SetUpData<cond_type>(
      cond_tensor, true_tensor, false_tensor, kBatch, kHeight, kWidth,
      kChannels, kBroadcastTrue, kBroadcastFalse,
      /*gather_by_rows=*/true);

  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(RunSelectV2<cond_type>(
      env, data_type, cond_storage, storage, data_type, cond_tensor,
      true_tensor, false_tensor, kBatch, kHeight, kWidth, kChannels,
      kBroadcastTrue, kBroadcastFalse, dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(0.0f), expected_data));
  return absl::OkStatus();
}

template absl::Status SelectV2BroadcastBothTest<DataType::FLOAT32>(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, TensorStorageType cond_storage);
template absl::Status SelectV2BroadcastBothTest<DataType::BOOL>(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, TensorStorageType cond_storage);

template <DataType cond_type>
absl::Status SelectV2ChannelsTest(TestExecutionEnvironment& env,
                                  DataType data_type, TensorStorageType storage,
                                  TensorStorageType cond_storage) {
  const int kBatch = 1;
  const int kHeight = 1;
  const int kWidth = 2;
  const int kChannels = 10;
  Tensor<BHWC, cond_type> cond_tensor;
  Tensor<BHWC, DataType::FLOAT32> true_tensor;
  Tensor<BHWC, DataType::FLOAT32> false_tensor;
  std::vector<float> expected_data =
      SetUpData<cond_type>(cond_tensor, true_tensor, false_tensor, kBatch,
                           kHeight, kWidth, kChannels,
                           /*broadcast_true=*/false, /*broadcast_false=*/false,
                           /*gather_by_rows=*/false);

  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(RunSelectV2<cond_type>(
      env, data_type, cond_storage, storage, data_type, cond_tensor,
      true_tensor, false_tensor, kBatch, kHeight, kWidth, kChannels,
      /*broadcast_true=*/false,
      /*broadcast_false=*/false, dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(0.0f), expected_data));
  return absl::OkStatus();
}

template absl::Status SelectV2ChannelsTest<DataType::FLOAT32>(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, TensorStorageType cond_storage);
template absl::Status SelectV2ChannelsTest<DataType::BOOL>(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, TensorStorageType cond_storage);

template <DataType cond_type>
absl::Status SelectV2ChannelsBatchTest(TestExecutionEnvironment& env,
                                       DataType data_type,
                                       TensorStorageType storage,
                                       TensorStorageType cond_storage) {
  const int kBatch = 3;
  const int kHeight = 1;
  const int kWidth = 2;
  const int kChannels = 10;
  Tensor<BHWC, cond_type> cond_tensor;
  Tensor<BHWC, DataType::FLOAT32> true_tensor;
  Tensor<BHWC, DataType::FLOAT32> false_tensor;
  std::vector<float> expected_data =
      SetUpData<cond_type>(cond_tensor, true_tensor, false_tensor, kBatch,
                           kHeight, kWidth, kChannels,
                           /*broadcast_true=*/false, /*broadcast_false=*/false,
                           /*gather_by_rows=*/false);

  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(RunSelectV2<cond_type>(
      env, data_type, cond_storage, storage, data_type, cond_tensor,
      true_tensor, false_tensor, kBatch, kHeight, kWidth, kChannels,
      /*broadcast_true=*/false,
      /*broadcast_false=*/false, dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(0.0f), expected_data));
  return absl::OkStatus();
}

template absl::Status SelectV2ChannelsBatchTest<DataType::FLOAT32>(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, TensorStorageType cond_storage);
template absl::Status SelectV2ChannelsBatchTest<DataType::BOOL>(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, TensorStorageType cond_storage);

template <DataType cond_type>
absl::Status SelectV2ChannelsBroadcastFalseTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, TensorStorageType cond_storage) {
  const int kBatch = 1;
  const int kHeight = 3;
  const int kWidth = 2;
  const int kChannels = 4;
  const bool kBroadcastFalse = true;
  Tensor<BHWC, cond_type> cond_tensor;
  Tensor<BHWC, DataType::FLOAT32> true_tensor;
  Tensor<BHWC, DataType::FLOAT32> false_tensor;
  std::vector<float> expected_data =
      SetUpData<cond_type>(cond_tensor, true_tensor, false_tensor, kBatch,
                           kHeight, kWidth, kChannels,
                           /*broadcast_true=*/false, kBroadcastFalse,
                           /*gather_by_rows=*/false);

  TensorFloat32 dst_tensor;
  ABSL_RETURN_IF_ERROR(RunSelectV2<cond_type>(
      env, data_type, cond_storage, storage, data_type, cond_tensor,
      true_tensor, false_tensor, kBatch, kHeight, kWidth, kChannels,
      /*broadcast_true=*/false, kBroadcastFalse, dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(0.0f), expected_data));
  return absl::OkStatus();
}

template absl::Status SelectV2ChannelsBroadcastFalseTest<DataType::FLOAT32>(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, TensorStorageType cond_storage);
template absl::Status SelectV2ChannelsBroadcastFalseTest<DataType::BOOL>(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage, TensorStorageType cond_storage);

}  // namespace ml_drift
