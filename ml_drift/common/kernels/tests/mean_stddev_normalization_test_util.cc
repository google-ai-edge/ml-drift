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

#include "ml_drift/common/kernels/tests/mean_stddev_normalization_test_util.h"

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/mean_stddev_normalization.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
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
TensorFloat32 NormalizeHWCGroupRef(const TensorFloat32& src, int groups,
                                   double variance_bias) {
  TensorFloat32 dst = src;
  const int group_size = src.shape.c / groups;
  const int reduction_size = src.shape.h * src.shape.w * group_size;
  for (int b = 0; b < src.shape.b; ++b) {
    for (int g = 0; g < groups; ++g) {
      double mean = 0.0;
      for (int h = 0; h < src.shape.h; ++h) {
        for (int w = 0; w < src.shape.w; ++w) {
          for (int c = g * group_size; c < (g + 1) * group_size; ++c) {
            mean += src.data[src.shape.LinearIndex({b, h, w, c})];
          }
        }
      }
      mean /= static_cast<double>(reduction_size);
      double variance = 0.0;
      for (int h = 0; h < src.shape.h; ++h) {
        for (int w = 0; w < src.shape.w; ++w) {
          for (int c = g * group_size; c < (g + 1) * group_size; ++c) {
            float val = src.data[src.shape.LinearIndex({b, h, w, c})];
            variance += (val - mean) * (val - mean);
          }
        }
      }
      variance /= static_cast<double>(reduction_size);
      double stddev = sqrt(variance + variance_bias);
      for (int h = 0; h < src.shape.h; ++h) {
        for (int w = 0; w < src.shape.w; ++w) {
          for (int c = g * group_size; c < (g + 1) * group_size; ++c) {
            double val = src.data[src.shape.LinearIndex({b, h, w, c})];
            double normalized = (val - mean) / stddev;
            dst.data[dst.shape.LinearIndex({b, h, w, c})] = normalized;
          }
        }
      }
    }
  }
  return dst;
}

absl::Status MeanStddevNormSeparateBatchesUnit(TestExecutionEnvironment& env,
                                               DataType data_type,
                                               TensorStorageType storage,
                                               float mean, float diff,
                                               float tolerance) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 2, 4);
  src_tensor.data = {mean - 2 * diff, mean - diff,     mean + diff,
                     mean + 2 * diff, mean - 2 * diff, mean - diff,
                     mean + diff,     mean + 2 * diff};
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  auto operation =
      CreateMeanStdDevNormalization(op_def, env.GetGpuInfo(), src_tensor.shape);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor},
      std::make_unique<MeanStdDevNormalization>(std::move(operation)),
      BHWC(1, 1, 2, 4), &dst_tensor));

  std::vector<float> expected_output;
  if (diff == 0.0f) {
    expected_output.assign({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  } else {
    const float ksqrt16 = std::sqrt(1.6f);
    const float ksqrt04 = std::sqrt(0.4f);
    expected_output.assign({-ksqrt16, -ksqrt04, ksqrt04, ksqrt16, -ksqrt16,
                            -ksqrt04, ksqrt04, ksqrt16});
  }
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(tolerance), expected_output));

  TensorFloat32 dst_tensor_single_step;
  auto operation_single_step = CreateMeanStdDevNormalization(
      op_def, env.GetGpuInfo(), src_tensor.shape,
      /*variance_bias=*/1.0e-8f, /*two_step=*/false);
  RETURN_IF_ERROR(
      env.ExecuteGPUOperation({src_tensor},
                              std::make_unique<MeanStdDevNormalization>(
                                  std::move(operation_single_step)),
                              BHWC(1, 1, 2, 4), &dst_tensor_single_step));
  EXPECT_THAT(dst_tensor_single_step.data,
              Pointwise(FloatNear(tolerance), expected_output));
  return absl::OkStatus();
}

}  // namespace

// Parameterized test: mean, difference, tolerance.
// Input is constructed as [mean-2*diff, mean-diff, mean+diff, mean+2*diff]
absl::Status MeanStddevNormSeparateBatchesTest(TestExecutionEnvironment& env,
                                               DataType data_type,
                                               TensorStorageType storage) {
  // zero mean, zero variance
  RETURN_IF_ERROR(MeanStddevNormSeparateBatchesUnit(env, data_type, storage,
                                                    0.0f, 0.0f, 0.0f));

  // zero mean, small variance
  RETURN_IF_ERROR(MeanStddevNormSeparateBatchesUnit(env, data_type, storage,
                                                    0.0f, 0.01f, 2.63e-4f));

  // zero mean, large variance
  RETURN_IF_ERROR(MeanStddevNormSeparateBatchesUnit(env, data_type, storage,
                                                    0.0f, 100.0f, 2.63e-4f));

  // small mean, zero variance
  RETURN_IF_ERROR(MeanStddevNormSeparateBatchesUnit(env, data_type, storage,
                                                    0.01f, 0.0f, 0.0f));

  // small mean, small variance
  RETURN_IF_ERROR(MeanStddevNormSeparateBatchesUnit(env, data_type, storage,
                                                    0.01f, 0.01f, 3.57e-4f));

  // small mean, large variance
  RETURN_IF_ERROR(MeanStddevNormSeparateBatchesUnit(env, data_type, storage,
                                                    1.0f, 100.0f, 2.63e-4f));

  // large mean, zero variance
  RETURN_IF_ERROR(MeanStddevNormSeparateBatchesUnit(env, data_type, storage,
                                                    100.0f, 0.0f, 0.0f));

  // large mean, small variance
  RETURN_IF_ERROR(MeanStddevNormSeparateBatchesUnit(env, data_type, storage,
                                                    100.0f, 1.0f, 2.63e-4f));

  // large mean, large variance
  RETURN_IF_ERROR(MeanStddevNormSeparateBatchesUnit(env, data_type, storage,
                                                    100.0f, 100.0f, 2.63e-4f));

  return absl::OkStatus();
}

absl::Status MeanStddevNormalizationAllBatchesTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(9, 1, 1, 4);
  src_tensor.data = {
      0.0f,    0.0f,    0.0f,   0.0f,    // zero mean, zero variance
      -0.02f,  -0.01f,  0.01f,  0.02f,   // zero mean, small variance
      -200.0f, -100.0f, 100.0f, 200.0f,  // zero mean, large variance
      0.01f,   0.01f,   0.01f,  0.01f,   // small mean, zero variance
      -0.01f,  0.0f,    0.02f,  0.03f,   // small mean, small variance
      -199.0f, -99.0f,  101.0f, 201.0f,  // small mean, large variance
      100.0f,  100.0f,  100.0f, 100.0f,  // large mean, zero variance
      98.0f,   99.0f,   101.0f, 102.0f,  // large mean, small variance
      -100.0f, 0.0f,    200.0f, 300.0f,  // large mean, large variance
  };
  const float eps = data_type == DataType::FLOAT32 ? 2.53e-05f : 5.0e-4f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  TensorFloat32 dst_tensor;
  auto operation = CreateMeanStdDevNormalization(op_def, env.GetGpuInfo(),
                                                  src_tensor.shape);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor},
      std::make_unique<MeanStdDevNormalization>(std::move(operation)),
      BHWC(9, 1, 1, 4), &dst_tensor));

  const float ksqrt16 = std::sqrt(1.6f);
  const float ksqrt04 = std::sqrt(0.4f);
  const std::vector<float> expected_output = {
      0.0f,     0.0f,     0.0f,    0.0f,     // zero mean, zero variance
      -ksqrt16, -ksqrt04, ksqrt04, ksqrt16,  // zero mean, small variance
      -ksqrt16, -ksqrt04, ksqrt04, ksqrt16,  // zero mean, large variance
      0.0f,     0.0f,     0.0f,    0.0f,     // small mean, zero variance
      -ksqrt16, -ksqrt04, ksqrt04, ksqrt16,  // small mean, small variance
      -ksqrt16, -ksqrt04, ksqrt04, ksqrt16,  // small mean, large variance
      0.0f,     0.0f,     0.0f,    0.0f,     // large mean, zero variance
      -ksqrt16, -ksqrt04, ksqrt04, ksqrt16,  // large mean, small variance
      -ksqrt16, -ksqrt04, ksqrt04, ksqrt16,  // large mean, large variance
  };
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), expected_output));

  TensorFloat32 dst_tensor_single_step;
  auto operation_single_step = CreateMeanStdDevNormalization(
      op_def, env.GetGpuInfo(), src_tensor.shape,
      /*variance_bias=*/1.0e-8f, /*two_step=*/false);
  RETURN_IF_ERROR(
      env.ExecuteGPUOperation({src_tensor},
                              std::make_unique<MeanStdDevNormalization>(
                                  std::move(operation_single_step)),
                              BHWC(9, 1, 1, 4), &dst_tensor_single_step));
  EXPECT_THAT(dst_tensor_single_step.data,
              Pointwise(FloatNear(eps), expected_output));
  return absl::OkStatus();
}

absl::Status MeanStddevNormalizationLargeVectorTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage) {
  const float mean = 100.0f;
  const float diff = 1.0f;
  // Some large vector that is not a round multiple of any SIMD vector sizes.
  constexpr int kVectorSize = 16 * 16 + 16 + 1;

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 2, kVectorSize);
  src_tensor.data.resize(kVectorSize * 2);
  // First input is mean.
  src_tensor.data[0] = mean;
  src_tensor.data[kVectorSize] = mean;
  // Rest is alternating between mean + diff and mean - diff.
  for (int i = 1; i < kVectorSize - 1; i += 2) {
    src_tensor.data[i + 0] = mean + diff;
    src_tensor.data[i + 1] = mean - diff;
    src_tensor.data[kVectorSize + i + 0] = mean + diff;
    src_tensor.data[kVectorSize + i + 1] = mean - diff;
  }

  const float eps = data_type == DataType::FLOAT32 ? 5.0e-7f : 8.60e-4f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  auto operation =
      CreateMeanStdDevNormalization(op_def, env.GetGpuInfo(), src_tensor.shape);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor},
      std::make_unique<MeanStdDevNormalization>(std::move(operation)),
      BHWC(1, 1, 2, kVectorSize), &dst_tensor));

  std::vector<float> expected_output(kVectorSize * 2);
  // First output should be 0.
  expected_output[0] = 0.0;
  expected_output[kVectorSize] = 0.0;
  // Rest should be alternating between ±√(N/(N-1)).
  const float expected_elem = std::sqrt(static_cast<double>(kVectorSize) /
                                        static_cast<double>(kVectorSize - 1));
  for (int i = 1; i < kVectorSize - 1; i += 2) {
    expected_output[i + 0] = +expected_elem;
    expected_output[i + 1] = -expected_elem;
    expected_output[kVectorSize + i + 0] = +expected_elem;
    expected_output[kVectorSize + i + 1] = -expected_elem;
  }
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), expected_output));

  if (data_type != DataType::FLOAT32) {
    TensorFloat32 dst_tensor_single_step;
    auto operation_single_step = CreateMeanStdDevNormalization(
        op_def, env.GetGpuInfo(), src_tensor.shape,
        /*variance_bias=*/1.0e-8f, /*two_step=*/false);
    RETURN_IF_ERROR(env.ExecuteGPUOperation(
        {src_tensor},
        std::make_unique<MeanStdDevNormalization>(
            std::move(operation_single_step)),
        BHWC(1, 1, 2, kVectorSize), &dst_tensor_single_step));
    EXPECT_THAT(dst_tensor_single_step.data,
                Pointwise(FloatNear(eps), expected_output));
  }
  return absl::OkStatus();
}

absl::Status HWCGroupNormalizationTest(TestExecutionEnvironment& env,
                                       DataType data_type,
                                       TensorStorageType storage,
                                       int group_size) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 6, 32, 24);
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = std::sin(i);
  }
  const float kVarianceBias = 1.0e-5f;
  Tensor<Linear, DataType::FLOAT32> gamma;
  gamma.shape = Linear(src_tensor.shape.c);
  gamma.data.resize(gamma.shape.DimensionsProduct());
  for (int i = 0; i < gamma.data.size(); ++i) {
    gamma.data[i] = 1.0f;
  }
  Tensor<Linear, DataType::FLOAT32> beta;
  beta.shape = Linear(src_tensor.shape.c);
  beta.data.resize(beta.shape.DimensionsProduct());
  for (int i = 0; i < beta.data.size(); ++i) {
    beta.data[i] = 0.0f;
  }

  const int kNumGroups = src_tensor.shape.c / group_size;
  TensorFloat32 output_ref =
      NormalizeHWCGroupRef(src_tensor, kNumGroups, kVarianceBias);
  const float eps = data_type == DataType::FLOAT32 ? 3e-05f : 2e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;

  HWCGroupNormalization operation =
      HWCGroupNormalization(op_def, env.GetGpuInfo(), src_tensor.shape,
                            kNumGroups, kVarianceBias, gamma, beta);

  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor},
      std::make_unique<HWCGroupNormalization>(std::move(operation)),
      src_tensor.shape, &dst_tensor));

  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), output_ref.data));
  return absl::OkStatus();
}

absl::Status HWCGroupNormalizationBatchTest(TestExecutionEnvironment& env,
                                            DataType data_type,
                                            TensorStorageType storage,
                                            int group_size) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(2, 6, 32, 24);
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = std::sin(i);
  }
  const float kVarianceBias = 1.0e-5f;
  Tensor<Linear, DataType::FLOAT32> gamma;
  gamma.shape = Linear(src_tensor.shape.c);
  gamma.data.resize(gamma.shape.DimensionsProduct());
  for (int i = 0; i < gamma.data.size(); ++i) {
    gamma.data[i] = 1.0f;
  }
  Tensor<Linear, DataType::FLOAT32> beta;
  beta.shape = Linear(src_tensor.shape.c);
  beta.data.resize(beta.shape.DimensionsProduct());
  for (int i = 0; i < beta.data.size(); ++i) {
    beta.data[i] = 0.0f;
  }

  const int kNumGroups = src_tensor.shape.c / group_size;
  TensorFloat32 output_ref =
      NormalizeHWCGroupRef(src_tensor, kNumGroups, kVarianceBias);
  const float eps = data_type == DataType::FLOAT32 ? 3e-05f : 2e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  TensorFloat32 dst_tensor;

  HWCGroupNormalization operation =
      HWCGroupNormalization(op_def, env.GetGpuInfo(), src_tensor.shape,
                            kNumGroups, kVarianceBias, gamma, beta);

  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor},
      std::make_unique<HWCGroupNormalization>(std::move(operation)),
      src_tensor.shape, &dst_tensor));

  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), output_ref.data));
  return absl::OkStatus();
}

absl::Status RMSNormalizationTest(TestExecutionEnvironment& exec_env,
                                  const TensorFloat32& src_tensor,
                                  const OperationDef& op_def,
                                  DataType data_type) {
  float rms_eps = 1e-6f;
  TensorFloat32 dst_ref_tensor = RMSNormalizationReference(src_tensor, rms_eps);

  auto operation = CreateRMSNormalization(op_def, exec_env.GetGpuInfo(),
                                          src_tensor.shape, rms_eps);
  float error_eps =
      GetEpsilon(data_type, exec_env.GetGpuInfo()) * src_tensor.shape.c;
  TensorFloat32 dst_tensor;
  RETURN_IF_ERROR(exec_env.ExecuteGPUOperation(
      src_tensor,
      std::make_unique<MeanStdDevNormalization>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(error_eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status RMSNormalizationBigTest(TestExecutionEnvironment& env,
                                     DataType data_type,
                                     TensorStorageType storage) {
  auto src_shape = BHWC(1, 15, 13, 17);
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  RETURN_IF_ERROR(RMSNormalizationTest(env, src_tensor, op_def, data_type));

  return absl::OkStatus();
}

absl::Status StatisticalTopKTest(TestExecutionEnvironment& env,
                                 DataType data_type,
                                 TensorStorageType storage) {
  auto src_shape = BHWC(1, 15, 13, 17);
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

  const float stddev_multiplier = 1.0f;
  TensorFloat32 dst_ref_tensor =
      StatisticalTopKReference(src_tensor, stddev_multiplier);

  std::unique_ptr<GPUOperation> operation = CreateStatisticalTopK(
      op_def, env.GetGpuInfo(), src_tensor.shape, stddev_multiplier);
  float error_eps =
      GetEpsilon(data_type, env.GetGpuInfo()) * src_tensor.shape.c;
  TensorFloat32 dst_tensor;
  RETURN_IF_ERROR(env.ExecuteGPUOperation(src_tensor, std::move(operation),
                                          dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(error_eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status MeanStddevNormalization5DTest(TestExecutionEnvironment& env,
                                           DataType data_type,
                                           TensorStorageType storage) {
  Tensor<BHWDC, DataType::FLOAT32> src_tensor;
  src_tensor.shape = BHWDC(1, 1, 4, 1, 4);
  src_tensor.data = {
      0.0f,    0.0f,   0.0f,    0.0f,    // zero mean, zero variance
      -0.02f,  0.02f,  -0.02f,  0.02f,   // zero mean, small variance
      -200.0f, 200.0f, -200.0f, 200.0f,  // zero mean, large variance
      100.0f,  100.0f, 100.0f,  100.0f,  // large mean, zero variance
  };
  const float eps = data_type == DataType::FLOAT32 ? 2.53e-05f : 5.0e-4f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWDC});
  Tensor<BHWDC, DataType::FLOAT32> dst_tensor;

  // We mock a BHWC shape for CreateMeanStdDevNormalization since it currently
  // takes BHWC and deduces workgroups based on that.
  BHWC mock_shape = BHWC(1, 1, 4, 4);
  auto operation =
      CreateMeanStdDevNormalization(op_def, env.GetGpuInfo(), mock_shape);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor},
      std::make_unique<MeanStdDevNormalization>(std::move(operation)),
      src_tensor.shape, &dst_tensor));

  const std::vector<float> expected_output = {
      0.0f,  0.0f, 0.0f,  0.0f, -1.0f, 1.0f, -1.0f, 1.0f,
      -1.0f, 1.0f, -1.0f, 1.0f, 0.0f,  0.0f, 0.0f,  0.0f,
  };
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), expected_output));
  return absl::OkStatus();
}

}  // namespace ml_drift
