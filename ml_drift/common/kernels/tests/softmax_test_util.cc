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

#include "ml_drift/common/kernels/tests/softmax_test_util.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernels/softmax.h"
#include "ml_drift/common/kernels/softmax1x1.h"
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

namespace {

template <typename T>
absl::Status RuntimeChannelsTest(
    std::function<T(const OperationDef&, const GpuInfo&, const BHWC&,
                    const SoftmaxRuntimeCheckDesc&)>
        create_softmax,
    std::function<T(const OperationDef&, const GpuInfo&, const BHWC&,
                    const SoftmaxRuntimeCheckDesc&)>
        create_softmax_reduce,
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(2, 1, 1, 6);
  int ch = src_tensor.shape.c;
  src_tensor.data = {std::log(1.0f), std::log(1.0f), std::log(1.0f),
                     std::log(1.0f), std::log(2.0f), std::log(4.0f),
                     std::log(1.0f), std::log(1.0f), std::log(1.0f),
                     std::log(1.0f), std::log(2.0f), std::log(4.0f)};
  std::vector<float> post_exp(ch);
  std::transform(src_tensor.data.begin(), src_tensor.data.begin() + ch,
                 post_exp.begin(), [](float x) { return std::exp(x); });
  TensorInt32 params_tensor;
  params_tensor.shape = BHWC(1, 1, 1, 1);

  for (int k = 1; k <= ch; ++k) {
      const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
      SoftmaxRuntimeCheckDesc runtime_check = {.end_ch_index = 0};
      T operation = create_softmax(op_def, env.GetGpuInfo(), src_tensor.shape,
                                   runtime_check);

      TensorDescriptor src_td = op_def.src_tensors[0];
      src_td.UploadData(src_tensor);
      TensorDescriptor params_td = {DataType::INT32, TensorStorageType::BUFFER,
                                    Layout::HWC};

      params_tensor.data = {k};
      params_td.UploadData(params_tensor);
      TensorDescriptor dst_td = op_def.dst_tensors[0];
      dst_td.SetBHWCShape(src_tensor.shape);
      RETURN_IF_ERROR(
          env.ExecuteGPUOperation({&src_td, &params_td}, {&dst_td},
                                  std::make_unique<T>(std::move(operation))));
      TensorFloat32 dst_tensor;
      dst_td.DownloadData(&dst_tensor);

      std::vector<float> expected_results(k);
      float sum_window =
          std::accumulate(post_exp.begin(), post_exp.begin() + k, 0.0f);
      std::transform(post_exp.begin(), post_exp.begin() + k,
                     expected_results.begin(),
                     [&sum_window](float x) { return x / sum_window; });

      // We only check the elements within the sliding window of each batch
      // because we make no guarantees about the values past the
      // runtime-specified channel bounds.
      std::vector<float> first_batch(dst_tensor.data.begin(),
                                     dst_tensor.data.begin() + k);
      EXPECT_THAT(first_batch, Pointwise(FloatNear(eps), expected_results));
      std::vector<float> second_batch(dst_tensor.data.begin() + ch,
                                      dst_tensor.data.begin() + ch + k);
      EXPECT_THAT(second_batch, Pointwise(FloatNear(eps), expected_results));

      T operation_part0 = create_softmax_reduce(
          op_def, env.GetGpuInfo(), src_tensor.shape, runtime_check);
      TensorDescriptor exp_td = {data_type, storage, Layout::BHWC};
      exp_td.SetBHWCShape(
          BHWC(src_tensor.shape.b, src_tensor.shape.h, src_tensor.shape.w, 2));
      RETURN_IF_ERROR(env.ExecuteGPUOperation(
          {&src_td, &params_td}, {&exp_td},
          std::make_unique<T>(std::move(operation_part0))));
      OperationDef op_def_part1;
      op_def_part1.src_tensors.push_back({data_type, storage, Layout::BHWC});
      op_def_part1.src_tensors.push_back({data_type, storage, Layout::BHWC});
      op_def_part1.dst_tensors.push_back({data_type, storage, Layout::BHWC});
      GPUOperation operation_part1 = CreateSoftmaxFinal(op_def_part1);
      RETURN_IF_ERROR(env.ExecuteGPUOperation(
          {&src_td, &exp_td}, {&dst_td},
          std::make_unique<GPUOperation>(std::move(operation_part1))));
      TensorFloat32 reduce_dst_tensor;
      dst_td.DownloadData(&reduce_dst_tensor);
      std::vector<float> first_batch_reduce(reduce_dst_tensor.data.begin(),
                                            reduce_dst_tensor.data.begin() + k);
      EXPECT_THAT(first_batch_reduce,
                  Pointwise(FloatNear(eps), expected_results));
      std::vector<float> second_batch_reduce(
          reduce_dst_tensor.data.begin() + ch,
          reduce_dst_tensor.data.begin() + ch + k);
      EXPECT_THAT(second_batch_reduce,
                  Pointwise(FloatNear(eps), expected_results));
  }
  return absl::OkStatus();
}

absl::Status SoftmaxTest(std::unique_ptr<GPUOperation>&& operation,
                         TestExecutionEnvironment& exec_env,
                         const BHWC& src_shape, const OperationDef& op_def) {
  SoftmaxAttributes attr;
  attr.axis = Axis::CHANNELS;
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  TensorFloat32 dst_ref_tensor = SoftmaxReference(attr, src_tensor);

  TensorFloat32 dst_tensor;
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(src_tensor, std::move(operation),
                                         dst_ref_tensor.shape, &dst_tensor));
  const float eps =
      op_def.src_tensors[0].GetDataType() == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status SoftmaxReduceTest(std::unique_ptr<GPUOperation>&& operation,
                               TestExecutionEnvironment& exec_env,
                               const BHWC& src_shape,
                               const OperationDef& op_def) {
  SoftmaxAttributes attr;
  attr.axis = Axis::CHANNELS;
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  TensorFloat32 dst_ref_tensor = SoftmaxReduceReference(attr, src_tensor);

  TensorFloat32 dst_tensor;
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(src_tensor, std::move(operation),
                                         dst_ref_tensor.shape, &dst_tensor));
  const float eps =
      op_def.src_tensors[0].GetDataType() == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status SoftmaxReduceRuntimeChannelsTest(
    std::unique_ptr<GPUOperation>&& operation,
    TestExecutionEnvironment& exec_env, const TensorFloat32& src_tensor,
    int channels_count, const OperationDef& op_def) {
  SoftmaxAttributes softmax_attr;
  softmax_attr.axis = Axis::CHANNELS;
  SliceAttributes slice_attr;
  slice_attr.starts = BHWC(0, 0, 0, 0);
  slice_attr.starts.c = 0;
  slice_attr.ends = src_tensor.shape;
  slice_attr.ends.c = channels_count;
  slice_attr.strides = BHWC(1, 1, 1, 1);
  TensorFloat32 temp = SliceReference(slice_attr, src_tensor).value();
  TensorFloat32 dst_ref_tensor = SoftmaxReduceReference(softmax_attr, temp);

  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.UploadData(src_tensor);
  TensorInt32 params_tensor;
  params_tensor.shape = BHWC(1, 1, 1, 1);
  params_tensor.data = {channels_count};
  TensorDescriptor params_td = {DataType::INT32, TensorStorageType::BUFFER,
                                Layout::HWC};
  params_td.UploadData(params_tensor);
  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(dst_ref_tensor.shape);

  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation({&src_td, &params_td}, {&dst_td},
                                         std::move(operation)));
  TensorFloat32 dst_tensor;
  dst_td.DownloadData(&dst_tensor);
  const float eps =
      op_def.src_tensors[0].GetDataType() == DataType::FLOAT32 ? 1e-6f : 2e-3f;
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_tensor.data));
  return absl::OkStatus();
}

template <typename T>
absl::Status RuntimeChannelsTest(
    std::function<T(const OperationDef&, const GpuInfo&, const BHWC&,
                    const SoftmaxRuntimeCheckDesc&)>
        create_softmax,
    TestExecutionEnvironment& exec_env, const TensorFloat32& src_tensor,
    const OperationDef& op_def) {
  SoftmaxRuntimeCheckDesc runtime_check = {.end_ch_index = 0};

  std::vector<float> post_exp(src_tensor.data.size());
  std::transform(src_tensor.data.begin(), src_tensor.data.end(),
                 post_exp.begin(), [](float x) { return std::exp(x); });

  for (int k = 1; k <= src_tensor.shape.c; k += 64) {
    TensorDescriptor src_td = op_def.src_tensors[0];
    src_td.UploadData(src_tensor);
    TensorInt32 params_tensor;
    params_tensor.shape = BHWC(1, 1, 1, 1);
    params_tensor.data = {k};
    TensorDescriptor params_td = {DataType::INT32, TensorStorageType::BUFFER,
                                  Layout::HWC};
    params_td.UploadData(params_tensor);
    TensorDescriptor dst_td = op_def.dst_tensors[0];
    dst_td.SetBHWCShape(src_tensor.shape);
    T operation = create_softmax(op_def, exec_env.GetGpuInfo(),
                                 src_tensor.shape, runtime_check);
    MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
        {&src_td, &params_td}, {&dst_td},
        std::make_unique<T>(std::move(operation))));
    TensorFloat32 dst_tensor;
    dst_td.DownloadData(&dst_tensor);
    for (int i = 0; i < post_exp.size(); i += src_tensor.shape.c) {
      int results_size = k;
      // Only compare the entries within the window.
      std::vector<float> expected_results(results_size);
      float sum = std::accumulate(post_exp.begin() + i + k - results_size,
                                  post_exp.begin() + i + k, 0.0f);
      std::transform(post_exp.begin() + i + k - results_size,
                     post_exp.begin() + i + k, expected_results.begin(),
                     [&sum](float x) { return x / sum; });
      std::vector<float> actual_results(
          dst_tensor.data.begin() + i + k - results_size,
          dst_tensor.data.begin() + i + k);
      const float eps = op_def.src_tensors[0].GetDataType() == DataType::FLOAT32
                            ? 1e-6f
                            : 1e-3f;
      EXPECT_THAT(actual_results, Pointwise(FloatNear(eps), expected_results));
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status SoftmaxTest(TestExecutionEnvironment& env, DataType data_type,
                         TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {std::log(1.0f), std::log(2.0f), std::log(3.0f),
                     std::log(4.0f)};
  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  Softmax operation = CreateSoftmax(op_def, env.GetGpuInfo(), src_tensor.shape);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Softmax>(std::move(operation)),
      BHWC(1, 2, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps),
                        {1.0f / 3.0f, 2.0f / 3.0f, 3.0f / 7.0f, 4.0f / 7.0f}));

  TensorFloat32 exp_tensor;
  Softmax op_part0 =
      CreateSoftmaxReduce(op_def, env.GetGpuInfo(), src_tensor.shape);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Softmax>(std::move(op_part0)),
      BHWC(1, 2, 1, 2), &exp_tensor));

  OperationDef op_def_part1;
  op_def_part1.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def_part1.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def_part1.dst_tensors.push_back({data_type, storage, Layout::HWC});
  GPUOperation op_part1 = CreateSoftmaxFinal(op_def_part1);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor, exp_tensor},
      std::make_unique<GPUOperation>(std::move(op_part1)), BHWC(1, 2, 1, 2),
      &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps),
                        {1.0f / 3.0f, 2.0f / 3.0f, 3.0f / 7.0f, 4.0f / 7.0f}));
  return absl::OkStatus();
}

absl::Status SoftmaxWGTest(TestExecutionEnvironment& env, DataType data_type,
                           TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 256);
  src_tensor.data = std::vector<float>(256, 0.0f);

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  Softmax operation =
      CreateSoftmaxReduce(op_def, env.GetGpuInfo(), src_tensor.shape);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Softmax>(std::move(operation)),
      BHWC(1, 1, 1, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), {1.0f / 256.0f}));
  return absl::OkStatus();
}

absl::Status SoftmaxReduceTest(TestExecutionEnvironment& env,
                               DataType data_type, TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 5);
  src_tensor.data = {std::log(1.0f), std::log(2.0f), std::log(3.0f),
                     std::log(4.0f), -std::numeric_limits<float>::infinity()};

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

  TensorFloat32 exp_tensor;
  Softmax op_part0 =
      CreateSoftmaxReduce(op_def, env.GetGpuInfo(), src_tensor.shape);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Softmax>(std::move(op_part0)),
      BHWC(1, 1, 1, 2), &exp_tensor));
  EXPECT_THAT(exp_tensor.data,
              Pointwise(FloatNear(eps), {4.0f / 10.0f, std::log(4.0f)}));
  return absl::OkStatus();
}

absl::Status SoftmaxBigNumberTest(TestExecutionEnvironment& env,
                                  DataType data_type,
                                  TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  double doubles[4] = {1.0, 2.0, 3.0, 100.0};
  // exp(100) is inf in float (32 bit) but representable in double (64 bit)
  src_tensor.data.resize(4);
  src_tensor.data[0] = doubles[0];
  src_tensor.data[1] = doubles[1];
  src_tensor.data[2] = doubles[2];
  src_tensor.data[3] = doubles[3];
  if (!std::isinf(std::exp(src_tensor.data[3]))) {
    return absl::InternalError("exp(100.0f) not inf in float (32 bit)");
  }
  if (std::isinf(std::exp(doubles[3]))) {
    return absl::InternalError("exp(100.0) inf in double (64 bit)");
  }
  double s0 = std::exp(doubles[0]) + std::exp(doubles[1]);
  double s1 = std::exp(doubles[2]) + std::exp(doubles[3]);

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  Softmax operation = CreateSoftmax(op_def, env.GetGpuInfo(), src_tensor.shape);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Softmax>(std::move(operation)),
      BHWC(1, 2, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps),
                        {static_cast<float>(std::exp(doubles[0]) / s0),
                         static_cast<float>(std::exp(doubles[1]) / s0),
                         static_cast<float>(std::exp(doubles[2]) / s1),
                         static_cast<float>(std::exp(doubles[3]) / s1)}));

  TensorFloat32 exp_tensor;
  Softmax op_part0 =
      CreateSoftmaxReduce(op_def, env.GetGpuInfo(), src_tensor.shape);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Softmax>(std::move(op_part0)),
      BHWC(1, 2, 1, 2), &exp_tensor));

  OperationDef op_def_part1;
  op_def_part1.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def_part1.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def_part1.dst_tensors.push_back({data_type, storage, Layout::HWC});
  GPUOperation op_part1 = CreateSoftmaxFinal(op_def_part1);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor, exp_tensor},
      std::make_unique<GPUOperation>(std::move(op_part1)), BHWC(1, 2, 1, 2),
      &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps),
                        {static_cast<float>(std::exp(doubles[0]) / s0),
                         static_cast<float>(std::exp(doubles[1]) / s0),
                         static_cast<float>(std::exp(doubles[2]) / s1),
                         static_cast<float>(std::exp(doubles[3]) / s1)}));
  return absl::OkStatus();
}

absl::Status SoftmaxRuntimeChannelsTest(TestExecutionEnvironment& env,
                                        DataType data_type,
                                        TensorStorageType storage) {
  return RuntimeChannelsTest<Softmax>(CreateSoftmax, CreateSoftmaxReduce, env,
                                      data_type, storage);
}

absl::Status SoftmaxBigTest(TestExecutionEnvironment& env, DataType data_type,
                            TensorStorageType storage) {
  auto src_shape = BHWC(1, 3, 2, 7);
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  auto op = CreateSoftmax(op_def, env.GetGpuInfo(), src_shape);
  return SoftmaxTest(std::make_unique<Softmax>(std::move(op)), env, src_shape,
                     op_def);
}

absl::Status SoftmaxBatchedBigTest(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage) {
  auto src_shape = BHWC(5, 3, 2, 7);
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  auto op = CreateSoftmax(op_def, env.GetGpuInfo(), src_shape);
  return SoftmaxTest(std::make_unique<Softmax>(std::move(op)), env, src_shape,
                     op_def);
}

absl::Status SoftmaxReduceBigTest(TestExecutionEnvironment& env,
                                  DataType data_type,
                                  TensorStorageType storage) {
  auto src_shape = BHWC(1, 1, 128, 512);
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  Softmax operation = CreateSoftmaxReduce(op_def, env.GetGpuInfo(), src_shape);
  return SoftmaxReduceTest(std::make_unique<Softmax>(std::move(operation)), env,
                           src_shape, op_def);
}

absl::Status SoftmaxReduceRuntimeChannelsBigTest(TestExecutionEnvironment& env,
                                                 DataType data_type,
                                                 TensorStorageType storage) {
  auto src_shape = BHWC(1, 1, 128, 512);
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  {
    SoftmaxRuntimeCheckDesc runtime_check = {.end_ch_index = 0};

    Softmax operation = CreateSoftmaxReduce(op_def, env.GetGpuInfo(),
                                            src_tensor.shape, runtime_check);
    MLD_EXPECT_OK(SoftmaxReduceRuntimeChannelsTest(
        std::make_unique<Softmax>(std::move(operation)), env, src_tensor,
        /*channels_count=*/12, op_def));
  }
  return absl::OkStatus();
}

absl::Status SoftmaxRuntimeChannelsBigTest(TestExecutionEnvironment& env,
                                           DataType data_type,
                                           TensorStorageType storage) {
  auto src_shape = BHWC(2, 1, 2, 256);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  MLD_EXPECT_OK(
      RuntimeChannelsTest<Softmax>(CreateSoftmax, env, src_tensor, op_def));
  return absl::OkStatus();
}

absl::Status Softmax1x1Test(TestExecutionEnvironment& env, DataType data_type,
                            TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 4);
  src_tensor.data = {std::log(1.0f), std::log(2.0f), std::log(3.0f),
                     std::log(4.0f)};

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  Softmax1x1 operation =
      CreateSoftmax1x1(op_def, env.GetGpuInfo(), src_tensor.shape);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Softmax1x1>(std::move(operation)),
      BHWC(1, 1, 1, 4), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {0.1f, 0.2f, 0.3f, 0.4f}));

  TensorFloat32 exp_tensor;
  Softmax1x1 op_part0 =
      CreateSoftmax1x1Reduce(op_def, env.GetGpuInfo(), src_tensor.shape);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Softmax1x1>(std::move(op_part0)),
      BHWC(1, 1, 1, 4), &exp_tensor));

  OperationDef op_def_part1;
  op_def_part1.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def_part1.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def_part1.dst_tensors.push_back({data_type, storage, Layout::HWC});
  GPUOperation op_part1 = CreateSoftmaxFinal(op_def_part1);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor, exp_tensor},
      std::make_unique<GPUOperation>(std::move(op_part1)), BHWC(1, 1, 1, 4),
      &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps), {0.1f, 0.2f, 0.3f, 0.4f}));
  return absl::OkStatus();
}

absl::Status Softmax1x1BigNumberTest(TestExecutionEnvironment& env,
                                     DataType data_type,
                                     TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 4);
  double doubles[4] = {1.0, 2.0, 3.0, 100.0};
  // exp(100) is inf in float (32 bit) but representable in double (64 bit)
  src_tensor.data.resize(4);
  src_tensor.data[0] = doubles[0];
  src_tensor.data[1] = doubles[1];
  src_tensor.data[2] = doubles[2];
  src_tensor.data[3] = doubles[3];
  if (!std::isinf(std::exp(src_tensor.data[3]))) {
    return absl::InternalError("exp(100.0f) not inf in float (32 bit)");
  }
  if (std::isinf(std::exp(doubles[3]))) {
    return absl::InternalError("exp(100.0) inf in double (64 bit)");
  }
  double s0 = std::exp(doubles[0]) + std::exp(doubles[1]) +
              std::exp(doubles[2]) + std::exp(doubles[3]);

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  Softmax1x1 operation =
      CreateSoftmax1x1(op_def, env.GetGpuInfo(), src_tensor.shape);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Softmax1x1>(std::move(operation)),
      BHWC(1, 1, 1, 4), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps),
                        {static_cast<float>(std::exp(doubles[0]) / s0),
                         static_cast<float>(std::exp(doubles[1]) / s0),
                         static_cast<float>(std::exp(doubles[2]) / s0),
                         static_cast<float>(std::exp(doubles[3]) / s0)}));

  TensorFloat32 exp_tensor;
  Softmax1x1 op_part0 =
      CreateSoftmax1x1Reduce(op_def, env.GetGpuInfo(), src_tensor.shape);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Softmax1x1>(std::move(op_part0)),
      BHWC(1, 1, 1, 4), &exp_tensor));

  OperationDef op_def_part1;
  op_def_part1.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def_part1.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def_part1.dst_tensors.push_back({data_type, storage, Layout::HWC});
  GPUOperation op_part1 = CreateSoftmaxFinal(op_def_part1);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor, exp_tensor},
      std::make_unique<GPUOperation>(std::move(op_part1)), BHWC(1, 1, 1, 4),
      &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps),
                        {static_cast<float>(std::exp(doubles[0]) / s0),
                         static_cast<float>(std::exp(doubles[1]) / s0),
                         static_cast<float>(std::exp(doubles[2]) / s0),
                         static_cast<float>(std::exp(doubles[3]) / s0)}));
  return absl::OkStatus();
}

absl::Status Softmax1x1RuntimeChannelsTest(TestExecutionEnvironment& env,
                                           DataType data_type,
                                           TensorStorageType storage) {
  return RuntimeChannelsTest<Softmax1x1>(
      CreateSoftmax1x1, CreateSoftmax1x1Reduce, env, data_type, storage);
}

absl::Status Softmax1x1Custom1Test(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 128);
  src_tensor.data.resize(128, -2.381976325057654e+38);
  src_tensor.data[127] = 1;

  std::vector<float> expected_results(128, 0.0f);
  expected_results[127] = 1.0f;

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::FLOAT32, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::FLOAT32, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  Softmax1x1 operation =
      CreateSoftmax1x1(op_def, env.GetGpuInfo(), src_tensor.shape);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Softmax1x1>(std::move(operation)),
      BHWC(1, 1, 1, 128), &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), expected_results));
  return absl::OkStatus();
}

absl::Status Softmax1x1BigTest(TestExecutionEnvironment& env,
                               DataType data_type, TensorStorageType storage) {
  auto src_shape = BHWC(1, 1, 1, 129);
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  auto op = CreateSoftmax1x1(op_def, env.GetGpuInfo(), src_shape);
  return SoftmaxTest(std::make_unique<Softmax1x1>(std::move(op)), env,
                     src_shape, op_def);
}

absl::Status Softmax1x1BatchedBigTest(TestExecutionEnvironment& env,
                                      DataType data_type,
                                      TensorStorageType storage) {
  auto src_shape = BHWC(3, 1, 1, 129);
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  auto op = CreateSoftmax1x1(op_def, env.GetGpuInfo(), src_shape);
  return SoftmaxTest(std::make_unique<Softmax1x1>(std::move(op)), env,
                     src_shape, op_def);
}

absl::Status Softmax1x1ReduceBigTest(TestExecutionEnvironment& env,
                                     DataType data_type,
                                     TensorStorageType storage) {
  auto src_shape = BHWC(1, 1, 4, 512);
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  Softmax1x1 operation =
      CreateSoftmax1x1Reduce(op_def, env.GetGpuInfo(), src_shape);
  return SoftmaxReduceTest(std::make_unique<Softmax1x1>(std::move(operation)),
                           env, src_shape, op_def);
}

absl::Status Softmax1x1ReduceRuntimeChannelsBigTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage) {
  auto src_shape = BHWC(1, 1, 4, 512);
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  {
    SoftmaxRuntimeCheckDesc runtime_check = {.end_ch_index = 0};

    Softmax1x1 operation = CreateSoftmax1x1Reduce(
        op_def, env.GetGpuInfo(), src_tensor.shape, runtime_check);
    MLD_EXPECT_OK(SoftmaxReduceRuntimeChannelsTest(
        std::make_unique<Softmax1x1>(std::move(operation)), env, src_tensor,
        /*channels_count=*/12, op_def));
  }
  return absl::OkStatus();
}

absl::Status Softmax1x1RuntimeChannelsBigTest(TestExecutionEnvironment& env,
                                              DataType data_type,
                                              TensorStorageType storage) {
  auto src_shape = BHWC(2, 2, 1, 256);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  MLD_EXPECT_OK(RuntimeChannelsTest<Softmax1x1>(CreateSoftmax1x1, env, src_tensor,
                                            op_def));
  return absl::OkStatus();
}

absl::Status Softmax5DTest(TestExecutionEnvironment& env, DataType data_type,
                           TensorStorageType storage) {
  BHWDC src_shape(1, 1, 2, 2, 2);  // B, H, W, D, C
  auto src_tensor = MakeSyntheticTensor(src_shape);
  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWDC});

  // Reference implementation: flatten to 4D, run reference, restore to 5D.
  TensorFloat32 flattened_src;
  flattened_src.shape =
      BHWC(src_shape.b, src_shape.h, src_shape.w * src_shape.d, src_shape.c);
  flattened_src.data = src_tensor.data;

  SoftmaxAttributes attr;
  attr.axis = Axis::CHANNELS;
  TensorFloat32 dst_ref_tensor = SoftmaxReference(attr, flattened_src);

  Tensor5DFloat32 dst_ref_5d;
  dst_ref_5d.shape = src_shape;
  dst_ref_5d.data = dst_ref_tensor.data;

  // Create operation
  // We need to pass a BHWC shape to CreateSoftmax where Height is H * D.
  BHWC collapsed_shape(src_shape.b, src_shape.h * src_shape.d, src_shape.w,
                       src_shape.c);
  Softmax operation = CreateSoftmax(op_def, env.GetGpuInfo(), collapsed_shape);

  Tensor5DFloat32 dst_tensor;
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Softmax>(std::move(operation)),
      dst_ref_5d.shape, &dst_tensor));

  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_5d.data));
  return absl::OkStatus();
}

absl::Status Softmax1x15DTest(TestExecutionEnvironment& env, DataType data_type,
                              TensorStorageType storage) {
  BHWDC src_shape(1, 1, 2, 2, 2);  // B, H, W, D, C
  auto src_tensor = MakeSyntheticTensor(src_shape);
  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWDC});

  // Reference implementation: flatten to 4D, run reference, restore to 5D.
  TensorFloat32 flattened_src;
  flattened_src.shape =
      BHWC(src_shape.b, src_shape.h, src_shape.w * src_shape.d, src_shape.c);
  flattened_src.data = src_tensor.data;

  SoftmaxAttributes attr;
  attr.axis = Axis::CHANNELS;
  TensorFloat32 dst_ref_tensor = SoftmaxReference(attr, flattened_src);

  Tensor5DFloat32 dst_ref_5d;
  dst_ref_5d.shape = src_shape;
  dst_ref_5d.data = dst_ref_tensor.data;

  // Create operation
  // We need to pass a BHWC shape to CreateSoftmax1x1 where Height is H * D.
  BHWC collapsed_shape(src_shape.b, src_shape.h * src_shape.d, src_shape.w,
                       src_shape.c);
  Softmax1x1 operation =
      CreateSoftmax1x1(op_def, env.GetGpuInfo(), collapsed_shape);

  Tensor5DFloat32 dst_tensor;
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Softmax1x1>(std::move(operation)),
      dst_ref_5d.shape, &dst_tensor));

  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref_5d.data));
  return absl::OkStatus();
}

}  // namespace ml_drift
