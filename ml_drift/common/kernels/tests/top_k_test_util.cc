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

#include "ml_drift/common/kernels/tests/top_k_test_util.h"

#include <stdlib.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/top_k.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

absl::Status TopKTest(TestExecutionEnvironment& env, DataType data_type,
                      TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 2, 4);
  src_tensor.data = {7.0f, 1.0f, 13.0f, 12.0f, 5.0f, 4.0f, 2.0f, -9.0f};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  TensorDescriptor src, dst_max, dst_ind;
  src = op_def.src_tensors[0];
  src.UploadData(src_tensor);
  dst_max.SetBHWCShape(BHWC(1, 1, 1, 4));
  dst_ind.SetBHWCShape(BHWC(1, 1, 1, 4));
  TopKOp operation = CreateTopK(env.GetGpuInfo(), op_def);
  ABSL_RETURN_IF_ERROR(
      env.ExecuteGPUOperation({&src}, {&dst_max, &dst_ind},
                              std::make_unique<TopKOp>(std::move(operation))));
  Tensor<BHWC, DataType::FLOAT32> dst_max_tensor;
  dst_max.DownloadData(&dst_max_tensor);
  Tensor<BHWC, DataType::INT32> dst_ind_tensor;
  dst_ind.DownloadData(&dst_ind_tensor);

  EXPECT_EQ(dst_max_tensor.data,
            std::vector<float>({13.0f, 12.0f, 7.0f, 5.0f}));
  EXPECT_EQ(dst_ind_tensor.data, std::vector<int32_t>({2, 3, 0, 4}));
  return absl::OkStatus();
}

absl::Status TopKPartialReductionTest(TestExecutionEnvironment& env,
                                      DataType data_type,
                                      TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 3, 4);
  src_tensor.data = {7.0f,  1.0f, 13.0f, 12.0f, 5.0f,  4.0f,
                     -2.0f, 9.0f, 9.0f,  1.0f,  -3.0f, 2.0f};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  TensorDescriptor src, dst_max, dst_ind;
  src = op_def.src_tensors[0];
  src.UploadData(src_tensor);
  dst_max.SetBHWCShape(BHWC(1, 1, 2, 4));
  dst_ind.SetBHWCShape(BHWC(1, 1, 2, 4));
  TopKOp operation = CreateTopK(env.GetGpuInfo(), op_def);
  ABSL_RETURN_IF_ERROR(
      env.ExecuteGPUOperation({&src}, {&dst_max, &dst_ind},
                              std::make_unique<TopKOp>(std::move(operation))));
  Tensor<BHWC, DataType::FLOAT32> dst_max_tensor;
  dst_max.DownloadData(&dst_max_tensor);
  Tensor<BHWC, DataType::INT32> dst_ind_tensor;
  dst_ind.DownloadData(&dst_ind_tensor);

  EXPECT_EQ(dst_max_tensor.data, std::vector<float>({13.0f, 12.0f, 9.0f, 7.0f,
                                                     9.0f, 2.0f, 1.0f, -3.0f}));
  EXPECT_EQ(dst_ind_tensor.data,
            std::vector<int32_t>({2, 3, 7, 0, 8, 11, 9, 10}));
  return absl::OkStatus();
}

absl::Status TopKBigTest(TestExecutionEnvironment& env, DataType data_type,
                         TensorStorageType storage) {
  const int kTopKSize = 4;
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 64000, 4);
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  unsigned int seed = 0;
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    double rand_value = static_cast<double>(rand_r(&seed)) / RAND_MAX;
    src_tensor.data[i] = static_cast<float>(
        static_cast<half>((rand_value * 2.0 - 1.0) * 1024.0 * 32.0f));
  }

  std::vector<std::pair<float, int32_t>> values(src_tensor.data.size());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    values[i] = {src_tensor.data[i], i};
  }

  std::stable_sort(
      values.begin(), values.end(),
      [](const std::pair<float, int32_t>& a,
         const std::pair<float, int32_t>& b) { return a.first > b.first; });

  std::vector<float> ref_max(kTopKSize);
  std::vector<int32_t> ref_ind(kTopKSize);
  for (int i = 0; i < kTopKSize; ++i) {
    ref_max[i] = static_cast<float>(static_cast<half>(values[i].first));
    ref_ind[i] = values[i].second;
  }

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  TensorDescriptor src, dst_max, dst_ind;
  src = op_def.src_tensors[0];
  src.UploadData(src_tensor);
  dst_max.SetBHWCShape(BHWC(1, 1, 1, kTopKSize));
  dst_ind.SetBHWCShape(BHWC(1, 1, 1, kTopKSize));
  TopKOp operation = CreateTopK(env.GetGpuInfo(), op_def);
  ABSL_RETURN_IF_ERROR(
      env.ExecuteGPUOperation({&src}, {&dst_max, &dst_ind},
                              std::make_unique<TopKOp>(std::move(operation))));
  Tensor<BHWC, DataType::FLOAT32> dst_max_tensor;
  dst_max.DownloadData(&dst_max_tensor);
  Tensor<BHWC, DataType::INT32> dst_ind_tensor;
  dst_ind.DownloadData(&dst_ind_tensor);

  EXPECT_EQ(dst_max_tensor.data, ref_max);
  EXPECT_EQ(dst_ind_tensor.data, ref_ind);
  return absl::OkStatus();
}

absl::Status TopKBig2StepTest(TestExecutionEnvironment& env, DataType data_type,
                              TensorStorageType storage) {
  const int kTopKSize = 4;
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 64000, 4);
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  unsigned int seed = 0;
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    double rand_value = static_cast<double>(rand_r(&seed)) / RAND_MAX;
    src_tensor.data[i] = static_cast<float>(
        static_cast<half>((rand_value * 2.0 - 1.0) * 1024.0 * 32.0f));
  }

  std::vector<std::pair<float, int32_t>> values(src_tensor.data.size());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    values[i] = {src_tensor.data[i], i};
  }

  std::stable_sort(
      values.begin(), values.end(),
      [](const std::pair<float, int32_t>& a,
         const std::pair<float, int32_t>& b) { return a.first > b.first; });

  std::vector<float> ref_max(kTopKSize);
  std::vector<int32_t> ref_ind(kTopKSize);
  for (int i = 0; i < kTopKSize; ++i) {
    ref_max[i] = values[i].first;
    ref_ind[i] = values[i].second;
  }

  OperationDef op_def_first;
  op_def_first.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def_first.dst_tensors.push_back({data_type, storage, Layout::HWC});
  op_def_first.dst_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  TensorDescriptor src, interm_max, interm_ind;
  src = op_def_first.src_tensors[0];
  src.UploadData(src_tensor);
  interm_max.SetBHWCShape(BHWC(1, 1, 1024, 4));
  interm_ind.SetBHWCShape(BHWC(1, 1, 1024, 4));
  TopKOp op_first = CreateTopK(env.GetGpuInfo(), op_def_first);
  ABSL_RETURN_IF_ERROR(
      env.ExecuteGPUOperation({&src}, {&interm_max, &interm_ind},
                              std::make_unique<TopKOp>(std::move(op_first))));
  OperationDef op_def_second;
  op_def_second.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def_second.src_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  op_def_second.dst_tensors.push_back({data_type, storage, Layout::HWC});
  op_def_second.dst_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  TopKOp op_second = CreateTopK(env.GetGpuInfo(), op_def_second);
  TensorDescriptor dst_max, dst_ind;
  src.UploadData(src_tensor);
  dst_max.SetBHWCShape(BHWC(1, 1, 1, kTopKSize));
  dst_ind.SetBHWCShape(BHWC(1, 1, 1, kTopKSize));
  ABSL_RETURN_IF_ERROR(
      env.ExecuteGPUOperation({&interm_max, &interm_ind}, {&dst_max, &dst_ind},
                              std::make_unique<TopKOp>(std::move(op_second))));
  Tensor<BHWC, DataType::FLOAT32> dst_max_tensor;
  dst_max.DownloadData(&dst_max_tensor);
  Tensor<BHWC, DataType::INT32> dst_ind_tensor;
  dst_ind.DownloadData(&dst_ind_tensor);

  EXPECT_EQ(dst_max_tensor.data, ref_max);
  EXPECT_EQ(dst_ind_tensor.data, ref_ind);
  return absl::OkStatus();
}

absl::Status TopKBig2StepFirstStepNoWgReductionTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage) {
  const int kTopKSize = 4;
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 64000, 4);
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  unsigned int seed = 0;
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    double rand_value = static_cast<double>(rand_r(&seed)) / RAND_MAX;
    src_tensor.data[i] = static_cast<float>(
        static_cast<half>((rand_value * 2.0 - 1.0) * 1024.0 * 32.0f));
  }

  std::vector<std::pair<float, int32_t>> values(src_tensor.data.size());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    values[i] = {src_tensor.data[i], i};
  }

  std::stable_sort(
      values.begin(), values.end(),
      [](const std::pair<float, int32_t>& a,
         const std::pair<float, int32_t>& b) { return a.first > b.first; });

  std::vector<float> ref_max(kTopKSize);
  std::vector<int32_t> ref_ind(kTopKSize);
  for (int i = 0; i < kTopKSize; ++i) {
    ref_max[i] = values[i].first;
    ref_ind[i] = values[i].second;
  }

  OperationDef op_def_first;
  op_def_first.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def_first.dst_tensors.push_back({data_type, storage, Layout::HWC});
  op_def_first.dst_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  TensorDescriptor src, interm_max, interm_ind;
  src = op_def_first.src_tensors[0];
  src.UploadData(src_tensor);
  interm_max.SetBHWCShape(BHWC(1, 1, 1024, 4));
  interm_ind.SetBHWCShape(BHWC(1, 1, 1024, 4));
  TopKOp op_first = CreateTopK(env.GetGpuInfo(), op_def_first,
                               /*use_wg_reduction*/ false);
  ABSL_RETURN_IF_ERROR(
      env.ExecuteGPUOperation({&src}, {&interm_max, &interm_ind},
                              std::make_unique<TopKOp>(std::move(op_first))));
  OperationDef op_def_second;
  op_def_second.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def_second.src_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  op_def_second.dst_tensors.push_back({data_type, storage, Layout::HWC});
  op_def_second.dst_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  TopKOp op_second = CreateTopK(env.GetGpuInfo(), op_def_second);
  TensorDescriptor dst_max, dst_ind;
  src.UploadData(src_tensor);
  dst_max.SetBHWCShape(BHWC(1, 1, 1, kTopKSize));
  dst_ind.SetBHWCShape(BHWC(1, 1, 1, kTopKSize));
  ABSL_RETURN_IF_ERROR(
      env.ExecuteGPUOperation({&interm_max, &interm_ind}, {&dst_max, &dst_ind},
                              std::make_unique<TopKOp>(std::move(op_second))));
  Tensor<BHWC, DataType::FLOAT32> dst_max_tensor;
  dst_max.DownloadData(&dst_max_tensor);
  Tensor<BHWC, DataType::INT32> dst_ind_tensor;
  dst_ind.DownloadData(&dst_ind_tensor);

  EXPECT_EQ(dst_max_tensor.data, ref_max);
  EXPECT_EQ(dst_ind_tensor.data, ref_ind);
  return absl::OkStatus();
}

absl::Status TopKIterativeTest(TestExecutionEnvironment& env,
                               DataType data_type, TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 2, 4);
  src_tensor.data = {7.0f, 1.0f, 13.0f, 12.0f, 5.0f, 4.0f, 2.0f, -9.0f};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  TensorDescriptor src, dst_max, dst_ind;
  TensorDescriptor prev_max, prev_ind;
  src = op_def.src_tensors[0];
  src.UploadData(src_tensor);
  dst_max.SetBHWCShape(BHWC(1, 1, 1, 8));
  dst_ind.SetBHWCShape(BHWC(1, 1, 1, 8));
  Tensor<BHWC, DataType::FLOAT32> dst_max_tensor;
  dst_max_tensor.shape = BHWC(1, 1, 1, 8);
  dst_max_tensor.data.resize(dst_max_tensor.shape.DimensionsProduct());
  Tensor<BHWC, DataType::INT32> dst_ind_tensor;
  dst_ind_tensor.shape = BHWC(1, 1, 1, 8);
  dst_ind_tensor.data.resize(dst_ind_tensor.shape.DimensionsProduct());
  for (int i = 0; i < 2; ++i) {
    int k_offset = i * 4;
    TopKOp operation = CreateTopK(env.GetGpuInfo(), op_def,
                                  /*use_wg_reduction*/ true, k_offset);
    std::vector<TensorDescriptor*> src_cpu = {&src};
    std::vector<TensorDescriptor*> dst_cpu = {&dst_max, &dst_ind};
    if (i != 0) {
      src_cpu.push_back(&prev_max);
      src_cpu.push_back(&prev_ind);
    }
    ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
        src_cpu, dst_cpu, std::make_unique<TopKOp>(std::move(operation))));
    prev_max = dst_max;
    prev_ind = dst_ind;
    Tensor<BHWC, DataType::FLOAT32> temp_max_tensor;
    dst_max.DownloadData(&temp_max_tensor);
    Tensor<BHWC, DataType::INT32> temp_ind_tensor;
    dst_ind.DownloadData(&temp_ind_tensor);
    for (int j = 0; j < 4; ++j) {
      dst_max_tensor.data[i * 4 + j] = temp_max_tensor.data[i * 4 + j];
      dst_ind_tensor.data[i * 4 + j] = temp_ind_tensor.data[i * 4 + j];
    }
  }

  EXPECT_EQ(dst_max_tensor.data, std::vector<float>({13.0f, 12.0f, 7.0f, 5.0f,
                                                     4.0f, 2.0f, 1.0f, -9.0f}));
  EXPECT_EQ(dst_ind_tensor.data,
            std::vector<int32_t>({2, 3, 0, 4, 5, 6, 1, 7}));
  return absl::OkStatus();
}

absl::Status TopKIterativeBigTest(TestExecutionEnvironment& env,
                                  DataType data_type,
                                  TensorStorageType storage) {
  const int kTopKSize = 40;
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 64000, 4);
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  unsigned int seed = 0;
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    double rand_value = static_cast<double>(rand_r(&seed)) / RAND_MAX;
    src_tensor.data[i] = static_cast<float>(
        static_cast<half>((rand_value * 2.0 - 1.0) * 1024.0 * 32.0f));
  }

  std::vector<std::pair<float, int32_t>> values(src_tensor.data.size());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    values[i] = {src_tensor.data[i], i};
  }

  std::stable_sort(
      values.begin(), values.end(),
      [](const std::pair<float, int32_t>& a,
         const std::pair<float, int32_t>& b) { return a.first > b.first; });

  std::vector<float> ref_max(kTopKSize);
  std::vector<int32_t> ref_ind(kTopKSize);
  for (int i = 0; i < kTopKSize; ++i) {
    ref_max[i] = values[i].first;
    ref_ind[i] = values[i].second;
  }

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  TensorDescriptor src, dst_max, dst_ind;
  TensorDescriptor prev_max, prev_ind;
  src = op_def.src_tensors[0];
  src.UploadData(src_tensor);
  dst_max.SetBHWCShape(BHWC(1, 1, 1, kTopKSize));
  dst_ind.SetBHWCShape(BHWC(1, 1, 1, kTopKSize));
  Tensor<BHWC, DataType::FLOAT32> dst_max_tensor;
  dst_max_tensor.shape = BHWC(1, 1, 1, kTopKSize);
  dst_max_tensor.data.resize(dst_max_tensor.shape.DimensionsProduct());
  Tensor<BHWC, DataType::INT32> dst_ind_tensor;
  dst_ind_tensor.shape = BHWC(1, 1, 1, kTopKSize);
  dst_ind_tensor.data.resize(dst_ind_tensor.shape.DimensionsProduct());
  for (int i = 0; i < kTopKSize / 4; ++i) {
    int k_offset = i * 4;
    TopKOp operation = CreateTopK(env.GetGpuInfo(), op_def,
                                  /*use_wg_reduction*/ true, k_offset);
    std::vector<TensorDescriptor*> src_cpu = {&src};
    std::vector<TensorDescriptor*> dst_cpu = {&dst_max, &dst_ind};
    if (i != 0) {
      src_cpu.push_back(&prev_max);
      src_cpu.push_back(&prev_ind);
    }
    ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
        src_cpu, dst_cpu, std::make_unique<TopKOp>(std::move(operation))));
    prev_max = dst_max;
    prev_ind = dst_ind;
    Tensor<BHWC, DataType::FLOAT32> temp_max_tensor;
    dst_max.DownloadData(&temp_max_tensor);
    Tensor<BHWC, DataType::INT32> temp_ind_tensor;
    dst_ind.DownloadData(&temp_ind_tensor);
    for (int j = 0; j < 4; ++j) {
      dst_max_tensor.data[i * 4 + j] = temp_max_tensor.data[i * 4 + j];
      dst_ind_tensor.data[i * 4 + j] = temp_ind_tensor.data[i * 4 + j];
    }
  }
  EXPECT_EQ(dst_max_tensor.data, ref_max);
  EXPECT_EQ(dst_ind_tensor.data, ref_ind);
  return absl::OkStatus();
}

absl::Status TopKIterative2StepBigTest(TestExecutionEnvironment& env,
                                       TensorStorageType storage) {
  const int kTopKSize = 40;
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 64000, 4);
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  unsigned int seed = 0;
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    double rand_value = static_cast<double>(rand_r(&seed)) / RAND_MAX;
    src_tensor.data[i] = (rand_value * 2.0 - 1.0) * 1024.0 * 32.0f;
  }

  std::vector<std::pair<float, int32_t>> values(src_tensor.data.size());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    values[i] = {src_tensor.data[i], i};
  }

  std::stable_sort(
      values.begin(), values.end(),
      [](const std::pair<float, int32_t>& a,
         const std::pair<float, int32_t>& b) { return a.first > b.first; });

  std::vector<float> ref_max(kTopKSize);
  std::vector<int32_t> ref_ind(kTopKSize);
  for (int i = 0; i < kTopKSize; ++i) {
    ref_max[i] = values[i].first;
    ref_ind[i] = values[i].second;
  }

  OperationDef op_def_first;
  const DataType data_type = DataType::FLOAT32;
  op_def_first.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def_first.dst_tensors.push_back({data_type, storage, Layout::HWC});
  op_def_first.dst_tensors.push_back({DataType::INT32, storage, Layout::HWC});

  OperationDef op_def_second;
  op_def_second.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def_second.src_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  op_def_second.dst_tensors.push_back({data_type, storage, Layout::HWC});
  op_def_second.dst_tensors.push_back({DataType::INT32, storage, Layout::HWC});

  TensorDescriptor src, dst_max, dst_ind, interm_max, interm_ind;
  TensorDescriptor prev_max, prev_ind;
  src = op_def_first.src_tensors[0];
  src.UploadData(src_tensor);
  dst_max.SetBHWCShape(BHWC(1, 1, 1, kTopKSize));
  dst_ind.SetBHWCShape(BHWC(1, 1, 1, kTopKSize));
  interm_max.SetBHWCShape(BHWC(1, 1, 1024, 4));
  interm_ind.SetBHWCShape(BHWC(1, 1, 1024, 4));
  Tensor<BHWC, DataType::FLOAT32> dst_max_tensor;
  dst_max_tensor.shape = BHWC(1, 1, 1, kTopKSize);
  dst_max_tensor.data.resize(dst_max_tensor.shape.DimensionsProduct());
  Tensor<BHWC, DataType::INT32> dst_ind_tensor;
  dst_ind_tensor.shape = BHWC(1, 1, 1, kTopKSize);
  dst_ind_tensor.data.resize(dst_ind_tensor.shape.DimensionsProduct());
  for (int i = 0; i < kTopKSize / 4; ++i) {
    int k_offset = i * 4;
    TopKOp op_first = CreateTopK(env.GetGpuInfo(), op_def_first,
                                 /*use_wg_reduction*/ false, k_offset);
    std::vector<TensorDescriptor*> src_cpu = {&src};
    std::vector<TensorDescriptor*> dst_cpu = {&interm_max, &interm_ind};
    if (i != 0) {
      src_cpu.push_back(&prev_max);
      src_cpu.push_back(&prev_ind);
    }
    ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
        src_cpu, dst_cpu, std::make_unique<TopKOp>(std::move(op_first))));

    TopKOp op_second = CreateTopK(env.GetGpuInfo(), op_def_second,
                                  /*use_wg_reduction*/ true, k_offset);
    ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
        {&interm_max, &interm_ind}, {&dst_max, &dst_ind},
        std::make_unique<TopKOp>(std::move(op_second))));

    prev_max = dst_max;
    prev_ind = dst_ind;
    Tensor<BHWC, DataType::FLOAT32> temp_max_tensor;
    dst_max.DownloadData(&temp_max_tensor);
    Tensor<BHWC, DataType::INT32> temp_ind_tensor;
    dst_ind.DownloadData(&temp_ind_tensor);
    for (int j = 0; j < 4; ++j) {
      dst_max_tensor.data[i * 4 + j] = temp_max_tensor.data[i * 4 + j];
      dst_ind_tensor.data[i * 4 + j] = temp_ind_tensor.data[i * 4 + j];
    }
  }
  EXPECT_EQ(dst_max_tensor.data, ref_max);
  EXPECT_EQ(dst_ind_tensor.data, ref_ind);
  return absl::OkStatus();
}

}  // namespace ml_drift
