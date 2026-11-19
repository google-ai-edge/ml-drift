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

#include "ml_drift/common/kernels/bucket_argsort.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {
namespace {

TensorInt32 argsort_cpu(const TensorInt32& values) {
  TensorInt32 indices;
  indices.shape = values.shape;
  indices.data.resize(values.data.size());
  for (int i = 0; i < values.data.size(); ++i) {
    indices.data[i] = i;
  }
  std::stable_sort(indices.data.begin(), indices.data.end(), [&](int a, int b) {
    return values.data[a] < values.data[b];
  });
  return indices;
}

TEST(ArgsortTest, SmallScaleReferenceTest) {
  const int num_buckets = 4;
  const int num_elements = 8;
  TensorInt32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, num_elements);
  src_tensor.data = {3, 1, 3, 0, 2, 1, 0, 2};

  // Expected sorting (indices of elements grouped by their bucket ID):
  // Bucket 0: indices 3, 6
  // Bucket 1: indices 1, 5
  // Bucket 2: indices 4, 7
  // Bucket 3: indices 0, 2
  std::vector<int> expected_indices = {3, 6, 1, 5, 4, 7, 0, 2};
  TensorInt32 cpu_output_tensor = argsort_cpu(src_tensor);

  OperationDef op_def;
  TensorDescriptor src_td = TensorDescriptor{
      DataType::kInt32, TensorStorageType::kBuffer, Layout::kBHWC};
  src_td.SetBHWCShape(BHWC(1, 1, 1, num_elements));
  op_def.src_tensors.push_back(src_td);
  TensorDescriptor dst_td = TensorDescriptor{
      DataType::kInt32, TensorStorageType::kBuffer, Layout::kBHWC};
  dst_td.SetBHWCShape(BHWC(1, 1, 1, num_elements));
  op_def.dst_tensors.push_back(dst_td);

  src_td.UploadDataRaw(absl::MakeConstSpan(src_tensor.data));

  BucketArgsortOp op = CreateBucketArgsort(op_def, exec_env->GetGpuInfo(),
                                           num_buckets, num_elements);
  ABSL_ASSERT_OK(exec_env->ExecuteGPUOperation(
      {&src_td}, {&dst_td}, std::make_unique<BucketArgsortOp>(std::move(op))));

  Tensor<BHWC, DataType::kInt32> gpu_output_tensor;
  dst_td.DownloadData(&gpu_output_tensor);

  EXPECT_EQ(gpu_output_tensor.data, expected_indices);
  EXPECT_EQ(cpu_output_tensor.data, expected_indices);
}

class ArgsortScaleTest : public ::testing::TestWithParam<size_t> {};

TEST_P(ArgsortScaleTest, GpuEquivalence) {
  const size_t num_buckets = 128;
  const size_t num_elements = GetParam();
  TensorInt32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, num_elements);
  src_tensor.data.resize(num_elements);
  unsigned int seed = 42;
  for (size_t i = 0; i < num_elements; ++i) {
    src_tensor.data[i] = rand_r(&seed) % static_cast<int>(num_buckets);
  }
  TensorInt32 cpu_output_tensor = argsort_cpu(src_tensor);

  OperationDef op_def;
  TensorDescriptor src_td = TensorDescriptor{
      DataType::kInt32, TensorStorageType::kBuffer, Layout::kBHWC};
  src_td.SetBHWCShape(BHWC(1, 1, 1, num_elements));
  op_def.src_tensors.push_back(src_td);
  TensorDescriptor dst_td = TensorDescriptor{
      DataType::kInt32, TensorStorageType::kBuffer, Layout::kBHWC};
  dst_td.SetBHWCShape(BHWC(1, 1, 1, num_elements));
  op_def.dst_tensors.push_back(dst_td);

  src_td.UploadDataRaw(absl::MakeConstSpan(src_tensor.data));

  BucketArgsortOp op = CreateBucketArgsort(op_def, exec_env->GetGpuInfo(),
                                           num_buckets, num_elements);
  ABSL_ASSERT_OK(exec_env->ExecuteGPUOperation(
      {&src_td}, {&dst_td}, std::make_unique<BucketArgsortOp>(std::move(op))));

  Tensor<BHWC, DataType::kInt32> gpu_output_tensor;
  dst_td.DownloadData(&gpu_output_tensor);

  EXPECT_EQ(gpu_output_tensor.data, cpu_output_tensor.data);
}

INSTANTIATE_TEST_SUITE_P(LargeScaleCpuGpuEquivalenceTest, ArgsortScaleTest,
                         ::testing::Values(1024, 1));

}  // namespace
}  // namespace ml_drift
