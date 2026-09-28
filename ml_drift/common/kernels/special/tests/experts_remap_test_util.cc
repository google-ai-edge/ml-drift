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

#include "ml_drift/common/kernels/special/tests/experts_remap_test_util.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/special/experts_remap.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_ref_ops.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

namespace {
absl::Status ExpertsRemapTest(const TensorInt32& src_tensor, int num_experts,
                              TestExecutionEnvironment* env) {
  const int seq_size = src_tensor.shape.w;

  auto [groups_map, groups_sizes] = GroupsMapReference(src_tensor, num_experts);

  TensorDescriptor src_desc =
      TensorDescriptor(DataType::INT32, TensorStorageType::BUFFER, Layout::HWC);
  src_desc.SetBHWCShape(src_tensor.shape);
  src_desc.UploadData(src_tensor);

  TensorDescriptor dst_desc =
      TensorDescriptor(DataType::INT32, TensorStorageType::BUFFER, Layout::HWC);
  dst_desc.SetBHWCShape(BHWC(1, num_experts, seq_size, 2));

  TensorDescriptor dst_count_desc =
      TensorDescriptor(DataType::INT32, TensorStorageType::BUFFER, Layout::HWC);
  dst_count_desc.SetBHWCShape(BHWC(1, 1, 1, num_experts));

  ABSL_RETURN_IF_ERROR(
      env->ExecuteGPUOperation({&src_desc}, {&dst_desc, &dst_count_desc},
                               CreateExpertsRemapOp(src_desc, dst_desc)));

  TensorInt32 dst_tensor;
  dst_desc.DownloadData(&dst_tensor);

  TensorInt32 dst_count_tensor;
  dst_count_desc.DownloadData(&dst_count_tensor);

  EXPECT_THAT(dst_count_tensor.data,
              ::testing::ElementsAreArray(groups_sizes.data));

  for (int i = 0; i < num_experts; ++i) {
    for (int j = 0; j < seq_size; ++j) {
      if (j < groups_sizes.data[i]) {
        EXPECT_EQ(dst_tensor.data[(i * seq_size + j) * 2 + 0],
                  groups_map.data[(i * seq_size + j) * 2 + 0]);
        EXPECT_EQ(dst_tensor.data[(i * seq_size + j) * 2 + 1],
                  groups_map.data[(i * seq_size + j) * 2 + 1]);
      }
    }
  }
  return absl::OkStatus();
}
}  // namespace

absl::Status ExpertsRemapTest(TestExecutionEnvironment* env) {
  TensorInt32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 6, 4);
  // clang-format off
  src_tensor.data = {
    0,  2,  5,  6,
    3,  4,  6,  10,
    7,  8,  9,  12,
    0,  3,  5,  13,
    2,  7,  9,  14,
    3,  8, 10,  15,
  };
  // clang-format on
  return ExpertsRemapTest(src_tensor, 16, env);
}

absl::Status ExpertsRemapBigTest(TestExecutionEnvironment* env, int seq_size,
                                 int num_active_experts, int num_experts) {
  TensorInt32 active_expert_ids =
      GenerateGroupIds(BHWC(1, 1, seq_size, num_active_experts), num_experts);
  return ExpertsRemapTest(active_expert_ids, num_experts, env);
}

absl::Status OffsetsTest(TestExecutionEnvironment* env) {
  TensorInt32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 12);
  src_tensor.data = {
      1, 2, 5, 6, 3, 4, 6, 10, 7, 8, 9, 12,
  };

  TensorInt32 ref_tensor;
  ref_tensor.shape = BHWC(1, 1, 1, 12);
  ref_tensor.data = {
      0, 1, 3, 8, 14, 17, 21, 27, 37, 44, 52, 61,
  };

  TensorDescriptor src_desc =
      TensorDescriptor(DataType::INT32, TensorStorageType::BUFFER, Layout::HWC);
  src_desc.SetBHWCShape(src_tensor.shape);
  src_desc.UploadData(src_tensor);

  TensorDescriptor dst_desc =
      TensorDescriptor(DataType::INT32, TensorStorageType::BUFFER, Layout::HWC);
  dst_desc.SetBHWCShape(BHWC(1, 1, 1, 12));

  ABSL_RETURN_IF_ERROR(
      env->ExecuteGPUOperation({&src_desc}, {&dst_desc}, CreateOffsetsOp()));

  TensorInt32 dst_tensor;
  dst_desc.DownloadData(&dst_tensor);

  EXPECT_THAT(dst_tensor.data, ::testing::ElementsAreArray(ref_tensor.data));

  return absl::OkStatus();
}

absl::Status OffsetsBigTest(TestExecutionEnvironment* env, int size) {
  TensorInt32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, size);
  src_tensor.data.resize(size);
  absl::BitGen gen;
  for (int i = 0; i < size; ++i) {
    src_tensor.data[i] = absl::Uniform(gen, 0, 100);
  }

  TensorInt32 ref_tensor;
  ref_tensor.shape = BHWC(1, 1, 1, size);
  ref_tensor.data.resize(size);
  ref_tensor.data[0] = 0;
  for (int i = 1; i < size; ++i) {
    ref_tensor.data[i] = ref_tensor.data[i - 1] + src_tensor.data[i - 1];
  }

  TensorDescriptor src_desc =
      TensorDescriptor(DataType::INT32, TensorStorageType::BUFFER, Layout::HWC);
  src_desc.SetBHWCShape(src_tensor.shape);
  src_desc.UploadData(src_tensor);

  TensorDescriptor dst_desc =
      TensorDescriptor(DataType::INT32, TensorStorageType::BUFFER, Layout::HWC);
  dst_desc.SetBHWCShape(BHWC(1, 1, 1, size));

  ABSL_RETURN_IF_ERROR(
      env->ExecuteGPUOperation({&src_desc}, {&dst_desc}, CreateOffsetsOp()));

  TensorInt32 dst_tensor;
  dst_desc.DownloadData(&dst_tensor);

  EXPECT_THAT(dst_tensor.data, ::testing::ElementsAreArray(ref_tensor.data));

  return absl::OkStatus();
}

absl::Status LinearizeMapTest(TestExecutionEnvironment* env, int seq_size,
                              int num_active_experts, int num_experts) {
  TensorInt32 active_expert_ids =
      GenerateGroupIds(BHWC(1, 1, seq_size, num_active_experts), num_experts);

  auto [groups_map, groups_sizes] =
      GroupsMapReference(active_expert_ids, num_experts);
  auto [packed_groups_map, groups_offsets] =
      PackedGroupsMapReference(groups_map, groups_sizes);

  TensorDescriptor src_desc =
      TensorDescriptor(DataType::INT32, TensorStorageType::BUFFER, Layout::HWC);
  src_desc.SetBHWCShape(active_expert_ids.shape);
  src_desc.UploadData(active_expert_ids);

  TensorDescriptor map_desc =
      TensorDescriptor(DataType::INT32, TensorStorageType::BUFFER, Layout::HWC);
  map_desc.SetBHWCShape(BHWC(1, num_experts, seq_size, 2));

  TensorDescriptor count_desc =
      TensorDescriptor(DataType::INT32, TensorStorageType::BUFFER, Layout::HWC);
  count_desc.SetBHWCShape(BHWC(1, 1, 1, num_experts));

  ABSL_RETURN_IF_ERROR(
      env->ExecuteGPUOperation({&src_desc}, {&map_desc, &count_desc},
                               CreateExpertsRemapOp(src_desc, map_desc)));

  TensorInt32 count_tensor;
  count_desc.DownloadData(&count_tensor);
  EXPECT_THAT(count_tensor.data,
              ::testing::ElementsAreArray(groups_sizes.data));

  TensorInt32 map2d_tensor;
  map_desc.DownloadData(&map2d_tensor);

  for (int i = 0; i < num_experts; ++i) {
    for (int j = 0; j < seq_size; ++j) {
      if (j < groups_sizes.data[i]) {
        EXPECT_EQ(map2d_tensor.data[(i * seq_size + j) * 2 + 0],
                  groups_map.data[(i * seq_size + j) * 2 + 0]);
        EXPECT_EQ(map2d_tensor.data[(i * seq_size + j) * 2 + 1],
                  groups_map.data[(i * seq_size + j) * 2 + 1]);
      }
    }
  }

  TensorDescriptor offsets_desc =
      TensorDescriptor(DataType::INT32, TensorStorageType::BUFFER, Layout::HWC);
  offsets_desc.SetBHWCShape(BHWC(1, 1, 1, num_experts));

  ABSL_RETURN_IF_ERROR(env->ExecuteGPUOperation({&count_desc}, {&offsets_desc},
                                                CreateOffsetsOp()));

  TensorInt32 offsets_tensor;
  offsets_desc.DownloadData(&offsets_tensor);
  EXPECT_THAT(offsets_tensor.data,
              ::testing::ElementsAreArray(groups_offsets.data));

  TensorDescriptor linearized_map_desc =
      TensorDescriptor(DataType::INT32, TensorStorageType::BUFFER, Layout::HWC);
  linearized_map_desc.SetBHWCShape(
      BHWC(1, 1, num_active_experts * seq_size, 2));

  ABSL_RETURN_IF_ERROR(env->ExecuteGPUOperation(
      {&map_desc, &count_desc, &offsets_desc}, {&linearized_map_desc},
      CreateLinearizeMapOp(map_desc, linearized_map_desc)));

  TensorInt32 linearized_map_tensor;
  linearized_map_desc.DownloadData(&linearized_map_tensor);

  EXPECT_THAT(linearized_map_tensor.data,
              ::testing::ElementsAreArray(packed_groups_map.data));
  return absl::OkStatus();
}

absl::Status ExpertsRemapToTest(TestExecutionEnvironment* env, int seq_size,
                                int num_active_experts, int num_experts,
                                int channels) {
  const DataType data_type = DataType::FLOAT32;
  const TensorStorageType storage_type = TensorStorageType::BUFFER;
  TensorFloat32 src_tensor =
      MakeSyntheticTensor(BHWC(1, 1, seq_size, channels));
  TensorInt32 active_expert_ids =
      GenerateGroupIds(BHWC(1, 1, seq_size, num_active_experts), num_experts);

  auto [groups_map, groups_sizes] =
      GroupsMapReference(active_expert_ids, num_experts);
  auto [packed_groups_map, groups_offsets] =
      PackedGroupsMapReference(groups_map, groups_sizes);

  auto dst_ref = RemapToReference(src_tensor, packed_groups_map);

  TensorDescriptor src_desc =
      TensorDescriptor(data_type, storage_type, Layout::HWC);
  src_desc.UploadData(src_tensor);
  TensorDescriptor experts_packed_map_desc =
      TensorDescriptor(DataType::INT32, storage_type, Layout::HWC);
  experts_packed_map_desc.UploadData(packed_groups_map);

  TensorDescriptor dst_desc =
      TensorDescriptor(data_type, storage_type, Layout::HWC);
  dst_desc.SetBHWCShape(dst_ref.shape);

  ABSL_RETURN_IF_ERROR(env->ExecuteGPUOperation(
      {&src_desc, &experts_packed_map_desc}, {&dst_desc},
      CreateExpertsRemapToOp(src_desc, experts_packed_map_desc, dst_desc)));

  TensorFloat32 dst_tensor;
  dst_desc.DownloadData(&dst_tensor);

  float eps = GetEpsilon(data_type, env->GetGpuInfo());
  EXPECT_THAT(dst_tensor.data,
              testing::Pointwise(testing::FloatNear(eps), dst_ref.data));

  return absl::OkStatus();
}

absl::Status ExpertsRemapFromTest(TestExecutionEnvironment* env, int seq_size,
                                  int num_active_experts, int num_experts,
                                  int channels) {
  const DataType data_type = DataType::FLOAT32;
  const TensorStorageType storage_type = TensorStorageType::BUFFER;
  TensorFloat32 packed_tensor =
      MakeSyntheticTensor(BHWC(1, 1, seq_size * num_active_experts, channels));
  TensorInt32 active_expert_ids =
      GenerateGroupIds(BHWC(1, 1, seq_size, num_active_experts), num_experts);

  auto [groups_map, groups_sizes] =
      GroupsMapReference(active_expert_ids, num_experts);
  auto [packed_groups_map, groups_offsets] =
      PackedGroupsMapReference(groups_map, groups_sizes);

  auto dst_ref =
      RemapFromReference(packed_tensor, packed_groups_map, num_active_experts);

  TensorDescriptor src_desc =
      TensorDescriptor(data_type, storage_type, Layout::HWC);
  src_desc.UploadData(packed_tensor);
  TensorDescriptor experts_packed_map_desc =
      TensorDescriptor(DataType::INT32, storage_type, Layout::HWC);
  experts_packed_map_desc.UploadData(packed_groups_map);

  TensorDescriptor dst_desc =
      TensorDescriptor(data_type, storage_type, Layout::HWC);
  dst_desc.SetBHWCShape(dst_ref.shape);

  ABSL_RETURN_IF_ERROR(env->ExecuteGPUOperation(
      {&src_desc, &experts_packed_map_desc}, {&dst_desc},
      CreateExpertsRemapFromOp(src_desc, experts_packed_map_desc, dst_desc)));

  TensorFloat32 dst_tensor;
  dst_desc.DownloadData(&dst_tensor);

  float eps = GetEpsilon(data_type, env->GetGpuInfo());
  EXPECT_THAT(dst_tensor.data,
              testing::Pointwise(testing::FloatNear(eps), dst_ref.data));

  return absl::OkStatus();
}

}  // namespace ml_drift
