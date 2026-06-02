// Copyright 2025 The ML Drift Authors.
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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/kernels/tests/strided_slice_test_util.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::ValuesIn;

// Non float tests.
using StridedSliceTypedTest = TestWithParam<TensorStorageType>;

TEST_P(StridedSliceTypedTest, StridedSliceBoolTest) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::BOOL)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(GetParam());
  }
  MLD_ASSERT_OK(StridedSliceBoolTest(*exec_env, GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    StridedSliceTypedTestSuite, StridedSliceTypedTest,
    ValuesIn(GetTensorStoragesTypes()),
    [](const TestParamInfo<StridedSliceTypedTest::ParamType>& info) {
      return absl::StrReplaceAll(ToString(info.param), {{":", ""}});
    });

class StridedSliceFloatTest : public DataTypeTest {};

TEST_P(StridedSliceFloatTest, StridedSliceTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(StridedSliceTest(*exec_env, data_type(), storage()));
}

TEST_P(StridedSliceFloatTest, StridedSliceChannelsAlignedx4) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  auto src_shape = BHWC(1, 6, 10, 8);

  SliceAttributes attr;
  attr.starts = BHWC(0, 1, 3, 4);
  attr.ends = BHWC(src_shape.b, 5, 9, 8);
  attr.strides = BHWC(1, 2, 3, 1);

  MLD_ASSERT_OK(
      StridedSliceBigTest(*exec_env, attr, src_shape, data_type(), storage()));
}

TEST_P(StridedSliceFloatTest, StridedSliceChannelsAlignedx4Batched) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  auto src_shape = BHWC(7, 6, 10, 8);

  SliceAttributes attr;
  attr.starts = BHWC(0, 1, 3, 4);
  attr.ends = BHWC(6, 5, 9, 8);
  attr.strides = BHWC(2, 2, 3, 1);

  MLD_ASSERT_OK(
      StridedSliceBigTest(*exec_env, attr, src_shape, data_type(), storage()));
}

TEST_P(StridedSliceFloatTest, StridedSliceChannelsUnaligned) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  auto src_shape = BHWC(1, 6, 10, 12);

  SliceAttributes attr;
  attr.starts = BHWC(0, 1, 3, 1);
  attr.ends = BHWC(src_shape.b, 5, 9, 11);
  attr.strides = BHWC(1, 2, 3, 2);

  MLD_ASSERT_OK(
      StridedSliceBigTest(*exec_env, attr, src_shape, data_type(), storage()));
}

TEST_P(StridedSliceFloatTest, StridedSliceChannelsUnalignedBatched) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  auto src_shape = BHWC(9, 6, 10, 12);

  SliceAttributes attr;
  attr.starts = BHWC(1, 1, 3, 1);
  attr.ends = BHWC(8, 5, 9, 11);
  attr.strides = BHWC(3, 2, 3, 2);

  MLD_ASSERT_OK(
      StridedSliceBigTest(*exec_env, attr, src_shape, data_type(), storage()));
}

TEST_P(StridedSliceFloatTest, StridedSlice3DTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(StridedSlice3DTest(*exec_env, data_type(), storage()));
}

TEST_P(StridedSliceFloatTest, StridedSlice3DPaddedGridTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  MLD_ASSERT_OK(StridedSlice3DPaddedGridTest(*exec_env, data_type(), storage()));
}

TEST_P(StridedSliceFloatTest, StridedSlice3DChannelsAlignedx4) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  auto src_shape = BHWDC(1, 6, 10, 2, 8);

  Slice3DAttributes attr;
  attr.starts = BHWDC(0, 1, 3, 0, 4);
  attr.ends = BHWDC(src_shape.b, 5, 9, 2, 8);
  attr.strides = BHWDC(1, 2, 3, 1, 1);

  MLD_ASSERT_OK(StridedSlice3DBigTest(*exec_env, attr, src_shape, data_type(),
                                  storage()));
}

TEST_P(StridedSliceFloatTest, StridedSlice3DChannelsUnalignedBatched) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  auto src_shape = BHWDC(9, 6, 10, 3, 12);

  Slice3DAttributes attr;
  attr.starts = BHWDC(1, 1, 3, 0, 1);
  attr.ends = BHWDC(8, 5, 9, 2, 11);
  attr.strides = BHWDC(3, 2, 3, 1, 2);

  MLD_ASSERT_OK(StridedSlice3DBigTest(*exec_env, attr, src_shape, data_type(),
                                  storage()));
}

INSTANTIATE_TEST_SUITE_P(
    FloatTestSuite, StridedSliceFloatTest,
    Combine(ValuesIn(GetFloatTypes()), ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<StridedSliceFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

}  // namespace ml_drift
