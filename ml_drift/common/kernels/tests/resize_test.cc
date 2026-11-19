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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/kernels/tests/resize_test_util.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

class ResizeFloatTest : public DataTypeTest {};

TEST_P(ResizeFloatTest, ResizeBilinearAlignedTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ResizeBilinearAlignedTest(*exec_env, data_type(), storage()));
}

TEST_P(ResizeFloatTest, ResizeBilinearNonAlignedTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ResizeBilinearNonAlignedTest(*exec_env, data_type(), storage()));
}

TEST_P(ResizeFloatTest, ResizeBilinearWithoutHalfPixelTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      ResizeBilinearWithoutHalfPixelTest(*exec_env, data_type(), storage()));
}

TEST_P(ResizeFloatTest, ResizeBilinearWithHalfPixelTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ResizeBilinearWithHalfPixelTest(*exec_env, data_type(), storage()));
}

TEST_P(ResizeFloatTest, ResizeNearestTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ResizeNearestTest(*exec_env, data_type(), storage()));
}

TEST_P(ResizeFloatTest, ResizeNearestAlignCornersTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ResizeNearestAlignCornersTest(*exec_env, data_type(), storage()));
}

TEST_P(ResizeFloatTest, ResizeNearestHalfPixelCentersTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      ResizeNearestHalfPixelCentersTest(*exec_env, data_type(), storage()));
}

TEST_P(ResizeFloatTest, ResizeBilinearAlignedBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ResizeBilinearAlignedBigTest(*exec_env, data_type(), storage()));
}

TEST_P(ResizeFloatTest, ResizeBilinearNonAlignedBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ResizeBilinearNonAlignedBigTest(*exec_env, data_type(), storage()));
}

TEST_P(ResizeFloatTest, ResizeBilinearAlignedBatchedBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      ResizeBilinearAlignedBatchedBigTest(*exec_env, data_type(), storage()));
}

TEST_P(ResizeFloatTest, ResizeBilinearNonAlignedBatchedBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ResizeBilinearNonAlignedBatchedBigTest(*exec_env, data_type(),
                                                   storage()));
}

TEST_P(ResizeFloatTest, ResizeNearestBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ResizeNearestBigTest(*exec_env, data_type(), storage()));
}

TEST_P(ResizeFloatTest, ResizeNearestBatchedBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ResizeNearestBatchedBigTest(*exec_env, data_type(), storage()));
}

TEST_P(ResizeFloatTest, ResizeBilinear3DAlignedBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ResizeBilinear3DAlignedBigTest(*exec_env, data_type(), storage()));
}

TEST_P(ResizeFloatTest, ResizeBilinear3DNonAlignedBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      ResizeBilinear3DNonAlignedBigTest(*exec_env, data_type(), storage()));
}

TEST_P(ResizeFloatTest, ResizeBilinear3DAlignedBatchedBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(
      ResizeBilinear3DAlignedBatchedBigTest(*exec_env, data_type(), storage()));
}

TEST_P(ResizeFloatTest, ResizeBilinear3DNonAlignedBatchedBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ResizeBilinear3DNonAlignedBatchedBigTest(*exec_env, data_type(),
                                                     storage()));
}

TEST_P(ResizeFloatTest, ResizeNearest3DBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ResizeNearest3DBigTest(*exec_env, data_type(), storage()));
}

TEST_P(ResizeFloatTest, ResizeNearest3DBatchedBigTest) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  ABSL_ASSERT_OK(ResizeNearest3DBatchedBigTest(*exec_env, data_type(), storage()));
}

INSTANTIATE_TEST_SUITE_P(
    ResizeFloatTestSuite, ResizeFloatTest,
    Combine(ValuesIn(GetFloatTypes()), ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<ResizeFloatTest::ParamType>& info) {
      return ToString(info.param);
    });

}  // namespace ml_drift
