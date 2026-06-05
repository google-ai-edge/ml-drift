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

#include "ml_drift/common/tensor.h"

#include <cstddef>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"

using ::testing::Eq;
using ::testing::IsEmpty;

namespace ml_drift {
namespace {

class TensorTest : public testing::TestWithParam<bool> {
 public:
  void SetUp() override { absl::c_iota(reference_data_, 1); }

  const BHWC& Shape() const { return shape_; }
  const std::vector<float>& ReferenceData() const { return reference_data_; }
  bool OwnsData() const { return GetParam(); }

  TensorFloat32 CreateTensor() {
    TensorFloat32 tensor;
    if (OwnsData()) {
      tensor = MakeZeroTensor<BHWC, DataType::FLOAT32>(Shape());
      absl::c_copy(ReferenceData(), tensor.data.begin());
    } else {
      tensor.shape = Shape();
      tensor.spanned_data = absl::Span<float>(reference_data_);
    }
    return tensor;
  }

 private:
  const BHWC shape_ = BHWC(2, 3, 4, 5);
  std::vector<float> reference_data_ =
      std::vector<float>(shape_.DimensionsProduct());
};

TEST_P(TensorTest, DefaultConstructedTensorIsEmpty) {
  TensorFloat32 tensor;
  EXPECT_THAT(tensor, IsEmpty());
  EXPECT_THAT(tensor.size(), Eq(0));
  EXPECT_THAT(tensor.shape, IsEmpty());
}

TEST_P(TensorTest, TensorIsNotEmpty) {
  TensorFloat32 tensor = CreateTensor();
  EXPECT_THAT(tensor, Not(IsEmpty()));
}

TEST_P(TensorTest, TensorSizeMatchesShape) {
  TensorFloat32 tensor = CreateTensor();
  EXPECT_THAT(tensor.size(), Eq(Shape().DimensionsProduct()));
  EXPECT_THAT(tensor.shape, Eq(Shape()));
}

TEST_P(TensorTest, TensorDataMatchesOwnedData) {
  TensorFloat32 tensor = CreateTensor();
  if (OwnsData()) {
    EXPECT_THAT(tensor.Data(), Eq(tensor.data.data()));
  } else {
    EXPECT_THAT(tensor.Data(), Eq(ReferenceData().data()));
  }
}

TEST_P(TensorTest, TensorMoveDataForCpuRearrangeWork) {
  TensorFloat32 tensor = CreateTensor();
  const size_t old_size = tensor.size();
  size_t new_size = tensor.size() + 16;
  tensor.MoveDataForCpuRearrange(new_size);
  EXPECT_THAT(tensor.size(), Eq(new_size));
  for (size_t i = 0; i < old_size; ++i) {
    EXPECT_THAT(tensor.Get(i), Eq(ReferenceData()[i]))
        << "element " << i << " doesn't match expected value.";
  }
}

TEST_P(TensorTest, TensorSetAndGetWork) {
  TensorFloat32 tensor = CreateTensor();
  for (size_t i = 0; i < tensor.size(); ++i) {
    EXPECT_THAT(tensor.Get(i), Eq(ReferenceData()[i]))
        << "element " << i << " doesn't match expected value.";
  }
  for (size_t i = 0; i < tensor.size(); ++i) {
    tensor.Set(i, ReferenceData()[i] + 1);
  }
  // Recheck values after setting them.
  for (size_t i = 0; i < tensor.size(); ++i) {
    // Note: we add OwnsData() because if the tensor doesn't own the data then
    // the reference data is also modified when calling `Set()`above.
    EXPECT_THAT(tensor.Get(i), Eq(ReferenceData()[i] + OwnsData()))
        << "element " << i << " doesn't match expected value.";
  }
}

INSTANTIATE_TEST_SUITE_P(
    Test, TensorTest, testing::Bool(),
    [](const testing::TestParamInfo<TensorTest::ParamType>& info) {
      return std::string(info.param ? "OwningData" : "NotOwningData");
    });

}  // namespace
}  // namespace ml_drift
