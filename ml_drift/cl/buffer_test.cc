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

#include "ml_drift/cl/buffer.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "ml_drift/cl/cl_test.h"
#include "ml_drift/common/types.h"

using ::testing::Eq;
using ::testing::FloatNear;
using ::testing::Pointwise;

namespace ml_drift {
namespace cl {
namespace {

TEST_F(OpenCLTest, BufferTestFloat) {
  const std::vector<float> kInput = {1.0f, 2.0f, 3.0f, -4.0f, 5.1f};
  Buffer buffer;
  ABSL_ASSERT_OK(CreateReadWriteBuffer(sizeof(float) * kInput.size(),
                                  &env_.context(), &buffer));
  ABSL_ASSERT_OK(buffer.WriteData(env_.queue(), absl::MakeConstSpan(kInput)));
  std::vector<float> output;
  ABSL_ASSERT_OK(buffer.ReadData<float>(env_.queue(), &output));

  const std::vector<float> kExpected = {1.0f, 2.0f, 3.0f, -4.0f, 5.1f};
  EXPECT_THAT(output, Pointwise(Eq(), kExpected));
}

TEST_F(OpenCLTest, BufferTestHalf) {
  // 2^-10 is the maximum error for values in range [0.0, 2.2].
  constexpr float kEps = 1.0f / (1 << 10);

  const std::vector<half> kInput = {half(1.4f), half(2.1f), half(2.2f)};
  Buffer buffer;
  ABSL_ASSERT_OK(CreateReadWriteBuffer(sizeof(half) * kInput.size(), &env_.context(),
                                  &buffer));
  ABSL_ASSERT_OK(buffer.WriteData(env_.queue(), absl::MakeConstSpan(kInput)));
  std::vector<half> output_fp16;
  ABSL_ASSERT_OK(buffer.ReadData<half>(env_.queue(), &output_fp16));
  std::vector<float> output_fp32;
  output_fp32.reserve(output_fp16.size());
  for (const half& fp16 : output_fp16) {
    output_fp32.push_back(static_cast<float>(fp16));
  }

  const std::vector<float> kExpected = {1.4f, 2.1f, 2.2f};
  EXPECT_THAT(output_fp32, Pointwise(FloatNear(kEps), kExpected));
}

}  // namespace
}  // namespace cl
}  // namespace ml_drift
