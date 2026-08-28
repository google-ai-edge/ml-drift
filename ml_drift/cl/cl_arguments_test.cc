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

#include "ml_drift/cl/cl_arguments.h"

#include <memory>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/match.h"
#include "ml_drift/common/access_type.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"

namespace ml_drift {
namespace cl {
TEST(CLArgumentsTest, TestSelectorResolve) {
  BufferDescriptor desc;
  desc.element_type = DataType::FLOAT32;
  desc.element_size = 4;
  desc.memory_type = MemoryType::GLOBAL;

  Arguments args;
  args.AddObjectRef("weights", AccessType::READ,
                    std::make_unique<BufferDescriptor>(std::move(desc)));
  std::string sample_code = R"(
__kernel void main_function($0) {
  if (a < 3) {
    value = args.weights.Read(id);
  }
})";

  CLArguments cl_args;
  GpuInfo gpu_info;
  ABSL_ASSERT_OK(cl_args.Init(gpu_info, /*context=*/nullptr, &args, &sample_code));
  EXPECT_TRUE(absl::StrContains(sample_code, "value = weights.Read(id);"));
  EXPECT_TRUE(
      absl::StrContains(sample_code, "__global float4* weights_buffer"));
}

}  // namespace cl
}  // namespace ml_drift
