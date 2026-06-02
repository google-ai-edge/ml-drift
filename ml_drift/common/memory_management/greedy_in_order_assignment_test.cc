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

#include "ml_drift/common/memory_management/greedy_in_order_assignment.h"

#include <cstddef>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "ml_drift/common/memory_management/types.h"

namespace ml_drift {
namespace {

using ::testing::ElementsAre;

// [T0]  [T2]  [T4]
//  |     |     |
// op0   op1   op2
//  |     |     |
// [T1]  [T3]  [T5]
TEST(GreedyInOrderAssignmentTest, ForestOf3SimpleGraphs) {
  std::vector<TensorUsageRecord<size_t>> usage_records = {
      {256, 0, 0},  // Tensor 0
      {256, 0, 3},  // Tensor 1
      {256, 0, 1},  // Tensor 2
      {256, 1, 3},  // Tensor 3
      {256, 0, 2},  // Tensor 4
      {256, 2, 3}   // Tensor 5
  };

  ObjectsAssignment<size_t> assignment;
  EXPECT_TRUE(GreedyInOrderAssignment(usage_records, &assignment).ok());

  // Tensor 0 -> Object 0
  // Tensor 1 -> Object 1
  // Tensor 2 -> Object 2
  // Tensor 3 -> Object 0
  // Tensor 4 -> Object 3
  // Tensor 5 -> Object 2
  EXPECT_THAT(assignment.object_ids, ElementsAre(0, 1, 2, 0, 3, 2));
  EXPECT_THAT(assignment.object_sizes, ElementsAre(256, 256, 256, 256));
}

}  // namespace
}  // namespace ml_drift
