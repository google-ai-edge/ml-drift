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

#include "ml_drift/common/task/weights_layout.h"

#include <vector>

#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "ml_drift/common/shape.h"

namespace ml_drift {
namespace {

WeightsDescription CreateWeightsDescription(bool is_i4o4,
                                            const std::vector<Axis>& axes) {
  WeightsDescription desc;
  desc.layout = WeightsLayout::kCustomGroups;
  if (is_i4o4) {
    desc.group_sizes.push_back({Axis::OUTPUT_CHANNELS, 4});
    desc.group_sizes.push_back({Axis::INPUT_CHANNELS, 4});
  } else {
    desc.group_sizes.push_back({Axis::INPUT_CHANNELS, 4});
    desc.group_sizes.push_back({Axis::OUTPUT_CHANNELS, 4});
  }
  for (Axis axis : axes) {
    desc.group_sizes.push_back({axis, 0});
  }
  return desc;
}

TEST(WeightsLayoutTest, IsOIOI4O4) {
  WeightsDescription desc;
  desc.layout = WeightsLayout::kCustomGroups;
  desc.group_sizes = {{Axis::OUTPUT_CHANNELS, 4},
                      {Axis::INPUT_CHANNELS, 4},
                      {Axis::OUTPUT_CHANNELS, 0},
                      {Axis::INPUT_CHANNELS, 0},
                      {Axis::OUTPUT_CHANNELS, 0}};
  EXPECT_TRUE(desc.IsOISpatialOGroupI4O4());
}

TEST(WeightsLayoutTest, IsIOO4I4) {
  WeightsDescription desc;
  desc.layout = WeightsLayout::kCustomGroups;
  desc.group_sizes = {{Axis::INPUT_CHANNELS, 4},
                      {Axis::OUTPUT_CHANNELS, 4},
                      {Axis::OUTPUT_CHANNELS, 0},
                      {Axis::INPUT_CHANNELS, 0}};
  EXPECT_TRUE(desc.IsOISpatialOGroupO4I4());
}

TEST(WeightsLayoutTest, IsOII4O4) {
  WeightsDescription desc;
  desc.layout = WeightsLayout::kCustomGroups;
  desc.group_sizes = {{Axis::OUTPUT_CHANNELS, 4},
                      {Axis::INPUT_CHANNELS, 4},
                      {Axis::INPUT_CHANNELS, 0},
                      {Axis::OUTPUT_CHANNELS, 0}};
  EXPECT_TRUE(desc.IsOISpatialOGroupI4O4());
}

TEST(WeightsLayoutTest, IsIIO4I4) {
  WeightsDescription desc;
  desc.layout = WeightsLayout::kCustomGroups;
  desc.group_sizes = {{Axis::INPUT_CHANNELS, 4},
                      {Axis::OUTPUT_CHANNELS, 4},
                      {Axis::INPUT_CHANNELS, 16},
                      {Axis::INPUT_CHANNELS, 0}};
  EXPECT_FALSE(desc.IsOISpatialOGroupO4I4());
}

TEST(WeightsLayoutTest, IsOIHWOI4O4) {
  WeightsDescription desc;
  desc.layout = WeightsLayout::kCustomGroups;
  desc.group_sizes = {{Axis::OUTPUT_CHANNELS, 4}, {Axis::INPUT_CHANNELS, 4},
                      {Axis::OUTPUT_CHANNELS, 0}, {Axis::WIDTH, 0},
                      {Axis::HEIGHT, 0},          {Axis::INPUT_CHANNELS, 0},
                      {Axis::OUTPUT_CHANNELS, 0}};
  EXPECT_TRUE(desc.IsOISpatialOGroupI4O4());
}

TEST(WeightsLayoutTest, IsOWHIOO4I4) {
  WeightsDescription desc;
  desc.layout = WeightsLayout::kCustomGroups;
  desc.group_sizes = {{Axis::INPUT_CHANNELS, 4},  {Axis::OUTPUT_CHANNELS, 4},
                      {Axis::OUTPUT_CHANNELS, 0}, {Axis::INPUT_CHANNELS, 0},
                      {Axis::HEIGHT, 0},          {Axis::WIDTH, 0},
                      {Axis::OUTPUT_CHANNELS, 0}};
  EXPECT_FALSE(desc.IsOISpatialOGroupO4I4());
}

TEST(WeightsLayoutTest, IsIHOI4O4) {
  WeightsDescription desc;
  desc.layout = WeightsLayout::kCustomGroups;
  desc.group_sizes = {{Axis::OUTPUT_CHANNELS, 4},
                      {Axis::INPUT_CHANNELS, 4},
                      {Axis::OUTPUT_CHANNELS, 0},
                      {Axis::HEIGHT, 0},
                      {Axis::INPUT_CHANNELS, 0}};
  EXPECT_TRUE(desc.IsOISpatialOGroupI4O4());
}

TEST(WeightsLayoutTest, IsHIOO4I4) {
  WeightsDescription desc;
  desc.layout = WeightsLayout::kCustomGroups;
  desc.group_sizes = {{Axis::INPUT_CHANNELS, 4},
                      {Axis::OUTPUT_CHANNELS, 4},
                      {Axis::OUTPUT_CHANNELS, 0},
                      {Axis::INPUT_CHANNELS, 0},
                      {Axis::HEIGHT, 0}};
  EXPECT_FALSE(desc.IsOISpatialOGroupO4I4());
}

TEST(WeightsLayoutTest, IsI4O4) {
  WeightsDescription desc;
  desc.layout = WeightsLayout::kCustomGroups;
  desc.group_sizes = {{Axis::OUTPUT_CHANNELS, 4}, {Axis::INPUT_CHANNELS, 4}};
  EXPECT_TRUE(desc.IsOISpatialOGroupI4O4());
}

TEST(WeightsLayoutTest, IsOISpatialOGroupSingleAxis) {
  for (Axis axis : {Axis::OUTPUT_CHANNELS, Axis::INPUT_CHANNELS, Axis::DEPTH,
                    Axis::HEIGHT, Axis::WIDTH}) {
    EXPECT_TRUE(CreateWeightsDescription(true, {axis}).IsOISpatialOGroupI4O4());
    EXPECT_TRUE(
        CreateWeightsDescription(false, {axis}).IsOISpatialOGroupO4I4());
  }

  for (Axis axis : {Axis::BATCH, Axis::CHANNELS, Axis::UNKNOWN, Axis::VALUE}) {
    EXPECT_FALSE(
        CreateWeightsDescription(true, {axis}).IsOISpatialOGroupI4O4());
    EXPECT_FALSE(
        CreateWeightsDescription(false, {axis}).IsOISpatialOGroupO4I4());
  }
}

}  // namespace
}  // namespace ml_drift
