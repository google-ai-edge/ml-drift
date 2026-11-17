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
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"

namespace ml_drift {
namespace {

WeightsDescription CreateWeightsDescription(bool is_i4o4,
                                            const std::vector<Axis>& axes) {
  WeightsDescription desc;
  desc.layout = WeightsLayout::kCustomGroups;
  if (is_i4o4) {
    desc.group_sizes.push_back({Axis::kOutputChannels, 4});
    desc.group_sizes.push_back({Axis::kInputChannels, 4});
  } else {
    desc.group_sizes.push_back({Axis::kInputChannels, 4});
    desc.group_sizes.push_back({Axis::kOutputChannels, 4});
  }
  for (Axis axis : axes) {
    desc.group_sizes.push_back({axis, 0});
  }
  return desc;
}

TEST(WeightsLayoutTest, IsOIOI4O4) {
  WeightsDescription desc;
  desc.layout = WeightsLayout::kCustomGroups;
  desc.group_sizes = {{Axis::kOutputChannels, 4},
                      {Axis::kInputChannels, 4},
                      {Axis::kOutputChannels, 0},
                      {Axis::kInputChannels, 0},
                      {Axis::kOutputChannels, 0}};
  EXPECT_TRUE(desc.IsOISpatialOGroupI4O4());
}

TEST(WeightsLayoutTest, IsIOO4I4) {
  WeightsDescription desc;
  desc.layout = WeightsLayout::kCustomGroups;
  desc.group_sizes = {{Axis::kInputChannels, 4},
                      {Axis::kOutputChannels, 4},
                      {Axis::kOutputChannels, 0},
                      {Axis::kInputChannels, 0}};
  EXPECT_TRUE(desc.IsOISpatialOGroupO4I4());
}

TEST(WeightsLayoutTest, IsOII4O4) {
  WeightsDescription desc;
  desc.layout = WeightsLayout::kCustomGroups;
  desc.group_sizes = {{Axis::kOutputChannels, 4},
                      {Axis::kInputChannels, 4},
                      {Axis::kInputChannels, 0},
                      {Axis::kOutputChannels, 0}};
  EXPECT_TRUE(desc.IsOISpatialOGroupI4O4());
}

TEST(WeightsLayoutTest, IsIIO4I4) {
  WeightsDescription desc;
  desc.layout = WeightsLayout::kCustomGroups;
  desc.group_sizes = {{Axis::kInputChannels, 4},
                      {Axis::kOutputChannels, 4},
                      {Axis::kInputChannels, 16},
                      {Axis::kInputChannels, 0}};
  EXPECT_FALSE(desc.IsOISpatialOGroupO4I4());
}

TEST(WeightsLayoutTest, IsOIHWOI4O4) {
  WeightsDescription desc;
  desc.layout = WeightsLayout::kCustomGroups;
  desc.group_sizes = {{Axis::kOutputChannels, 4}, {Axis::kInputChannels, 4},
                      {Axis::kOutputChannels, 0}, {Axis::kWidth, 0},
                      {Axis::kHeight, 0},         {Axis::kInputChannels, 0},
                      {Axis::kOutputChannels, 0}};
  EXPECT_TRUE(desc.IsOISpatialOGroupI4O4());
}

TEST(WeightsLayoutTest, IsOWHIOO4I4) {
  WeightsDescription desc;
  desc.layout = WeightsLayout::kCustomGroups;
  desc.group_sizes = {{Axis::kInputChannels, 4},  {Axis::kOutputChannels, 4},
                      {Axis::kOutputChannels, 0}, {Axis::kInputChannels, 0},
                      {Axis::kHeight, 0},         {Axis::kWidth, 0},
                      {Axis::kOutputChannels, 0}};
  EXPECT_FALSE(desc.IsOISpatialOGroupO4I4());
}

TEST(WeightsLayoutTest, IsIHOI4O4) {
  WeightsDescription desc;
  desc.layout = WeightsLayout::kCustomGroups;
  desc.group_sizes = {{Axis::kOutputChannels, 4},
                      {Axis::kInputChannels, 4},
                      {Axis::kOutputChannels, 0},
                      {Axis::kHeight, 0},
                      {Axis::kInputChannels, 0}};
  EXPECT_TRUE(desc.IsOISpatialOGroupI4O4());
}

TEST(WeightsLayoutTest, IsHIOO4I4) {
  WeightsDescription desc;
  desc.layout = WeightsLayout::kCustomGroups;
  desc.group_sizes = {{Axis::kInputChannels, 4},
                      {Axis::kOutputChannels, 4},
                      {Axis::kOutputChannels, 0},
                      {Axis::kInputChannels, 0},
                      {Axis::kHeight, 0}};
  EXPECT_FALSE(desc.IsOISpatialOGroupO4I4());
}

TEST(WeightsLayoutTest, IsI4O4) {
  WeightsDescription desc;
  desc.layout = WeightsLayout::kCustomGroups;
  desc.group_sizes = {{Axis::kOutputChannels, 4}, {Axis::kInputChannels, 4}};
  EXPECT_TRUE(desc.IsOISpatialOGroupI4O4());
}

TEST(WeightsLayoutTest, IsOISpatialOGroupSingleAxis) {
  for (Axis axis : {Axis::kOutputChannels, Axis::kInputChannels, Axis::kDepth,
                    Axis::kHeight, Axis::kWidth}) {
    EXPECT_TRUE(CreateWeightsDescription(true, {axis}).IsOISpatialOGroupI4O4());
    EXPECT_TRUE(
        CreateWeightsDescription(false, {axis}).IsOISpatialOGroupO4I4());
  }

  for (Axis axis :
       {Axis::kBatch, Axis::kChannels, Axis::kUnknown, Axis::kValue}) {
    EXPECT_FALSE(
        CreateWeightsDescription(true, {axis}).IsOISpatialOGroupI4O4());
    EXPECT_FALSE(
        CreateWeightsDescription(false, {axis}).IsOISpatialOGroupO4I4());
  }
}

TEST(WeightsLayoutTest, CustomGroupsI2gr0O8I2gr1O2Properties) {
  WeightsDescription desc;
  desc.type = DataType::kFloat32;
  desc.layout = WeightsLayout::kCustomGroupsI2gr0O8I2gr1O2;
  desc.output_group_size = 8;
  desc.group_sizes = {{Axis::kOutputChannels, 4}, {Axis::kInputChannels, 4},
                      {Axis::kWidth, 0},          {Axis::kHeight, 0},
                      {Axis::kInputChannels, 0},  {Axis::kOutputChannels, 0}};

  EXPECT_EQ(ToString(WeightsLayout::kCustomGroupsI2gr0O8I2gr1O2),
            "kCustomGroupsI2gr0O8I2gr1O2");
  EXPECT_EQ(ToString(desc),
            "float32_kCustomGroupsI2gr0O8I2gr1O2_o4_i4_w_h_i_o");
  EXPECT_TRUE(desc.IsLinearLayout());
  EXPECT_FALSE(desc.IsI4O4());
  EXPECT_FALSE(desc.IsO4I4());
  EXPECT_FALSE(desc.IsOISpatialOGroupI4O4());
  EXPECT_FALSE(desc.IsOISpatialOGroupO4I4());
  EXPECT_EQ(desc.GetOutputGroupSize(), 1);

  WeightsDescription same_desc = desc;
  same_desc.output_group_size = 2;
  EXPECT_EQ(desc, same_desc);

  WeightsDescription diff_desc = desc;
  diff_desc.group_sizes.back().first = Axis::kInputChannels;
  EXPECT_NE(desc, diff_desc);
}

TEST(WeightsLayoutTest, CustomGroupsOgr1Ogr2Ogr0Properties) {
  WeightsDescription desc;
  desc.type = DataType::kFloat16;
  desc.layout = WeightsLayout::kCustomGroupsOgr1Ogr2Ogr0;
  desc.output_group_size = 8;
  desc.params0 = {2, 4, 8};
  desc.group_sizes = {{Axis::kInputChannels, 16}, {Axis::kOutputChannels, 3},
                      {Axis::kInputChannels, 0},  {Axis::kWidth, 0},
                      {Axis::kHeight, 0},         {Axis::kOutputChannels, 0}};

  EXPECT_EQ(ToString(WeightsLayout::kCustomGroupsOgr1Ogr2Ogr0),
            "kCustomGroupsOgr1Ogr2Ogr0");
  EXPECT_EQ(ToString(desc),
            "float16_kCustomGroupsO4gr1O8gr2O2gr0_i16_o3_i_w_h_o");
  EXPECT_TRUE(desc.IsLinearLayout());
  EXPECT_FALSE(desc.IsI4O4());
  EXPECT_FALSE(desc.IsO4I4());
  EXPECT_FALSE(desc.IsOISpatialOGroupI4O4());
  EXPECT_FALSE(desc.IsOISpatialOGroupO4I4());
  EXPECT_EQ(desc.GetOutputGroupSize(), 1);

  WeightsDescription same_desc = desc;
  same_desc.output_group_size = 2;
  EXPECT_EQ(desc, same_desc);

  WeightsDescription diff_o_gr2 = desc;
  diff_o_gr2.params0.o_gr2_size = 2;
  EXPECT_NE(desc, diff_o_gr2);

  WeightsDescription diff_desc = desc;
  diff_desc.group_sizes.back().first = Axis::kInputChannels;
  EXPECT_NE(desc, diff_desc);
}

TEST(WeightsLayoutTest, CustomGroupsOgr0Igr1Ogr1Igr0Properties) {
  WeightsDescription desc;
  desc.type = DataType::kFloat16;
  desc.layout = WeightsLayout::kCustomGroupsOgr0Igr1Ogr1Igr0;
  desc.output_group_size = 8;
  desc.params1 = {2, 8, 16, 4};
  desc.group_sizes = {{Axis::kOutputChannels, 3},
                      {Axis::kInputChannels, 0},
                      {Axis::kWidth, 0},
                      {Axis::kHeight, 0},
                      {Axis::kOutputChannels, 0}};

  EXPECT_EQ(ToString(WeightsLayout::kCustomGroupsOgr0Igr1Ogr1Igr0),
            "kCustomGroupsOgr0Igr1Ogr1Igr0");
  EXPECT_EQ(ToString(desc),
            "float16_kCustomGroupsO16gr0I8gr1O4gr1I2gr0_o3_i_w_h_o");
  EXPECT_TRUE(desc.IsLinearLayout());
  EXPECT_FALSE(desc.IsI4O4());
  EXPECT_FALSE(desc.IsO4I4());
  EXPECT_FALSE(desc.IsOISpatialOGroupI4O4());
  EXPECT_FALSE(desc.IsOISpatialOGroupO4I4());
  EXPECT_EQ(desc.GetOutputGroupSize(), 1);

  WeightsDescription same_desc = desc;
  same_desc.output_group_size = 2;
  same_desc.params0 = {9, 9, 9};
  EXPECT_EQ(desc, same_desc);

  WeightsDescription diff_o_gr1 = desc;
  diff_o_gr1.params1.o_gr1_size = 2;
  EXPECT_NE(desc, diff_o_gr1);

  WeightsDescription diff_desc = desc;
  diff_desc.group_sizes.back().first = Axis::kInputChannels;
  EXPECT_NE(desc, diff_desc);
}

}  // namespace
}  // namespace ml_drift
