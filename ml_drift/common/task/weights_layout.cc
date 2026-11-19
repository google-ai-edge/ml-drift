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

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"

namespace ml_drift {

std::string ToString(const WeightsDescription& desc) {
  std::string result;
  if (desc.layout == WeightsLayout::kCustomGroupsOgr1Ogr2Ogr0) {
    result = absl::StrCat(
        ToString(desc.type), "_kCustomGroupsO", desc.params0.o_gr1_size, "gr1O",
        desc.params0.o_gr2_size, "gr2O", desc.params0.o_gr0_size, "gr0");
  } else if (desc.layout == WeightsLayout::kCustomGroupsOgr0Igr1Ogr1Igr0) {
    result = absl::StrCat(
        ToString(desc.type), "_kCustomGroupsO", desc.params1.o_gr0_size, "gr0I",
        desc.params1.i_gr1_size, "gr1O", desc.params1.o_gr1_size, "gr1I",
        desc.params1.i_gr0_size, "gr0");
  } else {
    result = absl::StrCat(ToString(desc.type), "_", ToString(desc.layout));
  }
  if (desc.layout == WeightsLayout::kCustomGroups ||
      desc.layout == WeightsLayout::kCustomGroupsI2gr0O8I2gr1O2 ||
      desc.layout == WeightsLayout::kCustomGroupsOgr1Ogr2Ogr0 ||
      desc.layout == WeightsLayout::kCustomGroupsOgr0Igr1Ogr1Igr0) {
    for (const auto& [axis, group_size] : desc.group_sizes) {
      absl::StrAppend(&result, "_", std::string(1, ToString(axis)[0]));
      if (group_size > 0) {
        absl::StrAppend(&result, group_size);
      }
    }
    return result;
  } else if (desc.layout == WeightsLayout::kISpatialOI4O4UnalignedIO) {
    return result;
  } else {
    return absl::StrCat(result, "_", desc.output_group_size);
  }
}

namespace {
bool IsCustomGroupsI4O4(const std::vector<std::pair<Axis, int>>& group_sizes) {
  return group_sizes.size() >= 2 &&
         group_sizes[0].first == Axis::kOutputChannels &&
         group_sizes[0].second == 4 &&
         group_sizes[1].first == Axis::kInputChannels &&
         group_sizes[1].second == 4;
}

bool IsCustomGroupsO4I4(const std::vector<std::pair<Axis, int>>& group_sizes) {
  return group_sizes.size() >= 2 &&
         group_sizes[0].first == Axis::kInputChannels &&
         group_sizes[0].second == 4 &&
         group_sizes[1].first == Axis::kOutputChannels &&
         group_sizes[1].second == 4;
}

std::vector<Axis> ExtractAxisWithoutFirst4x4Block(
    const std::vector<std::pair<Axis, int>>& group_sizes) {
  std::vector<Axis> result;
  for (int i = 2; i < group_sizes.size(); ++i) {
    result.push_back(group_sizes[i].first);
  }
  return result;
}

template <typename T>
bool IsSubsequence(const std::vector<T>& expected, const std::vector<T>& sub) {
  int expected_idx = 0;
  for (const auto& item : sub) {
    while (expected_idx < expected.size() && expected[expected_idx] != item) {
      ++expected_idx;
    }
    if (expected_idx == expected.size()) {
      return false;
    }
    ++expected_idx;
  }
  return true;
}

bool IsOISpatialOGroup(const std::vector<Axis>& axes) {
  const std::vector<Axis> expected_axes = {
      Axis::kOutputChannels, Axis::kWidth,         Axis::kHeight,
      Axis::kDepth,          Axis::kInputChannels, Axis::kOutputChannels};
  return IsSubsequence(expected_axes, axes);
}

}  // namespace

bool WeightsDescription::IsI4O4() const {
  switch (layout) {
    case WeightsLayout::kOSpatialIOGroupI4O4:
    case WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4:
    case WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4:
      return true;
    case WeightsLayout::kOSpatialIOGroupO4I4:
    case WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4:
      return false;
    case WeightsLayout::kCustomGroups:
      return IsCustomGroupsI4O4(group_sizes);
    case WeightsLayout::kUnknown:
    case WeightsLayout::kISpatialOI4O4UnalignedIO:
    case WeightsLayout::kCustomGroupsI2gr0O8I2gr1O2:
    case WeightsLayout::kCustomGroupsOgr1Ogr2Ogr0:
    case WeightsLayout::kCustomGroupsOgr0Igr1Ogr1Igr0:
      return false;
  }
}

bool WeightsDescription::IsO4I4() const {
  switch (layout) {
    case WeightsLayout::kOSpatialIOGroupO4I4:
    case WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4:
      return true;
    case WeightsLayout::kOSpatialIOGroupI4O4:
    case WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4:
    case WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4:
      return false;
    case WeightsLayout::kCustomGroups:
      return IsCustomGroupsO4I4(group_sizes);
    case WeightsLayout::kUnknown:
    case WeightsLayout::kISpatialOI4O4UnalignedIO:
    case WeightsLayout::kCustomGroupsI2gr0O8I2gr1O2:
    case WeightsLayout::kCustomGroupsOgr1Ogr2Ogr0:
    case WeightsLayout::kCustomGroupsOgr0Igr1Ogr1Igr0:
      return false;
  }
}

int WeightsDescription::GetOutputGroupSize() const {
  switch (layout) {
    case WeightsLayout::kOSpatialIOGroupI4O4:
    case WeightsLayout::kOSpatialIOGroupO4I4:
    case WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4:
    case WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4:
    case WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4:
      return output_group_size;
    case WeightsLayout::kUnknown:
    case WeightsLayout::kCustomGroups:
    case WeightsLayout::kCustomGroupsI2gr0O8I2gr1O2:
    case WeightsLayout::kCustomGroupsOgr1Ogr2Ogr0:
    case WeightsLayout::kCustomGroupsOgr0Igr1Ogr1Igr0:
    case WeightsLayout::kISpatialOI4O4UnalignedIO:
      return 1;
  }
}

bool WeightsDescription::IsLinearLayout() const {
  return !(layout == WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4 ||
           layout == WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4 ||
           layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4);
}

bool WeightsDescription::operator==(const WeightsDescription& t) const {
  return type == t.type && layout == t.layout &&
         GetOutputGroupSize() == t.GetOutputGroupSize() &&
         spatial_remap == t.spatial_remap && group_sizes == t.group_sizes &&
         (layout != WeightsLayout::kCustomGroupsOgr1Ogr2Ogr0 ||
          params0 == t.params0) &&
         (layout != WeightsLayout::kCustomGroupsOgr0Igr1Ogr1Igr0 ||
          params1 == t.params1);
}

bool WeightsDescription::IsOISpatialOGroupI4O4() const {
  if (layout != WeightsLayout::kCustomGroups ||
      !IsCustomGroupsI4O4(group_sizes) || !spatial_remap.empty()) {
    return false;
  }
  return IsOISpatialOGroup(ExtractAxisWithoutFirst4x4Block(group_sizes));
}

bool WeightsDescription::IsOISpatialOGroupO4I4() const {
  if (layout != WeightsLayout::kCustomGroups ||
      !IsCustomGroupsO4I4(group_sizes) || !spatial_remap.empty()) {
    return false;
  }
  return IsOISpatialOGroup(ExtractAxisWithoutFirst4x4Block(group_sizes));
}

std::string ToString(const WeightsLayout& layout) {
  switch (layout) {
    case WeightsLayout::kOSpatialIOGroupI4O4:
      return "kOSpatialIOGroupI4O4";
    case WeightsLayout::kOSpatialIOGroupO4I4:
      return "kOSpatialIOGroupO4I4";
    case WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4:
      return "k2DX4I4YIsSpatialIAndXIsOOGroupO4";
    case WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4:
      return "k2DX4O4YIsSpatialIAndXIsOOGroupI4";
    case WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4:
      return "k2DYIsSpatialIOAndXIsOGroupI4O4";
    case WeightsLayout::kISpatialOI4O4UnalignedIO:
      return "kISpatialOI4O4UnalignedIO";
    case WeightsLayout::kCustomGroups:
      return "kCustomGroups";
    case WeightsLayout::kCustomGroupsI2gr0O8I2gr1O2:
      return "kCustomGroupsI2gr0O8I2gr1O2";
    case WeightsLayout::kCustomGroupsOgr1Ogr2Ogr0:
      return "kCustomGroupsOgr1Ogr2Ogr0";
    case WeightsLayout::kCustomGroupsOgr0Igr1Ogr1Igr0:
      return "kCustomGroupsOgr0Igr1Ogr1Igr0";
    case WeightsLayout::kUnknown:
      return "Unknown";
  }
}

}  // namespace ml_drift
