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

#include "absl/strings/str_cat.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"

namespace ml_drift {

std::string ToString(const WeightsDescription& desc) {
  std::string result =
      absl::StrCat(ToString(desc.type), "_", ToString(desc.layout));
  if (desc.layout == WeightsLayout::kCustomGroups) {
    for (const auto& [axis, group_size] : desc.group_sizes) {
      absl::StrAppend(&result, "_", std::string(1, ToString(axis)[0]));
      if (group_size > 0) {
        absl::StrAppend(&result, group_size);
      }
    }
    return result;
  } else if (desc.layout == WeightsLayout::kOICustomSpatialI4O4 ||
             desc.layout == WeightsLayout::kOICustomSpatialO4I4 ||
             desc.layout == WeightsLayout::kISpatialOI4O4UnalignedIO) {
    return result;
  } else {
    return absl::StrCat(result, "_", desc.output_group_size);
  }
}

bool WeightsDescription::IsI4O4() const {
  switch (layout) {
    case WeightsLayout::kOSpatialIOGroupI4O4:
    case WeightsLayout::kOISpatialOGroupI4O4:
    case WeightsLayout::kOICustomSpatialI4O4:
    case WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4:
    case WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4:
      return true;
    case WeightsLayout::kOSpatialIOGroupO4I4:
    case WeightsLayout::kOISpatialOGroupO4I4:
    case WeightsLayout::kOICustomSpatialO4I4:
    case WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4:
      return false;
    case WeightsLayout::kCustomGroups:
      return group_sizes.size() >= 2 &&
             group_sizes[0].first == Axis::OUTPUT_CHANNELS &&
             group_sizes[0].second == 4 &&
             group_sizes[1].first == Axis::INPUT_CHANNELS &&
             group_sizes[1].second == 4;
    case WeightsLayout::kUnknown:
    case WeightsLayout::kISpatialOI4O4UnalignedIO:
      return false;
  }
}

bool WeightsDescription::IsO4I4() const {
  switch (layout) {
    case WeightsLayout::kOSpatialIOGroupO4I4:
    case WeightsLayout::kOISpatialOGroupO4I4:
    case WeightsLayout::kOICustomSpatialO4I4:
    case WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4:
      return true;
    case WeightsLayout::kOSpatialIOGroupI4O4:
    case WeightsLayout::kOISpatialOGroupI4O4:
    case WeightsLayout::kOICustomSpatialI4O4:
    case WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4:
    case WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4:
      return false;
    case WeightsLayout::kCustomGroups:
      return group_sizes.size() >= 2 &&
             group_sizes[0].first == Axis::INPUT_CHANNELS &&
             group_sizes[0].second == 4 &&
             group_sizes[1].first == Axis::OUTPUT_CHANNELS &&
             group_sizes[1].second == 4;
    case WeightsLayout::kUnknown:
    case WeightsLayout::kISpatialOI4O4UnalignedIO:
      return false;
  }
}

int WeightsDescription::GetOutputGroupSize() const {
  switch (layout) {
    case WeightsLayout::kOSpatialIOGroupI4O4:
    case WeightsLayout::kOSpatialIOGroupO4I4:
    case WeightsLayout::kOISpatialOGroupI4O4:
    case WeightsLayout::kOISpatialOGroupO4I4:
    case WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4:
    case WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4:
    case WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4:
      return output_group_size;
    case WeightsLayout::kOICustomSpatialI4O4:
    case WeightsLayout::kOICustomSpatialO4I4:
      return 1;
    case WeightsLayout::kUnknown:
    case WeightsLayout::kCustomGroups:
    case WeightsLayout::kISpatialOI4O4UnalignedIO:
      return 1;
  }
}

bool WeightsDescription::IsCustomSpatial() const {
  return layout == WeightsLayout::kOICustomSpatialI4O4 ||
         layout == WeightsLayout::kOICustomSpatialO4I4;
}

bool WeightsDescription::IsLinearLayout() const {
  return !(layout == WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4 ||
           layout == WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4 ||
           layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4);
}

bool WeightsDescription::operator==(const WeightsDescription& t) const {
  const bool equal_spatial_remap =
      IsCustomSpatial() ? spatial_remap == t.spatial_remap : true;
  return type == t.type && layout == t.layout &&
         GetOutputGroupSize() == t.GetOutputGroupSize() && equal_spatial_remap;
}

std::string ToString(const WeightsLayout& layout) {
  switch (layout) {
    case WeightsLayout::kOSpatialIOGroupI4O4:
      return "kOSpatialIOGroupI4O4";
    case WeightsLayout::kOSpatialIOGroupO4I4:
      return "kOSpatialIOGroupO4I4";
    case WeightsLayout::kOISpatialOGroupI4O4:
      return "kOISpatialOGroupI4O4";
    case WeightsLayout::kOISpatialOGroupO4I4:
      return "kOISpatialOGroupO4I4";
    case WeightsLayout::kOICustomSpatialI4O4:
      return "kOICustomSpatialI4O4";
    case WeightsLayout::kOICustomSpatialO4I4:
      return "kOICustomSpatialO4I4";
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
    case WeightsLayout::kUnknown:
      return "Unknown";
  }
}

}  // namespace ml_drift
