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

#ifndef ML_DRIFT_COMMON_TASK_WEIGHTS_LAYOUT_H_
#define ML_DRIFT_COMMON_TASK_WEIGHTS_LAYOUT_H_

#include <string>
#include <utility>
#include <vector>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"

namespace ml_drift {

enum class WeightsLayout {
  kUnknown,
  // Spatial is DHW/HW depending on amount of spatial dimensions (Depth, Height,
  // Width).
  kOSpatialIOGroupI4O4,
  kOSpatialIOGroupO4I4,
  kISpatialOI4O4UnalignedIO,
  k2DX4I4YIsSpatialIAndXIsOOGroupO4,
  k2DX4O4YIsSpatialIAndXIsOOGroupI4,
  k2DYIsSpatialIOAndXIsOGroupI4O4,
  kCustomGroups,
};

struct WeightsDescription {
  DataType type;
  WeightsLayout layout;
  // applicable with layouts that have OGroup.
  int output_group_size;  // OGroup size
  std::vector<int> spatial_remap;  // optional, applicable with CustomGroups.
  // applicable with CustomGroups layout.
  std::vector<std::pair<Axis, int>> group_sizes;

  int GetOutputGroupSize() const;
  bool IsI4O4() const;
  bool IsO4I4() const;
  bool IsLinearLayout() const;

  bool IsOISpatialOGroupI4O4() const;
  bool IsOISpatialOGroupO4I4() const;

  bool operator==(const WeightsDescription& t) const;
  bool operator!=(const WeightsDescription& t) const { return !(*this == t); }
};

std::string ToString(const WeightsDescription& desc);
std::string ToString(const WeightsLayout& layout);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_TASK_WEIGHTS_LAYOUT_H_
