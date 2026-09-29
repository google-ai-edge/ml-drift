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

#include "ml_drift/common/shape.h"

#include <cstdint>
#include <ostream>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace ml_drift {
namespace {

struct GetAxisByIndexFunc {
  template <Layout T>
  Axis operator()() const {
    return GetAxis<T>(index);
  }
  int32_t index;
};

struct GetIndexByAxisFunc {
  template <Layout T>
  int operator()() const {
    return GetAxisIndex<T>(axis);
  }
  Axis axis;
};

struct NumAxisFunc {
  template <Layout T>
  int operator()() const {
    return Size<T>();
  }
};

}  // namespace

std::string ToString(Axis axis) {
  switch (axis) {
    case Axis::kBatch:
      return "batch";
    case Axis::kChannels:
      return "channels";
    case Axis::kInputChannels:
      return "input_channels";
    case Axis::kOutputChannels:
      return "output_channels";
    case Axis::kHeight:
      return "height";
    case Axis::kWidth:
      return "width";
    case Axis::kValue:
      return "value";
    case Axis::kDepth:
      return "depth";
    case Axis::kUnknown:
      return "unknown";
  }
  return "undefined";
}

std::string ToString(Layout layout) {
  switch (layout) {
    case Layout::kScalar:
      return "scalar";
    case Layout::kLinear:
      return "linear";
    case Layout::kHW:
      return "hw";
    case Layout::kHWD:
      return "hwd";
    case Layout::kCHW:
      return "chw";
    case Layout::kHWC:
      return "hwc";
    case Layout::kHWDC:
      return "hwdc";
    case Layout::kOHWI:
      return "ohwi";
    case Layout::kIHWO:
      return "ihwo";
    case Layout::kOIHW:
      return "oihw";
    case Layout::kIOHW:
      return "iohw";
    case Layout::kBHWC:
      return "bhwc";
    case Layout::kBHWDC:
      return "bhwdc";
    case Layout::kOHWDI:
      return "ohwdi";
    case Layout::kHWIO:
      return "hwio";
    case Layout::kUnknown:
      return "unknown";
  }
  return "undefined";
}

std::ostream& operator<<(std::ostream& os, const Layout& l) {
  return os << ToString(l);
}

Axis GetAxis(Layout layout, int32_t index) {
  return DispatchByLayout(layout, GetAxisByIndexFunc{index});
}

int GetAxisIndex(Layout layout, Axis axis) {
  return DispatchByLayout(layout, GetIndexByAxisFunc{axis});
}

bool HasAxis(Layout layout, Axis axis) {
  return GetAxisIndex(layout, axis) >= 0;
}

int Size(Layout layout) { return DispatchByLayout(layout, NumAxisFunc()); }

std::string ToString(const Shape& s) {
  return absl::StrCat("{", ToString(s.layout), ", {",
                      absl::StrJoin(s.dimensions, ", "), "}}");
}

}  // namespace ml_drift
