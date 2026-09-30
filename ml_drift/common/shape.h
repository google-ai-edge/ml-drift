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

#ifndef ML_DRIFT_COMMON_SHAPE_H_
#define ML_DRIFT_COMMON_SHAPE_H_

#include <stddef.h>

#include <array>
#include <cstdint>
#include <functional>
#include <numeric>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace ml_drift {

enum class Axis {
  kUnknown = 0,
  kChannels = 1,
  kInputChannels = 2,
  kOutputChannels = 3,
  kHeight = 4,
  kWidth = 5,
  kBatch = 6,
  kValue = 7,
  kDepth = 8,

  // Deprecated aliases:
  UNKNOWN = kUnknown,
  CHANNELS = kChannels,
  INPUT_CHANNELS = kInputChannels,
  OUTPUT_CHANNELS = kOutputChannels,
  HEIGHT = kHeight,
  WIDTH = kWidth,
  BATCH = kBatch,
  VALUE = kValue,
  DEPTH = kDepth,
};

std::string ToString(Axis axis);

// Layout represents axis order.
enum class Layout {
  kUnknown = 0,
  kScalar = 1,
  kLinear = 2,
  kHW = 3,
  kCHW = 4,
  kHWC = 5,
  kOIHW = 6,
  kOHWI = 7,
  kIHWO = 8,
  kIOHW = 9,
  kBHWC = 10,
  kHWDC = 11,
  kBHWDC = 12,
  kHWD = 13,
  kOHWDI = 14,
  kHWIO = 15,

  // Deprecated aliases:
  UNKNOWN = kUnknown,
  SCALAR = kScalar,
  LINEAR = kLinear,
  HW = kHW,
  CHW = kCHW,
  HWC = kHWC,
  OIHW = kOIHW,
  OHWI = kOHWI,
  IHWO = kIHWO,
  IOHW = kIOHW,
  BHWC = kBHWC,
  HWDC = kHWDC,
  BHWDC = kBHWDC,
  HWD = kHWD,
  OHWDI = kOHWDI,
  HWIO = kHWIO,
};

std::string ToString(Layout l);

// Stream operator for Layout.
std::ostream& operator<<(std::ostream& os, const Layout& l);

// Returns number of axis for the fixed layout.
template <Layout T>
constexpr int Size();

// Returns number of axis for the given layout.
int Size(Layout layout);

// Returns Axis for the given index and fixed layout.
template <Layout T>
constexpr Axis GetAxis(int index);

// Returns axis for the given layout and index.
Axis GetAxis(Layout layout, int32_t index);

// Returns axis index for the given axis and fixed layout.
template <Layout T>
constexpr int GetAxisIndex(Axis axis);

// Returns axis index for the given layout and axis.
int GetAxisIndex(Layout layout, Axis axis);

// Checks if fixed layout has given axis
template <Layout T>
constexpr bool HasAxis(Axis axis);

// Checks if given layout has given axis
bool HasAxis(Layout layout, Axis axis);

// Stores Layout(axis set and order) and value for dimensions.
struct Shape {
  Shape() : layout(Layout::kUnknown), dimensions() {}

  explicit Shape(Layout t) : layout(t), dimensions(Size(t)) {}

  Shape(Layout t, std::vector<int32_t> d)
      : layout(t), dimensions(std::move(d)) {}

  bool operator==(const Shape& other) const {
    return (layout == other.layout) && (dimensions == other.dimensions);
  }

  bool operator!=(const Shape& other) const { return !operator==(other); }

  // All methods below are matching same methods defined in StrongShape to
  // make sure generic algorithms work both ways.

  // Returns back a dimension or -1 if it is not found.
  template <Axis D>
  int32_t get() const;
  int32_t get(Axis axis) const;

  template <Axis D>
  bool set(int32_t t);
  bool set(Axis axis, int32_t t);

  Axis axis(int index) const { return GetAxis(layout, index); }

  int index(Axis axis) const { return GetAxisIndex(layout, axis); }

  bool has(Axis axis) const { return HasAxis(layout, axis); }

  int64_t DimensionsProduct() const {
    return std::accumulate(dimensions.begin(), dimensions.end(), 1LL,
                           std::multiplies<int64_t>());
  }

  Layout layout = Layout::kUnknown;

  std::vector<int32_t> dimensions;
};

std::string ToString(const Shape& s);

// StrongShape provides convenient explicit access to dimensions stored in
// shape, e.g. StrongShape<Layout::HW> s; provides s.h and s.w accessors.
//
// There is a conversion possible both ways between Shape and StrongShape.
//
//   OIHW oihw;  // specific shape
//   Shape l = oihw.ToShape();
//
//   OHWI other;  // notice not the same but compatible shape.
//   if (!other.Adopt(l)) {
//     // error handling
//   }
//
// StrongShape supports the following set of operations:
//
//   // Returns number of axis in the shape class.
//   static constexpr int size();
//
//   // Returns Axis for the given index or Axis::UNKNOWN if index
//   // falls outside of the defined range in this shape.
//   static constexpr Axis axis(int index);
//
//   // Returns index for the given axis or -1 if axis is not defined in this
//   // shape.
//   static constexpr int index(Axis axis);
//
//   // Getters
//   int32_t get(int index) const;
//   int32_t get(Axis axis) const;
//   int32_t get<Axis>() const;
//
//   // Setters that return false if set was not successful.
//   bool set(int index, int32_t v);
//   bool set(Axis axis, int32_t v);
//   bool set<Axis>(int32_t v);
//
//   // Returns shape's layout.
//   static const Layout layout;
//
//   // Turns specific shape into generic shape.
//   Shape ToShape() const;
//
//   // Copies all dimensions from the given shape.
//   bool Adopt(const Shape&);
//
template <Layout L>
struct StrongShape;

using Scalar = StrongShape<Layout::kScalar>;
using Linear = StrongShape<Layout::kLinear>;
using HW = StrongShape<Layout::kHW>;
using HWD = StrongShape<Layout::kHWD>;

// Common tensor shape for CNN models working with images.
using CHW = StrongShape<Layout::kCHW>;
using HWC = StrongShape<Layout::kHWC>;
using HWDC = StrongShape<Layout::kHWDC>;
using BHWC = StrongShape<Layout::kBHWC>;
using BHWDC = StrongShape<Layout::kBHWDC>;

// Tensor shape used in convolution_2d weights.
using OIHW = StrongShape<Layout::kOIHW>;
using OHWI = StrongShape<Layout::kOHWI>;
using IHWO = StrongShape<Layout::kIHWO>;
using IOHW = StrongShape<Layout::kIOHW>;
using HWIO = StrongShape<Layout::kHWIO>;

// Tensor shape used in convolution_3d weights.
using OHWDI = StrongShape<Layout::kOHWDI>;

// -----------------------------------------------------------------------------
// Everything below are internal implementation details.
// -----------------------------------------------------------------------------

namespace internal_shape {

template <Axis T>
struct AxisTraits;

#define ML_DRIFT_AXIS_TRAITS(AxisName, HolderName)           \
  template <>                                                \
  struct AxisTraits<Axis::AxisName> {                        \
    struct Holder {                                          \
      int32_t HolderName;                                    \
                                                             \
     protected:                                              \
      int32_t operator()() const { return HolderName; }      \
      void operator()(int32_t value) { HolderName = value; } \
    };                                                       \
                                                             \
    using dimension_holder_type = Holder;                    \
  }

ML_DRIFT_AXIS_TRAITS(CHANNELS, c);
ML_DRIFT_AXIS_TRAITS(HEIGHT, h);
ML_DRIFT_AXIS_TRAITS(WIDTH, w);
ML_DRIFT_AXIS_TRAITS(INPUT_CHANNELS, i);
ML_DRIFT_AXIS_TRAITS(OUTPUT_CHANNELS, o);
ML_DRIFT_AXIS_TRAITS(BATCH, b);
ML_DRIFT_AXIS_TRAITS(VALUE, v);
ML_DRIFT_AXIS_TRAITS(DEPTH, d);

#undef ML_DRIFT_AXIS_TRAITS

template <int N, Axis... As>
struct StrongShapeImpl;

template <int N>
struct StrongShapeImpl<N> {
  static constexpr int size() { return N; }

  static constexpr Axis axis(int) { return Axis::kUnknown; }

  static constexpr int index(Axis) { return -1; }

  static constexpr bool has(Axis) { return false; }

  int32_t get(Axis) const { return -1; }

  int32_t get(int) const { return -1; }

  template <Axis B>
  int32_t get() const {
    return -1;
  }

  bool set(Axis, int32_t) { return false; }

  bool set(int, int32_t) { return false; }

  template <Axis B>
  bool set(int32_t) {
    return false;
  }
};

// Used to deduce number of axis, and to be a child of a proper holder to
// provide access to the dimension by name
template <int N, Axis A, Axis... As>
struct StrongShapeImpl<N, A, As...>
    : public AxisTraits<A>::dimension_holder_type,
      public StrongShapeImpl<N + 1, As...> {
  using dimension_holder_type = typename AxisTraits<A>::dimension_holder_type;

  using rest_type = StrongShapeImpl<N + 1, As...>;

  StrongShapeImpl() : dimension_holder_type{0}, rest_type() {}

  template <typename... Ts>
  explicit StrongShapeImpl(int32_t t, Ts... ts)
      : dimension_holder_type{t}, rest_type(ts...) {}

  static constexpr Axis axis(int index) {
    return index == N ? A : rest_type::axis(index);
  }

  static constexpr int index(Axis axis) {
    return axis == A ? N : rest_type::index(axis);
  }

  static constexpr bool has(Axis axis) {
    return axis == A ? true : rest_type::has(axis);
  }

  int32_t get(Axis axis) const {
    return axis == A ? dimension_holder_type::operator()()
                     : rest_type::get(axis);
  }

  template <Axis B>
  int32_t get() const {
    return B == A ? dimension_holder_type::operator()()
                  : rest_type::template get<B>();
  }

  int32_t get(int index) const {
    return index == N ? dimension_holder_type::operator()()
                      : rest_type::get(index);
  }

  bool set(Axis axis, int32_t t) {
    if (axis == A) {
      dimension_holder_type::operator()(t);
      return true;
    }
    return rest_type::set(axis, t);
  }

  bool set(int index, int32_t t) {
    if (index == N) {
      dimension_holder_type::operator()(t);
      return true;
    }
    return rest_type::set(index, t);
  }

  template <Axis B>
  bool set(int32_t t) {
    if (A == B) {
      dimension_holder_type::operator()(t);
      return true;
    }
    return rest_type::template set<B>(t);
  }
};

template <Layout T>
struct LayoutTraits;

#define ML_DRIFT_LAYOUT_TRAITS(LayoutName, ...)                \
  template <>                                                  \
  struct LayoutTraits<Layout::LayoutName> {                    \
    using strong_shape_type = StrongShapeImpl<0, __VA_ARGS__>; \
  }

ML_DRIFT_LAYOUT_TRAITS(HW, Axis::kHeight, Axis::kWidth);
ML_DRIFT_LAYOUT_TRAITS(HWD, Axis::kHeight, Axis::kWidth, Axis::kDepth);
ML_DRIFT_LAYOUT_TRAITS(OHWI, Axis::kOutputChannels, Axis::kHeight, Axis::kWidth,
                       Axis::kInputChannels);
ML_DRIFT_LAYOUT_TRAITS(OIHW, Axis::kOutputChannels, Axis::kInputChannels,
                       Axis::kHeight, Axis::kWidth);
ML_DRIFT_LAYOUT_TRAITS(IOHW, Axis::kInputChannels, Axis::kOutputChannels,
                       Axis::kHeight, Axis::kWidth);
ML_DRIFT_LAYOUT_TRAITS(IHWO, Axis::kInputChannels, Axis::kHeight, Axis::kWidth,
                       Axis::kOutputChannels);
ML_DRIFT_LAYOUT_TRAITS(CHW, Axis::kChannels, Axis::kHeight, Axis::kWidth);
ML_DRIFT_LAYOUT_TRAITS(HWC, Axis::kHeight, Axis::kWidth, Axis::kChannels);
ML_DRIFT_LAYOUT_TRAITS(HWDC, Axis::kHeight, Axis::kWidth, Axis::kDepth,
                       Axis::kChannels);
ML_DRIFT_LAYOUT_TRAITS(LINEAR, Axis::kValue);
ML_DRIFT_LAYOUT_TRAITS(SCALAR, Axis::kValue);
ML_DRIFT_LAYOUT_TRAITS(BHWC, Axis::kBatch, Axis::kHeight, Axis::kWidth,
                       Axis::kChannels);
ML_DRIFT_LAYOUT_TRAITS(BHWDC, Axis::kBatch, Axis::kHeight, Axis::kWidth,
                       Axis::kDepth, Axis::kChannels);
ML_DRIFT_LAYOUT_TRAITS(OHWDI, Axis::kOutputChannels, Axis::kHeight,
                       Axis::kWidth, Axis::kDepth, Axis::kInputChannels);
ML_DRIFT_LAYOUT_TRAITS(HWIO, Axis::kHeight, Axis::kWidth, Axis::kInputChannels,
                       Axis::kOutputChannels);

#undef ML_DRIFT_LAYOUT_TRAITS

template <>
struct LayoutTraits<Layout::kUnknown> {
  using strong_shape_type = StrongShapeImpl<0>;
};

template <Axis A>
struct DimensionGetterFixedAxisFunc {
  template <Layout T>
  int32_t operator()() const {
    constexpr int i = GetAxisIndex<T>(A);
    return i >= 0 && static_cast<size_t>(i) < l->dimensions.size() ?
        l->dimensions[i] : -1;
  }
  const Shape* l;
};

struct DimensionGetterFunc {
  template <Layout T>
  int32_t operator()() const {
    int i = GetAxisIndex<T>(axis);
    return i >= 0 && static_cast<size_t>(i) < l->dimensions.size() ?
        l->dimensions[i] : -1;
  }
  Axis axis;
  const Shape* l;
};

template <Axis A>
struct DimensionSetterFixedAxisFunc {
  template <Layout T>
  bool operator()() const {
    constexpr int i = GetAxisIndex<T>(A);
    if (i >= 0 && static_cast<size_t>(i) < l->dimensions.size()) {
      l->dimensions[i] = v;
      return true;
    }
    return false;
  }
  Shape* l;
  int32_t v;
};

struct DimensionSetterFunc {
  template <Layout T>
  bool operator()() const {
    int i = GetAxisIndex<T>(axis);
    if (i >= 0 && static_cast<size_t>(i) < l->dimensions.size()) {
      l->dimensions[i] = v;
      return true;
    }
    return false;
  }
  Axis axis;
  Shape* l;
  int32_t v;
};

template <Layout L>
struct ToShapeFunc {
  template <Layout T>
  bool operator()() const {
    for (int i = 0; i < StrongShape<L>::size(); ++i) {
      int index = GetAxisIndex<T>(StrongShape<L>::axis(i));
      if (index < 0) return false;
      shape->set(i, l.dimensions[index]);
    }
    return true;
  }

  StrongShape<L>* shape;
  const Shape& l;
};

}  // namespace internal_shape

// template <Axis... As>
template <Layout L>
struct StrongShape : public internal_shape::LayoutTraits<L>::strong_shape_type {
  using strong_shape_type =
      typename internal_shape::LayoutTraits<L>::strong_shape_type;
  StrongShape() = default;

  template <typename... Ts>
  explicit StrongShape(Ts... t) : strong_shape_type(t...) {}

  constexpr static Layout layout = L;

  bool operator==(const StrongShape<L>& shape) const {
    // TODO: implement better alternative.
    return this->ToShape() == shape.ToShape();
  }

  bool operator!=(const StrongShape<L>& shape) const {
    // TODO: implement better alternative.
    return this->ToShape() != shape.ToShape();
  }
  bool empty() const { return DimensionsProduct() == 0; }

  // Turns StrongShape into generic shape.
  Shape ToShape() const {
    std::vector<int32_t> dimensions(StrongShape::size());
    for (int i = 0; i < StrongShape::size(); ++i) {
      dimensions[i] = StrongShape::get(i);
    }
    return Shape(L, std::move(dimensions));
  }

  // @return all dimensions multiplied
  int64_t DimensionsProduct() const {
    int64_t product = 1;
    for (int i = 0; i < StrongShape::size(); ++i) {
      product *= StrongShape::get(i);
    }
    return product;
  }

  // Translates given coordinates of the layout into a linear index assuming
  // dimensions are sorted in tensor access order e.g. if you access
  // foobar[i][j][k] order of coordinates should be i,j,k.
  int64_t LinearIndex(
      const std::array<int32_t, StrongShape::size()>& coordinates) const {
    int64_t index = coordinates[0];
    for (int i = 1; i < StrongShape::size(); ++i) {
      index = index * StrongShape::get(i) + coordinates[i];
    }
    return index;
  }

  // Copies all dimensions from the given generic shape into specific shape.
  // It requires shape to have all axis defined in the given
  // StrongShape. For example:
  //   - If this shape is OHWI but given shape is OIHW, Adopt will copy all
  //     dimensions and return true.
  //   - If this shape is OIHW but input shape is HW, Adopt will copy H and W
  //     dimensions and return true, but if this shape is HW and given shape
  //     OIHW, then Adopt will return false because not all axis are present in
  //     the input shape.
  //
  // @return false if generic shape is not compatible.
  bool Adopt(const Shape& shape) {
    return DispatchByLayout(shape.layout,
                            internal_shape::ToShapeFunc<L>{this, shape});
  }

  // For all axis defined in a given shape copies values to this shape.
  // Therefore, it is possible to copy dimensions from CHW to BCHW, but not
  // the other way around.
  //
  // BCHW bchw;
  // CHW chw;
  // bchw.CopyAllGivenAxis(chw);  --> true
  // chw.CopyAllGivenAxis(bchw);  --> false
  //
  // @return false if axis in source shape is not defined here, thus value
  //         was not copied.
  template <Layout B>
  bool CopyAllGivenAxis(const StrongShape<B>& source) {
    for (int i = 0; i < source.size(); ++i) {
      if (!StrongShape::set(source.axis(i), source.get(i))) {
        return false;
      }
    }
    return true;
  }

  // For all axis defined in this shape copies values from the given shape.
  //
  // BCHW bchw;
  // CHW chw;
  // bchw.CopyAllDefinedAxis(chw);  --> false
  // chw.CopyAllDefinedAxis(bchw);  --> true
  //
  // @return false if given shape does not have axis defined here,
  //         therefore a value was not copied.
  template <Layout B>
  bool CopyAllDefinedAxis(const StrongShape<B>& source) {
    for (int i = 0; i < StrongShape::size(); ++i) {
      int source_index = source.index(StrongShape::axis(i));
      if (source_index < 0) {
        return false;
      }
      StrongShape::set(i, source.get(source_index));  // always true
    }
    return true;
  }

  // Copies values only for matching axis.
  template <Layout B>
  void CopyMatchingAxis(const StrongShape<B>& source) {
    for (int i = 0; i < StrongShape::size(); ++i) {
      StrongShape::set(source.axis(i), source.get(i));
    }
  }

  // AbslHash function for using in flat hash containers.
  template <typename H>
  friend H AbslHashValue(H hash_state, const StrongShape& strong_shape) {
    for (size_t i = 0; i < strong_shape.size(); ++i) {
      hash_state = H::combine(std::move(hash_state), strong_shape.get(i));
    }
    return hash_state;
  }
};

template <Layout T>
inline std::string ToString(const StrongShape<T>& s) {
  return ToString(s.ToShape());
}

template <Layout L>
constexpr Layout StrongShape<L>::layout;

template <class F>
auto DispatchByLayout(Layout type, F f)
    -> decltype(f.template operator()<Layout::kUnknown>()) {
  switch (type) {
    case Layout::kHW:
      return f.template operator()<Layout::kHW>();
    case Layout::kHWD:
      return f.template operator()<Layout::kHWD>();
    case Layout::kHWC:
      return f.template operator()<Layout::kHWC>();
    case Layout::kHWDC:
      return f.template operator()<Layout::kHWDC>();
    case Layout::kCHW:
      return f.template operator()<Layout::kCHW>();
    case Layout::kOIHW:
      return f.template operator()<Layout::kOIHW>();
    case Layout::kIOHW:
      return f.template operator()<Layout::kIOHW>();
    case Layout::kOHWI:
      return f.template operator()<Layout::kOHWI>();
    case Layout::kIHWO:
      return f.template operator()<Layout::kIHWO>();
    case Layout::kLinear:
      return f.template operator()<Layout::kLinear>();
    case Layout::kScalar:
      return f.template operator()<Layout::kScalar>();
    case Layout::kBHWC:
      return f.template operator()<Layout::kBHWC>();
    case Layout::kBHWDC:
      return f.template operator()<Layout::kBHWDC>();
    case Layout::kOHWDI:
      return f.template operator()<Layout::kOHWDI>();
    case Layout::kHWIO:
      return f.template operator()<Layout::kHWIO>();
    case Layout::kUnknown:
      return f.template operator()<Layout::kUnknown>();
  }
}

template <Layout T>
constexpr int Size() {
  return StrongShape<T>::size();
}

template <Layout T>
constexpr Axis GetAxis(int index) {
  return StrongShape<T>::axis(index);
}

template <Layout T>
constexpr int GetAxisIndex(Axis axis) {
  return StrongShape<T>::index(axis);
}

template <Layout T>
constexpr bool HasAxis(Axis axis) {
  return StrongShape<T>::has(axis);
}

template <Axis D>
inline int32_t Shape::get() const {
  return DispatchByLayout(
      layout, internal_shape::DimensionGetterFixedAxisFunc<D>{this});
}

inline int32_t Shape::get(Axis axis) const {
  return DispatchByLayout(layout,
                          internal_shape::DimensionGetterFunc{axis, this});
}

template <Axis D>
inline bool Shape::set(int32_t t) {
  return DispatchByLayout(
      layout, internal_shape::DimensionSetterFixedAxisFunc<D>{this, t});
}

inline bool Shape::set(Axis axis, int32_t t) {
  return DispatchByLayout(layout,
                          internal_shape::DimensionSetterFunc{axis, this, t});
}

template <Layout T>
std::ostream& operator<<(std::ostream& ostream, const StrongShape<T>& shape) {
  ostream << ToString(shape);
  return ostream;
}

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_SHAPE_H_
