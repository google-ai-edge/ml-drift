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

#ifndef ML_DRIFT_COMMON_TENSOR_MATCHER_H_
#define ML_DRIFT_COMMON_TENSOR_MATCHER_H_

#include <ostream>
#include <tuple>

#include "gmock/gmock.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

template <typename TupleMatcher>
class TensorFloat32EqMatcher {
 public:
  TensorFloat32EqMatcher(const TupleMatcher& tuple_matcher,
                         const TensorFloat32& rhs)
      : tuple_matcher_(tuple_matcher), rhs_(rhs) {}

  // Make TensorEqMatcher movable only (The copy operations are implicitly
  // deleted).
  TensorFloat32EqMatcher(TensorFloat32EqMatcher&& other) = default;
  TensorFloat32EqMatcher& operator=(TensorFloat32EqMatcher&& other) = default;

  template <typename T>
  operator testing::Matcher<T>() const {  // NOLINT
    return testing::Matcher<T>(new Impl(tuple_matcher_, rhs_));
  }

  class Impl : public testing::MatcherInterface<TensorFloat32> {
   public:
    typedef ::std::tuple<float, float> InnerMatcherArg;

    Impl(const TupleMatcher& tuple_matcher, const TensorFloat32& rhs)
        : mono_tuple_matcher_(
              testing::SafeMatcherCast<InnerMatcherArg>(tuple_matcher)),
          rhs_(rhs) {}

    // Make Impl movable only (The copy operations are implicitly deleted).
    Impl(Impl&& other) = default;
    Impl& operator=(Impl&& other) = default;

    // Define what gtest framework will print for the Expected field.
    void DescribeTo(std::ostream* os) const override {
      *os << "tensor which has the shape of " << rhs_.shape
          << ", where each value and its corresponding expected value ";
      mono_tuple_matcher_.DescribeTo(os);
    }

    bool MatchAndExplain(
        TensorFloat32 lhs,
        testing::MatchResultListener* listener) const override {
      if (lhs.shape != rhs_.shape) {
        *listener << "which is different from the expected shape "
                  << rhs_.shape;
        return false;
      }
      const float* left = lhs.data.data();
      const float* right = rhs_.data.data();
      for (int b = 0; b < lhs.shape.b; b++) {
        for (int h = 0; h < lhs.shape.h; h++) {
          for (int w = 0; w < lhs.shape.w; w++) {
            for (int c = 0; c < lhs.shape.c; c++) {
              if (listener->IsInterested()) {
                testing::StringMatchResultListener inner_listener;
                if (!mono_tuple_matcher_.MatchAndExplain({*left, *right},
                                                         &inner_listener)) {
                  *listener << "where the value pair (";
                  testing::internal::UniversalPrint(*left, listener->stream());
                  *listener << ", ";
                  testing::internal::UniversalPrint(*right, listener->stream());
                  *listener << ") with coordinate "
                            << absl::StrFormat("b: %i, h: %i, w: %i, c: %i", b,
                                               h, w, c)
                            << " doesn't match";
                  testing::internal::PrintIfNotEmpty(inner_listener.str(),
                                                     listener->stream());
                  return false;
                }
              } else {
                if (!mono_tuple_matcher_.Matches({*left, *right})) return false;
              }
              left++;
              right++;
            }
          }
        }
      }
      return true;
    }

   private:
    const testing::Matcher<InnerMatcherArg> mono_tuple_matcher_;
    const TensorFloat32 rhs_;
  };

 private:
  const TupleMatcher tuple_matcher_;
  const TensorFloat32 rhs_;
};

template <typename TupleMatcherT>
inline TensorFloat32EqMatcher<TupleMatcherT> TensorEq(
    const TupleMatcherT& matcher, const TensorFloat32& rhs) {
  return TensorFloat32EqMatcher<TupleMatcherT>(matcher, rhs);
}

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_TENSOR_MATCHER_H_
