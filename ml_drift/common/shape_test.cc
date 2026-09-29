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

#include <vector>

#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"

namespace ml_drift {
namespace {

TEST(OIHW, Smoke) {
  OIHW OIHW;

  // Test 4 different versions of setters.
  OIHW.i = 1;
  ASSERT_TRUE(OIHW.set<Axis::kOutputChannels>(2));
  ASSERT_TRUE(OIHW.set(Axis::kHeight, 3));
  ASSERT_TRUE(OIHW.set(3, 4));

  // Make sure invalid setters return false.
  ASSERT_FALSE(OIHW.set(5, 10));
  ASSERT_FALSE(OIHW.set(Axis::kChannels, 10));
  ASSERT_FALSE(OIHW.set<Axis::kChannels>(10));

  // Test 4 different versions of getters
  EXPECT_EQ(1, OIHW.get(Axis::kInputChannels));
  EXPECT_EQ(2, OIHW.o);
  EXPECT_EQ(3, OIHW.get(2));
  EXPECT_EQ(4, OIHW.get<Axis::kWidth>());

  // Make sure getters that fall outside of a range return invalid axis.
  EXPECT_EQ(-1, OIHW.get(5));
  EXPECT_EQ(-1, OIHW.get(Axis::kChannels));
  EXPECT_EQ(-1, OIHW.get<Axis::kChannels>());

  // Check axis indices are all correct.
  ASSERT_EQ(4, OIHW.size());
  std::vector<Axis> expected = {Axis::kOutputChannels, Axis::kInputChannels,
                                Axis::kHeight, Axis::kWidth};
  for (int i = 0; i < OIHW.size(); ++i) {
    Axis axis = OIHW.axis(i);
    ASSERT_EQ(expected[i], axis);
    ASSERT_EQ(i, OIHW.index(axis));
  }

  // Check equivalent conversions.
  OHWI ohwi;
  ASSERT_TRUE(ohwi.CopyAllDefinedAxis(OIHW));
  EXPECT_EQ(ohwi.o, OIHW.o);
  EXPECT_EQ(ohwi.i, OIHW.i);
  EXPECT_EQ(ohwi.h, OIHW.h);
  EXPECT_EQ(ohwi.w, OIHW.w);

  ohwi = OHWI(10, 20, 30, 40);
  ASSERT_TRUE(OIHW.CopyAllGivenAxis(ohwi));
  EXPECT_EQ(ohwi.o, OIHW.o);
  EXPECT_EQ(ohwi.i, OIHW.i);
  EXPECT_EQ(ohwi.h, OIHW.h);
  EXPECT_EQ(ohwi.w, OIHW.w);

  EXPECT_TRUE(ohwi.has(Axis::kWidth));
  EXPECT_FALSE(ohwi.has(Axis::kDepth));
}

TEST(Layout, Smoke) {
  EXPECT_EQ(4, Size<Layout::kOIHW>());
  EXPECT_EQ(4, Size(Layout::kOIHW));
  std::vector<Axis> expected = {Axis::kOutputChannels, Axis::kInputChannels,
                                Axis::kHeight, Axis::kWidth};
  for (int i = 0; i < Size<Layout::kOIHW>(); ++i) {
    Axis axis = GetAxis<Layout::kOIHW>(i);
    ASSERT_EQ(expected[i], axis);
    ASSERT_EQ(axis, GetAxis(Layout::kOIHW, i));
    ASSERT_EQ(i, GetAxisIndex<Layout::kOIHW>(axis));
    ASSERT_EQ(i, GetAxisIndex(Layout::kOIHW, axis));
  }
  EXPECT_EQ(Axis::kUnknown, GetAxis(Layout::kOIHW, 5));
  EXPECT_EQ(-1, GetAxisIndex<Layout::kOIHW>(Axis::kChannels));
  EXPECT_EQ(-1, GetAxisIndex<Layout::kOIHW>(Axis::kChannels));
  EXPECT_TRUE(HasAxis<Layout::kOHWDI>(Axis::kDepth));
  EXPECT_FALSE(HasAxis<Layout::kOHWDI>(Axis::kChannels));
}

TEST(Shape, Smoke) {
  Shape s(Layout::kOIHW, {1, 2, 3, 4});
  EXPECT_TRUE(s.set(Axis::kHeight, 10));
  EXPECT_TRUE(s.set<Axis::kWidth>(20));
  EXPECT_FALSE(s.set(Axis::kBatch, 10));
  EXPECT_FALSE(s.set<Axis::kBatch>(20));

  ASSERT_EQ(10, s.get<Axis::kHeight>());
  ASSERT_EQ(20, s.get(Axis::kWidth));
  EXPECT_EQ(20, s.dimensions[3]);

  EXPECT_TRUE(s.has(Axis::kHeight));
  EXPECT_FALSE(s.has(Axis::kDepth));

  OIHW oihw(1, 2, 10, 20);
  Shape s2 = oihw.ToShape();
  EXPECT_EQ(s2.layout, oihw.layout);
  EXPECT_EQ(s.layout, s2.layout);
  EXPECT_EQ(s.dimensions, s2.dimensions);

  // Convert layout into compatible shape.
  OHWI ohwi;
  ASSERT_TRUE(ohwi.Adopt(s2));
  EXPECT_EQ(1, ohwi.o);
  EXPECT_EQ(2, ohwi.i);
  EXPECT_EQ(10, ohwi.h);
  EXPECT_EQ(20, ohwi.w);
}

}  // namespace
}  // namespace ml_drift
