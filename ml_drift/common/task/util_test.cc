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

#include "ml_drift/common/task/util.h"

#include <cstddef>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "absl/status/status.h"
#include "ml_drift/common/gpu_info.h"

namespace ml_drift {

TEST(ParseArguments, TestOneArg) {
  std::string text = "func(arg0)";
  size_t close_bracket_pos;
  std::vector<std::string> args;
  MLD_ASSERT_OK(ParseArguments(text, text.find('('), &close_bracket_pos, &args));
  EXPECT_EQ(args.size(), 1);
  EXPECT_EQ(args[0], "arg0");
  EXPECT_EQ(close_bracket_pos, text.size() - 1);
}

TEST(ParseArguments, TestTwoArgs) {
  std::string text = "func(arg0, arg1)";
  size_t close_bracket_pos;
  std::vector<std::string> args;
  MLD_ASSERT_OK(ParseArguments(text, text.find('('), &close_bracket_pos, &args));
  EXPECT_EQ(args.size(), 2);
  EXPECT_EQ(args[0], "arg0");
  EXPECT_EQ(args[1], "arg1");
  EXPECT_EQ(close_bracket_pos, text.size() - 1);
}

TEST(ParseArguments, TestThreeArgs) {
  std::string text = "func( arg0 ,     arg1,arg2)";
  size_t close_bracket_pos;
  std::vector<std::string> args;
  MLD_ASSERT_OK(ParseArguments(text, text.find('('), &close_bracket_pos, &args));
  EXPECT_EQ(args.size(), 3);
  EXPECT_EQ(args[0], "arg0");
  EXPECT_EQ(args[1], "arg1");
  EXPECT_EQ(args[2], "arg2");
  EXPECT_EQ(close_bracket_pos, text.size() - 1);
}

TEST(ParseArguments, TestOneComplexArg) {
  std::string text = "func(arg0<int>)";
  size_t close_bracket_pos;
  std::vector<std::string> args;
  MLD_ASSERT_OK(ParseArguments(text, text.find('('), &close_bracket_pos, &args));
  EXPECT_EQ(args.size(), 1);
  EXPECT_EQ(args[0], "arg0<int>");
  EXPECT_EQ(close_bracket_pos, text.size() - 1);
}

TEST(ParseArguments, TestTwoComplexArgs) {
  std::string text = "func(arg0<int(2,2)>, arg1[23, int(2,2), {code}])";
  size_t close_bracket_pos;
  std::vector<std::string> args;
  MLD_ASSERT_OK(ParseArguments(text, text.find('('), &close_bracket_pos, &args));
  EXPECT_EQ(args.size(), 2);
  EXPECT_EQ(args[0], "arg0<int(2,2)>");
  EXPECT_EQ(args[1], "arg1[23, int(2,2), {code}]");
  EXPECT_EQ(close_bracket_pos, text.size() - 1);
}

TEST(ParseArguments, TestMixedArgs) {
  std::string text = "func(arg0<int(2,2)>, arg1, arg2[23, int(2,2)],  [0,0])";
  size_t close_bracket_pos;
  std::vector<std::string> args;
  MLD_ASSERT_OK(ParseArguments(text, text.find('('), &close_bracket_pos, &args));
  EXPECT_EQ(args.size(), 4);
  EXPECT_EQ(args[0], "arg0<int(2,2)>");
  EXPECT_EQ(args[1], "arg1");
  EXPECT_EQ(args[2], "arg2[23, int(2,2)]");
  EXPECT_EQ(args[3], "[0,0]");
  EXPECT_EQ(close_bracket_pos, text.size() - 1);
}

TEST(ParseArguments, TestArgWithLeftShift) {
  std::string text = "func(arg0, arg1 >> 4)";
  size_t close_bracket_pos;
  std::vector<std::string> args;
  MLD_ASSERT_OK(ParseArguments(text, text.find('('), &close_bracket_pos, &args));
  EXPECT_EQ(args.size(), 2);
  EXPECT_EQ(args[0], "arg0");
  EXPECT_EQ(args[1], "arg1 >> 4");
  EXPECT_EQ(close_bracket_pos, text.size() - 1);
}

TEST(ParseArguments, TestArgWithRightShift) {
  std::string text = "func(arg0 << 4, arg1)";
  size_t close_bracket_pos;
  std::vector<std::string> args;
  MLD_ASSERT_OK(ParseArguments(text, text.find('('), &close_bracket_pos, &args));
  EXPECT_EQ(args.size(), 2);
  EXPECT_EQ(args[0], "arg0 << 4");
  EXPECT_EQ(args[1], "arg1");
  EXPECT_EQ(close_bracket_pos, text.size() - 1);
}

TEST(ParseArguments, TestTemplateArgWithLeftShift) {
  std::string text = "func<a>>4>()";
  size_t close_bracket_pos;
  std::vector<std::string> args;
  MLD_ASSERT_OK(ParseArguments(text, text.find('<'), &close_bracket_pos, &args));
  EXPECT_EQ(args.size(), 1);
  EXPECT_EQ(args[0], "a>>4");
  EXPECT_EQ(close_bracket_pos, text.size() - 3);
}

TEST(ParseArguments, TestNestedTemplateArg) {
  std::string text = "func<vec4<float> >()";
  size_t close_bracket_pos;
  std::vector<std::string> args;
  MLD_ASSERT_OK(ParseArguments(text, text.find('<'), &close_bracket_pos, &args));
  EXPECT_EQ(args.size(), 1);
  EXPECT_EQ(args[0], "vec4<float>");
  EXPECT_EQ(close_bracket_pos, text.size() - 3);
}

TEST(ParseArguments, TestNoArgs) {
  std::string text = "func()";
  size_t close_bracket_pos;
  std::vector<std::string> args;
  MLD_ASSERT_OK(ParseArguments(text, text.find('('), &close_bracket_pos, &args));
  EXPECT_EQ(args.size(), 0);
  EXPECT_EQ(close_bracket_pos, text.size() - 1);
}

TEST(ParseArguments, Test2NoArgs) {
  std::string text = "func(  )";
  size_t close_bracket_pos;
  std::vector<std::string> args;
  MLD_ASSERT_OK(ParseArguments(text, text.find('('), &close_bracket_pos, &args));
  EXPECT_EQ(args.size(), 0);
  EXPECT_EQ(close_bracket_pos, text.size() - 1);
}

TEST(ParseArguments, TestFailNoArgument) {
  std::string text = "func(a, , b)";
  size_t close_bracket_pos;
  std::vector<std::string> args;
  EXPECT_FALSE(
      ParseArguments(text, text.find('('), &close_bracket_pos, &args).ok());
}

TEST(ParseArguments, TestFailNoClosingBracket) {
  std::string text = "func(a, b";
  size_t close_bracket_pos;
  std::vector<std::string> args;
  EXPECT_FALSE(
      ParseArguments(text, text.find('('), &close_bracket_pos, &args).ok());
}

TEST(ParseArguments, TestFailWrongClosingBracket) {
  std::string text = "func(a, b}";
  size_t close_bracket_pos;
  std::vector<std::string> args;
  EXPECT_FALSE(
      ParseArguments(text, text.find('('), &close_bracket_pos, &args).ok());
}

TEST(ParseArguments, TestFailWrongNestedClosingBracket) {
  std::string text = "func(a[0}, b)";
  size_t close_bracket_pos;
  std::vector<std::string> args;
  EXPECT_FALSE(
      ParseArguments(text, text.find('('), &close_bracket_pos, &args).ok());
}

TEST(ParseArguments, TestFailWrongNestedBracketsOrder) {
  std::string text = "func(a][, b)";
  size_t close_bracket_pos;
  std::vector<std::string> args;
  EXPECT_FALSE(
      ParseArguments(text, text.find('('), &close_bracket_pos, &args).ok());
}

TEST(ParseSystemFunction, TestInt32x4ToInt8x16AsVec4x4) {
  std::string result;
  ASSERT_EQ(PerformSystemFunction(GpuInfo(), "Int32x4ToInt8x16AsVec4x4", {}, {},
                                  &result),
            absl::NotFoundError(
                "Int32x4ToInt8x16AsVec4x4 must have 1 template argument and 5 "
                "arguments"));
  MLD_ASSERT_OK(PerformSystemFunction(GpuInfo(), "Int32x4ToInt8x16AsVec4x4",
                                  {"a", "b", "c", "d", "e"}, {"float"},
                                  &result));
  EXPECT_EQ(result, R"(
  b.x = ucl::Convert<float>(ucl::Convert<char>((a.x) & 255));
  b.y = ucl::Convert<float>(ucl::Convert<char>((a.x >>  8u) & 255));
  b.z = ucl::Convert<float>(ucl::Convert<char>((a.x >> 16u) & 255));
  b.w = ucl::Convert<float>(ucl::Convert<char>((a.x >> 24u) & 255));
  c.x = ucl::Convert<float>(ucl::Convert<char>((a.y) & 255));
  c.y = ucl::Convert<float>(ucl::Convert<char>((a.y >>  8u) & 255));
  c.z = ucl::Convert<float>(ucl::Convert<char>((a.y >> 16u) & 255));
  c.w = ucl::Convert<float>(ucl::Convert<char>((a.y >> 24u) & 255));
  d.x = ucl::Convert<float>(ucl::Convert<char>((a.z) & 255));
  d.y = ucl::Convert<float>(ucl::Convert<char>((a.z >>  8u) & 255));
  d.z = ucl::Convert<float>(ucl::Convert<char>((a.z >> 16u) & 255));
  d.w = ucl::Convert<float>(ucl::Convert<char>((a.z >> 24u) & 255));
  e.x = ucl::Convert<float>(ucl::Convert<char>((a.w) & 255));
  e.y = ucl::Convert<float>(ucl::Convert<char>((a.w >>  8u) & 255));
  e.z = ucl::Convert<float>(ucl::Convert<char>((a.w >> 16u) & 255));
  e.w = ucl::Convert<float>(ucl::Convert<char>((a.w >> 24u) & 255));
)");
}

}  // namespace ml_drift
