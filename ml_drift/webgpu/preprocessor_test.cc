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

#include "ml_drift/webgpu/preprocessor.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "ml_drift/common/gpu_info.h"

namespace ml_drift {
namespace webgpu {

namespace {

WebGpuInfo GetWebGpuInfo() {
  WebGpuInfo result;
  result.supports_fp16 = true;
  return result;
}

}  // namespace

TEST(PreprocessorTest, ResolveTypes) {
  std::string input = R"(
MAIN_FUNCTION($0) {
  int a0;
  float a1;
  float4 a2;
  half4 a3;
  char4 a4;
  ushort4 a5;
  bool a6;
}
)";
  ExtensionsInfo extensions_info;
  MLD_ASSERT_OK(ConvertToWGSL(GetWebGpuInfo(), &input, &extensions_info));
  EXPECT_EQ(input, R"(
MAIN_FUNCTION($0) {
  var a0 : i32;
  var a1 : f32;
  var a2 : vec4<f32>;
  var a3 : vec4<f16>;
  var a4 : vec4<i32>;
  var a5 : vec4<u32>;
  var a6 : bool;
}
)");
}

TEST(PreprocessorTest, RemoveFloatPostfix) {
  std::string input = R"(
MAIN_FUNCTION($0) {
  float a1 = 0.8934f;
  float a2 = 0.8934;
  float a3 = 34.0f;
}
)";
  ExtensionsInfo extensions_info;
  MLD_ASSERT_OK(ConvertToWGSL(GetWebGpuInfo(), &input, &extensions_info));
  EXPECT_EQ(input, R"(
MAIN_FUNCTION($0) {
  var a1 : f32= 0.8934;
  var a2 : f32= 0.8934;
  var a3 : f32= 34.0;
}
)");
}

TEST(PreprocessorTest, ResolveSingleStatementIf) {
  std::string input = R"(
MAIN_FUNCTION($0) {
  if (X < Y) return;
}
)";
  ExtensionsInfo extensions_info;
  MLD_ASSERT_OK(ConvertToWGSL(GetWebGpuInfo(), &input, &extensions_info));
  EXPECT_EQ(input, R"(
MAIN_FUNCTION($0) {
  if (X < Y) {return;}
}
)");
}

TEST(PreprocessorTest, ResolveDoWhileLoop) {
  std::string input = R"(
MAIN_FUNCTION($0) {
  do {
    x += 2;
  } while (s < 5);
}
)";
  ExtensionsInfo extensions_info;
  MLD_ASSERT_OK(ConvertToWGSL(GetWebGpuInfo(), &input, &extensions_info));
  EXPECT_EQ(input, R"(
MAIN_FUNCTION($0) {
  loop {
    x += 2;
  if(! (s < 5)){break;}
} ;
}
)");
}

TEST(PreprocessorTest, ResolveForLoop) {
  std::string input = R"(
MAIN_FUNCTION($0) {
  for (i = 0; i < 10; i += 1, s += 2) {
    x += 2;
  }
}
)";
  ExtensionsInfo extensions_info;
  MLD_ASSERT_OK(ConvertToWGSL(GetWebGpuInfo(), &input, &extensions_info));
  std::string expected = R"(
MAIN_FUNCTION($0) {
    {i = 0;
  loop {
    if (!( i < 10)) {break;}
    )";
  expected += R"(
    x += 2;
  ;
    continuing {
       i += 1;
       s += 2;
    }
  }
}
}
)";
  EXPECT_EQ(input, expected);
}

TEST(PreprocessorTest, ResolveTernaryOperatorPass) {
  std::string input = R"(
MAIN_FUNCTION($0) {
  x = a ? b : c;
}
)";
  ExtensionsInfo extensions_info;
  MLD_ASSERT_OK(ConvertToWGSL(GetWebGpuInfo(), &input, &extensions_info));
  EXPECT_EQ(input, R"(
MAIN_FUNCTION($0) {
  x =  select( c,  b , a );
}
)");
}

TEST(PreprocessorTest, ResolvePreIncrement) {
  std::string input = R"(
MAIN_FUNCTION($0) {
  ++x;
}
)";
  ExtensionsInfo extensions_info;
  MLD_ASSERT_OK(ConvertToWGSL(GetWebGpuInfo(), &input, &extensions_info));
  EXPECT_EQ(input, R"(
MAIN_FUNCTION($0) {
  x=x+1;
}
)");
}

TEST(PreprocessorTest, MoveLocalMemToGlobalSpace) {
  std::string input = R"(
MAIN_FUNCTION($0) {
  __local float cache[32][4];
}
)";
  ExtensionsInfo extensions_info;
  MLD_ASSERT_OK(ConvertToWGSL(GetWebGpuInfo(), &input, &extensions_info));
  EXPECT_EQ(input, R"(
var<workgroup> cache : array<array<f32, 4>, 32>;
MAIN_FUNCTION($0) {
  }
)");
}

TEST(PreprocessorTest, ParseLongComments) {
  std::string input = R"(
MAIN_FUNCTION($0) {
  /* some comment */int x;
  /* more
   int foo;
   and more
  */
  int y;
}
)";
  ExtensionsInfo extensions_info;
  MLD_ASSERT_OK(ConvertToWGSL(GetWebGpuInfo(), &input, &extensions_info));
  EXPECT_EQ(input, R"(
MAIN_FUNCTION($0) {
  /* some comment */var x : i32;
  /* more
   int foo;
   and more
  */
  var y : i32;
}
)");
}

TEST(PreprocessorTest, ParseShortComments) {
  std::string input = R"(
MAIN_FUNCTION($0) {
  int x; // some comment;
  // int foo;
  int y;
}
)";
  ExtensionsInfo extensions_info;
  MLD_ASSERT_OK(ConvertToWGSL(GetWebGpuInfo(), &input, &extensions_info));
  EXPECT_EQ(input, R"(
MAIN_FUNCTION($0) {
  var x : i32; // some comment;
  // int foo;
  var y : i32;
}
)");
}

TEST(PreprocessorTest, ResolveMacros) {
  std::string input = R"(
MAIN_FUNCTION($0) {
  float x = fabs(1.0f);
  float y = rsqrt(1.0f);
}
)";
  ExtensionsInfo extensions_info;
  MLD_ASSERT_OK(ConvertToWGSL(GetWebGpuInfo(), &input, &extensions_info));
  EXPECT_EQ(input, R"(
MAIN_FUNCTION($0) {
  var x : f32= abs(1.0);
  var y : f32= inverseSqrt(1.0);
}
)");
}

TEST(PreprocessorTest, ResolveDefine) {
  std::string input = R"(
#define FOO(x) x.foo
MAIN_FUNCTION($0) {
  float y = FOO(z);
}
)";
  ExtensionsInfo extensions_info;
  MLD_ASSERT_OK(ConvertToWGSL(GetWebGpuInfo(), &input, &extensions_info));
  EXPECT_EQ(input, R"(

MAIN_FUNCTION($0) {
  var y : f32= z.foo;
}
)");
}

TEST(PreprocessorTest, ParseOperators) {
  std::string input = R"(
MAIN_FUNCTION($0) {
  int x = 1;
  x <<= 5;
  int y = 1 << 5;
}
)";
  ExtensionsInfo extensions_info;
  MLD_ASSERT_OK(ConvertToWGSL(GetWebGpuInfo(), &input, &extensions_info));
  EXPECT_EQ(input, R"(
MAIN_FUNCTION($0) {
  var x : i32= 1;
  x <<= 5;
  var y : i32= 1 << 5;
}
)");
}

TEST(PreprocessorTest, ParseEnableExtensions) {
  std::string input = R"(
enable ext0;
enable ext1;
MAIN_FUNCTION($0) {
}
)";
  ExtensionsInfo extensions_info;
  MLD_ASSERT_OK(ConvertToWGSL(GetWebGpuInfo(), &input, &extensions_info));
  EXPECT_EQ(input, R"(


MAIN_FUNCTION($0) {
}
)");
  EXPECT_EQ(extensions_info.enable_extensions.size(), 2);
  EXPECT_EQ(extensions_info.enable_extensions[0], "ext0");
  EXPECT_EQ(extensions_info.enable_extensions[1], "ext1");
}

TEST(PreprocessorTest, ParseLanguageExtensions) {
  std::string input = R"(
requires ext0;
requires ext1;
MAIN_FUNCTION($0) {
}
)";
  ExtensionsInfo extensions_info;
  MLD_ASSERT_OK(ConvertToWGSL(GetWebGpuInfo(), &input, &extensions_info));
  EXPECT_EQ(input, R"(


MAIN_FUNCTION($0) {
}
)");
  EXPECT_EQ(extensions_info.language_extensions.size(), 2);
  EXPECT_EQ(extensions_info.language_extensions[0], "ext0");
  EXPECT_EQ(extensions_info.language_extensions[1], "ext1");
}

TEST(PreprocessorTest, ParseMixedExtensions) {
  std::string input = R"(
requires ext0;
enable ext3;
requires ext1;
MAIN_FUNCTION($0) {
}
)";
  ExtensionsInfo extensions_info;
  MLD_ASSERT_OK(ConvertToWGSL(GetWebGpuInfo(), &input, &extensions_info));
  EXPECT_EQ(input, R"(



MAIN_FUNCTION($0) {
}
)");
  EXPECT_EQ(extensions_info.language_extensions.size(), 2);
  EXPECT_EQ(extensions_info.language_extensions[0], "ext0");
  EXPECT_EQ(extensions_info.language_extensions[1], "ext1");
  EXPECT_EQ(extensions_info.enable_extensions.size(), 1);
  EXPECT_EQ(extensions_info.enable_extensions[0], "ext3");
}

}  // namespace webgpu
}  // namespace ml_drift
