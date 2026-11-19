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

#include "ml_drift/common/data_type.h"

#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"

namespace ml_drift {
namespace {

TEST(DataTypeTest, GlslShaderDataTypes) {
  // Float16
  EXPECT_EQ("float", ToGlslShaderDataType(DataType::kFloat16));
  EXPECT_EQ("mediump float",
            ToGlslShaderDataType(DataType::kFloat16, 1, /*add_precision*/ true,
                                 /*explicit_fp16 */ false));
  EXPECT_EQ("float16_t",
            ToGlslShaderDataType(DataType::kFloat16, 1, /*add_precision*/ false,
                                 /*explicit_fp16 */ true));
  EXPECT_EQ("float16_t",
            ToGlslShaderDataType(DataType::kFloat16, 1, /*add_precision*/ true,
                                 /*explicit_fp16 */ true));

  // vec4 Float16
  EXPECT_EQ("vec4", ToGlslShaderDataType(DataType::kFloat16, 4));
  EXPECT_EQ("mediump vec4",
            ToGlslShaderDataType(DataType::kFloat16, 4, /*add_precision*/ true,
                                 /*explicit_fp16 */ false));
  EXPECT_EQ("f16vec4",
            ToGlslShaderDataType(DataType::kFloat16, 4, /*add_precision*/ false,
                                 /*explicit_fp16 */ true));
  EXPECT_EQ("f16vec4",
            ToGlslShaderDataType(DataType::kFloat16, 4, /*add_precision*/ true,
                                 /*explicit_fp16 */ true));

  // Float32
  EXPECT_EQ("float", ToGlslShaderDataType(DataType::kFloat32));
  EXPECT_EQ("highp float", ToGlslShaderDataType(DataType::kFloat32, 1,
                                                /*add_precision*/ true));
  EXPECT_EQ("float", ToGlslShaderDataType(DataType::kFloat32, 1,
                                          /*add_precision*/ false));

  // vec2 Float32
  EXPECT_EQ("vec2", ToGlslShaderDataType(DataType::kFloat32, 2));
  EXPECT_EQ("highp vec2", ToGlslShaderDataType(DataType::kFloat32, 2,
                                               /*add_precision*/ true));
  EXPECT_EQ("vec2", ToGlslShaderDataType(DataType::kFloat32, 2,
                                         /*add_precision*/ false));

  // Int
  EXPECT_EQ("int",
            ToGlslShaderDataType(DataType::kInt64, 1, /*add_precision*/ false));
  EXPECT_EQ("int",
            ToGlslShaderDataType(DataType::kInt32, 1, /*add_precision*/ false));
  EXPECT_EQ("int",
            ToGlslShaderDataType(DataType::kInt16, 1, /*add_precision*/ false));
  EXPECT_EQ("int",
            ToGlslShaderDataType(DataType::kInt8, 1, /*add_precision*/ false));
  EXPECT_EQ("int",
            ToGlslShaderDataType(DataType::kInt64, 1, /*add_precision*/ true));
  EXPECT_EQ("highp int",
            ToGlslShaderDataType(DataType::kInt32, 1, /*add_precision*/ true));
  EXPECT_EQ("mediump int",
            ToGlslShaderDataType(DataType::kInt16, 1, /*add_precision*/ true));
  EXPECT_EQ("lowp int",
            ToGlslShaderDataType(DataType::kInt8, 1, /*add_precision*/ true));

  // Uint
  EXPECT_EQ("uint", ToGlslShaderDataType(DataType::kUint64, 1,
                                         /*add_precision*/ false));
  EXPECT_EQ("uint", ToGlslShaderDataType(DataType::kUint32, 1,
                                         /*add_precision*/ false));
  EXPECT_EQ("uint", ToGlslShaderDataType(DataType::kUint16, 1,
                                         /*add_precision*/ false));
  EXPECT_EQ("uint",
            ToGlslShaderDataType(DataType::kUint8, 1, /*add_precision*/ false));
  EXPECT_EQ("uint",
            ToGlslShaderDataType(DataType::kUint64, 1, /*add_precision*/ true));
  EXPECT_EQ("highp uint",
            ToGlslShaderDataType(DataType::kUint32, 1, /*add_precision*/ true));
  EXPECT_EQ("mediump uint",
            ToGlslShaderDataType(DataType::kUint16, 1, /*add_precision*/ true));
  EXPECT_EQ("lowp uint",
            ToGlslShaderDataType(DataType::kUint8, 1, /*add_precision*/ true));

  EXPECT_EQ("bool", ToGlslShaderDataType(DataType::kBool));
  EXPECT_EQ("bvec4", ToGlslShaderDataType(DataType::kBool, 4));
  EXPECT_EQ("bool",
            ToGlslShaderDataType(DataType::kBool, 1, /*add_precision*/ true));
  EXPECT_EQ("bool", ToGlslShaderDataType(DataType::kBool, 1,
                                         /*add_precision*/ false));
}

}  // namespace
}  // namespace ml_drift
