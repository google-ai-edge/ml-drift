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

// testing function from unnamed namespace
#include "ml_drift/cl/util.h"

#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "ml_drift/common/gpu_info.h"
namespace ml_drift {
namespace cl {

TEST(QualcommOpenClCompilerVersionParsing, Base) {
  auto result =
      GetQualcommOpenClCompilerVersion("random text Compiler E031.79.53.41");
  EXPECT_EQ(result.major, 79);
  EXPECT_EQ(result.minor, 53);
  EXPECT_EQ(result.patch, 41);
}

TEST(QualcommOpenClCompilerVersionParsing, WrongFormat0) {
  auto result =
      GetQualcommOpenClCompilerVersion("random text Assembler A337.79.53.41");
  EXPECT_EQ(result.major, 0);
  EXPECT_EQ(result.minor, 0);
  EXPECT_EQ(result.patch, 0);
}

TEST(QualcommOpenClCompilerVersionParsing, WrongFormat1) {
  auto result =
      GetQualcommOpenClCompilerVersion("random text Compiler E031.79.53.4");
  EXPECT_EQ(result.major, 0);
  EXPECT_EQ(result.minor, 0);
  EXPECT_EQ(result.patch, 0);
}

TEST(QualcommOpenClCompilerVersionParsing, WrongFormat2) {
  auto result =
      GetQualcommOpenClCompilerVersion("random text Compiler E031:79:53:41");
  EXPECT_EQ(result.major, 0);
  EXPECT_EQ(result.minor, 0);
  EXPECT_EQ(result.patch, 0);
}

TEST(QualcommOpenClCompilerVersionParsing, WrongFormat3) {
  auto result =
      GetQualcommOpenClCompilerVersion("random text Compiler E031.79.x53.41");
  EXPECT_EQ(result.major, 0);
  EXPECT_EQ(result.minor, 0);
  EXPECT_EQ(result.patch, 0);
}

TEST(QualcommOpenClCompilerVersionParsing, WrongFormat4) {
  auto result =
      GetQualcommOpenClCompilerVersion("random text Compiler E031.a9.53.41");
  EXPECT_EQ(result.major, 0);
  EXPECT_EQ(result.minor, 0);
  EXPECT_EQ(result.patch, 0);
}

}  // namespace cl
}  // namespace ml_drift
