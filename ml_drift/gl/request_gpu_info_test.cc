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

#include "ml_drift/gl/request_gpu_info.h"

#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "ml_drift/common/gpu_info.h"

namespace ml_drift {
namespace gl {

TEST(AdrenoGlDriverVersionParsing, Real0) {
  auto result = GeAdrenoDriverVersion(
      "OpenGL ES 3.2 V@0615.91 (GIT@a63fb93269, I16b8254a37, 1724729505) "
      "(Date:08/27/24)");
  EXPECT_EQ(result.major, 615);
  EXPECT_EQ(result.minor, 91);
  EXPECT_EQ(result.patch, 0);
}

TEST(AdrenoGlDriverVersionParsing, Real1) {
  auto result = GeAdrenoDriverVersion(
      "OpenGL ES 3.2 V@0502.11.5 (GIT@db7e7066a2, I1fad33e957, 1746950372) "
      "(Date:05/11/25)");
  EXPECT_EQ(result.major, 502);
  EXPECT_EQ(result.minor, 11);
  EXPECT_EQ(result.patch, 5);
}

TEST(AdrenoGlDriverVersionParsing, Real2) {
  auto result = GeAdrenoDriverVersion(
      "OpenGL ES 3.2 V@0530.0 (GIT@f3aa497c33, I7c31881d50, 1639653329) "
      "(Date:12/16/21)");
  EXPECT_EQ(result.major, 530);
  EXPECT_EQ(result.minor, 0);
  EXPECT_EQ(result.patch, 0);
}

TEST(AdrenoGlDriverVersionParsing, OnlyMajor) {
  auto result = GeAdrenoDriverVersion("V@0555");
  EXPECT_EQ(result.major, 555);
  EXPECT_EQ(result.minor, 0);
  EXPECT_EQ(result.patch, 0);
}

TEST(AdrenoGlDriverVersionParsing, ExtraParts) {
  auto result = GeAdrenoDriverVersion("V@0555.44.3.2");
  EXPECT_EQ(result.major, 555);
  EXPECT_EQ(result.minor, 44);
  EXPECT_EQ(result.patch, 3);
}

TEST(AdrenoGlDriverVersionParsing, WrongFormat0) {
  auto result = GeAdrenoDriverVersion("V234.12");
  EXPECT_EQ(result.major, 0);
  EXPECT_EQ(result.minor, 0);
  EXPECT_EQ(result.patch, 0);
}

TEST(AdrenoGlDriverVersionParsing, WrongFormat1) {
  auto result = GeAdrenoDriverVersion("V@234.abc");
  EXPECT_EQ(result.major, 234);
  EXPECT_EQ(result.minor, 0);
  EXPECT_EQ(result.patch, 0);
}

TEST(AdrenoGlDriverVersionParsing, WrongFormat2) {
  auto result = GeAdrenoDriverVersion("V@234.12.abc");
  EXPECT_EQ(result.major, 234);
  EXPECT_EQ(result.minor, 12);
  EXPECT_EQ(result.patch, 0);
}

TEST(AdrenoGlDriverVersionParsing, WrongFormat3) {
  auto result = GeAdrenoDriverVersion("");
  EXPECT_EQ(result.major, 0);
  EXPECT_EQ(result.minor, 0);
  EXPECT_EQ(result.patch, 0);
}

TEST(AdrenoGlDriverVersionParsing, WrongFormat4) {
  auto result = GeAdrenoDriverVersion("V@");
  EXPECT_EQ(result.major, 0);
  EXPECT_EQ(result.minor, 0);
  EXPECT_EQ(result.patch, 0);
}

}  // namespace gl
}  // namespace ml_drift
