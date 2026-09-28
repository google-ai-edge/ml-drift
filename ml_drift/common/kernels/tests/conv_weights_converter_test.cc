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

#include <array>
#include <string>
#include <tuple>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/conv_weights_converter_test_util.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/task/weights_layout.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::Test;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;
using ::testing::WithParamInterface;

namespace {
// Dst Weight Layouts to test
std::array<WeightsDescription, 5> WeightsDescsIntOHWIToUInt() {
  return {
      {{.layout = WeightsLayout::kOSpatialIOGroupI4O4, .output_group_size = 16},
       {.layout = WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4,
        .output_group_size = 16},
       {.layout = WeightsLayout::kOSpatialIOGroupO4I4, .output_group_size = 16},
       {.layout = WeightsLayout::kCustomGroups,
        .group_sizes = {{Axis::kInputChannels, 2},
                        {Axis::kOutputChannels, 4},
                        {Axis::kInputChannels, 2},
                        {Axis::kInputChannels, 3},
                        {Axis::kOutputChannels, 5},
                        {Axis::kInputChannels, 0},
                        {Axis::kWidth, 0},
                        {Axis::kHeight, 0},
                        {Axis::kOutputChannels, 0}}},
       {.layout = WeightsLayout::kCustomGroups,
        .group_sizes = {{Axis::kInputChannels, 4},
                        {Axis::kOutputChannels, 4},
                        {Axis::kInputChannels, 2},
                        {Axis::kOutputChannels, 3},
                        {Axis::kInputChannels, 0},
                        {Axis::kWidth, 0},
                        {Axis::kHeight, 0},
                        {Axis::kOutputChannels, 0}}}}};
};

std::array<WeightsDescription, 5> WeightsDescsIntOHWIToFloat() {
  return {
      {{.layout = WeightsLayout::kOSpatialIOGroupI4O4, .output_group_size = 16},
       {.layout = WeightsLayout::kOSpatialIOGroupO4I4, .output_group_size = 16},
       {.layout = WeightsLayout::kCustomGroups,
        .group_sizes = {{Axis::kInputChannels, 2},
                        {Axis::kOutputChannels, 4},
                        {Axis::kInputChannels, 2},
                        {Axis::kInputChannels, 3},
                        {Axis::kOutputChannels, 5},
                        {Axis::kInputChannels, 0},
                        {Axis::kWidth, 0},
                        {Axis::kHeight, 0},
                        {Axis::kOutputChannels, 0}}},
       {.layout = WeightsLayout::kCustomGroups,
        .group_sizes = {{Axis::kInputChannels, 4},
                        {Axis::kOutputChannels, 4},
                        {Axis::kInputChannels, 2},
                        {Axis::kOutputChannels, 3},
                        {Axis::kInputChannels, 0},
                        {Axis::kWidth, 0},
                        {Axis::kHeight, 0},
                        {Axis::kOutputChannels, 0}}},
       {.layout = WeightsLayout::kCustomGroups,
        .group_sizes = {{Axis::kInputChannels, 32},
                        {Axis::kOutputChannels, 16},
                        {Axis::kOutputChannels, 1},
                        {Axis::kInputChannels, 0},
                        {Axis::kWidth, 0},
                        {Axis::kHeight, 0},
                        {Axis::kOutputChannels, 0}}}}};
};

std::array<WeightsDescription, 9> WeightsDescsIntToFloat() {
  return {{{
               .layout = WeightsLayout::kOSpatialIOGroupI4O4,
               .output_group_size = 16,
           },
           {
               .layout = WeightsLayout::kOSpatialIOGroupO4I4,
               .output_group_size = 16,
           },
           {
               .layout = WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4,
               .output_group_size = 2,
           },
           {
               .layout = WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4,
               .output_group_size = 2,
           },
           {
               .layout = WeightsLayout::kISpatialOI4O4UnalignedIO,
           },
           {
               .layout = WeightsLayout::kCustomGroups,
               .group_sizes = {{Axis::kOutputChannels, 8},
                               {Axis::kInputChannels, 8},
                               {Axis::kOutputChannels, 4},
                               {Axis::kInputChannels, 0},
                               {Axis::kWidth, 0},
                               {Axis::kHeight, 0},
                               {Axis::kOutputChannels, 0}},
           },
           {
               .layout = WeightsLayout::kCustomGroups,
               .group_sizes = {{Axis::kInputChannels, 4},
                               {Axis::kOutputChannels, 4},
                               {Axis::kWidth, 0},
                               {Axis::kHeight, 0},
                               {Axis::kInputChannels, 0},
                               {Axis::kOutputChannels, 0}},
           },
           {
               .layout = WeightsLayout::kCustomGroups,
               .group_sizes = {{Axis::kOutputChannels, 4},
                               {Axis::kInputChannels, 4},
                               {Axis::kWidth, 0},
                               {Axis::kHeight, 0},
                               {Axis::kOutputChannels, 0},
                               {Axis::kInputChannels, 0}},
           },
           {
               .layout = WeightsLayout::kCustomGroups,
               .group_sizes = {{Axis::kInputChannels, 32},
                               {Axis::kOutputChannels, 16},
                               {Axis::kOutputChannels, 1},
                               {Axis::kInputChannels, 0},
                               {Axis::kWidth, 0},
                               {Axis::kHeight, 0},
                               {Axis::kOutputChannels, 0}},
           }}};
};

std::array<WeightsDescription, 8> WeightsDescsI8ToUI8() {
  return {{{
               .type = DataType::kUint8,
               .layout = WeightsLayout::kOSpatialIOGroupI4O4,
               .output_group_size = 16,
           },
           {
               .type = DataType::kUint8,
               .layout = WeightsLayout::kOSpatialIOGroupO4I4,
               .output_group_size = 16,
           },
           {.type = DataType::kUint8,
            .layout = WeightsLayout::kCustomGroups,
            .group_sizes = {{Axis::kInputChannels, 2},
                            {Axis::kOutputChannels, 4},
                            {Axis::kInputChannels, 2},
                            {Axis::kInputChannels, 3},
                            {Axis::kOutputChannels, 5},
                            {Axis::kInputChannels, 0},
                            {Axis::kWidth, 0},
                            {Axis::kHeight, 0},
                            {Axis::kOutputChannels, 0}}},
           {.type = DataType::kUint8,
            .layout = WeightsLayout::kCustomGroups,
            .group_sizes = {{Axis::kInputChannels, 4},
                            {Axis::kOutputChannels, 4},
                            {Axis::kInputChannels, 2},
                            {Axis::kOutputChannels, 3},
                            {Axis::kInputChannels, 0},
                            {Axis::kWidth, 0},
                            {Axis::kHeight, 0},
                            {Axis::kOutputChannels, 0}}},
           {
               .type = DataType::kUint8,
               .layout = WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4,
               .output_group_size = 16,
           },
           {
               .type = DataType::kUint8,
               .layout = WeightsLayout::kCustomGroups,
               .group_sizes = {{Axis::kInputChannels, 4},
                               {Axis::kOutputChannels, 4},
                               {Axis::kInputChannels, 0},
                               {Axis::kOutputChannels, 0}},
           },
           {
               .type = DataType::kUint8,
               .layout = WeightsLayout::kCustomGroups,
               .group_sizes = {{Axis::kOutputChannels, 4},
                               {Axis::kInputChannels, 4},
                               {Axis::kOutputChannels, 0},
                               {Axis::kInputChannels, 0}},
           },
           {
               .type = DataType::kUint8,
               .layout = WeightsLayout::kCustomGroups,
               .group_sizes = {{Axis::kInputChannels, 32},
                               {Axis::kOutputChannels, 16},
                               {Axis::kOutputChannels, 1},
                               {Axis::kInputChannels, 0},
                               {Axis::kWidth, 0},
                               {Axis::kHeight, 0},
                               {Axis::kOutputChannels, 0}},
           }}};
};

std::array<WeightsDescription, 9> WeightsDescsF32ToF32() {
  return {{{
               .layout = WeightsLayout::kOSpatialIOGroupI4O4,
               .output_group_size = 16,
           },
           {
               .layout = WeightsLayout::kOSpatialIOGroupO4I4,
               .output_group_size = 16,
           },
           {
               .layout = WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4,
               .output_group_size = 2,
           },
           {
               .layout = WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4,
               .output_group_size = 2,
           },
           {
               .layout = WeightsLayout::kCustomGroups,
               .group_sizes = {{Axis::kOutputChannels, 8},
                               {Axis::kInputChannels, 8},
                               {Axis::kOutputChannels, 4},
                               {Axis::kInputChannels, 0},
                               {Axis::kWidth, 0},
                               {Axis::kHeight, 0},
                               {Axis::kOutputChannels, 0}},
           },
           {.layout = WeightsLayout::kCustomGroups,
            .group_sizes = {{Axis::kInputChannels, 2},
                            {Axis::kOutputChannels, 4},
                            {Axis::kInputChannels, 2},
                            {Axis::kInputChannels, 3},
                            {Axis::kOutputChannels, 5},
                            {Axis::kInputChannels, 0},
                            {Axis::kWidth, 0},
                            {Axis::kHeight, 0},
                            {Axis::kOutputChannels, 0}}},
           {.layout = WeightsLayout::kCustomGroups,
            .group_sizes = {{Axis::kInputChannels, 4},
                            {Axis::kOutputChannels, 4},
                            {Axis::kInputChannels, 2},
                            {Axis::kOutputChannels, 3},
                            {Axis::kInputChannels, 0},
                            {Axis::kWidth, 0},
                            {Axis::kHeight, 0},
                            {Axis::kOutputChannels, 0}}},
           {.layout = WeightsLayout::kCustomGroups,
            .group_sizes = {{Axis::kInputChannels, 32},
                            {Axis::kOutputChannels, 16},
                            {Axis::kOutputChannels, 1},
                            {Axis::kInputChannels, 0},
                            {Axis::kWidth, 0},
                            {Axis::kHeight, 0},
                            {Axis::kOutputChannels, 0}}},
           {
               .layout = WeightsLayout::kISpatialOI4O4UnalignedIO,
           }}};
};

std::array<WeightsDescription, 7> WeightsDescsUITo8Bit() {
  return {{{
               .layout = WeightsLayout::kOSpatialIOGroupI4O4,
               .output_group_size = 16,
           },
           {
               .layout = WeightsLayout::kOSpatialIOGroupO4I4,
               .output_group_size = 16,
           },
           {.layout = WeightsLayout::kCustomGroups,
            .group_sizes = {{Axis::kInputChannels, 2},
                            {Axis::kOutputChannels, 4},
                            {Axis::kInputChannels, 2},
                            {Axis::kInputChannels, 8},
                            {Axis::kOutputChannels, 2},
                            {Axis::kInputChannels, 0},
                            {Axis::kWidth, 0},
                            {Axis::kHeight, 0},
                            {Axis::kOutputChannels, 0}}},
           {.layout = WeightsLayout::kCustomGroups,
            .group_sizes = {{Axis::kInputChannels, 4},
                            {Axis::kOutputChannels, 4},
                            {Axis::kInputChannels, 3},
                            {Axis::kOutputChannels, 5},
                            {Axis::kInputChannels, 0},
                            {Axis::kWidth, 0},
                            {Axis::kHeight, 0},
                            {Axis::kOutputChannels, 0}}},
           {
               .layout = WeightsLayout::kCustomGroups,
               .group_sizes = {{Axis::kInputChannels, 4},
                               {Axis::kOutputChannels, 4},
                               {Axis::kInputChannels, 0},
                               {Axis::kOutputChannels, 0}},
           },
           {
               .layout = WeightsLayout::kCustomGroups,
               .group_sizes = {{Axis::kOutputChannels, 4},
                               {Axis::kInputChannels, 4},
                               {Axis::kOutputChannels, 0},
                               {Axis::kInputChannels, 0}},
           },
           {
               .layout = WeightsLayout::kCustomGroups,
               .group_sizes = {{Axis::kInputChannels, 32},
                               {Axis::kOutputChannels, 16},
                               {Axis::kOutputChannels, 1},
                               {Axis::kInputChannels, 0},
                               {Axis::kWidth, 0},
                               {Axis::kHeight, 0},
                               {Axis::kOutputChannels, 0}},
           }}};
};

std::array<WeightsDescription, 4> WeightsDescsUITo4Bit() {
  return {{{
               .layout = WeightsLayout::kOSpatialIOGroupI4O4,
               .output_group_size = 16,
           },
           {
               .layout = WeightsLayout::kOSpatialIOGroupO4I4,
               .output_group_size = 16,
           },
           {.layout = WeightsLayout::kCustomGroups,
            .group_sizes = {{Axis::kInputChannels, 2},
                            {Axis::kOutputChannels, 4},
                            {Axis::kInputChannels, 2},
                            {Axis::kInputChannels, 8},
                            {Axis::kOutputChannels, 2},
                            {Axis::kInputChannels, 0},
                            {Axis::kWidth, 0},
                            {Axis::kHeight, 0},
                            {Axis::kOutputChannels, 0}}},
           {.layout = WeightsLayout::kCustomGroups,
            .group_sizes = {{Axis::kInputChannels, 4},
                            {Axis::kOutputChannels, 4},
                            {Axis::kInputChannels, 3},
                            {Axis::kOutputChannels, 5},
                            {Axis::kInputChannels, 0},
                            {Axis::kWidth, 0},
                            {Axis::kHeight, 0},
                            {Axis::kOutputChannels, 0}}}}};
};

std::array<WeightsDescription, 12> WeightsDescsCustomGroups() {
  return {{{
               .type = DataType::kFloat32,
               .layout = WeightsLayout::kCustomGroups,
               .group_sizes = {{Axis::kInputChannels, 8},
                               {Axis::kOutputChannels, 12},
                               {Axis::kInputChannels, 0},
                               {Axis::kOutputChannels, 0}},
           },
           {
               .type = DataType::kFloat32,
               .layout = WeightsLayout::kCustomGroups,
               .group_sizes = {{Axis::kOutputChannels, 8},
                               {Axis::kInputChannels, 12},
                               {Axis::kOutputChannels, 0},
                               {Axis::kInputChannels, 0}},
           },
           {
               .type = DataType::kFloat32,
               .layout = WeightsLayout::kCustomGroups,
               .group_sizes = {{Axis::kInputChannels, 4},
                               {Axis::kOutputChannels, 4},
                               {Axis::kInputChannels, 0},
                               {Axis::kOutputChannels, 0}},
           },
           {
               .type = DataType::kFloat32,
               .layout = WeightsLayout::kCustomGroups,
               .group_sizes = {{Axis::kOutputChannels, 4},
                               {Axis::kInputChannels, 4},
                               {Axis::kOutputChannels, 0},
                               {Axis::kInputChannels, 0}},
           },
           {
               .type = DataType::kFloat32,
               .layout = WeightsLayout::kCustomGroups,
               .group_sizes = {{Axis::kInputChannels, 4},
                               {Axis::kOutputChannels, 4},
                               {Axis::kInputChannels, 4},
                               {Axis::kOutputChannels, 0},
                               {Axis::kInputChannels, 0}},
           },
           {
               .type = DataType::kFloat32,
               .layout = WeightsLayout::kCustomGroups,
               .group_sizes = {{Axis::kOutputChannels, 4},
                               {Axis::kInputChannels, 4},
                               {Axis::kOutputChannels, 4},
                               {Axis::kInputChannels, 0},
                               {Axis::kOutputChannels, 0}},
           },
           {
               .type = DataType::kFloat32,
               .layout = WeightsLayout::kCustomGroups,
               .group_sizes = {{Axis::kInputChannels, 4},
                               {Axis::kOutputChannels, 4},
                               {Axis::kInputChannels, 4},
                               {Axis::kOutputChannels, 4},
                               {Axis::kInputChannels, 0},
                               {Axis::kOutputChannels, 0}},
           },
           {
               .type = DataType::kFloat32,
               .layout = WeightsLayout::kCustomGroups,
               .group_sizes = {{Axis::kOutputChannels, 4},
                               {Axis::kInputChannels, 4},
                               {Axis::kOutputChannels, 4},
                               {Axis::kInputChannels, 4},
                               {Axis::kOutputChannels, 0},
                               {Axis::kInputChannels, 0}},
           },
           {
               .type = DataType::kFloat32,
               .layout = WeightsLayout::kCustomGroups,
               .group_sizes = {{Axis::kInputChannels, 2},
                               {Axis::kOutputChannels, 2},
                               {Axis::kInputChannels, 2},
                               {Axis::kOutputChannels, 2},
                               {Axis::kInputChannels, 0},
                               {Axis::kOutputChannels, 0}},
           },
           {
               .type = DataType::kFloat32,
               .layout = WeightsLayout::kCustomGroups,
               .group_sizes = {{Axis::kOutputChannels, 2},
                               {Axis::kInputChannels, 4},
                               {Axis::kOutputChannels, 2},
                               {Axis::kInputChannels, 0},
                               {Axis::kOutputChannels, 0}},
           },
           {
               .type = DataType::kFloat32,
               .layout = WeightsLayout::kCustomGroups,
               .group_sizes = {{Axis::kInputChannels, 2},
                               {Axis::kOutputChannels, 4},
                               {Axis::kInputChannels, 2},
                               {Axis::kOutputChannels, 0},
                               {Axis::kInputChannels, 0}},
           },
           {
               .type = DataType::kFloat32,
               .layout = WeightsLayout::kCustomGroups,
               .group_sizes = {{Axis::kInputChannels, 32},
                               {Axis::kOutputChannels, 16},
                               {Axis::kOutputChannels, 1},
                               {Axis::kInputChannels, 0},
                               {Axis::kWidth, 0},
                               {Axis::kHeight, 0},
                               {Axis::kOutputChannels, 0}},
           }}};
};

std::array<WeightsDescription, 3> WeightsDescsCustomGroupsOIorIO() {
  return {{{
               .type = DataType::kFloat32,
               .layout = WeightsLayout::kCustomGroups,
               .group_sizes = {{Axis::kInputChannels, 0},
                               {Axis::kOutputChannels, 0}},
           },
           {
               .type = DataType::kFloat32,
               .layout = WeightsLayout::kCustomGroups,
               .group_sizes = {{Axis::kOutputChannels, 0},
                               {Axis::kInputChannels, 0}},
           },
           {
               .type = DataType::kFloat32,
               .layout = WeightsLayout::kCustomGroups,
               .group_sizes = {{Axis::kInputChannels, 32},
                               {Axis::kOutputChannels, 16},
                               {Axis::kOutputChannels, 1},
                               {Axis::kInputChannels, 0},
                               {Axis::kWidth, 0},
                               {Axis::kHeight, 0},
                               {Axis::kOutputChannels, 0}},
           }}};
};
}  // namespace

class OutTest : public Test,
                public WithParamInterface<
                    std::tuple<DataType, TensorStorageType, WeightsLayout>> {};

TEST_P(OutTest, 1x1OutX4Test) {
  auto [data_type, storage, weights_layout] = GetParam();
  if (!exec_env->IsStorageSupported(storage, data_type)) {
    GTEST_SKIP() << "Unsupported data_type " << ToString(data_type)
                 << " and storage type: " << ToString(storage);
  }
  ABSL_ASSERT_OK(ConverterToConvWeights1x1OutX4Test(*exec_env, data_type, storage,
                                               weights_layout));
}

TEST_P(OutTest, 1x1OutX4UnalignedTest) {
  auto [data_type, storage, weights_layout] = GetParam();
  if (!exec_env->IsStorageSupported(storage, data_type)) {
    GTEST_SKIP() << "Unsupported data_type " << ToString(data_type)
                 << " and storage type: " << ToString(storage);
  }
  ABSL_ASSERT_OK(ConverterToConvWeights1x1OutX4UnalignedTest(
      *exec_env, data_type, storage, weights_layout));
}

TEST_P(OutTest, 1x1OutX2Test) {
  auto [data_type, storage, weights_layout] = GetParam();
  if (!exec_env->IsStorageSupported(storage, data_type)) {
    GTEST_SKIP() << "Unsupported data_type " << ToString(data_type)
                 << " and storage type: " << ToString(storage);
  }
  ABSL_ASSERT_OK(ConverterToConvWeights1x1OutX2Test(*exec_env, data_type, storage,
                                               weights_layout));
}

TEST_P(OutTest, OutX2Test) {
  auto [data_type, storage, weights_layout] = GetParam();
  if (!exec_env->IsStorageSupported(storage, data_type)) {
    GTEST_SKIP() << "Unsupported data_type " << ToString(data_type)
                 << " and storage type: " << ToString(storage);
  }
  ABSL_ASSERT_OK(ConverterToConvWeightsOutX2Test(*exec_env, data_type, storage,
                                            weights_layout));
}

INSTANTIATE_TEST_SUITE_P(
    ConvWeightsConverterOutTestSuite, OutTest,
    Combine(ValuesIn(GetFloatTypes()),
            ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D()),
            ValuesIn({WeightsLayout::kOSpatialIOGroupI4O4,
                      WeightsLayout::kOSpatialIOGroupO4I4})),
    [](const TestParamInfo<OutTest::ParamType>& info) {
      return absl::StrReplaceAll(
          absl::StrCat(ToString(std::get<0>(info.param)), "_",
                       ToString(std::get<1>(info.param)), "_",
                       ToString(std::get<2>(info.param))),
          {{":", ""}});
    });

class X4TexturesTest
    : public Test,
      public WithParamInterface<
          std::tuple<DataType, TensorStorageType, WeightsLayout>> {};

TEST_P(X4TexturesTest, TransposedWeights4xTexturesTest) {
  auto [data_type, storage, weights_layout] = GetParam();
  if (!exec_env->IsStorageSupported(storage, data_type)) {
    GTEST_SKIP() << "Unsupported data_type " << ToString(data_type)
                 << " and storage type: " << ToString(storage);
  }
  ABSL_ASSERT_OK(ConverterToConvWeights4xTexturesTest(*exec_env, data_type, storage,
                                                 weights_layout));
}

INSTANTIATE_TEST_SUITE_P(
    ConvWeightsConverter4xTexturesTestSuite, X4TexturesTest,
    Combine(ValuesIn(GetFloatTypes()),
            ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D()),
            ValuesIn({WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4,
                      WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4})),
    [](const TestParamInfo<X4TexturesTest::ParamType>& info) {
      return absl::StrReplaceAll(
          absl::StrCat(ToString(std::get<0>(info.param)), "_",
                       ToString(std::get<1>(info.param)), "_",
                       ToString(std::get<2>(info.param))),
          {{":", ""}});
    });

class F32OHWItoF32Test : public Test,
                         public WithParamInterface<std::tuple<DataType, OHWI>> {
};

TEST_P(F32OHWItoF32Test, Float32OHWItoFloat32Test) {
  auto [data_type, weights_shape] = GetParam();
  if (!exec_env->IsStorageSupported(TensorStorageType::kBuffer, data_type)) {
    GTEST_SKIP() << "Unsupported data_type " << ToString(data_type);
  }
  ABSL_ASSERT_OK(ConverterToConvWeightsFloat32OHWItoFloat32Test(*exec_env, data_type,
                                                           weights_shape));
}

INSTANTIATE_TEST_SUITE_P(
    ConvWeightsConverterF32OHWItoF32TestSuite, F32OHWItoF32Test,
    Combine(ValuesIn(GetFloatTypes()),
            ValuesIn({OHWI(16, 1, 1, 16), OHWI(13, 1, 1, 16),
                      OHWI(16, 1, 1, 13), OHWI(13, 1, 1, 13),
                      OHWI(16, 3, 2, 16), OHWI(13, 3, 2, 16),
                      OHWI(16, 3, 2, 13), OHWI(13, 3, 2, 13)})),
    [](const TestParamInfo<F32OHWItoF32Test::ParamType>& info) {
      const OHWI weights_shape = std::get<1>(info.param);
      return absl::StrReplaceAll(
          absl::StrCat(ToString(std::get<0>(info.param)), "_", weights_shape.o,
                       "_", weights_shape.h, "_", weights_shape.w, "_",
                       weights_shape.i),
          {{":", ""}});
    });

class FromIntOHWITest
    : public Test,
      public WithParamInterface<std::tuple<OHWI, WeightsDescription>> {};

TEST_P(FromIntOHWITest, Int8OHWIToUint8Test) {
  auto [weights_shape, conv_weight_desc] = GetParam();
  if (!exec_env->IsStorageSupported(TensorStorageType::kBuffer,
                                    DataType::kInt8) ||
      !exec_env->IsStorageSupported(TensorStorageType::kBuffer,
                                    DataType::kUint8)) {
    GTEST_SKIP() << "Unsupported int8/uint8 buffer storage";
  }
  conv_weight_desc.type = DataType::kUint8;
  ABSL_ASSERT_OK(ConverterToConvWeightsInt8OHWIToUint8Test(*exec_env, weights_shape,
                                                      conv_weight_desc));
}

TEST_P(FromIntOHWITest, Int2OHWIToUint2Test) {
  auto [weights_shape, conv_weight_desc] = GetParam();
  if (!exec_env->IsStorageSupported(TensorStorageType::kBuffer,
                                    DataType::kInt8)) {
    GTEST_SKIP() << "Unsupported data_type " << ToString(DataType::kInt8)
                 << " and storage type: "
                 << ToString(TensorStorageType::kBuffer);
  }
  conv_weight_desc.type = DataType::kUint2;
  ABSL_ASSERT_OK(ConverterToConvWeightsInt2OHWIToUint2Test(*exec_env, weights_shape,
                                                      conv_weight_desc));
}

TEST_P(FromIntOHWITest, Int4OHWIToUint4Test) {
  auto [weights_shape, conv_weight_desc] = GetParam();
  if (!exec_env->IsStorageSupported(TensorStorageType::kBuffer,
                                    DataType::kInt8)) {
    GTEST_SKIP() << "Unsupported data_type " << ToString(DataType::kInt8)
                 << " and storage type: "
                 << ToString(TensorStorageType::kBuffer);
  }
  conv_weight_desc.type = DataType::kUint4;
  ABSL_ASSERT_OK(ConverterToConvWeightsInt4OHWIToUint4Test(*exec_env, weights_shape,
                                                      conv_weight_desc));
}

INSTANTIATE_TEST_SUITE_P(
    ConverterToConvWeightsTestSuite, FromIntOHWITest,
    Combine(ValuesIn({// aligned shapes
                      OHWI(4, 1, 1, 4), OHWI(4, 1, 1, 16), OHWI(16, 1, 1, 4),
                      OHWI(16, 1, 1, 16), OHWI(32, 1, 1, 16),
                      OHWI(32, 1, 1, 32), OHWI(48, 1, 1, 32),
                      OHWI(56, 1, 1, 48), OHWI(4, 3, 2, 16), OHWI(4, 2, 3, 16),
                      // unaligned shapes
                      OHWI(4, 1, 1, 3), OHWI(3, 1, 1, 16), OHWI(7, 1, 1, 11),
                      OHWI(15, 1, 1, 16), OHWI(19, 1, 1, 17),
                      OHWI(31, 1, 1, 29), OHWI(43, 1, 1, 33),
                      OHWI(57, 1, 1, 47), OHWI(7, 3, 2, 5), OHWI(5, 2, 3, 7)}),
            ValuesIn(WeightsDescsIntOHWIToUInt())),
    [](const TestParamInfo<FromIntOHWITest::ParamType>& info) {
      const OHWI weights_shape = std::get<0>(info.param);
      const std::string weights_shape_str =
          absl::StrCat(weights_shape.o, "_", weights_shape.h, "_",
                       weights_shape.w, "_", weights_shape.i);
      return absl::StrCat(weights_shape_str, "_",
                          ToString(std::get<1>(info.param)));
    });

class FromIntOHWIToFloatTest
    : public Test,
      public WithParamInterface<
          std::tuple<DataType, OHWI, WeightsDescription>> {};

TEST_P(FromIntOHWIToFloatTest, Int2OHWIToFloatTest) {
  auto [data_type, weights_shape, conv_weight_desc] = GetParam();
  conv_weight_desc.type = data_type;
  ABSL_ASSERT_OK(ConverterToConvWeightsInt2OHWIToFloatTest(*exec_env, weights_shape,
                                                      conv_weight_desc));
}

TEST_P(FromIntOHWIToFloatTest, Int4OHWIToFloatTest) {
  auto [data_type, weights_shape, conv_weight_desc] = GetParam();
  conv_weight_desc.type = data_type;
  ABSL_ASSERT_OK(ConverterToConvWeightsInt4OHWIToFloatTest(*exec_env, weights_shape,
                                                      conv_weight_desc));
}

TEST_P(FromIntOHWIToFloatTest, Int8OHWIToFloatTest) {
  auto [data_type, weights_shape, conv_weight_desc] = GetParam();
  conv_weight_desc.type = data_type;
  ABSL_ASSERT_OK(ConverterToConvWeightsInt8OHWIToFloatTest(*exec_env, weights_shape,
                                                      conv_weight_desc));
}

INSTANTIATE_TEST_SUITE_P(
    ConverterToConvWeightsTestSuite, FromIntOHWIToFloatTest,
    // TODO: use GetFloatTypes()
    Combine(ValuesIn({DataType::kFloat32}),
            ValuesIn({// aligned shapes
                      OHWI(4, 1, 1, 4), OHWI(4, 1, 1, 16), OHWI(16, 1, 1, 4),
                      OHWI(16, 1, 1, 16), OHWI(32, 1, 1, 16),
                      OHWI(32, 1, 1, 32), OHWI(48, 1, 1, 32),
                      OHWI(56, 1, 1, 48), OHWI(4, 3, 2, 16), OHWI(4, 2, 3, 16),
                      // unaligned shapes
                      OHWI(4, 1, 1, 3), OHWI(3, 1, 1, 16), OHWI(7, 1, 1, 11),
                      OHWI(15, 1, 1, 16), OHWI(19, 1, 1, 17),
                      OHWI(31, 1, 1, 29), OHWI(43, 1, 1, 33),
                      OHWI(57, 1, 1, 47), OHWI(7, 3, 2, 5), OHWI(5, 2, 3, 7)}),
            ValuesIn(WeightsDescsIntOHWIToFloat())),
    [](const TestParamInfo<FromIntOHWIToFloatTest::ParamType>& info) {
      const std::string precision_name =
          absl::StrReplaceAll(ToString(std::get<0>(info.param)), {{":", ""}});
      const OHWI weights_shape = std::get<1>(info.param);
      const std::string weights_shape_str =
          absl::StrCat(weights_shape.o, "_", weights_shape.h, "_",
                       weights_shape.w, "_", weights_shape.i);
      return absl::StrCat(precision_name, "_", weights_shape_str, "_",
                          ToString(std::get<2>(info.param)));
    });

class OTileI2Test : public Test,
                    public WithParamInterface<std::tuple<OHWI, int, int>> {};

TEST_P(OTileI2Test, OSpatialIOGroupITileOTileI2) {
  auto [shape, i_tile_size, output_group_size] = GetParam();
  WeightsDescription conv_weight_desc = {
      .layout = WeightsLayout::kCustomGroups,
      .group_sizes = {{Axis::kInputChannels, 2},
                      {Axis::kOutputChannels, 4},
                      {Axis::kInputChannels, 2},
                      {Axis::kInputChannels, i_tile_size / 2},
                      {Axis::kOutputChannels, output_group_size},
                      {Axis::kInputChannels, 0},
                      {Axis::kWidth, 0},
                      {Axis::kHeight, 0},
                      {Axis::kOutputChannels, 0}},
  };
  ABSL_ASSERT_OK(ConverterToOSpatialIOGroupITileOTileIXTest(*exec_env, shape,
                                                       conv_weight_desc));
}

INSTANTIATE_TEST_SUITE_P(
    ConverterToConvWeightsOTileI2TestSuite, OTileI2Test,
    Combine(ValuesIn({OHWI(64, 1, 1, 64), OHWI(13, 1, 1, 27),
                      OHWI(111, 1, 1, 33), OHWI(13, 2, 3, 11),
                      OHWI(11, 3, 2, 23)}),
            ValuesIn({2, 4, 6, 8, 10}), ValuesIn({2, 4, 6})),
    [](const TestParamInfo<OTileI2Test::ParamType>& info) {
      const OHWI weights_shape = std::get<0>(info.param);
      const std::string weights_shape_str =
          absl::StrCat(weights_shape.o, "_", weights_shape.h, "_",
                       weights_shape.w, "_", weights_shape.i);
      return absl::StrReplaceAll(
          absl::StrCat(weights_shape_str, "_",
                       std::to_string(std::get<1>(info.param)), "_",
                       std::to_string(std::get<2>(info.param))),
          {{":", ""}});
    });

class OTileI4Test : public Test,
                    public WithParamInterface<std::tuple<OHWI, int, int, int>> {
};

TEST_P(OTileI4Test, OSpatialIOGroupITileOTileI4) {
  auto [shape, i_tile_size, o_tile_size, output_group_size] = GetParam();
  WeightsDescription conv_weight_desc = {
      .layout = WeightsLayout::kCustomGroups,
      .group_sizes = {{Axis::kInputChannels, 4},
                      {Axis::kOutputChannels, o_tile_size},
                      {Axis::kInputChannels, i_tile_size},
                      {Axis::kOutputChannels, output_group_size},
                      {Axis::kInputChannels, 0},
                      {Axis::kWidth, 0},
                      {Axis::kHeight, 0},
                      {Axis::kOutputChannels, 0}}};
  ABSL_ASSERT_OK(ConverterToOSpatialIOGroupITileOTileIXTest(*exec_env, shape,
                                                       conv_weight_desc));
}

INSTANTIATE_TEST_SUITE_P(
    ConverterToConvWeightsOTileI4TestSuite, OTileI4Test,
    Combine(ValuesIn({OHWI(64, 1, 1, 64), OHWI(13, 1, 1, 27),
                      OHWI(111, 1, 1, 33), OHWI(13, 2, 3, 11),
                      OHWI(11, 3, 2, 23)}),
            /*i_tile_size=*/ValuesIn({2, 4, 6, 8, 10}),
            /*o_tile_size=*/ValuesIn({4, 8, 12}),
            /*output_group_size=*/ValuesIn({2, 4, 6})),
    [](const TestParamInfo<OTileI4Test::ParamType>& info) {
      const OHWI weights_shape = std::get<0>(info.param);
      const std::string weights_shape_str =
          absl::StrCat(weights_shape.o, "_", weights_shape.h, "_",
                       weights_shape.w, "_", weights_shape.i);
      return absl::StrReplaceAll(
          absl::StrCat(weights_shape_str, "_",
                       std::to_string(std::get<1>(info.param)), "_",
                       std::to_string(std::get<2>(info.param)), "_",
                       std::to_string(std::get<3>(info.param))),
          {{":", ""}});
    });

class IntToFloatTest
    : public Test,
      public WithParamInterface<
          std::tuple<std::pair<OHWI, int>, WeightsLayout, WeightsDescription>> {
  // accessor methods
 protected:
  OHWI GetWeightsShape() const { return std::get<0>(GetParam()).first; }
  int GetSrcChQuantGroups() const { return std::get<0>(GetParam()).second; }
  WeightsLayout GetSrcLayout() const { return std::get<1>(GetParam()); }
  WeightsDescription GetDstWeightsDesc() const {
    return std::get<2>(GetParam());
  }
};

TEST_P(IntToFloatTest, Int8ToFloat) {
  WeightsDescription dst_weights_desc = GetDstWeightsDesc();
  ABSL_ASSERT_OK(Int8ToFloatWeightsConverterTest(*exec_env, GetWeightsShape(),
                                            GetSrcChQuantGroups(),
                                            GetSrcLayout(), dst_weights_desc));
}

TEST_P(IntToFloatTest, Int4ToFloat) {
  WeightsDescription dst_weights_desc = GetDstWeightsDesc();
  ABSL_ASSERT_OK(Int4ToFloatWeightsConverterTest(*exec_env, GetWeightsShape(),
                                            GetSrcChQuantGroups(),
                                            GetSrcLayout(), dst_weights_desc));
}

TEST_P(IntToFloatTest, Int2ToFloat) {
  WeightsDescription dst_weights_desc = GetDstWeightsDesc();
  ABSL_ASSERT_OK(Int2ToFloatWeightsConverterTest(*exec_env, GetWeightsShape(),
                                            GetSrcChQuantGroups(),
                                            GetSrcLayout(), dst_weights_desc));
}

INSTANTIATE_TEST_SUITE_P(
    ConverterToConvWeightsIntToFloatTestSuite, IntToFloatTest,
    Combine(ValuesIn({std::make_pair(OHWI(128, 1, 1, 128), 1),
                      std::make_pair(OHWI(124, 1, 1, 128), 1),
                      std::make_pair(OHWI(128, 1, 1, 124), 1),
                      std::make_pair(OHWI(124, 1, 1, 124), 1),
                      std::make_pair(OHWI(132, 1, 1, 4 * 20), 4),
                      std::make_pair(OHWI(132, 1, 1, 5 * 20), 5),
                      std::make_pair(OHWI(132, 4, 1, 92), 1)}),
            ValuesIn({WeightsLayout::kOSpatialIOGroupI4O4,
                      WeightsLayout::kOSpatialIOGroupO4I4,
                      WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4}),
            ValuesIn(WeightsDescsIntToFloat())),
    [](const TestParamInfo<IntToFloatTest::ParamType>& info) {
      const OHWI weights_shape = std::get<0>(info.param).first;
      const std::string weights_shape_str =
          absl::StrCat(weights_shape.o, "_", weights_shape.h, "_",
                       weights_shape.w, "_", weights_shape.i);
      return absl::StrReplaceAll(
          absl::StrCat(weights_shape_str, "_", std::get<0>(info.param).second,
                       "_", ToString(std::get<1>(info.param)), "_",
                       ToString(std::get<2>(info.param))),
          {{":", ""}});
    });

TEST(IntToFloatRuntimeTest, Int8ToFloatRuntimeInput) {
  ABSL_ASSERT_OK(Int8ToFloatWeightsWithRuntimeInputTest(*exec_env));
}

TEST(IntToFloatRuntimeTest, Int8ToFloatRuntimeOutput) {
  ABSL_ASSERT_OK(Int8ToFloatWeightsWithRuntimeOutputTest(*exec_env));
}

TEST(FloatWeightsRingedTest, RingedOFloat32) {
  ABSL_ASSERT_OK(FloatWeightsWithRingedOTest(*exec_env, DataType::kFloat32));
}

TEST(FloatWeightsRingedTest, RingedOFloat16) {
  if (!exec_env->IsStorageSupported(TensorStorageType::kBuffer,
                                    DataType::kFloat16)) {
    GTEST_SKIP() << "Unsupported data type: FLOAT16";
  }
  ABSL_ASSERT_OK(FloatWeightsWithRingedOTest(*exec_env, DataType::kFloat16));
}

TEST(FloatWeightsRingedTest, RingedIFloat32) {
  ABSL_ASSERT_OK(FloatWeightsWithRingedITest(*exec_env, DataType::kFloat32));
}

TEST(FloatWeightsRingedTest, RingedIFloat16) {
  if (!exec_env->IsStorageSupported(TensorStorageType::kBuffer,
                                    DataType::kFloat16)) {
    GTEST_SKIP() << "Unsupported data type: FLOAT16";
  }
  ABSL_ASSERT_OK(FloatWeightsWithRingedITest(*exec_env, DataType::kFloat16));
}

class I8ToUI8Test
    : public Test,
      public WithParamInterface<
          std::tuple<WeightsLayout, WeightsDescription, int, int>> {};

TEST_P(I8ToUI8Test, I8ToUI8) {
  WeightsDescription dst_weights_desc = std::get<1>(GetParam());
  ABSL_ASSERT_OK(Int8ToUint8WeightsConverterTest(
      *exec_env, std::get<0>(GetParam()), dst_weights_desc,
      /*i_channels=*/std::get<2>(GetParam()),
      /*o_channels=*/std::get<3>(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(
    ConverterToConvWeightsI8ToUI8TestSuite, I8ToUI8Test,
    Combine(ValuesIn({WeightsLayout::kOSpatialIOGroupO4I4,
                      WeightsLayout::kOSpatialIOGroupI4O4,
                      WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4}),
            ValuesIn(WeightsDescsI8ToUI8()),

            // For input channels:
            // * Channel 7 is to test the case that has paddings in the slice
            //     dimension.
            // * Channel 8 is to test the case that has no paddings.
            ValuesIn({7, 8, 128 - 4}), /* i_channels */

            // For output channels (assume destination output group size is 16):
            // * Channel 7 is to test the case that has paddings in the slice
            //     dimension.
            // * Channel (dst_output_group_size + 1) * 4 = 68 is to test the
            //     case that has paddings in the OGroup dimension.
            // * Channel dst_output_group_size * 2 * 4 = 128 is to test the case
            //     that has no paddings.
            ValuesIn({7, 68, 128, 128 + 4})), /* o_channels */
    [](const TestParamInfo<I8ToUI8Test::ParamType>& info) {
      return absl::StrReplaceAll(
          absl::StrCat(ToString(std::get<0>(info.param)), "_",
                       ToString(std::get<1>(info.param)), "_", "i_channels",
                       std::get<2>(info.param), "_", "o_channels",
                       std::get<3>(info.param)),
          {{":", ""}});
    });

class FloatToFloatTest
    : public Test,
      public WithParamInterface<
          std::tuple<OHWI, WeightsLayout, WeightsDescription>> {};

TEST_P(FloatToFloatTest, F2F) {
  WeightsDescription dst_weights_desc = std::get<2>(GetParam());
  ABSL_ASSERT_OK(FloatToFloatWeightsConverterTest(*exec_env, std::get<0>(GetParam()),
                                             std::get<1>(GetParam()),
                                             dst_weights_desc));
}

INSTANTIATE_TEST_SUITE_P(
    ConverterToConvWeightsF32ToF32TestSuite, FloatToFloatTest,
    Combine(ValuesIn({OHWI(132, 1, 1, 124), OHWI(108, 3, 2, 140)}),
            ValuesIn({WeightsLayout::kOSpatialIOGroupI4O4,
                      WeightsLayout::kOSpatialIOGroupO4I4}),
            ValuesIn(WeightsDescsF32ToF32())),
    [](const TestParamInfo<FloatToFloatTest::ParamType>& info) {
      const OHWI weights_shape = std::get<0>(info.param);
      const std::string weights_shape_str =
          absl::StrCat(weights_shape.o, "_", weights_shape.h, "_",
                       weights_shape.w, "_", weights_shape.i);
      return absl::StrReplaceAll(
          absl::StrCat(weights_shape_str, "_",
                       ToString(std::get<1>(info.param)), "_",
                       ToString(std::get<2>(info.param))),
          {{":", ""}});
    });

class UITo8BitTest
    : public Test,
      public WithParamInterface<std::tuple<WeightsLayout, WeightsDescription>> {
 protected:
  WeightsLayout GetSrcLayout() const { return std::get<0>(GetParam()); }
};

TEST_P(UITo8BitTest, UI8ToI8) {
  WeightsDescription dst_weights_desc = std::get<1>(GetParam());
  dst_weights_desc.type = DataType::kInt8;
  ABSL_ASSERT_OK(Uint8ToInt8WeightsConverterTest(*exec_env, GetSrcLayout(),
                                            dst_weights_desc));
}

TEST_P(UITo8BitTest, UI4ToI8) {
  WeightsDescription dst_weights_desc = std::get<1>(GetParam());
  dst_weights_desc.type = DataType::kInt8;
  ABSL_ASSERT_OK(Uint4ToInt8WeightsConverterTest(*exec_env, GetSrcLayout(),
                                            dst_weights_desc));
}

TEST_P(UITo8BitTest, UI2ToI8) {
  WeightsDescription dst_weights_desc = std::get<1>(GetParam());
  dst_weights_desc.type = DataType::kInt8;
  ABSL_ASSERT_OK(Uint2ToInt8WeightsConverterTest(*exec_env, GetSrcLayout(),
                                            dst_weights_desc));
}

TEST_P(UITo8BitTest, UI8ToUI8) {
  WeightsDescription dst_weights_desc = std::get<1>(GetParam());
  dst_weights_desc.type = DataType::kUint8;
  ABSL_ASSERT_OK(Uint8ToUint8WeightsConverterTest(*exec_env, GetSrcLayout(),
                                             dst_weights_desc));
}

TEST_P(UITo8BitTest, UI4ToUI8) {
  WeightsDescription dst_weights_desc = std::get<1>(GetParam());
  dst_weights_desc.type = DataType::kUint8;
  ABSL_ASSERT_OK(Uint4ToUint8WeightsConverterTest(*exec_env, GetSrcLayout(),
                                             dst_weights_desc));
}

TEST_P(UITo8BitTest, UI2ToUI8) {
  WeightsDescription dst_weights_desc = std::get<1>(GetParam());
  dst_weights_desc.type = DataType::kUint8;
  ABSL_ASSERT_OK(Uint2ToUint8WeightsConverterTest(*exec_env, GetSrcLayout(),
                                             dst_weights_desc));
}

INSTANTIATE_TEST_SUITE_P(
    ConverterToConvWeightsUTo8bitTestSuite, UITo8BitTest,
    Combine(ValuesIn({WeightsLayout::kOSpatialIOGroupI4O4,
                      WeightsLayout::kOSpatialIOGroupO4I4,
                      WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4}),
            ValuesIn(WeightsDescsUITo8Bit())),
    [](const TestParamInfo<UITo8BitTest::ParamType>& info) {
      return absl::StrReplaceAll(
          absl::StrCat(ToString(std::get<0>(info.param)), "_",
                       ToString(std::get<1>(info.param))),
          {{":", ""}});
    });

class UITo4BitTest
    : public Test,
      public WithParamInterface<std::tuple<WeightsLayout, WeightsDescription>> {
 protected:
  WeightsLayout GetSrcLayout() const { return std::get<0>(GetParam()); }
};

TEST_P(UITo4BitTest, UI4ToI4) {
  WeightsDescription dst_weights_desc = std::get<1>(GetParam());
  dst_weights_desc.type = DataType::kInt4;
  ABSL_ASSERT_OK(Uint4ToInt4WeightsConverterTest(*exec_env, GetSrcLayout(),
                                            dst_weights_desc));
}

TEST_P(UITo4BitTest, UI2ToI4) {
  WeightsDescription dst_weights_desc = std::get<1>(GetParam());
  dst_weights_desc.type = DataType::kInt4;
  ABSL_ASSERT_OK(Uint2ToInt4WeightsConverterTest(*exec_env, GetSrcLayout(),
                                            dst_weights_desc));
}

INSTANTIATE_TEST_SUITE_P(
    ConverterToConvWeightsUTo4bitTestSuite, UITo4BitTest,
    Combine(ValuesIn({WeightsLayout::kOSpatialIOGroupI4O4,
                      WeightsLayout::kOSpatialIOGroupO4I4,
                      WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4}),
            ValuesIn(WeightsDescsUITo4Bit())),
    [](const TestParamInfo<UITo4BitTest::ParamType>& info) {
      return absl::StrReplaceAll(
          absl::StrCat(ToString(std::get<0>(info.param)), "_",
                       ToString(std::get<1>(info.param))),
          {{":", ""}});
    });

class OHWIToCustomGroupsTest
    : public Test,
      public WithParamInterface<
          std::tuple<OHWI, TensorStorageType, DataType, WeightsDescription>> {};

TEST_P(OHWIToCustomGroupsTest, FromOHWI) {
  auto [weights_shape, storage, src_type, dst_desc] = GetParam();
  if (!exec_env->IsStorageSupported(storage, src_type) ||
      !exec_env->IsStorageSupported(TensorStorageType::kBuffer,
                                    dst_desc.type)) {
    GTEST_SKIP() << "Unsupported storage type: " << ToString(storage);
  }
  ABSL_ASSERT_OK(ConverterToCustomGroupsTest(*exec_env, weights_shape, storage,
                                        src_type, dst_desc));
}

INSTANTIATE_TEST_SUITE_P(
    OHWIToCustomGroupsTestSuite, OHWIToCustomGroupsTest,
    Combine(ValuesIn({OHWI(3, 1, 1, 3), OHWI(13, 1, 1, 11), OHWI(49, 1, 1, 37),
                      OHWI(79, 1, 1, 82)}),
            ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D()),
            ValuesIn({DataType::kFloat32}),
            ValuesIn(WeightsDescsCustomGroups())),
    [](const TestParamInfo<OHWIToCustomGroupsTest::ParamType>& info) {
      const OHWI weights_shape = std::get<0>(info.param);
      const std::string weights_shape_str =
          absl::StrCat(weights_shape.o, "_", weights_shape.h, "_",
                       weights_shape.w, "_", weights_shape.i);
      return absl::StrReplaceAll(
          absl::StrCat(ToString(std::get<1>(info.param)), "_",
                       weights_shape_str, "_",
                       ToString(std::get<2>(info.param)), "_to_",
                       ToString(std::get<3>(info.param))),
          {{":", ""}});
    });

INSTANTIATE_TEST_SUITE_P(
    OHWIToCustomGroupsOIorIOTestSuite, OHWIToCustomGroupsTest,
    Combine(ValuesIn({OHWI(4, 1, 1, 4), OHWI(16, 1, 1, 12), OHWI(48, 1, 1, 36),
                      OHWI(76, 1, 1, 84)}),
            ValuesIn(GetTensorStoragesTypesWithoutSingleTexture2D()),
            ValuesIn({DataType::kFloat32}),
            ValuesIn(WeightsDescsCustomGroupsOIorIO())),
    [](const TestParamInfo<OHWIToCustomGroupsTest::ParamType>& info) {
      const OHWI weights_shape = std::get<0>(info.param);
      const std::string weights_shape_str =
          absl::StrCat(weights_shape.o, "_", weights_shape.h, "_",
                       weights_shape.w, "_", weights_shape.i);
      return absl::StrReplaceAll(
          absl::StrCat(ToString(std::get<1>(info.param)), "_",
                       weights_shape_str, "_",
                       ToString(std::get<2>(info.param)), "_to_",
                       ToString(std::get<3>(info.param))),
          {{":", ""}});
    });

}  // namespace ml_drift
