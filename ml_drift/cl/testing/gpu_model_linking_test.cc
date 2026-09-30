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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/status.h"
#include "ml_drift/cl/testing/cl_test.h"
#include "ml_drift/common/gpu_model_linking_test_util.h"

namespace ml_drift {
namespace cl {
namespace {

TEST_F(OpenCLOperationTest, LinkingConvolutionAndCosOp) {
  ABSL_ASSERT_OK(TestLinkingConvolutionAndCosOp(&exec_env_));
}

TEST_F(OpenCLOperationTest, LinkingConvolution2InputMul2InputMul) {
  ABSL_ASSERT_OK(TestLinkingConvolution2InputMul2InputMul(&exec_env_));
}

TEST_F(OpenCLOperationTest, LinkingConvolution2InputBroadcastMul2InputMul) {
  ABSL_ASSERT_OK(TestLinkingConvolution2InputBroadcastMul2InputMul(&exec_env_));
}

TEST_F(OpenCLOperationTest, LinkingConvolution2InputMul2InputBroadcastMul) {
  ABSL_ASSERT_OK(TestLinkingConvolution2InputMul2InputBroadcastMul(&exec_env_));
}

TEST_F(OpenCLOperationTest, LinkingConvolution2InputMul2InputMulCos) {
  ABSL_ASSERT_OK(TestLinkingConvolution2InputMul2InputMulCos(&exec_env_));
}

TEST_F(OpenCLOperationTest, LinkingConvolutionFirstTanh2InputDiff) {
  ABSL_ASSERT_OK(TestLinkingConvolutionFirstTanh2InputDiff(&exec_env_));
}

TEST_F(OpenCLOperationTest, LinkingConvolutionSecondTanh2InputDiff) {
  ABSL_ASSERT_OK(TestLinkingConvolutionSecondTanh2InputDiff(&exec_env_));
}

TEST_F(OpenCLOperationTest, LinkingConvolutionFirstTanhSecondCos2InputDiff) {
  ABSL_ASSERT_OK(TestLinkingConvolutionFirstTanhSecondCos2InputDiff(&exec_env_));
}

TEST_F(OpenCLOperationTest, LinkingComplex0) {
  ABSL_ASSERT_OK(TestLinkingComplex0(&exec_env_));
}

TEST_F(OpenCLOperationTest, LinkingConvElem2InputAddElemsOp) {
  ABSL_ASSERT_OK(TestLinkingConvElem2InputAddElemsOp(&exec_env_));
}

TEST_F(OpenCLOperationTest, LinkingSliceCastOp) {
  ABSL_ASSERT_OK(TestLinkingSliceCastOp(&exec_env_));
}

TEST_F(OpenCLOperationTest, LinkingAddAddMulOp) {
  ABSL_ASSERT_OK(
      TestLinkingAddAddMulOp(&exec_env_, /*use_second_input_add=*/true));
}

TEST_F(OpenCLOperationTest, LinkingAddMulOp) {
  ABSL_ASSERT_OK(
      TestLinkingAddAddMulOp(&exec_env_, /*use_second_input_add=*/false));
}

TEST_F(OpenCLOperationTest, LinkingConcatAndCosOp) {
  ABSL_ASSERT_OK(TestLinkingConcatAndCosOp(&exec_env_));
}

TEST_F(OpenCLOperationTest, LinkingCosAndCosOp) {
  ABSL_ASSERT_OK(TestLinkingCosAndCosOp(&exec_env_));
}

TEST_F(OpenCLOperationTest, FloatCastToBoolCastToFloat) {
  ABSL_ASSERT_OK(TestFloatCastToBoolCastToFloat(&exec_env_));
}

TEST_F(OpenCLOperationTest, ReshapeTranspose) {
  ABSL_ASSERT_OK(TestReshapeTranspose(&exec_env_));
}

TEST_F(OpenCLOperationTest, ReshapeTransposeReshape) {
  ABSL_ASSERT_OK(TestReshapeTransposeReshape(&exec_env_));
}

TEST_F(OpenCLOperationTest, TwoInputTwise) {
  ABSL_ASSERT_OK(TestTwoInputTwise(&exec_env_));
}

TEST_F(OpenCLOperationTest, ConvWithPaddedAdd) {
  ABSL_ASSERT_OK(TestConvWithPaddedAdd(&exec_env_));
}

}  // namespace
}  // namespace cl
}  // namespace ml_drift
