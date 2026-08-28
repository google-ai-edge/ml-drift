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

#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/status.h"
#include "ml_drift/cl/testing/cl_test.h"
#include "ml_drift/common/gpu_model_linking_test_util.h"

namespace ml_drift {
namespace cl {
namespace {

TEST_F(OpenCLOperationTest, LinkingConvolutionAndCosOp) {
  auto status = TestLinkingConvolutionAndCosOp(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, LinkingConvolution2InputMul2InputMul) {
  auto status = TestLinkingConvolution2InputMul2InputMul(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, LinkingConvolution2InputBroadcastMul2InputMul) {
  auto status = TestLinkingConvolution2InputBroadcastMul2InputMul(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, LinkingConvolution2InputMul2InputBroadcastMul) {
  auto status = TestLinkingConvolution2InputMul2InputBroadcastMul(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, LinkingConvolution2InputMul2InputMulCos) {
  auto status = TestLinkingConvolution2InputMul2InputMulCos(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, LinkingConvolutionFirstTanh2InputDiff) {
  auto status = TestLinkingConvolutionFirstTanh2InputDiff(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, LinkingConvolutionSecondTanh2InputDiff) {
  auto status = TestLinkingConvolutionSecondTanh2InputDiff(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, LinkingConvolutionFirstTanhSecondCos2InputDiff) {
  auto status = TestLinkingConvolutionFirstTanhSecondCos2InputDiff(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, LinkingComplex0) {
  auto status = TestLinkingComplex0(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, LinkingConvElem2InputAddElemsOp) {
  auto status = TestLinkingConvElem2InputAddElemsOp(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, LinkingSliceCastOp) {
  auto status = TestLinkingSliceCastOp(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, LinkingAddAddMulOp) {
  auto status = TestLinkingAddAddMulOp(&exec_env_,
                                       /*use_second_input_add=*/true);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, LinkingAddMulOp) {
  auto status =
      TestLinkingAddAddMulOp(&exec_env_, /*use_second_input_add=*/false);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, LinkingConcatAndCosOp) {
  auto status = TestLinkingConcatAndCosOp(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, LinkingCosAndCosOp) {
  auto status = TestLinkingCosAndCosOp(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, FloatCastToBoolCastToFloat) {
  auto status = TestFloatCastToBoolCastToFloat(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, ReshapeTranspose) {
  auto status = TestReshapeTranspose(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, ReshapeTransposeReshape) {
  auto status = TestReshapeTransposeReshape(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, TwoInputTwise) {
  auto status = TestTwoInputTwise(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST_F(OpenCLOperationTest, ConvWithPaddedAdd) {
  auto status = TestConvWithPaddedAdd(&exec_env_);
  ASSERT_TRUE(status.ok()) << status.message();
}

}  // namespace
}  // namespace cl
}  // namespace ml_drift
