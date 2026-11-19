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

#ifndef ML_DRIFT_COMMON_GPU_MODEL_LINKING_TEST_UTIL_H_
#define ML_DRIFT_COMMON_GPU_MODEL_LINKING_TEST_UTIL_H_

#include "absl/status/status.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

//    input
//      |
// convolution
//      |
//   cosinus
//      |
//   output
absl::Status TestLinkingConvolutionAndCosOp(TestExecutionEnvironment* env);

//      input0
//        |
//   convolution   input1
//          \        /
//        multiplication0    input2
//                 \         /
//               multiplication1
//                      |
//                    output
absl::Status TestLinkingConvolution2InputMul2InputMul(
    TestExecutionEnvironment* env);

//      input0(32x32x128)
//        |
//   convolution       input1(32x32x1)
//        |                /
//   conv_out(32x32x16)   /
//          \            /
//          broadcast_mul   input2(32x32x16)
//                |            /
//       mul_out(32x32x16)    /
//                 \         /
//               multiplication
//                      |
//                    output(32x32x16)
absl::Status TestLinkingConvolution2InputBroadcastMul2InputMul(
    TestExecutionEnvironment* env);

//      input0(32x32x128)
//        |
//   convolution       input1(32x32x16)
//        |                /
//   conv_out(32x32x16)   /
//          \            /
//          multiplication   input2(1x1x16)
//                |            /
//       mul_out(32x32x16)    /
//                 \         /
//               broadcast_mul
//                      |
//                    output(32x32x16)
absl::Status TestLinkingConvolution2InputMul2InputBroadcastMul(
    TestExecutionEnvironment* env);

//      input0
//        |
//   convolution   input1
//          \        /
//        multiplication0    input2
//                 \         /
//               multiplication1
//                      |
//                   cosinus
//                      |
//                   output
absl::Status TestLinkingConvolution2InputMul2InputMulCos(
    TestExecutionEnvironment* env);

//      input
//        |
//   convolution
//     /     \
//   tanh     |
//     \     /
//  substraction
//        |
//     output
absl::Status TestLinkingConvolutionFirstTanh2InputDiff(
    TestExecutionEnvironment* env);

//      input
//        |
//   convolution
//     /     \
//    |     tanh
//     \     /
//  substraction
//        |
//     output
absl::Status TestLinkingConvolutionSecondTanh2InputDiff(
    TestExecutionEnvironment* env);

//      input
//        |
//   convolution
//     /     \
//   tanh    cos
//     \     /
//  substraction
//        |
//     output
absl::Status TestLinkingConvolutionFirstTanhSecondCos2InputDiff(
    TestExecutionEnvironment* env);

//      input
//        |
//   convolution
//      /    \
//   tanh    cos
//    /     /   \
//   |    prelu  sin
//   |      \   /
//   |       pow
//   |        |
//   |       exp
//    \       |
//  substraction
//        |
//     output
absl::Status TestLinkingComplex0(TestExecutionEnvironment* env);

//                input1
//                  |
//              convolution
//                  |
//         input0  cos
//             \   /
//              add
//               |
//              cos
//               |
//              sin
//               |
//              abs
//               |
//             output
absl::Status TestLinkingConvElem2InputAddElemsOp(TestExecutionEnvironment* env);

//     input1
//       |
//     slice
//       |
//      cast
//       |
//     output
absl::Status TestLinkingSliceCastOp(TestExecutionEnvironment* env);

//       input
//         |
//      Reshape
//       /   \
//     Add   Add (Optional)
//       \   /
//        Mul
//         |
//       output
absl::Status TestLinkingAddAddMulOp(TestExecutionEnvironment* env,
                                    bool use_second_input_add);

absl::Status TestLinkingConcatAndCosOp(TestExecutionEnvironment* env);

//    input
//      |
//   cosinus
//      |
//   cosinus
//      |
//   output
absl::Status TestLinkingCosAndCosOp(TestExecutionEnvironment* env);

//    float
//      |
//   cast_op
//      |
//    bool
//      |
//   cast_op
//      |
//    float
absl::Status TestFloatCastToBoolCastToFloat(TestExecutionEnvironment* env);

//   input_tensor
//        |
//     reshape
//        |
//  interm_tensor
//        |
//    transpose
//        |
//  output_tensor
absl::Status TestReshapeTranspose(TestExecutionEnvironment* env);

//   input_tensor
//        |
//     reshape
//        |
//  interm_tensor
//        |
//    transpose
//        |
//  interm_tensor
//        |
//    reshape
//        |
//  output_tensor
absl::Status TestReshapeTransposeReshape(TestExecutionEnvironment* env);

//   input_tensor
//        |
//     reshape
//     /     \
//   sin    cos
//     \     /
//  substraction
//     /     \
//   sin    cos
//     \     /
//  substraction
//        |
//  output_tensor
absl::Status TestTwoInputTwise(TestExecutionEnvironment* env);

//   input_tensor
//     /     \
//   conv    pad
//     \     /
//    addition
//        |
//  output_tensor
absl::Status TestConvWithPaddedAdd(TestExecutionEnvironment* env);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_GPU_MODEL_LINKING_TEST_UTIL_H_
