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

#ifndef ML_DRIFT_COMMON_KERNELS_TESTS_ELEMENTWISE_TEST_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_TESTS_ELEMENTWISE_TEST_UTIL_H_

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"

namespace ml_drift {

absl::Status AbsTest(TestExecutionEnvironment& env,
                     DataType data_type,
                     TensorStorageType storage);
absl::Status CosTest(TestExecutionEnvironment& env,
                     DataType data_type,
                     TensorStorageType storage);
absl::Status CosIntTest(TestExecutionEnvironment& env,
                        TensorStorageType storage);
absl::Status CopyTest(TestExecutionEnvironment& env,
                      DataType data_type,
                      TensorStorageType storage);
absl::Status EluTest(TestExecutionEnvironment& env,
                     DataType data_type,
                     TensorStorageType storage);
absl::Status ExpTest(TestExecutionEnvironment& env,
                     DataType data_type,
                     TensorStorageType storage);
absl::Status FloorTest(TestExecutionEnvironment& env,
                       DataType data_type,
                       TensorStorageType storage);
absl::Status FloorDivTest(TestExecutionEnvironment& env,
                          DataType data_type,
                          TensorStorageType storage);
absl::Status FloorDivIntTest(TestExecutionEnvironment& env,
                             TensorStorageType storage);
absl::Status FloorModTest(TestExecutionEnvironment& env,
                          DataType data_type,
                          TensorStorageType storage);
absl::Status FloorModIntTest(TestExecutionEnvironment& env,
                             TensorStorageType storage);
absl::Status GeluTest(TestExecutionEnvironment& env,
                      DataType data_type,
                      TensorStorageType storage);
absl::Status HardSwishTest(TestExecutionEnvironment& env,
                           DataType data_type,
                           TensorStorageType storage);
absl::Status LogTest(TestExecutionEnvironment& env,
                     DataType data_type,
                     TensorStorageType storage);
absl::Status NegTest(TestExecutionEnvironment& env,
                     DataType data_type,
                     TensorStorageType storage);
absl::Status RoundTest(TestExecutionEnvironment& env,
                       DataType data_type,
                       TensorStorageType storage);
absl::Status RsqrtTest(TestExecutionEnvironment& env,
                       DataType data_type,
                       TensorStorageType storage);
absl::Status SigmoidTest(TestExecutionEnvironment& env,
                         DataType data_type,
                         TensorStorageType storage);
absl::Status SignTest(TestExecutionEnvironment& env,
                      DataType data_type,
                      TensorStorageType storage);

absl::Status SignInt8Test(TestExecutionEnvironment& env,
                          TensorStorageType storage);
absl::Status SinTest(TestExecutionEnvironment& env,
                     DataType data_type,
                     TensorStorageType storage);
absl::Status SqrtTest(TestExecutionEnvironment& env,
                      DataType data_type,
                      TensorStorageType storage);
absl::Status SquareTest(TestExecutionEnvironment& env,
                        DataType data_type,
                        TensorStorageType storage);
absl::Status TanhTest(TestExecutionEnvironment& env,
                      DataType data_type,
                      TensorStorageType storage);
absl::Status SubTest(TestExecutionEnvironment& env,
                     DataType data_type,
                     TensorStorageType storage);
absl::Status ShiftLeftTest(TestExecutionEnvironment& env,
                           TensorStorageType storage);
absl::Status ShiftRightTest(TestExecutionEnvironment& env,
                            TensorStorageType storage);
absl::Status SquaredDiffTest(TestExecutionEnvironment& env,
                             DataType data_type,
                             TensorStorageType storage);
absl::Status DivTest(TestExecutionEnvironment& env,
                     DataType data_type,
                     TensorStorageType storage);
absl::Status ModInt3Test(TestExecutionEnvironment& env,
                         TensorStorageType storage);
absl::Status ModUint5Test(TestExecutionEnvironment& env,
                          TensorStorageType storage);
absl::Status PowTest(TestExecutionEnvironment& env,
                     DataType data_type,
                     TensorStorageType storage);
absl::Status PowNegativeBaseTest(TestExecutionEnvironment& env,
                                 DataType data_type,
                                 TensorStorageType storage);
absl::Status PowNegativeBaseScalarTest(TestExecutionEnvironment& env,
                                       DataType data_type,
                                       TensorStorageType storage);
absl::Status AddTest(TestExecutionEnvironment& env,
                     DataType data_type,
                     TensorStorageType storage);
absl::Status AddWithConstantBHWCTensorTest(TestExecutionEnvironment& env,
                                           DataType data_type,
                                           TensorStorageType storage);
absl::Status AddTiledTest(TestExecutionEnvironment& env,
                          DataType data_type,
                          TensorStorageType storage);
absl::Status Atan2Test(TestExecutionEnvironment& env,
                       DataType data_type,
                       TensorStorageType storage);
absl::Status Atan2IntTest(TestExecutionEnvironment& env,
                          TensorStorageType storage);
absl::Status MaximumTest(TestExecutionEnvironment& env,
                         DataType data_type,
                         TensorStorageType storage);
absl::Status MaximumInt8Test(TestExecutionEnvironment& env,
                             TensorStorageType storage);
absl::Status MaximumWithScalarTest(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage);
absl::Status MaximumWithIntScalarTest(TestExecutionEnvironment& env,
                                      TensorStorageType storage);
absl::Status MaximumWithUintScalarTest(TestExecutionEnvironment& env,
                                       TensorStorageType storage);
absl::Status MaximumWithConstantLinearTensorTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage);
absl::Status MaximumWithConstantBHWCTensorTest(TestExecutionEnvironment& env,
                                               DataType data_type,
                                               TensorStorageType storage);
absl::Status MaximumWithConstantBHWCTensorBroadcastChannelsTest(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage);
absl::Status MinimumTest(TestExecutionEnvironment& env,
                         DataType data_type,
                         TensorStorageType storage);
absl::Status MinimumInt8Test(TestExecutionEnvironment& env,
                             TensorStorageType storage);
absl::Status MinimumWithScalarTest(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage);
absl::Status MulTest(TestExecutionEnvironment& env,
                     DataType data_type,
                     TensorStorageType storage);
absl::Status MulBroadcastHWTest(TestExecutionEnvironment& env,
                                DataType data_type,
                                TensorStorageType storage);
absl::Status MulBroadcastChannelsTest(TestExecutionEnvironment& env,
                                      DataType data_type,
                                      TensorStorageType storage);
absl::Status SubWithScalarAtFirstPositionTest(TestExecutionEnvironment& env,
                                              DataType data_type,
                                              TensorStorageType storage);
absl::Status LessTest(TestExecutionEnvironment& env, TensorStorageType storage);
absl::Status LessEqualTest(TestExecutionEnvironment& env,
                           TensorStorageType storage);
absl::Status GreaterTest(TestExecutionEnvironment& env,
                         TensorStorageType storage);
absl::Status GreaterEqualTest(TestExecutionEnvironment& env,
                              TensorStorageType storage);
absl::Status EqualTest(TestExecutionEnvironment& env,
                       TensorStorageType storage);
absl::Status NotEqualTest(TestExecutionEnvironment& env,
                          TensorStorageType storage);
absl::Status CosBroadcastTest(TestExecutionEnvironment& env,
                              DataType data_type,
                              TensorStorageType storage);
absl::Status MaximumScalarBroadcastInputTest(TestExecutionEnvironment& env,
                                             DataType data_type,
                                             TensorStorageType storage);
absl::Status MulLinearBroadcastInputTest(TestExecutionEnvironment& env,
                                         DataType data_type,
                                         TensorStorageType storage);
absl::Status MulBroadcastBothInputsTest(TestExecutionEnvironment& env,
                                        DataType data_type,
                                        TensorStorageType storage);
absl::Status MishTest(TestExecutionEnvironment& env,
                      DataType data_type,
                      TensorStorageType storage);

absl::Status LogicalAndTest(TestExecutionEnvironment& env,
                            TensorStorageType storage);
absl::Status LogicalAndInt8Test(TestExecutionEnvironment& env,
                                TensorStorageType storage);
absl::Status LogicalOrTest(TestExecutionEnvironment& env,
                           TensorStorageType storage);
absl::Status LogicalOrInt8Test(TestExecutionEnvironment& env,
                               TensorStorageType storage);
absl::Status LogicalNotTest(TestExecutionEnvironment& env,
                            TensorStorageType storage);
absl::Status LogicalNotInt8Test(TestExecutionEnvironment& env,
                                TensorStorageType storage);
absl::Status LogicalXorTest(TestExecutionEnvironment& env,
                            TensorStorageType storage);
absl::Status LogicalXorInt8Test(TestExecutionEnvironment& env,
                                TensorStorageType storage);

absl::Status Add5DTest(TestExecutionEnvironment& env, DataType data_type,
                       TensorStorageType storage);

absl::Status OneInputWithBroadcast5DTest(TestExecutionEnvironment& env,
                                         DataType data_type,
                                         TensorStorageType storage);

absl::Status WithBroadcast5DTest(TestExecutionEnvironment& env,
                                 DataType data_type, TensorStorageType storage);
absl::Status WithBroadcast5DPaddedGridTest(TestExecutionEnvironment& env,
                                           DataType data_type,
                                           TensorStorageType storage);
absl::Status TwoInputWithBroadcast5DTest(TestExecutionEnvironment& env,
                                         DataType data_type,
                                         TensorStorageType storage);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_TESTS_ELEMENTWISE_TEST_UTIL_H_
