// Copyright 2024 The ML Drift Authors.
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

#include "ml_drift/common/kernels/tests/lstm_test_util.h"

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/lstm.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_ref_ops.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace {
absl::Status LSTMTest(TestExecutionEnvironment& exec_env,
                      const TensorFloat32& prev_state,
                      const TensorFloat32& intermediate_tensor,
                      const OperationDef& op_def) {
  std::vector<TensorFloat32> dsts_ref =
      LSTMReference(intermediate_tensor, prev_state);

  auto operation = CreateLSTM(op_def, exec_env.GetGpuInfo());

  std::vector<TensorFloat32> dsts(2);
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      {intermediate_tensor, prev_state},
      std::make_unique<GPUOperation>(std::move(operation)),
      {dsts_ref[0].shape, dsts_ref[1].shape}, {&dsts[0], &dsts[1]}));
  EXPECT_THAT(dsts[0].data, Pointwise(FloatNear(0.005), dsts_ref[0].data));
  EXPECT_THAT(dsts[1].data, Pointwise(FloatNear(0.005), dsts_ref[1].data));
  return absl::OkStatus();
}

}  // namespace

absl::Status LstmTest(TestExecutionEnvironment& env,
                      CalculationsPrecision precision,
                      TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 16);
  src_tensor.data = {
      -std::log(2.0f), -std::log(2.0f), -std::log(2.0f), -std::log(2.0f),
      std::log(3.0f),  std::log(3.0f),  std::log(3.0f),  std::log(3.0f),
      -std::log(4.0f), -std::log(4.0f), -std::log(4.0f), -std::log(4.0f),
      -std::log(5.0f), -std::log(5.0f), -std::log(5.0f), -std::log(5.0f)};
  // input_gate = 1.0 / (1.0 + exp(log(2.0f))) = 1.0 / 3.0;
  // new_input = tanh(log(3.0f)) = (exp(2 * log(3.0f)) - 1) / exp(2 * log(3.0f))
  // + 1 = (9 - 1) / (9 + 1) = 0.8;
  // forget_gate = 1.0 / (1.0 + exp(log(4.0f)))
  //  = 1.0 / 5.0;
  // output_gate = 1.0 / (1.0 + exp(log(5.0f))) = 1.0 / 6.0;
  // new_st = input_gate * new_input + forget_gate * prev_st
  //   = 1.0 / 3.0 * 0.8 + 1.0 / 5.0 * prev_st
  //   = 4.0 / 15.0 + 3.0 / 15.0 = 7.0 / 15.0
  // activation = output_gate * tanh(new_st)
  TensorFloat32 prev_state;
  prev_state.shape = BHWC(1, 1, 1, 4);
  prev_state.data = {1.0f, 2.0f, 3.0f, 4.0f};

  const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 2e-3f;
  OperationDef op_def;
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  TensorFloat32 new_state;
  TensorFloat32 new_activ;
  GPUOperation operation = CreateLSTM(op_def, env.GetGpuInfo());
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor, prev_state},
      std::make_unique<GPUOperation>(std::move(operation)),
      {BHWC(1, 1, 1, 4), BHWC(1, 1, 1, 4)}, {&new_state, &new_activ}));
  EXPECT_THAT(new_state.data,
              Pointwise(FloatNear(eps),
                        {7.0 / 15.0, 10.0 / 15.0, 13.0 / 15.0, 16.0 / 15.0}))
      ;
  TensorFloat32 expected_output;
  expected_output.data = {
      static_cast<float>((1.0 / 6.0) * std::tanh(7.0 / 15.0)),
      static_cast<float>((1.0 / 6.0) * std::tanh(10.0 / 15.0)),
      static_cast<float>((1.0 / 6.0) * std::tanh(13.0 / 15.0)),
      static_cast<float>((1.0 / 6.0) * std::tanh(16.0 / 15.0))};
  EXPECT_THAT(new_activ.data, Pointwise(FloatNear(eps), expected_output.data));
  return absl::OkStatus();
}

absl::Status LstmBigTest(TestExecutionEnvironment& env,
                         CalculationsPrecision precision,
                         TensorStorageType storage) {
  const DataType data_type = DeduceDataTypeFromPrecision(precision);
  const int state_size = 96;
  const int BATCH_SIZE = 5;
  BHWC state_shape(BATCH_SIZE, 1, 1, state_size);
  BHWC src_shape(BATCH_SIZE, 1, 1, state_size * 4);

  TensorFloat32 prev_state = MakeSyntheticTensor(state_shape);
  TensorFloat32 intermediate_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  return LSTMTest(env, prev_state, intermediate_tensor, op_def);
}

}  // namespace ml_drift
