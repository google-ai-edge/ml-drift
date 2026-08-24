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

#include "ml_drift/common/kernels/special/dw3x3_conv32to16_dw3x3_conv16to16_add_conv16to32.h"

#include <memory>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "absl/status/status.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_ref_ops.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

using ::testing::Combine;
using ::testing::FloatNear;
using ::testing::Pointwise;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

namespace {

absl::Status DW3x3Conv32To16DW3x3Conv16To16AddConv16To32Test(
    TestExecutionEnvironment* exec_env, const OperationDef& op_def) {
  DepthwiseConvolution2DAttributes dw_attr_0;
  dw_attr_0.padding.prepended = HW(1, 1);
  dw_attr_0.padding.appended = HW(1, 1);
  dw_attr_0.strides = HW(1, 1);
  dw_attr_0.dilations = HW(1, 1);
  dw_attr_0.weights = MakeSyntheticTensor(OHWI(1, 3, 3, 32));
  dw_attr_0.bias = MakeSyntheticTensor(Linear(32));

  DepthwiseConvolution2DAttributes dw_attr_1;
  dw_attr_1.padding.prepended = HW(1, 1);
  dw_attr_1.padding.appended = HW(1, 1);
  dw_attr_1.strides = HW(1, 1);
  dw_attr_1.dilations = HW(1, 1);
  dw_attr_1.weights = MakeSyntheticTensor(OHWI(1, 3, 3, 16));
  dw_attr_1.bias = MakeSyntheticTensor(Linear(16));

  Convolution2DAttributes conv32to16;
  conv32to16.padding.prepended = HW(0, 0);
  conv32to16.padding.appended = HW(0, 0);
  conv32to16.strides = HW(1, 1);
  conv32to16.dilations = HW(1, 1);
  conv32to16.weights = MakeSyntheticTensor(OHWI(16, 1, 1, 32));
  conv32to16.bias = MakeSyntheticTensor(Linear(16));

  Convolution2DAttributes conv16to16;
  conv16to16.padding.prepended = HW(0, 0);
  conv16to16.padding.appended = HW(0, 0);
  conv16to16.strides = HW(1, 1);
  conv16to16.dilations = HW(1, 1);
  conv16to16.weights = MakeSyntheticTensor(OHWI(16, 1, 1, 16));
  conv16to16.bias = MakeSyntheticTensor(Linear(16));

  Convolution2DAttributes conv16to32;
  conv16to32.padding.prepended = HW(0, 0);
  conv16to32.padding.appended = HW(0, 0);
  conv16to32.strides = HW(1, 1);
  conv16to32.dilations = HW(1, 1);
  conv16to32.weights = MakeSyntheticTensor(OHWI(32, 1, 1, 16));
  conv16to32.bias = MakeSyntheticTensor(Linear(32));

  PReLUAttributes prelu_0;
  prelu_0.alpha = MakeSyntheticTensor(Linear(16));

  PReLUAttributes prelu_1;
  prelu_1.alpha = MakeSyntheticTensor(Linear(32));

  auto src_shape_0 = BHWC(1, 14, 16, 32);
  auto src_shape_1 = BHWC(1, 14, 16, 16);
  TensorFloat32 src_tensor_0 = MakeSyntheticTensor(src_shape_0);
  TensorFloat32 src_tensor_1 = MakeSyntheticTensor(src_shape_1);

  TensorFloat32 interm_0 =
      DepthWiseConvolutionReference(dw_attr_0, src_tensor_0);
  TensorFloat32 interm_1 =
      DepthWiseConvolutionReference(dw_attr_1, src_tensor_1);
  TensorFloat32 interm_2 = ConvolutionReference(conv32to16, interm_0);
  TensorFloat32 interm_3 = ConvolutionReference(conv16to16, interm_1);
  TensorFloat32 interm_4 = AddTableReference({interm_2, interm_3});
  TensorFloat32 interm_5 = PReLUReference(prelu_0, interm_4);
  TensorFloat32 interm_6 = ConvolutionReference(conv16to32, interm_5);
  TensorFloat32 out_ref = PReLUReference(prelu_1, interm_6);

  auto operation = CreateDW3x3Conv32To16DW3x3Conv16To16AddConv16To32(
      op_def, dw_attr_0, conv32to16, dw_attr_1, conv16to16, prelu_0, conv16to32,
      prelu_1);

  TensorFloat32 dst_0;
  MLD_EXPECT_OK(exec_env->ExecuteGPUOperation(
      {src_tensor_0, src_tensor_1},
      std::make_unique<GPUOperation>(std::move(operation)), out_ref.shape,
      &dst_0));
  EXPECT_THAT(dst_0.data, Pointwise(FloatNear(0.1f), out_ref.data));
  return absl::OkStatus();
}

}  // namespace

TEST_P(DataTypeTest, DW3x3Conv32To16DW3x3Conv16To16AddConv16To32Test) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type(), storage(), Layout::HWC});
  op_def.src_tensors.push_back({data_type(), storage(), Layout::HWC});
  op_def.dst_tensors.push_back({data_type(), storage(), Layout::HWC});
  if (!op_def.src_tensors[0].SupportsZeroClamp(Axis::WIDTH,
                                               exec_env->GetGpuInfo()) ||
      !op_def.src_tensors[0].SupportsZeroClamp(Axis::HEIGHT,
                                               exec_env->GetGpuInfo())) {
    GTEST_SKIP() << "Source tensor does not support zero clamp.";
  }
  MLD_EXPECT_OK(DW3x3Conv32To16DW3x3Conv16To16AddConv16To32Test(exec_env, op_def));
}

INSTANTIATE_TEST_SUITE_P(
    Suite, DataTypeTest,
    Combine(ValuesIn(GetFloatTypes()), ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<DataTypeTest::ParamType>& info) {
      return ToString(info.param);
    });
}  // namespace ml_drift
