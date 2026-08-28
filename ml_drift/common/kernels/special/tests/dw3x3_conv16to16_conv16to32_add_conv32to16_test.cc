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

#include "ml_drift/common/kernels/special/dw3x3_conv16to16_conv16to32_add_conv32to16.h"

#include <memory>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
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

absl::Status DW3x3Conv16To16Conv16To32AddConv32To16Test0(
    TestExecutionEnvironment* exec_env, const OperationDef& op_def) {
  DepthwiseConvolution2DAttributes dw_attr;
  dw_attr.padding.prepended = HW(1, 1);
  dw_attr.padding.appended = HW(1, 1);
  dw_attr.strides = HW(1, 1);
  dw_attr.dilations = HW(1, 1);
  dw_attr.weights = MakeSyntheticTensor(OHWI(1, 3, 3, 16));
  dw_attr.bias = MakeSyntheticTensor(Linear(16));

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

  Convolution2DAttributes conv32to16;
  conv32to16.padding.prepended = HW(0, 0);
  conv32to16.padding.appended = HW(0, 0);
  conv32to16.strides = HW(1, 1);
  conv32to16.dilations = HW(1, 1);
  conv32to16.weights = MakeSyntheticTensor(OHWI(16, 1, 1, 32));
  conv32to16.bias = MakeSyntheticTensor(Linear(16));

  PReLUAttributes prelu_0;
  prelu_0.alpha = MakeSyntheticTensor(Linear(16));

  PReLUAttributes prelu_1;
  prelu_1.alpha = MakeSyntheticTensor(Linear(32));

  PReLUAttributes prelu_2;
  prelu_2.alpha = MakeSyntheticTensor(Linear(16));

  auto src_shape_0 = BHWC(1, 14, 16, 16);
  auto src_shape_1 = BHWC(1, 14, 16, 32);
  TensorFloat32 src_tensor_0 = MakeSyntheticTensor(src_shape_0);
  TensorFloat32 src_tensor_1 = MakeSyntheticTensor(src_shape_1);

  TensorFloat32 interm_0 = DepthWiseConvolutionReference(dw_attr, src_tensor_0);
  TensorFloat32 interm_1 = ConvolutionReference(conv16to16, interm_0);
  TensorFloat32 interm_2 = PReLUReference(prelu_0, interm_1);
  TensorFloat32 interm_3 = ConvolutionReference(conv16to32, interm_2);
  TensorFloat32 interm_4 = AddTableReference({interm_3, src_tensor_1});
  TensorFloat32 out_ref_0 = PReLUReference(prelu_1, interm_4);
  TensorFloat32 interm_5 = ConvolutionReference(conv32to16, out_ref_0);
  TensorFloat32 out_ref_1 = PReLUReference(prelu_2, interm_5);

  auto operation = CreateDW3x3Conv16To16Conv16To32AddConv32To16(
      op_def, dw_attr, conv16to16, prelu_0, conv16to32, prelu_1, conv32to16,
      prelu_2, true);

  TensorFloat32 dst_0, dst_1;
  ABSL_EXPECT_OK(exec_env->ExecuteGPUOperation(
      {src_tensor_0, src_tensor_1},
      std::make_unique<GPUOperation>(std::move(operation)),
      {out_ref_0.shape, out_ref_1.shape}, {&dst_0, &dst_1}));
  EXPECT_THAT(dst_0.data, Pointwise(FloatNear(0.2f), out_ref_0.data));
  EXPECT_THAT(dst_1.data, Pointwise(FloatNear(0.2f), out_ref_1.data));
  return absl::OkStatus();
}

absl::Status DW3x3Conv16To16Conv16To32AddConv32To16Test2(
    TestExecutionEnvironment* exec_env, const OperationDef& op_def) {
  DepthwiseConvolution2DAttributes dw_attr;
  dw_attr.padding.prepended = HW(1, 1);
  dw_attr.padding.appended = HW(1, 1);
  dw_attr.strides = HW(1, 1);
  dw_attr.dilations = HW(1, 1);
  dw_attr.weights = MakeSyntheticTensor(OHWI(1, 3, 3, 16));
  dw_attr.bias = MakeSyntheticTensor(Linear(16));

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

  Convolution2DAttributes conv32to16;
  conv32to16.padding.prepended = HW(0, 0);
  conv32to16.padding.appended = HW(0, 0);
  conv32to16.strides = HW(1, 1);
  conv32to16.dilations = HW(1, 1);
  conv32to16.weights = MakeSyntheticTensor(OHWI(16, 1, 1, 32));
  conv32to16.bias = MakeSyntheticTensor(Linear(16));

  PReLUAttributes prelu_0;
  prelu_0.alpha = MakeSyntheticTensor(Linear(16));

  PReLUAttributes prelu_1;
  prelu_1.alpha = MakeSyntheticTensor(Linear(32));

  PReLUAttributes prelu_2;
  prelu_2.alpha = MakeSyntheticTensor(Linear(16));

  PadAttributes pad_attr;
  pad_attr.prepended = BHWC(0, 0, 0, 0);
  pad_attr.appended = BHWC(0, 0, 0, 24);

  auto src_shape_0 = BHWC(1, 14, 16, 16);
  auto src_shape_1 = BHWC(1, 14, 16, 8);
  TensorFloat32 src_tensor_0 = MakeSyntheticTensor(src_shape_0);
  TensorFloat32 src_tensor_1 = MakeSyntheticTensor(src_shape_1);

  TensorFloat32 interm_0 = DepthWiseConvolutionReference(dw_attr, src_tensor_0);
  TensorFloat32 interm_1 = ConvolutionReference(conv16to16, interm_0);
  TensorFloat32 interm_2 = PReLUReference(prelu_0, interm_1);
  TensorFloat32 interm_3 = ConvolutionReference(conv16to32, interm_2);
  TensorFloat32 interm_4 = PaddingReference(pad_attr, src_tensor_1);
  TensorFloat32 interm_5 = AddTableReference({interm_3, interm_4});
  TensorFloat32 out_ref_0 = PReLUReference(prelu_1, interm_5);
  TensorFloat32 interm_6 = ConvolutionReference(conv32to16, out_ref_0);
  TensorFloat32 out_ref_1 = PReLUReference(prelu_2, interm_6);

  auto operation = CreateDW3x3Conv16To16Conv16To32AddConv32To16(
      op_def, dw_attr, conv16to16, prelu_0, conv16to32, prelu_1, conv32to16,
      prelu_2, false);

  TensorFloat32 dst_0, dst_1;
  ABSL_EXPECT_OK(exec_env->ExecuteGPUOperation(
      {src_tensor_0, src_tensor_1},
      std::make_unique<GPUOperation>(std::move(operation)),
      {out_ref_0.shape, out_ref_1.shape}, {&dst_0, &dst_1}));
  EXPECT_THAT(dst_0.data, Pointwise(FloatNear(0.2f), out_ref_0.data));
  EXPECT_THAT(dst_1.data, Pointwise(FloatNear(0.2f), out_ref_1.data));
  return absl::OkStatus();
}

}  // namespace

TEST_P(DataTypeTest, DW3x3Conv16To16Conv16To32AddConv32To16Test0) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type(), storage(), Layout::HWC});
  op_def.src_tensors.push_back({data_type(), storage(), Layout::HWC});
  op_def.dst_tensors.push_back({data_type(), storage(), Layout::HWC});
  op_def.dst_tensors.push_back({data_type(), storage(), Layout::HWC});
  if (!op_def.src_tensors[0].SupportsZeroClamp(Axis::WIDTH,
                                               exec_env->GetGpuInfo()) ||
      !op_def.src_tensors[0].SupportsZeroClamp(Axis::HEIGHT,
                                               exec_env->GetGpuInfo())) {
    GTEST_SKIP() << "Source tensor does not support zero clamp.";
  }
  ABSL_EXPECT_OK(DW3x3Conv16To16Conv16To32AddConv32To16Test0(exec_env, op_def));
}

TEST_P(DataTypeTest, DW3x3Conv16To16Conv16To32AddConv32To16Test2) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type(), storage(), Layout::HWC});
  op_def.src_tensors.push_back({data_type(), storage(), Layout::HWC});
  op_def.dst_tensors.push_back({data_type(), storage(), Layout::HWC});
  op_def.dst_tensors.push_back({data_type(), storage(), Layout::HWC});
  if (!op_def.src_tensors[0].SupportsZeroClamp(Axis::WIDTH,
                                               exec_env->GetGpuInfo()) ||
      !op_def.src_tensors[0].SupportsZeroClamp(Axis::HEIGHT,
                                               exec_env->GetGpuInfo())) {
    GTEST_SKIP() << "Source tensor does not support zero clamp.";
  }
  ABSL_EXPECT_OK(DW3x3Conv16To16Conv16To32AddConv32To16Test2(exec_env, op_def));
}

INSTANTIATE_TEST_SUITE_P(
    Suite, DataTypeTest,
    Combine(ValuesIn(GetFloatTypes()), ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<DataTypeTest::ParamType>& info) {
      return ToString(info.param);
    });

}  // namespace ml_drift
