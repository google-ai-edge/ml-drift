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

#include "ml_drift/common/kernels/special/dw7x7_conv2to6_concat_conv8to8.h"

#include <memory>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/operations.h"
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
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::ValuesIn;

namespace {

absl::Status DW7x7Conv2To6ConcatConv8to8Test(TestExecutionEnvironment* exec_env,
                                             const OperationDef& op_def) {
  ConcatAttributes concat_attr;
  concat_attr.axis = Axis::CHANNELS;

  DepthwiseConvolution2DAttributes dw_attr;
  dw_attr.padding.prepended = HW(2, 2);
  dw_attr.padding.appended = HW(3, 3);
  dw_attr.strides = HW(2, 2);
  dw_attr.dilations = HW(1, 1);
  dw_attr.weights = MakeSyntheticTensor(OHWI(1, 7, 7, 2));
  dw_attr.bias = MakeSyntheticTensor(Linear(2));

  Convolution2DAttributes conv2to6;
  conv2to6.padding.prepended = HW(0, 0);
  conv2to6.padding.appended = HW(0, 0);
  conv2to6.strides = HW(1, 1);
  conv2to6.dilations = HW(1, 1);
  conv2to6.weights = MakeSyntheticTensor(OHWI(6, 1, 1, 2));
  conv2to6.bias = MakeSyntheticTensor(Linear(6));

  Convolution2DAttributes conv8to8;
  conv8to8.padding.prepended = HW(0, 0);
  conv8to8.padding.appended = HW(0, 0);
  conv8to8.strides = HW(1, 1);
  conv8to8.dilations = HW(1, 1);
  conv8to8.weights = MakeSyntheticTensor(OHWI(8, 1, 1, 8));
  conv8to8.bias = MakeSyntheticTensor(Linear(8));

  PReLUAttributes prelu_0;
  prelu_0.alpha = MakeSyntheticTensor(Linear(6));

  PReLUAttributes prelu_1;
  prelu_1.alpha = MakeSyntheticTensor(Linear(8));

  Pooling2DAttributes pool_attr;
  pool_attr.padding.prepended = HW(0, 0);
  pool_attr.padding.appended = HW(0, 0);
  pool_attr.strides = HW(2, 2);
  pool_attr.kernel = HW(2, 2);
  pool_attr.type = PoolingType::MAX;

  auto src_shape = BHWC(1, 14, 16, 2);
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  TensorFloat32 interm_1 = DepthWiseConvolutionReference(dw_attr, src_tensor);
  TensorFloat32 interm_2 = MaxPoolingReference(pool_attr, src_tensor)[0];
  TensorFloat32 interm_3 = ConvolutionReference(conv2to6, interm_1);
  TensorFloat32 interm_3_prelu = PReLUReference(prelu_0, interm_3);
  TensorFloat32 result0_ref =
      ConcatReference(concat_attr, {interm_3_prelu, interm_2});
  TensorFloat32 interm_5 = ConvolutionReference(conv8to8, result0_ref);
  TensorFloat32 result1_ref = PReLUReference(prelu_1, interm_5);

  auto operation = CreateDW7x7Conv2To6ConcatConv8to8(
      op_def, dw_attr, conv2to6, prelu_0, conv8to8, prelu_1);

  TensorFloat32 dst_0, dst_1;
  MLD_EXPECT_OK(exec_env->ExecuteGPUOperation(
      {src_tensor}, std::make_unique<GPUOperation>(std::move(operation)),
      {result0_ref.shape, result1_ref.shape}, {&dst_0, &dst_1}));
  EXPECT_THAT(dst_0.data, Pointwise(FloatNear(0.1f), result0_ref.data));
  EXPECT_THAT(dst_1.data, Pointwise(FloatNear(0.1f), result1_ref.data));
  return absl::OkStatus();
}

}  // namespace

using StorageTest = TestWithParam<TensorStorageType>;

TEST_P(StorageTest, DW7x7Conv2To6ConcatConv8to8Test) {
  if (!IsDW7x7Conv2To6ConcatConv8to8Supported(exec_env->GetGpuInfo())) {
    GTEST_SKIP() << "Skipped unsupported DW7x7Conv2To6ConcatConv8to8 Test.";
  }
  const DataType data_type = DataType::FLOAT16;
  const TensorStorageType storage = GetParam();
  if (!exec_env->IsStorageSupported(storage, data_type)) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type)
                 << " storage type: " << ToString(storage);
  }
  OperationDef op_def;
  op_def.src_tensors.push_back(
      {DataType::FLOAT16, TensorStorageType::SINGLE_TEXTURE_2D, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::FLOAT16, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::FLOAT16, storage, Layout::HWC});
  MLD_EXPECT_OK(DW7x7Conv2To6ConcatConv8to8Test(exec_env, op_def));
}

INSTANTIATE_TEST_SUITE_P(Suite, StorageTest, ValuesIn(GetTensorStoragesTypes()),
                         [](const TestParamInfo<StorageTest::ParamType>& info) {
                           return absl::StrReplaceAll(ToString(info.param),
                                                      {{":", ""}});
                         });

}  // namespace ml_drift
