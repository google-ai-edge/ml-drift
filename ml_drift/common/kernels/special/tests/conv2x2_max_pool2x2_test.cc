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

#include "ml_drift/common/kernels/special/conv2x2_max_pool2x2.h"

#include <memory>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
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

using ::testing::Combine;
using ::testing::FloatNear;
using ::testing::Pointwise;
using ::testing::TestParamInfo;
using ::testing::ValuesIn;

namespace {

absl::Status Conv2x2MaxPool2x2Test(TestExecutionEnvironment* exec_env,
                                   const OperationDef& op_def) {
  Convolution2DAttributes conv2x2;
  conv2x2.padding.prepended = HW(0, 0);
  conv2x2.padding.appended = HW(0, 0);
  conv2x2.strides = HW(2, 2);
  conv2x2.dilations = HW(1, 1);
  conv2x2.weights = MakeSyntheticTensor(OHWI(16, 2, 2, 8));
  conv2x2.bias = MakeSyntheticTensor(Linear(16));

  PReLUAttributes prelu;
  prelu.alpha = MakeSyntheticTensor(Linear(16));

  Pooling2DAttributes pool_attr;
  pool_attr.padding.prepended = HW(0, 0);
  pool_attr.padding.appended = HW(0, 0);
  pool_attr.strides = HW(2, 2);
  pool_attr.kernel = HW(2, 2);
  pool_attr.type = PoolingType::MAX;

  auto src_shape = BHWC(1, 7, 8, 8);
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  TensorFloat32 interm_0 = ConvolutionReference(conv2x2, src_tensor);
  TensorFloat32 out_ref_0 = PReLUReference(prelu, interm_0);
  TensorFloat32 out_ref_1 = MaxPoolingReference(pool_attr, src_tensor)[0];

  auto operation = CreateConv2x2MaxPool2x2(op_def, conv2x2, prelu);

  TensorFloat32 dst_0, dst_1;
  MLD_EXPECT_OK(exec_env->ExecuteGPUOperation(
      {src_tensor}, std::make_unique<GPUOperation>(std::move(operation)),
      {out_ref_0.shape, out_ref_1.shape}, {&dst_0, &dst_1}));
  EXPECT_THAT(dst_0.data, Pointwise(FloatNear(0.02f), out_ref_0.data));
  EXPECT_THAT(dst_1.data, Pointwise(FloatNear(0.02f), out_ref_1.data));
  return absl::OkStatus();
}

}  // namespace

TEST_P(DataTypeTest, Conv2x2MaxPool2x2) {
  if (!exec_env->IsStorageSupported(storage(), data_type())) {
    GTEST_SKIP() << "Unsupported data type: " << ToString(data_type())
                 << " storage type: " << ToString(storage());
  }
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type(), storage(), Layout::HWC});
  op_def.dst_tensors.push_back({data_type(), storage(), Layout::HWC});
  op_def.dst_tensors.push_back({data_type(), storage(), Layout::HWC});
  if (!op_def.src_tensors[0].SupportsZeroClamp(Axis::WIDTH,
                                               exec_env->GetGpuInfo()) ||
      !op_def.src_tensors[0].SupportsZeroClamp(Axis::HEIGHT,
                                               exec_env->GetGpuInfo())) {
    GTEST_SKIP() << "Source tensor does not support zero clamp.";
  }
  MLD_EXPECT_OK(Conv2x2MaxPool2x2Test(exec_env, op_def));
}

INSTANTIATE_TEST_SUITE_P(
    Conv2x2MaxPool2x2TestSuite, DataTypeTest,
    Combine(ValuesIn(GetFloatTypes()), ValuesIn(GetTensorStoragesTypes())),
    [](const TestParamInfo<DataTypeTest::ParamType>& info) {
      return ToString(info.param);
    });

}  // namespace ml_drift
