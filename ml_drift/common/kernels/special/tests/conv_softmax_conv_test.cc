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

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <memory>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "xnnpack.h"  // from @XNNPACK
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/special/conv_softmax_conv.h"
#include "ml_drift/common/kernels/tests/kernel_test.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_ref_ops.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::ValuesIn;

namespace {

TensorFloat32 SoftmaxConv1x1Base(const Convolution2DAttributes& attr,
                                 const TensorFloat32& src) {
  TensorFloat32 dst = MakeZeroTensor(CalculateOutputShape(src.shape, attr));
  const auto& weights = GetFloatWeights(attr);

  for (int dst_ch = 0; dst_ch < dst.shape.c; dst_ch++) {
    for (int dst_y = 0; dst_y < dst.shape.h; ++dst_y) {
      for (int dst_x = 0; dst_x < dst.shape.w; ++dst_x) {
        float dst_sum = 0.0;
        float exp_sum = 0.0;
        for (int src_ch = 0; src_ch < src.shape.c; src_ch++) {
          const int src_index =
              src.shape.LinearIndex({0, dst_y, dst_x, src_ch});
          const int f_index = weights.shape.LinearIndex({dst_ch, 0, 0, src_ch});
          dst_sum += exp(src.data[src_index]) * weights.data[f_index];
          exp_sum += exp(src.data[src_index]);
        }
        const int dst_index = dst.shape.LinearIndex({0, dst_y, dst_x, dst_ch});
        dst_sum = dst_sum / exp_sum;
        dst_sum += attr.bias.data.empty() ? 0 : attr.bias.data[dst_ch];
        dst.data[dst_index] = dst_sum;
      }
    }
  }
  return dst;
}

TensorFloat32 SoftmaxWithMaxAdjustmentConv1x1(
    const Convolution2DAttributes& attr, const TensorFloat32& src) {
  TensorFloat32 dst = MakeZeroTensor(CalculateOutputShape(src.shape, attr));
  const auto& weights = GetFloatWeights(attr);

  for (int dst_ch = 0; dst_ch < dst.shape.c; dst_ch++) {
    for (int dst_y = 0; dst_y < dst.shape.h; ++dst_y) {
      for (int dst_x = 0; dst_x < dst.shape.w; ++dst_x) {
        float dst_sum = 0.0;
        float exp_sum_adj = 0.0;
        float max_val = -FLT_MAX;
        for (int src_ch = 0; src_ch < src.shape.c; src_ch++) {
          const int src_index =
              src.shape.LinearIndex({0, dst_y, dst_x, src_ch});
          const int f_index = weights.shape.LinearIndex({dst_ch, 0, 0, src_ch});
          float src_val = src.data[src_index];
          float new_max_val = std::max(max_val, src_val);
          float src_val_exp_adj = exp(src_val - new_max_val);
          float scale = exp(max_val - new_max_val);
          float new_exp_sum_adj = exp_sum_adj * scale + src_val_exp_adj;

          // float scale2 = exp_sum_adj / new_exp_sum_adj;
          // src_val_exp_adj /= new_exp_sum_adj;
          // dst_sum *= scale * scale2;
          //  or
          dst_sum *= scale;

          dst_sum = dst_sum + src_val_exp_adj * weights.data[f_index];
          max_val = new_max_val;
          exp_sum_adj = new_exp_sum_adj;
        }
        const int dst_index = dst.shape.LinearIndex({0, dst_y, dst_x, dst_ch});
        dst_sum = dst_sum / exp_sum_adj;
        dst_sum += attr.bias.data.empty() ? 0 : attr.bias.data[dst_ch];
        dst.data[dst_index] = dst_sum;
      }
    }
  }
  return dst;
}

TensorFloat32 Conv1x1BatchedHeight(
    const TensorFloat32& src,
    const ml_drift::Tensor<OHWI, DataType::kFloat32>& weights) {
  const BHWC dst_shape =
      BHWC(src.shape.b, src.shape.h, src.shape.w, weights.shape.o);
  TensorFloat32 dst = MakeZeroTensor(dst_shape);

  for (int dst_ch = 0; dst_ch < dst.shape.c; dst_ch++) {
    for (int dst_y = 0; dst_y < dst.shape.h; ++dst_y) {
      for (int dst_x = 0; dst_x < dst.shape.w; ++dst_x) {
        float dst_sum = 0.0;
        for (int src_ch = 0; src_ch < src.shape.c; src_ch++) {
          const int src_index =
              src.shape.LinearIndex({0, dst_y, dst_x, src_ch});
          const int f_index =
              weights.shape.LinearIndex({dst_ch, 0, dst_y, src_ch});
          dst_sum += src.data[src_index] * weights.data[f_index];
        }
        const int dst_index = dst.shape.LinearIndex({0, dst_y, dst_x, dst_ch});
        dst.data[dst_index] = dst_sum;
      }
    }
  }
  return dst;
}

}  // namespace

TEST(SoftaxConv1x1, Base) {
  const int src_channels = 32;
  const int dst_channels = 32;
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  attr.weights = MakeSyntheticTensor(OHWI(dst_channels, 1, 1, src_channels));
  attr.bias = MakeSyntheticTensor(Linear(dst_channels));

  SoftmaxAttributes sfm_attr;
  sfm_attr.axis = Axis::kChannels;

  auto src_shape = BHWC(1, 8, 8, src_channels);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  TensorFloat32 interm_tensor = SoftmaxReference(sfm_attr, src_tensor);
  TensorFloat32 ref_tensor = ConvolutionReference(attr, interm_tensor);

  TensorFloat32 dst0_tensor = SoftmaxConv1x1Base(attr, src_tensor);

  EXPECT_THAT(ref_tensor.data,
              testing::Pointwise(testing::FloatNear(1e-6f), dst0_tensor.data));

  TensorFloat32 dst1_tensor = SoftmaxWithMaxAdjustmentConv1x1(attr, src_tensor);
  EXPECT_THAT(ref_tensor.data,
              testing::Pointwise(testing::FloatNear(1e-6f), dst1_tensor.data));
}

using SoftmaxConv1x1BigTest = TestWithParam<TensorStorageType>;

TEST_P(SoftmaxConv1x1BigTest, BatchedConvSoftmaxBatchedConv) {
  if (!exec_env->IsStorageSupported(GetParam(), DataType::kFloat16)) {
    GTEST_SKIP() << "Unsupported storage: " << ToString(GetParam());
  }
  int batch_size = 8;
  int width = 32;
  int src_ch = 40;
  int interm_ch = 128;
  int dst_ch = 40;
  ml_drift::Tensor<OHWI, DataType::kFloat32> weights0 =
      MakeSyntheticTensor(OHWI(interm_ch, 1, batch_size, src_ch));
  weights0.data.resize(weights0.shape.DimensionsProduct() +
                       XNN_EXTRA_BYTES / sizeof(float));
  ml_drift::Tensor<OHWI, DataType::kFloat32> weights1 =
      MakeSyntheticTensor(OHWI(dst_ch, 1, batch_size, interm_ch));
  weights1.data.resize(weights1.shape.DimensionsProduct() +
                       XNN_EXTRA_BYTES / sizeof(float));

  auto src_shape = BHWC(1, batch_size, width, src_ch);
  auto dst_shape = BHWC(1, batch_size, width, dst_ch);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);

  OperationDef op_def;
  auto precision = CalculationsPrecision::kF16;
  auto data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, GetParam(), Layout::kHWC});
  op_def.dst_tensors.push_back({data_type, GetParam(), Layout::kHWC});

  TensorFloat32 interm0_tensor = Conv1x1BatchedHeight(src_tensor, weights0);

  TensorFloat32 interm1_tensor;

  SoftmaxAttributes attr;
  attr.axis = Axis::kChannels;
  interm1_tensor = SoftmaxReference(attr, interm0_tensor);

  TensorFloat32 dst_ref_tensor = Conv1x1BatchedHeight(interm1_tensor, weights1);

  ConvSoftmaxConv::WeightsDesc weights_desc;
  weights_desc.constant = true;
  weights_desc.weights0 = &weights0;
  weights_desc.weights1 = &weights1;
  auto operation = CreateConvSoftmaxConv(exec_env->GetGpuInfo(), op_def,
                                         precision, weights_desc);
  int mads_amount = interm_ch + dst_ch;
  float eps = GetEpsilon(precision, exec_env->GetGpuInfo()) * mads_amount;

  TensorFloat32 dst_tensor;
  ABSL_EXPECT_OK(exec_env->ExecuteGPUOperation(
      src_tensor, std::make_unique<ConvSoftmaxConv>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              testing::Pointwise(testing::FloatNear(eps), dst_ref_tensor.data));
}

INSTANTIATE_TEST_SUITE_P(
    Suite, SoftmaxConv1x1BigTest, ValuesIn(GetTensorStoragesTypes()),
    [](const TestParamInfo<SoftmaxConv1x1BigTest::ParamType>& info) {
      return absl::StrReplaceAll(ToString(info.param), {{":", ""}});
    });

}  // namespace ml_drift
