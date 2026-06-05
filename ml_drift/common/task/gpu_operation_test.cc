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

#include "ml_drift/common/task/gpu_operation.h"

#include <memory>
#include <string>
#include <utility>

#include "benchmark/benchmark.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "xnnpack.h"  // from @XNNPACK
#include "ml_drift/common/access_type.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernels/conv_generic.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"

namespace ml_drift {
namespace {

TEST(GpuOperationTest, RemovesComments) {
  GPUOperation op;
  op.code_ = R"(
    int foo;// comment
    int bar;/* comment */
/*
     multi line
     comment
     */
    int baz; /* foo */// more comment)";
  GpuInfo gpu_info;
  MLD_EXPECT_OK(op.AssembleCode(gpu_info));
  EXPECT_EQ(op.code_, R"(
    int foo;
    int bar;

    int baz; )");
}

TEST(GpuOperationTest, RemovesCommentsEndsWithLongComment) {
  GPUOperation op;
  op.code_ = R"(
    int foo;/*
    comment */)";
  GpuInfo gpu_info;
  MLD_EXPECT_OK(op.AssembleCode(gpu_info));
  EXPECT_EQ(op.code_, R"(
    int foo;)");
}

TEST(GpuOperationTest, MarksActiveArguments) {
  GPUOperation op;
  op.args_.AddFloat("f1", 1.0f);
  op.args_.AddFloat("f2", 2.0f);
  op.args_.AddHalf("h1", half(1.0f));
  op.args_.AddHalf("h2", half(2.0f));
  op.args_.AddInt("i1", 1);
  op.args_.AddInt("i2", 2);
  op.args_.AddUint("u1", 1);
  op.args_.AddUint("u2", 2);

  const std::string code = R"(
    float a = args.f1;
    float b = args.f2a;
    float c = zargs.f2;
    half d =args.h1;
    int e =  args.i1+args.u1+_args.u2;args.)";
  op.code_ = code;
  GpuInfo gpu_info;
  MLD_EXPECT_OK(op.AssembleCode(gpu_info));
  EXPECT_EQ(op.code_, code);

  const auto& args = op.args_;
  EXPECT_TRUE(args.GetFloatValues().at("f1").active);
  EXPECT_TRUE(args.GetHalfValues().at("h1").active);
  EXPECT_TRUE(args.GetIntValues().at("i1").active);
  EXPECT_TRUE(args.GetUintValues().at("u1").active);

  EXPECT_FALSE(args.GetFloatValues().at("f2").active);
  EXPECT_FALSE(args.GetHalfValues().at("h2").active);
  EXPECT_FALSE(args.GetIntValues().at("i2").active);
  EXPECT_FALSE(args.GetUintValues().at("u2").active);
}

TEST(GpuOperationTest, TestNoSelector) {
  GPUOperation op;

  BufferDescriptor desc;
  desc.element_type = DataType::FLOAT32;
  desc.element_size = 4;
  desc.memory_type = MemoryType::GLOBAL;

  op.args_.AddObjectRef("weights", AccessType::READ,
                    std::make_unique<BufferDescriptor>(std::move(desc)));

  op.code_ = R"(
    if (a < 3) {
      value = args.weights.UnknownSelector(id);
    }
  )";
  EXPECT_FALSE(op.AssembleCode(GpuInfo()).ok());
}

void BM_AssembleCode(benchmark::State& state) {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;
  Convolution2DAttributes conv_attr;
  conv_attr.padding.prepended = HW(0, 0);
  conv_attr.padding.appended = HW(0, 0);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  auto& attr_weights =
      conv_attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(16, 1, 1, 128);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  conv_attr.bias.shape = Linear(16);
  conv_attr.bias.data.resize(conv_attr.bias.shape.DimensionsProduct());

  OperationDef op_def;
  op_def.src_tensors.push_back(
      {DataType::FLOAT32, TensorStorageType::TEXTURE_2D, Layout::HWC});
  op_def.dst_tensors.push_back(
      {DataType::FLOAT32, TensorStorageType::TEXTURE_2D, Layout::HWC});

  for (auto s : state) {
    ConvGeneric conv_operation = CreateConvGeneric(
        gpu_info, op_def, CalculationsPrecision::F32, conv_attr);
    MLD_EXPECT_OK(conv_operation.AssembleCode(gpu_info));
  }
}
BENCHMARK(BM_AssembleCode);

}  // namespace
}  // namespace ml_drift
