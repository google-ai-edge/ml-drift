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

#include "ml_drift/common/kernels/tests/rope_test_util.h"

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/rope.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

absl::Status RoPETest(TestExecutionEnvironment& env, DataType data_type,
                      TensorStorageType storage) {
  TensorFloat32 src_l;
  src_l.shape = BHWC(1, 1, 4, 4);
  src_l.data = {0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
                -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f};
  TensorFloat32 src_r;
  src_r.shape = BHWC(1, 1, 4, 4);
  src_r.data = {0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
                -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f};

  TensorFloat32 pos_tensor;
  pos_tensor.shape = BHWC(1, 1, 4, 1);
  pos_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f};

  const float eps = data_type == DataType::FLOAT32 ? 5e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_l;
  TensorFloat32 dst_r;
  RoPEAttributes attr;
  GPUOperation operation = CreateRoPE(env.GetGpuInfo(), op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_l, src_r, pos_tensor},
      std::make_unique<GPUOperation>(std::move(operation)),
      {BHWC(1, 1, 4, 4), BHWC(1, 1, 4, 4)}, {&dst_l, &dst_r}));
  EXPECT_THAT(
      dst_l.data,
      Pointwise(FloatNear(eps),
                {0.0f, 0.895171f, 1.979900f, 2.996999f, -5.301777f, 3.906986f,
                 5.878808f, 6.985986f, 1.131112f, -1.319633f, -2.908664f,
                 -3.987982f, -0.515794f, -3.189856f, -6.714476f, -7.967936f}));
  EXPECT_THAT(
      dst_r.data,
      Pointwise(FloatNear(eps),
                {0.0f, 1.094838f, 2.019900f, 3.002999f, 1.972603f, 5.893680f,
                 6.118792f, 7.013986f, 0.848872f, -2.501714f, -3.088637f,
                 -4.011982f, 7.052231f, -7.862876f, -7.274326f, -8.031936f}));
  return absl::OkStatus();
}

absl::Status SplitRoPEConcatTest(TestExecutionEnvironment& env,
                                 DataType data_type,
                                 TensorStorageType storage) {
  TensorFloat32 src;
  src.shape = BHWC(1, 1, 4, 8);
  src.data = {0.0f,  1.0f,  2.0f,  3.0f,  0.0f,  1.0f,  2.0f,  3.0f,
              4.0f,  5.0f,  6.0f,  7.0f,  4.0f,  5.0f,  6.0f,  7.0f,
              -1.0f, -2.0f, -3.0f, -4.0f, -1.0f, -2.0f, -3.0f, -4.0f,
              -5.0f, -6.0f, -7.0f, -8.0f, -5.0f, -6.0f, -7.0f, -8.0f};

  TensorFloat32 pos_tensor;
  pos_tensor.shape = BHWC(1, 1, 4, 1);
  pos_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f};

  const float eps = data_type == DataType::FLOAT32 ? 5e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst;
  RoPEAttributes attr;
  GPUOperation operation =
      CreateSplitRoPEConcat(env.GetGpuInfo(), op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src, pos_tensor}, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 1, 4, 8), &dst));
  EXPECT_THAT(
      dst.data,
      Pointwise(FloatNear(eps),
                {0.0f,       0.895171f,  1.979900f,  2.996999f,  0.0f,
                 1.094838f,  2.019900f,  3.002999f,  -5.301777f, 3.906986f,
                 5.878808f,  6.985986f,  1.972603f,  5.893680f,  6.118792f,
                 7.013986f,  1.131112f,  -1.319633f, -2.908664f, -3.987982f,
                 0.848872f,  -2.501714f, -3.088637f, -4.011982f, -0.515794f,
                 -3.189856f, -6.714476f, -7.967936f, 7.052231f,  -7.862876f,
                 -7.274326f, -8.031936f}));
  return absl::OkStatus();
}

absl::Status SplitRoPEConcatIntPositionTest(TestExecutionEnvironment& env,
                                            DataType data_type,
                                            TensorStorageType storage) {
  TensorFloat32 src;
  src.shape = BHWC(1, 1, 4, 8);
  src.data = {0.0f,  1.0f,  2.0f,  3.0f,  0.0f,  1.0f,  2.0f,  3.0f,
              4.0f,  5.0f,  6.0f,  7.0f,  4.0f,  5.0f,  6.0f,  7.0f,
              -1.0f, -2.0f, -3.0f, -4.0f, -1.0f, -2.0f, -3.0f, -4.0f,
              -5.0f, -6.0f, -7.0f, -8.0f, -5.0f, -6.0f, -7.0f, -8.0f};

  TensorInt32 pos_tensor;
  pos_tensor.shape = BHWC(1, 1, 4, 1);
  pos_tensor.data = {1, 2, 3, 4};

  const float eps = data_type == DataType::FLOAT32 ? 5e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.UploadData(src);
  TensorDescriptor pos_td = op_def.src_tensors[1];
  pos_td.UploadData(pos_tensor);
  TensorDescriptor dst_td = op_def.dst_tensors[0];
  dst_td.SetBHWCShape(BHWC(1, 1, 4, 8));
  RoPEAttributes attr;
  GPUOperation operation =
      CreateSplitRoPEConcat(env.GetGpuInfo(), op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_td, &pos_td}, {&dst_td},
      std::make_unique<GPUOperation>(std::move(operation))));
  TensorFloat32 dst;
  dst_td.DownloadData(&dst);
  EXPECT_THAT(
      dst.data,
      Pointwise(FloatNear(eps),
                {0.0f,       0.895171f,  1.979900f,  2.996999f,  0.0f,
                 1.094838f,  2.019900f,  3.002999f,  -5.301777f, 3.906986f,
                 5.878808f,  6.985986f,  1.972603f,  5.893680f,  6.118792f,
                 7.013986f,  1.131112f,  -1.319633f, -2.908664f, -3.987982f,
                 0.848872f,  -2.501714f, -3.088637f, -4.011982f, -0.515794f,
                 -3.189856f, -6.714476f, -7.967936f, 7.052231f,  -7.862876f,
                 -7.274326f, -8.031936f}));
  return absl::OkStatus();
}

absl::Status SplitRoPEConcatInterleavedAxialTest(TestExecutionEnvironment& env,
                                                 DataType data_type,
                                                 TensorStorageType storage) {
  TensorFloat32 src;
  src.shape = BHWC(1, 2, 2, 8);
  src.data = {
      1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
      0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
      1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
  };

  TensorFloat32 pos_tensor;
  pos_tensor.shape = BHWC(1, 1, 2, 1);
  pos_tensor.data = {1.0f, 2.0f};

  const float eps = data_type == DataType::FLOAT32 ? 5e-6f : 1e-2f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst;
  RoPEAttributes attr;
  attr.kernel_type = RoPEKernelType::INTERLEAVED_2D;
  GPUOperation operation =
      CreateSplitRoPEConcat(env.GetGpuInfo(), op_def, attr);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src, pos_tensor}, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 2, 8), &dst));

  auto expected_pair = [](float pos_val) {
    return std::vector<float>{
        std::cos(pos_val / 1.0f), std::sin(pos_val / 1.0f),
        std::cos(pos_val / 100.0f), std::sin(pos_val / 100.0f)};
  };
  std::vector<float> expected;
  // (Y=0, X=0): X%2 = 0 (1.0f), Y%2 = 0 (1.0f)
  for (float v : expected_pair(1.0f)) expected.push_back(v);
  for (float v : expected_pair(1.0f)) expected.push_back(v);
  // (Y=0, X=1): X%2 = 1 (2.0f), Y%2 = 0 (1.0f)
  for (float v : expected_pair(2.0f)) expected.push_back(v);
  for (float v : expected_pair(1.0f)) expected.push_back(v);
  // (Y=1, X=0): X%2 = 0 (1.0f), Y%2 = 1 (2.0f)
  for (float v : expected_pair(1.0f)) expected.push_back(v);
  for (float v : expected_pair(2.0f)) expected.push_back(v);
  // (Y=1, X=1): X%2 = 1 (2.0f), Y%2 = 1 (2.0f)
  for (float v : expected_pair(2.0f)) expected.push_back(v);
  for (float v : expected_pair(2.0f)) expected.push_back(v);

  EXPECT_THAT(dst.data, Pointwise(FloatNear(eps), expected));
  return absl::OkStatus();
}

}  // namespace ml_drift
