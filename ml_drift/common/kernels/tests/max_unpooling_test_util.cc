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

#include "ml_drift/common/kernels/tests/max_unpooling_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/max_unpooling.h"
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

absl::Status MaxUnpoolingTest(TestExecutionEnvironment& env, DataType data_type,
                              TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 1);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};
  TensorFloat32 src_ind_tensor;
  src_ind_tensor.shape = BHWC(1, 2, 2, 1);
  src_ind_tensor.data = {0.1f, 1.1f, 2.1f, 3.1f};

  MaxUnpooling2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(2, 2);
  attr.kernel = HW(2, 2);

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateMaxUnpooling(env.GetGpuInfo(), op_def, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor, src_ind_tensor},
      std::make_unique<GPUOperation>(std::move(operation)), BHWC(1, 4, 4, 1),
      &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps),
                        {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                         0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 3.0f}));
  return absl::OkStatus();
}

namespace {
absl::Status MaxUnpoolingTest(TestExecutionEnvironment& exec_env,
                              const MaxUnpooling2DAttributes& attr,
                              const std::vector<TensorFloat32>& src_tensors,
                              const OperationDef& op_def) {
  TensorFloat32 dst_ref_tensor =
      MaxUnpoolingReference(attr, src_tensors[0], src_tensors[1]);

  std::vector<const TensorFloat32*> src_ptrs = {&src_tensors[0],
                                                &src_tensors[1]};

  auto operation = CreateMaxUnpooling(exec_env.GetGpuInfo(), op_def, attr);

  TensorFloat32 dst_tensor;
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      {src_tensors[0], src_tensors[1]},
      std::make_unique<GPUOperation>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.001f), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status MaxUnpoolingTest(TestExecutionEnvironment& exec_env,
                              const MaxUnpooling3DAttributes& attr,
                              const std::vector<Tensor5DFloat32>& src_tensors,
                              const OperationDef& op_def) {
  Tensor5DFloat32 dst_ref_tensor =
      MaxUnpoolingReference(attr, src_tensors[0], src_tensors[1]);

  std::vector<const Tensor5DFloat32*> src_ptrs = {&src_tensors[0],
                                                  &src_tensors[1]};

  auto operation = CreateMaxUnpooling(exec_env.GetGpuInfo(), op_def, attr);

  Tensor5DFloat32 dst_tensor;
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      {src_tensors[0], src_tensors[1]},
      std::make_unique<GPUOperation>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.001f), dst_ref_tensor.data));
  return absl::OkStatus();
}
}  // namespace

absl::Status MaxUnpoolingBigTest(TestExecutionEnvironment& env,
                                 DataType data_type,
                                 TensorStorageType storage) {
  MaxUnpooling2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(2, 2);
  attr.kernel = HW(2, 2);

  auto src_shape = BHWC(1, 7, 11, 8);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  TensorFloat32 src_ind_tensor = MakeZeroTensor(src_shape);
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_ind_tensor.data[i] = i % (attr.kernel.w * attr.kernel.h) + 0.1f;
  }

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  return MaxUnpoolingTest(env, attr, {src_tensor, src_ind_tensor}, op_def);
}

absl::Status MaxUnpoolingBatchedBigTest(TestExecutionEnvironment& env,
                                        DataType data_type,
                                        TensorStorageType storage) {
  MaxUnpooling2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(2, 2);
  attr.kernel = HW(2, 2);

  auto src_shape = BHWC(3, 7, 11, 8);

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  TensorFloat32 src_ind_tensor = MakeZeroTensor(src_shape);
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_ind_tensor.data[i] = i % (attr.kernel.w * attr.kernel.h) + 0.1f;
  }

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  return MaxUnpoolingTest(env, attr, {src_tensor, src_ind_tensor}, op_def);
}

absl::Status MaxUnpooling3DBigTest(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage) {
  MaxUnpooling3DAttributes attr;
  attr.padding.prepended = HWD(0, 0, 0);
  attr.padding.appended = HWD(0, 0, 0);
  attr.strides = HWD(2, 2, 2);
  attr.kernel = HWD(2, 2, 2);

  auto src_shape = BHWDC(1, 7, 11, 5, 8);

  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  Tensor5DFloat32 src_ind_tensor = MakeZeroTensor(src_shape);
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_ind_tensor.data[i] =
        i % (attr.kernel.w * attr.kernel.h * attr.kernel.d) + 0.1f;
  }

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWDC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWDC});
  return MaxUnpoolingTest(env, attr, {src_tensor, src_ind_tensor}, op_def);
}

absl::Status MaxUnpooling3DBatchedBigTest(TestExecutionEnvironment& env,
                                          DataType data_type,
                                          TensorStorageType storage) {
  MaxUnpooling3DAttributes attr;
  attr.padding.prepended = HWD(0, 0, 0);
  attr.padding.appended = HWD(0, 0, 0);
  attr.strides = HWD(2, 2, 2);
  attr.kernel = HWD(2, 2, 2);

  auto src_shape = BHWDC(3, 7, 11, 5, 8);

  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  Tensor5DFloat32 src_ind_tensor = MakeZeroTensor(src_shape);
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_ind_tensor.data[i] =
        i % (attr.kernel.w * attr.kernel.h * attr.kernel.d) + 0.1f;
  }

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWDC});
  return MaxUnpoolingTest(env, attr, {src_tensor, src_ind_tensor}, op_def);
}

}  // namespace ml_drift
