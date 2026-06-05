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

#include "ml_drift/common/kernels/tests/reshape_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/reshape.h"
#include "ml_drift/common/kernels/reshapex4.h"
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

// 2^-9 should be the maximum error for values in range [0.0, 8.0]
constexpr float kEps = 1.0f / (1 << 9);

absl::Status ReshapeTest(TestExecutionEnvironment& env, DataType data_type,
                         TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 3);
  src_tensor.data = {0.5f, -1.1f, -2.2f, 3.1f, 1.2f, 2.9f};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateReshape(op_def);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 3, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(kEps), {0.5f, -1.1f, -2.2f,
                                                           3.1f, 1.2f, 2.9f}));
  return absl::OkStatus();
}

absl::Status Reshapex4Test(TestExecutionEnvironment& env, DataType data_type,
                           TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 8);
  src_tensor.data = {0.5f, -1.1f, -2.2f, 3.1f, 1.2f, 2.9f, 4.2f, -1.9f};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  Reshapex4 operation = CreateReshapex4(op_def);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Reshapex4>(std::move(operation)),
      BHWC(1, 1, 2, 4), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps),
                        {0.5f, -1.1f, -2.2f, 3.1f, 1.2f, 2.9f, 4.2f, -1.9f}));
  return absl::OkStatus();
}

absl::Status ReshapeBigTest(TestExecutionEnvironment& env, DataType data_type,
                            TensorStorageType storage, const BHWC& src_shape,
                            const BHWC& dst_shape) {
  ReshapeAttributes attr;
  attr.new_shape = dst_shape;

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  TensorFloat32 dst_ref_tensor = ReshapeReference(attr, src_tensor);

  OperationDef op_def;
  const Layout src_layout = src_shape.b == 1 ? Layout::HWC : Layout::BHWC;
  const Layout dst_layout = dst_shape.b == 1 ? Layout::HWC : Layout::BHWC;
  op_def.src_tensors.push_back({data_type, storage, src_layout});
  op_def.dst_tensors.push_back({data_type, storage, dst_layout});
  GPUOperation operation = CreateReshape(op_def);

  TensorFloat32 dst_tensor;
  MLD_EXPECT_OK(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.001f), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status Reshape3DBigTest(TestExecutionEnvironment& env, DataType data_type,
                              TensorStorageType storage, const BHWDC& src_shape,
                              const BHWDC& dst_shape) {
  Reshape3DAttributes attr;
  attr.new_shape = dst_shape;

  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  Tensor5DFloat32 dst_ref_tensor = ReshapeReference(attr, src_tensor);

  OperationDef op_def;
  const Layout src_layout = src_shape.b == 1 ? Layout::HWDC : Layout::BHWDC;
  const Layout dst_layout = dst_shape.b == 1 ? Layout::HWDC : Layout::BHWDC;
  op_def.src_tensors.push_back({data_type, storage, src_layout});
  op_def.dst_tensors.push_back({data_type, storage, dst_layout});
  GPUOperation operation = CreateReshape(op_def);

  Tensor5DFloat32 dst_tensor;
  MLD_EXPECT_OK(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.001f), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status Reshapex4Test(TestExecutionEnvironment& env, DataType data_type,
                           TensorStorageType storage, const BHWC& src_shape,
                           const BHWC& dst_shape) {
  ReshapeAttributes attr;
  attr.new_shape = dst_shape;

  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  TensorFloat32 dst_ref_tensor = ReshapeReference(attr, src_tensor);

  OperationDef op_def;
  const Layout src_layout = src_shape.b == 1 ? Layout::HWC : Layout::BHWC;
  const Layout dst_layout = dst_shape.b == 1 ? Layout::HWC : Layout::BHWC;
  op_def.src_tensors.push_back({data_type, storage, src_layout});
  op_def.dst_tensors.push_back({data_type, storage, dst_layout});
  Reshapex4 operation = CreateReshapex4(op_def);
  TensorFloat32 dst_tensor;
  MLD_EXPECT_OK(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Reshapex4>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.001f), dst_ref_tensor.data));
  return absl::OkStatus();
}

}  // namespace ml_drift
