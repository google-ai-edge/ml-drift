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

#include "ml_drift/common/kernels/tests/space_to_depth_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/space_to_depth.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace ml_drift {

// 2^-8 should be the maximum error for values in range [0.0, 16.0]
constexpr float kEps = 1.0f / (1 << 8);

absl::Status SpaceToDepthTensorShape1x2x2x1BlockSize2Test(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 1);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f};
  const SpaceToDepthAttributes attr = {.block_size = 2};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateSpaceToDepth(op_def, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 1, 1, 4), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps), {1.0f, 2.0f, 3.0f, 4.0f}));
  return absl::OkStatus();
}

absl::Status SpaceToDepthTensorShape1x2x2x2BlockSize2Test(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 2);
  src_tensor.data = {1.4f, 2.3f, 3.2f, 4.1f, 5.4f, 6.3f, 7.2f, 8.1f};
  const SpaceToDepthAttributes attr = {.block_size = 2};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateSpaceToDepth(op_def, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 1, 1, 8), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps),
                        {1.4f, 2.3f, 3.2f, 4.1f, 5.4f, 6.3f, 7.2f, 8.1f}));
  return absl::OkStatus();
}

absl::Status SpaceToDepthTensorShape1x2x2x3BlockSize2Test(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 3);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                     7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  const SpaceToDepthAttributes attr = {.block_size = 2};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateSpaceToDepth(op_def, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 1, 1, 12), &dst_tensor));
  EXPECT_THAT(
      dst_tensor.data,
      Pointwise(FloatNear(kEps), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
                                  8.0f, 9.0f, 10.0f, 11.0f, 12.0f}));
  return absl::OkStatus();
}

absl::Status SpaceToDepthTensorShape1x4x4x1BlockSize2Test(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 4, 4, 1);
  src_tensor.data = {1.0f, 2.0f,  5.0f,  6.0f,  3.0f,  4.0f,  7.0f,  8.0f,
                     9.0f, 10.0f, 13.0f, 14.0f, 11.0f, 12.0f, 15.0f, 16.0f};
  const SpaceToDepthAttributes attr = {.block_size = 2};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateSpaceToDepth(op_def, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 2, 4), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps),
                        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                         10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f}));
  return absl::OkStatus();
}

absl::Status DepthToSpaceFrom1x1x1x4To1x2x2x1Test(TestExecutionEnvironment& env,
                                                  DataType data_type,
                                                  TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 4);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f};
  const SpaceToDepthAttributes attr = {.block_size = 2};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateDepthToSpace(op_def, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 2, 1), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps), {1.0f, 2.0f, 3.0f, 4.0f}));
  return absl::OkStatus();
}

absl::Status DepthToSpaceFrom1x1x1x16To1x2x2x4Test(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 16);
  src_tensor.data = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                     9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  const SpaceToDepthAttributes attr = {.block_size = 2};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateDepthToSpace(op_def, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 2, 2, 4), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps),
                        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                         10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f}));
  return absl::OkStatus();
}

absl::Status DepthToSpaceFrom1x2x2x16To1x4x4x4Test(
    TestExecutionEnvironment& env, DataType data_type,
    TensorStorageType storage) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 16);
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = static_cast<float>(i + 1);
  }
  std::vector<float> ref(src_tensor.data.size());
  const int block_size = 2;
  for (int i = 0; i < ref.size(); ++i) {
    int dst_ch = i % 4;
    int dst_x = (i / 4) % 4;
    int dst_y = i / 16;
    int block_x = dst_x % block_size;
    int src_x = dst_x / block_size;
    int block_y = dst_y % block_size;
    int src_y = dst_y / block_size;
    int block_id = block_y * block_size + block_x;
    int src_ch = block_id * 4 + dst_ch;
    int src_linear = (src_y * 2 + src_x) * 16 + src_ch;
    ref[(dst_y * 4 + dst_x) * 4 + dst_ch] = src_linear + 1;
  }
  const SpaceToDepthAttributes attr = {.block_size = block_size};

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateDepthToSpace(op_def, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 4, 4, 4), &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(kEps), ref));
  return absl::OkStatus();
}

}  // namespace ml_drift
