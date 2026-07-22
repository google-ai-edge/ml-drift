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

#include "ml_drift/common/kernels/tests/winograd_test_util.h"

#include <cmath>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/winograd.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_ref_ops.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/util.h"
#include "ml_drift/common/winograd_util.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

absl::Status Winograd3x3ForwardTiledTest(TestExecutionEnvironment& env,
                                         DataType data_type,
                                         TensorStorageType storage,
                                         const BHWC& src_shape, int tile_size) {
  TensorFloat32 src_tensor;
  src_tensor.shape = src_shape;
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = sin(i);
  }
  Padding2D padding;
  padding.prepended = HW(1, 1);
  padding.appended = HW(1, 1);

  TensorFloat32 dst_ref = Winograd3x3ForwardRef(src_tensor, padding, tile_size);

  float eps = data_type == DataType::FLOAT32 ? 1e-5f : 1e-2f;
  if (tile_size == 8) {
    eps = data_type == DataType::FLOAT32 ? 2e-5f : 0.2f;
  }
  if (!env.GetGpuInfo().IsRoundToNearestSupported()) {
    eps *= 4.0f;
  }
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  Winograd3x3TiledXForward operation = CreateWinograd3x3TiledXForward(
      env.GetGpuInfo(), op_def, padding, tile_size);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor,
      std::make_unique<Winograd3x3TiledXForward>(std::move(operation)),
      dst_ref.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref.data));
  return absl::OkStatus();
}

absl::Status Winograd3x3BackwardTiledTest(TestExecutionEnvironment& env,
                                          DataType data_type,
                                          TensorStorageType storage,
                                          const BHWC& initial_shape,
                                          int tile_size) {
  const int tile_size_inner = tile_size - 2;
  const int tile_size_outer = tile_size;
  const int w_tiles = DivideRoundUp(initial_shape.w, tile_size_inner);
  const int h_tiles = DivideRoundUp(initial_shape.h, tile_size_inner);

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(initial_shape.b, tile_size_outer * tile_size_outer,
                          w_tiles * h_tiles, initial_shape.c);
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = sin(i);
  }

  Tensor<Linear, DataType::FLOAT32> biases;
  biases.shape = Linear(initial_shape.c);
  biases.data.resize(biases.shape.DimensionsProduct());
  for (int i = 0; i < biases.data.size(); ++i) {
    biases.data[i] = 0.0f;
  }

  const BHWC dst_shape = initial_shape;
  TensorFloat32 dst_ref =
      Winograd3x3BackwardRef(src_tensor, dst_shape, tile_size);

  float eps = data_type == DataType::FLOAT32 ? 1e-5f : 1e-2f;
  if (tile_size == 8) {
    eps = data_type == DataType::FLOAT32 ? 4e-5f : 0.3f;
  }
  if (!env.GetGpuInfo().IsRoundToNearestSupported()) {
    eps *= 4.0f;
  }
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  Winograd3x3TiledXBackward operation = CreateWinograd3x3TiledXBackward(
      env.GetGpuInfo(), op_def, biases, tile_size);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor,
      std::make_unique<Winograd3x3TiledXBackward>(std::move(operation)),
      dst_ref.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref.data));
  return absl::OkStatus();
}

absl::Status Winograd4x4To36Test(TestExecutionEnvironment& env,
                                 DataType data_type, TensorStorageType storage,
                                 int tile_size) {
  const int tile_size_inner = tile_size - 2;
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, tile_size_inner, tile_size_inner, 1);
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = sin(i);
  }
  Padding2D padding;
  padding.prepended = HW(1, 1);
  padding.appended = HW(1, 1);

  TensorFloat32 dst_ref = Winograd3x3ForwardRef(src_tensor, padding, tile_size);

  float eps = data_type == DataType::FLOAT32 ? 1e-5f : 1e-2f;
  if (!env.GetGpuInfo().IsRoundToNearestSupported()) {
    eps *= 4.0f;
  }
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;

  Winograd4x4To36 operation =
      CreateWinograd4x4To36(op_def, padding, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Winograd4x4To36>(std::move(operation)),
      dst_ref.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref.data));
  return absl::OkStatus();
}

absl::Status Winograd4x4To36BatchTest(TestExecutionEnvironment& env,
                                      DataType data_type,
                                      TensorStorageType storage,
                                      int tile_size) {
  const int tile_size_inner = tile_size - 2;
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(3, tile_size_inner, tile_size_inner, 1);
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = sin(i);
  }
  Padding2D padding;
  padding.prepended = HW(1, 1);
  padding.appended = HW(1, 1);

  TensorFloat32 dst_ref = Winograd3x3ForwardRef(src_tensor, padding, tile_size);

  float eps = data_type == DataType::FLOAT32 ? 1e-5f : 1e-2f;
  if (!env.GetGpuInfo().IsRoundToNearestSupported()) {
    eps *= 4.0f;
  }
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
  TensorFloat32 dst_tensor;

  Winograd4x4To36 operation =
      CreateWinograd4x4To36(op_def, padding, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Winograd4x4To36>(std::move(operation)),
      dst_ref.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref.data));
  return absl::OkStatus();
}

absl::Status Winograd36To4x4Test(TestExecutionEnvironment& env,
                                 DataType data_type, TensorStorageType storage,
                                 int tile_size) {
  const int tile_size_inner = tile_size - 2;
  const int tile_size_outer = tile_size;
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, tile_size_outer * tile_size_outer, 1, 1);
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = sin(i);
  }

  Tensor<Linear, DataType::FLOAT32> biases;
  biases.shape = Linear(1);
  biases.data.resize(biases.shape.DimensionsProduct());
  for (int i = 0; i < biases.data.size(); ++i) {
    biases.data[i] = 0.0f;
  }

  const BHWC dst_shape = BHWC(1, tile_size_inner, tile_size_inner, 1);
  TensorFloat32 dst_ref =
      Winograd3x3BackwardRef(src_tensor, dst_shape, tile_size);

  float eps = data_type == DataType::FLOAT32 ? 1e-5f : 1e-2f;
  if (!env.GetGpuInfo().IsRoundToNearestSupported()) {
    eps *= 4.0f;
  }
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  Winograd36To4x4 operation = CreateWinograd36To4x4(op_def, biases);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<Winograd36To4x4>(std::move(operation)),
      dst_ref.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), dst_ref.data));
  return absl::OkStatus();
}

absl::Status Winograd3x3To36Test(TestExecutionEnvironment& env) {
  const int kTileSize = 6;
  Tensor<OHWI, DataType::FLOAT32> weights =
      MakeSyntheticTensor(OHWI(8, 3, 3, 12));
  Tensor<OHWI, DataType::FLOAT32> wino_weights_ref;
  RearrangeWeightsToWinograd3x3TileNxN(weights, &wino_weights_ref, kTileSize);

  Tensor<Linear, DataType::FLOAT32> src_tensor_gpu;
  src_tensor_gpu.shape = Linear(weights.shape.DimensionsProduct());
  src_tensor_gpu.data = weights.data;
  TensorDescriptor src_desc = CreateConstantLinearTensorDescriptor(
      DataType::FLOAT32, TensorStorageType::BUFFER, src_tensor_gpu);

  // reinterpreting weights as OHWI-BHWC tensor
  TensorFloat32 wino_weights_gpu;
  wino_weights_gpu.shape =
      BHWC(weights.shape.o, kTileSize, kTileSize, weights.shape.i);
  TensorDescriptor dst_desc(DataType::FLOAT32, TensorStorageType::BUFFER,
                            Layout::BHWC);
  dst_desc.SetBHWCShape(wino_weights_gpu.shape);
  Winograd3x3To36 operation = Winograd3x3To36(src_desc, dst_desc);
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<Winograd3x3To36>(std::move(operation))));
  dst_desc.DownloadData(&wino_weights_gpu);
  const float eps = 1e-5f;
  wino_weights_ref.data.resize(wino_weights_ref.shape.DimensionsProduct());
  EXPECT_THAT(wino_weights_gpu.data,
              Pointwise(FloatNear(eps), wino_weights_ref.data));
  return absl::OkStatus();
}

}  // namespace ml_drift
