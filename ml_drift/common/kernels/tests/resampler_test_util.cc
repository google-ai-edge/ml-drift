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

#include "ml_drift/common/kernels/tests/resampler_test_util.h"

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/resampler.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

absl::Status ResamplerIdentityTest(TestExecutionEnvironment& env,
                                   DataType data_type,
                                   TensorStorageType storage,
                                   const BHWC& shape) {
  TensorFloat32 src_tensor;
  src_tensor.shape = shape;
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = std::sin(i);
  }
  TensorFloat32 warp_tensor;
  warp_tensor.shape = BHWC(1, shape.h, shape.w, 2);
  warp_tensor.data.resize(warp_tensor.shape.DimensionsProduct());
  for (int y = 0; y < shape.h; ++y) {
    for (int x = 0; x < shape.w; ++x) {
      warp_tensor.data[(y * shape.w + x) * 2 + 0] = x;
      warp_tensor.data[(y * shape.w + x) * 2 + 1] = y;
    }
  }

  const float eps = data_type == DataType::FLOAT32 ? 1e-6f : 1e-3f;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateResampler(env.GetGpuInfo(), op_def);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {src_tensor, warp_tensor},
      std::make_unique<GPUOperation>(std::move(operation)), src_tensor.shape,
      &dst_tensor));
  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(eps), src_tensor.data));
  return absl::OkStatus();
}

}  // namespace ml_drift
