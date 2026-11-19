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

#include "ml_drift/common/kernels/tests/random_philox_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/random_philox.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

absl::Status RandomNormalPhiloxTest(TestExecutionEnvironment& env,
                                    DataType data_type,
                                    TensorStorageType storage) {
  const float eps = data_type == DataType::kFloat32 ? 1e-6f : 1e-2f;
  OperationDef op_def;
  op_def.dst_tensors.push_back({data_type, storage, Layout::kHWC});
  TensorFloat32 dst_tensor;
  GPUOperation operation = CreateRandomNormalPhilox(env.GetGpuInfo(), op_def);
  ABSL_RETURN_IF_ERROR(operation.args_.SetInt("seed", 1));
  ABSL_RETURN_IF_ERROR(operation.args_.SetInt("seed2", 1));
  std::vector<TensorFloat32> src_cpu = {};
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_cpu, std::make_unique<GPUOperation>(std::move(operation)),
      BHWC(1, 1, 1, 12), &dst_tensor));
  //  tf.random.stateless_normal(shape=[12], seed=(1, 1), mean=0.0,
  //  stddev=1.0, alg='philox')
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(eps),
                        {1.6652163f, 1.366376f, 0.7786316f, 0.9834321f,
                         1.6551187f, -0.6363001f, -0.4229284f, 0.63195646f,
                         0.6605189f, -0.6906152f, 3.1515226f, 1.970373f}));
  return absl::OkStatus();
}

}  // namespace ml_drift
