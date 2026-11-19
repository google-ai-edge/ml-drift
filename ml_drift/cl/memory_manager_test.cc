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

#include "ml_drift/cl/memory_manager.h"

#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/status_matchers.h"
#include "ml_drift/cl/cl_test.h"
#include "ml_drift/cl/tensor.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"

namespace ml_drift {
namespace cl {
namespace {

TEST_F(OpenCLTest, MemoryManagerSharingSavesMemory) {
  GpuModel model;

  // Define an intermediate tensor of size 1024 floats (4KB)
  TensorDescriptor desc(DataType::kFloat32, TensorStorageType::kBuffer,
                        Layout::kBHWC);
  desc.SetBHWCShape(BHWC(1, 1, 1, 1024));

  model.tensors[1] = desc;

  auto& node = model.nodes.emplace_back();
  node.inputs = {1};

  MemoryManager memory_manager;
  ExternalTensorsInfo ext_tensors;

  // Allocate memory for Model 1
  std::vector<std::unique_ptr<Tensor>> temp_tensors_1;
  ABSL_ASSERT_OK(memory_manager
                .AllocateMemory(model, env_.device().GetInfo(), ext_tensors,
                                temp_tensors_1, &env_.context())
                .status());

  // Peak intermediate memory should be exactly 4KB (4096 bytes)
  EXPECT_EQ(memory_manager.GetSizeOfMemoryAllocatedForIntermediateTensors(),
            4096);

  // Allocate memory for Model 2 (sequential run)
  std::vector<std::unique_ptr<Tensor>> temp_tensors_2;
  ABSL_ASSERT_OK(memory_manager
                .AllocateMemory(model, env_.device().GetInfo(), ext_tensors,
                                temp_tensors_2, &env_.context())
                .status());

  // Since Model 2 runs sequentially, it should reuse Model 1's intermediate
  // buffer. The total allocated intermediate memory should STILL be 4KB.
  EXPECT_EQ(memory_manager.GetSizeOfMemoryAllocatedForIntermediateTensors(),
            4096);
}

}  // namespace
}  // namespace cl
}  // namespace ml_drift
