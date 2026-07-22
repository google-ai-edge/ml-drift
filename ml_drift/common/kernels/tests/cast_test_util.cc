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

#include "ml_drift/common/kernels/tests/cast_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/cast.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

absl::Status CastBaseTest(TestExecutionEnvironment& env,
                          TensorStorageType src_storage,
                          TensorStorageType dst_storage) {
  Tensor<BHWC, DataType::FLOAT32> src;
  src.shape = BHWC(1, 2, 1, 2);
  src.data = {0.0f, -1.3f, -7.4f, 12.45f};

  Tensor<BHWC, DataType::INT32> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {0, -1, -7, 12};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::FLOAT32, src_storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::INT32, dst_storage, Layout::HWC});
  TensorDescriptor src_desc, dst_desc;
  src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src);
  dst_desc.SetBHWCShape(BHWC(1, 2, 1, 2));
  GPUOperation operation = CreateCast(op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::INT32> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

absl::Status CastToBoolTest(TestExecutionEnvironment& env,
                            TensorStorageType src_storage,
                            TensorStorageType dst_storage) {
  Tensor<BHWC, DataType::FLOAT32> src;
  src.shape = BHWC(1, 2, 1, 2);
  src.data = {0.0f, -1.3f, -7.4f, 12.45f};

  Tensor<BHWC, DataType::BOOL> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {false, true, true, true};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::FLOAT32, src_storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::BOOL, dst_storage, Layout::HWC});
  TensorDescriptor src_desc, dst_desc;
  src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src);
  dst_desc.SetBHWCShape(BHWC(1, 2, 1, 2));
  GPUOperation operation = CreateCast(op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::BOOL> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

absl::Status CastFromBoolTest(TestExecutionEnvironment& env,
                              TensorStorageType src_storage,
                              TensorStorageType dst_storage) {
  Tensor<BHWC, DataType::BOOL> src;
  src.shape = BHWC(1, 2, 1, 2);
  src.data = {false, true, true, true};

  Tensor<BHWC, DataType::FLOAT32> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {0.0, 1.0, 1.0, 1.0};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::BOOL, src_storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::FLOAT32, dst_storage, Layout::HWC});
  TensorDescriptor src_desc, dst_desc;
  src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src);
  dst_desc.SetBHWCShape(BHWC(1, 2, 1, 2));
  GPUOperation operation = CreateCast(op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::FLOAT32> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

absl::Status CastToBfloatTest(TestExecutionEnvironment& env,
                              TensorStorageType src_storage,
                              TensorStorageType dst_storage) {
  Tensor<BHWC, DataType::BFLOAT16> src;
  src.shape = BHWC(1, 2, 1, 2);
  src.data = ml_drift::FloatToBFloat({0, -1, -7, 12});

  Tensor<BHWC, DataType::INT32> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = {0, -1, -7, 12};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::BFLOAT16, src_storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::INT32, dst_storage, Layout::HWC});
  TensorDescriptor src_desc, dst_desc;
  src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src);
  dst_desc.SetBHWCShape(BHWC(1, 2, 1, 2));
  GPUOperation operation = CreateCast(op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::INT32> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

absl::Status CastFromBfloatTest(TestExecutionEnvironment& env,
                                TensorStorageType src_storage,
                                TensorStorageType dst_storage) {
  Tensor<BHWC, DataType::INT32> src;
  src.shape = BHWC(1, 2, 1, 2);
  src.data = {0, -1, -7, 12};

  Tensor<BHWC, DataType::BFLOAT16> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 2);
  ref_tensor.data = ml_drift::FloatToBFloat({0, -1, -7, 12});

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::INT32, src_storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::BFLOAT16, dst_storage, Layout::HWC});
  TensorDescriptor src_desc, dst_desc;
  src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src);
  dst_desc.SetBHWCShape(BHWC(1, 2, 1, 2));
  GPUOperation operation = CreateCast(op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::BFLOAT16> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

}  // namespace ml_drift
