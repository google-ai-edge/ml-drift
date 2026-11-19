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

#include "ml_drift/common/kernels/tests/bitcast_test_util.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/bitcast.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

absl::Status BitcastGrowTest(TestExecutionEnvironment& env,
                             TensorStorageType src_storage,
                             TensorStorageType dst_storage) {
  Tensor<BHWC, DataType::kUint32> src_tensor;
  src_tensor.shape = BHWC(2, 1, 2, 1);
  // Backwards for endianness.
  src_tensor.data = {0x03020100, 0x07060504, 0x0B0A0908, 0x0F0E0D0C};
  BHWC dst_shape = BHWC(2, 1, 2, 4);

  Tensor<BHWC, DataType::kUint8> ref_tensor;
  ref_tensor.shape = dst_shape;
  ref_tensor.data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::kUint32, src_storage, Layout::kBHWC});
  op_def.dst_tensors.push_back({DataType::kUint8, dst_storage, Layout::kBHWC});
  TensorDescriptor src_desc, dst_desc;
  src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src_tensor);
  dst_desc.SetBHWCShape(dst_shape);
  GPUOperation operation = CreateBitcast(op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::kUint8> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

absl::Status BitcastShrinkTest(TestExecutionEnvironment& env,
                               TensorStorageType src_storage,
                               TensorStorageType dst_storage) {
  Tensor<BHWC, DataType::kUint8> src_tensor;
  src_tensor.shape = BHWC(2, 2, 1, 4);
  src_tensor.data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  BHWC dst_shape = BHWC(2, 2, 1, 1);

  Tensor<BHWC, DataType::kUint32> ref_tensor;
  ref_tensor.shape = dst_shape;
  ref_tensor.data = {0x03020100, 0x07060504, 0x0B0A0908, 0x0F0E0D0C};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::kUint8, src_storage, Layout::kBHWC});
  op_def.dst_tensors.push_back({DataType::kUint32, dst_storage, Layout::kBHWC});
  TensorDescriptor src_desc, dst_desc;
  src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src_tensor);
  dst_desc.SetBHWCShape(dst_shape);
  GPUOperation operation = CreateBitcast(op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::kUint32> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

absl::Status BitcastStableTest(TestExecutionEnvironment& env,
                               TensorStorageType src_storage,
                               TensorStorageType dst_storage) {
  Tensor<BHWC, DataType::kFloat32> src_tensor;
  src_tensor.shape = BHWC(2, 1, 1, 2);
  src_tensor.data = {-5.538, 2.001, 15.8, 111.99};
  BHWC dst_shape = BHWC(2, 1, 1, 2);

  Tensor<BHWC, DataType::kUint32> ref_tensor;
  ref_tensor.shape = dst_shape;
  ref_tensor.data = {0xC0B1374C, 0x40001062, 0x417CCCCD, 0x42dffae1};

  OperationDef op_def;
  op_def.src_tensors.push_back(
      {DataType::kFloat32, src_storage, Layout::kBHWC});
  op_def.dst_tensors.push_back({DataType::kUint32, dst_storage, Layout::kBHWC});
  TensorDescriptor src_desc, dst_desc;
  src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src_tensor);
  dst_desc.SetBHWCShape(dst_shape);
  GPUOperation operation = CreateBitcast(op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::kUint32> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

absl::Status BitcastFromBoolToUcharTest(TestExecutionEnvironment& env,
                                        TensorStorageType src_storage,
                                        TensorStorageType dst_storage) {
  Tensor<BHWC, DataType::kBool> src_tensor;
  src_tensor.shape = BHWC(2, 1, 1, 8);
  src_tensor.data = {true, false, false, false, false, false, false, false,
                     true, true,  true,  true,  true,  true,  true,  true};
  BHWC dst_shape = BHWC(2, 1, 1, 1);

  Tensor<BHWC, DataType::kUint8> ref_tensor;
  ref_tensor.shape = dst_shape;
  ref_tensor.data = {0x01, 0xFF};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::kBool, src_storage, Layout::kBHWC});
  op_def.dst_tensors.push_back({DataType::kUint8, dst_storage, Layout::kBHWC});
  TensorDescriptor src_desc, dst_desc;
  src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src_tensor);
  dst_desc.SetBHWCShape(dst_shape);
  GPUOperation operation = CreateBitcast(op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::kUint8> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

absl::Status BitcastFromBoolToUshortTest(TestExecutionEnvironment& env,
                                         TensorStorageType src_storage,
                                         TensorStorageType dst_storage) {
  Tensor<BHWC, DataType::kBool> src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 16);
  src_tensor.data = {true, false, false, false, false, false, false, false,
                     true, true,  true,  true,  true,  true,  true,  true};
  BHWC dst_shape = BHWC(1, 1, 1, 1);

  Tensor<BHWC, DataType::kUint16> ref_tensor;
  ref_tensor.shape = dst_shape;
  ref_tensor.data = {0xFF01};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::kBool, src_storage, Layout::kBHWC});
  op_def.dst_tensors.push_back({DataType::kUint16, dst_storage, Layout::kBHWC});
  TensorDescriptor src_desc, dst_desc;
  src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src_tensor);
  dst_desc.SetBHWCShape(dst_shape);
  GPUOperation operation = CreateBitcast(op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::kUint16> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

absl::Status BitcastFromBoolToFloatTest(TestExecutionEnvironment& env,
                                        TensorStorageType src_storage,
                                        TensorStorageType dst_storage) {
  Tensor<BHWC, DataType::kBool> src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 32);
  // 0xC0B1374C
  src_tensor.data = {false, false, true,  true,  false, false, true,  false,
                     true,  true,  true,  false, true,  true,  false, false,
                     true,  false, false, false, true,  true,  false, true,
                     false, false, false, false, false, false, true,  true};

  BHWC dst_shape = BHWC(1, 1, 1, 1);

  Tensor<BHWC, DataType::kFloat32> ref_tensor;
  ref_tensor.shape = dst_shape;
  ref_tensor.data = {-5.538};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::kBool, src_storage, Layout::kBHWC});
  op_def.dst_tensors.push_back(
      {DataType::kFloat32, dst_storage, Layout::kBHWC});
  TensorDescriptor src_desc, dst_desc;
  src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src_tensor);
  dst_desc.SetBHWCShape(dst_shape);
  GPUOperation operation = CreateBitcast(op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::kFloat32> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

absl::Status BitcastFromUcharToBoolTest(TestExecutionEnvironment& env,
                                        TensorStorageType src_storage,
                                        TensorStorageType dst_storage) {
  Tensor<BHWC, DataType::kUint8> src_tensor;
  src_tensor.shape = BHWC(2, 1, 1, 1);
  src_tensor.data = {0x80, 0xFF};
  BHWC dst_shape = BHWC(2, 1, 1, 8);

  Tensor<BHWC, DataType::kBool> ref_tensor;
  ref_tensor.shape = dst_shape;
  ref_tensor.data = {false, false, false, false, false, false, false, true,
                     true,  true,  true,  true,  true,  true,  true,  true};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::kUint8, src_storage, Layout::kBHWC});
  op_def.dst_tensors.push_back({DataType::kBool, dst_storage, Layout::kBHWC});
  TensorDescriptor src_desc, dst_desc;
  src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src_tensor);
  dst_desc.SetBHWCShape(dst_shape);
  GPUOperation operation = CreateBitcast(op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::kBool> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

absl::Status BitcastFromShortToBoolTest(TestExecutionEnvironment& env,
                                        TensorStorageType src_storage,
                                        TensorStorageType dst_storage) {
  Tensor<BHWC, DataType::kInt16> src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 1);
  src_tensor.data = {static_cast<int16_t>(0xFF08)};
  BHWC dst_shape = BHWC(1, 1, 1, 16);

  Tensor<BHWC, DataType::kBool> ref_tensor;
  ref_tensor.shape = dst_shape;
  ref_tensor.data = {false, false, false, true, false, false, false, false,
                     true,  true,  true,  true, true,  true,  true,  true};

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::kUint16, src_storage, Layout::kBHWC});
  op_def.dst_tensors.push_back({DataType::kBool, dst_storage, Layout::kBHWC});
  TensorDescriptor src_desc, dst_desc;
  src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src_tensor);
  dst_desc.SetBHWCShape(dst_shape);
  GPUOperation operation = CreateBitcast(op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::kBool> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

absl::Status BitcastFromFloatToBoolTest(TestExecutionEnvironment& env,
                                        TensorStorageType src_storage,
                                        TensorStorageType dst_storage) {
  Tensor<BHWC, DataType::kFloat32> src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 1);
  src_tensor.data = {-5.538};
  // 0xC0B1374C
  BHWC dst_shape = BHWC(1, 1, 1, 32);

  Tensor<BHWC, DataType::kBool> ref_tensor;
  ref_tensor.shape = dst_shape;
  ref_tensor.data = {false, false, true,  true,  false, false, true,  false,
                     true,  true,  true,  false, true,  true,  false, false,
                     true,  false, false, false, true,  true,  false, true,
                     false, false, false, false, false, false, true,  true};

  OperationDef op_def;
  op_def.src_tensors.push_back(
      {DataType::kFloat32, src_storage, Layout::kBHWC});
  op_def.dst_tensors.push_back({DataType::kBool, dst_storage, Layout::kBHWC});
  TensorDescriptor src_desc, dst_desc;
  src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src_tensor);
  dst_desc.SetBHWCShape(dst_shape);
  GPUOperation operation = CreateBitcast(op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::kBool> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

absl::Status BitcastFromFloatToUint8Test(TestExecutionEnvironment& env,
                                         TensorStorageType src_storage,
                                         TensorStorageType dst_storage) {
  Tensor<BHWC, DataType::kFloat32> src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 1);
  src_tensor.data = {-5.538};
  // 0xC0B1374C
  BHWC dst_shape = BHWC(1, 1, 1, 4);

  Tensor<BHWC, DataType::kUint8> ref_tensor;
  ref_tensor.shape = dst_shape;
  ref_tensor.data = {0x4C, 0x37, 0xB1, 0xC0};

  OperationDef op_def;
  op_def.src_tensors.push_back(
      {DataType::kFloat32, src_storage, Layout::kBHWC});
  op_def.dst_tensors.push_back({DataType::kUint8, dst_storage, Layout::kBHWC});
  TensorDescriptor src_desc, dst_desc;
  src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src_tensor);
  dst_desc.SetBHWCShape(dst_shape);
  GPUOperation operation = CreateBitcast(op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::kUint8> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

absl::Status BitcastFromUint8ToFloatTest(TestExecutionEnvironment& env,
                                         TensorStorageType src_storage,
                                         TensorStorageType dst_storage) {
  Tensor<BHWC, DataType::kUint8> src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 4);
  src_tensor.data = {0x4C, 0x37, 0xB1, 0xC0};
  BHWC dst_shape = BHWC(1, 1, 1, 1);

  Tensor<BHWC, DataType::kFloat32> ref_tensor;
  ref_tensor.shape = dst_shape;
  ref_tensor.data = {-5.538};  // 0xC0B1374C

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::kUint8, src_storage, Layout::kBHWC});
  op_def.dst_tensors.push_back(
      {DataType::kFloat32, dst_storage, Layout::kBHWC});
  TensorDescriptor src_desc, dst_desc;
  src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src_tensor);
  dst_desc.SetBHWCShape(dst_shape);
  GPUOperation operation = CreateBitcast(op_def, env.GetGpuInfo());
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::kFloat32> dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

}  // namespace ml_drift
