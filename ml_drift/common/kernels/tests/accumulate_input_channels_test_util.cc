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

#include "ml_drift/common/kernels/tests/accumulate_input_channels_test_util.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/accumulate_input_channels.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

absl::Status AccumulateInputChannelsInt8ToInt32Test(
    TestExecutionEnvironment& env, const TensorStorageType& dst_storage,
    const OHWI& weights_shape) {
  TensorDescriptor src_raw_ohwi = TensorDescriptor(
      DataType::INT8, TensorStorageType::BUFFER, Layout::LINEAR);
  src_raw_ohwi.SetBHWCShape(BHWC(1, 1, 1, weights_shape.i * weights_shape.o));
  TensorDescriptor dst_descriptor =
      TensorDescriptor(DataType::INT32, dst_storage, Layout::LINEAR);
  dst_descriptor.SetBHWCShape(BHWC(1, 1, 1, weights_shape.o));

  OperationDef op_def;
  op_def.src_tensors.push_back(src_raw_ohwi);
  op_def.dst_tensors.push_back(dst_descriptor);
  GPUOperation operation =
      CreateAccumulateInputChannels(op_def, weights_shape, DataType::INT8);

  // Upload input data.
  std::vector<int8_t> src_raw(weights_shape.DimensionsProduct());
  std::vector<int> reference_output(weights_shape.o);
  for (int i = 0; i < weights_shape.o; ++i) {
    int sum = 0;
    for (int j = 0; j < weights_shape.i; ++j) {
      int8_t value = i % 256 - 128;
      sum += value;
      src_raw[i * weights_shape.i + j] = value;
    }
    reference_output[i] = sum;
  }
  src_raw_ohwi.UploadDataRaw(absl::MakeConstSpan(src_raw));

  std::vector<TensorDescriptor*> src_cpu_desc_ptrs{&src_raw_ohwi};
  std::vector<TensorDescriptor*> dst_cpu_desc_ptrs{&dst_descriptor};
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_cpu_desc_ptrs, dst_cpu_desc_ptrs,
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::INT32> dst_tensor;
  dst_descriptor.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, reference_output);
  return absl::OkStatus();
}

absl::Status AccumulateInputChannelsInt4ToInt32Test(
    TestExecutionEnvironment& env, const TensorStorageType& dst_storage,
    const OHWI& weights_shape) {
  const TensorStorageType src_storage = TensorStorageType::BUFFER;
  const int int4_elements_per_int8_element =
      SizeInBitsOf(DataType::INT8) / SizeInBitsOf(DataType::INT4);
  const int int8_elements_per_int32_element =
      SizeInBitsOf(DataType::INT32) / SizeInBitsOf(DataType::INT8);
  const int src_size_packed = DivideRoundUp(
      weights_shape.DimensionsProduct(), int4_elements_per_int8_element);

  TensorDescriptor src_descriptor =
      TensorDescriptor(DataType::INT32, src_storage, Layout::LINEAR);
  src_descriptor.SetBHWCShape(BHWC(1, 1, 1, src_size_packed));
  TensorDescriptor dst_descriptor =
      TensorDescriptor(DataType::INT32, dst_storage, Layout::LINEAR);
  dst_descriptor.SetBHWCShape(BHWC(1, 1, 1, weights_shape.o));

  OperationDef op_def;
  op_def.src_tensors.push_back(src_descriptor);
  op_def.dst_tensors.push_back(dst_descriptor);
  GPUOperation operation =
      CreateAccumulateInputChannels(op_def, weights_shape, DataType::INT4);

  // Upload input data.
  //
  // In case of OHWI(4, 1, 1, 8):
  //   the input int4 data is:
  //     Output channels | Input channels-->
  //       |     -8 -7 -6 -5 -4 -3 -2 -1
  //       |      0  1  2  3  4  5  6  7
  //       v     -8 -7 -6 -5 -4 -3 -2 -1
  //              0  1  2  3  4  5  6  7
  //
  //   the expected output is:
  //     Channels-->
  //              -36 28 -36 28
  //
  std::vector<int8_t> src_raw(src_size_packed);
  std::vector<int> reference_output(weights_shape.o);
  for (int i = 0; i < weights_shape.o; ++i) {
    int sum = 0;
    for (int j = 0; j < weights_shape.i; j += int4_elements_per_int8_element) {
      int8_t lower_int4 = (i * weights_shape.i + j) % 16 - 8;
      int8_t upper_int4 = (i * weights_shape.i + j + 1) % 16 - 8;
      sum += lower_int4 + upper_int4;
      src_raw[(i * weights_shape.i + j) / int4_elements_per_int8_element] =
          (upper_int4 << 4) | (lower_int4 & 0x0F);
    }
    reference_output[i] = sum;
  }
  src_descriptor.SetBHWCShape(
      BHWC(1, 1, 1,
           DivideRoundUp(src_size_packed, int8_elements_per_int32_element)));
  src_descriptor.UploadDataRaw(absl::MakeConstSpan(src_raw));

  std::vector<TensorDescriptor*> src_cpu_desc_ptrs{&src_descriptor};
  std::vector<TensorDescriptor*> dst_cpu_desc_ptrs{&dst_descriptor};
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_cpu_desc_ptrs, dst_cpu_desc_ptrs,
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::INT32> dst_tensor;
  dst_descriptor.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, reference_output);
  return absl::OkStatus();
}

absl::Status AccumulateInputChannelsInt2ToInt32Test(
    TestExecutionEnvironment& env, const TensorStorageType& dst_storage,
    const OHWI& weights_shape) {
  const TensorStorageType src_storage = TensorStorageType::BUFFER;
  const int int2_elements_per_int8_element =
      SizeInBitsOf(DataType::INT8) / SizeInBitsOf(DataType::INT2);
  const int src_size_packed = DivideRoundUp(
      weights_shape.DimensionsProduct(), int2_elements_per_int8_element);

  TensorDescriptor src_descriptor =
      TensorDescriptor(DataType::INT8, src_storage, Layout::LINEAR);
  src_descriptor.SetBHWCShape(BHWC(1, 1, 1, src_size_packed));
  TensorDescriptor dst_descriptor =
      TensorDescriptor(DataType::INT32, dst_storage, Layout::LINEAR);
  dst_descriptor.SetBHWCShape(BHWC(1, 1, 1, weights_shape.o));

  OperationDef op_def;
  op_def.src_tensors.push_back(src_descriptor);
  op_def.dst_tensors.push_back(dst_descriptor);
  GPUOperation operation =
      CreateAccumulateInputChannels(op_def, weights_shape, DataType::INT2);

  // Upload input data.
  //
  // In case of OHWI(4, 1, 1, 8):
  //   the input int4 data is:
  //     Output channels | Input channels-->
  //       |     -2 -1  0  1 -2 -1  0  1
  //       |     -2 -1  0  1 -2 -1  0  1
  //       |     -2 -1  0  1 -2 -1  0  1
  //       |     -2 -1  0  1 -2 -1  0  1
  //
  //   the expected output is:
  //     Channels-->
  //              -4 -4 -4 -4
  //
  std::vector<int8_t> src_raw(src_size_packed);
  std::vector<int> reference_output(weights_shape.o);
  for (int i = 0; i < weights_shape.o; ++i) {
    int sum = 0;
    for (int j = 0; j < weights_shape.i; j += int2_elements_per_int8_element) {
      int8_t int2_val_0 = (i * weights_shape.i + j) % 4 - 2;
      int8_t int2_val_1 = (i * weights_shape.i + j + 1) % 4 - 2;
      int8_t int2_val_2 = (i * weights_shape.i + j + 2) % 4 - 2;
      int8_t int2_val_3 = (i * weights_shape.i + j + 3) % 4 - 2;
      sum += int2_val_0 + int2_val_1 + int2_val_2 + int2_val_3;
      src_raw[(i * weights_shape.i + j) / int2_elements_per_int8_element] =
          (int2_val_3 << 6 & 0xC0) | (int2_val_2 << 4 & 0x30) |
          (int2_val_1 << 2 & 0x0C) | (int2_val_0 & 0x03);
    }
    reference_output[i] = sum;
  }
  src_descriptor.SetBHWCShape(BHWC(1, 1, 1, src_size_packed));
  src_descriptor.UploadDataRaw(absl::MakeConstSpan(src_raw));

  std::vector<TensorDescriptor*> src_cpu_desc_ptrs{&src_descriptor};
  std::vector<TensorDescriptor*> dst_cpu_desc_ptrs{&dst_descriptor};
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_cpu_desc_ptrs, dst_cpu_desc_ptrs,
      std::make_unique<GPUOperation>(std::move(operation))));

  Tensor<BHWC, DataType::INT32> dst_tensor;
  dst_descriptor.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, reference_output);
  return absl::OkStatus();
}

}  // namespace ml_drift
