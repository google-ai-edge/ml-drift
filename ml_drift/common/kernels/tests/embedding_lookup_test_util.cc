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

#include "ml_drift/common/kernels/tests/embedding_lookup_test_util.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "xnnpack.h"  // from @XNNPACK
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/embedding_lookup.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

uint8_t Int4AsUint4(int8_t value) { return value >= 0 ? value : (value + 16); }

absl::Status EmbeddingLookupTest(TestExecutionEnvironment& env,
                                 DataType dst_type, TensorStorageType storage) {
  TensorFloat32 lookup_table;
  lookup_table.shape = BHWC(1, 4, 4, 1);
  lookup_table.data.resize(lookup_table.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < lookup_table.shape.DimensionsProduct(); ++i) {
    lookup_table.data[i] = i + 1;
  }

  TensorInt32 id;
  id.shape = BHWC(1, 1, 1, 1);
  id.data = {1};

  EmbeddingLookupAttributes attr;
  attr.original_weights_shape =
      OHWI(lookup_table.shape.h, 1, 1, lookup_table.shape.w);
  Tensor<OHWI, DataType::FLOAT32> weights_float32;
  weights_float32.data = lookup_table.data;
  weights_float32.shape.h = 1;
  weights_float32.shape.w = 1;
  weights_float32.shape.o = lookup_table.shape.h;
  weights_float32.shape.i = lookup_table.shape.w;
  attr.weights = weights_float32;
  attr.weights_type = EmbeddingLookupAttributes::WeightsType::kFloat32;

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  op_def.dst_tensors.push_back({dst_type, storage, Layout::HWC});
  TensorDescriptor src_0, src_1, dst;
  src_0 = op_def.src_tensors[0];
  src_0.UploadData(id);
  dst.SetBHWCShape(BHWC(1, 1, 1, 4));
  GPUOperation operation =
      CreateEmbeddingLookup(op_def, env.GetGpuInfo(), attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0}, {&dst}, std::make_unique<GPUOperation>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.0f), {5.0f, 6.0f, 7.0f, 8.0f}));
  return absl::OkStatus();
}

absl::Status EmbeddingLookupSeqLen2Test(TestExecutionEnvironment& env,
                                        DataType dst_type,
                                        TensorStorageType storage) {
  TensorFloat32 lookup_table;
  lookup_table.shape = BHWC(1, 4, 4, 1);
  lookup_table.data.resize(lookup_table.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < lookup_table.shape.DimensionsProduct(); ++i) {
    lookup_table.data[i] = i + 1;
  }

  TensorInt32 id;
  id.shape = BHWC(1, 1, 1, 2);
  id.data = {1, 2};

  EmbeddingLookupAttributes attr;
  attr.original_weights_shape =
      OHWI(lookup_table.shape.h, 1, 1, lookup_table.shape.w);
  Tensor<OHWI, DataType::FLOAT32> weights_float32;
  weights_float32.data = lookup_table.data;
  weights_float32.shape.h = 1;
  weights_float32.shape.w = 1;
  weights_float32.shape.o = lookup_table.shape.h;
  weights_float32.shape.i = lookup_table.shape.w;
  attr.weights = weights_float32;
  attr.weights_type = EmbeddingLookupAttributes::WeightsType::kFloat32;

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  op_def.dst_tensors.push_back({dst_type, storage, Layout::HWC});
  TensorDescriptor src, dst;
  src = op_def.src_tensors[0];
  src.UploadData(id);
  src.SetBHWCShape(BHWC(1, 1, 1, 2));
  dst.SetBHWCShape(BHWC(1, 1, 2, 4));
  GPUOperation operation =
      CreateEmbeddingLookup(op_def, env.GetGpuInfo(), attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src}, {&dst}, std::make_unique<GPUOperation>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.0f),
                        {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}));
  return absl::OkStatus();
}

absl::Status EmbeddingLookupInt8Test(TestExecutionEnvironment& env,
                                     DataType dst_type,
                                     TensorStorageType storage) {
  Tensor<BHWC, DataType::INT8> lookup_table;
  lookup_table.shape = BHWC(1, 4, 4, 1);
  lookup_table.data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

  TensorInt32 id;
  id.shape = BHWC(1, 1, 1, 1);
  id.data = {1};

  Tensor<OHWI, DataType::FLOAT32> scale_tensor;
  Tensor<OHWI, DataType::FLOAT32> zero_point_tensor;
  scale_tensor.shape = OHWI(lookup_table.shape.h, 1, 1, 1);
  scale_tensor.data.resize(scale_tensor.shape.DimensionsProduct());
  for (int i = 0; i < scale_tensor.data.size(); ++i) {
    scale_tensor.data[i] = 1;
  }
  zero_point_tensor.shape = OHWI(lookup_table.shape.h, 1, 1, 1);
  zero_point_tensor.data.resize(zero_point_tensor.shape.DimensionsProduct());
  for (int i = 0; i < zero_point_tensor.data.size(); ++i) {
    zero_point_tensor.data[i] = 0;
  }
  EmbeddingLookupAttributes attr;
  attr.weights_scale = scale_tensor;
  attr.weights_zero_point = zero_point_tensor;
  attr.original_weights_shape =
      OHWI(lookup_table.shape.h, 1, 1, lookup_table.shape.w);
  Tensor<OHWI, DataType::INT8> weights_int8;
  weights_int8.data = lookup_table.data;
  weights_int8.shape.h = 1;
  weights_int8.shape.w = 1;
  weights_int8.shape.o = lookup_table.shape.h;
  weights_int8.shape.i = lookup_table.shape.w;
  attr.weights = weights_int8;
  attr.weights_type = EmbeddingLookupAttributes::WeightsType::kInt8;

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  op_def.dst_tensors.push_back({dst_type, storage, Layout::HWC});
  TensorDescriptor src_0, dst;
  src_0 = op_def.src_tensors[0];
  src_0.UploadData(id);
  dst.SetBHWCShape(BHWC(1, 1, 1, 4));
  GPUOperation operation =
      CreateEmbeddingLookup(op_def, env.GetGpuInfo(), attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0}, {&dst}, std::make_unique<GPUOperation>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.0f), {5.0f, 6.0f, 7.0f, 8.0f}));
  return absl::OkStatus();
}

absl::Status EmbeddingLookupInt4Test(TestExecutionEnvironment& env,
                                     DataType dst_type,
                                     TensorStorageType storage) {
  Tensor<BHWC, DataType::INT8> LookupTable;
  LookupTable.shape = BHWC(1, 4, 4, 1);
  LookupTable.data = {0, 1, 2, 3, 4, 5, 6, 7, 0, -1, -2, -3, -4, -5, -6, -7};

  std::vector<uint8_t> packed_data(LookupTable.data.size() / 2);
  for (int i = 0; i < LookupTable.data.size() / 2; ++i) {
    uint8_t part0 = Int4AsUint4(LookupTable.data[i * 2]);
    uint8_t part1 = Int4AsUint4(LookupTable.data[i * 2 + 1]);
    packed_data[i] = (part1 << 4) | part0;
  }

  TensorInt32 Ids;
  Ids.shape = BHWC(1, 1, 1, 1);
  Ids.data = {1};

  Tensor<OHWI, DataType::FLOAT32> scale_tensor;
  Tensor<OHWI, DataType::FLOAT32> zero_point_tensor;
  scale_tensor.shape = OHWI(LookupTable.shape.h, 1, 1, 1);
  scale_tensor.data.resize(scale_tensor.shape.DimensionsProduct());
  for (int i = 0; i < scale_tensor.data.size(); ++i) {
    scale_tensor.data[i] = 1;
  }
  zero_point_tensor.shape = OHWI(LookupTable.shape.h, 1, 1, 1);
  zero_point_tensor.data.resize(zero_point_tensor.shape.DimensionsProduct());
  for (int i = 0; i < zero_point_tensor.data.size(); ++i) {
    zero_point_tensor.data[i] = 0;
  }
  EmbeddingLookupAttributes attr;
  attr.weights_scale = scale_tensor;
  attr.weights_zero_point = zero_point_tensor;
  attr.original_weights_shape =
      OHWI(LookupTable.shape.h, 1, 1, LookupTable.shape.w);
  attr.weights_type = EmbeddingLookupAttributes::WeightsType::kInt4;
  Tensor<OHWI, DataType::UINT8> weights_int4;
  weights_int4.data = packed_data;
  weights_int4.shape.h = 1;
  weights_int4.shape.w = 1;
  weights_int4.shape.o = LookupTable.shape.h;
  weights_int4.shape.i = LookupTable.shape.w;
  attr.weights = weights_int4;

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  op_def.dst_tensors.push_back({dst_type, storage, Layout::HWC});
  TensorDescriptor src_0, dst;
  src_0 = op_def.src_tensors[0];
  src_0.UploadData(Ids);
  dst.SetBHWCShape(BHWC(1, 1, 1, 4));
  GPUOperation operation =
      CreateEmbeddingLookup(op_def, env.GetGpuInfo(), attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0}, {&dst}, std::make_unique<GPUOperation>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.0f), {4.0f, 5.0f, 6.0f, 7.0f}));
  return absl::OkStatus();
}

absl::Status EmbeddingLookupInt4NegativeTest(TestExecutionEnvironment& env,
                                             DataType dst_type,
                                             TensorStorageType storage) {
  Tensor<BHWC, DataType::INT8> LookupTable;
  LookupTable.shape = BHWC(1, 4, 4, 1);
  LookupTable.data = {0, 1, 2, 3, 4, 5, 6, 7, 0, -1, -2, -3, -4, -5, -6, -7};
  std::vector<uint8_t> packed_data(LookupTable.data.size() / 2);
  for (int i = 0; i < LookupTable.data.size() / 2; ++i) {
    uint8_t part0 = Int4AsUint4(LookupTable.data[i * 2]);
    uint8_t part1 = Int4AsUint4(LookupTable.data[i * 2 + 1]);
    packed_data[i] = (part1 << 4) | part0;
  }

  TensorInt32 Ids;
  Ids.shape = BHWC(1, 1, 1, 1);
  Ids.data = {3};

  Tensor<OHWI, DataType::FLOAT32> scale_tensor;
  Tensor<OHWI, DataType::FLOAT32> zero_point_tensor;
  scale_tensor.shape = OHWI(LookupTable.shape.h, 1, 1, 1);
  scale_tensor.data.resize(scale_tensor.shape.DimensionsProduct());
  for (int i = 0; i < scale_tensor.data.size(); ++i) {
    scale_tensor.data[i] = 1;
  }
  zero_point_tensor.shape = OHWI(LookupTable.shape.h, 1, 1, 1);
  zero_point_tensor.data.resize(zero_point_tensor.shape.DimensionsProduct());
  for (int i = 0; i < zero_point_tensor.data.size(); ++i) {
    zero_point_tensor.data[i] = 0;
  }
  EmbeddingLookupAttributes attr;
  attr.weights_scale = scale_tensor;
  attr.weights_zero_point = zero_point_tensor;
  attr.original_weights_shape =
      OHWI(LookupTable.shape.h, 1, 1, LookupTable.shape.w);
  attr.weights_type = EmbeddingLookupAttributes::WeightsType::kInt4;
  Tensor<OHWI, DataType::UINT8> weights_int4;
  weights_int4.data = packed_data;
  weights_int4.shape.h = 1;
  weights_int4.shape.w = 1;
  weights_int4.shape.o = LookupTable.shape.h;
  weights_int4.shape.i = LookupTable.shape.w;
  attr.weights = weights_int4;

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::INT32, storage, Layout::HWC});
  op_def.dst_tensors.push_back({dst_type, storage, Layout::HWC});
  TensorDescriptor src_0, dst;
  src_0 = op_def.src_tensors[0];
  src_0.UploadData(Ids);
  dst.SetBHWCShape(BHWC(1, 1, 1, 4));
  GPUOperation operation =
      CreateEmbeddingLookup(op_def, env.GetGpuInfo(), attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0}, {&dst}, std::make_unique<GPUOperation>(std::move(operation))));
  TensorFloat32 dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.0f), {-4.0f, -5.0f, -6.0f, -7.0f}));
  return absl::OkStatus();
}

}  // namespace ml_drift
