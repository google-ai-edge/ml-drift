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

#include "ml_drift/common/kernels/tests/strided_slice_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/kernels/strided_slice.h"
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

absl::Status StridedSliceTest(TestExecutionEnvironment& env, DataType data_type,
                              TensorStorageType storage) {
  // 2^-7 should be the maximum error for values in range [0.0, 21.4].
  constexpr float kEps = 1.0f / (1 << 7);

  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 3, 2, 4);
  src_tensor.data = {0.1f,  0.2f,  0.3f,  0.4f,  1.1f,  1.2f,  1.3f,  1.4f,
                     10.1f, 10.2f, 10.3f, 10.4f, 11.1f, 11.2f, 11.3f, 11.4f,
                     20.1f, 20.2f, 20.3f, 20.4f, 21.1f, 21.2f, 21.3f, 21.4f};

  SliceAttributes attr;
  attr.starts = BHWC(0, 1, 0, 1);
  attr.ends = BHWC(src_tensor.shape.b, 2, 2, 3);
  attr.strides = BHWC(1, 1, 2, 2);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
  TensorFloat32 dst_tensor;
  StridedSlice operation = CreateStridedSlice(op_def, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<StridedSlice>(std::move(operation)),
      BHWC(1, 2, 1, 2), &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(kEps), {10.2f, 10.4f, 20.2f, 20.4f}));
  return absl::OkStatus();
}

absl::Status StridedSliceBoolTest(TestExecutionEnvironment& env,
                                  TensorStorageType storage) {
  TensorBool src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 2);
  src_tensor.data = {false, true};

  SliceAttributes attr;
  attr.starts = BHWC(0, 0, 0, 0);
  attr.ends = BHWC(1, 1, 1, 1);
  attr.strides = BHWC(1, 1, 1, 1);

  OperationDef op_def;
  op_def.src_tensors.push_back({DataType::BOOL, storage, Layout::HWC});
  op_def.dst_tensors.push_back({DataType::BOOL, storage, Layout::HWC});

  TensorDescriptor src_desc, dst_desc;
  src_desc = op_def.src_tensors[0];
  src_desc.UploadData(src_tensor);
  dst_desc.SetBHWCShape(BHWC(1, 1, 1, 1));

  StridedSlice operation = CreateStridedSlice(op_def, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_desc}, {&dst_desc},
      std::make_unique<StridedSlice>(std::move(operation))));

  std::vector<unsigned char> ref_data = {false};
  TensorBool dst_tensor;
  dst_desc.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_data);
  return absl::OkStatus();
}

absl::Status StridedSliceBigTest(TestExecutionEnvironment& exec_env,
                                 const SliceAttributes& attr,
                                 const BHWC& src_shape, DataType data_type,
                                 TensorStorageType storage) {
  TensorFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  ASSIGN_OR_RETURN(TensorFloat32 dst_ref_tensor,
                   SliceReference(attr, src_tensor));

  OperationDef op_def;
  const Layout layout = src_shape.b > 1 ? Layout::BHWC : Layout::HWC;
  op_def.src_tensors.push_back({data_type, storage, layout});
  op_def.dst_tensors.push_back({data_type, storage, layout});
  StridedSlice operation = CreateStridedSlice(op_def, attr);

  TensorFloat32 dst_tensor;
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      src_tensor, std::make_unique<StridedSlice>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.001f), dst_ref_tensor.data));
  return absl::OkStatus();
}

absl::Status StridedSlice3DTest(TestExecutionEnvironment& env,
                                DataType data_type, TensorStorageType storage) {
  constexpr float kEps = 1.0f / (1 << 7);

  Tensor5DFloat32 src_tensor;
  src_tensor.shape = BHWDC(1, 3, 2, 2, 4);
  src_tensor.data = {0.1f,  0.2f,  0.3f,  0.4f,  1.1f,  1.2f,  1.3f,  1.4f,
                     10.1f, 10.2f, 10.3f, 10.4f, 11.1f, 11.2f, 11.3f, 11.4f,
                     20.1f, 20.2f, 20.3f, 20.4f, 21.1f, 21.2f, 21.3f, 21.4f,
                     30.1f, 30.2f, 30.3f, 30.4f, 31.1f, 31.2f, 31.3f, 31.4f,
                     40.1f, 40.2f, 40.3f, 40.4f, 41.1f, 41.2f, 41.3f, 41.4f,
                     50.1f, 50.2f, 50.3f, 50.4f, 51.1f, 51.2f, 51.3f, 51.4f};

  Slice3DAttributes attr;
  attr.starts = BHWDC(0, 1, 0, 1, 1);
  attr.ends = BHWDC(src_tensor.shape.b, 2, 2, 2, 3);
  attr.strides = BHWDC(1, 1, 2, 1, 2);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWDC});
  Tensor5DFloat32 dst_tensor;
  StridedSlice operation = CreateStridedSlice(op_def, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<StridedSlice>(std::move(operation)),
      BHWDC(1, 1, 1, 1, 1), &dst_tensor));

  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(kEps), {21.2f}));
  return absl::OkStatus();
}

absl::Status StridedSlice3DPaddedGridTest(TestExecutionEnvironment& env,
                                          DataType data_type,
                                          TensorStorageType storage) {
  constexpr float kEps = 1.0f / (1 << 7);

  Tensor5DFloat32 src_tensor;
  src_tensor.shape = BHWDC(1, 7, 1, 3, 1);
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = static_cast<float>(i + 1);
  }

  Slice3DAttributes attr;
  attr.starts = BHWDC(0, 0, 0, 0, 0);
  attr.ends = src_tensor.shape;
  attr.strides = BHWDC(1, 1, 1, 1, 1);

  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, storage, Layout::BHWDC});
  op_def.dst_tensors.push_back({data_type, storage, Layout::BHWDC});
  Tensor5DFloat32 dst_tensor;
  StridedSlice operation = CreateStridedSlice(op_def, attr);
  RETURN_IF_ERROR(env.ExecuteGPUOperation(
      src_tensor, std::make_unique<StridedSlice>(std::move(operation)),
      src_tensor.shape, &dst_tensor));

  EXPECT_THAT(dst_tensor.data, Pointwise(FloatNear(kEps), src_tensor.data));
  return absl::OkStatus();
}

absl::Status StridedSlice3DBigTest(TestExecutionEnvironment& exec_env,
                                   const Slice3DAttributes& attr,
                                   const BHWDC& src_shape, DataType data_type,
                                   TensorStorageType storage) {
  Tensor5DFloat32 src_tensor = MakeSyntheticTensor(src_shape);
  ASSIGN_OR_RETURN(Tensor5DFloat32 dst_ref_tensor,
                   SliceReference(attr, src_tensor));

  OperationDef op_def;
  const Layout layout = Layout::BHWDC;
  op_def.src_tensors.push_back({data_type, storage, layout});
  op_def.dst_tensors.push_back({data_type, storage, layout});
  StridedSlice operation = CreateStridedSlice(op_def, attr);

  Tensor5DFloat32 dst_tensor;
  MLD_EXPECT_OK(exec_env.ExecuteGPUOperation(
      src_tensor, std::make_unique<StridedSlice>(std::move(operation)),
      dst_ref_tensor.shape, &dst_tensor));
  EXPECT_THAT(dst_tensor.data,
              Pointwise(FloatNear(0.001f), dst_ref_tensor.data));
  return absl::OkStatus();
}

}  // namespace ml_drift
