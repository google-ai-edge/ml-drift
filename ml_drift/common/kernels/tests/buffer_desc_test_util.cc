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

#include "ml_drift/common/kernels/tests/buffer_desc_test_util.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

absl::Status ReadAsI16SelectorTest(TestExecutionEnvironment& env) {
  Tensor<BHWC, DataType::kInt32> src;
  src.shape = BHWC(1, 1, 1, 4);
  src.data = {0x0001ffff, 0x0002fffe, 0x0003fffd, 0x0004fffc};

  Tensor<BHWC, DataType::kInt32> ref_tensor;
  ref_tensor.shape = BHWC(1, 1, 1, 8);
  ref_tensor.data = {-1, 1, -2, 2, -3, 3, -4, 4};

  // The test build a custom Gpu operation that reads from SRC (INT32) buffer
  // and use `ReadAsI16` to pase each INT32 value as two INT16 values, and then
  // cast the two INT16 values to two INT32 values to write to DST buffer.
  //
  // For example, an INT32 value read from SRC buffer is 0x0001ffff, then it
  // will be parsed as two INT16 values: 0x0001 and 0xffff.
  OperationDef op_def;
  op_def.src_tensors.push_back(
      {DataType::kInt32, TensorStorageType::kBuffer, Layout::kLinear});
  op_def.dst_tensors.push_back(
      {DataType::kInt32, TensorStorageType::kBuffer, Layout::kLinear});
  TensorDescriptor src_0, dst;
  src_0 = op_def.src_tensors[0];
  src_0.UploadData(src);
  dst.SetBHWCShape(BHWC(1, 1, 1, 8));
  GPUOperation operation;
  BufferDescriptor desc;
  desc.element_type = DataType::kInt32;
  desc.element_size = 1;
  operation.AddSrcBuffer("src", desc);
  operation.AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  operation.code_ = R"(
MAIN_FUNCTION($0) {
  int S = ucl::GetGlobalId<2>();
  int buffer_index = S * 4;
  int4 result;
  result.x = args.src.ReadAsI16(buffer_index);
  result.y = args.src.ReadAsI16(buffer_index + 1);
  result.z = args.src.ReadAsI16(buffer_index + 2);
  result.w = args.src.ReadAsI16(buffer_index + 3);
  args.dst_tensor.WriteLinear(result, S);
}
)";
  operation.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  ABSL_RETURN_IF_ERROR(env.ExecuteGPUOperation(
      {&src_0}, {&dst}, std::make_unique<GPUOperation>(std::move(operation))));
  Tensor<BHWC, DataType::kInt32> dst_tensor;
  dst.DownloadData(&dst_tensor);
  EXPECT_EQ(dst_tensor.data, ref_tensor.data);
  return absl::OkStatus();
}

}  // namespace ml_drift
