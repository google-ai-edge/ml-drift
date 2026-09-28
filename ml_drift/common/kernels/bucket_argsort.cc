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

#include "ml_drift/common/kernels/bucket_argsort.h"

#include <cstddef>
#include <string>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"

namespace ml_drift {

std::string GetBucketArgsortCode(size_t num_buckets, size_t num_elements) {
  std::string code = R"(
MAIN_FUNCTION($0) {
  int local_id = ucl::GetLocalId<0>();
  int num_elements = args.src_tensor.Channels();

  __local int hist[NUM_BUCKETS];
  __local int offsets[NUM_BUCKETS];

  int count = 0;
  int cur_id = -1;
  for (int i = 0; i < num_elements; ++i) {
    args.src_tensor.ReadPerChannel<int>(cur_id, 0, 0, i, 0);
    if (cur_id == local_id) {
      count++;
    }
  }
  hist[local_id] = count;
  ucl::SyncThreads<WorkGroup, Local>();

  if (local_id == 0) {
    int current_offset = 0;
    for (int b = 0; b < NUM_BUCKETS; ++b) {
      offsets[b] = current_offset;
      current_offset += hist[b];
    }
  }
  ucl::SyncThreads<WorkGroup, Local>();

  int output_pos = offsets[local_id];
  for (int i = 0; i < num_elements; ++i) {
    args.src_tensor.ReadPerChannel<int>(cur_id, 0, 0, i, 0);
    if (cur_id == local_id) {
      args.dst_tensor.Write(i, output_pos);
      output_pos++;
    }
  }
}
)";
  absl::StrReplaceAll({{"NUM_BUCKETS", std::to_string(num_buckets)},
                       {"NUM_ELEMENTS", std::to_string(num_elements)}}, &code);
  return code;
}

BucketArgsortOp CreateBucketArgsort(const OperationDef& op_def,
                                    const GpuInfo& gpu_info,
                                    size_t num_buckets, size_t num_elements) {
  ABSL_CHECK(op_def.src_tensors[0].GetDataType() == DataType::kInt32 &&
             op_def.dst_tensors[0].GetDataType() == DataType::kInt32 &&
             op_def.src_tensors[0].GetBHWCShape().c ==
                 op_def.src_tensors[0].GetBHWCShape().DimensionsProduct() &&
             op_def.dst_tensors[0].GetBHWCShape().c ==
                 op_def.dst_tensors[0].GetBHWCShape().DimensionsProduct())
      << "Only supported for flat tensors with int32 data type.";
  ABSL_LOG_IF(WARNING, num_buckets > 256)
      << "Bucket sorting algorithm is inefficient for large number of buckets.";
  ABSL_CHECK(op_def.dst_tensors[0].GetStorageType() ==
             TensorStorageType::kBuffer)
      << "Only supported for buffer storage type.";

  BucketArgsortOp op(num_buckets);
  op.AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  BufferDescriptor dst_desc;
  dst_desc.element_type = DataType::kInt32;
  dst_desc.element_size = 1;
  op.AddDstBuffer("dst_tensor", dst_desc);
  op.code_ = GetBucketArgsortCode(num_buckets, num_elements);
  return op;
}

}  // namespace ml_drift
