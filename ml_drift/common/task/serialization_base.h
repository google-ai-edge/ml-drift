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

#ifndef ML_DRIFT_COMMON_TASK_SERIALIZATION_BASE_H_
#define ML_DRIFT_COMMON_TASK_SERIALIZATION_BASE_H_

#include "absl/status/status.h"
#include "flatbuffers/buffer.h"
#include "flatbuffers/flatbuffer_builder.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/serialization_base_generated.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

flatbuffers::Offset<data::Int2> Encode(const int2& v,
                                       flatbuffers::FlatBufferBuilder* builder);

flatbuffers::Offset<data::Int3> Encode(const int3& v,
                                       flatbuffers::FlatBufferBuilder* builder);

flatbuffers::Offset<data::TensorDescriptor> Encode(
    const TensorDescriptor& desc, flatbuffers::FlatBufferBuilder* builder);
absl::Status Decode(const data::TensorDescriptor* fb_desc,
                    TensorDescriptor* desc);

flatbuffers::Offset<data::GPUOperation> Encode(
    const GPUOperation& op, flatbuffers::FlatBufferBuilder* builder);
absl::Status Decode(const data::GPUOperation* fb_op, GPUOperation* op);

DataType ToEnum(data::DataType type);
TensorStorageType ToEnum(data::TensorStorageType type);
Layout ToEnum(data::Layout type);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_TASK_SERIALIZATION_BASE_H_
