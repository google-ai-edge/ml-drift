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

#ifndef ML_DRIFT_COMMON_KERNELS_ELEMENTWISE_H_
#define ML_DRIFT_COMMON_KERNELS_ELEMENTWISE_H_

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"

namespace ml_drift {

// Creates a simple elementwise operation with one input, for example
// log, sin, cos, etc.
ElementwiseDescriptor CreateElementwiseOneInput(const GpuInfo& gpu_info,
                                                const OperationType& op_type,
                                                const DataType& data_type);

// Creates a simple elementwise operation with one input, for example
// log, sin, cos, etc.
GPUOperation CreateElementwiseOneInput(const GpuInfo& gpu_info,
                                       const OperationDef& definition,
                                       const OperationType& op_type);

// Creates a simple elementwise operation with one input, for example
// log, sin, cos, etc.
GPUOperation CreateElementwiseOneInput(const GpuInfo& gpu_info,
                                       const TensorDescriptor& src,
                                       const TensorDescriptor& dst,
                                       const OperationType& op_type);

inline BHWDC ExpandToBHWDC(const BHWC& shape) {
  return BHWDC(shape.b, shape.h, shape.w, 1, shape.c);
}

GPUOperation CreateElementwiseOneInputWithBroadcast(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type, const BHWDC& input_shape,
    const BHWDC& output_shape);

inline GPUOperation CreateElementwiseOneInputWithBroadcast(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type, const BHWC& input_shape,
    const BHWC& output_shape) {
  return CreateElementwiseOneInputWithBroadcast(gpu_info, definition, op_type,
                                                ExpandToBHWDC(input_shape),
                                                ExpandToBHWDC(output_shape));
}

// Creates a simple elementwise operation with two inputs (first input is
// runtime tensor and second input is constant or linear/hwc tensor), for
// example sub, div and etc.
GPUOperation CreateElementwise(const GpuInfo& gpu_info,
                               const OperationDef& definition,
                               const OperationType& op_type,
                               const ElementwiseAttributes& attr);

// Creates a simple elementwise operation with two inputs (first input is
// runtime tensor and second input is constant or linear/hwc tensor), for
// example sub, div and etc. Can broadcast input.
GPUOperation CreateElementwiseWithBroadcast(const GpuInfo& gpu_info,
                                            const OperationDef& definition,
                                            const OperationType& op_type,
                                            const ElementwiseAttributes& attr,
                                            const BHWDC& input_shape,
                                            const BHWDC& output_shape);

inline GPUOperation CreateElementwiseWithBroadcast(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type, const ElementwiseAttributes& attr,
    const BHWC& input_shape, const BHWC& output_shape) {
  return CreateElementwiseWithBroadcast(gpu_info, definition, op_type, attr,
                                        ExpandToBHWDC(input_shape),
                                        ExpandToBHWDC(output_shape));
}

// Creates a simple elementwise operation with two inputs (2 runtime tensors),
// for example sub, div and etc.
GPUOperation CreateElementwiseTwoInput(const GpuInfo& gpu_info,
                                       const OperationDef& definition,
                                       const OperationType& op_type,
                                       const BHWDC& second_shape,
                                       const BHWDC& dst_shape);

inline GPUOperation CreateElementwiseTwoInput(const GpuInfo& gpu_info,
                                              const OperationDef& definition,
                                              const OperationType& op_type,
                                              const BHWC& second_shape,
                                              const BHWC& dst_shape) {
  return CreateElementwiseTwoInput(gpu_info, definition, op_type,
                                   ExpandToBHWDC(second_shape),
                                   ExpandToBHWDC(dst_shape));
}

// Creates a simple elementwise operation with two inputs (2 runtime tensors),
// for example sub, div and etc. Can broadcast first and second input
// simultaneously.
GPUOperation CreateElementwiseTwoInputWithBroadcast(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type, const BHWDC& first_input_shape,
    const BHWDC& second_input_shape, const BHWDC& output_shape,
    const ElementwiseAttributes& attr);

inline GPUOperation CreateElementwiseTwoInputWithBroadcast(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OperationType& op_type, const BHWC& first_input_shape,
    const BHWC& second_input_shape, const BHWC& output_shape,
    const ElementwiseAttributes& attr) {
  return CreateElementwiseTwoInputWithBroadcast(
      gpu_info, definition, op_type, ExpandToBHWDC(first_input_shape),
      ExpandToBHWDC(second_input_shape), ExpandToBHWDC(output_shape), attr);
}

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_ELEMENTWISE_H_
