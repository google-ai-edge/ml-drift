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

#ifndef ML_DRIFT_COMMON_KERNELS_FULLY_CONNECTED_UTIL_H_
#define ML_DRIFT_COMMON_KERNELS_FULLY_CONNECTED_UTIL_H_

#include <string>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/types.h"

namespace ml_drift {
namespace fc {

void AddRuntimeParam(const ConvRuntimeCheckDesc& runtime_check,
                     GPUOperation* op);

std::string ReadScaleZeroPointBlock(const std::string& o_slice,
                                    const OHWI& scale_zp_shape,
                                    bool has_zero_point, DataType weights_type);
std::string ReadScaleZeroPointLinear(const std::string& o_slice,
                                     const OHWI& scale_zp_shape,
                                     bool has_zero_point,
                                     DataType weights_type);
std::string ReadScaleZeroPointScalar(bool has_zero_point,
                                     DataType weights_type);

std::string WeightsScaleAddBias(const std::string& w_scale,
                                const std::string& w_bias, bool isI4O4,
                                bool use_fma);

std::string AccumulateFloat(const std::string& r_name,
                            const std::string& src_name,
                            CalculationsPrecision precision, bool isI4O4,
                            bool use_fma);

std::string AccumulateUint(const std::string& r_name,
                           const std::string& src_name,
                           const GpuInfo& gpu_info);
std::string AdjustUintSum(const std::string& r_name, DataType weights_type);

// initialize: int dst_end_slice
std::string GetActiveDstSlices(const ConvRuntimeCheckDesc& runtime_check);
// initialize: int ring_o_offset, int ring_size
std::string GetRingOOffset(const ConvRuntimeCheckDesc& runtime_check);
// initialize: int ring_i_offset, int ring_size
std::string GetRingIOffset(const ConvRuntimeCheckDesc& runtime_check);
// initialize: int dst_w, int w_group_size, int w_group_offset
std::string GetPackedGroupsParams(const ConvRuntimeCheckDesc& runtime_check,
                                  int dim_id, int block_size);
// initialize: int w_batch_id, optionally dst_h
std::string GetWeightsBatchId(int runtime_batch_ids = 0);

int GetLocalBatchSize(const GpuInfo& gpu_info, CalculationsPrecision precision,
                      int block_spatial, const int3& work_group_size);

std::string GetReductionCode(int first, int last, int local_batch_size,
                             const std::string& thread_id,
                             const std::string& reduction_local_id,
                             const std::string& reduction_size,
                             const std::string& mem_name);

std::string GenerateDstWrite(const BHWC& block_size,
                             const ConvRuntimeCheckDesc& runtime_check,
                             bool has_bias, bool batched_weights,
                             int runtime_batch_ids);

bool IsQuantized(DataType weights_type);
bool IsScalarQuantized(DataType weights_type, const OHWI& scale_zp_shape);
bool IsLinearQuantized(DataType weights_type, const OHWI& scale_zp_shape);
bool IsBlockQuantized(DataType weights_type, const OHWI& scale_zp_shape);

void AddWeightsScaleZeroPointArguments(const ExternalWeights& weights,
                                       GPUOperation* op);
// Adds weights arguments and scale/zero-point arguments to the GPU operation.
void AddWeightsArguments(const ExternalWeights& weights, int vec_size,
                         GPUOperation* op);
void AddSparseWeightsArguments(const ExternalWeights& weights, int vec_size,
                               GPUOperation* op);

int3 GetBlockSpatialCoords(int linear_spatial, const BHWC& shape);
DataType GetDataTypeForWeights(DataType weights_type);
bool UseFMA(const GpuInfo& gpu_info);
BHWC GetBlockSize(const BHWC* dst_shape_ptr, bool batched_weights);

}  // namespace fc
}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_FULLY_CONNECTED_UTIL_H_
