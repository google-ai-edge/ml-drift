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

#ifndef ML_DRIFT_COMMON_KERNELS_SPECIAL_CONV_SOFTMAX_CONV_H_
#define ML_DRIFT_COMMON_KERNELS_SPECIAL_CONV_SOFTMAX_CONV_H_

#include <vector>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/wave_memory_util.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class ConvSoftmaxConv : public GPUOperation {
 public:
  struct WeightsDesc {
    // conv weights can be constant or runtime
    // if constant, weights0 and weights1 must be initialized
    // if non constant, src_ch, interm_ch and dst_ch must be initialized
    bool constant = true;
    Tensor<OHWI, DataType::kFloat32>* weights0 = nullptr;
    Tensor<OHWI, DataType::kFloat32>* weights1 = nullptr;
    int src_ch = 0;
    int interm_ch = 0;
    int dst_ch = 0;
  };
  enum class WeightsUploadType {
    kGlobal,
    kGlobalToLocalByThreads,
    kWaveMemory,
    kIntelWaveMatmul,
  };
  struct Params {
    // applicable for kWaveMemory only
    int GetDataSizePerWave() {
      const int flt_size = SizeOf(weights_data_type);
      // Base CONV block:
      //   s_out += s_in.x * wave_weight_flt16.s0123;
      //   s_out += s_in.y * wave_weight_flt16.s4567;
      //   s_out += s_in.z * wave_weight_flt16.s89ab;
      //   s_out += s_in.w * wave_weight_flt16.scdef;
      const int base_conv_blocks = block_slices_out;
      const int flt_values_per_base_block = 16;
      return base_conv_blocks * flt_values_per_base_block * flt_size;
    }

    BufferDescriptor GetWeightsBufferDesc(const GpuInfo& gpu_info) const {
      BufferDescriptor desc;
      desc.element_type = weights_data_type;
      desc.element_size = 4;
      if (weights_upload_type == WeightsUploadType::kWaveMemory) {
        desc = GetBufferDescForWaveMemoryUpload(gpu_info, weights_data_type);
      } else {
        desc.memory_type = MemoryType::kGlobal;
      }
      return desc;
    }

    WeightsUploadType weights_upload_type;
    DataType weights_data_type;  // used for weights and biases
    bool dot_conv = false;
    int block_x = 1;
    int block_interm_slices = 1;
    int block_slices_in = 1;
    int block_slices_out = 1;
    int block_slices_size = 1;
    int wave_size = 1;  // for kWaveMemory / kIntelWaveMatmul
  };

  ConvSoftmaxConv() = default;
  int3 GetGridSize() const override;
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override;

  // Move only
  ConvSoftmaxConv(ConvSoftmaxConv&& kernel) = default;
  ConvSoftmaxConv& operator=(ConvSoftmaxConv&& kernel) = default;
  ConvSoftmaxConv(const ConvSoftmaxConv&) = delete;
  ConvSoftmaxConv& operator=(const ConvSoftmaxConv&) = delete;

 private:
  friend ConvSoftmaxConv CreateConvSoftmaxConv(
      const GpuInfo& gpu_info, const OperationDef& definition,
      CalculationsPrecision precision,
      const ConvSoftmaxConv::WeightsDesc& weights_desc);

  ConvSoftmaxConv(CalculationsPrecision precision, const Params& params);

  Params params_;
};

// returns vector with size = 2, 2 WeightsDescription
std::vector<WeightsDescription> GetWeightsDescsForConvSoftmaxConv(
    const GpuInfo& gpu_info, CalculationsPrecision precision, int src_ch,
    int interm_ch, int dst_ch);

bool IsConvSoftmaxConvSupported(const GpuInfo& gpu_info,
                                CalculationsPrecision precision, int src_ch,
                                int interm_ch, int dst_ch);

ConvSoftmaxConv CreateConvSoftmaxConv(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const ConvSoftmaxConv::WeightsDesc& weights_desc);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_SPECIAL_CONV_SOFTMAX_CONV_H_
