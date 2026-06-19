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

#ifndef ML_DRIFT_COMMON_KERNELS_CONV_GENERIC_H_
#define ML_DRIFT_COMMON_KERNELS_CONV_GENERIC_H_

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class ConvGeneric : public GPUOperation {
 public:
  enum class WeightsUploadType {
    kConstantMemory,
    kGlobalMemory,
    kIntelWave16MatMul,
    kLocalMemoryWGLoad,
    kLocalMemory,
    kTexturesX4,  // 4 textures for weights
    kWaveMemory,
  };
  struct KernelParams {
    int4 block_size;  // WHDS
    bool fixed_work_group_size;
    int3 work_group_size;
    int3 work_group_launch_order;
    bool linear_spatial;  // spatial dimensions are Width/Height/Depth
    bool linear_all;  // linear_spatial & linear_all can not be used together,
                      // linear_all can not be used with WeightsUploadTypes
                      // that use workgroups(subgroups) for
                      // uploading(kLocalMemory for example).
    int src_depth_loop_size;
    bool need_src_loop = true;
    bool need_dst_loop = true;
    bool slices_loop_first =
        true;  // if false, spatial loops will be first(most internal), (false)
               // supported only in specific combinations. If false selected,
               // src_depth_loop_size must be 1 and src_desc.IsLinear must be
               // false.
    bool unroll_x_loop = false;  // applicable with slices_loop_first = false;
    WeightsUploadType weights_upload_type;
    WeightsLayout weights_layout;
    // weights_type and weights_element_size is how we bind weights buffer to
    // kernel. It can differ from ConvParams.weights_data_type. For example when
    // we use int8 weights, we use INT8 as ConvParams.weights_data_type, but we
    // can bind this buffer as uint32x4
    DataType weights_type;     // as kernel argument
    int weights_element_size;  // as kernel argument

    // by default linear all launch threads in batch-spatial-slices order, with
    // this option it will be batch-slice-spatial
    bool linear_all_slices_first = false;

    // Must be in ascending order
    std::vector<int> simd_sizes = {1};

    // for WeightsLayout::kCustomGroups;
    std::vector<std::pair<Axis, int>> group_sizes;

    bool AreWeightsBuffer() const {
      return weights_upload_type != WeightsUploadType::kTexturesX4;
    }
  };
  struct ConvParams {
    WeightsDescription weights_desc;         // only used for ExternalWeights
    OHWI scale_zp_shape = OHWI(1, 1, 1, 1);  // only used for ExternalWeights
    bool has_zero_point = false;             // only used for ExternalWeights
    int src_group_slices = 0;                // only used for ExternalWeights
    TensorDescriptor src_desc;
    PackedType src_packed_type;  // applicable for 8/4 bit convolutions
    CalculationsPrecision precision = CalculationsPrecision::F32;
    int4 stride = int4(1, 1, 1, 1);
    int4 padding_prepended = int4(0, 0, 0, 0);
    int4 padding_appended = int4(0, 0, 0, 0);
    int4 kernel_size = int4(1, 1, 1, 1);
    int4 dilation = int4(1, 1, 1, 1);
    int groups_count = 1;  // convolution groups
    int src_slices = 0;
    int dst_slices = 0;
    DataType weights_data_type;  // used for weights and biases
    bool different_weights_for_height = false;
    bool IsXKernelIs1() const {
      return kernel_size.x == 1 && stride.x == 1 && dilation.x == 1 &&
             padding_prepended.x == 0 && padding_appended.x == 0;
    }
    bool IsYKernelIs1() const {
      return kernel_size.y == 1 && stride.y == 1 && dilation.y == 1 &&
             padding_prepended.y == 0 && padding_appended.y == 0;
    }
    bool IsZKernelIs1() const {
      return kernel_size.z == 1 && stride.z == 1 && dilation.z == 1 &&
             padding_prepended.z == 0 && padding_appended.z == 0;
    }
    bool IsTrivialKernelSize() const {
      return IsXKernelIs1() && IsYKernelIs1() && IsZKernelIs1();
    }
    bool Is8Bit() const {
      return weights_data_type == DataType::INT8 ||
             weights_data_type == DataType::UINT8;
    }
    bool Is4Bit() const {
      return weights_data_type == DataType::INT4 ||
             weights_data_type == DataType::UINT4;
    }
    bool IsSrcPackedW4C4() const {
      return Is8Bit() && (src_packed_type == PackedType::kInt8W4C4 ||
                          src_packed_type == PackedType::kUint8W4C4);
    }
    bool IsSrcPackedC16() const {
      return Is8Bit() && (src_packed_type == PackedType::kInt8C16 ||
                          src_packed_type == PackedType::kUint8C16);
    }
    bool IsSrcPackedC32() const {
      return Is4Bit() && (src_packed_type == PackedType::kInt4C32 ||
                          src_packed_type == PackedType::kUint4C32);
    }
    bool HasGroups() const { return groups_count != 1; }
    bool softmax_input_activation = false;
    bool has_bias = true;
    ConvRuntimeCheckDesc runtime_check;
  };
  ConvGeneric() = default;
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override;
  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;

  WeightsDescription GetWeightsDescription() const {
    if (conv_params_.weights_desc.layout != WeightsLayout::kUnknown) {
      return conv_params_.weights_desc;
    }
    if (kernel_params_.weights_upload_type ==
        ConvGeneric::WeightsUploadType::kIntelWave16MatMul) {
      WeightsDescription desc;
      desc.type = conv_params_.weights_data_type;
      desc.layout = WeightsLayout::kCustomGroups;
      if (conv_params_.Is8Bit()) {
        desc.group_sizes.push_back({Axis::INPUT_CHANNELS, 2});
      } else if (conv_params_.Is4Bit()) {
        desc.group_sizes.push_back({Axis::INPUT_CHANNELS, 4});
      }
      desc.group_sizes.push_back({Axis::OUTPUT_CHANNELS, 4});
      if (conv_params_.Is8Bit()) {
        // keep last block in i2o4i2 layout to work with gpu conversion kernel
        desc.group_sizes.push_back({Axis::INPUT_CHANNELS, 2});
        desc.group_sizes.push_back({Axis::INPUT_CHANNELS, 8});
      } else {
        desc.group_sizes.push_back({Axis::INPUT_CHANNELS, 16});
      }
      desc.group_sizes.push_back(
          {Axis::OUTPUT_CHANNELS, kernel_params_.block_size.w});
      desc.group_sizes.push_back({Axis::INPUT_CHANNELS, 0});
      if (conv_params_.kernel_size.x != 1) {
        desc.group_sizes.push_back({Axis::WIDTH, 0});
      }
      if (conv_params_.kernel_size.y != 1) {
        desc.group_sizes.push_back({Axis::HEIGHT, 0});
      }
      if (conv_params_.different_weights_for_height) {
        desc.group_sizes.push_back({Axis::WIDTH, 0});
        desc.group_sizes.push_back({Axis::HEIGHT, 0});
      }
      if (conv_params_.kernel_size.z != 1) {
        desc.group_sizes.push_back({Axis::DEPTH, 0});
      }
      desc.group_sizes.push_back({Axis::OUTPUT_CHANNELS, 0});
      return desc;
    }
    WeightsDescription desc;
    desc.type = conv_params_.weights_data_type;
    desc.layout = kernel_params_.weights_layout;
    if (kernel_params_.weights_layout == WeightsLayout::kCustomGroups) {
      desc.group_sizes = kernel_params_.group_sizes;
    } else {
      desc.output_group_size = kernel_params_.block_size.w;
    }
    return desc;
  }

  std::string_view GetDebugName() const override { return "conv_generic"; }

  // Move only
  ConvGeneric(ConvGeneric&& operation) = default;
  ConvGeneric& operator=(ConvGeneric&& operation) = default;
  ConvGeneric(const ConvGeneric&) = delete;
  ConvGeneric& operator=(const ConvGeneric&) = delete;

 private:
  ConvGeneric(const OperationDef& definition, CalculationsPrecision precision,
              const Convolution2DAttributes& attr, const GpuInfo& gpu_info,
              const BHWC* dst_shape = nullptr);
  ConvGeneric(const OperationDef& definition, CalculationsPrecision precision,
              const Convolution2DAttributes& attr, const OHWI& weights_shape,
              const GpuInfo& gpu_info, const BHWC* dst_shape = nullptr);
  ConvGeneric(const OperationDef& definition, CalculationsPrecision precision,
              const FullyConnectedAttributes& attr, const GpuInfo& gpu_info,
              const BHWC* dst_shape = nullptr);
  ConvGeneric(const OperationDef& definition, CalculationsPrecision precision,
              const Convolution3DAttributes& attr, const GpuInfo& gpu_info,
              const BHWDC* dst_shape = nullptr);

  void GenerateCode(const OperationDef& definition, const GpuInfo& gpu_info);

  template <DataType T>
  void UploadWeights(const Tensor<OHWI, T>& weights);
  void UploadWeightsI4(const Tensor<OHWI, DataType::INT8>& weights_i4);
  void UploadWeightsI8AsU8(const Tensor<OHWI, DataType::INT8>& weights);

  template <DataType T>
  void UploadWeights(const Tensor<OHWDI, T>& weights);

  template <DataType T>
  void UploadBias(const GpuInfo& gpu_info, const Tensor<Linear, T>& bias);

  void InitArgs(const OperationDef& definition);
  void AddRuntimeWeightsDef();

  MemoryType GetBufferWeightsMemoryType() const {
    return kernel_params_.weights_upload_type ==
                   ConvGeneric::WeightsUploadType::kConstantMemory
               ? MemoryType::CONSTANT
               : MemoryType::GLOBAL;
  }

  friend ConvGeneric CreateConvGeneric(const GpuInfo& gpu_info,
                                       const OperationDef& definition,
                                       CalculationsPrecision precision,
                                       const Convolution2DAttributes& attr,
                                       const BHWC* dst_shape);

  friend ConvGeneric CreateConvGeneric(const GpuInfo& gpu_info,
                                       const OperationDef& definition,
                                       CalculationsPrecision precision,
                                       const FullyConnectedAttributes& attr,
                                       const BHWC* dst_shape);

  friend ConvGeneric CreateConvGenericExternalWeights(
      const GpuInfo& gpu_info, const OperationDef& definition,
      CalculationsPrecision precision, const Convolution2DAttributes& attr,
      const TensorDescriptor* bias, const BHWC* dst_shape,
      const TensorDescriptor* src_exp, bool different_weights_for_height,
      const ConvRuntimeCheckDesc& runtime_check);
  friend ConvGeneric CreateConvGenericExternalWeights(
      const GpuInfo& gpu_info, const OperationDef& definition,
      CalculationsPrecision precision, const ExternalWeights& weights,
      const TensorDescriptor* bias, const BHWC* dst_shape,
      const TensorDescriptor* src_exp, bool different_weights_for_height,
      const ConvRuntimeCheckDesc& runtime_check);

  friend ConvGeneric CreateConvGeneric3D(const GpuInfo& gpu_info,
                                         const OperationDef& definition,
                                         CalculationsPrecision precision,
                                         const Convolution3DAttributes& attr,
                                         const BHWDC* dst_shape);

  friend ConvGeneric CreateConvGenericInt8(
      const GpuInfo& gpu_info, const OperationDef& definition,
      PackedType src_packed_type, const Tensor<OHWI, DataType::INT8>& weights,
      const BHWC* dst_shape);
  friend ConvGeneric CreateConvGenericInt8ExternalWeights(
      const GpuInfo& gpu_info, const OperationDef& definition,
      PackedType src_packed_type, const OHWI& weights_shape,
      const BHWC* dst_shape);

  friend ConvGeneric CreateConvGenericInt4(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const Tensor<OHWI, DataType::INT8>& weights, const BHWC* dst_shape);
  friend ConvGeneric CreateConvGenericInt4ExternalWeights(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const OHWI& weights_shape, const BHWC* dst_shape);

  ConvParams conv_params_;
  KernelParams kernel_params_;
};

template <DataType T>
void ConvGeneric::UploadBias(const GpuInfo& gpu_info,
                             const Tensor<Linear, T>& bias) {
  if (!conv_params_.has_bias) return;
  TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
      gpu_info, conv_params_.weights_data_type, bias);
  args_.AddObject("biases", std::make_unique<TensorDescriptor>(
                                std::move(bias_tensor_desc)));
}

template <DataType T>
void ConvGeneric::UploadWeights(const Tensor<OHWI, T>& weights) {
  const auto weights_desc = GetWeightsDescription();
  const int elements_count =
      GetTotalElementsCountForLayout(weights_desc, weights.shape);

  std::vector<uint8_t> weights_data(elements_count * SizeOf(weights_desc.type));
  RearrangeWeights(weights, weights_desc, absl::MakeSpan(weights_data));

  if (kernel_params_.AreWeightsBuffer()) {
    BufferDescriptor desc;
    desc.element_type = kernel_params_.weights_type;
    desc.element_size = kernel_params_.weights_element_size;
    desc.memory_type = GetBufferWeightsMemoryType();
    desc.size = weights_data.size();
    desc.data = std::move(weights_data);
    args_.AddObject("weights",
                    std::make_unique<BufferDescriptor>(std::move(desc)));
  } else {
    uint2 tex_size = Get2dResourceSize(weights_desc, weights.shape);
    int sub_size = SizeOf(weights_desc.type) * 4 * tex_size.x * tex_size.y;
    for (int i = 0; i < 4; ++i) {
      TensorDescriptor desc = CreateConstantHWVec4TensorDescriptor(
          weights_desc.type, TensorStorageType::TEXTURE_2D, tex_size.x,
          tex_size.y, weights_data.data() + sub_size * i);
      args_.AddObject("weights" + std::to_string(i),
                      std::make_unique<TensorDescriptor>(std::move(desc)));
    }
  }
}

template <DataType T>
void ConvGeneric::UploadWeights(const Tensor<OHWDI, T>& weights) {
  const auto weights_desc = GetWeightsDescription();
  const int elements_count =
      GetTotalElementsCountForLayout(weights_desc, weights.shape);

  std::vector<uint8_t> weights_data(elements_count * SizeOf(weights_desc.type));
  RearrangeWeights(weights, weights_desc, absl::MakeSpan(weights_data));

  if (kernel_params_.AreWeightsBuffer()) {
    BufferDescriptor desc;
    desc.element_type = weights_desc.type;
    desc.element_size = 4;
    desc.memory_type = GetBufferWeightsMemoryType();
    desc.size = weights_data.size();
    desc.data = std::move(weights_data);
    args_.AddObject("weights",
                    std::make_unique<BufferDescriptor>(std::move(desc)));
  } else {
    uint2 tex_size = Get2dResourceSize(weights_desc, weights.shape);
    int sub_size = SizeOf(weights_desc.type) * 4 * tex_size.x * tex_size.y;
    for (int i = 0; i < 4; ++i) {
      TensorDescriptor desc = CreateConstantHWVec4TensorDescriptor(
          weights_desc.type, TensorStorageType::TEXTURE_2D, tex_size.x,
          tex_size.y, weights_data.data() + sub_size * i);
      args_.AddObject("weights" + std::to_string(i),
                      std::make_unique<TensorDescriptor>(std::move(desc)));
    }
  }
}

// Creates a generic convolution operation.
ConvGeneric CreateConvGeneric(const GpuInfo& gpu_info,
                              const OperationDef& definition,
                              CalculationsPrecision precision,
                              const Convolution2DAttributes& attr,
                              const BHWC* dst_shape = nullptr);

// Creates a generic convolution operation for fully connected layers.
ConvGeneric CreateConvGeneric(const GpuInfo& gpu_info,
                              const OperationDef& definition,
                              CalculationsPrecision precision,
                              const FullyConnectedAttributes& attr,
                              const BHWC* dst_shape = nullptr);

// Creates a generic convolution operation with external weights.
ConvGeneric CreateConvGenericExternalWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision, const Convolution2DAttributes& attr,
    const TensorDescriptor* bias, const BHWC* dst_shape,
    const TensorDescriptor* src_exp, bool different_weights_for_height,
    const ConvRuntimeCheckDesc& runtime_check = {});

bool SupportsConvGeneric(const GpuInfo& gpu_info,
                         CalculationsPrecision precision,
                         const WeightsDescription& weights_desc,
                         const OHWI& weights_shape);

ConvGeneric CreateConvGenericExternalWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision, const ExternalWeights& weights,
    const TensorDescriptor* bias, const BHWC* dst_shape,
    const TensorDescriptor* src_exp, bool different_weights_for_height,
    const ConvRuntimeCheckDesc& runtime_check = {});

// Creates a generic 3D convolution operation.
ConvGeneric CreateConvGeneric3D(const GpuInfo& gpu_info,
                                const OperationDef& definition,
                                CalculationsPrecision precision,
                                const Convolution3DAttributes& attr,
                                const BHWDC* dst_shape = nullptr);

// Checks if the generic INT8 convolution is supported on the given GPU.
bool SupportsConvGenericInt8(const GpuInfo& gpu_info);
// Returns the source packed type for the generic INT8 convolution.
PackedType GetConvGenericInt8SrcType(const GpuInfo& gpu_info,
                                     const BHWC& src_shape);
// Checks if UINT8 math should be used for INT8 weights.
bool UseUint8MathForInt8Weights(const GpuInfo& gpu_info);

// Creates a generic INT8 convolution operation.
ConvGeneric CreateConvGenericInt8(const GpuInfo& gpu_info,
                                  const OperationDef& definition,
                                  PackedType src_packed_type,
                                  const Tensor<OHWI, DataType::INT8>& weights,
                                  const BHWC* dst_shape = nullptr);

// Creates a generic INT8 convolution operation with external weights.
ConvGeneric CreateConvGenericInt8ExternalWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    PackedType src_packed_type, const OHWI& weights_shape,
    const BHWC* dst_shape = nullptr);

// Checks if the generic INT4 convolution is supported on the given GPU.
bool SupportsConvGenericInt4(const GpuInfo& gpu_info, const BHWC& src_shape);
// Returns the source packed type for the generic INT4 convolution.
PackedType GetConvGenericInt4SrcType(const GpuInfo& gpu_info,
                                     const BHWC& src_shape);
// Creates a generic INT4 convolution operation.
ConvGeneric CreateConvGenericInt4(const GpuInfo& gpu_info,
                                  const OperationDef& definition,
                                  const Tensor<OHWI, DataType::INT8>& weights,
                                  const BHWC* dst_shape = nullptr);

// Creates a generic INT4 convolution operation with external weights.
ConvGeneric CreateConvGenericInt4ExternalWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OHWI& weights_shape, const BHWC* dst_shape = nullptr);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_KERNELS_CONV_GENERIC_H_
