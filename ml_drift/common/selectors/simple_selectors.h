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

#ifndef ML_DRIFT_COMMON_SELECTORS_SIMPLE_SELECTORS_H_
#define ML_DRIFT_COMMON_SELECTORS_SIMPLE_SELECTORS_H_

#include <memory>
#include <set>
#include <vector>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/model_hints.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

std::unique_ptr<GPUOperation> SelectLSTM(const OperationDef& op_def,
                                         const GpuInfo& gpu_info);

std::unique_ptr<GPUOperation> SelectReLU(const ReLUAttributes& attr,
                                         const OperationDef& op_def);

std::unique_ptr<GPUOperation> SelectPReLU(const PReLUAttributes& attr,
                                          const GpuInfo& gpu_info,
                                          const OperationDef& op_def);

std::unique_ptr<GPUOperation> SelectPooling(const Pooling2DAttributes& attr,
                                            const GpuInfo& gpu_info,
                                            const OperationDef& op_def);

std::unique_ptr<GPUOperation> SelectMaxUnpooling(
    const MaxUnpooling2DAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def);

void SelectAdd(const OperationDef& op_def, const std::vector<int>& channels,
               int dst_channels, std::unique_ptr<GPUOperation>* ptr);

absl::Status SelectResize(const Resize2DAttributes& attr,
                          const OperationDef& op_def,
                          std::unique_ptr<GPUOperation>* ptr);

std::unique_ptr<GPUOperation> SelectReverse(const ReverseAttributes& attr,
                                            const OperationDef& op_def);

std::unique_ptr<GPUOperation> SelectResampler(const OperationDef& op_def,
                                              const GpuInfo& gpu_info);

absl::Status SelectConcat(const ConcatAttributes& attr,
                          const std::vector<int>& channels,
                          const OperationDef& op_def, const GpuInfo& gpu_info,
                          std::unique_ptr<GPUOperation>* ptr);

std::unique_ptr<GPUOperation> SelectPadding(const GpuInfo& gpu_info,
                                            const PadAttributes& attr,
                                            const OperationDef& op_def);

std::unique_ptr<GPUOperation> SelectReduce(const std::set<Axis>& axis_to_reduce,
                                           const BHWC& src_shape,
                                           OperationType op_type,
                                           const OperationDef& op_def,
                                           const GpuInfo& gpu_info);

std::unique_ptr<GPUOperation> SelectSoftmax(
    const GpuInfo& gpu_info, const BHWC& shape, const OperationDef& op_def,
    const SoftmaxRuntimeCheckDesc& runtime_check = {});

void SelectSpaceToDepth(const SpaceToDepthAttributes& attr,
                        const OperationDef& op_def,
                        std::unique_ptr<GPUOperation>* ptr);

void SelectDepthToSpace(const SpaceToDepthAttributes& attr,
                        const OperationDef& op_def,
                        std::unique_ptr<GPUOperation>* ptr);

void SelectEmbeddingLookup(const EmbeddingLookupAttributes& attr,
                           const OperationDef& op_def, const GpuInfo& gpu_info,
                           std::unique_ptr<GPUOperation>* ptr);

void SelectSplit(const SplitAttributes& attr, const GpuInfo& gpu_info,
                 const std::vector<int>& channels, const OperationDef& op_def,
                 std::unique_ptr<GPUOperation>* ptr);

std::unique_ptr<GPUOperation> SelectTile(const OperationDef& op_def);

std::unique_ptr<GPUOperation> SelectWinograd3x3Forward(
    const GpuInfo& gpu_info, const Padding2D& padding,
    const OperationDef& op_def, int tile_size);

std::unique_ptr<GPUOperation> SelectWinograd3x3Backward(
    const GpuInfo& gpu_info, const OperationDef& op_def, int tile_size,
    const Tensor<Linear, DataType::FLOAT32>& biases);

std::unique_ptr<GPUOperation> SelectQuantizeAndDequantize(
    const QuantizeAndDequantizeAttributes& attr, const OperationDef& op_def);

std::unique_ptr<GPUOperation> SelectStaticRangeQuantization(
    const QuantizeAndDequantizeAttributes& attr, const OperationDef& op_def);

void SelectCumsum(const OperationDef& op_def, const CumsumAttributes& attr,
                  std::unique_ptr<GPUOperation>* ptr);

void SelectOneHot(const OperationDef& op_def, const OneHotAttributes& attr,
                  std::unique_ptr<GPUOperation>* ptr);

void SelectSelectV2(const OperationDef& op_def, const SelectV2Attributes& attr,
                    std::unique_ptr<GPUOperation>* ptr);

std::unique_ptr<GPUOperation> SelectConverterToConvWeights(
    const GpuInfo& gpu_info, const OHWI& weights_shape,
    const WeightsDescription& weights_desc, const OperationDef& op_def,
    ModelHints hints, Layout input_layout,
    const TensorDescriptor* scale_desc = nullptr,
    const TensorDescriptor* zero_point_desc = nullptr);

std::unique_ptr<GPUOperation> SelectDynamicUpdateSlice(
    const OperationDef& op_def, const GpuInfo& gpu_info);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_SELECTORS_SIMPLE_SELECTORS_H_
