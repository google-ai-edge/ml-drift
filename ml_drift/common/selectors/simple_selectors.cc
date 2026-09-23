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

#include "ml_drift/common/selectors/simple_selectors.h"

#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernels/add.h"
#include "ml_drift/common/kernels/concat_xy.h"
#include "ml_drift/common/kernels/concat_z.h"
#include "ml_drift/common/kernels/conv_weights_converter.h"
#include "ml_drift/common/kernels/cumsum.h"
#include "ml_drift/common/kernels/dynamic_update_slice.h"
#include "ml_drift/common/kernels/embedding_lookup.h"
#include "ml_drift/common/kernels/lstm.h"
#include "ml_drift/common/kernels/max_unpooling.h"
#include "ml_drift/common/kernels/one_hot.h"
#include "ml_drift/common/kernels/padding.h"
#include "ml_drift/common/kernels/pooling.h"
#include "ml_drift/common/kernels/prelu.h"
#include "ml_drift/common/kernels/quantize_and_dequantize.h"
#include "ml_drift/common/kernels/reduce.h"
#include "ml_drift/common/kernels/relu.h"
#include "ml_drift/common/kernels/resampler.h"
#include "ml_drift/common/kernels/resize.h"
#include "ml_drift/common/kernels/reverse.h"
#include "ml_drift/common/kernels/scatter_nd.h"
#include "ml_drift/common/kernels/select_v2.h"
#include "ml_drift/common/kernels/space_to_depth.h"
#include "ml_drift/common/kernels/split.h"
#include "ml_drift/common/kernels/tile.h"
#include "ml_drift/common/kernels/winograd.h"
#include "ml_drift/common/model_hints.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

std::unique_ptr<GPUOperation> SelectLSTM(const OperationDef& op_def,
                                         const GpuInfo& gpu_info) {
  return std::make_unique<GPUOperation>(CreateLSTM(op_def, gpu_info));
}

std::unique_ptr<GPUOperation> SelectReLU(const ReLUAttributes& attr,
                                         const OperationDef& op_def) {
  return std::make_unique<GPUOperation>(CreateReLU(op_def, attr));
}

std::unique_ptr<GPUOperation> SelectPReLU(const PReLUAttributes& attr,
                                          const GpuInfo& gpu_info,
                                          const OperationDef& op_def) {
  return std::make_unique<GPUOperation>(CreatePReLU(gpu_info, op_def, attr));
}

std::unique_ptr<GPUOperation> SelectPooling(const Pooling2DAttributes& attr,
                                            const GpuInfo& gpu_info,
                                            const OperationDef& op_def) {
  return std::make_unique<GPUOperation>(CreatePooling(op_def, gpu_info, attr));
}

std::unique_ptr<GPUOperation> SelectMaxUnpooling(
    const MaxUnpooling2DAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def) {
  return std::make_unique<GPUOperation>(
      CreateMaxUnpooling(gpu_info, op_def, attr));
}

void SelectAdd(const OperationDef& op_def, const std::vector<int>& channels,
               int dst_channels, std::unique_ptr<GPUOperation>* ptr) {
  GPUOperation operation = CreateAdd(op_def, channels, dst_channels);
  *ptr = std::make_unique<GPUOperation>(std::move(operation));
}

std::unique_ptr<GPUOperation> SelectDynamicUpdateSlice(
    const OperationDef& op_def, const GpuInfo& gpu_info) {
  return std::make_unique<GPUOperation>(
      CreateDynamicUpdateSlice(op_def, gpu_info));
}

std::unique_ptr<GPUOperation> SelectScatterNd(const OperationDef& op_def) {
  return std::make_unique<GPUOperation>(CreateScatterNd(op_def));
}

std::unique_ptr<GPUOperation> SelectResampler(const OperationDef& op_def,
                                              const GpuInfo& gpu_info) {
  GPUOperation operation = CreateResampler(gpu_info, op_def);
  return std::make_unique<GPUOperation>(std::move(operation));
}

absl::Status SelectResize(const Resize2DAttributes& attr,
                          const OperationDef& op_def,
                          std::unique_ptr<GPUOperation>* ptr) {
  Resize operation = CreateResize(op_def, attr);
  *ptr = std::make_unique<Resize>(std::move(operation));
  return absl::OkStatus();
}

std::unique_ptr<GPUOperation> SelectReverse(const ReverseAttributes& attr,
                                            const OperationDef& op_def) {
  GPUOperation operation = CreateReverse(op_def, attr);
  return std::make_unique<GPUOperation>(std::move(operation));
}

absl::Status SelectConcat(const ConcatAttributes& attr,
                          const std::vector<int>& channels,
                          const OperationDef& op_def, const GpuInfo& gpu_info,
                          std::unique_ptr<GPUOperation>* ptr) {
  switch (attr.axis) {
    case Axis::CHANNELS: {
      GPUOperation operation = CreateConcatZ(op_def, channels, gpu_info);
      *ptr = std::make_unique<GPUOperation>(std::move(operation));
      return absl::OkStatus();
    }
    case Axis::BATCH:
    case Axis::DEPTH:
    case Axis::HEIGHT:
    case Axis::WIDTH: {
      GPUOperation operation = CreateConcatXY(op_def, attr);
      *ptr = std::make_unique<GPUOperation>(std::move(operation));
      return absl::OkStatus();
    }
    default:
      return absl::UnimplementedError("No concat for this axis.");
  }
}

void SelectSpaceToDepth(const SpaceToDepthAttributes& attr,
                        const OperationDef& op_def,
                        std::unique_ptr<GPUOperation>* ptr) {
  GPUOperation operation = CreateSpaceToDepth(op_def, attr);
  *ptr = std::make_unique<GPUOperation>(std::move(operation));
}

void SelectDepthToSpace(const SpaceToDepthAttributes& attr,
                        const OperationDef& op_def,
                        std::unique_ptr<GPUOperation>* ptr) {
  GPUOperation operation = CreateDepthToSpace(op_def, attr);
  *ptr = std::make_unique<GPUOperation>(std::move(operation));
}

void SelectEmbeddingLookup(const EmbeddingLookupAttributes& attr,
                           const OperationDef& op_def, const GpuInfo& gpu_info,
                           std::unique_ptr<GPUOperation>* ptr) {
  GPUOperation operation = CreateEmbeddingLookup(op_def, gpu_info, attr);
  *ptr = std::make_unique<GPUOperation>(std::move(operation));
}

void SelectSplit(const SplitAttributes& attr, const GpuInfo& gpu_info,
                 const std::vector<int>& channels, const OperationDef& op_def,
                 std::unique_ptr<GPUOperation>* ptr) {
  Split operation = CreateSplit(gpu_info, op_def, attr, channels);
  *ptr = std::make_unique<Split>(std::move(operation));
}

std::unique_ptr<GPUOperation> SelectPadding(const GpuInfo& gpu_info,
                                            const PadAttributes& attr,
                                            const OperationDef& op_def,
                                            int src_channels) {
  return std::make_unique<GPUOperation>(
      CreatePadding(gpu_info, op_def, attr, src_channels));
}

std::unique_ptr<GPUOperation> SelectReduce(const std::set<Axis>& axis_to_reduce,
                                           const BHWC& src_shape,
                                           OperationType op_type,
                                           const OperationDef& op_def,
                                           const GpuInfo& gpu_info) {
  return std::make_unique<Reduce>(
      CreateReduce(axis_to_reduce, src_shape, op_type, op_def, gpu_info));
}

std::unique_ptr<GPUOperation> SelectTile(const OperationDef& op_def) {
  return std::make_unique<GPUOperation>(
      CreateTile(op_def.src_tensors[0], op_def.dst_tensors[0],
                 op_def.src_tensors[0].GetBHWCShape().c % 4 == 0));
}

std::unique_ptr<GPUOperation> SelectWinograd3x3Forward(
    const GpuInfo& gpu_info, const Padding2D& padding,
    const OperationDef& op_def, int tile_size) {
  if (gpu_info.IsAMD() && tile_size == 6) {
    Winograd4x4To36 operation =
        CreateWinograd4x4To36(op_def, padding, gpu_info);
    return std::make_unique<Winograd4x4To36>(std::move(operation));
  }
  return std::make_unique<Winograd3x3TiledXForward>(
      CreateWinograd3x3TiledXForward(gpu_info, op_def, padding, tile_size));
}

std::unique_ptr<GPUOperation> SelectWinograd3x3Backward(
    const GpuInfo& gpu_info, const OperationDef& op_def, int tile_size,
    const Tensor<Linear, DataType::FLOAT32>& biases) {
  if (gpu_info.IsAMD() && tile_size == 6) {
    Winograd36To4x4 operation = CreateWinograd36To4x4(op_def, biases);
    return std::make_unique<Winograd36To4x4>(std::move(operation));
  }
  return std::make_unique<Winograd3x3TiledXBackward>(
      CreateWinograd3x3TiledXBackward(gpu_info, op_def, biases, tile_size));
}

std::unique_ptr<GPUOperation> SelectQuantizeAndDequantize(
    const QuantizeAndDequantizeAttributes& attr, const OperationDef& op_def) {
  return std::make_unique<GPUOperation>(
      CreateQuantizeAndDequantize(op_def, attr));
}

std::unique_ptr<GPUOperation> SelectStaticRangeQuantization(
    const QuantizeAndDequantizeAttributes& attr, const OperationDef& op_def) {
  return std::make_unique<GPUOperation>(
      CreateStaticRangeQuantization(op_def, attr));
}

void SelectCumsum(const OperationDef& op_def, const CumsumAttributes& attr,
                  std::unique_ptr<GPUOperation>* ptr) {
  Cumsum operation = CreateCumsum(op_def, attr);
  *ptr = std::make_unique<Cumsum>(std::move(operation));
}

void SelectOneHot(const OperationDef& op_def, const OneHotAttributes& attr,
                  std::unique_ptr<GPUOperation>* ptr) {
  GPUOperation operation = CreateOneHot(op_def, attr);
  *ptr = std::make_unique<GPUOperation>(std::move(operation));
}

void SelectSelectV2(const OperationDef& op_def, const SelectV2Attributes& attr,
                    std::unique_ptr<GPUOperation>* ptr) {
  GPUOperation operation = CreateSelectV2(op_def, attr);
  *ptr = std::make_unique<GPUOperation>(std::move(operation));
}

std::unique_ptr<GPUOperation> SelectConverterToConvWeights(
    const GpuInfo& gpu_info, const OHWI& weights_shape,
    const WeightsDescription& weights_desc, const OperationDef& op_def,
    ModelHints hints, Layout input_layout, const TensorDescriptor* scale_desc,
    const TensorDescriptor* zero_point_desc) {
  return std::make_unique<ConverterToConvWeights>(
      ConverterToConvWeights(gpu_info, op_def, weights_shape, weights_desc,
                             input_layout, scale_desc, zero_point_desc));
}

}  // namespace ml_drift
