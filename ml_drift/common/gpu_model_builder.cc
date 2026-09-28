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

#include "ml_drift/common/gpu_model_builder.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/flops_util.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/kernels/accumulate_input_channels.h"
#include "ml_drift/common/kernels/bitcast.h"
#include "ml_drift/common/kernels/cast.h"
#include "ml_drift/common/kernels/conv_apple_mpp.h"
#include "ml_drift/common/kernels/conv_generic.h"
#include "ml_drift/common/kernels/conv_wave_matrix.h"
#include "ml_drift/common/kernels/conv_weights_converter.h"
#include "ml_drift/common/kernels/cumsum.h"
#include "ml_drift/common/kernels/elementwise.h"
#include "ml_drift/common/kernels/embedding_lookup.h"
#include "ml_drift/common/kernels/fully_connected.h"
#include "ml_drift/common/kernels/gather.h"
#include "ml_drift/common/kernels/mean_stddev_normalization.h"
#include "ml_drift/common/kernels/positional_embedding.h"
#include "ml_drift/common/kernels/quantize_and_dequantize.h"
#include "ml_drift/common/kernels/reduce.h"
#include "ml_drift/common/kernels/reshape.h"
#include "ml_drift/common/kernels/reshapex4.h"
#include "ml_drift/common/kernels/rope.h"
#include "ml_drift/common/kernels/select_v2.h"
#include "ml_drift/common/kernels/softmax.h"
#include "ml_drift/common/kernels/softmax1x1.h"
#include "ml_drift/common/kernels/space_to_depth.h"
#include "ml_drift/common/kernels/special/conv_softmax_conv.h"
#include "ml_drift/common/kernels/strided_slice.h"
#include "ml_drift/common/kernels/top_k.h"
#include "ml_drift/common/kernels/transpose.h"
#include "ml_drift/common/kernels/winograd.h"
#include "ml_drift/common/merge_nodes.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/model_hints.h"
#include "ml_drift/common/op_attrs.h"
#include "ml_drift/common/op_base.h"
#include "ml_drift/common/op_registry.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/selectors/convolution_selector.h"
#include "ml_drift/common/selectors/convolution_transposed_selector.h"
#include "ml_drift/common/selectors/dw_convolution_selector.h"
#include "ml_drift/common/selectors/simple_selectors.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"
#include "ml_drift/common/winograd_util.h"

namespace ml_drift {
namespace {

const size_t kDefaultBatchSize = 32;

int GetMaxSrcImages(const GpuInfo& gpu_info) {
  // only in webgpu we bind src images as storage textures vs sampled textures
  // in other apis.
  return gpu_info.IsApiWebGpu() ? gpu_info.GetMaxImageArguments()
                                : gpu_info.GetMaxTextureArguments();
}

std::string GetConvOpNameSuffix(const GPUOperation& op) {
  return op.GetDebugName().empty() ? ""
                                   : absl::StrCat("(", op.GetDebugName(), ")");
}

std::string ToString(WeightsManager::TargetWeightsType type) {
  switch (type) {
    case WeightsManager::TargetWeightsType::kStandard:
      return "kStandard";
    case WeightsManager::TargetWeightsType::kWinograd3x3:
      return "kWinograd3x3";
    case WeightsManager::TargetWeightsType::kWeightsSumI:
      return "kWeightsSumI";
    default:
      return "Unknown";
  }
}

inline size_t GetElementsCountForDataType(const DataType& data_type,
                                          const OHWI& weights_shape) {
  size_t num_values = weights_shape.DimensionsProduct();
  switch (data_type) {
    case DataType::INT4:
    case DataType::UINT4:
      // Raw Int4 or UInt4 weights are stored in packed format: two int4
      // values are stored in one element (one byte).
      return DivideRoundUp(num_values, SizeInBitsOf(DataType::INT8) /
                                           SizeInBitsOf(DataType::INT4));
    case DataType::INT2:
    case DataType::UINT2:
      // Raw Int2 or UInt2 weights are stored in packed format: four int2
      // values are stored in one element (one byte).
      return DivideRoundUp(num_values, SizeInBitsOf(DataType::INT8) /
                                           SizeInBitsOf(DataType::INT2));
    default:
      return num_values;
  }
}

ExternalWeights ToExternalWeights(const GpuModelBuilder::Weights& weights) {
  ExternalWeights external_weights;
  external_weights.desc = weights.desc;
  external_weights.shape = weights.shape;
  if (weights.scale) {
    external_weights.scale_zp_shape = weights.scale_zp_shape;
    external_weights.scale = &(weights.scale->tensor_desc);
  }
  if (weights.zero_point) {
    external_weights.zero_point = &(weights.zero_point->tensor_desc);
  }
  return external_weights;
}

}  // namespace

GpuModelBuilder GpuModelBuilder::CreateBuilder() const {
  GpuModelBuilderOptions options;
  options.hints = hints_;
  options.storage = default_storage_;
  options.use_f32_accum_for_f16_convolutions =
      use_f32_accum_for_f16_convolutions_;
  return GpuModelBuilder(gpu_info_, options);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::AddTensor(
    const TensorDescriptor& tensor_desc) {
  tensor_desc.CopyWithoutData(&gpu_model_.tensors[id_counter_]);
  GpuModelBuilder::TensorHandle tensor_handle;
  tensor_handle.id = id_counter_;
  tensor_handle.tensor_desc = gpu_model_.tensors[id_counter_];
  id_counter_++;
  return tensor_handle;
}

std::vector<GpuModelBuilder::TensorHandle> GpuModelBuilder::AddTensors(
    const std::vector<TensorDescriptor>& tensor_descs) {
  std::vector<GpuModelBuilder::TensorHandle> dsts;
  dsts.reserve(tensor_descs.size());
  for (const auto& tensor_desc : tensor_descs) {
    dsts.push_back(AddTensor(tensor_desc));
  }
  return dsts;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::AddTensor(
    int b, int h, int w, int c, TensorStorageType storage_type,
    DataType data_type) {
  Layout layout = b == 1 ? Layout::HWC : Layout::BHWC;
  TensorDescriptor tensor_desc =
      TensorDescriptor{data_type, storage_type, layout};
  auto status =
      tensor_desc.UpdateToSupportedStorageType(gpu_info_, BHWC(b, h, w, c));
  tensor_desc.SetBHWCShape(BHWC(b, h, w, c));
  return AddTensor(tensor_desc);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::AddTensor(int b, int h, int w,
                                                         int c,
                                                         DataType data_type) {
  return AddTensor(b, h, w, c, default_storage_, data_type);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::AddTensor(const BHWC& shape,
                                                         DataType data_type) {
  return AddTensor(shape.b, shape.h, shape.w, shape.c, default_storage_,
                   data_type);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::AddTensor(
    int b, int h, int w, int d, int c, TensorStorageType storage_type,
    DataType data_type) {
  const bool has_b = b > 1;
  const bool has_d = d > 1;
  Layout layout;
  if (has_b && has_d) {
    layout = Layout::BHWDC;
  } else if (has_b) {
    layout = Layout::BHWC;
  } else if (has_d) {
    layout = Layout::HWDC;
  } else {
    layout = Layout::HWC;
  }
  TensorDescriptor tensor_desc =
      TensorDescriptor{data_type, storage_type, layout};
  auto status =
      tensor_desc.UpdateToSupportedStorageType(gpu_info_, BHWDC(b, h, w, d, c));
  tensor_desc.SetBHWDCShape(BHWDC(b, h, w, d, c));
  return AddTensor(tensor_desc);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::AddTensor(int b, int h, int w,
                                                         int d, int c,
                                                         DataType data_type) {
  return AddTensor(b, h, w, d, c, default_storage_, data_type);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::AddTensor(const BHWDC& shape,
                                                         DataType data_type) {
  return AddTensor(shape.b, shape.h, shape.w, shape.d, shape.c,
                   default_storage_, data_type);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::AddLinearTensor(
    int x, DataType data_type) {
  TensorDescriptor tensor_desc =
      TensorDescriptor{data_type, default_storage_, Layout::LINEAR};
  BHWC shape = BHWC(1, 1, 1, x);
  auto status = tensor_desc.UpdateToSupportedStorageType(gpu_info_, shape);
  tensor_desc.SetBHWCShape(shape);
  return AddTensor(tensor_desc);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::AddConstantTensor(
    TensorDescriptor&& tensor_desc) {
  gpu_model_.const_tensors[id_counter_] = std::move(tensor_desc);
  GpuModelBuilder::TensorHandle tensor_handle;
  tensor_handle.id = id_counter_;
  gpu_model_.const_tensors[id_counter_].CopyWithoutData(
      &tensor_handle.tensor_desc);
  id_counter_++;
  return tensor_handle;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::AddConstantTensor(
    const Tensor<Linear, DataType::FLOAT32>& tensor, DataType data_type) {
  TensorDescriptor tensor_desc =
      CreateConstantLinearTensorDescriptor(gpu_info_, data_type, tensor);
  return AddConstantTensor(std::move(tensor_desc));
}

GpuModelBuilder::TensorHandle GpuModelBuilder::AddConstantTensor(
    const TensorFloat32& tensor, DataType data_type) {
  Layout layout = tensor.shape.b == 1 ? Layout::HWC : Layout::BHWC;
  TensorDescriptor tensor_desc =
      TensorDescriptor{data_type, default_storage_, layout};
  auto status =
      tensor_desc.UpdateToSupportedStorageType(gpu_info_, tensor.shape);
  tensor_desc.UploadData(tensor);
  return AddConstantTensor(std::move(tensor_desc));
}

absl::StatusOr<GpuModelBuilder::TensorHandle> GpuModelBuilder::GetTensor(
    ValueId id) const {
  GpuModelBuilder::TensorHandle tensor_handle;
  tensor_handle.id = id;
  if (auto it = gpu_model_.tensors.find(id); it != gpu_model_.tensors.end()) {
    tensor_handle.tensor_desc = it->second;
    return tensor_handle;
  }
  if (auto it = gpu_model_.const_tensors.find(id);
      it != gpu_model_.const_tensors.end()) {
    it->second.CopyWithoutData(&tensor_handle.tensor_desc);
    return tensor_handle;
  }
  return absl::NotFoundError("Tensor not found");
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Cast(
    const GpuModelBuilder::TensorHandle& src, DataType dst_type) {
  const BHWC new_shape = src.tensor_desc.GetBHWCShape();
  GpuModelBuilder::TensorHandle dst = AddTensor(new_shape, dst_type);
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);
  GPUOperation operation = CreateCast(op_def, gpu_info_);
  AddGpuOperation(std::vector<ValueId>{src.id}, std::vector<ValueId>{dst.id},
                  std::make_unique<GPUOperation>(std::move(operation)), "cast");
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::BitCast(
    const GpuModelBuilder::TensorHandle& src, DataType dst_type) {
  const BHWC new_shape = src.tensor_desc.GetBHWCShape();
  GpuModelBuilder::TensorHandle dst = AddTensor(new_shape, dst_type);
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);
  GPUOperation operation = CreateBitcast(op_def, gpu_info_);
  AddGpuOperation(std::vector<ValueId>{src.id}, std::vector<ValueId>{dst.id},
                  std::make_unique<GPUOperation>(std::move(operation)),
                  "bit_cast");
  return dst;
}

std::vector<GpuModelBuilder::TensorHandle>
GpuModelBuilder::GetWinograd3x3Weights(
    const Tensor<OHWI, DataType::FLOAT32>& weights,
    WeightsDescription weights_desc, int tile_size) {
  std::vector<TensorHandle> wino_weights;
  if (hints_.winograd_runtime_weights_conversion) {
    TensorHandle weights_linear_handle;
    {
      Tensor<Linear, DataType::FLOAT32> weights_linear;
      weights_linear.shape = Linear(weights.shape.DimensionsProduct());
      weights_linear.data = weights.data;
      TensorDescriptor tensor_desc = CreateConstantLinearTensorDescriptor(
          gpu_info_, weights_desc.type, weights_linear);
      weights_linear_handle = AddConstantTensor(std::move(tensor_desc));
    }
    wino_weights = GetWinograd3x3WeightsFromOHWI(weights_linear_handle,
                                                 weights.shape, weights_desc);
  } else {
    Tensor<OHWI, DataType::FLOAT32> wino_weights_cpu;
    RearrangeWeightsToWinograd3x3TileNxN(weights, &wino_weights_cpu, tile_size);

    auto weights_descs =
        GetTensorDescriptorsForWeightsLayout(wino_weights_cpu, weights_desc);
    wino_weights.resize(weights_descs.size());
    for (int i = 0; i < weights_descs.size(); ++i) {
      wino_weights[i] = AddConstantTensor(std::move(weights_descs[i]));
    }
  }
  return wino_weights;
}

std::vector<GpuModelBuilder::TensorHandle>
GpuModelBuilder::GetWinograd3x3WeightsFromOHWI(
    const TensorHandle& weights_ohwi, const OHWI& weights_shape,
    WeightsDescription dst_weights_desc) {
  const OHWI wino_weights_shape =
      GetWinograd3x3TileNxNWeightsShape(weights_shape, /*tile_size=*/6);
  auto wino_weights_as_bhwc_handle =
      AddTensor(BHWC(wino_weights_shape.o, wino_weights_shape.h,
                     wino_weights_shape.w, wino_weights_shape.i),
                weights_ohwi.tensor_desc.GetDataType());
  Winograd3x3To36 operation = Winograd3x3To36(
      weights_ohwi.tensor_desc, wino_weights_as_bhwc_handle.tensor_desc);
  AddGpuOperation(std::vector<ValueId>{weights_ohwi.id},
                  std::vector<ValueId>{wino_weights_as_bhwc_handle.id},
                  std::make_unique<Winograd3x3To36>(std::move(operation)),
                  "weights_to_winograd");

  return WeightsConversion(wino_weights_as_bhwc_handle, Layout::OHWI,
                           dst_weights_desc, wino_weights_shape);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::GetWeightsSumIFromRawOHWI(
    const GpuModelBuilder::TensorHandle& src_weights,
    const OHWI& weights_shape,
    const DataType& src_data_type) {
  TensorDescriptor dst_desc = TensorDescriptor(
      DataType::INT32, TensorStorageType::BUFFER, Layout::LINEAR);
  dst_desc.SetBHWCShape(BHWC(1, 1, 1, weights_shape.o));
  const auto dst = AddTensor(dst_desc);

  OperationDef op_def;
  op_def.src_tensors.push_back(src_weights.tensor_desc);
  op_def.dst_tensors.push_back(std::move(dst_desc));

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = "raw_ohwi_weights_to_weights_sum_i";
  gpu_node.gpu_operation = std::make_unique<GPUOperation>(
      CreateAccumulateInputChannels(op_def, weights_shape, src_data_type));
  gpu_node.inputs = {src_weights.id};
  gpu_node.outputs.push_back(dst.id);
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::WinoConvolution(
    const GpuModelBuilder::TensorHandle& src,
    const Convolution2DAttributes& attr, int tile_size) {
  const auto& weights = GetFloatWeights(attr);
  const BHWC input_shape = src.tensor_desc.GetBHWCShape();
  const BHWC output_shape = CalculateOutputShape(input_shape, attr);
  GpuModelBuilder::TensorHandle dst =
      AddTensor(output_shape, src.tensor_desc.GetDataType());
  const int tile_size_outer = tile_size;
  const int tile_size_inner = tile_size_outer - 2;
  const int tiles_x = DivideRoundUp(output_shape.w, tile_size_inner);
  const int tiles_y = DivideRoundUp(output_shape.h, tile_size_inner);

  const BHWC src_transformed_shape{input_shape.b,
                                   tile_size_outer * tile_size_outer,
                                   tiles_x * tiles_y, input_shape.c};
  const BHWC dst_transformed_shape{input_shape.b,
                                   tile_size_outer * tile_size_outer,
                                   tiles_x * tiles_y, output_shape.c};

  GpuModelBuilder::TensorHandle src_transformed =
      AddTensor(src_transformed_shape, src.tensor_desc.GetDataType());
  GpuModelBuilder::TensorHandle dst_transformed =
      AddTensor(dst_transformed_shape, src.tensor_desc.GetDataType());

  const OHWI wino_weights_shape(weights.shape.o, tile_size, tile_size,
                                weights.shape.i);
  std::unique_ptr<GPUOperation> conv_op;
  WeightsDescription weights_desc;
  {
    OperationDef conv_def;
    conv_def.src_tensors.push_back(src_transformed.tensor_desc);
    conv_def.dst_tensors.push_back(dst_transformed.tensor_desc);

    Convolution2DAttributes wino_attr;
    wino_attr.padding.prepended = HW(0, 0);
    wino_attr.padding.appended = HW(0, 0);
    wino_attr.strides = HW(1, 1);
    wino_attr.dilations = HW(1, 1);
    auto& wino_weights =
        wino_attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
    wino_weights.shape = wino_weights_shape;

    conv_op = SelectConvolutionWithExternalWeights(
        wino_attr, /*bias_desc_ptr*/ nullptr, dst_transformed_shape, gpu_info_,
        conv_def, GetConvPrecision(src.tensor_desc.GetDataType()), hints_,
        &weights_desc,
        /*src_exp=*/nullptr,
        /*different_weights_for_height=*/true);
  }

  std::vector<TensorHandle> wino_weights;
  if (weights_manager_ &&
      weights_manager_->ShouldOffloadPreparationToGPU(
          gpu_info_, weights.shape,
          WeightsManager::TargetWeightsType::kWinograd3x3)) {
    const OHWI wino_weights_shape =
        GetWinograd3x3TileNxNWeightsShape(weights.shape, /*tile_size=*/6);
    auto input_weights = AddTensors(
        GetTensorDescriptorsForWeightsLayout(wino_weights_shape, weights_desc));
    std::vector<ValueId> weights_ids(input_weights.size());
    for (int i = 0; i < input_weights.size(); ++i) {
      weights_ids[i] = input_weights[i].id;
    }
    weights_manager_->RegisterWinograd3x3WeightsConversion(
        weights_ids, weights_desc, weights.shape, DataType::FLOAT32,
        weights.Data());
    wino_weights = input_weights;
  } else {
    wino_weights = GetWinograd3x3Weights(weights, weights_desc, tile_size);
  }

  gpu_model_.nodes.reserve(gpu_model_.nodes.size() + 3);

  gpu_model_.nodes.push_back({});
  auto& winograd_up = gpu_model_.nodes.back();
  gpu_model_.nodes.push_back({});
  auto& conv = gpu_model_.nodes.back();
  gpu_model_.nodes.push_back({});
  auto& winograd_down = gpu_model_.nodes.back();

  OperationDef winograd_up_def;
  winograd_up_def.src_tensors.push_back(src.tensor_desc);
  winograd_up_def.dst_tensors.push_back(src_transformed.tensor_desc);
  winograd_up.gpu_operation = SelectWinograd3x3Forward(
      gpu_info_, attr.padding, winograd_up_def, tile_size_outer);
  winograd_up.inputs = {src.id};
  winograd_up.outputs = {src_transformed.id};
  winograd_up.name = "winograd_3x3_forward";
  winograd_up.op_name = absl::StrCat(attr.op_name, "_winograd_3x3_forward");

  conv.inputs = {src_transformed.id};
  for (const auto& weight : wino_weights) {
    conv.inputs.push_back(weight.id);
  }
  conv.outputs = {dst_transformed.id};
  conv.gpu_operation = std::move(conv_op);
  conv.name = absl::StrCat("convolution_winograd_3x3",
                           GetConvOpNameSuffix(*conv.gpu_operation));
  conv.op_name = absl::StrCat(attr.op_name, "_convolution_winograd_3x3");
  conv.gpu_operation->flops_ = GetConvolutionWinograd3x3TileNxNFlops(
      output_shape, weights.shape, tile_size_outer);

  OperationDef winograd_down_def;
  winograd_down_def.src_tensors.push_back(dst_transformed.tensor_desc);
  winograd_down_def.dst_tensors.push_back(dst.tensor_desc);
  winograd_down.inputs = {dst_transformed.id};
  winograd_down.outputs = {dst.id};
  auto bias_copy = attr.bias;
  if (bias_copy.shape.v < weights.shape.o) {
    bias_copy.shape = Linear(weights.shape.o);
    bias_copy.data.resize(weights.shape.o);
  }
  winograd_down.gpu_operation = SelectWinograd3x3Backward(
      gpu_info_, winograd_down_def, tile_size_outer, bias_copy);
  winograd_down.name = "winograd_3x3_backward";
  winograd_down.op_name = absl::StrCat(attr.op_name, "_winograd_3x3_backward");
  return dst;
}

std::vector<GpuModelBuilder::TensorHandle> GpuModelBuilder::GetWeights(
    const Tensor<OHWI, DataType::FLOAT32>& weights,
    WeightsDescription weights_desc) {
  if (weights_manager_ && weights_manager_->ShouldOffloadPreparationToGPU(
                              gpu_info_, weights.shape,
                              WeightsManager::TargetWeightsType::kStandard)) {
    auto input_weights =
        AddTensors(GetTensorDescriptorsForWeightsLayout(weights, weights_desc));
    std::vector<ValueId> weights_ids(input_weights.size());
    for (int i = 0; i < input_weights.size(); ++i) {
      weights_ids[i] = input_weights[i].id;
    }
    weights_manager_->RegisterWeightsConversion(
        weights_ids, weights_desc, weights.shape, DataType::FLOAT32,
        weights.Data());
  }

  if (hints_.Check(ModelHints::kReuseConvWeights) && weights.id != -1) {
    int64_t weight_id = weights.id;
    if (auto it = std::find_if(
            shared_weights_.begin(), shared_weights_.end(),
            [weight_id, &weights_desc](const SharedWeights& desc) {
              return desc.weights_id == weight_id && desc.desc == weights_desc;
            });
        it != shared_weights_.end()) {
      return it->weights_handles;
    }
  }

  auto weights_descs =
      GetTensorDescriptorsForWeightsLayout(weights, weights_desc);
  std::vector<TensorHandle> weights_handles(weights_descs.size());
  for (int i = 0; i < weights_descs.size(); ++i) {
    weights_handles[i] = AddConstantTensor(std::move(weights_descs[i]));
  }
  if (hints_.Check(ModelHints::kReuseConvWeights) && weights.id != -1) {
    SharedWeights shared_weights_desc;
    shared_weights_desc.weights_id = weights.id;
    shared_weights_desc.desc = weights_desc;
    shared_weights_desc.weights_handles = weights_handles;
    shared_weights_.push_back(std::move(shared_weights_desc));
  }
  return weights_handles;
}

std::vector<GpuModelBuilder::TensorHandle> GpuModelBuilder::GetWeights(
    const Convolution2DAttributes& attr, WeightsDescription weights_desc) {
  if (weights_manager_ && weights_manager_->ShouldOffloadPreparationToGPU(
                              gpu_info_,
                              std::visit([](const auto& w) { return w.shape; },
                                         attr.weights),
                              WeightsManager::TargetWeightsType::kStandard)) {
    auto input_weights = AddTensors(GetTensorDescriptorsForWeightsLayout(
        std::visit([](const auto& w) { return w.shape; }, attr.weights),
        weights_desc));
    std::vector<ValueId> weights_ids(input_weights.size());
    for (int i = 0; i < input_weights.size(); ++i) {
      weights_ids[i] = input_weights[i].id;
    }
    if (std::holds_alternative<Tensor<OHWI, DataType::INT4>>(attr.weights)) {
      const auto& weights =
          std::get<Tensor<OHWI, DataType::INT4>>(attr.weights);
      ABSL_CHECK(!attr.scale.empty())
          << "Quantization scale is not provided for INT4 weights.";
      weights_manager_->RegisterWeightsConversion(
          weights_ids, weights_desc, weights.shape, DataType::INT4,
          weights.Data(), absl::MakeSpan(attr.scale.Data(), attr.scale.size()),
          absl::MakeSpan(attr.zero_point.Data(), attr.zero_point.size()));
    } else if (std::holds_alternative<Tensor<OHWI, DataType::INT8>>(
                   attr.weights)) {
      const auto& weights =
          std::get<Tensor<OHWI, DataType::INT8>>(attr.weights);
      ABSL_CHECK(!attr.scale.empty())
          << "Quantization scale is not provided for INT8 weights.";
      weights_manager_->RegisterWeightsConversion(
          weights_ids, weights_desc, weights.shape, DataType::INT8,
          weights.Data(), absl::MakeSpan(attr.scale.Data(), attr.scale.size()),
          absl::MakeSpan(attr.zero_point.Data(), attr.zero_point.size()));
    } else {
      const Tensor<OHWI, DataType::FLOAT32>& weights = GetFloatWeights(attr);
      weights_manager_->RegisterWeightsConversion(
          weights_ids, weights_desc, weights.shape, DataType::FLOAT32,
          weights.Data());
    }
    return input_weights;
  }

  const Tensor<OHWI, DataType::FLOAT32>& weights = GetFloatWeights(attr);
  return GetWeights(weights, weights_desc);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Convolution(
    const GpuModelBuilder::TensorHandle& src,
    const Convolution2DAttributes& attr) {
  const auto& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  BHWC dst_shape = CalculateOutputShape(src.tensor_desc.GetBHWCShape(), attr);
  GpuModelBuilder::TensorHandle dst =
      AddTensor(dst_shape, src.tensor_desc.GetDataType());
  const int total_spatial_size = dst_shape.b * dst_shape.h * dst_shape.w;
  if (total_spatial_size <=
          GetRecommendedMaxTotalSpatialSize(
              gpu_info_, GetConvPrecision(src.tensor_desc.GetDataType())) &&
      IsConvEquivalentToFullyConnected(attr)) {
    FullyConnectedAttributes fc_attr;
    fc_attr.op_name = attr.op_name;
    fc_attr.weights = GetFloatWeights(attr);
    fc_attr.bias = attr.bias;
    return FullyConnected(src, fc_attr);
  }

  bool can_use_winograd = !hints_.Check(ModelHints::kNoWinogradOptimizations);
  if (can_use_winograd && hints_.winograd_runtime_weights_conversion) {
    can_use_winograd = weights_shape.i % 4 == 0 && weights_shape.o % 4 == 0;
  }
  constexpr int kTileSize = 6;
  if (can_use_winograd && IsSuitableForWinograd3x3(attr) &&
      IsRecommendedForWinograd3x3TileNxN(attr, gpu_info_, dst_shape,
                                         kTileSize)) {
    return WinoConvolution(src, attr, kTileSize);
  }

  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);
  std::string op_name =
      absl::StrCat("convolution", weights_shape.h, "x", weights_shape.w);

  const bool prefer_weights_preparation_on_gpu =
      weights_manager_ && weights_manager_->ShouldOffloadPreparationToGPU(
                              gpu_info_, weights_shape,
                              WeightsManager::TargetWeightsType::kStandard);
  // If prefer_external_weights is true, we'll first select the best-performing
  // kernel, and then use external weights if the selected kernel supports it.
  // The kernels above this line do not support both internal and external
  // weights, and have better performance than the kernels below, so this
  // preference will not be accounted if they are selected.
  const bool prefer_external_weights =
      (hints_.Check(ModelHints::kReuseConvWeights) &&
       std::visit([](const auto& w) { return w.id; }, attr.weights) != -1) ||
      prefer_weights_preparation_on_gpu;
  if (prefer_external_weights) {
    WeightsDescription weights_desc;
    TensorHandle bias_th;
    TensorDescriptor* bias_tensor_desc_ptr = nullptr;
    if (!attr.bias.data.empty()) {
      bias_th = AddConstantTensor(attr.bias, dst.tensor_desc.GetDataType());
      bias_tensor_desc_ptr = &bias_th.tensor_desc;
    }
    auto op = SelectConvolutionWithExternalWeights(
        attr, bias_tensor_desc_ptr, dst_shape, gpu_info_, op_def,
        GetConvPrecision(src.tensor_desc.GetDataType()), hints_, &weights_desc,
        /*src_exp=*/nullptr,
        /*different_weights_for_height=*/false);
    op->flops_ = GetConvolutionFlops(dst_shape, weights_shape);
    const auto weights_handles = GetWeights(attr, weights_desc);
    std::vector<ValueId> input_ids = {src.id};
    for (const auto& weight : weights_handles) {
      input_ids.push_back(weight.id);
    }
    if (!attr.bias.data.empty()) {
      input_ids.push_back(bias_th.id);
    }
    absl::StrAppend(&op_name, GetConvOpNameSuffix(*op));
    AddGpuOperation(input_ids, {dst.id}, std::move(op), op_name);
    return dst;
  }

  auto op = SelectConvolution(attr, dst_shape, gpu_info_, op_def,
                              GetConvPrecision(src.tensor_desc.GetDataType()),
                              hints_);
  op->flops_ = GetConvolutionFlops(dst_shape, weights_shape);
  absl::StrAppend(&op_name, GetConvOpNameSuffix(*op));
  AddGpuOperation(std::vector<ValueId>{src.id}, std::vector<ValueId>{dst.id},
                  std::move(op), op_name);
  return dst;
}

absl::StatusOr<GpuModelBuilder::TensorHandle> GpuModelBuilder::Convolution(
    const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& weights,
    const GpuModelBuilder::TensorHandle* bias,
    const Convolution2DAttributes& attr) {
  const auto& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  BHWC dst_shape = CalculateOutputShape(src.tensor_desc.GetBHWCShape(), attr);
  const int total_spatial_size = dst_shape.b * dst_shape.h * dst_shape.w;
  if (total_spatial_size <=
          GetRecommendedMaxTotalSpatialSize(
              gpu_info_, GetConvPrecision(src.tensor_desc.GetDataType())) &&
      IsConvEquivalentToFullyConnected(attr) &&
      IsFullyConnectedWeightsAreSpatialTensorSupported(weights_shape)) {
    return FullyConnectedExternalSpatialWeights(src, weights, bias);
  }

  GpuModelBuilder::TensorHandle dst =
      AddTensor(dst_shape, src.tensor_desc.GetDataType());
  std::unique_ptr<GPUOperation> conv_op;
  WeightsDescription conv_weights_desc;
  {
    OperationDef op_def;
    op_def.src_tensors.push_back(src.tensor_desc);
    op_def.dst_tensors.push_back(dst.tensor_desc);

    const TensorDescriptor* bias_desc_ptr = bias ? &bias->tensor_desc : nullptr;
    conv_op = SelectConvolutionWithExternalWeights(
        attr, bias_desc_ptr, dst_shape, gpu_info_, op_def,
        GetConvPrecision(src.tensor_desc.GetDataType()), hints_,
        &conv_weights_desc, /*src_exp=*/nullptr,
        /*different_weights_for_height=*/false);
  }

  const DataType float_type = src.tensor_desc.GetDataType();
  TensorHandle scale_handle;
  const TensorHandle* scale_handle_ptr = nullptr;
  if (!attr.scale.data.empty()) {
    scale_handle = GetWeightsScale(attr.scale, float_type);
    scale_handle_ptr = &scale_handle;
  }
  TensorHandle zp_handle;
  const TensorHandle* zp_handle_ptr = nullptr;
  if (!attr.zero_point.data.empty()) {
    zp_handle = GetWeightsZeroPoint(attr.zero_point, float_type);
    zp_handle_ptr = &zp_handle;
  }

  std::vector<TensorHandle> conv_weights =
      WeightsConversion(weights, Layout::OHWI, conv_weights_desc, weights_shape,
                        scale_handle_ptr, zp_handle_ptr);

  gpu_model_.nodes.push_back({});
  auto& conv_node = gpu_model_.nodes.back();
  conv_node.inputs = {src.id};
  for (const auto& conv_weight : conv_weights) {
    conv_node.inputs.push_back(conv_weight.id);
  }
  if (bias) {
    conv_node.inputs.push_back(bias->id);
  }
  conv_node.outputs = {dst.id};
  conv_node.gpu_operation = std::move(conv_op);
  conv_node.name =
      absl::StrCat("convolution", weights_shape.h, "x", weights_shape.w,
                   GetConvOpNameSuffix(*conv_node.gpu_operation));
  conv_node.gpu_operation->flops_ =
      GetConvolutionFlops(dst_shape, weights_shape);
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::ConvolutionTransposed(
    const GpuModelBuilder::TensorHandle& src,
    const ConvolutionTransposedAttributes& attr) {
  const BHWC src_shape = src.tensor_desc.GetBHWCShape();
  BHWC dst_shape = CalculateOutputShape(src_shape, attr);
  GpuModelBuilder::TensorHandle dst =
      AddTensor(dst_shape, src.tensor_desc.GetDataType());
  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.op_name = attr.op_name;
  gpu_node.inputs = {src.id};
  gpu_node.outputs = {dst.id};
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);
  gpu_node.gpu_operation = SelectConvolutionTransposed(
      attr, gpu_info_, op_def, GetConvPrecision(src.tensor_desc.GetDataType()));
  gpu_node.name = absl::StrCat("conv_transposed", attr.weights.shape.h, "x",
                               attr.weights.shape.w,
                               GetConvOpNameSuffix(*gpu_node.gpu_operation));
  gpu_node.gpu_operation->flops_ =
      GetConvolutionTransposedFlops(src_shape, attr.weights.shape);
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::ConvolutionTransposed(
    const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& weights,
    const ConvolutionTransposedAttributes& attr) {
  const BHWC src_shape = src.tensor_desc.GetBHWCShape();
  BHWC dst_shape = CalculateOutputShape(src_shape, attr);
  GpuModelBuilder::TensorHandle dst =
      AddTensor(dst_shape, src.tensor_desc.GetDataType());

  std::unique_ptr<GPUOperation> conv_op;
  WeightsDescription conv_weights_desc;
  {
    OperationDef op_def;
    op_def.src_tensors.push_back(src.tensor_desc);
    op_def.src_tensors.push_back(src.tensor_desc);
    op_def.src_tensors[1] = {src.tensor_desc.GetDataType(),
                             TensorStorageType::BUFFER, Layout::HWC};
    op_def.dst_tensors.push_back(dst.tensor_desc);

    conv_op = SelectConvolutionTransposedExternalWeights(
        attr, gpu_info_, op_def,
        GetConvPrecision(src.tensor_desc.GetDataType()), &conv_weights_desc);
  }

  std::vector<TensorHandle> conv_weights = WeightsConversion(
      weights, Layout::OHWI, conv_weights_desc, attr.weights.shape);

  gpu_model_.nodes.push_back({});
  auto& conv_node = gpu_model_.nodes.back();
  conv_node.inputs = {src.id};
  for (const auto& conv_weight : conv_weights) {
    conv_node.inputs.push_back(conv_weight.id);
  }
  conv_node.outputs = {dst.id};
  conv_node.gpu_operation = std::move(conv_op);
  conv_node.name = absl::StrCat("conv_transposed", attr.weights.shape.h, "x",
                                attr.weights.shape.w,
                                GetConvOpNameSuffix(*conv_node.gpu_operation));
  conv_node.gpu_operation->flops_ =
      GetConvolutionTransposedFlops(src_shape, attr.weights.shape);
  return dst;
}

WeightsDescription GpuModelBuilder::GetFullyConnectedWeightsDesc(
    DataType data_type, const OHWI& weights_shape) const {
  return ml_drift::GetFullyConnectedWeightsDesc(data_type, weights_shape);
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
GpuModelBuilder::FullyConnectedExternalWeights(
    const TensorHandle& src, const Weights& weights, const TensorHandle* biases,
    const TensorHandle* src_exp, const ConvRuntimeCheckDesc& runtime_check,
    const TensorHandle* runtime_check_tensor) {
  const auto conv_precision = GetConvPrecision(src.tensor_desc.GetDataType());
  BHWC dst_shape = src.tensor_desc.GetBHWCShape();
  dst_shape.c = weights.shape.o;
  const bool different_weights_for_height = weights.shape.h != 1;

  if (biases && biases->tensor_desc.GetBHWCShape().c != dst_shape.c) {
    ABSL_LOG(ERROR) << "Bias tensor has different number of channels than "
                       "the output tensor.";
  }

  int total_spatial_size = dst_shape.b * dst_shape.w;
  if (!different_weights_for_height) {
    total_spatial_size *= dst_shape.h;
  }
  const TensorDescriptor* bias_desc = biases ? &biases->tensor_desc : nullptr;

  const TensorDescriptor* src_exp_td =
      src_exp ? &src_exp->tensor_desc : nullptr;
  if (total_spatial_size <=
      GetRecommendedMaxTotalSpatialSize(gpu_info_, conv_precision)) {
    GpuModelBuilder::TensorHandle dst =
        AddTensor(dst_shape, src.tensor_desc.GetDataType());
    gpu_model_.nodes.push_back({});
    auto& gpu_node = gpu_model_.nodes.back();
    gpu_node.name = "fully_connected";
    if (src_exp) {
      gpu_node.name += " + src softmax";
    }

    ExternalWeights external_weights = ToExternalWeights(weights);
    if (runtime_check.ring_o_offset_index.has_value() &&
        runtime_check.ring_size.value()) {
      external_weights.shape.o = runtime_check.ring_size.value();
    }
    if (runtime_check.ring_i_offset_index.has_value() &&
        runtime_check.ring_size.value()) {
      external_weights.shape.i = runtime_check.ring_size.value();
    }
    ABSL_ASSIGN_OR_RETURN(auto fc_op,
                          CreateFullyConnectedExternalWeights(
                              gpu_info_, conv_precision, src.tensor_desc,
                              dst.tensor_desc, external_weights, bias_desc,
                              &dst_shape, src_exp_td, runtime_check));
    gpu_node.gpu_operation =
        std::make_unique<ml_drift::FullyConnected>(std::move(fc_op));
    gpu_node.inputs = {src.id, weights.weights.id};
    if (biases) {
      gpu_node.inputs.push_back(biases->id);
    }
    if (src_exp) {
      gpu_node.inputs.push_back(src_exp->id);
    }
    if (runtime_check_tensor &&
        (runtime_check.src_end_ch_index.has_value() ||
         runtime_check.dst_end_ch_index.has_value() ||
         runtime_check.ring_o_offset_index.has_value() ||
         runtime_check.ring_i_offset_index.has_value())) {
      gpu_node.inputs.push_back(runtime_check_tensor->id);
    }
    gpu_node.outputs = {dst.id};
    gpu_node.gpu_operation->flops_ =
        GetConvolutionFlops(dst_shape, weights.shape) / weights.shape.h;
    return dst;
  } else {
    return FullyConnectedSrcFloatExternalWeightsWithConversion(
        src, weights, biases, src_exp, runtime_check, runtime_check_tensor);
  }
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
GpuModelBuilder::FullyConnectedExternalSpatialWeights(
    const TensorHandle& src, const TensorHandle& weights,
    const TensorHandle* biases) {
  const BHWC& weights_bhwc_shape = weights.tensor_desc.GetBHWCShape();
  const OHWI weights_shape = OHWI(weights_bhwc_shape.b, weights_bhwc_shape.h,
                                  weights_bhwc_shape.w, weights_bhwc_shape.c);
  BHWC dst_shape = src.tensor_desc.GetBHWCShape();
  dst_shape.c = weights_shape.o;
  GpuModelBuilder::TensorHandle dst =
      AddTensor(dst_shape, src.tensor_desc.GetDataType());

  if (biases && biases->tensor_desc.GetBHWCShape().c != dst_shape.c) {
    ABSL_LOG(ERROR) << "Bias tensor has different number of channels than "
                       "the output tensor.";
  }

  const bool different_weights_for_height = weights_shape.h != 1;
  const int total_spatial_size =
      dst_shape.b * dst_shape.w *
      (!different_weights_for_height ? dst_shape.h : 1);

  if (total_spatial_size >
      GetRecommendedMaxTotalSpatialSize(
          gpu_info_, GetConvPrecision(src.tensor_desc.GetDataType()))) {
    return absl::UnimplementedError(
        "Total spatial size is too large for FC runtime weights, should use "
        "CONVOLUTION instead.");
  }

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = "fully_connected_runtime_weights";

  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.src_tensors.push_back(weights.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);

  const TensorDescriptor* bias_desc = biases ? &biases->tensor_desc : nullptr;
  ABSL_ASSIGN_OR_RETURN(
      auto fc_op,
      CreateFullyConnectedWeightsAreSpatialTensor(
          gpu_info_, op_def, GetConvPrecision(src.tensor_desc.GetDataType()),
          weights_shape, bias_desc, &dst_shape, /*wg_size=*/nullptr));
  gpu_node.gpu_operation =
      std::make_unique<ml_drift::FullyConnected>(std::move(fc_op));
  gpu_node.inputs = {src.id, weights.id};
  if (biases) {
    gpu_node.inputs.push_back(biases->id);
  }
  gpu_node.outputs = {dst.id};
  gpu_node.gpu_operation->flops_ =
      GetConvolutionFlops(dst_shape, weights_shape) / weights_shape.h;
  return dst;
}

std::vector<GpuModelBuilder::TensorHandle> GpuModelBuilder::WeightsConversion(
    const GpuModelBuilder::TensorHandle& src_weights, Layout src_layout,
    const WeightsDescription& dst_desc, const OHWI& weights_shape,
    const TensorHandle* weights_scale, const TensorHandle* weights_zero_point) {
  const auto dsts =
      AddTensors(GetTensorDescriptorsForWeightsLayout(weights_shape, dst_desc));

  OperationDef op_def;
  op_def.src_tensors.push_back(src_weights.tensor_desc);
  for (auto& dst : dsts) {
    op_def.dst_tensors.push_back(dst.tensor_desc);
  }

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  if (src_layout == Layout::OHWI) {
    if (src_weights.tensor_desc.GetLayout() == Layout::LINEAR) {
      gpu_node.name = "ohwi";
    } else {
      gpu_node.name = "bhwc_as_ohwi";
    }
  } else if (src_layout == Layout::HWIO) {
    gpu_node.name = "bhwc_as_hwio";
  }
  gpu_node.name +=
      absl::StrCat("_", ToString(src_weights.tensor_desc.GetDataType()), "_to_",
                   ToString(dst_desc.layout), "_", ToString(dst_desc.type));
  const TensorDescriptor* weights_scale_td =
      weights_scale ? &weights_scale->tensor_desc : nullptr;
  const TensorDescriptor* weights_zero_point_td =
      weights_zero_point ? &weights_zero_point->tensor_desc : nullptr;
  gpu_node.gpu_operation = SelectConverterToConvWeights(
      gpu_info_, weights_shape, dst_desc, op_def, hints_, src_layout,
      weights_scale_td, weights_zero_point_td);
  gpu_node.inputs = {src_weights.id};
  if (weights_scale) {
    gpu_node.inputs.push_back(weights_scale->id);
  }
  if (weights_zero_point) {
    gpu_node.inputs.push_back(weights_zero_point->id);
  }
  for (auto& dst : dsts) {
    gpu_node.outputs.push_back(dst.id);
  }
  return dsts;
}

std::vector<GpuModelBuilder::TensorHandle> GpuModelBuilder::WeightsConversion(
    const TensorHandle& src_weights, const TensorHandle* weights_scale,
    const TensorHandle* weights_zero_point, const WeightsDescription& src_desc,
    const WeightsDescription& dst_desc, const OHWI& weights_shape,
    const ConvRuntimeCheckDesc& runtime_check,
    const TensorHandle* runtime_check_tensor) {
  Weights weights;
  weights.weights = src_weights;
  weights.desc = src_desc;
  weights.shape = weights_shape;
  if (weights_scale) {
    weights.scale = *weights_scale;
  }
  if (weights_zero_point) {
    weights.zero_point = *weights_zero_point;
  }
  return WeightsConversion(weights, dst_desc, runtime_check,
                           runtime_check_tensor);
}

std::vector<GpuModelBuilder::TensorHandle> GpuModelBuilder::WeightsConversion(
    const Weights& weights, const WeightsDescription& dst_desc,
    const ConvRuntimeCheckDesc& runtime_check,
    const TensorHandle* runtime_check_tensor) {
  const auto dsts =
      AddTensors(GetTensorDescriptorsForWeightsLayout(weights.shape, dst_desc));

  OperationDef op_def;
  op_def.src_tensors.push_back(weights.weights.tensor_desc);
  for (auto& dst : dsts) {
    op_def.dst_tensors.push_back(dst.tensor_desc);
  }

  ExternalWeights external_weights = ToExternalWeights(weights);

  WeightsConverter converter(gpu_info_, op_def, external_weights, dst_desc,
                             runtime_check);

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = "weights_convert_" + ToString(weights.desc.type) + "_to_" +
                  ToString(dst_desc.type);
  gpu_node.gpu_operation =
      std::make_unique<WeightsConverter>(std::move(converter));
  gpu_node.inputs = {weights.weights.id};
  if (weights.scale) {
    gpu_node.inputs.push_back(weights.scale->id);
  }
  if (weights.zero_point) {
    gpu_node.inputs.push_back(weights.zero_point->id);
  }
  if (runtime_check_tensor && (runtime_check.src_end_ch_index.has_value() ||
                               runtime_check.dst_end_ch_index.has_value() ||
                               runtime_check.ring_o_offset_index.has_value() ||
                               runtime_check.ring_i_offset_index.has_value())) {
    gpu_node.inputs.push_back(runtime_check_tensor->id);
  }
  for (auto& dst : dsts) {
    gpu_node.outputs.push_back(dst.id);
  }

  return dsts;
}

GpuModelBuilder::TensorHandle
GpuModelBuilder::FullyConnectedSrcFloatExternalWeightsWithConversion(
    const TensorHandle& src, const Weights& weights, const TensorHandle* biases,
    const TensorHandle* src_exp, const ConvRuntimeCheckDesc& runtime_check,
    const TensorHandle* runtime_check_tensor) {
  BHWC dst_shape = src.tensor_desc.GetBHWCShape();
  dst_shape.c = weights.shape.o;
  GpuModelBuilder::TensorHandle dst =
      AddTensor(dst_shape, src.tensor_desc.GetDataType());
  const bool different_weights_for_height = weights.shape.h != 1;

  const auto conv_precision =
        GetConvPrecision(src.tensor_desc.GetDataType());

  const bool use_apple_mpp =
      SupportsConvAppleMPP(gpu_info_) &&
      conv_precision == CalculationsPrecision::F16 &&
      src.tensor_desc.GetBHWCShape().c % 32 == 0;
  auto src_handle = use_apple_mpp && !src_exp ? ToDHWBCC4(src) : src;

  const bool ringed_weights = runtime_check.ring_o_offset_index.has_value() ||
                              runtime_check.ring_i_offset_index.has_value();

  const uint64_t flops =
      GetConvolutionFlops(dst_shape, weights.shape) / weights.shape.h;
  const uint64_t src_weights_size =
      weights.shape.DimensionsProduct() * SizeInBitsOf(weights.desc.type) / 8;
  const uint64_t dst_weights_size =
      weights.shape.DimensionsProduct() *
      SizeInBitsOf(src.tensor_desc.GetDataType()) / 8;
  double flops_per_byte = 32;
  // const double flops_per_byte = device_compute_gflops/device_bandwidth_gbs;
  if (gpu_info_.IsApple()) {
    // const double device_bandwidth_gbs = 120;  // in MLDrift conversion
    // const double device_compute_gflops = 13000;  // in MLDrift convolution
    // flops_per_byte ~110 for M5/A19, GPUs with NA
    // const double device_bandwidth_gbs = 75;  // in MLDrift conversion
    // const double device_compute_gflops = 2500;  // in MLDrift convolution
    // flops_per_byte ~32 for M4/A18 and below, GPUs without NA
    flops_per_byte = use_apple_mpp ? 110 : 32;
  } else if (gpu_info_.IsIntel()) {
    // TODO: sorokin MLDrift conversion has bad results for Intel GPUs.
    // Intel LNL:
    // const double device_bandwidth_gbs = 40;  // in some cases
    // const double device_compute_gflops = 10000;  // in MLDrift convolution
    flops_per_byte = 250;
  }
  // const double conv_gflops = flops * 1e-9;
  // const double conv_time_s = conv_gflops / device_compute_gflops;
  // const double weights_bytes = src_weights_size + dst_weights_size;
  // const double convert_weights_time_s = weights_gbytes /
  //                                       device_bandwidth_gbs;
  //  double conversion_cost = convert_weights_time_s * 100 / conv_time_s;
  //  (weights_gbytes / device_bandwidth_gbs) * 100 /
  //    (conv_gflops / device_compute_gflops);
  //  weights_gbytes * 100 / conv_gflops *
  //  (device_compute_gflops / device_bandwidth_gbs);
  //  weights_gbytes * 100 / conv_gflops * flops_per_byte;
  const double conversion_cost =
      100.0 * (src_weights_size + dst_weights_size) / flops * flops_per_byte;
  // if conversion cost is more than 20% of the convolution cost, it is
  // recommended to use a single convolution. Convolution with weights
  // conversion is usually ~20% slower than convolution without weights
  // conversion(on Apple GPUs in MLDrift implementation).
  const bool recommended_single_conv = !ringed_weights && conversion_cost > 20;

  std::unique_ptr<GPUOperation> conv_op;
  WeightsDescription conv_weights_desc;
  bool conv_need_scale_zp = false;
  {
    OperationDef op_def;
    op_def.src_tensors.push_back(src_handle.tensor_desc);
    op_def.dst_tensors.push_back(dst.tensor_desc);

    Convolution2DAttributes attr;
    attr.padding.prepended = HW(0, 0);
    attr.padding.appended = HW(0, 0);
    attr.strides = HW(1, 1);
    attr.dilations = HW(1, 1);
    auto& attr_weights =
        attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
    attr_weights.shape = weights.shape;

    const TensorDescriptor* src_exp_td =
        src_exp ? &src_exp->tensor_desc : nullptr;
    const TensorDescriptor* bias_td = biases ? &biases->tensor_desc : nullptr;

    ExternalWeights external_weights = ToExternalWeights(weights);

    if (recommended_single_conv && use_apple_mpp &&
        SupportsConvAppleMPP(gpu_info_, external_weights)) {
      auto conv_apple_mpp = CreateConvAppleMPPExternalWeights(
          op_def.src_tensors[0], op_def.dst_tensors[0], external_weights,
          bias_td, src_exp_td, different_weights_for_height, runtime_check);
      conv_weights_desc = weights.desc;
      conv_op = std::make_unique<ConvAppleMPP>(std::move(conv_apple_mpp));
      conv_need_scale_zp = true;
    } else if (recommended_single_conv &&
               SupportsConvWaveMatrix(gpu_info_, conv_precision,
                                      external_weights)) {
      auto conv_wave_matrix = CreateConvWaveMatrixExternalWeights(
          op_def, conv_precision, dst_shape, external_weights, gpu_info_,
          bias_td, src_exp_td, different_weights_for_height, runtime_check);
      conv_weights_desc = weights.desc;
      conv_op = std::make_unique<ConvWaveMatrix>(std::move(conv_wave_matrix));
      conv_need_scale_zp = true;
    } else if (recommended_single_conv &&
               SupportsConvGeneric(gpu_info_, conv_precision,
                                   external_weights)) {
      auto conv_generic = CreateConvGenericExternalWeights(
          gpu_info_, op_def, conv_precision, external_weights, bias_td,
          &dst_shape, src_exp_td, different_weights_for_height, runtime_check);
      conv_weights_desc = weights.desc;
      conv_op = std::make_unique<ConvGeneric>(std::move(conv_generic));
      conv_need_scale_zp = true;
    } else if (use_apple_mpp) {
      ConvAppleMPP conv_mpp = CreateConvAppleMPPExternalWeights(
          op_def.src_tensors[0], op_def.dst_tensors[0], weights.shape, bias_td,
          src_exp_td, different_weights_for_height, runtime_check);
      conv_weights_desc = conv_mpp.GetWeightsDescription();
      conv_op = std::make_unique<ConvAppleMPP>(std::move(conv_mpp));
    } else {
      conv_op = SelectConvolutionWithExternalWeights(
          attr, bias_td, dst_shape, gpu_info_, op_def, conv_precision, hints_,
          &conv_weights_desc, src_exp_td, different_weights_for_height,
          runtime_check);
    }
  }

  std::vector<TensorHandle> conv_weights = {weights.weights};
  if (ringed_weights || conv_weights_desc != weights.desc) {
    conv_weights = WeightsConversion(weights, conv_weights_desc, runtime_check,
                                     runtime_check_tensor);
  }

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = absl::StrCat("convolution", GetConvOpNameSuffix(*conv_op));
  if (src_exp) {
    gpu_node.name += " + src softmax";
  }
  gpu_node.inputs.push_back(src_handle.id);
  for (const auto& conv_weight : conv_weights) {
    gpu_node.inputs.push_back(conv_weight.id);
  }
  if (conv_need_scale_zp && weights.scale) {
    gpu_node.inputs.push_back(weights.scale->id);
  }
  if (conv_need_scale_zp && weights.zero_point) {
    gpu_node.inputs.push_back(weights.zero_point->id);
  }
  if (biases) {
    gpu_node.inputs.push_back(biases->id);
  }
  if (src_exp) {
    gpu_node.inputs.push_back(src_exp->id);
  }
  if (runtime_check_tensor && (runtime_check.src_end_ch_index.has_value() ||
                               runtime_check.dst_end_ch_index.has_value())) {
    gpu_node.inputs.push_back(runtime_check_tensor->id);
  }

  gpu_node.outputs = {dst.id};
  gpu_node.gpu_operation = std::move(conv_op);
  gpu_node.gpu_operation->flops_ = flops;

  return dst;
}

std::vector<GpuModelBuilder::TensorHandle> GpuModelBuilder::Quantize(
    const TensorHandle& src, PackedType quantized_type, bool calculate_sum) {
  const BHWC src_shape = src.tensor_desc.GetBHWCShape();
  auto quantized = AddTensor(GetShapeForPackedType(src_shape, quantized_type),
                             ToSpatialTensorType(quantized_type));
  const int params_count = calculate_sum ? 3 : 2;
  // Some of the parameters are accumulated values and can easily overflow in
  // fp16. Using fp32 type for them.
  auto src_params =
      AddTensor(BHWC(src_shape.b, src_shape.h, src_shape.w, params_count),
                DataType::FLOAT32);

  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(quantized.tensor_desc);
  op_def.dst_tensors.push_back(src_params.tensor_desc);
  auto quant_op = CreateQuantization(op_def, quantized_type, gpu_info_,
                                     src_shape, calculate_sum);

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = "quantize_" + ToString(src.tensor_desc.GetDataType()) +
                  "_to_" + ToString(quantized_type);
  gpu_node.inputs = {src.id};
  gpu_node.outputs = {quantized.id, src_params.id};
  gpu_node.gpu_operation = std::move(quant_op);
  return {quantized, src_params};
}

GpuModelBuilder::TensorHandle GpuModelBuilder::ToDHWBCC4(
    const TensorHandle& src) {
  TensorDescriptor tensor_desc =
      TensorDescriptor{src.tensor_desc.GetDataType(), TensorStorageType::BUFFER,
                       src.tensor_desc.GetLayout(),
                       TensorDescriptor::PhysicalLayout1D::kDHWBCC4};
  tensor_desc.SetBHWCShape(src.tensor_desc.GetBHWCShape());
  auto new_tensor = AddTensor(tensor_desc);
  AddGpuOperation({src}, {new_tensor},
                  std::make_unique<GPUOperation>(CreateElementwiseOneInput(
                      gpu_info_, src.tensor_desc, new_tensor.tensor_desc,
                      OperationType::COPY)),
                  "to_dhwbcc4");
  return new_tensor;
}

GpuModelBuilder::TensorHandle
GpuModelBuilder::FullyConnectedInt8QuantizedWithSrcQuantization(
    const TensorHandle& src, const Weights& weights, const TensorHandle* biases,
    WeightsDescription* conv_weights_desc_ptr) {
  const BHWC src_shape = src.tensor_desc.GetBHWCShape();
  const BHWC dst_shape =
      BHWC(src_shape.b, src_shape.h, src_shape.w, weights.shape.o);
  auto dst = AddTensor(dst_shape, src.tensor_desc.GetDataType());

  const auto src_packed_type = GetConvolutionInt8SrcType(gpu_info_, src_shape);
  const bool use_uint8_math = gpu_info_.IsPowerVR();
  const bool calculate_sum = weights.zero_point.has_value() || use_uint8_math;
  auto outputs = Quantize(src, src_packed_type, calculate_sum);
  auto src_quantized = outputs[0];
  if (SupportsConvAppleMPP(gpu_info_)) {
    src_quantized = ToDHWBCC4(src_quantized);
  }
  auto src_params = outputs[1];

  std::unique_ptr<GPUOperation> conv_int8;
  WeightsDescription conv_weights_desc;
  {
    const DataType dst_conv_type =
        use_uint8_math ? DataType::UINT32 : DataType::INT32;
    TensorDescriptor interm_dst = {dst_conv_type, default_storage_,
                                   dst.tensor_desc.GetLayout()};
    OperationDef op_def;
    op_def.src_tensors.push_back(src_quantized.tensor_desc);
    op_def.dst_tensors.push_back(interm_dst);

    const TensorDescriptor* zero_point_desc =
        weights.zero_point.has_value() ? &weights.zero_point->tensor_desc
                                       : nullptr;
    conv_int8 =
        SelectConvolutionInt8(gpu_info_, op_def, src_packed_type, weights.shape,
                              dst_shape, &conv_weights_desc);

    auto dequant_op = CreateDequantization(
        weights.shape, gpu_info_, interm_dst, dst.tensor_desc,
        src_params.tensor_desc, weights.sum_i->tensor_desc,
        weights.scale->tensor_desc, zero_point_desc);
    auto s = conv_int8->AddOperation(gpu_info_, &dequant_op);
    if (!s.ok()) {
      ABSL_LOG(ERROR) << s.message();
    }
  }
  const std::string dequant_name =
      "dequantize_to_" + ToString(dst.tensor_desc.GetDataType());

  std::vector<TensorHandle> conv_weights;
  conv_weights = {weights.weights};
  if (weights.desc.layout != WeightsLayout::kUnknown) {
    if (conv_weights_desc != weights.desc) {
      conv_weights = WeightsConversion(weights, conv_weights_desc);
    }
  } else if (conv_weights_desc_ptr != nullptr) {
    *conv_weights_desc_ptr = conv_weights_desc;
  }

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name =
      absl::StrCat("convolution_int8", GetConvOpNameSuffix(*conv_int8), " -> ",
                   dequant_name);
  gpu_node.inputs.push_back(src_quantized.id);
  for (const auto& conv_weight : conv_weights) {
    gpu_node.inputs.push_back(conv_weight.id);
  }
  gpu_node.inputs.push_back(src_params.id);
  gpu_node.inputs.push_back(weights.sum_i->id);
  gpu_node.inputs.push_back(weights.scale->id);
  if (weights.zero_point) {
    gpu_node.inputs.push_back(weights.zero_point->id);
  }
  gpu_node.outputs = {dst.id};
  gpu_node.gpu_operation = std::move(conv_int8);
  gpu_node.gpu_operation->flops_ =
      GetConvolutionFlops(dst_shape, weights.shape);
  if (biases) {
    dst = Add(dst, *biases);
  }
  return dst;
}

GpuModelBuilder::TensorHandle
GpuModelBuilder::FullyConnectedInt4QuantizedWithSrcQuantization(
    const TensorHandle& src, const Weights& weights,
    const TensorHandle* biases) {
  const BHWC src_shape = src.tensor_desc.GetBHWCShape();
  const BHWC dst_shape =
      BHWC(src_shape.b, src_shape.h, src_shape.w, weights.shape.o);
  auto dst = AddTensor(dst_shape, src.tensor_desc.GetDataType());

  auto outputs = Quantize(src, GetConvolutionInt4SrcType(gpu_info_, src_shape),
                          /*calculate_sum=*/weights.zero_point.has_value());
  auto src_quantized = outputs[0];
  auto src_params = outputs[1];

  std::unique_ptr<GPUOperation> conv_int4;
  WeightsDescription conv_weights_desc;
  {
    TensorDescriptor interm_dst = {DataType::INT32, default_storage_,
                                   dst.tensor_desc.GetLayout()};
    OperationDef op_def;
    op_def.src_tensors.push_back(src_quantized.tensor_desc);
    op_def.dst_tensors.push_back(interm_dst);

    const TensorDescriptor* zero_point_desc =
        weights.zero_point.has_value() ? &weights.zero_point->tensor_desc
                                       : nullptr;
    conv_int4 = SelectConvolutionInt4(gpu_info_, op_def, weights.shape,
                                      dst_shape, &conv_weights_desc);

    auto dequant_op = CreateDequantization(
        weights.shape, gpu_info_, interm_dst, dst.tensor_desc,
        src_params.tensor_desc, weights.sum_i->tensor_desc,
        weights.scale->tensor_desc, zero_point_desc);
    auto s = conv_int4->AddOperation(gpu_info_, &dequant_op);
  }
  const std::string dequant_name =
      "dequantize_to_" + ToString(dst.tensor_desc.GetDataType());

  std::vector<TensorHandle> conv_weights = {weights.weights};
  if (conv_weights_desc != weights.desc) {
    conv_weights = WeightsConversion(weights, conv_weights_desc);
  }

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name =
      absl::StrCat("convolution_int4", GetConvOpNameSuffix(*conv_int4), " -> ",
                   dequant_name);
  gpu_node.inputs.push_back(src_quantized.id);
  for (const auto& conv_weight : conv_weights) {
    gpu_node.inputs.push_back(conv_weight.id);
  }
  gpu_node.inputs.push_back(src_params.id);
  gpu_node.inputs.push_back(weights.sum_i->id);
  gpu_node.inputs.push_back(weights.scale->id);
  if (weights.zero_point) {
    gpu_node.inputs.push_back(weights.zero_point->id);
  }
  gpu_node.outputs = {dst.id};
  gpu_node.gpu_operation = std::move(conv_int4);
  gpu_node.gpu_operation->flops_ =
      GetConvolutionFlops(dst_shape, weights.shape);
  if (biases) {
    dst = Add(dst, *biases);
  }
  return dst;
}

WeightsDescription GpuModelBuilder::GetFullyConnectedInt8WeightsDesc(
    const OHWI& weights_shape) {
  return ml_drift::GetFullyConnectedInt8WeightsDesc(
      gpu_info_, weights_shape,
      hints_.Check(ModelHints::kPreferTextureWeights));
}

GpuModelBuilder::TensorHandle GpuModelBuilder::EmbeddingLookup(
    const TensorHandle& src, const Weights& weights, DataType dst_type,
    Axis lookup_axis) {
  BHWC new_shape = src.tensor_desc.GetBHWCShape();
  new_shape.c = weights.shape.i;
  auto dst = AddTensor(new_shape, dst_type);

  const TensorDescriptor* weights_scale_desc =
      weights.scale ? &weights.scale->tensor_desc : nullptr;
  const TensorDescriptor* weights_zero_point_desc =
      weights.zero_point ? &weights.zero_point->tensor_desc : nullptr;
  auto gpu_op = CreateEmbeddingLookupExternalWeights(
      src.tensor_desc, dst.tensor_desc, weights.weights.tensor_desc,
      weights.desc, weights.shape, weights_scale_desc, weights_zero_point_desc,
      lookup_axis);
  std::vector<ValueId> src_ids = {src.id, weights.weights.id};
  gpu_op.read_size_ = src.tensor_desc.GetMemorySizeInBytes();
  // EmbeddingLookup not reading entire weights tensor.
  gpu_op.read_size_ += dst.tensor_desc.GetMemorySizeInBytes() /
                       SizeInBitsOf(dst_type) * SizeInBitsOf(weights.desc.type);
  if (weights.scale) {
    src_ids.push_back(weights.scale->id);
  }
  if (weights.zero_point) {
    src_ids.push_back(weights.zero_point->id);
  }
  AddGpuOperation({src_ids}, {dst.id},
                  std::make_unique<GPUOperation>(std::move(gpu_op)),
                  "embedding_lookup");
  return dst;
}

GpuModelBuilder::TensorHandle
GpuModelBuilder::FullyConnectedInt8ExternalWeights(const TensorHandle& src,
                                                   const Weights& weights,
                                                   const TensorHandle* biases) {
  return FullyConnectedInt8ExternalWeights(src, weights, biases, nullptr, {},
                                           nullptr);
}

GpuModelBuilder::TensorHandle
GpuModelBuilder::FullyConnectedInt8ExternalWeights(
    const TensorHandle& src, const Weights& weights, const TensorHandle* biases,
    const TensorHandle* src_exp, const ConvRuntimeCheckDesc& runtime_check,
    const TensorHandle* runtime_check_tensor) {
  const auto conv_precision = GetConvPrecision(src.tensor_desc.GetDataType());
  BHWC dst_shape = src.tensor_desc.GetBHWCShape();
  dst_shape.c = weights.shape.o;
  GpuModelBuilder::TensorHandle dst =
      AddTensor(dst_shape, src.tensor_desc.GetDataType());

  if (biases && biases->tensor_desc.GetBHWCShape().c != dst_shape.c) {
    ABSL_LOG(ERROR) << "Bias tensor has different number of channels than "
                       "the output tensor.";
  }

  ABSL_QCHECK(weights.scale.has_value())
      << "FullyConnectedInt8ExternalWeights requires weights.scale";
  ABSL_QCHECK(weights.scale_zp_shape.i > 0)
      << "FullyConnectedInt8ExternalWeights requires weights.scale_zp_shape";

  const bool different_weights_for_height = weights.shape.h != 1;
  int total_spatial_size = dst_shape.b * dst_shape.w;
  if (!different_weights_for_height) {
    total_spatial_size *= dst_shape.h;
  }

  const TensorDescriptor* bias_desc = biases ? &biases->tensor_desc : nullptr;
  const bool grouped_quantization = weights.scale_zp_shape.i != 1;

  const bool ringed_weights = runtime_check.ring_o_offset_index.has_value() ||
                              runtime_check.ring_i_offset_index.has_value();

  if (total_spatial_size <=
          GetRecommendedMaxTotalSpatialSize(gpu_info_, conv_precision) ||
      ringed_weights) {
    gpu_model_.nodes.push_back({});
    auto& gpu_node = gpu_model_.nodes.back();
    gpu_node.name = "fc1x1_int8_weights";
    if (src_exp) {
      gpu_node.name += " + src softmax";
    }

    const TensorDescriptor* src_exp_td =
        src_exp ? &src_exp->tensor_desc : nullptr;
    ExternalWeights external_weights = ToExternalWeights(weights);
    if (runtime_check.ring_o_offset_index.has_value() &&
        runtime_check.ring_size.value()) {
      external_weights.shape.o = runtime_check.ring_size.value();
    }
    if (runtime_check.ring_i_offset_index.has_value() &&
        runtime_check.ring_size.value()) {
      external_weights.shape.i = runtime_check.ring_size.value();
    }
    auto fc_op = CreateFullyConnectedExternalWeights(
        gpu_info_, conv_precision, src.tensor_desc, dst.tensor_desc,
        external_weights, bias_desc, &dst_shape, src_exp_td, runtime_check);
    ABSL_QCHECK_OK(fc_op);

    gpu_node.gpu_operation =
        std::make_unique<ml_drift::FullyConnected>(std::move(*fc_op));

    gpu_node.inputs = {src.id, weights.weights.id, weights.scale->id};
    if (weights.zero_point) {
      gpu_node.inputs.push_back(weights.zero_point->id);
    }
    if (biases) {
      gpu_node.inputs.push_back(biases->id);
    }
    if (src_exp) {
      gpu_node.inputs.push_back(src_exp->id);
    }
    if (runtime_check_tensor &&
        (runtime_check.src_end_ch_index.has_value() ||
         runtime_check.dst_end_ch_index.has_value() || ringed_weights)) {
      gpu_node.inputs.push_back(runtime_check_tensor->id);
    }
    gpu_node.outputs = {dst.id};
    gpu_node.gpu_operation->flops_ =
        GetConvolutionFlops(dst_shape, weights.shape);
    return dst;
  } else if (!grouped_quantization && weights.sum_i.has_value() &&
             SupportsConvolutionInt8(gpu_info_,
                                     src.tensor_desc.GetBHWCShape()) &&
             !hints_.Check(ModelHints::kDisallow8bitConvs)) {
    ABSL_QCHECK(!runtime_check.HasValues())
        << "The support for runtime_check need to be implemented.";
    return FullyConnectedInt8QuantizedWithSrcQuantization(src, weights, biases);
  } else {
    return FullyConnectedSrcFloatExternalWeightsWithConversion(
        src, weights, biases, src_exp, runtime_check, runtime_check_tensor);
  }
}

WeightsDescription GpuModelBuilder::GetFullyConnectedInt4WeightsDesc(
    const OHWI& weights_shape) {
  return ml_drift::GetFullyConnectedInt4WeightsDesc(
      gpu_info_, weights_shape,
      hints_.Check(ModelHints::kPreferTextureWeights));
}

GpuModelBuilder::TensorHandle
GpuModelBuilder::FullyConnectedInt4ExternalWeights(const TensorHandle& src,
                                                   const Weights& weights,
                                                   const TensorHandle* biases) {
  BHWC dst_shape = src.tensor_desc.GetBHWCShape();
  dst_shape.c = weights.shape.o;
  GpuModelBuilder::TensorHandle dst =
      AddTensor(dst_shape, src.tensor_desc.GetDataType());

  if (biases && biases->tensor_desc.GetBHWCShape().c != dst_shape.c) {
    ABSL_LOG(ERROR) << "Bias tensor has different number of channels than "
                       "the output tensor.";
  }

  ABSL_QCHECK(weights.scale.has_value())
      << "FullyConnectedInt4ExternalWeights requires weights.scale";
  ABSL_QCHECK(weights.scale_zp_shape.i > 0)
      << "FullyConnectedInt4ExternalWeights requires weights.scale_zp_shape";

  const int total_spatial_size = dst_shape.b * dst_shape.h * dst_shape.w;
  const TensorDescriptor* bias_desc = biases ? &biases->tensor_desc : nullptr;
  const bool grouped_quantization = weights.scale_zp_shape.i != 1;

  if (total_spatial_size <=
      GetRecommendedMaxTotalSpatialSize(
          gpu_info_, GetConvPrecision(src.tensor_desc.GetDataType()))) {
    gpu_model_.nodes.push_back({});
    auto& gpu_node = gpu_model_.nodes.back();
    gpu_node.name = "fc1x1_int4_weights";

    ExternalWeights external_weights = ToExternalWeights(weights);
    auto fc_op = CreateFullyConnectedExternalWeights(
        gpu_info_, GetConvPrecision(src.tensor_desc.GetDataType()),
        src.tensor_desc, dst.tensor_desc, external_weights, bias_desc,
        &dst_shape);
    ABSL_QCHECK_OK(fc_op);

    gpu_node.gpu_operation =
        std::make_unique<ml_drift::FullyConnected>(std::move(*fc_op));
    gpu_node.inputs = {src.id, weights.weights.id, weights.scale->id};
    if (weights.zero_point) {
      gpu_node.inputs.push_back(weights.zero_point->id);
    }
    if (biases) {
      gpu_node.inputs.push_back(biases->id);
    }
    gpu_node.outputs = {dst.id};
    gpu_node.gpu_operation->flops_ =
        GetConvolutionFlops(dst_shape, weights.shape);
  } else if (weights.sum_i.has_value() &&
             SupportsConvolutionInt4(gpu_info_,
                                     src.tensor_desc.GetBHWCShape()) &&
             hints_.Check(ModelHints::kAllow4bitConvs)) {
    return FullyConnectedInt4QuantizedWithSrcQuantization(src, weights, biases);
  } else if (!grouped_quantization && weights.sum_i.has_value() &&
             SupportsConvolutionInt8(gpu_info_,
                                     src.tensor_desc.GetBHWCShape()) &&
             !hints_.Check(ModelHints::kDisallow8bitConvs)) {
    return FullyConnectedInt8QuantizedWithSrcQuantization(src, weights, biases);
  } else {
    return FullyConnectedSrcFloatExternalWeightsWithConversion(src, weights,
                                                               biases);
  }
  return dst;
}

WeightsDescription GpuModelBuilder::GetFullyConnectedInt2WeightsDesc(
    const OHWI& weights_shape) {
  return ml_drift::GetFullyConnectedInt2WeightsDesc(
      gpu_info_, weights_shape,
      hints_.Check(ModelHints::kPreferTextureWeights));
}

GpuModelBuilder::TensorHandle
GpuModelBuilder::FullyConnectedInt2ExternalWeights(const TensorHandle& src,
                                                   const Weights& weights,
                                                   const TensorHandle* biases) {
  BHWC dst_shape = src.tensor_desc.GetBHWCShape();
  dst_shape.c = weights.shape.o;
  GpuModelBuilder::TensorHandle dst =
      AddTensor(dst_shape, src.tensor_desc.GetDataType());

  if (biases && biases->tensor_desc.GetBHWCShape().c != dst_shape.c) {
    ABSL_LOG(ERROR) << "Bias tensor has different number of channels than "
                       "the output tensor.";
  }

  ABSL_QCHECK(weights.scale.has_value())
      << "FullyConnectedInt2ExternalWeights requires weights.scale";
  ABSL_QCHECK(weights.scale_zp_shape.i > 0)
      << "FullyConnectedInt2ExternalWeights requires weights.scale_zp_shape";

  const int total_spatial_size = dst_shape.b * dst_shape.h * dst_shape.w;
  const TensorDescriptor* bias_desc = biases ? &biases->tensor_desc : nullptr;
  const bool grouped_quantization = weights.scale_zp_shape.i != 1;

  if (total_spatial_size <=
      GetRecommendedMaxTotalSpatialSize(
          gpu_info_, GetConvPrecision(src.tensor_desc.GetDataType()))) {
    gpu_model_.nodes.push_back({});
    auto& gpu_node = gpu_model_.nodes.back();
    gpu_node.name = "fc1x1_int2_weights";

    ExternalWeights external_weights = ToExternalWeights(weights);
    auto fc_op = CreateFullyConnectedExternalWeights(
        gpu_info_, GetConvPrecision(src.tensor_desc.GetDataType()),
        src.tensor_desc, dst.tensor_desc, external_weights, bias_desc,
        &dst_shape);
    ABSL_QCHECK_OK(fc_op);

    gpu_node.gpu_operation =
        std::make_unique<ml_drift::FullyConnected>(std::move(*fc_op));
    gpu_node.inputs = {src.id, weights.weights.id, weights.scale->id};
    if (weights.zero_point) {
      gpu_node.inputs.push_back(weights.zero_point->id);
    }
    if (biases) {
      gpu_node.inputs.push_back(biases->id);
    }
    gpu_node.outputs = {dst.id};
    gpu_node.gpu_operation->flops_ =
        GetConvolutionFlops(dst_shape, weights.shape);
    return dst;
  } else if (weights.sum_i.has_value() &&
             SupportsConvolutionInt4(gpu_info_,
                                     src.tensor_desc.GetBHWCShape()) &&
             hints_.Check(ModelHints::kAllow4bitConvs)) {
    return FullyConnectedInt4QuantizedWithSrcQuantization(src, weights, biases);
  } else if (!grouped_quantization && weights.sum_i.has_value() &&
             SupportsConvolutionInt8(gpu_info_,
                                     src.tensor_desc.GetBHWCShape()) &&
             !hints_.Check(ModelHints::kDisallow8bitConvs)) {
    return FullyConnectedInt8QuantizedWithSrcQuantization(src, weights, biases);
  } else {
    return FullyConnectedSrcFloatExternalWeightsWithConversion(src, weights,
                                                               biases);
  }
  return dst;
}

GpuModelBuilder::Weights GpuModelBuilder::GetWeights(
    const std::variant<Tensor<OHWI, DataType::INT8>,
                       Tensor<OHWI, DataType::INT2>>& weights) {
  const OHWI weights_shape =
      std::visit([](const auto& w) { return w.shape; }, weights);

  WeightsDescription weights_desc = ml_drift::GetFullyConnectedInt2WeightsDesc(
      gpu_info_, weights_shape,
      hints_.Check(ModelHints::kPreferTextureWeights));

  const int elements_count =
      GetTotalElementsCountForLayout(weights_desc, weights_shape) / 4;
  std::vector<uint8_t> weights_data(elements_count);

  if (std::holds_alternative<Tensor<OHWI, DataType::INT2>>(weights)) {
    const auto& int2_weights = std::get<Tensor<OHWI, DataType::INT2>>(weights);
    if (weights_desc.layout == WeightsLayout::kOSpatialIOGroupI4O4 ||
        weights_desc.layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
      Tensor<OHWI, DataType::UINT8> uint8_weights;
      uint8_weights.shape = int2_weights.shape;
      uint8_weights.data.assign(int2_weights.data.begin(),
                                int2_weights.data.end());
      auto status = RearrangeWeightsUInt2Packed(uint8_weights, weights_desc,
                                                absl::MakeSpan(weights_data),
                                                {}, 2, false);
      ABSL_CHECK(status.ok())
          << "Failed to rearrange INT2 weights: " << status.message();
    } else {
      ABSL_CHECK(false) << "Unsupported layout for packed INT2 weights";
    }
  } else {
    const auto& int8_weights = std::get<Tensor<OHWI, DataType::INT8>>(weights);
    RearrangeWeightsInt8AsUint2(int8_weights, weights_desc,
                                absl::MakeSpan(weights_data), 2, 2u);
  }

  TensorDescriptor weights_td;
  if (weights_desc.IsLinearLayout()) {
    weights_td = TensorDescriptor(DataType::UINT8, TensorStorageType::BUFFER,
                                  Layout::LINEAR);
    weights_td.SetBHWCShape(BHWC(1, 1, 1, weights_data.size()));
  } else {
    DataType texture_type = DataType::UINT8;
    weights_td = TensorDescriptor(texture_type, TensorStorageType::TEXTURE_2D,
                                  Layout::HW);
    uint2 tex_size = Get2dResourceSize(weights_desc, weights_shape);
    tex_size.x /= 4;  // because we store 16 elements per pixel
    weights_td.SetBHWDCShape(BHWDC(1, tex_size.y, tex_size.x, 1, 4));
  }
  weights_td.UploadDataRaw(absl::MakeConstSpan(weights_data));

  Weights external_weights;
  external_weights.weights = AddConstantTensor(std::move(weights_td));
  external_weights.desc = weights_desc;
  external_weights.shape = weights_shape;
  return external_weights;
}

GpuModelBuilder::Weights GpuModelBuilder::GetWeights(
    const std::variant<Tensor<OHWI, DataType::INT8>,
                       Tensor<OHWI, DataType::INT4>>& weights) {
  const OHWI weights_shape =
      std::visit([](const auto& w) { return w.shape; }, weights);

  WeightsDescription weights_desc = ml_drift::GetFullyConnectedInt4WeightsDesc(
      gpu_info_, weights_shape,
      hints_.Check(ModelHints::kPreferTextureWeights));

  const int elements_count =
      GetTotalElementsCountForLayout(weights_desc, weights_shape) / 4;
  std::vector<uint8_t> weights_data(elements_count);

  if (std::holds_alternative<Tensor<OHWI, DataType::INT4>>(weights)) {
    const auto& int4_weights = std::get<Tensor<OHWI, DataType::INT4>>(weights);
    if (weights_desc.layout == WeightsLayout::kOSpatialIOGroupI4O4 ||
        weights_desc.layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
      Tensor<OHWI, DataType::UINT8> uint8_weights;
      uint8_weights.shape = int4_weights.shape;
      uint8_weights.data.assign(int4_weights.data.begin(),
                                int4_weights.data.end());
      auto status = RearrangeWeightsUInt4Packed(uint8_weights, weights_desc,
                                                absl::MakeSpan(weights_data),
                                                {}, 8, false);
      ABSL_CHECK(status.ok())
          << "Failed to rearrange INT4 weights: " << status.message();
    } else {
      ABSL_CHECK(false) << "Unsupported layout for packed INT4 weights";
    }
  } else {
    const auto& int8_weights = std::get<Tensor<OHWI, DataType::INT8>>(weights);
    RearrangeWeightsInt8AsUint4(int8_weights, weights_desc,
                                absl::MakeSpan(weights_data), 8, 8u);
  }

  TensorDescriptor weights_td;
  if (weights_desc.IsLinearLayout()) {
    weights_td = TensorDescriptor(DataType::UINT8, TensorStorageType::BUFFER,
                                  Layout::LINEAR);
    weights_td.SetBHWCShape(BHWC(1, 1, 1, weights_data.size()));
  } else {
    DataType texture_type = DataType::UINT16;
    weights_td = TensorDescriptor(texture_type, TensorStorageType::TEXTURE_2D,
                                  Layout::HW);
    uint2 tex_size = Get2dResourceSize(weights_desc, weights_shape);
    tex_size.x /= 4;  // because we store 16 elements per pixel
    weights_td.SetBHWDCShape(BHWDC(1, tex_size.y, tex_size.x, 1, 4));
  }
  weights_td.UploadDataRaw(absl::MakeConstSpan(weights_data));

  Weights external_weights;
  external_weights.weights = AddConstantTensor(std::move(weights_td));
  external_weights.desc = weights_desc;
  external_weights.shape = weights_shape;
  return external_weights;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::GetWeightsScale(
    const Tensor<OHWI, DataType::FLOAT32>& scale, DataType float_type) {
  auto weights_scale_td =
      ScaleOrZeroPointToTensorDesc(gpu_info_, scale, float_type);
  return AddConstantTensor(std::move(weights_scale_td));
}

GpuModelBuilder::TensorHandle GpuModelBuilder::GetWeightsZeroPoint(
    const Tensor<OHWI, DataType::INT32>& zero_point, DataType float_type) {
  Tensor<OHWI, DataType::FLOAT32> float_zp;
  float_zp.shape = zero_point.shape;
  float_zp.data.resize(zero_point.size());
  for (size_t i = 0; i < zero_point.size(); ++i) {
    float_zp.data[i] = static_cast<float>(zero_point.Data()[i]);
  }
  auto weights_zero_point_td =
      ScaleOrZeroPointToTensorDesc(gpu_info_, float_zp, float_type);
  return AddConstantTensor(std::move(weights_zero_point_td));
}

GpuModelBuilder::TensorHandle GpuModelBuilder::FullyConnected(
    const GpuModelBuilder::TensorHandle& src,
    const FullyConnectedInt4Attributes& attr) {
  const DataType float_type = src.tensor_desc.GetDataType();
  Weights weights = GetWeights(attr.weights);

  if (!attr.scale.empty()) {
    weights.scale_zp_shape = attr.scale.shape;
    weights.scale = GetWeightsScale(attr.scale, float_type);
  }

  if (!attr.zero_point.empty()) {
    weights.zero_point = GetWeightsZeroPoint(attr.zero_point, float_type);
  }

  GpuModelBuilder::TensorHandle bias_handle;
  GpuModelBuilder::TensorHandle* bias_handle_ptr = nullptr;
  if (!attr.bias.empty()) {
    bias_handle = AddConstantTensor(attr.bias, float_type);
    bias_handle_ptr = &bias_handle;
  }
  return FullyConnectedInt4ExternalWeights(src, weights, bias_handle_ptr);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::FullyConnected(
    const GpuModelBuilder::TensorHandle& src,
    const FullyConnectedInt2Attributes& attr) {
  const DataType float_type = src.tensor_desc.GetDataType();
  Weights weights = GetWeights(attr.weights);

  if (!attr.scale.empty()) {
    weights.scale_zp_shape = attr.scale.shape;
    weights.scale = GetWeightsScale(attr.scale, float_type);
  }

  if (!attr.zero_point.empty()) {
    weights.zero_point = GetWeightsZeroPoint(attr.zero_point, float_type);
  }

  GpuModelBuilder::TensorHandle bias_handle;
  GpuModelBuilder::TensorHandle* bias_handle_ptr = nullptr;
  if (!attr.bias.empty()) {
    bias_handle = AddConstantTensor(attr.bias, float_type);
    bias_handle_ptr = &bias_handle;
  }
  return FullyConnectedInt2ExternalWeights(src, weights, bias_handle_ptr);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::DepthwiseConvolution(
    const GpuModelBuilder::TensorHandle& src,
    const DepthwiseConvolution2DAttributes& attr) {
  BHWC dst_shape = CalculateOutputShape(src.tensor_desc.GetBHWCShape(), attr);
  GpuModelBuilder::TensorHandle dst =
      AddTensor(dst_shape, src.tensor_desc.GetDataType());
  const auto& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name =
      absl::StrCat("depthwise_conv", weights_shape.h, "x", weights_shape.w);
  gpu_node.op_name = attr.op_name;
  gpu_node.inputs = {src.id};
  gpu_node.outputs = {dst.id};
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);
  gpu_node.gpu_operation = SelectDWConvolution(
      attr, gpu_info_, op_def, GetConvPrecision(src.tensor_desc.GetDataType()));
  gpu_node.gpu_operation->flops_ =
      GetDepthwiseConvolutionFlops(dst_shape, weights_shape);
  return dst;
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
GpuModelBuilder::DepthwiseConvolution(
    const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& weights,
    const DepthwiseConvolution2DAttributes& attr) {
  const BHWC& weights_shape = weights.tensor_desc.GetBHWCShape();
  if (weights_shape.b != 1) {
    return absl::UnimplementedError(
        "No support of depthwise runtime weights with channel multiplier "
        "!= 1");
  }
  const auto src_type = src.tensor_desc.GetDataType();
  const auto weights_type = weights.tensor_desc.GetDataType();
  if (src_type != weights_type) {
    return absl::UnimplementedError(absl::StrCat(
        "No support of depthwise runtime weights with different data types. "
        "src: ",
        ToString(src_type), " weights: ", ToString(weights_type)));
  }
  BHWC dst_shape = CalculateOutputShape(src.tensor_desc.GetBHWCShape(), attr);
  auto dst = AddTensor(dst_shape, src_type);

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  const auto& dw_weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  gpu_node.name = absl::StrCat("depthwise_conv", dw_weights_shape.h, "x",
                               dw_weights_shape.w, "_runtime_weights");
  gpu_node.op_name = attr.op_name;
  gpu_node.inputs = {src.id, weights.id};
  gpu_node.outputs = {dst.id};
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.src_tensors.push_back(weights.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);
  gpu_node.gpu_operation = SelectDWConvolutionExternalWeights(
      attr, gpu_info_, op_def, GetConvPrecision(src_type));
  gpu_node.gpu_operation->flops_ =
      GetDepthwiseConvolutionFlops(dst_shape, dw_weights_shape);
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::FullyConnected(
    const GpuModelBuilder::TensorHandle& src,
    const FullyConnectedAttributes& attr) {
  const DataType float_type = src.tensor_desc.GetDataType();
  const CalculationsPrecision precision = GetConvPrecision(float_type);
  BHWC dst_shape = CalculateOutputShape(src.tensor_desc.GetBHWCShape(), attr);
  const int total_spatial_size = dst_shape.b * dst_shape.h * dst_shape.w;
  GpuModelBuilder::TensorHandle dst = AddTensor(dst_shape, float_type);

  GpuModelBuilder::TensorHandle bias_th;
  const TensorDescriptor* bias_td_ptr = nullptr;
  if (!attr.bias.empty()) {
    bias_th = AddConstantTensor(attr.bias, float_type);
    bias_td_ptr = &bias_th.tensor_desc;
  }

  std::unique_ptr<GPUOperation> op;
  WeightsDescription weights_desc;
  std::string op_name;
  if (total_spatial_size >
      GetRecommendedMaxTotalSpatialSize(gpu_info_, precision)) {
    OperationDef op_def;
    op_def.src_tensors.push_back(src.tensor_desc);
    op_def.dst_tensors.push_back(dst.tensor_desc);
    Convolution2DAttributes conv_attr;
    conv_attr.op_name = attr.op_name;
    auto weights_tensor = Tensor<OHWI, DataType::FLOAT32>();
    weights_tensor.shape = attr.weights.shape;
    conv_attr.weights = weights_tensor;
    conv_attr.bias = std::move(attr.bias);
    conv_attr.padding.appended = HW(0, 0);
    conv_attr.padding.prepended = HW(0, 0);
    conv_attr.strides = HW(1, 1);
    conv_attr.dilations = HW(1, 1);
    op = SelectConvolutionWithExternalWeights(
        conv_attr, bias_td_ptr, dst_shape, gpu_info_, op_def, precision, hints_,
        &weights_desc,
        /*src_exp=*/nullptr,
        /*different_weights_for_height=*/false);
    op_name = "convolution1x1" + GetConvOpNameSuffix(*op);
  } else {
    weights_desc = ml_drift::GetFullyConnectedWeightsDesc(gpu_info_, float_type,
                                                          attr.weights.shape);

    ExternalWeights external_weights;
    external_weights.desc = weights_desc;
    external_weights.shape = attr.weights.shape;
    auto fc_with_status = CreateFullyConnectedExternalWeights(
        gpu_info_, precision, src.tensor_desc, dst.tensor_desc,
        external_weights, bias_td_ptr, &dst_shape);
    if (!fc_with_status.ok()) {
      ABSL_LOG(ERROR) << fc_with_status.status().message();
      return dst;
    } else {
      op = std::make_unique<ml_drift::FullyConnected>(
          std::move(fc_with_status.value()));
    }
    op_name = "fully_connected";
  }
  op->flops_ = GetConvolutionFlops(dst_shape, attr.weights.shape);
  std::vector<ValueId> input_ids = {src.id};
  const auto weights_handles = GetWeights(attr.weights, weights_desc);
  for (const auto& weight : weights_handles) {
    input_ids.push_back(weight.id);
  }
  if (!attr.bias.empty()) {
    input_ids.push_back(bias_th.id);
  }
  AddGpuOperation(input_ids, {dst.id}, std::move(op), op_name);
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::FullyConnected(
    const GpuModelBuilder::TensorHandle& src,
    const FullyConnectedInt8Attributes& attr) {
  const BHWC src_shape = src.tensor_desc.GetBHWCShape();
  const BHWC dst_shape(src_shape.b, src_shape.h, src_shape.w,
                       attr.weights.shape.o);
  const int total_spatial_size = dst_shape.b * dst_shape.h * dst_shape.w;
  const DataType float_type = src.tensor_desc.GetDataType();
  if (!attr.zero_point.empty()) {
    ABSL_QCHECK_EQ(attr.scale.shape, attr.zero_point.shape);
  }
  if (weights_manager_ && weights_manager_->ShouldOffloadPreparationToGPU(
                              gpu_info_, attr.weights.shape,
                              WeightsManager::TargetWeightsType::kStandard)) {
    auto weights_desc = ml_drift::GetFullyConnectedInt8WeightsDesc(
        gpu_info_, attr.weights.shape,
        hints_.Check(ModelHints::kPreferTextureWeights));

    auto input_weights = AddTensors(
        GetTensorDescriptorsForWeightsLayout(attr.weights.shape, weights_desc));
    ABSL_QCHECK_EQ(input_weights.size(), 1);
    weights_manager_->RegisterWeightsConversion(
        {input_weights[0].id}, weights_desc, attr.weights.shape, DataType::INT8,
        attr.weights.Data());

    auto scale_th = GetWeightsScale(attr.scale, float_type);

    GpuModelBuilder::TensorHandle zp_th;
    GpuModelBuilder::TensorHandle* zp_th_ptr = nullptr;
    if (!attr.zero_point.empty()) {
      zp_th = GetWeightsZeroPoint(attr.zero_point, float_type);
      zp_th_ptr = &zp_th;
    }

    GpuModelBuilder::TensorHandle bias_th;
    GpuModelBuilder::TensorHandle* bias_th_ptr = nullptr;
    if (!attr.bias.data.empty()) {
      bias_th = AddConstantTensor(attr.bias, float_type);
      bias_th_ptr = &bias_th;
    }
    const Weights external_weights = CreateExternalWeights(
        input_weights[0], weights_desc, attr.weights.shape, attr.scale.shape,
        &scale_th, zp_th_ptr);
    return FullyConnectedInt8ExternalWeights(src, external_weights,
                                             bias_th_ptr);
  }

  if (total_spatial_size <=
      GetRecommendedMaxTotalSpatialSize(
          gpu_info_, GetConvPrecision(src.tensor_desc.GetDataType()))) {
    GpuModelBuilder::TensorHandle dst = AddTensor(dst_shape, float_type);
    gpu_model_.nodes.push_back({});
    auto& gpu_node = gpu_model_.nodes.back();
    gpu_node.name = "fully_connected_int8";
    gpu_node.op_name = absl::StrCat(attr.op_name, "_fully_connected_int8");
    gpu_node.inputs = {src.id};
    gpu_node.outputs = {dst.id};

    WeightsDescription weights_desc =
        ml_drift::GetFullyConnectedInt8WeightsDesc(gpu_info_,
                                                   attr.weights.shape);
    TensorDescriptor weights_i8_td =
        GetTensorDescriptorForWeightsLayout(attr.weights, weights_desc);
    auto weights_th = AddConstantTensor(std::move(weights_i8_td));
    gpu_node.inputs.push_back(weights_th.id);

    ExternalWeights external_weights;
    external_weights.desc = weights_desc;
    external_weights.shape = attr.weights.shape;
    if (!attr.scale.empty()) {
      external_weights.scale_zp_shape = attr.scale.shape;
    }
    GpuModelBuilder::TensorHandle scale_th;
    if (attr.scale.shape.DimensionsProduct() == 1) {
      external_weights.scalar_scale = attr.scale.Data()[0];
    } else if (!attr.scale.empty()) {
      scale_th = GetWeightsScale(attr.scale, float_type);
      external_weights.scale = &scale_th.tensor_desc;
      gpu_node.inputs.push_back(scale_th.id);
    }

    GpuModelBuilder::TensorHandle zero_point_th;
    if (attr.zero_point.shape.DimensionsProduct() == 1) {
      external_weights.scalar_zero_point = attr.zero_point.Data()[0];
    } else if (!attr.zero_point.empty()) {
      zero_point_th = GetWeightsZeroPoint(attr.zero_point, float_type);
      external_weights.zero_point = &zero_point_th.tensor_desc;
      gpu_node.inputs.push_back(zero_point_th.id);
    }

    GpuModelBuilder::TensorHandle bias_th;
    GpuModelBuilder::TensorHandle* bias_th_ptr = nullptr;
    TensorDescriptor* bias_td_ptr = nullptr;
    if (!attr.bias.data.empty()) {
      bias_th = AddConstantTensor(attr.bias, float_type);
      bias_th_ptr = &bias_th;
      bias_td_ptr = &bias_th.tensor_desc;
      gpu_node.inputs.push_back(bias_th.id);
    }

    auto fc = CreateFullyConnectedExternalWeights(
        gpu_info_, GetConvPrecision(src.tensor_desc.GetDataType()),
        src.tensor_desc, dst.tensor_desc, external_weights, bias_td_ptr,
        &dst_shape);
    if (!fc.ok()) {
      ABSL_LOG(ERROR) << fc.status().message();
    } else {
      gpu_node.gpu_operation =
          std::make_unique<ml_drift::FullyConnected>(std::move(fc.value()));
    }
    gpu_node.gpu_operation->flops_ =
        GetConvolutionFlops(dst_shape, attr.weights.shape);
    return dst;
  }

  const auto& scale_shape = attr.scale.shape;
  const bool blockwise =
      scale_shape.DimensionsProduct() != 1 &&            // tensor-wise
      scale_shape.DimensionsProduct() != scale_shape.o;  // channel-wise

  // Avoid using fp16 -> int8 DRQ -> int8 conv if this path is roughly estimated
  // to be slower than native fp16 conv, or with blockwise quantized weights.
  const bool do_not_use_int8_on_mali =
      gpu_info_.IsMali() && !(src_shape.c >= 128 && dst_shape.c >= 128);
  const bool do_not_use_int8_on_adreno =
      gpu_info_.IsAdreno() && !(src_shape.c >= 256 && dst_shape.c >= 256);
  const bool do_not_use_int8_on_powervr =
      gpu_info_.IsPowerVR() && !(src_shape.c >= 512 && dst_shape.c >= 512);
  if (SupportsConvolutionInt8(gpu_info_, src_shape) &&
      !do_not_use_int8_on_mali && !do_not_use_int8_on_adreno &&
      !do_not_use_int8_on_powervr &&
      !hints_.Check(ModelHints::kDisallow8bitConvs) && !blockwise) {
    GpuModelBuilder::TensorHandle scale_handle;
    {
      Tensor<Linear, DataType::FLOAT32> scale_float;
      scale_float.shape = Linear(attr.weights.shape.o);
      scale_float.data.resize(attr.weights.shape.o);
      for (int i = 0; i < attr.weights.shape.o; ++i) {
        scale_float.data[i] =
            attr.scale.Data()[std::min(i, attr.scale.shape.o - 1)];
      }
      scale_handle = AddConstantTensor(scale_float, float_type);
    }

    GpuModelBuilder::TensorHandle zp_handle;
    GpuModelBuilder::TensorHandle* zp_handle_ptr = nullptr;
    {
      Tensor<Linear, DataType::FLOAT32> zp_float;
      zp_float.shape = Linear(attr.weights.shape.o);
      zp_float.data.resize(attr.weights.shape.o);
      bool all_zeroes = true;
      for (int i = 0; i < attr.weights.shape.o; ++i) {
        zp_float.data[i] =
            attr.zero_point.Data()[std::min(i, attr.zero_point.shape.o - 1)];
        all_zeroes &= zp_float.data[i] == 0.0f;
      }
      if (!all_zeroes) {
        zp_handle = AddConstantTensor(zp_float, float_type);
        zp_handle_ptr = &zp_handle;
      }
    }

    GpuModelBuilder::TensorHandle weights_sum_i_handle;
    {
      auto weights_sum_i = GetWeightsAccumulatedInputChannels(attr.weights);
      auto weights_sum_i_td =
          CreateConstantLinearTensorDescriptor(gpu_info_, weights_sum_i);
      weights_sum_i_handle = AddConstantTensor(std::move(weights_sum_i_td));
    }

    GpuModelBuilder::TensorHandle bias_handle;
    GpuModelBuilder::TensorHandle* bias_handle_ptr = nullptr;
    if (!attr.bias.data.empty()) {
      bias_handle = AddConstantTensor(attr.bias, float_type);
      bias_handle_ptr = &bias_handle;
    }

    Weights external_weights;
    external_weights.shape = attr.weights.shape;
    external_weights.scale_zp_shape = scale_shape;
    external_weights.scale = scale_handle;
    if (zp_handle_ptr) {
      external_weights.zero_point = *zp_handle_ptr;
    }
    external_weights.sum_i = weights_sum_i_handle;
    external_weights.desc.layout = WeightsLayout::kUnknown;
    // Create id, load data later.
    external_weights.weights = AddConstantTensor({});
    WeightsDescription conv_weights_desc;
    TensorHandle result = FullyConnectedInt8QuantizedWithSrcQuantization(
        src, external_weights, bias_handle_ptr, &conv_weights_desc);
    gpu_model_.const_tensors[external_weights.weights.id] =
        GetTensorDescriptorForWeightsLayout(attr.weights, conv_weights_desc);
    return result;
  }

  // Dequantize weights to fp32 and use fp conv 1x1.
  FullyConnectedAttributes attr_f32 = ToFloat32(attr);
  Convolution2DAttributes conv_attr;
  conv_attr.op_name = attr.op_name;
  conv_attr.weights = std::move(attr_f32.weights);
  conv_attr.bias = std::move(attr_f32.bias);
  conv_attr.padding.appended = HW(0, 0);
  conv_attr.padding.prepended = HW(0, 0);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  return Convolution(src, conv_attr);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::PositionalEmbedding(
    const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& position) {
  BHWC dst_shape = src.tensor_desc.GetBHWCShape();
  GpuModelBuilder::TensorHandle dst =
      AddTensor(dst_shape, src.tensor_desc.GetDataType());

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = "add_position_emb";
  gpu_node.inputs = {src.id};
  gpu_node.inputs.push_back(position.id);
  gpu_node.outputs = {dst.id};
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.src_tensors.push_back(position.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);
  gpu_node.gpu_operation = std::make_unique<GPUOperation>(
      CreatePositionalEmbedding(gpu_info_, op_def));
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::QuantizeAndDequantize(
    const GpuModelBuilder::TensorHandle& src,
    const QuantizeAndDequantizeAttributes& attr) {
  auto dst =
      AddTensor(src.tensor_desc.GetBHWCShape(), src.tensor_desc.GetDataType());
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);
  auto gpu_op = SelectQuantizeAndDequantize(attr, op_def);
  AddGpuOperation(std::vector<ValueId>{src.id}, std::vector<ValueId>{dst.id},
                  std::move(gpu_op), "quantize_and_dequantize");
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::StaticRangeQuantization(
    const GpuModelBuilder::TensorHandle& src,
    const QuantizeAndDequantizeAttributes& attr) {
  auto dst =
      AddTensor(src.tensor_desc.GetBHWCShape(), src.tensor_desc.GetDataType());
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);
  auto gpu_op = SelectStaticRangeQuantization(attr, op_def);
  AddGpuOperation(std::vector<ValueId>{src.id}, std::vector<ValueId>{dst.id},
                  std::move(gpu_op), "range_quantization");
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::ReLU(
    const GpuModelBuilder::TensorHandle& src, const ReLUAttributes& attr) {
  BHWC dst_shape = src.tensor_desc.GetBHWCShape();
  GpuModelBuilder::TensorHandle dst =
      AddTensor(dst_shape, src.tensor_desc.GetDataType());

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = "relu";
  gpu_node.inputs = {src.id};
  gpu_node.outputs = {dst.id};
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);
  gpu_node.gpu_operation = SelectReLU(attr, op_def);
  return dst;
}

std::vector<GpuModelBuilder::TensorHandle> GpuModelBuilder::RoPE(
    const GpuModelBuilder::TensorHandle& src_l,
    const GpuModelBuilder::TensorHandle& src_r,
    const GpuModelBuilder::TensorHandle& position, const RoPEAttributes& attr) {
  BHWC new_shape = src_l.tensor_desc.GetBHWCShape();
  GpuModelBuilder::TensorHandle dst_l =
      AddTensor(new_shape, src_l.tensor_desc.GetDataType());
  GpuModelBuilder::TensorHandle dst_r =
      AddTensor(new_shape, src_r.tensor_desc.GetDataType());

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = "RoPE";
  gpu_node.inputs = {src_l.id, src_r.id, position.id};
  gpu_node.outputs = {dst_l.id, dst_r.id};
  OperationDef op_def;
  op_def.src_tensors.push_back(src_l.tensor_desc);
  op_def.src_tensors.push_back(src_r.tensor_desc);
  op_def.src_tensors.push_back(position.tensor_desc);
  op_def.dst_tensors.push_back(dst_l.tensor_desc);
  op_def.dst_tensors.push_back(dst_r.tensor_desc);
  gpu_node.gpu_operation =
      std::make_unique<GPUOperation>(CreateRoPE(gpu_info_, op_def, attr));
  return {dst_l, dst_r};
}

GpuModelBuilder::TensorHandle GpuModelBuilder::SplitRoPEConcatInternal(
    const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& position, const RoPEAttributes& attr) {
  BHWC new_shape = src.tensor_desc.GetBHWCShape();
  GpuModelBuilder::TensorHandle dst =
      AddTensor(new_shape, src.tensor_desc.GetDataType());

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = "SplitRoPEConcat";
  gpu_node.inputs = {src.id, position.id};
  gpu_node.outputs = {dst.id};
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.src_tensors.push_back(position.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);
  gpu_node.gpu_operation = std::make_unique<GPUOperation>(
      CreateSplitRoPEConcat(gpu_info_, op_def, attr));
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::SplitRoPEConcat(
    const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& position, const RoPEAttributes& attr) {
  GpuModelBuilder::TensorHandle dst;
  const int channels = src.tensor_desc.GetBHWCShape().c;
  if (channels % 8 == 0) {
    dst = SplitRoPEConcatInternal(src, position, attr);
  } else {
    auto qs = Split(src, Axis::CHANNELS, channels / 2);
    qs = RoPE(qs[0], qs[1], position, attr);
    dst = Concat(qs[0], qs[1], Axis::CHANNELS);
  }
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Softmax(
    const GpuModelBuilder::TensorHandle& src,
    const SoftmaxRuntimeCheckDesc& runtime_check,
    const GpuModelBuilder::TensorHandle* runtime_check_tensor) {
  BHWC dst_shape = src.tensor_desc.GetBHWCShape();
  const int total_spatial_size = dst_shape.b * dst_shape.h * dst_shape.w;
  const float spatial_per_cu =
      static_cast<float>(total_spatial_size) / gpu_info_.GetComputeUnitsCount();
  if (spatial_per_cu >= 256.0f) {
    auto reduced_exp = SoftmaxReduce(src, runtime_check, runtime_check_tensor);
    return SoftmaxElementwise(src, reduced_exp);
  }

  GpuModelBuilder::TensorHandle dst;
  if (src.tensor_desc.HasAxis(Axis::DEPTH)) {
    dst = AddTensor(src.tensor_desc.GetBHWDCShape(),
                    src.tensor_desc.GetDataType());
  } else {
    dst = AddTensor(dst_shape, src.tensor_desc.GetDataType());
  }

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = "softmax";
  gpu_node.inputs = {src.id};
  gpu_node.outputs = {dst.id};
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);

  const int spatial_size = dst_shape.b * dst_shape.w * dst_shape.h;
  const int spatial_size_per_cu =
      DivideRoundUp(spatial_size, gpu_info_.GetComputeUnitsCount());
  if (spatial_size_per_cu >= 4 ||
      (spatial_size >= 8 && dst_shape.c >= 1024 * 8)) {
    ml_drift::Softmax operation =
        CreateSoftmax(op_def, gpu_info_, dst_shape, runtime_check);
    gpu_node.gpu_operation =
        std::make_unique<ml_drift::Softmax>(std::move(operation));
  } else {
    Softmax1x1 operation =
        CreateSoftmax1x1(op_def, gpu_info_, dst_shape, runtime_check);
    gpu_node.gpu_operation = std::make_unique<Softmax1x1>(std::move(operation));
  }

  if (runtime_check_tensor && runtime_check.end_ch_index.has_value()) {
    gpu_node.inputs.push_back(runtime_check_tensor->id);
  }
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::SoftmaxReduce(
    const GpuModelBuilder::TensorHandle& src,
    const SoftmaxRuntimeCheckDesc& runtime_check,
    const GpuModelBuilder::TensorHandle* runtime_check_tensor) {
  GpuModelBuilder::TensorHandle reduced_exp_tensor;
  const BHWC src_shape = src.tensor_desc.GetBHWCShape();
  if (src.tensor_desc.HasAxis(Axis::DEPTH)) {
    BHWDC reduced_exp_shape = src.tensor_desc.GetBHWDCShape();
    reduced_exp_shape.c = 4;
    reduced_exp_tensor =
        AddTensor(reduced_exp_shape, src.tensor_desc.GetDataType());
  } else {
    BHWC reduced_exp_shape = src.tensor_desc.GetBHWCShape();
    reduced_exp_shape.c = 4;
    reduced_exp_tensor =
        AddTensor(reduced_exp_shape, src.tensor_desc.GetDataType());
  }

  gpu_model_.nodes.reserve(gpu_model_.nodes.size() + 3);
  gpu_model_.nodes.push_back({});
  auto& softmax_node = gpu_model_.nodes.back();
  softmax_node.name = "softmax (reduce)";
  softmax_node.inputs = {src.id};
  softmax_node.outputs = {reduced_exp_tensor.id};
  OperationDef softmax_def;
  softmax_def.src_tensors.push_back(src.tensor_desc);
  softmax_def.dst_tensors.push_back(reduced_exp_tensor.tensor_desc);

  const int spatial_size = src_shape.b * src_shape.w * src_shape.h;
  const int spatial_size_per_cu =
      DivideRoundUp(spatial_size, gpu_info_.GetComputeUnitsCount());
  if (spatial_size_per_cu >= 4) {
    ml_drift::Softmax operation =
        CreateSoftmaxReduce(softmax_def, gpu_info_, src_shape, runtime_check);
    softmax_node.gpu_operation =
        std::make_unique<ml_drift::Softmax>(std::move(operation));
  } else {
    Softmax1x1 operation = CreateSoftmax1x1Reduce(softmax_def, gpu_info_,
                                                  src_shape, runtime_check);
    softmax_node.gpu_operation =
        std::make_unique<Softmax1x1>(std::move(operation));
  }
  if (runtime_check_tensor && runtime_check.end_ch_index.has_value()) {
    softmax_node.inputs.push_back(runtime_check_tensor->id);
  }
  return reduced_exp_tensor;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::SoftmaxElementwise(
    const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& reduced_exp) {
  BHWC dst_shape = src.tensor_desc.GetBHWCShape();
  GpuModelBuilder::TensorHandle dst =
      AddTensor(dst_shape, src.tensor_desc.GetDataType());

  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.src_tensors.push_back(reduced_exp.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);
  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = "softmax (elementwise)";
  gpu_node.inputs = {src.id, reduced_exp.id};
  gpu_node.outputs = {dst.id};
  gpu_node.gpu_operation =
      std::make_unique<GPUOperation>(CreateSoftmaxFinal(op_def, dst_shape.c));
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Reshape(
    const GpuModelBuilder::TensorHandle& src, const BHWC& new_shape) {
  return Reshape(src,
                 BHWDC(new_shape.b, new_shape.h, new_shape.w, 1, new_shape.c));
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Reshape(
    const GpuModelBuilder::TensorHandle& src, const BHWDC& new_shape) {
  GpuModelBuilder::TensorHandle dst =
      AddTensor(new_shape.b, new_shape.h, new_shape.w, new_shape.d, new_shape.c,
                default_storage_, src.tensor_desc.GetDataType());

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = "reshape";
  gpu_node.inputs = {src.id};
  gpu_node.outputs = {dst.id};
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);
  Reshape3DAttributes attr;
  attr.new_shape = dst.tensor_desc.GetBHWDCShape();
  const int src_channels = src.tensor_desc.GetBHWDCShape().c;
  const int dst_channels = dst.tensor_desc.GetBHWDCShape().c;
  if (src_channels % 4 == 0 && dst_channels % 4 == 0 &&
      !op_def.src_tensors[0].HasAxis(Axis::DEPTH) &&
      !op_def.dst_tensors[0].HasAxis(Axis::DEPTH)) {
    Reshapex4 operation = CreateReshapex4(op_def);
    gpu_node.gpu_operation = std::make_unique<Reshapex4>(std::move(operation));
    gpu_node.gpu_operation->ResolveReorderFinalShape(attr.new_shape);
  } else {
    GPUOperation operation = CreateReshape(op_def);
    gpu_node.gpu_operation =
        std::make_unique<GPUOperation>(std::move(operation));
  }
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Transpose(
    const GpuModelBuilder::TensorHandle& src, const BHWC& perm) {
  TransposeAttributes attr;
  attr.perm = perm;
  const BHWC new_shape =
      CalculateOutputShape(src.tensor_desc.GetBHWCShape(), attr);
  GpuModelBuilder::TensorHandle dst =
      AddTensor(new_shape, src.tensor_desc.GetDataType());
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);
  GPUOperation operation = CreateTranspose(op_def, attr);
  AddGpuOperation(std::vector<ValueId>{src.id}, std::vector<ValueId>{dst.id},
                  std::make_unique<GPUOperation>(std::move(operation)),
                  "transpose");
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Transpose(
    const GpuModelBuilder::TensorHandle& src, const BHWDC& perm) {
  Transpose3DAttributes attr;
  attr.perm = perm;
  const BHWDC src_shape = src.tensor_desc.GetBHWDCShape();
  const BHWDC new_shape =
      BHWDC(src_shape.get(perm.b), src_shape.get(perm.h), src_shape.get(perm.w),
            src_shape.get(perm.d), src_shape.get(perm.c));
  GpuModelBuilder::TensorHandle dst =
      AddTensor(new_shape.b, new_shape.h, new_shape.w, new_shape.d, new_shape.c,
                default_storage_, src.tensor_desc.GetDataType());
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);
  GPUOperation operation = CreateTranspose(op_def, attr);
  AddGpuOperation(std::vector<ValueId>{src.id}, std::vector<ValueId>{dst.id},
                  std::make_unique<GPUOperation>(std::move(operation)),
                  "transpose");
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Elementwise(
    const GpuModelBuilder::TensorHandle& src, OperationType op_type) {
  const BHWC new_shape = src.tensor_desc.GetBHWCShape();
  GpuModelBuilder::TensorHandle dst =
      AddTensor(new_shape, src.tensor_desc.GetDataType());

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = ToString(op_type);
  gpu_node.inputs = {src.id};
  gpu_node.outputs = {dst.id};
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);

  GPUOperation operation =
      CreateElementwiseOneInput(gpu_info_, op_def, op_type);
  gpu_node.gpu_operation = std::make_unique<GPUOperation>(std::move(operation));
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Elementwise(
    const GpuModelBuilder::TensorHandle& src, OperationType op_type,
    float value) {
  ElementwiseAttributes attr;
  attr.param = value;
  return Elementwise(src, attr, op_type);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Elementwise(
    const GpuModelBuilder::TensorHandle& src, OperationType op_type,
    double value) {
  ElementwiseAttributes attr;
  attr.param = static_cast<float>(value);
  return Elementwise(src, attr, op_type);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Elementwise(
    const GpuModelBuilder::TensorHandle& src, OperationType op_type,
    int value) {
  ElementwiseAttributes attr;
  attr.param = value;
  return Elementwise(src, attr, op_type);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Elementwise(
    const GpuModelBuilder::TensorHandle& src, const ElementwiseAttributes& attr,
    OperationType op_type) {
  const BHWC new_shape = src.tensor_desc.GetBHWCShape();
  const DataType data_type =
      IsLogicalOp(op_type) ? DataType::BOOL : src.tensor_desc.GetDataType();
  GpuModelBuilder::TensorHandle dst = AddTensor(new_shape, data_type);

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = ToString(op_type);
  gpu_node.inputs = {src.id};
  gpu_node.outputs = {dst.id};
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);

  GPUOperation operation = CreateElementwise(gpu_info_, op_def, op_type, attr);
  gpu_node.gpu_operation = std::make_unique<GPUOperation>(std::move(operation));
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Elementwise(
    const GpuModelBuilder::TensorHandle& left,
    const GpuModelBuilder::TensorHandle& right, OperationType op_type) {
  const BHWC new_shape = left.tensor_desc.GetBHWCShape();
  const DataType data_type =
      IsLogicalOp(op_type) ? DataType::BOOL : left.tensor_desc.GetDataType();
  GpuModelBuilder::TensorHandle dst = AddTensor(new_shape, data_type);

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = ToString(op_type);
  gpu_node.inputs = {left.id, right.id};
  gpu_node.outputs = {dst.id};
  OperationDef op_def;
  op_def.src_tensors.push_back(left.tensor_desc);
  op_def.src_tensors.push_back(right.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);

  GPUOperation operation = CreateElementwiseTwoInput(
      gpu_info_, op_def, op_type, right.tensor_desc.GetBHWCShape(), new_shape);
  gpu_node.gpu_operation = std::make_unique<GPUOperation>(std::move(operation));
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Add(
    const GpuModelBuilder::TensorHandle& src, float value) {
  ElementwiseAttributes attr;
  attr.param = value;
  return Elementwise(src, attr, OperationType::ADD);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Add(
    const GpuModelBuilder::TensorHandle& src, double value) {
  ElementwiseAttributes attr;
  attr.param = static_cast<float>(value);
  return Elementwise(src, attr, OperationType::ADD);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Add(
    const GpuModelBuilder::TensorHandle& src, int value) {
  ElementwiseAttributes attr;
  attr.param = value;
  return Elementwise(src, attr, OperationType::ADD);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Add(
    const GpuModelBuilder::TensorHandle& src,
    const Tensor<Linear, DataType::FLOAT32>& value) {
  ElementwiseAttributes attr;
  attr.param = value;
  return Elementwise(src, attr, OperationType::ADD);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Add(
    const GpuModelBuilder::TensorHandle& left,
    const GpuModelBuilder::TensorHandle& right) {
  return Elementwise(left, right, OperationType::ADD);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Multiplication(
    const GpuModelBuilder::TensorHandle& src, float value) {
  ElementwiseAttributes attr;
  attr.param = value;
  return Elementwise(src, attr, OperationType::MUL);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Multiplication(
    const GpuModelBuilder::TensorHandle& src, double value) {
  ElementwiseAttributes attr;
  attr.param = static_cast<float>(value);
  return Elementwise(src, attr, OperationType::MUL);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Multiplication(
    const GpuModelBuilder::TensorHandle& src, int value) {
  ElementwiseAttributes attr;
  attr.param = value;
  return Elementwise(src, attr, OperationType::MUL);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Multiplication(
    const GpuModelBuilder::TensorHandle& src,
    const Tensor<Linear, DataType::FLOAT32>& value) {
  ElementwiseAttributes attr;
  attr.param = value;
  return Elementwise(src, attr, OperationType::MUL);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Multiplication(
    const GpuModelBuilder::TensorHandle& left,
    const GpuModelBuilder::TensorHandle& right) {
  return Elementwise(left, right, OperationType::MUL);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Tile(
    const GpuModelBuilder::TensorHandle& src, Axis axis, int tiles_count) {
  BHWC new_shape = src.tensor_desc.GetBHWCShape();
  new_shape.set(axis, new_shape.get(axis) * tiles_count);
  return Tile(src, new_shape);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Tile(
    const GpuModelBuilder::TensorHandle& src, const BHWC& new_shape) {
  GpuModelBuilder::TensorHandle dst =
      AddTensor(new_shape, src.tensor_desc.GetDataType());

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = "tile";
  gpu_node.inputs = {src.id};
  gpu_node.outputs = {dst.id};
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);

  gpu_node.gpu_operation = SelectTile(op_def);
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Tile(
    const GpuModelBuilder::TensorHandle& src, const BHWDC& new_shape) {
  GpuModelBuilder::TensorHandle dst =
      AddTensor(new_shape, src.tensor_desc.GetDataType());

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = "tile";
  gpu_node.inputs = {src.id};
  gpu_node.outputs = {dst.id};
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);

  gpu_node.gpu_operation = SelectTile(op_def);
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Reduce(
    const GpuModelBuilder::TensorHandle& src, Reduce::Type reduce_type,
    const std::set<Axis>& axis) {
  MeanAttributes attr;
  if (reduce_type == Reduce::Type::kMaximumIndex) {
    attr.dims = {*axis.begin()};  // MaximumIndex only supports 1 axis.
  } else {
    attr.dims = axis;
  }
  BHWDC new_shape = src.tensor_desc.GetBHWDCShape();
  for (const auto& axis : attr.dims) {
    new_shape.set(axis, 1);
  }
  const DataType dst_type = reduce_type == Reduce::Type::kMaximumIndex
                                ? DataType::INT32
                                : src.tensor_desc.GetDataType();
  GpuModelBuilder::TensorHandle dst = AddTensor(new_shape, dst_type);

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = Reduce::TypeToString(reduce_type);
  gpu_node.inputs = {src.id};
  gpu_node.outputs = {dst.id};
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);

  gpu_node.gpu_operation = std::make_unique<ml_drift::Reduce>(
      CreateReduce(attr.dims, src.tensor_desc.GetBHWDCShape(), reduce_type,
                   op_def, gpu_info_));
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Reduce(
    const GpuModelBuilder::TensorHandle& src, OperationType op_type,
    const std::set<Axis>& axis) {
  return Reduce(src, GetReduceTypeFromOperationType(op_type), axis);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Elementwise(
    const GpuModelBuilder::TensorHandle& src, ElementwiseDescriptor&& op_desc,
    const std::string& name) {
  const BHWC new_shape = src.tensor_desc.GetBHWCShape();
  GpuModelBuilder::TensorHandle dst =
      AddTensor(new_shape, src.tensor_desc.GetDataType());

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = name;
  gpu_node.inputs = {src.id};
  gpu_node.outputs = {dst.id};
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);

  GPUOperation operation = CreateGpuOperation(op_def, std::move(op_desc));
  gpu_node.gpu_operation = std::make_unique<GPUOperation>(std::move(operation));
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Gather(
    const TensorHandle& src, const TensorHandle& indices, Axis axis) {
  BHWC new_shape = src.tensor_desc.GetBHWCShape();
  new_shape.set(axis, indices.tensor_desc.GetBHWCShape().c);
  auto dst = AddTensor(new_shape, src.tensor_desc.GetDataType());

  GatherAttributes attr;
  attr.axis = axis;
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.src_tensors.push_back(indices.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);
  AddGpuOperation({src, indices}, dst,
                  std::make_unique<GPUOperation>(CreateGather(op_def, attr)),
                  "gather");
  return dst;
}

void GpuModelBuilder::AddGpuOperation(
    const std::vector<GpuModelBuilder::TensorHandle>& srcs,
    const std::vector<GpuModelBuilder::TensorHandle>& dsts,
    std::unique_ptr<GPUOperation>&& operation, const std::string& name) {
  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = name;
  for (const auto& src : srcs) {
    gpu_node.inputs.push_back(src.id);
  }
  for (const auto& dst : dsts) {
    gpu_node.outputs.push_back(dst.id);
  }
  gpu_node.gpu_operation = std::move(operation);
}

void GpuModelBuilder::AddGpuOperation(const std::vector<ValueId>& src_ids,
                                      const std::vector<ValueId>& dst_ids,
                                      std::unique_ptr<GPUOperation>&& operation,
                                      const std::string& name) {
  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = name;
  gpu_node.inputs = src_ids;
  gpu_node.outputs = dst_ids;
  gpu_node.gpu_operation = std::move(operation);
}

void GpuModelBuilder::AddGpuOperation(
    const std::vector<GpuModelBuilder::TensorHandle>& srcs,
    const GpuModelBuilder::TensorHandle& dst,
    std::unique_ptr<GPUOperation>&& operation, const std::string& name) {
  AddGpuOperation(srcs, std::vector<GpuModelBuilder::TensorHandle>{dst},
                  std::move(operation), name);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::ConcatInternal(
    const std::vector<GpuModelBuilder::TensorHandle>& srcs, Axis axis) {
  GpuModelBuilder::TensorHandle dst = AddTensor(
      GetOutputShapeConcat(srcs, axis), srcs[0].tensor_desc.GetDataType());
  std::vector<int> channels;
  channels.push_back(srcs[0].tensor_desc.GetBHWCShape().c);
  for (int i = 1; i < srcs.size(); ++i) {
    channels.push_back(srcs[i].tensor_desc.GetBHWCShape().c);
  }
  ConcatAttributes concat_attr;
  concat_attr.axis = axis;

  gpu_model_.nodes.push_back({});
  auto& concat_node = gpu_model_.nodes.back();
  concat_node.name = "concat_" + ToString(axis);
  for (int i = 0; i < srcs.size(); ++i) {
    concat_node.inputs.push_back(srcs[i].id);
  }
  concat_node.outputs = {dst.id};
  OperationDef concat_def;
  for (int i = 0; i < srcs.size(); ++i) {
    concat_def.src_tensors.push_back(srcs[i].tensor_desc);
  }
  concat_def.dst_tensors.push_back(dst.tensor_desc);
  auto status = SelectConcat(concat_attr, channels, concat_def, gpu_info_,
                             &concat_node.gpu_operation);
  if (!status.ok()) {
    ABSL_LOG(ERROR) << status.message();
  }
  return dst;
}

BHWC GpuModelBuilder::GetOutputShapeConcat(
    const std::vector<GpuModelBuilder::TensorHandle>& srcs, Axis axis) {
  BHWC dst_shape = srcs[0].tensor_desc.GetBHWCShape();
  for (int i = 1; i < srcs.size(); ++i) {
    dst_shape.set(axis, dst_shape.get(axis) +
                            srcs[i].tensor_desc.GetBHWCShape().get(axis));
  }
  return dst_shape;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Concat(
    const std::vector<GpuModelBuilder::TensorHandle>& srcs, Axis axis) {
  GpuModelBuilder::TensorHandle dst = AddTensor(
      GetOutputShapeConcat(srcs, axis), srcs[0].tensor_desc.GetDataType());
  const int max_src_images = GetMaxSrcImages(gpu_info_);
  const int max_src_buffers = gpu_info_.GetMaxBufferArguments();
  int max_inputs = std::max(2, std::min(max_src_images, max_src_buffers) - 4);
  if (gpu_info_.IsMali()) {
    // Mali can fail clEnqueueNDRangeKernel with "Out of resources" when it
    // receives too big kernel.
    max_inputs = std::min(8, max_inputs);
  }

  const int groups = DivideRoundUp(srcs.size(), max_inputs);
  for (int g = 0; g < groups; ++g) {
    std::vector<GpuModelBuilder::TensorHandle> new_srcs;
    if (g != 0) {
      // Concatenated tensor from previous concatenation operations.
      new_srcs.push_back(dst);
    }
    for (int i = 0; i < max_inputs; ++i) {
      int src_index = g * max_inputs + i;
      if (src_index >= srcs.size()) {
        break;
      }
      new_srcs.push_back(srcs[src_index]);
    }
    dst = ConcatInternal(new_srcs, axis);
  }
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Concat(
    const GpuModelBuilder::TensorHandle& first,
    const GpuModelBuilder::TensorHandle& second, Axis axis) {
  return ConcatInternal({first, second}, axis);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Cumsum(
    const GpuModelBuilder::TensorHandle& src, Axis axis) {
  GpuModelBuilder::TensorHandle dst =
      AddTensor(src.tensor_desc.GetBHWCShape(), src.tensor_desc.GetDataType());

  auto& cumsum_node = gpu_model_.nodes.emplace_back();
  cumsum_node.name = "cumsum";
  cumsum_node.inputs = {src.id};
  cumsum_node.outputs = {dst.id};
  OperationDef cumsum_def;
  cumsum_def.src_tensors.push_back(src.tensor_desc);
  cumsum_def.dst_tensors.push_back(dst.tensor_desc);
  auto cumsum_op =
      std::make_unique<GPUOperation>(CreateCumsum(cumsum_def, {axis}));
  cumsum_op->tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  cumsum_node.gpu_operation = std::move(cumsum_op);
  return dst;
}

void GpuModelBuilder::Copy(const TensorHandle& src, const TensorHandle& dst) {
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);
  auto op = CreateElementwiseOneInput(gpu_info_, op_def, OperationType::COPY);
  AddGpuOperation({src}, {dst}, std::make_unique<GPUOperation>(std::move(op)),
                  "copy");
}

GpuModelBuilder::TensorHandle GpuModelBuilder::LayerNormalization(
    const GpuModelBuilder::TensorHandle& src,
    const Tensor<Linear, DataType::FLOAT32>& gamma,
    const Tensor<Linear, DataType::FLOAT32>& beta, float epsilon) {
  const BHWC new_shape = src.tensor_desc.GetBHWCShape();
  GpuModelBuilder::TensorHandle dst =
      AddTensor(new_shape, src.tensor_desc.GetDataType());

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = "layer_normalization";
  gpu_node.inputs = {src.id};
  gpu_node.outputs = {dst.id};
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);

  MeanStdDevNormalization operation = CreateMeanStdDevNormalization(
      op_def, gpu_info_, new_shape, epsilon, gamma, beta, false);
  gpu_node.gpu_operation =
      std::make_unique<MeanStdDevNormalization>(std::move(operation));
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::HWCGroupNormalization(
    const GpuModelBuilder::TensorHandle& src, int groups,
    const Tensor<Linear, DataType::FLOAT32>& gamma,
    const Tensor<Linear, DataType::FLOAT32>& beta, float epsilon) {
  const BHWC new_shape = src.tensor_desc.GetBHWCShape();
  GpuModelBuilder::TensorHandle dst =
      AddTensor(new_shape, src.tensor_desc.GetDataType());

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = "hwcgroup_normalization";
  gpu_node.inputs = {src.id};
  gpu_node.outputs = {dst.id};
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);

  ml_drift::HWCGroupNormalization operation(op_def, gpu_info_, new_shape,
                                            groups, epsilon, gamma, beta);
  gpu_node.gpu_operation =
      std::make_unique<ml_drift::HWCGroupNormalization>(std::move(operation));
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::RMSNormalization(
    const TensorHandle& src, float epsilon,
    const Tensor<Linear, DataType::FLOAT32>* gamma,
    const Tensor<Linear, DataType::FLOAT32>* beta) {
  const BHWC new_shape = src.tensor_desc.GetBHWCShape();
  GpuModelBuilder::TensorHandle dst =
      AddTensor(new_shape, src.tensor_desc.GetDataType());

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = "rms_normalization";
  gpu_node.inputs = {src.id};
  gpu_node.outputs = {dst.id};
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);

  MeanStdDevNormalization operation =
      CreateRMSNormalization(op_def, gpu_info_, new_shape, epsilon);
  gpu_node.gpu_operation =
      std::make_unique<MeanStdDevNormalization>(std::move(operation));
  if (gamma) {
    dst = Multiplication(dst, *gamma);
  }
  if (beta) {
    dst = Add(dst, *beta);
  }
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::StatisticalTopK(
    const TensorHandle& src, float stddev_multiplier) {
  const BHWC new_shape = src.tensor_desc.GetBHWCShape();
  GpuModelBuilder::TensorHandle dst =
      AddTensor(new_shape, src.tensor_desc.GetDataType());

  gpu_model_.nodes.push_back({});
  GpuNode& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = "statistical_top_k";
  gpu_node.inputs = {src.id};
  gpu_node.outputs = {dst.id};

  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);
  gpu_node.gpu_operation =
      CreateStatisticalTopK(op_def, gpu_info_, new_shape, stddev_multiplier);
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::ResizeNearest(
    const GpuModelBuilder::TensorHandle& src, int scale, bool align_corners,
    bool half_pixel_centers) {
  Resize2DAttributes attr;
  attr.align_corners = align_corners;
  attr.half_pixel_centers = half_pixel_centers;
  attr.type = SamplingType::NEAREST;
  attr.new_shape.h = src.tensor_desc.GetBHWCShape().h * scale;
  attr.new_shape.w = src.tensor_desc.GetBHWCShape().w * scale;
  const BHWC new_shape =
      CalculateOutputShape(src.tensor_desc.GetBHWCShape(), attr);
  GpuModelBuilder::TensorHandle dst =
      AddTensor(new_shape, src.tensor_desc.GetDataType());

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = "resize";
  gpu_node.inputs = {src.id};
  gpu_node.outputs = {dst.id};
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);

  auto status = SelectResize(attr, op_def, &gpu_node.gpu_operation);
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::ResizeBilinear(
    const GpuModelBuilder::TensorHandle& src, const HW& new_shape,
    bool align_corners, bool half_pixel_centers) {
  Resize2DAttributes attr;
  attr.align_corners = align_corners;
  attr.half_pixel_centers = half_pixel_centers;
  attr.type = SamplingType::BILINEAR;
  attr.new_shape = new_shape;
  const BHWC output_shape =
      CalculateOutputShape(src.tensor_desc.GetBHWCShape(), attr);
  GpuModelBuilder::TensorHandle dst =
      AddTensor(output_shape, src.tensor_desc.GetDataType());

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = "resize_bilinear";
  gpu_node.inputs = {src.id};
  gpu_node.outputs = {dst.id};
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);

  auto status = SelectResize(attr, op_def, &gpu_node.gpu_operation);
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::DepthToSpace(
    const TensorHandle& src, int block_size) {
  const SpaceToDepthAttributes attr = {/*block_size=*/block_size};

  auto src_shape = src.tensor_desc.GetBHWCShape();
  const BHWC dst_shape =
      BHWC(src_shape.b, src_shape.h * block_size, src_shape.w * block_size,
           src_shape.c / (block_size * block_size));

  GpuModelBuilder::TensorHandle dst =
      AddTensor(dst_shape, src.tensor_desc.GetDataType());

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = "depth_to_space";
  gpu_node.inputs = {src.id};
  gpu_node.outputs = {dst.id};

  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);

  GPUOperation operation = CreateDepthToSpace(op_def, attr);
  gpu_node.gpu_operation = std::make_unique<GPUOperation>(std::move(operation));
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::SubTensor(
    const GpuModelBuilder::TensorHandle& src, const BHWC& start,
    const BHWC& size) {
  SliceAttributes attr;
  attr.starts = start;
  attr.ends = BHWC(start.b + size.b, start.h + size.h, start.w + size.w,
                   start.c + size.c);
  attr.strides = BHWC{1, 1, 1, 1};
  return StridedSlice(src, attr);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::StridedSlice(
    const GpuModelBuilder::TensorHandle& src, const SliceAttributes& attr) {
  auto src_shape = src.tensor_desc.GetBHWCShape();
  const BHWC dst_shape = CalculateOutputShape(src_shape, attr);
  GpuModelBuilder::TensorHandle dst =
      AddTensor(dst_shape, src.tensor_desc.GetDataType());

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = "strided_slice";
  gpu_node.inputs = {src.id};
  gpu_node.outputs = {dst.id};

  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);

  ml_drift::StridedSlice operation = CreateStridedSlice(op_def, attr);
  gpu_node.gpu_operation =
      std::make_unique<ml_drift::StridedSlice>(std::move(operation));
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::StridedSlice(
    const GpuModelBuilder::TensorHandle& src, const Slice3DAttributes& attr) {
  auto src_shape = src.tensor_desc.GetBHWDCShape();
  const BHWDC dst_shape = CalculateOutputShape(src_shape, attr);
  GpuModelBuilder::TensorHandle dst =
      AddTensor(dst_shape, src.tensor_desc.GetDataType());

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = "strided_slice_3d";
  gpu_node.inputs = {src.id};
  gpu_node.outputs = {dst.id};

  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);

  ml_drift::StridedSlice operation = CreateStridedSlice(op_def, attr);
  gpu_node.gpu_operation =
      std::make_unique<ml_drift::StridedSlice>(std::move(operation));
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Sampling(
    const GpuModelBuilder::TensorHandle& src_logits,
    const GpuModelBuilder::TensorHandle& src_indices,
    const GpuModelBuilder::TensorHandle& probabilities,
    const GpuModelBuilder::TensorHandle& params_i32_handle, int top_k_index) {
  auto src_shape = src_logits.tensor_desc.GetBHWCShape();
  BHWC dst_shape = src_shape;
  dst_shape.c = 1;
  auto dst = AddTensor(dst_shape, DataType::INT32);

  std::string code = absl::Substitute(R"(
MAIN_FUNCTION($$0) {
  int BW = ucl::GetGlobalId<0>();
  int H = ucl::GetGlobalId<1>();
  int B = BW % args.output.Batch();
  int W = BW / args.output.Batch();
  if (W >= args.output.Width() || H >= args.output.Height() || ucl::GetGlobalId<2>() != 0) return;

  args.src_logits.SetBatchRef(B);
  args.src_indices.SetBatchRef(B);
  args.probabilities.SetBatchRef(B);
  args.output.SetBatchRef(B);

  float probability = args.probabilities.Read<float>(W, H, 0).x;
  float cum_sum = 0.0f;
  int index = -1;
  int top_k = args.params_i32.Read($0);
  for (int s = 0; s < args.src_logits.Slices(); ++s) {
    float4 vals = ucl::Convert<float4>(args.src_logits.Read(W, H, s));
    int4 inds = args.src_indices.Read<int>(W, H, s);
    if (probability >= cum_sum && s * 4 + 0 < top_k) { index = inds.x; }
    cum_sum += vals.x;
    if (probability >= cum_sum && s * 4 + 1 < top_k) { index = inds.y; }
    cum_sum += vals.y;
    if (probability >= cum_sum && s * 4 + 2 < top_k) { index = inds.z; }
    cum_sum += vals.z;
    if (probability >= cum_sum && s * 4 + 3 < top_k) { index = inds.w; }
    cum_sum += vals.w;
  }
  int4 result = ucl::Init<int4>(index, 0, 0, 0);
  args.output.Write(result, W, H, 0);
})",
                                      top_k_index);

  GPUOperation custom_op;
  custom_op.AddSrcTensor("src_logits", src_logits.tensor_desc);
  custom_op.AddSrcTensor("src_indices", src_indices.tensor_desc);
  custom_op.AddSrcTensor("probabilities", probabilities.tensor_desc);
  BufferDescriptor params_i32_buffer;
  params_i32_buffer.element_type = DataType::INT32;
  params_i32_buffer.element_size = 1;
  custom_op.AddSrcBuffer("params_i32", params_i32_buffer);
  custom_op.AddDstTensor("output", dst.tensor_desc);
  custom_op.code_ = std::move(code);
  custom_op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_ZIs1;

  AddGpuOperation({src_logits, src_indices, probabilities, params_i32_handle},
                  {dst}, std::make_unique<GPUOperation>(std::move(custom_op)),
                  "sampling");
  return dst;
}

absl::StatusOr<GpuModelBuilder::TensorHandle> GpuModelBuilder::BatchedMatMul(
    const GpuModelBuilder::TensorHandle& left,
    const GpuModelBuilder::TensorHandle& right,
    const BatchedMatMulAttributes& batched_mat_mul_attr,
    const GpuModelBuilder::TensorHandle* src_exp,
    const ConvRuntimeCheckDesc& runtime_check,
    const TensorHandle* runtime_check_tensor) {
  GpuModelBuilder::TensorHandle mat_left =
      batched_mat_mul_attr.transpose_left ? Transpose(left, BHWC(0, 1, 3, 2))
                                          : left;
  BHWC left_shape = mat_left.tensor_desc.GetBHWCShape();
  BHWC right_shape = right.tensor_desc.GetBHWCShape();
  int out_channels =
      batched_mat_mul_attr.transpose_right ? right_shape.w : right_shape.c;
  BHWC dst_shape = BHWC(1, left_shape.h, left_shape.w, out_channels);
  OHWI weights_shape(right_shape.c, right_shape.b, right_shape.h,
                     right_shape.w);

  BHWDC mat_left_5d = mat_left.tensor_desc.GetBHWDCShape();
  int spatial_size = mat_left_5d.b * mat_left_5d.w * mat_left_5d.d;
  GpuModelBuilder::TensorHandle weights_handle = right;

  BHWDC dst_shape_5d = mat_left_5d;
  dst_shape_5d.c = out_channels;
  GpuModelBuilder::TensorHandle dst_handle =
      AddTensor(dst_shape_5d, left.tensor_desc.GetDataType());

  BHWC w_shape =
      BHWC(right_shape.w, right_shape.h, right_shape.b, right_shape.c);
  OHWI weights_shape_ohwi = OHWI(w_shape.b, w_shape.h, w_shape.w, w_shape.c);
  if (spatial_size <=
          GetRecommendedMaxTotalSpatialSize(
              gpu_info_, GetConvPrecision(left.tensor_desc.GetDataType())) &&
      batched_mat_mul_attr.transpose_right &&
      IsFullyConnectedWeightsAreSpatialTensorSupported(weights_shape_ohwi)) {
    auto weights = AddTensor(w_shape, right.tensor_desc.GetDataType());

    gpu_model_.nodes.push_back({});
    auto& fc_node = gpu_model_.nodes.back();
    fc_node.name = "batched_mat_mul_as_fc";
    if (runtime_check_tensor) {
      fc_node.name += " + runtime_check";
    }
    OperationDef op_def;
    op_def.src_tensors.push_back(mat_left.tensor_desc);
    op_def.src_tensors.push_back(weights.tensor_desc);
    op_def.dst_tensors.push_back(dst_handle.tensor_desc);
    ABSL_ASSIGN_OR_RETURN(
        auto fc_op,
        CreateFullyConnectedWeightsAreSpatialTensor(
            gpu_info_, op_def, GetConvPrecision(left.tensor_desc.GetDataType()),
            weights_shape_ohwi, nullptr, &dst_shape,
            /*wg_size=*/nullptr, runtime_check));
    fc_node.gpu_operation =
        std::make_unique<ml_drift::FullyConnected>(std::move(fc_op));
    fc_node.inputs = {mat_left.id, right.id};
    if (runtime_check_tensor && (runtime_check.src_end_ch_index.has_value() ||
                                 runtime_check.dst_end_ch_index.has_value())) {
      fc_node.inputs.push_back(runtime_check_tensor->id);
    }
    fc_node.outputs = {dst_handle.id};
    fc_node.gpu_operation->flops_ =
        GetConvolutionFlops(dst_shape, weights_shape) / weights_shape.h;
    return dst_handle;
  } else if (batched_mat_mul_attr.transpose_right) {
    // Add transpose for the second tensor. If we cannot use the
    // FullyConnected operation, we need to transpose the second tensor to
    // match the expected input shape of BMM.
    weights_handle = Transpose(right, BHWC(0, 1, 3, 2));
    weights_shape =
        OHWI(right_shape.w, right_shape.b, right_shape.h, right_shape.c);
  }

  std::unique_ptr<GPUOperation> conv_op;
  WeightsDescription conv_weights_desc;
  {
    OperationDef op_def;
    op_def.src_tensors.push_back(mat_left.tensor_desc);
    op_def.dst_tensors.push_back(dst_handle.tensor_desc);

    Convolution2DAttributes attr;
    attr.padding.prepended = HW(0, 0);
    attr.padding.appended = HW(0, 0);
    attr.strides = HW(1, 1);
    attr.dilations = HW(1, 1);
    auto& weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
    weights.shape = weights_shape;

    const TensorDescriptor* src_exp_td =
        src_exp ? &src_exp->tensor_desc : nullptr;
    conv_op = SelectConvolutionWithExternalWeights(
        attr, /*bias_desc*/ nullptr, dst_shape, gpu_info_, op_def,
        GetConvPrecision(left.tensor_desc.GetDataType()), hints_,
        &conv_weights_desc, src_exp_td,
        /*different_weights_for_height=*/true, runtime_check);
  }

  // The dwfh conv kernel (this BMM-as-conv path always uses
  // different_weights_for_height=true) indexes weights by output row
  // (0..dst height-1) and broadcasts the right-batch over the dst height,
  // so the layout's spatial extent (H*W) must be dst_shape.h, not the right
  // tensor's H*W. The converter derives its source-row divisor from W
  // (sp % W), which must stay the right tensor's batch, so grow H instead:
  // H*W = dst_shape.h with W unchanged.
  OHWI layout_weights_shape = weights_shape;
  layout_weights_shape.h = DivideRoundUp(dst_shape.h, weights_shape.w);
  std::vector<TensorHandle> conv_weights = WeightsConversion(
      weights_handle, Layout::HWIO, conv_weights_desc, layout_weights_shape);

  gpu_model_.nodes.push_back({});
  auto& conv_node = gpu_model_.nodes.back();
  conv_node.inputs = {mat_left.id};
  for (const auto& conv_weight : conv_weights) {
    conv_node.inputs.push_back(conv_weight.id);
  }
  if (src_exp) {
    conv_node.inputs.push_back(src_exp->id);
  }
  if (runtime_check_tensor && (runtime_check.src_end_ch_index.has_value() ||
                               runtime_check.dst_end_ch_index.has_value())) {
    conv_node.inputs.push_back(runtime_check_tensor->id);
  }
  conv_node.outputs = {dst_handle.id};
  conv_node.gpu_operation = std::move(conv_op);
  conv_node.name = absl::StrCat("mat_mul_as_convolution",
                                GetConvOpNameSuffix(*conv_node.gpu_operation));
  if (src_exp) {
    conv_node.name += " + src softmax";
  }
  if (runtime_check_tensor) {
    conv_node.name += " + runtime_check";
  }
  conv_node.gpu_operation->flops_ = 2.0 * dst_shape.b * dst_shape.h *
                                    dst_shape.w * dst_shape.c * weights_shape.i;
  return dst_handle;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Mask(
    const GpuModelBuilder::TensorHandle& true_tensor,
    GpuModelBuilder::TensorHandle* mask_tensor) {
  if (!mask_tensor) return true_tensor;
  return Add(true_tensor, *mask_tensor);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::SelectV2(
    const GpuModelBuilder::TensorHandle& cond,
    const GpuModelBuilder::TensorHandle& if_tensor,
    const GpuModelBuilder::TensorHandle& else_tensor) {
  BHWC cond_shape = cond.tensor_desc.GetBHWCShape();
  BHWC if_shape = if_tensor.tensor_desc.GetBHWCShape();
  BHWC else_shape = else_tensor.tensor_desc.GetBHWCShape();
  BHWC new_shape = cond_shape;
  new_shape.b = std::max(new_shape.b, std::max(if_shape.b, else_shape.b));
  new_shape.h = std::max(new_shape.h, std::max(if_shape.h, else_shape.h));
  new_shape.w = std::max(new_shape.w, std::max(if_shape.w, else_shape.w));
  new_shape.c = std::max(new_shape.c, std::max(if_shape.c, else_shape.c));
  GpuModelBuilder::TensorHandle dst =
      AddTensor(new_shape, if_tensor.tensor_desc.GetDataType());

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = "selectv2";
  gpu_node.inputs = {cond.id, if_tensor.id, else_tensor.id};
  gpu_node.outputs = {dst.id};
  OperationDef op_def;
  op_def.src_tensors.push_back(cond.tensor_desc);
  op_def.src_tensors.push_back(if_tensor.tensor_desc);
  op_def.src_tensors.push_back(else_tensor.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);

  GPUOperation operation = CreateSelectV2(op_def, {});
  gpu_node.gpu_operation = std::make_unique<GPUOperation>(std::move(operation));
  return dst;
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
GpuModelBuilder::SoftmaxBatchedMatMul(
    const GpuModelBuilder::TensorHandle& left,
    const GpuModelBuilder::TensorHandle& right) {
  if (UseFusedSoftmaxWithBatchedMatMul(left)) {
    auto reduced_exp_tensor = SoftmaxReduce(left);
    return BatchedMatMul(left, right, /*attr=*/{}, &reduced_exp_tensor);
  } else {
    return BatchedMatMul(Softmax(left), right);
  }
}

bool GpuModelBuilder::UseFusedSoftmaxWithBatchedMatMul(
    const GpuModelBuilder::TensorHandle& softmax_input) const {
  const auto& shape = softmax_input.tensor_desc.GetBHWCShape();
  bool fused_softmax =
      shape.w >
      GetRecommendedMaxTotalSpatialSize(
          gpu_info_, GetConvPrecision(softmax_input.tensor_desc.GetDataType()));
  if (gpu_info_.IsMali() && shape.c >= 64) {
    fused_softmax = false;
  }
  return fused_softmax;
}

void GpuModelBuilder::Split(const GpuModelBuilder::TensorHandle& src, Axis axis,
                            std::vector<GpuModelBuilder::TensorHandle>* dsts) {
  SplitAttributes split_attr;
  split_attr.axis = axis;

  gpu_model_.nodes.push_back({});
  auto& split_node = gpu_model_.nodes.back();
  split_node.name = "split";
  split_node.inputs = {src.id};
  split_node.outputs.reserve(dsts->size());
  for (int i = 0; i < dsts->size(); ++i) {
    split_node.outputs.push_back(dsts->at(i).id);
  }
  OperationDef split_def;
  split_def.src_tensors.push_back(src.tensor_desc);
  split_def.dst_tensors.reserve(dsts->size());
  for (int i = 0; i < dsts->size(); ++i) {
    split_def.dst_tensors.push_back(dsts->at(i).tensor_desc);
  }
  std::vector<int> channels;
  if (axis == Axis::CHANNELS) {
    channels.reserve(dsts->size());
    for (int i = 0; i < dsts->size(); ++i) {
      channels.push_back(dsts->at(i).tensor_desc.GetBHWCShape().c);
    }
  }
  SelectSplit(split_attr, gpu_info_, channels, split_def,
              &split_node.gpu_operation);
}

std::vector<GpuModelBuilder::TensorHandle> GpuModelBuilder::Split(
    const GpuModelBuilder::TensorHandle& src, Axis axis,
    const std::vector<int>& sizes) {
  BHWDC new_shape = src.tensor_desc.GetBHWDCShape();
  // Skip split if trivial case.
  if (sizes.size() == 1) {
    if ((axis == Axis::BATCH && new_shape.b == sizes[0]) ||
        (axis == Axis::HEIGHT && new_shape.h == sizes[0]) ||
        (axis == Axis::WIDTH && new_shape.w == sizes[0]) ||
        (axis == Axis::DEPTH && new_shape.d == sizes[0]) ||
        (axis == Axis::CHANNELS && new_shape.c == sizes[0])) {
      return {src};
    }
  }

  std::vector<GpuModelBuilder::TensorHandle> dsts;
  // We do some bookkeeping to ensure we don't overrun the max storage argument
  // counts. We assume that buffers are always 'storage', but imagebuffers and
  // textures are 'storage' when outputs and non-'storage' when inputs. If that
  // implicit agreement is ever broken, then this code will need to be updated
  // as well! Also, since we can't easily predetermine the storage type for a
  // tensor in advance, it's not easy for us to determine if chained outputs
  // will be a storage buffer or a storage texture, so to ensure correctness, we
  // just always add one to that count, even though that may artificially reduce
  // our maximum splitting factor in some (rare) cases when storage textures and
  // buffers are mixed. This could be improved if needed by actually creating
  // and checking the combined Tensor (DRest below) as we iterate. The actual
  // splitting algorithm is a simple recursive greedy linear splitting:
  // S -> [D0, D1, D2, D3, ..., DK]. If only 4 destination tensors "fit":
  // S -> [D0, D1, D2, DRest] and DRest --> [D3, ..., DK]
  uint32_t storage_buffer_arguments =
      (src.tensor_desc.GetStorageType() == TensorStorageType::BUFFER ||
       src.tensor_desc.GetStorageType() == TensorStorageType::UNKNOWN)
          ? 1
          : 0;
  uint32_t storage_texture_arguments = 0;

  const uint32_t max_image_arguments = gpu_info_.GetMaxImageArguments();
  const uint32_t max_buffer_arguments = gpu_info_.GetMaxBufferArguments();

  uint32_t max_total_arguments =
      std::max(max_buffer_arguments, max_image_arguments);
  if (gpu_info_.IsMali()) {
    // Mali can fail clEnqueueNDRangeKernel with "Out of resources" when it
    // receives too big kernel.
    // Replace single complex split to N with N simple kernels.
    max_total_arguments = 4;
  }

  int index = 0;
  for (; index < sizes.size() - 1; ++index) {
    new_shape.set(axis, sizes[index]);
    GpuModelBuilder::TensorHandle dst =
        AddTensor(new_shape, src.tensor_desc.GetDataType());
    dsts.push_back(dst);

    // We keep running tallies of storage arguments.
    const auto& storage_type = dst.tensor_desc.GetStorageType();
    if (storage_type == TensorStorageType::BUFFER ||
        storage_type == TensorStorageType::IMAGE_BUFFER ||
        storage_type == TensorStorageType::UNKNOWN) {
      storage_buffer_arguments++;
    } else {
      storage_texture_arguments++;
    }

    // If we're within one of our maximum storage argument bounds, then we exit
    // early and merge all remaining sizes into a single chunk for a follow-up
    // split op.
    if (storage_buffer_arguments + 1 == max_buffer_arguments ||
        storage_texture_arguments + 1 == max_image_arguments ||
        (storage_buffer_arguments + storage_texture_arguments + 1 ==
         max_total_arguments)) {
      index++;
      break;
    }
  }

  // Merge all remaining sizes into a single chunk, and make a destination
  // Tensor for it.
  int dst_last_size = 0;
  std::vector<int> remaining_sizes;
  for (; index < sizes.size(); index++) {
    dst_last_size += sizes[index];
    remaining_sizes.push_back(sizes[index]);
  }
  BHWDC dst_last_shape = src.tensor_desc.GetBHWDCShape();
  dst_last_shape.set(axis, dst_last_size);
  GpuModelBuilder::TensorHandle dst_last =
      AddTensor(dst_last_shape, src.tensor_desc.GetDataType());
  dsts.push_back(dst_last);

  // Perform first split.
  Split(src, axis, &dsts);

  // If final chunk needs additional splitting, recurse.
  if (remaining_sizes.size() != 1) {
    auto follow_up_split_results = Split(dsts.back(), axis, remaining_sizes);
    dsts.pop_back();
    dsts.insert(dsts.end(),
                std::make_move_iterator(follow_up_split_results.begin()),
                std::make_move_iterator(follow_up_split_results.end()));
  }
  return dsts;
}

std::vector<GpuModelBuilder::TensorHandle> GpuModelBuilder::Split(
    const GpuModelBuilder::TensorHandle& src, Axis axis, int tile_size) {
  const int tiles_count = src.tensor_desc.GetBHWCShape().get(axis) / tile_size;
  const std::vector<int> sizes(tiles_count, tile_size);
  return Split(src, axis, sizes);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Padding(
    const TensorHandle& src, const PadAttributes& attr) {
  const BHWC new_shape =
      CalculateOutputShape(src.tensor_desc.GetBHWCShape(), attr);
  GpuModelBuilder::TensorHandle dst =
      AddTensor(new_shape, src.tensor_desc.GetDataType());

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = "pad";
  gpu_node.inputs = {src.id};
  gpu_node.outputs = {dst.id};
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);
  gpu_node.gpu_operation =
      SelectPadding(gpu_info_, attr, op_def, src.tensor_desc.GetBHWCShape().c);
  return dst;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::Pooling(
    const TensorHandle& src, const Pooling2DAttributes& attr) {
  const BHWC new_shape =
      CalculateOutputShape(src.tensor_desc.GetBHWCShape(), attr);
  GpuModelBuilder::TensorHandle dst =
      AddTensor(new_shape, src.tensor_desc.GetDataType());

  gpu_model_.nodes.push_back({});
  auto& gpu_node = gpu_model_.nodes.back();
  gpu_node.name = "pooling";
  gpu_node.inputs = {src.id};
  gpu_node.outputs = {dst.id};
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);
  gpu_node.gpu_operation = SelectPooling(attr, gpu_info_, op_def);

  return dst;
}

//    a_tensor   b_tensor
//          \    /
//          matmul
//            |
//         softmax   c_tensor
//              \     /
//              matmul
//                |
//              output
absl::StatusOr<GpuModelBuilder::TensorHandle>
GpuModelBuilder::BatchedMatMulSoftmaxBatchedMatMul(
    const GpuModelBuilder::TensorHandle& a_tensor,
    const GpuModelBuilder::TensorHandle& b_tensor,
    const GpuModelBuilder::TensorHandle& c_tensor,
    GpuModelBuilder::TensorHandle* mask_tensor) {
  if (!mask_tensor && a_tensor.tensor_desc.GetBHWCShape().w >= 1024) {
    const int src_ch = a_tensor.tensor_desc.GetBHWCShape().c;
    const int interm_ch = b_tensor.tensor_desc.GetBHWCShape().c;
    const int dst_ch = c_tensor.tensor_desc.GetBHWCShape().c;
    if (IsConvSoftmaxConvSupported(
            gpu_info_, GetConvPrecision(a_tensor.tensor_desc.GetDataType()),
            src_ch, interm_ch, dst_ch)) {
      return BatchedMatMulSoftmaxBatchedMatMulSingleKernel(a_tensor, b_tensor,
                                                           c_tensor);
    }
  }
  return BatchedMatMulSoftmaxBatchedMatMulSeparateKernels(
      a_tensor, b_tensor, c_tensor, mask_tensor);
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
GpuModelBuilder::BatchedMatMulSoftmaxBatchedMatMulSeparateKernels(
    const GpuModelBuilder::TensorHandle& a_tensor,
    const GpuModelBuilder::TensorHandle& b_tensor,
    const GpuModelBuilder::TensorHandle& c_tensor,
    GpuModelBuilder::TensorHandle* mask_tensor) {
  int matrix_size = a_tensor.tensor_desc.GetBHWCShape().w *
                    b_tensor.tensor_desc.GetBHWCShape().c;
  GpuModelBuilder::TensorHandle att;
  bool tiled = false;
  int tile_size = 1;
  if (gpu_info_.IsIntel()) {
    if (matrix_size >= 4096 * 4096) {
      tile_size = 2;
      tiled = true;
    } else if (matrix_size >= 4096 * 2048) {
      tile_size = 4;
      tiled = true;
    } else if (matrix_size >= 2048 * 2048) {
      tile_size = 8;
      tiled = true;
    }
  } else if (gpu_info_.IsMali()) {
    if (matrix_size >= 4096 * 4096) {
      tile_size = 4;
      tiled = true;
    } else if (matrix_size >= 4096 * 2048) {
      tile_size = 8;
      tiled = true;
    }
  } else {
    if (matrix_size >= 4096 * 4096) {
      tile_size = 1;
      tiled = true;
    } else if (matrix_size >= 4096 * 2048) {
      tile_size = 2;
      tiled = true;
    } else if (matrix_size >= 2048 * 2048) {
      tile_size = 4;
      tiled = true;
    } else if (matrix_size >= 2048 * 1024) {
      tile_size = 8;
      tiled = true;
    }
  }
  tiled = tiled && a_tensor.tensor_desc.GetBHWCShape().h % tile_size == 0 &&
          a_tensor.tensor_desc.GetBHWCShape().h != tile_size;
  if (tiled) {
    auto q_tensors = Split(a_tensor, Axis::HEIGHT, tile_size);
    auto k_tensors = Split(b_tensor, Axis::HEIGHT, tile_size);
    auto v_tensors = Split(c_tensor, Axis::HEIGHT, tile_size);
    std::vector<GpuModelBuilder::TensorHandle> dsts(q_tensors.size());

    for (int i = 0; i < q_tensors.size(); ++i) {
      ABSL_ASSIGN_OR_RETURN(auto tmp,
                            BatchedMatMul(q_tensors[i], k_tensors[i]));
      tmp = Mask(tmp, mask_tensor);
      ABSL_ASSIGN_OR_RETURN(dsts[i], SoftmaxBatchedMatMul(tmp, v_tensors[i]));
    }

    att = Concat(dsts, Axis::HEIGHT);
  } else {
    ABSL_ASSIGN_OR_RETURN(att, BatchedMatMul(a_tensor, b_tensor));
    att = Mask(att, mask_tensor);
    ABSL_ASSIGN_OR_RETURN(att, SoftmaxBatchedMatMul(att, c_tensor));
  }
  return att;
}

//    a_tensor   b_tensor
//          \    /
//          matmul
//            |
//         softmax   c_tensor
//              \     /
//              matmul
//                |
//              output
GpuModelBuilder::TensorHandle
GpuModelBuilder::BatchedMatMulSoftmaxBatchedMatMulSingleKernel(
    const GpuModelBuilder::TensorHandle& a_tensor,
    const GpuModelBuilder::TensorHandle& b_tensor,
    const GpuModelBuilder::TensorHandle& c_tensor) {
  BHWC a_shape = a_tensor.tensor_desc.GetBHWCShape();
  BHWC b_shape = b_tensor.tensor_desc.GetBHWCShape();
  BHWC c_shape = c_tensor.tensor_desc.GetBHWCShape();
  const int width = a_shape.w;
  const int batch_size = a_shape.h;
  const int src_ch = a_shape.c;
  const int interm_ch = b_shape.c;
  const int dst_ch = c_shape.c;
  const OHWI weights_0_shape(interm_ch, 1, batch_size, src_ch);
  const OHWI weights_1_shape(dst_ch, 1, batch_size, interm_ch);
  GpuModelBuilder::TensorHandle dst = AddTensor(
      BHWC(1, batch_size, width, dst_ch), a_tensor.tensor_desc.GetDataType());

  std::vector<WeightsDescription> weights_descs =
      GetWeightsDescsForConvSoftmaxConv(
          gpu_info_, GetConvPrecision(a_tensor.tensor_desc.GetDataType()),
          src_ch, interm_ch, dst_ch);

  GpuModelBuilder::TensorHandle weights_0_tensor = WeightsConversion(
      b_tensor, Layout::HWIO, weights_descs[0], weights_0_shape)[0];
  GpuModelBuilder::TensorHandle weights_1_tensor = WeightsConversion(
      c_tensor, Layout::HWIO, weights_descs[1], weights_1_shape)[0];

  gpu_model_.nodes.push_back({});
  auto& conv_node = gpu_model_.nodes.back();

  conv_node.inputs = {a_tensor.id, weights_0_tensor.id, weights_1_tensor.id};
  conv_node.outputs = {dst.id};
  OperationDef conv_def;
  conv_def.src_tensors.push_back(a_tensor.tensor_desc);
  conv_def.src_tensors.push_back(weights_0_tensor.tensor_desc);
  conv_def.src_tensors.push_back(weights_1_tensor.tensor_desc);
  conv_def.dst_tensors.push_back(dst.tensor_desc);
  ConvSoftmaxConv::WeightsDesc weights_desc;
  weights_desc.constant = false;
  weights_desc.src_ch = src_ch;
  weights_desc.interm_ch = interm_ch;
  weights_desc.dst_ch = dst_ch;
  conv_node.gpu_operation =
      std::make_unique<ConvSoftmaxConv>(CreateConvSoftmaxConv(
          gpu_info_, conv_def,
          GetConvPrecision(a_tensor.tensor_desc.GetDataType()), weights_desc));
  conv_node.name = "matmul_softmax_matmul";
  {
    const int64_t conv0_flops_per_element = weights_0_shape.i * 2;
    const int64_t conv0_dst_elements = width * batch_size * weights_0_shape.o;
    const int64_t conv0_flops_count =
        conv0_dst_elements * conv0_flops_per_element;
    const int64_t conv1_flops_per_element = weights_1_shape.i * 2;
    const int64_t conv1_dst_elements = width * batch_size * weights_1_shape.o;
    const int64_t conv1_flops_count =
        conv1_dst_elements * conv1_flops_per_element;
    conv_node.gpu_operation->flops_ = conv0_flops_count + conv1_flops_count;
  }
  return dst;
}

absl::Status GpuModelBuilder::UpdateOutputTensor(
    const GpuModelBuilder::TensorHandle& output, ValueId new_id) {
  return UpdateOutputTensors(std::vector<GpuModelBuilder::TensorHandle>{output},
                             std::vector<ValueId>{new_id});
}

absl::Status GpuModelBuilder::UpdateOutputTensors(
    const std::vector<GpuModelBuilder::TensorHandle>& outputs,
    const std::vector<ValueId>& new_ids) {
  // this function is equivalent to
  // for (int i = 0; i < outputs.size(); ++i) {
  //  ValueId out_id = outputs[i].id;
  //  ValueId new_out_id = new_ids[i];
  //  CopyOp(out_id, new_out_id);
  // }
  // MergeNodes will merge the Copy nodes into the original nodes.
  for (int i = 0; i < outputs.size(); ++i) {
    ValueId out_id = outputs[i].id;
    ValueId new_out_id = new_ids[i];
    ABSL_ASSIGN_OR_RETURN(auto node_and_index,
                          GetNodeAndIndexByOutputId(out_id));
    auto& node = gpu_model_.nodes[node_and_index.first];
    int out_index = node_and_index.second;
    node.outputs[out_index] = new_out_id;
    ABSL_ASSIGN_OR_RETURN(auto new_tensor, GetTensor(new_out_id));
    if (new_tensor.tensor_desc.GetBHWDCShape() !=
        outputs[i].tensor_desc.GetBHWDCShape()) {
      // Shapes from GraphFloat32 can be calculated differently from
      // GpuModelBuilder.
      ABSL_LOG(WARNING)
          << "Shape mismatch in GpuModelBuilder::UpdateOutputTensors(output of "
          << node.name << "). Requested shape:"
          << ToString(new_tensor.tensor_desc.GetBHWDCShape())
          << " vs reference shape: "
          << ToString(outputs[i].tensor_desc.GetBHWDCShape());
    }
    ABSL_RETURN_IF_ERROR(node.gpu_operation->SetOutputDescriptor(
        out_index, new_tensor.tensor_desc));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::pair<int, int>> GpuModelBuilder::GetNodeAndIndexByOutputId(
    ValueId output_id) const {
  for (int node_index = gpu_model_.nodes.size() - 1; node_index >= 0;
       node_index--) {
    auto& node = gpu_model_.nodes[node_index];
    for (int out_index = 0; out_index < node.outputs.size(); ++out_index) {
      if (node.outputs[out_index] == output_id) {
        return std::make_pair(node_index, out_index);
      }
    }
  }
  return absl::NotFoundError("Output id not found");
}

absl::Status GpuModelBuilder::GetGpuModel(
    const std::vector<ValueId>& input_ids,
    const std::vector<ValueId>& output_ids, GpuModel* gpu_model) {
  std::vector<std::pair<ValueId, ValueId>> input_ids_and_refs(input_ids.size());
  for (int i = 0; i < input_ids.size(); ++i) {
    const ValueId id = input_ids[i];
    input_ids_and_refs[i] = {id, id};
  }
  std::vector<std::pair<ValueId, ValueId>> output_ids_and_refs(
      output_ids.size());
  for (int i = 0; i < output_ids.size(); ++i) {
    const ValueId id = output_ids[i];
    output_ids_and_refs[i] = {id, id};
  }
  return GetGpuModel(input_ids_and_refs, output_ids_and_refs, gpu_model);
}

absl::Status GpuModelBuilder::GetGpuModel(
    const std::vector<std::pair<ValueId, ValueId>>& input_ids_and_refs,
    const std::vector<std::pair<ValueId, ValueId>>& output_ids_and_refs,
    GpuModel* gpu_model) {
  gpu_model_.input_ids_and_refs = input_ids_and_refs;
  gpu_model_.output_ids_and_refs = output_ids_and_refs;
  ABSL_RETURN_IF_ERROR(MergeNodes(gpu_info_, &gpu_model_));
  ABSL_RETURN_IF_ERROR(AssembleCode(gpu_info_, &gpu_model_));
  ABSL_RETURN_IF_ERROR(ResolveArgs(&gpu_model_));
  ExpandSubgraphs(&gpu_model_);
  *gpu_model = std::move(gpu_model_);
  for (const auto& node : gpu_model->nodes) {
    if (!node.gpu_operation) {
      continue;
    }
    const int num_inputs = node.inputs.size();
    const int num_outputs = node.outputs.size();
    const int op_num_inputs = node.gpu_operation->GetSrcTensorsNames().size();
    const int op_num_outputs = node.gpu_operation->GetDstTensorsNames().size();
    if (num_inputs != op_num_inputs || num_outputs != op_num_outputs) {
      return absl::InternalError(absl::StrCat(
          "Node ", node.name, " has ", num_inputs, " inputs and ", num_outputs,
          " outputs, but the operation has ", op_num_inputs, " inputs and ",
          op_num_outputs, " outputs."));
    }
  }
  return absl::OkStatus();
}

GpuModelBuilder::TensorHandle GpuModelBuilder::MakeGelu(
    const GpuModelBuilder::TensorHandle& src) {
  return Elementwise(src, OperationType::GELU);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::MakeGeluTanh(
    const GpuModelBuilder::TensorHandle& src) {
  return Elementwise(src, OperationType::GELU_TANH_APPROX);
}

GPUOperation GpuModelBuilder::CreateNormalize(
    const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& mean,
    const GpuModelBuilder::TensorHandle& mean_squares,
    const GpuModelBuilder::TensorHandle& dst,
    const Tensor<HWC, DataType::FLOAT32>& gamma,
    const Tensor<HWC, DataType::FLOAT32>& beta, float epsilon) {
  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.src_tensors.push_back(mean.tensor_desc);
  op_def.src_tensors.push_back(mean_squares.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);

  GPUOperation op;
  op.AddSrcTensor("src_tensor", src.tensor_desc);
  op.AddSrcTensor("mean_tensor", mean.tensor_desc);
  op.AddSrcTensor("mean_sq_tensor", mean_squares.tensor_desc);
  op.AddDstTensor("dst", dst.tensor_desc);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  op.args_.AddFloat("variance_bias", epsilon);

  const BHWC shape = BHWC(1, gamma.shape.h, gamma.shape.w, gamma.shape.c);

  const DataType float_type = src.tensor_desc.GetDataType();
  TensorDescriptor gamma_tensor_desc =
      TensorDescriptor(float_type, default_storage_, Layout::HWC);
  auto status =
      gamma_tensor_desc.UpdateToSupportedStorageType(gpu_info_, shape);
  gamma_tensor_desc.UploadData(gamma);
  op.args_.AddObject("gamma", std::make_unique<TensorDescriptor>(
                                  std::move(gamma_tensor_desc)));

  TensorDescriptor beta_tensor_desc =
      TensorDescriptor(float_type, default_storage_, Layout::HWC);
  status = beta_tensor_desc.UpdateToSupportedStorageType(gpu_info_, shape);
  beta_tensor_desc.UploadData(beta);
  op.args_.AddObject(
      "beta", std::make_unique<TensorDescriptor>(std::move(beta_tensor_desc)));

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int X = linear_id / args.dst.Batch();\n";
    c += "  int B = linear_id % args.dst.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.mean_tensor.SetBatchRef(B);\n";
    c += "  args.mean_sq_tensor.SetBatchRef(B);\n";
    c += "  args.dst.SetBatchRef(B);\n";
  } else {
    c += "  int X = ucl::GetGlobalId<0>();\n";
  }
  c += R"(
    int Y = ucl::GetGlobalId<1>();
    int S = ucl::GetGlobalId<2>();
    if (X >= args.dst.Width() || Y >= args.dst.Height() || S >= args.dst.Slices()) return;
    float mean = args.mean_tensor.Read<float>(X, 0, 0).x;
    float mean_sq = args.mean_sq_tensor.Read<float>(X, 0, 0).x;
    float variance = mean_sq - mean * mean;
    float stddev_inv = rsqrt(variance + args.variance_bias);

    float4 t = args.src_tensor.Read<float>(X, Y, S);
    float4 t_normalized = (t - mean) * stddev_inv;
    t_normalized *= args.gamma.Read<float>(X, 0, S);
    t_normalized += args.beta.Read<float>(X, 0, S);
    args.dst::type result = ucl::Convert<args.dst::type>(t_normalized);
    args.dst.Write(result, X, Y, S);
  })";
  op.code_ = std::move(c);
  return op;
}

GpuModelBuilder::TensorHandle GpuModelBuilder::HWCGroupNorm(
    const GpuModelBuilder::TensorHandle& src, int groups, float epsilon,
    const Tensor<Linear, DataType::FLOAT32>& gamma,
    const Tensor<Linear, DataType::FLOAT32>& beta) {
  const auto& src_shape = src.tensor_desc.GetBHWCShape();
  const int group_size = src_shape.c / groups;
  if (group_size % 2 == 0 || group_size == 1) {
    return HWCGroupNormalization(src, groups, gamma, beta, epsilon);
  }

  Tensor<HWC, DataType::FLOAT32> gamma_hwc;
  gamma_hwc.shape = HWC(1, groups, src.tensor_desc.GetBHWCShape().c / groups);
  gamma_hwc.data.resize(gamma_hwc.shape.DimensionsProduct());
  Tensor<HWC, DataType::FLOAT32> beta_hwc;
  beta_hwc.shape = HWC(1, groups, src.tensor_desc.GetBHWCShape().c / groups);
  beta_hwc.data.resize(beta_hwc.shape.DimensionsProduct());

  for (int w = 0; w < gamma_hwc.shape.w; ++w) {
    for (int c = 0; c < gamma_hwc.shape.c; ++c) {
      gamma_hwc.data[gamma_hwc.shape.LinearIndex({0, w, c})] =
          gamma.data[w * gamma_hwc.shape.c + c];
    }
  }
  for (int w = 0; w < beta_hwc.shape.w; ++w) {
    for (int c = 0; c < beta_hwc.shape.c; ++c) {
      beta_hwc.data[beta_hwc.shape.LinearIndex({0, w, c})] =
          beta.data[w * beta_hwc.shape.c + c];
    }
  }

  auto x = Reshape(src, BHWC(src_shape.b, src_shape.h * src_shape.w, groups,
                             src_shape.c / groups));
  auto mean = Reduce(x, Reduce::Type::kMean, {Axis::HEIGHT, Axis::CHANNELS});
  auto mean_squares =
      Reduce(x, Reduce::Type::kMeanSquares, {Axis::HEIGHT, Axis::CHANNELS});

  auto normalized =
      AddTensor(x.tensor_desc.GetBHWCShape(), src.tensor_desc.GetDataType());

  GPUOperation normalized_op = CreateNormalize(
      x, mean, mean_squares, normalized, gamma_hwc, beta_hwc, epsilon);

  AddGpuOperation({x, mean, mean_squares}, normalized,
                  std::make_unique<GPUOperation>(std::move(normalized_op)),
                  "normalized");
  return Reshape(normalized, src_shape);
}

GpuModelBuilder::TensorHandle GpuModelBuilder::SiLU(
    const GpuModelBuilder::TensorHandle& src) {
  return Multiplication(src, Elementwise(src, OperationType::SIGMOID));
}

std::vector<GpuModelBuilder::TensorHandle> GpuModelBuilder::TopK(
    const GpuModelBuilder::TensorHandle& src, int top_k_size) {
  // TopKInternal expects src to be [1, BxHxW, C/4, 4]
  const auto& src_shape = src.tensor_desc.GetBHWCShape();
  if (src_shape.c % 4 != 0) {
    ABSL_LOG(ERROR)
        << "TopK requires src tensor C dimension to be divisible by 4.";
  }
  auto src_reshaped = Reshape(
      src,
      BHWC(1, src_shape.b * src_shape.h * src_shape.w, src_shape.c / 4, 4));
  auto results = TopKInternal(src_reshaped, top_k_size);
  auto out_max = Reshape(
      results[0], BHWC(src_shape.b, src_shape.h, src_shape.w, top_k_size));
  auto out_ind = Reshape(
      results[1], BHWC(src_shape.b, src_shape.h, src_shape.w, top_k_size));
  return {out_max, out_ind};
}

std::vector<GpuModelBuilder::TensorHandle> GpuModelBuilder::TopKInternal(
    const GpuModelBuilder::TensorHandle& src, int top_k_size) {
  constexpr int kMaxSizeForOneReductionStep = 4096;
  const auto& src_shape = src.tensor_desc.GetBHWCShape();
  auto dst_max = AddTensor(BHWC(1, src_shape.h, 1, top_k_size),
                           src.tensor_desc.GetDataType());
  auto dst_ind =
      AddTensor(BHWC(1, src_shape.h, 1, top_k_size), DataType::INT32);

  const int reduction_size = src_shape.w;
  for (int k_step = 0; k_step < DivideRoundUp(top_k_size, 4); ++k_step) {
    int k_offset = k_step * 4;
    std::vector<GpuModelBuilder::TensorHandle> srcs = {src};
    if (k_step != 0) {
      srcs.push_back(dst_max);
      srcs.push_back(dst_ind);

      // WebGPU does not allow read+write concurrently, so this fails if we use
      // the same tensors. Therefore we create new ones for each new step when
      // using one-step reduction. This is already not a problem for two-step
      // reduction.
      // TODO: b/409896028 - Handle this sort of issue more generically.
      if (reduction_size <= kMaxSizeForOneReductionStep &&
          gpu_info_.IsApiWebGpu()) {
        dst_max = AddTensor(BHWC(1, src_shape.h, 1, top_k_size),
                            src.tensor_desc.GetDataType());
        dst_ind =
            AddTensor(BHWC(1, src_shape.h, 1, top_k_size), DataType::INT32);
        Copy(srcs[1], dst_max);
        Copy(srcs[2], dst_ind);
      }
    }
    std::string name = "top_k step " + std::to_string(k_step);
    if (reduction_size <= kMaxSizeForOneReductionStep) {
      // one step reduction
      OperationDef op_def;
      for (const auto& source_tensor : srcs) {
        op_def.src_tensors.push_back(source_tensor.tensor_desc);
      }
      op_def.dst_tensors.push_back(dst_max.tensor_desc);
      op_def.dst_tensors.push_back(dst_ind.tensor_desc);

      TopKOp top_k_op =
          CreateTopK(gpu_info_, op_def, /*use_wg_reduction=*/true, k_offset);
      AddGpuOperation(srcs, {dst_max, dst_ind},
                      std::make_unique<TopKOp>(std::move(top_k_op)), name);
    } else {
      // two step reduction
      auto interm_max =
          AddTensor(BHWC(1, src_shape.h, kMaxSizeForOneReductionStep, 4),
                    src.tensor_desc.GetDataType());
      auto interm_ind =
          AddTensor(BHWC(1, src_shape.h, kMaxSizeForOneReductionStep, 4),
                    DataType::INT32);

      OperationDef op_def_first;
      for (const auto& source_tensor : srcs) {
        op_def_first.src_tensors.push_back(source_tensor.tensor_desc);
      }
      op_def_first.dst_tensors.push_back(interm_max.tensor_desc);
      op_def_first.dst_tensors.push_back(interm_ind.tensor_desc);
      TopKOp op_first = CreateTopK(gpu_info_, op_def_first,
                                   /*use_wg_reduction=*/false, k_offset);
      AddGpuOperation(srcs, {interm_max, interm_ind},
                      std::make_unique<TopKOp>(std::move(op_first)),
                      name + " part 0");
      OperationDef op_def_second;
      op_def_second.src_tensors.push_back(interm_max.tensor_desc);
      op_def_second.src_tensors.push_back(interm_ind.tensor_desc);
      op_def_second.dst_tensors.push_back(dst_max.tensor_desc);
      op_def_second.dst_tensors.push_back(dst_ind.tensor_desc);
      TopKOp op_second = CreateTopK(gpu_info_, op_def_second,
                                    /*use_wg_reduction=*/true, k_offset);
      AddGpuOperation({interm_max, interm_ind}, {dst_max, dst_ind},
                      std::make_unique<TopKOp>(std::move(op_second)),
                      name + " part 1");
    }
  }

  return {dst_max, dst_ind};
}

GpuModelBuilder::TensorHandle GpuModelBuilder::GetScalarTensor(
    const GpuModelBuilder::TensorHandle& tensor, BHWC coord) {
  return SubTensor(tensor, coord, BHWC{1, 1, 1, 1});
}

GpuModelBuilder::OptionalNodeContext GpuModelBuilder::BeginOptionalNodes(
    int tag, const TensorHandle& src) {
  return OptionalNodeContext{
      /*start_node=*/gpu_model_.nodes.size(),
      /*src=*/src,
      /*tag=*/tag,
  };
}

absl::Status GpuModelBuilder::EndOptionalNodes(OptionalNodeContext context,
                                               const TensorHandle& final_tensor,
                                               bool add_copy_to_src) {
  if (!add_copy_to_src) {
    for (size_t i = context.start_node; i < gpu_model_.nodes.size(); i++) {
      gpu_model_.nodes[i].optional_tag.insert(context.tag);
    }
    return absl::OkStatus();
  }
  if (final_tensor.tensor_desc != context.src.tensor_desc ||
      final_tensor.tensor_desc.GetBHWDCShape() !=
          context.src.tensor_desc.GetBHWDCShape()) {
    return absl::FailedPreconditionError(
        "Optional nodes must start and end with the same tensor shape.");
  }

  GPUOperation custom_op;
  custom_op.AddSrcTensor("src", final_tensor.tensor_desc);
  custom_op.AddDstTensor("dst", context.src.tensor_desc);
  std::string c = R"(
  MAIN_FUNCTION($0) {
    int linear_xb = ucl::GetGlobalId<0>();
    int X = linear_xb / args.dst.Batch();
    int B = linear_xb % args.dst.Batch();
    args.src.SetBatchRef(B);
    args.dst.SetBatchRef(B);
)";
  std::string coords = "X, Y";
  if (context.src.tensor_desc.HasAxis(Axis::DEPTH)) {
    coords += ", D";
    c += "    int linear_y = ucl::GetGlobalId<1>();\n";
    c += "    int Y = linear_y / args.dst.Depth();\n";
    c += "    int D = linear_y % args.dst.Depth();\n";
  } else {
    c += "    int Y = ucl::GetGlobalId<1>();\n";
  }
  coords += ", S";
  c += "    int S = ucl::GetGlobalId<2>();\n";

  c += "    if (X >= args.dst.Width() || Y >= args.dst.Height() || S >= "
       "args.dst.Slices()) {\n";
  c += "      return;\n";
  c += "    }\n";
  c += "    args.dst::type result = args.src.Read(" + coords + ");\n";
  c += "    args.dst.Write(result, " + coords + ");\n";
  c += "  }\n";

  custom_op.code_ = std::move(c);
  custom_op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  AddGpuOperation({final_tensor}, {context.src},
                  std::make_unique<GPUOperation>(std::move(custom_op)),
                  "copy_optional_result");

  for (size_t i = context.start_node; i < gpu_model_.nodes.size(); i++) {
    gpu_model_.nodes[i].optional_tag.insert(context.tag);
  }
  return absl::OkStatus();
}

absl::Status GpuModelBuilder::RegisterSubgraph(
    GpuModelBuilder subgraph_builder, const std::string& subgraph_id,
    const std::vector<TensorHandle>& inputs,
    const std::vector<TensorHandle>& outputs) {
  std::vector<ValueId> input_ids;
  input_ids.reserve(inputs.size());
  for (const auto& input : inputs) {
    input_ids.push_back(input.id);
  }
  std::vector<ValueId> output_ids;
  output_ids.reserve(outputs.size());
  for (const auto& output : outputs) {
    output_ids.push_back(output.id);
  }

  GpuModel subgraph_model;
  {
    subgraph_model = std::move(subgraph_builder.gpu_model_);
    for (auto id : input_ids) {
      subgraph_model.input_ids_and_refs.push_back({id, id});
    }
    for (auto id : output_ids) {
      subgraph_model.output_ids_and_refs.push_back({id, id});
    }
    ABSL_RETURN_IF_ERROR(MergeNodes(gpu_info_, &subgraph_model));
  }

  gpu_model_.subgraphs[subgraph_id] = std::move(subgraph_model);
  return absl::OkStatus();
}

absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>>
GpuModelBuilder::Subgraph(const std::string& subgraph_id,
                          const std::vector<TensorHandle>& inputs) {
  if (!HasSubgraph(subgraph_id)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Subgraph does not exist with id: ", subgraph_id));
  }

  GpuModel& subgraph = gpu_model_.subgraphs[subgraph_id];

  // Remap subgraph input/output ids to the new ids for the main graph.
  absl::flat_hash_map<ValueId, ValueId> id_map;
  for (int i = 0; i < inputs.size(); ++i) {
    ValueId subgraph_in = subgraph.input_ids_and_refs[i].first;
    id_map[subgraph_in] = inputs[i].id;

    const auto& sub_td = subgraph.tensors[subgraph_in];
    const auto& base_td = gpu_model_.const_tensors.contains(inputs[i].id)
                              ? gpu_model_.const_tensors[inputs[i].id]
                              : gpu_model_.tensors[inputs[i].id];
    if (sub_td != base_td ||
        sub_td.GetBHWDCShape() != base_td.GetBHWDCShape()) {
      return absl::InternalError("Subgraph input tensors don't match.");
    }
  }

  std::vector<TensorHandle> outputs(subgraph.output_ids_and_refs.size());
  for (int i = 0; i < outputs.size(); ++i) {
    ValueId subgraph_out = subgraph.output_ids_and_refs[i].first;
    auto new_out = AddTensor(subgraph.tensors[subgraph_out]);
    id_map[subgraph_out] = new_out.id;
    outputs[i] = new_out;
  }

  // TODO: b/328473621 - Implement subgraph node sharing in other backends.
  if (gpu_info_.IsApiWebGpu()) {
    // This op has inputs and outputs matching the subgraph's inputs and
    // outputs. It will be expanded into the subgraph nodes in MergeNodes().
    AddGpuOperation(inputs, outputs, std::make_unique<GPUOperation>(),
                    "subgraph " + subgraph_id);
    gpu_model_.nodes.back().subgraph_id = subgraph_id;
    return outputs;
  }

  for (const auto& [id, desc] : subgraph.tensors) {
    if (!id_map.contains(id)) {
      id_map[id] = AddTensor(desc).id;
    }
  }

  for (auto& [id, desc] : subgraph.const_tensors) {
    if (!id_map.contains(id)) {
      id_map[id] = AddConstantTensor(std::move(desc)).id;
    }
  }

  // Copy subgraph nodes to main graph and remap ids.
  for (GpuNode& node : subgraph.nodes) {
    for (ValueId& id : node.inputs) {
      id = id_map[id];
    }
    for (ValueId& id : node.outputs) {
      id = id_map[id];
    }
    gpu_model_.nodes.push_back(std::move(node));
  }

  // Erase subgraph since the nodes have been consumed.
  gpu_model_.subgraphs.erase(subgraph_id);

  return outputs;
}

GpuModelBuilder::Weights CreateExternalWeights(
    const GpuModelBuilder::TensorHandle& weights,
    const WeightsDescription& weights_desc, const OHWI& weights_shape,
    const OHWI& scale_zp_shape, const GpuModelBuilder::TensorHandle* scale,
    const GpuModelBuilder::TensorHandle* zero_point,
    const GpuModelBuilder::TensorHandle* sum_i) {
  GpuModelBuilder::Weights external_weights;
  external_weights.weights = weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_shape;
  external_weights.scale_zp_shape = scale_zp_shape;
  external_weights.scale = scale ? std::make_optional(*scale) : std::nullopt;
  external_weights.zero_point =
      zero_point ? std::make_optional(*zero_point) : std::nullopt;
  external_weights.sum_i = sum_i ? std::make_optional(*sum_i) : std::nullopt;
  return external_weights;
}

bool WeightsManager::IsGpuWeightsPreparationSupported(const GpuInfo& gpu_info) {
  // TODO(linchan): Enable weights preparation on Gpu for PowerVR, Broadcom,
  // and Mali GPUs.
  // Weights preparation on Gpu for PowerVR Gpu is currently disabled
  // because it's slow on Pixel 10 (Pixel 10: b/435509090) and doesn't
  // compatible with Broadcom GPUs.
  // Weights preparation on Gpu for Mali Gpu is currently disabled
  // because Mali GPUs crash during weight conversion due to memory
  // exhaustion and known driver issues (b/514701303).
  bool is_gpu_chip_supported =
      !gpu_info.IsPowerVR() && !gpu_info.IsBroadcom() && !gpu_info.IsMali();
  bool is_api_supported = gpu_info.gpu_api == GpuApi::kOpenCl ||
                          gpu_info.gpu_api == GpuApi::kWebGpu ||
                          gpu_info.gpu_api == GpuApi::kMetal;

  bool is_supported = is_gpu_chip_supported && is_api_supported;
  if (!is_supported) {
    ABSL_LOG(WARNING) << "Weights preparation on Gpu is disabled for PowerVR, "
                         "Broadcom, Mali GPUs and non-OpenCL/WebGPU/Metal "
                         "backends.";
  }
  return is_supported;
}

absl::Status WeightsManager::CreateConversionGpuModel(
    const GpuInfo& gpu_info, GpuModel* gpu_model,
    absl::flat_hash_map<ValueId, ValueId>* io_mapping,
    std::vector<WeightsManager::UploadWeightsInfo>* upload_weights_infos) {
  GpuModelBuilder model_builder(gpu_info, /*hints=*/{},
                                CalculationsPrecision::F32,
                                TensorStorageType::BUFFER);
  std::vector<ValueId> input_ids;
  std::vector<ValueId> output_ids;

  // Two conversion requests may use the same raw weight tensor (eg. kStandard
  // and kWeightsSumI). `raw_weight_tensor_id_map` is used to avoid one raw
  // weight tensor to be allocated multiple times.
  using RawWeightKey =
      std::tuple<const void*, OHWI, DataType, absl::Span<const float>,
                 absl::Span<const int>>;
  absl::flat_hash_map<RawWeightKey, GpuModelBuilder::TensorHandle>
      raw_weight_tensor_id_map;

  for (const auto& request : weights_conversion_requests_) {
    RawWeightKey key = {
        request.src_data_ptr,
        request.weights_shape,
        request.src_data_type,
        request.scale_data,
        request.zero_point_data,
    };
    auto it = raw_weight_tensor_id_map.find(key);
    if (it == raw_weight_tensor_id_map.end()) {
      it = raw_weight_tensor_id_map
               .emplace(key, AddRawWeightTensor(model_builder, request))
               .first;
      size_t elements_count = GetElementsCountForDataType(
          request.src_data_type, request.weights_shape);
      UploadWeightsInfo upload_info = {
          .input_id = it->second.id,
          .data = request.src_data_ptr,
          .size = elements_count * SizeOf(request.src_data_type),
      };
      upload_weights_infos->push_back(std::move(upload_info));
    }
    auto& weights_raw = it->second;
    std::vector<GpuModelBuilder::TensorHandle> internal_outputs;
    switch (request.dst_weights_type) {
      case TargetWeightsType::kStandard: {
        GpuModelBuilder::TensorHandle scale_handle;
        GpuModelBuilder::TensorHandle* scale_handle_ptr = nullptr;
        GpuModelBuilder::TensorHandle zp_handle;
        GpuModelBuilder::TensorHandle* zp_handle_ptr = nullptr;
        if (!request.scale_data.empty()) {
          auto scale_desc = TensorDescriptor(
              DataType::FLOAT32, TensorStorageType::BUFFER, Layout::LINEAR);
          scale_desc.SetBHWCShape(BHWC(1, 1, 1, request.scale_data.size()));
          scale_desc.UploadData(request.scale_data.data());
          scale_handle = model_builder.AddConstantTensor(std::move(scale_desc));
          scale_handle_ptr = &scale_handle;

          if (!request.zero_point_data.empty()) {
            TensorDescriptor zp_desc = TensorDescriptor(
                DataType::INT32, TensorStorageType::BUFFER, Layout::LINEAR);
            int zp_size = request.zero_point_data.size() == 1
                              ? request.scale_data.size()
                              : request.zero_point_data.size();
            zp_desc.SetBHWCShape(BHWC(1, 1, 1, zp_size));
            if (zp_size != request.zero_point_data.size()) {
              // Unzip the zero point data.
              std::vector<int> zp_data(zp_size);
              for (int i = 0; i < zp_size; ++i) {
                zp_data[i] = request.zero_point_data[0];
              }
              zp_desc.UploadData(zp_data.data());
            } else {
              zp_desc.UploadData(request.zero_point_data.data());
            }
            zp_handle = model_builder.AddConstantTensor(std::move(zp_desc));
            zp_handle_ptr = &zp_handle;
          }
        }
        internal_outputs = model_builder.WeightsConversion(
            weights_raw, Layout::OHWI, request.weights_desc,
            request.weights_shape, scale_handle_ptr, zp_handle_ptr);
      } break;
      case TargetWeightsType::kWinograd3x3:
        internal_outputs = model_builder.GetWinograd3x3WeightsFromOHWI(
            weights_raw, request.weights_shape, request.weights_desc);
        break;
      case TargetWeightsType::kWeightsSumI:
        // For kWeightsSumI, we need have a special case for INT4 as they will
        // be stored in INT32 format.
        if (request.src_data_type == DataType::INT4)
          weights_raw.tensor_desc.SetDataType(DataType::INT32);
        internal_outputs = {model_builder.GetWeightsSumIFromRawOHWI(
            weights_raw, request.weights_shape, request.src_data_type)};
        break;
      default:
        return absl::UnimplementedError(absl::Substitute(
            "The requested weights type $0 is not implemented yet.",
            ToString(request.dst_weights_type)));
    }
    input_ids.push_back(weights_raw.id);
    for (int i = 0; i < internal_outputs.size(); ++i) {
      (*io_mapping)[internal_outputs[i].id] = request.main_model_weights_ids[i];
      output_ids.push_back(internal_outputs[i].id);
    }
  }
  return model_builder.GetGpuModel(input_ids, output_ids, gpu_model);
}


TensorDescriptor MakeRawWeightTensorDescriptor(
    const WeightsManager::WeightsConversionRequest& request) {
  TensorDescriptor tensor_desc = TensorDescriptor(
      request.src_data_type, TensorStorageType::BUFFER, Layout::LINEAR);
  size_t elements_count =
      GetElementsCountForDataType(request.src_data_type, request.weights_shape);
  tensor_desc.SetBHWCShape(BHWC(1, 1, 1, elements_count));
  return tensor_desc;
}

GpuModelBuilder::TensorHandle WeightsManager::AddRawWeightTensor(
    GpuModelBuilder& model_builder, const WeightsConversionRequest& request) {
  TensorDescriptor tensor_desc = MakeRawWeightTensorDescriptor(request);
  return model_builder.AddTensor(std::move(tensor_desc));
}

bool WeightsManager::ShouldOffloadPreparationToGPU(
    const GpuInfo& gpu_info, const OHWI& weights_shape,
    TargetWeightsType preparation_type) {
  // Compared the latency between GPU and CPU of rearranging different sizes of
  // weights. The `threshold` is the weight size when the CPU latency and GPU
  // latency are close.
  size_t threshold = 1 << 24;
  switch (preparation_type) {
    case TargetWeightsType::kStandard:
      if (gpu_info.IsAdreno()) {
        // TODO: b/487838846 - Update the threshold to 1<<19 when the bug is
        // fixed.
        threshold = (1 << 22);
      } else if (gpu_info.IsMali()) {
        threshold = (1 << 21);
      }
      break;
    case TargetWeightsType::kWinograd3x3:
      threshold = (1 << 13) * 3 * 3;
      if (gpu_info.IsAdreno()) {
        threshold = (1 << 12) * 3 * 3;
      } else if (gpu_info.IsMali()) {
        threshold = (1 << 13) * 3 * 3;
      }
      break;
    default:
      break;
  }
  const size_t num_elements = weights_shape.DimensionsProduct();
  return num_elements >= threshold;
}

std::vector<WeightsManager::WeightsPrepOperationInfo>
WeightsManager::ConvertWeightsPrepRequestsToOperations(
    const GpuInfo& gpu_info, std::vector<WeightsConversionRequest>&& requests) {
  std::vector<WeightsManager::WeightsPrepOperationInfo> conversion_operations;
  conversion_operations.reserve(requests.size());

  for (const auto& request : requests) {
    if (request.dst_weights_type == TargetWeightsType::kStandard &&
        request.scale_data.empty()) {
      // Build TensorDescriptor for src and dst tensors.
      auto src_desc = MakeRawWeightTensorDescriptor(request);
      auto dst_descs = GetTensorDescriptorsForWeightsLayout(
          request.weights_shape, request.weights_desc);
      // Build the GpuOperation.
      OperationDef op_def;
      op_def.src_tensors.push_back(src_desc);
      for (auto& dst : dst_descs) {
        op_def.dst_tensors.push_back(dst);
      }
      auto src_layout = Layout::OHWI;
      auto gpu_operation = SelectConverterToConvWeights(
          gpu_info, request.weights_shape, request.weights_desc, op_def,
          /*hints=*/ModelHints{}, src_layout, nullptr, nullptr);
      size_t elements_count = GetElementsCountForDataType(
          request.src_data_type, request.weights_shape);
      WeightsManager::WeightsPrepOperationInfo op_info = {
          .main_model_weight_id = request.main_model_weights_ids[0],
          .src_desc = std::move(src_desc),
          .dst_descs = std::move(dst_descs),
          .gpu_operation = std::move(gpu_operation),
          .data_ptr = request.src_data_ptr,
          .size = elements_count * SizeOf(request.src_data_type),
      };
      conversion_operations.push_back(std::move(op_info));
    } else if (request.dst_weights_type == TargetWeightsType::kWeightsSumI) {
      auto src_desc = MakeRawWeightTensorDescriptor(request);
      if (request.src_data_type == DataType::INT4) {
        src_desc.SetDataType(DataType::INT32);
      }
      TensorDescriptor dst_desc = TensorDescriptor(
          DataType::INT32, TensorStorageType::BUFFER, Layout::LINEAR);
      dst_desc.SetBHWCShape(BHWC(1, 1, 1, request.weights_shape.o));

      OperationDef op_def;
      op_def.src_tensors.push_back(src_desc);
      op_def.dst_tensors.push_back(dst_desc);

      auto gpu_operation =
          std::make_unique<GPUOperation>(CreateAccumulateInputChannels(
              op_def, request.weights_shape, request.src_data_type));
      size_t elements_count = GetElementsCountForDataType(
          request.src_data_type, request.weights_shape);
      WeightsManager::WeightsPrepOperationInfo op_info = {
          .main_model_weight_id = request.main_model_weights_ids[0],
          .src_desc = MakeRawWeightTensorDescriptor(request),
          .dst_descs = {std::move(dst_desc)},
          .gpu_operation = std::move(gpu_operation),
          .data_ptr = request.src_data_ptr,
          .size = elements_count * SizeOf(request.src_data_type),
      };
      conversion_operations.push_back(std::move(op_info));
    } else {
      ABSL_LOG(FATAL) << absl::Substitute(
          "The requested weights type $0 is not implemented yet.",
          ToString(request.dst_weights_type));
      return {};
    }
  }
  return conversion_operations;
}

std::vector<std::vector<WeightsManager::WeightsPrepOperationInfo>>
WeightsManager::BatchGpuOperations(
    std::vector<WeightsPrepOperationInfo>&& operations,
    ScheduleStrategy schedule_strategy, size_t total_shared_tensor_size) {
  std::vector<std::vector<WeightsManager::WeightsPrepOperationInfo>> batches;
  switch (schedule_strategy) {
    case ScheduleStrategy::kDefaultBatch: {
      const size_t num_batches =
          (operations.size() + kDefaultBatchSize - 1) / kDefaultBatchSize;
      batches.reserve(num_batches);
      for (size_t i = 0; i < operations.size(); i += kDefaultBatchSize) {
        auto start_it = operations.begin() + i;
        auto end_it = operations.begin() +
                      std::min(i + kDefaultBatchSize, operations.size());
        batches.emplace_back(std::make_move_iterator(start_it),
                             std::make_move_iterator(end_it));
      }
      break;
    }
    case ScheduleStrategy::kBatchByMaxWeightSize: {
      // 10% was chosen through some very rough experiments and may need to be
      // tuned in the future.
      const size_t percentage_of_total_shared_tensor_size =
          total_shared_tensor_size / 10;
      const size_t min_sum_of_weight_sizes_per_batch = std::min<size_t>(
          256 * 1024 * 1024, percentage_of_total_shared_tensor_size);
      size_t max_size = min_sum_of_weight_sizes_per_batch;
      for (const auto& op : operations) {
        max_size = std::max(max_size, op.size);
      }
      std::vector<WeightsPrepOperationInfo> current_batch;
      size_t current_batch_size = 0;
      for (auto& op : operations) {
        if (!current_batch.empty() && current_batch_size + op.size > max_size) {
          batches.push_back(std::move(current_batch));
          current_batch.clear();
          current_batch_size = 0;
        }
        current_batch_size += op.size;
        current_batch.push_back(std::move(op));
      }
      if (!current_batch.empty()) {
        batches.push_back(std::move(current_batch));
      }
      break;
    }
    default:
      ABSL_LOG(FATAL) << "Unsupported schedule strategy: "
                      << static_cast<int>(schedule_strategy);
  }
  return batches;
}

absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>>
GpuModelBuilder::AppendOp(std::string_view op_name,
                          const std::vector<TensorHandle>& inputs,
                          const OpAttrs& attrs) {
  auto op = OpRegistry::Global().Create(op_name);
  if (!op) {
    return absl::NotFoundError(
        absl::StrCat("Operation not found in registry: ", op_name));
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<AttrSpec> specs,
                        OpRegistry::Global().GetAttrSpecs(op_name));

  OpAttrs mutable_attrs = attrs;
  ABSL_RETURN_IF_ERROR(ValidateAndNormalizeAttrs(specs, mutable_attrs));

  std::vector<TensorHandle> outputs;
  size_t initial_nodes = this->gpu_model_.nodes.size();
  uint64_t initial_tensors = this->id_counter_;

  absl::Status status = op->Build(*this, inputs, mutable_attrs, outputs);
  if (!status.ok()) {
    this->gpu_model_.nodes.resize(initial_nodes);
    for (uint64_t i = initial_tensors; i < id_counter_; ++i) {
      this->gpu_model_.tensors.erase(i);
      this->gpu_model_.const_tensors.erase(i);
    }
    this->id_counter_ = initial_tensors;
    return status;
  }

  for (size_t i = initial_nodes; i < this->gpu_model_.nodes.size(); ++i) {
    this->gpu_model_.nodes[i].name =
        absl::StrCat(op_name, "/", this->gpu_model_.nodes[i].name);
  }

  return outputs;
}

void GpuModelBuilder::AddSrcTensor(GPUOperation* op, const std::string& name,
                                   const TensorHandle& handle) {
  op->AddSrcTensor(name, handle.tensor_desc);
  for (int idx = gpu_model_.nodes.size() - 1; idx >= 0; --idx) {
    if (gpu_model_.nodes[idx].gpu_operation.get() == op) {
      gpu_model_.nodes[idx].inputs.push_back(handle.id);
      break;
    }
  }
}

void GpuModelBuilder::AddDstTensor(GPUOperation* op, const std::string& name,
                                   const TensorHandle& handle) {
  op->AddDstTensor(name, handle.tensor_desc);
  for (int idx = gpu_model_.nodes.size() - 1; idx >= 0; --idx) {
    if (gpu_model_.nodes[idx].gpu_operation.get() == op) {
      gpu_model_.nodes[idx].outputs.push_back(handle.id);
      break;
    }
  }
}

}  // namespace ml_drift
