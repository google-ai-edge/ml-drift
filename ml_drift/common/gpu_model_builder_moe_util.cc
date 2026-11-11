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

#include "ml_drift/common/gpu_model_builder_moe_util.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/kernels/conv_apple_mpp.h"
#include "ml_drift/common/kernels/conv_generic.h"
#include "ml_drift/common/kernels/conv_wave_matrix.h"
#include "ml_drift/common/kernels/conv_wave_memory.h"
#include "ml_drift/common/kernels/fully_connected.h"
#include "ml_drift/common/kernels/special/experts_remap.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

std::vector<GpuModelBuilder::TensorHandle> CreateExpertsRemap(
    GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& indices,
    int num_experts) {
  const BHWC& indices_shape = indices.tensor_desc.GetBHWCShape();
  auto experts_remap = builder.AddTensor(
      BHWC(1, num_experts, indices_shape.w, 2), DataType::kInt32);
  auto experts_remap_op =
      CreateExpertsRemapOp(indices.tensor_desc, experts_remap.tensor_desc);
  experts_remap_op->flops_ = indices_shape.DimensionsProduct();

  TensorDescriptor dst_count_desc = TensorDescriptor(
      DataType::kInt32, TensorStorageType::kBuffer, Layout::kHWC);
  dst_count_desc.SetBHWCShape(BHWC(1, 1, 1, num_experts));
  auto experts_count = builder.AddTensor(dst_count_desc);

  builder.AddGpuOperation({indices}, {experts_remap, experts_count},
                          std::move(experts_remap_op), "create_experts_remap");

  auto experts_offsets = builder.AddTensor(dst_count_desc);
  auto experts_offsets_op = CreateOffsetsOp();
  experts_offsets_op->flops_ = num_experts;
  builder.AddGpuOperation({experts_count}, {experts_offsets},
                          std::move(experts_offsets_op),
                          "create_experts_offsets");

  auto experts_remap_packed = builder.AddTensor(
      BHWC(1, 1, indices_shape.c * indices_shape.w, 2), DataType::kInt32);
  auto experts_remap_packed_op = CreateLinearizeMapOp(
      experts_remap.tensor_desc, experts_remap_packed.tensor_desc);
  experts_remap_packed_op->read_size_ =
      experts_remap_packed.tensor_desc.GetMemorySizeInBytes();
  experts_remap_packed_op->flops_ =
      experts_remap_packed.tensor_desc.GetBHWCShape().DimensionsProduct();
  builder.AddGpuOperation(
      {experts_remap, experts_count, experts_offsets}, {experts_remap_packed},
      std::move(experts_remap_packed_op), "linearize_experts_remap");

  auto storage = builder.default_storage();
  builder.SetDefaultStorage(TensorStorageType::kBuffer);
  auto experts_params =
      builder.Concat({experts_count, experts_offsets}, Axis::kChannels);
  builder.SetDefaultStorage(storage);

  return {experts_remap, experts_params, experts_remap_packed};
}

GpuModelBuilder::TensorHandle ExpertsRemapTo(
    GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& experts_remap,
    int num_active_experts) {
  const BHWC& src_shape = src.tensor_desc.GetBHWCShape();
  auto remapped_src = builder.AddTensor(
      BHWC(src_shape.b, 1, src_shape.w * num_active_experts, src_shape.c),
      src.tensor_desc.GetDataType());
  auto remap_to = CreateExpertsRemapToOp(
      src.tensor_desc, experts_remap.tensor_desc, remapped_src.tensor_desc);
  builder.AddGpuOperation({src, experts_remap}, {remapped_src},
                          std::move(remap_to), "experts_remap_to");
  return remapped_src;
}

GpuModelBuilder::TensorHandle ExpertsRemapFrom(
    GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& experts_remap,
    int num_active_experts) {
  const auto shape = src.tensor_desc.GetBHWCShape();
  auto remapped_dst = builder.AddTensor(
      BHWC(shape.b, num_active_experts, shape.w / num_active_experts, shape.c),
      src.tensor_desc.GetDataType());
  auto remap_from = CreateExpertsRemapFromOp(
      src.tensor_desc, experts_remap.tensor_desc, remapped_dst.tensor_desc);
  builder.AddGpuOperation({src, experts_remap}, {remapped_dst},
                          std::move(remap_from), "experts_remap_from");
  return remapped_dst;
}

absl::StatusOr<GpuModelBuilder::TensorHandle> MakeConvWithPackedGroups(
    GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& params,
    const GpuModelBuilder::Weights& weights, int num_active_experts) {
  const auto precision =
      builder.GetConvPrecision(src.tensor_desc.GetDataType());
  BHWC dst_shape = src.tensor_desc.GetBHWCShape();
  dst_shape.c = weights.shape.o;
  auto dst = builder.AddTensor(dst_shape, src.tensor_desc.GetDataType());

  OperationDef op_def;
  op_def.src_tensors.push_back(src.tensor_desc);
  op_def.dst_tensors.push_back(dst.tensor_desc);

  ConvRuntimeCheckDesc::PackedGroups packed_groups;
  packed_groups.params_offset = 0;
  packed_groups.num_groups = weights.shape.h;
  packed_groups.max_group_size =
      src.tensor_desc.GetBHWCShape().w / num_active_experts;
  int average_task_size =
      DivideRoundUp(src.tensor_desc.GetBHWCShape().w, packed_groups.num_groups);
  ConvRuntimeCheckDesc runtime_check;
  runtime_check.packed_groups = packed_groups;

  std::unique_ptr<GPUOperation> conv;
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

  Convolution2DAttributes conv_attr;
  conv_attr.padding.prepended = HW(0, 0);
  conv_attr.padding.appended = HW(0, 0);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  auto& conv_attr_weights =
      conv_attr.weights.emplace<Tensor<OHWI, DataType::kFloat32>>();
  conv_attr_weights.shape = weights.shape;

  std::vector<GpuModelBuilder::TensorHandle> src_ids;
  src_ids.push_back(src);
  bool weights_conversion = false;
  if (average_task_size >= 32 &&
      SupportsConvAppleMPP(builder.gpu_info(), external_weights) &&
      precision == CalculationsPrecision::kF16) {
    auto conv_apple_mpp = CreateConvAppleMPPExternalWeights(
        src.tensor_desc, dst.tensor_desc, external_weights,
        /*bias=*/nullptr, /*src_exp=*/nullptr,
        /*different_weights_for_height=*/true, runtime_check, &dst_shape);
    conv = std::make_unique<ConvAppleMPP>(std::move(conv_apple_mpp));
  } else if (average_task_size >= 32 &&
             SupportsConvWaveMatrix(builder.gpu_info(), precision,
                                    external_weights)) {
    auto conv_wave_matrix = CreateConvWaveMatrixExternalWeights(
        op_def, precision, dst_shape, external_weights, builder.gpu_info(),
        /*bias=*/nullptr, /*src_exp=*/nullptr,
        /*different_weights_for_height=*/true, runtime_check);
    conv = std::make_unique<ConvWaveMatrix>(std::move(conv_wave_matrix));
  } else if (average_task_size >= 32 &&
             SupportsConvGeneric(builder.gpu_info(), precision,
                                 external_weights)) {
    auto conv_generic = CreateConvGenericExternalWeights(
        builder.gpu_info(), op_def, precision, external_weights,
        /*bias=*/nullptr, &dst_shape,
        /*src_exp=*/nullptr, /*different_weights_for_height=*/true,
        runtime_check);
    conv = std::make_unique<ConvGeneric>(std::move(conv_generic));
  } else if (average_task_size >= 64 &&
             IsConvWaveMemorySupported(builder.gpu_info())) {
    auto conv_wave_memory = CreateConvWaveMemoryExternalWeights(
        builder.gpu_info(), op_def, precision, conv_attr,
        /*bias=*/nullptr, &dst_shape,
        /*src_exp=*/nullptr, /*different_weights_for_height=*/true,
        runtime_check);
    auto conv_weights_desc = conv_wave_memory.GetWeightsDescription();
    conv = std::make_unique<ConvWaveMemory>(std::move(conv_wave_memory));
    if (!(conv_weights_desc == weights.desc)) {
      weights_conversion = true;
      auto weights_tensors =
          builder.WeightsConversion(weights, conv_weights_desc);
      src_ids.push_back(weights_tensors[0]);
    }
  } else {
    ABSL_ASSIGN_OR_RETURN(
        auto conv_fc,
        CreateFullyConnectedExternalWeights(
            builder.gpu_info(), precision, op_def.src_tensors[0],
            op_def.dst_tensors[0], external_weights, /*bias=*/nullptr,
            &dst_shape, /*src_exp=*/nullptr, runtime_check));
    conv = std::make_unique<FullyConnected>(std::move(conv_fc));
  }
  if (!weights_conversion) {
    src_ids.push_back(weights.weights);
    if (weights.scale) {
      src_ids.push_back(*weights.scale);
    }
    if (weights.zero_point) {
      src_ids.push_back(*weights.zero_point);
    }
  }
  src_ids.push_back(params);
  conv->flops_ = dst_shape.DimensionsProduct() * weights.shape.i * 2;
  builder.AddGpuOperation(src_ids, {dst}, std::move(conv),
                          "conv_packed_groups_" + ToString(weights.desc.type));
  return dst;
}

absl::StatusOr<GpuModelBuilder::TensorHandle> MakeConvWithBatchIds(
    GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& ids,
    const GpuModelBuilder::Weights& weights) {
  const auto precision =
      builder.GetConvPrecision(src.tensor_desc.GetDataType());

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

  BHWC dst_shape = src.tensor_desc.GetBHWCShape();
  dst_shape.c = weights.shape.o;
  dst_shape.h = ids.tensor_desc.GetBHWCShape().c;
  auto conv = builder.AddTensor(dst_shape, src.tensor_desc.GetDataType());

  ABSL_ASSIGN_OR_RETURN(auto operation,
                        CreateFullyConnectedWeightsBatchIds(
                            builder.gpu_info(), precision, src.tensor_desc,
                            ids.tensor_desc, conv.tensor_desc, external_weights,
                            /*bias=*/nullptr, &dst_shape));

  operation.flops_ = dst_shape.DimensionsProduct() * weights.shape.i * 2;

  const int num_active_experts = dst_shape.h;
  const int num_experts = weights.shape.h;
  if (num_active_experts < num_experts) {
    uint64_t partial_read_size =
        weights.weights.tensor_desc.GetMemorySizeInBytes();
    if (weights.scale) {
      partial_read_size += weights.scale->tensor_desc.GetMemorySizeInBytes();
    }
    if (weights.zero_point) {
      partial_read_size +=
          weights.zero_point->tensor_desc.GetMemorySizeInBytes();
    }
    operation.read_size_ = src.tensor_desc.GetMemorySizeInBytes() +
                           ids.tensor_desc.GetMemorySizeInBytes() +
                           partial_read_size / num_experts * num_active_experts;
  }
  std::vector<GpuModelBuilder::TensorHandle> src_ids = {src, ids,
                                                        weights.weights};
  if (weights.scale) {
    src_ids.push_back(*weights.scale);
  }
  if (weights.zero_point) {
    src_ids.push_back(*weights.zero_point);
  }
  builder.AddGpuOperation(
      src_ids, {conv}, std::make_unique<FullyConnected>(std::move(operation)),
      "fc1x1_" + ToString(weights.desc.type) + "_batch_ids");
  return conv;
}

absl::StatusOr<GpuModelBuilder::TensorHandle> ScaleWithBatchIds(
    GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& inputs,
    const GpuModelBuilder::TensorHandle& active_ids,
    const GpuModelBuilder::TensorHandle& active_scales,
    const GpuModelBuilder::TensorHandle& scales) {
  auto input_shape = inputs.tensor_desc.GetBHWCShape();
  auto dst_shape = BHWC(input_shape.b, 1, input_shape.h, input_shape.c);
  auto dst = builder.AddTensor(dst_shape, inputs.tensor_desc.GetDataType());

  // src [B, S, AE, D];
  // active ids [B, 1, S, AE];
  // active scales [B, 1, S, AE];
  // scales [E];
  // dst [B, 1, S, D];
  GPUOperation op;
  op.AddSrcTensor("inputs", inputs.tensor_desc);
  op.AddSrcTensor("active_ids", active_ids.tensor_desc);
  op.AddSrcTensor("active_scales", active_scales.tensor_desc);
  op.AddSrcTensor("scales", scales.tensor_desc);
  op.AddDstTensor("dst", dst.tensor_desc);
  const float epsilon =
      (active_scales.tensor_desc.GetDataType() == DataType::kFloat32) ? 1e-12f
                                                                      : 1e-3f;
  op.args_.AddFloat("epsilon", epsilon);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  const std::string read_scale =
      scales.tensor_desc.GetLayout() == Layout::kLinear
          ? "args.scales.ReadPerChannel<float>(scale_value, expert_id);"
          : "args.scales.ReadPerChannel<float>(scale_value, 0, 0, expert_id);";

  op.code_ = R"(
MAIN_FUNCTION($0) {
  int linear_id = ucl::GetGlobalId<0>();
  int x = linear_id / args.dst.Batch();
  int b = linear_id % args.dst.Batch();
  int y = ucl::GetGlobalId<1>();
  int s = ucl::GetGlobalId<2>();
  if (x >= args.dst.Width() || y >= args.dst.Height() || s >= args.dst.Slices()) return;
  args.inputs.SetBatchRef(b);
  args.active_ids.SetBatchRef(b);
  args.active_scales.SetBatchRef(b);
  args.dst.SetBatchRef(b);
  float4 sum = ucl::Init<float4>(0.0f);
  float sum_scales = 0.0f;
  for (int ae_id = 0; ae_id < args.inputs.Width(); ++ae_id) {
    int expert_id;
    args.active_ids.ReadPerChannel(expert_id, x, 0, ae_id);
    float scale_value;
    )" + read_scale +
             R"(
    float4 in_value = ucl::Convert<float4>(args.inputs.Read(ae_id, x, s));
    in_value *= scale_value;
    float active_scale;
    args.active_scales.ReadPerChannel<float>(active_scale, x, 0, ae_id);
    sum_scales += active_scale;
    sum += in_value * active_scale;
  }
  sum_scales = max(sum_scales, args.epsilon);
  sum /= sum_scales;
  args.dst::type dst_value = ucl::Convert<args.dst::type>(sum);
  args.dst.Write(dst_value, x, y, s);
}
)";
  builder.AddGpuOperation({inputs, active_ids, active_scales, scales}, {dst},
                          std::make_unique<GPUOperation>(std::move(op)),
                          "scale_with_batch_ids");
  return dst;
}

}  // namespace ml_drift
