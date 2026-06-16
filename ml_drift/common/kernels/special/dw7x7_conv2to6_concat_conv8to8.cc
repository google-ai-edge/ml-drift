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

#include "ml_drift/common/kernels/special/dw7x7_conv2to6_concat_conv8to8.h"

#include <any>
#include <cstdint>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/flops_util.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/ir_model.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/qcom_thin_filter_desc.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"

namespace ml_drift {
namespace {

bool IsConvKernelXis1(const Convolution2DAttributes& conv_attr) {
  return std::visit([](const auto& w) { return w.shape.w; },
                    conv_attr.weights) == 1 &&
         conv_attr.dilations.w == 1 && conv_attr.strides.w == 1 &&
         conv_attr.padding.prepended.w == 0 &&
         conv_attr.padding.appended.w == 0;
}
bool IsConvKernelYis1(const Convolution2DAttributes& conv_attr) {
  return std::visit([](const auto& w) { return w.shape.h; },
                    conv_attr.weights) == 1 == 1 &&
         conv_attr.dilations.h == 1 && conv_attr.strides.h == 1 &&
         conv_attr.padding.prepended.h == 0 &&
         conv_attr.padding.appended.h == 0;
}

bool IsConv1x1(const Convolution2DAttributes& conv_attr) {
  return IsConvKernelXis1(conv_attr) && IsConvKernelYis1(conv_attr);
}

std::string GenerateConvolutionCode(const OperationDef& op_def) {
  std::string c;
  c += R"(

#define CONV(R, SRC, i) \
  R += SRC.x * constants[i + 0]; \
  R += SRC.y * constants[i + 1]; \
  R += SRC.z * constants[i + 2]; \
  R += SRC.w * constants[i + 3];

MAIN_FUNCTION($0) {

  __constant Type* constants = args.weights.GetPtr();

  int X = get_global_id(0);
  int Y = get_global_id(1);

  if (X >= args.dst_0.Width() || Y >= args.dst_0.Height()) return;

  int2 int_coord = (int2)(X * 2 + 1, Y * 2 + 1);
  float2 float_coord = convert_float2(int_coord);

  Type c0, c1;

  SType2 dw = constants[0].xy;
  c0.xy = constants[0].zw;
  c0.zw = constants[1].xy;
  c1.xy = constants[1].zw;

  dw.x += qcom_convolve_imageh(args.src.GetHandle(), smp_zero, float_coord, args.f7x7_0.GetHandle()).x;
  dw.y += qcom_convolve_imageh(args.src.GetHandle(), smp_zero, float_coord, args.f7x7_1.GetHandle()).y;
  Type t0 = qcom_read_imageh_2x2(args.src.GetHandle(), smp_zero, float_coord, 0);
  Type t1 = qcom_read_imageh_2x2(args.src.GetHandle(), smp_zero, float_coord, 1);
  c0.x += dw.x * constants[2].x + dw.y * constants[2].y;
  c0.y += dw.x * constants[2].z + dw.y * constants[2].w;
  c0.z += dw.x * constants[3].x + dw.y * constants[3].y;
  c0.w += dw.x * constants[3].z + dw.y * constants[3].w;
  c1.x += dw.x * constants[4].x + dw.y * constants[4].y;
  c1.y += dw.x * constants[4].z + dw.y * constants[4].w;
  c0 = max(c0, ucl::Init<Type>(0.0)) + min(c0, ucl::Init<Type>(0.0)) * constants[5];
  c1.xy = max(c1.xy, ucl::Init<SType2>(0.0)) + min(c1.xy, ucl::Init<SType2>(0.0)) * constants[6].xy;

  c1.z = max(max(t0.x, t0.y), max(t0.z, t0.w));
  c1.w = max(max(t1.x, t1.y), max(t1.z, t1.w));

  args.dst_0.Write(c0, X, Y, 0);
  args.dst_0.Write(c1, X, Y, 1);

if (X < args.dst_0.Width() + 1) {
  t0 = constants[7];
  t1 = constants[8];

  CONV(t0, c0, 9);
  CONV(t1, c0, 13);
  CONV(t0, c1, 17);
  CONV(t1, c1, 21);

  t0 = max(t0, ucl::Init<Type>(0.0)) + min(t0, ucl::Init<Type>(0.0)) * constants[25];
  t1 = max(t1, ucl::Init<Type>(0.0)) + min(t1, ucl::Init<Type>(0.0)) * constants[26];

  args.dst_1.Write(t0, X, Y, 0);
  args.dst_1.Write(t1, X, Y, 1);
}
}
  )";
  const DataType type = op_def.src_tensors[0].GetDataType();
  absl::StrReplaceAll(
      {{"SType", ToUclDataType(type, 1)}, {"Type", ToUclDataType(type, 4)}},
      &c);
  return c;
}

void UploadWeights(const std::vector<float>& constants, DataType data_type,
                   GPUOperation* op) {
  BufferDescriptor desc;
  desc.element_type = data_type;
  desc.element_size = 4;
  desc.memory_type = MemoryType::CONSTANT;
  desc.size = SizeOf(data_type) * constants.size();
  desc.data.resize(desc.size);

  if (data_type == DataType::FLOAT32) {
    memcpy(desc.data.data(), constants.data(), desc.size);
  } else {
    half* gpu_data_half = reinterpret_cast<half*>(desc.data.data());
    for (int i = 0; i < constants.size(); ++i) {
      gpu_data_half[i] = constants[i];
    }
  }
  op->args_.AddObject("weights",
                      std::make_unique<BufferDescriptor>(std::move(desc)));
}

void CreateFilterDataFromDepthwiseWeights(
    const DepthwiseConvolution2DAttributes& dw_attr, int dst_group,
    int src_layer, std::vector<uint8_t>* result) {
  const auto& weights = GetFloatWeights(dw_attr);
  result->resize(weights.shape.h * weights.shape.w * sizeof(half));
  half* gpu_data = reinterpret_cast<half*>(result->data());
  for (int y = 0; y < weights.shape.h; ++y) {
    for (int x = 0; x < weights.shape.w; ++x) {
      const int f_index = y * weights.shape.w + x;
      const int index = weights.shape.LinearIndex({dst_group, y, x, src_layer});
      gpu_data[f_index] = weights.data[index];
    }
  }
}

}  // namespace

bool IsDW7x7Conv2To6ConcatConv8to8Supported(const GpuInfo& gpu_info) {
  return gpu_info.SupportsExtension("cl_qcom_accelerated_image_ops");
}

GPUOperation CreateDW7x7Conv2To6ConcatConv8to8(
    const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& dw_attr,
    const Convolution2DAttributes& conv2to6, const PReLUAttributes& prelu0,
    const Convolution2DAttributes& conv8to8, const PReLUAttributes& prelu1) {
  std::vector<float> constants;
  constants.reserve(2);
  for (int i = 0; i < 2; ++i) {
    constants.push_back(dw_attr.bias.data[i]);
  }

  for (int i = 0; i < 6; ++i) {
    constants.push_back(conv2to6.bias.data[i]);
  }
  const auto& conv2to6_weights_data = GetFloatWeights(conv2to6).data;
  for (int i = 0; i < 12; ++i) {
    constants.push_back(conv2to6_weights_data[i]);
  }

  auto alpha0 = std::get_if<Tensor<Linear, DataType::FLOAT32>>(&prelu0.alpha);
  for (int i = 0; i < 6; ++i) {
    constants.push_back(alpha0->data[i]);
  }
  constants.push_back(0.0);
  constants.push_back(0.0);

  for (int i = 0; i < 8; ++i) {
    constants.push_back(conv8to8.bias.data[i]);
  }

  const auto& conv8to8_weights = GetFloatWeights(conv8to8);
  for (int s = 0; s < 2; ++s) {
    for (int d = 0; d < 2; ++d) {
      float4 filters[4];
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          const int src_ch = s * 4 + j;
          const int dst_ch = d * 4 + i;
          const int f_index =
              conv8to8_weights.shape.LinearIndex({dst_ch, 0, 0, src_ch});
          filters[i][j] = conv8to8_weights.data[f_index];
        }
      }
      float4 filters_new[4];
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          filters_new[i][j] = filters[j][i];
        }
      }
      for (int i = 0; i < 4; ++i) {
        constants.push_back(filters_new[i].x);
        constants.push_back(filters_new[i].y);
        constants.push_back(filters_new[i].z);
        constants.push_back(filters_new[i].w);
      }
    }
  }

  auto alpha1 = std::get_if<Tensor<Linear, DataType::FLOAT32>>(&prelu1.alpha);
  for (int i = 0; i < 8; ++i) {
    constants.push_back(alpha1->data[i]);
  }

  GPUOperation result;
  result.AddSrcTensor("src", definition.src_tensors[0]);
  result.AddDstTensor("dst_0", definition.dst_tensors[0]);
  result.AddDstTensor("dst_1", definition.dst_tensors[1]);
  result.code_ = GenerateConvolutionCode(definition);
  result.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_ZIs1;

  const auto& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, dw_attr.weights);
  QcomThinFilterDescriptor f0_desc;
  f0_desc.kernel_size_x = weights_shape.w;
  f0_desc.kernel_size_y = weights_shape.h;
  CreateFilterDataFromDepthwiseWeights(dw_attr, 0, 0, &f0_desc.data);
  QcomThinFilterDescriptor f1_desc;
  f1_desc.kernel_size_x = weights_shape.w;
  f1_desc.kernel_size_y = weights_shape.h;
  CreateFilterDataFromDepthwiseWeights(dw_attr, 0, 1, &f1_desc.data);

  result.args_.AddObject(
      "f7x7_0", std::make_unique<QcomThinFilterDescriptor>(std::move(f0_desc)));
  result.args_.AddObject(
      "f7x7_1", std::make_unique<QcomThinFilterDescriptor>(std::move(f1_desc)));
  UploadWeights(constants, definition.src_tensors[0].GetDataType(), &result);
  return result;
}

absl::Status TryDW7x7Conv2To6ConcatConv8to8(
    const GpuInfo& gpu_info, const GraphFloat32& graph, NodeId first_node_id,
    const std::set<NodeId>& consumed_nodes,
    std::set<NodeId>* new_consumed_nodes, GpuModelBuilder* model_builder) {
  if (!gpu_info.SupportsExtension("cl_qcom_accelerated_image_ops")) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  auto* dw_node = graph.GetNode(first_node_id);
  if (dw_node == nullptr) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  if (OperationTypeFromString(dw_node->operation.type) !=
      OperationType::DEPTHWISE_CONVOLUTION) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }

  auto dw_inputs = graph.FindInputs(dw_node->id);
  if (dw_inputs.empty()) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }

  ASSIGN_OR_RETURN(auto dw_handle, model_builder->GetTensor(dw_inputs[0]->id));

  if (model_builder->GetConvPrecision(dw_handle.tensor_desc.GetDataType()) !=
      CalculationsPrecision::F16) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }

  DepthwiseConvolution2DAttributes* dw_attr =
      std::any_cast<DepthwiseConvolution2DAttributes>(
          &dw_node->operation.attributes);
  const auto& dw_weights_shape =
      std::visit([](const auto& w) { return w.shape; }, dw_attr->weights);
  const bool kGoodDwWeights =
      dw_weights_shape.w == 7 && dw_weights_shape.h == 7 &&
      dw_weights_shape.i == 2 && dw_weights_shape.o == 1;
  const bool kGoodDwDilation =
      dw_attr->dilations.w == 1 && dw_attr->dilations.h == 1;
  const bool kGoodDwPadding =
      dw_attr->padding.prepended.w == 2 && dw_attr->padding.prepended.h == 2;
  if (!kGoodDwWeights || !kGoodDwDilation || !kGoodDwPadding) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }

  if (dw_handle.tensor_desc.GetStorageType() !=
      TensorStorageType::SINGLE_TEXTURE_2D) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  auto dw_outputs = graph.FindOutputs(dw_node->id);
  auto consumers = graph.FindConsumers(dw_outputs[0]->id);
  if (consumers.size() != 1) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }

  auto* conv1_node = consumers[0];
  if (conv1_node == nullptr) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  if (consumed_nodes.find(conv1_node->id) != consumed_nodes.end()) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  if (OperationTypeFromString(conv1_node->operation.type) !=
      OperationType::CONVOLUTION_2D) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  Convolution2DAttributes* conv1_attr =
      std::any_cast<Convolution2DAttributes>(&conv1_node->operation.attributes);
  const auto& conv1_weights_shape =
      std::visit([](const auto& w) { return w.shape; }, conv1_attr->weights);
  if (!IsConv1x1(*conv1_attr) || conv1_weights_shape.i != 2 ||
      conv1_weights_shape.o != 6) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  auto conv1_outputs = graph.FindOutputs(conv1_node->id);
  consumers = graph.FindConsumers(conv1_outputs[0]->id);
  if (consumers.size() != 1) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }

  auto* prelu1_node = consumers[0];
  if (prelu1_node == nullptr) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  if (consumed_nodes.find(prelu1_node->id) != consumed_nodes.end()) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  if (OperationTypeFromString(prelu1_node->operation.type) !=
      OperationType::PRELU) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  auto prelu1_outputs = graph.FindOutputs(prelu1_node->id);
  consumers = graph.FindConsumers(prelu1_outputs[0]->id);
  if (consumers.size() != 1) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }

  auto* concat_node = consumers[0];
  if (concat_node == nullptr) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  if (consumed_nodes.find(concat_node->id) != consumed_nodes.end()) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  if (OperationTypeFromString(concat_node->operation.type) !=
      OperationType::CONCAT) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  auto concat_outputs = graph.FindOutputs(concat_node->id);
  consumers = graph.FindConsumers(concat_outputs[0]->id);
  if (consumers.size() != 2) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }

  auto concat_inputs = graph.FindInputs(concat_node->id);
  if (concat_inputs.size() != 2) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  auto* pooling_node = graph.FindProducer(concat_inputs[1]->id);
  if (pooling_node == nullptr) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  if (consumed_nodes.find(pooling_node->id) != consumed_nodes.end()) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  if (OperationTypeFromString(pooling_node->operation.type) !=
      OperationType::POOLING_2D) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  Pooling2DAttributes* pooling_attr =
      std::any_cast<Pooling2DAttributes>(&pooling_node->operation.attributes);
  if (pooling_attr->type != PoolingType::MAX || pooling_attr->output_indices ||
      pooling_attr->kernel.w != 2 || pooling_attr->kernel.h != 2 ||
      pooling_attr->strides.w != 2 || pooling_attr->strides.h != 2 ||
      pooling_attr->padding.prepended.w != 0 ||
      pooling_attr->padding.prepended.h != 0) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  auto pooling_inputs = graph.FindInputs(pooling_node->id);
  if (pooling_inputs[0] != dw_inputs[0]) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }

  auto* conv2_node = consumers[0];
  if (conv2_node == nullptr) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  if (consumed_nodes.find(conv2_node->id) != consumed_nodes.end()) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  if (OperationTypeFromString(conv2_node->operation.type) !=
      OperationType::CONVOLUTION_2D) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  Convolution2DAttributes* conv2_attr =
      std::any_cast<Convolution2DAttributes>(&conv2_node->operation.attributes);
  const auto& conv2_weights_shape =
      std::visit([](const auto& w) { return w.shape; }, conv2_attr->weights);
  if (!IsConv1x1(*conv2_attr) || conv2_weights_shape.i != 8 ||
      conv2_weights_shape.o != 8) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  auto conv2_outputs = graph.FindOutputs(conv2_node->id);
  consumers = graph.FindConsumers(conv2_outputs[0]->id);
  if (consumers.size() != 1) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }

  auto* prelu2_node = consumers[0];
  if (prelu2_node == nullptr) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  if (consumed_nodes.find(prelu2_node->id) != consumed_nodes.end()) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  if (OperationTypeFromString(prelu2_node->operation.type) !=
      OperationType::PRELU) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  auto prelu2_outputs = graph.FindOutputs(prelu2_node->id);
  consumers = graph.FindConsumers(prelu2_outputs[0]->id);
  if (consumers.size() != 1) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }

  ASSIGN_OR_RETURN(auto out0_handle,
                   model_builder->GetTensor(concat_outputs[0]->id));
  ASSIGN_OR_RETURN(auto out1_handle,
                   model_builder->GetTensor(prelu2_outputs[0]->id));
  OperationDef op_def;
  op_def.src_tensors.push_back(dw_handle.tensor_desc);
  op_def.dst_tensors.push_back(out0_handle.tensor_desc);
  op_def.dst_tensors.push_back(out1_handle.tensor_desc);

  PReLUAttributes* prelu1_attr =
      std::any_cast<PReLUAttributes>(&prelu1_node->operation.attributes);
  PReLUAttributes* prelu2_attr =
      std::any_cast<PReLUAttributes>(&prelu2_node->operation.attributes);

  GPUOperation op = CreateDW7x7Conv2To6ConcatConv8to8(
      op_def, *dw_attr, *conv1_attr, *prelu1_attr, *conv2_attr, *prelu2_attr);
  op.flops_ = GetDepthwiseConvolutionFlops(dw_outputs[0]->tensor.shape,
                                           dw_weights_shape);
  op.flops_ +=
      GetConvolutionFlops(conv1_outputs[0]->tensor.shape, conv1_weights_shape);
  op.flops_ +=
      GetConvolutionFlops(conv2_outputs[0]->tensor.shape, conv2_weights_shape);

  model_builder->AddGpuOperation(
      std::vector<ValueId>({dw_inputs[0]->id}),
      std::vector<ValueId>({concat_outputs[0]->id, prelu2_outputs[0]->id}),
      std::make_unique<GPUOperation>(std::move(op)),
      "dw7x7->conv1x1->pooling->conv1x1");

  new_consumed_nodes->insert(dw_node->id);
  new_consumed_nodes->insert(conv1_node->id);
  new_consumed_nodes->insert(prelu1_node->id);
  new_consumed_nodes->insert(concat_node->id);
  new_consumed_nodes->insert(pooling_node->id);
  new_consumed_nodes->insert(conv2_node->id);
  new_consumed_nodes->insert(prelu2_node->id);
  return absl::OkStatus();
}

absl::Status TryDW7x7Conv2To6ConcatConv8to8(
    const GpuInfo& gpu_info, const ir::IrModel& ir_model,
    ir::IrOpId first_op_id, const absl::flat_hash_set<ir::IrOpId>& consumed_ops,
    absl::flat_hash_set<ir::IrOpId>* new_consumed_ops,
    GpuModelBuilder* model_builder) {
  if (!gpu_info.SupportsExtension("cl_qcom_accelerated_image_ops")) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  auto* dw_op = ir_model.op(first_op_id);
  if (dw_op == nullptr) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  if (OperationTypeFromString(dw_op->name) !=
      OperationType::DEPTHWISE_CONVOLUTION) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }

  auto dw_inputs = dw_op->inputs;
  if (dw_inputs.empty()) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }

  ASSIGN_OR_RETURN(auto dw_handle, model_builder->GetTensor(dw_inputs[0]));

  if (model_builder->GetConvPrecision(dw_handle.tensor_desc.GetDataType()) !=
      CalculationsPrecision::F16) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }

  auto dw_attr = std::any_cast<DepthwiseConvolution2DAttributes>(dw_op->attr);
  const auto& dw_weights_shape =
      std::visit([](const auto& w) { return w.shape; }, dw_attr.weights);
  const bool kGoodDwWeights =
      dw_weights_shape.w == 7 && dw_weights_shape.h == 7 &&
      dw_weights_shape.i == 2 && dw_weights_shape.o == 1;
  const bool kGoodDwDilation =
      dw_attr.dilations.w == 1 && dw_attr.dilations.h == 1;
  const bool kGoodDwPadding =
      dw_attr.padding.prepended.w == 2 && dw_attr.padding.prepended.h == 2;
  if (!kGoodDwWeights || !kGoodDwDilation || !kGoodDwPadding) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }

  auto dw_handle_or_err = model_builder->GetTensor(dw_inputs[0]);
  if (!dw_handle_or_err.ok() ||
      dw_handle_or_err.value().tensor_desc.GetStorageType() !=
          TensorStorageType::SINGLE_TEXTURE_2D) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  auto dw_outputs = dw_op->outputs;
  auto consumers = ir_model.FindConsumers(dw_outputs[0]);
  if (consumers.size() != 1) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }

  auto* conv1_op = consumers[0];
  if (conv1_op == nullptr) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  if (consumed_ops.find(conv1_op->id) != consumed_ops.end()) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  if (OperationTypeFromString(conv1_op->name) !=
      OperationType::CONVOLUTION_2D) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  auto conv1_attr = std::any_cast<Convolution2DAttributes>(conv1_op->attr);
  const auto& conv1_weights_shape =
      std::visit([](const auto& w) { return w.shape; }, conv1_attr.weights);
  if (!IsConv1x1(conv1_attr) || conv1_weights_shape.i != 2 ||
      conv1_weights_shape.o != 6) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  auto conv1_outputs = conv1_op->outputs;
  consumers = ir_model.FindConsumers(conv1_outputs[0]);
  if (consumers.size() != 1) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }

  auto* prelu1_op = consumers[0];
  if (prelu1_op == nullptr) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  if (consumed_ops.find(prelu1_op->id) != consumed_ops.end()) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  if (OperationTypeFromString(prelu1_op->name) != OperationType::PRELU) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  auto prelu1_outputs = prelu1_op->outputs;
  consumers = ir_model.FindConsumers(prelu1_outputs[0]);
  if (consumers.size() != 1) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }

  auto* concat_op = consumers[0];
  if (concat_op == nullptr) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  if (consumed_ops.find(concat_op->id) != consumed_ops.end()) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  if (OperationTypeFromString(concat_op->name) != OperationType::CONCAT) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  auto concat_outputs = concat_op->outputs;
  consumers = ir_model.FindConsumers(concat_outputs[0]);
  if (consumers.size() != 2) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }

  auto concat_inputs = concat_op->inputs;
  if (concat_inputs.size() != 2) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  auto* pooling_op = ir_model.FindProducer(concat_inputs[1]);
  if (pooling_op == nullptr) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  if (consumed_ops.find(pooling_op->id) != consumed_ops.end()) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  if (OperationTypeFromString(pooling_op->name) != OperationType::POOLING_2D) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  auto pooling_attr = std::any_cast<Pooling2DAttributes>(pooling_op->attr);
  if (pooling_attr.type != PoolingType::MAX || pooling_attr.output_indices ||
      pooling_attr.kernel.w != 2 || pooling_attr.kernel.h != 2 ||
      pooling_attr.strides.w != 2 || pooling_attr.strides.h != 2 ||
      pooling_attr.padding.prepended.w != 0 ||
      pooling_attr.padding.prepended.h != 0) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  auto pooling_inputs = pooling_op->inputs;
  if (pooling_inputs[0] != dw_inputs[0]) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }

  auto* conv2_op = consumers[0];
  if (conv2_op == nullptr) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  if (consumed_ops.find(conv2_op->id) != consumed_ops.end()) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  if (OperationTypeFromString(conv2_op->name) !=
      OperationType::CONVOLUTION_2D) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  auto conv2_attr = std::any_cast<Convolution2DAttributes>(conv2_op->attr);
  const auto& conv2_weights_shape =
      std::visit([](const auto& w) { return w.shape; }, conv2_attr.weights);
  if (!IsConv1x1(conv2_attr) || conv2_weights_shape.i != 8 ||
      conv2_weights_shape.o != 8) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  auto conv2_outputs = conv2_op->outputs;
  auto conv2_consumers = ir_model.FindConsumers(conv2_outputs[0]);
  if (conv2_consumers.size() != 1) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }

  auto* prelu2_op = conv2_consumers[0];
  if (prelu2_op == nullptr) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  if (consumed_ops.find(prelu2_op->id) != consumed_ops.end()) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  if (OperationTypeFromString(prelu2_op->name) != OperationType::PRELU) {
    return absl::NotFoundError("DW7x7Conv2To6ConcatConv8to8 not suitable.");
  }
  auto prelu2_outputs = prelu2_op->outputs;

  ASSIGN_OR_RETURN(auto out0_handle,
                   model_builder->GetTensor(concat_outputs[0]));
  ASSIGN_OR_RETURN(auto out1_handle,
                   model_builder->GetTensor(prelu2_outputs[0]));
  OperationDef op_def;
  op_def.src_tensors.push_back(dw_handle.tensor_desc);
  op_def.dst_tensors.push_back(out0_handle.tensor_desc);
  op_def.dst_tensors.push_back(out1_handle.tensor_desc);

  auto prelu1_attr = std::any_cast<PReLUAttributes>(prelu1_op->attr);
  auto prelu2_attr = std::any_cast<PReLUAttributes>(prelu2_op->attr);

  GPUOperation op = CreateDW7x7Conv2To6ConcatConv8to8(
      op_def, dw_attr, conv1_attr, prelu1_attr, conv2_attr, prelu2_attr);
  op.flops_ = GetDepthwiseConvolutionFlops(
      ir_model.tensor(dw_outputs[0])->desc.GetBHWCShape(), dw_weights_shape);
  op.flops_ += GetConvolutionFlops(
      ir_model.tensor(conv1_outputs[0])->desc.GetBHWCShape(),
      conv1_weights_shape);
  op.flops_ += GetConvolutionFlops(
      ir_model.tensor(conv2_outputs[0])->desc.GetBHWCShape(),
      conv2_weights_shape);

  model_builder->AddGpuOperation(
      std::vector<GpuModelBuilder::ValueId>(
          {static_cast<GpuModelBuilder::ValueId>(dw_inputs[0])}),
      std::vector<GpuModelBuilder::ValueId>(
          {static_cast<GpuModelBuilder::ValueId>(concat_outputs[0]),
           static_cast<GpuModelBuilder::ValueId>(prelu2_outputs[0])}),
      std::make_unique<GPUOperation>(std::move(op)),
      "dw7x7->conv1x1->pooling->conv1x1");

  new_consumed_ops->insert(dw_op->id);
  new_consumed_ops->insert(conv1_op->id);
  new_consumed_ops->insert(prelu1_op->id);
  new_consumed_ops->insert(concat_op->id);
  new_consumed_ops->insert(pooling_op->id);
  new_consumed_ops->insert(conv2_op->id);
  new_consumed_ops->insert(prelu2_op->id);
  return absl::OkStatus();
}

}  // namespace ml_drift
