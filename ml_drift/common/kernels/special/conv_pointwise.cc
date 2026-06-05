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

#include "ml_drift/common/kernels/special/conv_pointwise.h"

#include <any>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace {
struct ConvPointwiseAttributes {
  // (Slice start width, slice start height)
  std::vector<int2> offsets;
  // True if we use mean as the reduce op, false to use reduce_sum.
  bool mean;
};

std::string GenerateCode(const ConvPointwiseAttributes& attr,
                         CalculationsPrecision precision) {
  std::string c = R"(
MAIN_FUNCTION($0) {
  int linear_id = ucl::GetGlobalId<0>();
  int X = linear_id / args.dst_tensor.Batch();
  int B = linear_id % args.dst_tensor.Batch();
  args.weights_tensor.SetBatchRef(B);
  args.src_tensor.SetBatchRef(B);
  args.dst_tensor.SetBatchRef(B);
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.dst_tensor.Width() ||
      Y >= args.dst_tensor.Height() ||
      S >= args.dst_tensor.Slices()) return;
  int4 offset0 = args.offsets.Read(S * 2 + 0, 0);
  int4 offset1 = args.offsets.Read(S * 2 + 1, 0);
  AccType res = ucl::Init<AccType>(0.0f);
  Type last_mask;
  int last_src_ch = (args.src_tensor.Slices() - 1) * 4;
  last_mask.x = ucl::Init<SType>(1.0f);
  last_mask.y = last_src_ch + 1 < args.src_tensor.Channels() ? ucl::Init<SType>(1.0f) : ucl::Init<SType>(0.0f);
  last_mask.z = last_src_ch + 2 < args.src_tensor.Channels() ? ucl::Init<SType>(1.0f) : ucl::Init<SType>(0.0f);
  last_mask.w = last_src_ch + 3 < args.src_tensor.Channels() ? ucl::Init<SType>(1.0f) : ucl::Init<SType>(0.0f);
  for (int s = 0; s < args.src_tensor.Slices(); ++s) {
    Type src = args.src_tensor.Read(X, Y, s);
    Type w0 = args.weights_tensor.Read(X + offset0.x, Y + offset0.y, s);
    Type w1 = args.weights_tensor.Read(X + offset0.z, Y + offset0.w, s);
    Type w2 = args.weights_tensor.Read(X + offset1.x, Y + offset1.y, s);
    Type w3 = args.weights_tensor.Read(X + offset1.z, Y + offset1.w, s);
    Type mask = ucl::Init<Type>(1.0f);
    if (s == (args.src_tensor.Slices() - 1)) {
      mask = last_mask;
    }
    src *= mask;
    res.x += dot(src, w0);
    res.y += dot(src, w1);
    res.z += dot(src, w2);
    res.w += dot(src, w3);
  }
  Type result = ucl::Convert<Type>(res);
)";
  if (attr.mean) {
    c += "  result /= ucl::Convert<SType>(args.src_tensor.Channels());\n";
  }
  c += "  args.dst_tensor.Write(result, X, Y, S);\n";
  c += "}\n";
  DataType acc_type = precision == CalculationsPrecision::F16
                          ? DataType::FLOAT16
                          : DataType::FLOAT32;
  DataType type = DeduceDataTypeFromPrecision(precision);
  absl::StrReplaceAll({{"SType", ToUclDataType(type, 1)},
                       {"AccType", ToUclDataType(acc_type, 4)},
                       {"Type", ToUclDataType(type, 4)}},
                      &c);
  return c;
}

struct NodeContext {
  Node* node;
  std::vector<Value*> inputs;
  std::vector<Value*> outputs;
};

absl::Status IsNode(const GraphFloat32& graph, OperationType op_type,
                    int inputs_count, int outputs_count, Node* node,
                    NodeContext* node_context) {
  const std::string op_desc = ToString(op_type);
  node_context->node = node;
  if (node_context->node == nullptr) {
    return absl::NotFoundError(absl::StrCat("Invalid ", op_desc, " node."));
  }
  if (OperationTypeFromString(node_context->node->operation.type) != op_type) {
    return absl::InternalError(
        absl::StrCat("Not correct node type. Expected ", op_desc, ", received ",
                     node_context->node->operation.type));
  }
  node_context->inputs = graph.FindInputs(node_context->node->id);
  node_context->outputs = graph.FindOutputs(node_context->node->id);
  if (inputs_count != -1) {
    if (node_context->inputs.size() != inputs_count) {
      return absl::InternalError(
          absl::StrCat("Expected ", inputs_count, " input in a ", op_desc,
                       " node. Node has ", node_context->inputs.size()));
    }
  }
  if (node_context->outputs.size() != outputs_count) {
    return absl::InternalError(
        absl::StrCat("Expected ", outputs_count, " output in a ", op_desc,
                     " node. Node has ", node_context->outputs.size()));
  }
  return absl::OkStatus();
}

absl::Status IsMeanNode(const GraphFloat32& graph, Node* node,
                        NodeContext* node_context) {
  RETURN_IF_ERROR(IsNode(graph, OperationType::MEAN, 1, 1, node, node_context));
  auto mean_attr =
      std::any_cast<ReduceAttributes>(node_context->node->operation.attributes);
  if (mean_attr.dims != std::set<Axis>{Axis::CHANNELS}) {
    return absl::InternalError("Expected mean node with channels reduction.");
  }
  return absl::OkStatus();
}

absl::Status IsReduceSumNode(const GraphFloat32& graph, Node* node,
                             NodeContext* node_context) {
  RETURN_IF_ERROR(
      IsNode(graph, OperationType::REDUCE_SUM, 1, 1, node, node_context));
  auto reduce_attr =
      std::any_cast<ReduceAttributes>(node_context->node->operation.attributes);
  if (reduce_attr.dims != std::set<Axis>{Axis::CHANNELS}) {
    return absl::InternalError(
        "Expected reduce_sum node with channels reduction.");
  }
  return absl::OkStatus();
}

absl::Status IsMulNode(const GraphFloat32& graph, Node* node,
                       NodeContext* node_context) {
  RETURN_IF_ERROR(IsNode(graph, OperationType::MUL, 2, 1, node, node_context));
  if (node_context->inputs[0]->tensor.shape !=
      node_context->inputs[1]->tensor.shape) {
    return absl::InternalError("Expected mul node with 2 equal tensors.");
  }
  return absl::OkStatus();
}

absl::Status IsSliceNode(const GraphFloat32& graph, Node* node,
                         NodeContext* node_context) {
  RETURN_IF_ERROR(
      IsNode(graph, OperationType::SLICE, 1, 1, node, node_context));
  auto slice_attr =
      std::any_cast<SliceAttributes>(node_context->node->operation.attributes);
  if (slice_attr.strides != BHWC(1, 1, 1, 1)) {
    return absl::InternalError("Not valid attributes in slice node.");
  }
  return absl::OkStatus();
}

absl::Status IsConcatNode(const GraphFloat32& graph, Node* node,
                          NodeContext* node_context) {
  RETURN_IF_ERROR(
      IsNode(graph, OperationType::CONCAT, -1, 1, node, node_context));
  auto concat_attr =
      std::any_cast<ConcatAttributes>(node_context->node->operation.attributes);
  if (concat_attr.axis != Axis::CHANNELS) {
    return absl::InternalError("Not valid attributes in concat node.");
  }
  return absl::OkStatus();
}

absl::Status GetOffset(const GraphFloat32& graph, NodeId concat_input_node,
                       NodeId second_commom_input_id, int* offset_x,
                       int* offset_y, std::set<NodeId>* consumed_nodes) {
  NodeContext reduce_node, mul_node, slice_node;
  absl::Status status =
      IsMeanNode(graph, graph.FindProducer(concat_input_node), &reduce_node);
  if (!status.ok()) {
    RETURN_IF_ERROR(IsReduceSumNode(
        graph, graph.FindProducer(concat_input_node), &reduce_node));
  }
  RETURN_IF_ERROR(IsMulNode(
      graph, graph.FindProducer(reduce_node.inputs[0]->id), &mul_node));
  const ValueId slice_output_id =
      mul_node.inputs[0]->id == second_commom_input_id ? mul_node.inputs[1]->id
                                                       : mul_node.inputs[0]->id;
  RETURN_IF_ERROR(
      IsSliceNode(graph, graph.FindProducer(slice_output_id), &slice_node));
  auto slice_attr =
      std::any_cast<SliceAttributes>(slice_node.node->operation.attributes);
  *offset_x = slice_attr.starts.w;
  *offset_y = slice_attr.starts.h;
  consumed_nodes->insert(reduce_node.node->id);
  consumed_nodes->insert(mul_node.node->id);
  consumed_nodes->insert(slice_node.node->id);
  return absl::OkStatus();
}

}  // namespace

GPUOperation CreateConvPointwise(const OperationDef& definition,
                                 CalculationsPrecision precision,
                                 const ConvPointwiseAttributes& attr) {
  const int dst_channels = attr.offsets.size();
  const int dst_depth = DivideRoundUp(dst_channels, 4);
  std::vector<int32_t> offsets_data(dst_depth * 2 * 4, 0);
  for (int i = 0; i < attr.offsets.size(); ++i) {
    offsets_data[i * 2 + 0] = attr.offsets[i].x;
    offsets_data[i * 2 + 1] = attr.offsets[i].y;
  }
  for (int i = attr.offsets.size(); i < offsets_data.size() / 2; ++i) {
    offsets_data[i * 2 + 0] = attr.offsets.back().x;
    offsets_data[i * 2 + 1] = attr.offsets.back().y;
  }

  GPUOperation op;
  op.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  op.AddSrcTensor("weights_tensor", definition.src_tensors[1]);
  op.AddDstTensor("dst_tensor", definition.dst_tensors[0]);
  op.code_ = GenerateCode(attr, precision);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  TensorDescriptor desc = CreateConstantHWVec4TensorDescriptor(
      DataType::INT32, TensorStorageType::TEXTURE_2D, dst_depth * 2, 1,
      reinterpret_cast<uint8_t*>(offsets_data.data()));
  op.args_.AddObject("offsets", std::make_unique<TensorDescriptor>(desc));
  return op;
}

absl::Status TryFusedPointwiseConv(const GraphFloat32& graph,
                                   NodeId first_node_id,
                                   const std::set<NodeId>& consumed_nodes,
                                   std::set<NodeId>* new_consumed_nodes,
                                   GpuModelBuilder* model_builder) {
  NodeContext slice_node;
  RETURN_IF_ERROR(
      IsSliceNode(graph, graph.GetNode(first_node_id), &slice_node));
  const auto& first_commom_input = slice_node.inputs[0];
  auto slice_consumers = graph.FindConsumers(slice_node.outputs[0]->id);
  if (slice_consumers.size() != 1) {
    return absl::NotFoundError("FusedPointwiseConv not suitable.");
  }
  NodeContext mul_node;
  RETURN_IF_ERROR(IsMulNode(graph, slice_consumers[0], &mul_node));
  const auto& second_commom_input =
      mul_node.inputs[0]->id == slice_node.outputs[0]->id ? mul_node.inputs[1]
                                                          : mul_node.inputs[0];
  auto mul_consumers = graph.FindConsumers(mul_node.outputs[0]->id);
  if (mul_consumers.size() != 1) {
    return absl::NotFoundError("FusedPointwiseConv not suitable.");
  }
  NodeContext reduce_node;
  bool mean = true;
  absl::Status status = IsMeanNode(graph, mul_consumers[0], &reduce_node);
  if (!status.ok()) {
    RETURN_IF_ERROR(IsReduceSumNode(graph, mul_consumers[0], &reduce_node));
    mean = false;
  }
  auto reduce_consumers = graph.FindConsumers(reduce_node.outputs[0]->id);
  if (reduce_consumers.size() != 1) {
    return absl::NotFoundError("FusedPointwiseConv not suitable.");
  }
  NodeContext concat_node;
  RETURN_IF_ERROR(IsConcatNode(graph, reduce_consumers[0], &concat_node));
  ConvPointwiseAttributes op_attr;
  op_attr.mean = mean;
  std::set<NodeId> temp_consumed_nodes;
  for (const auto& concat_input : concat_node.inputs) {
    int offset_x, offset_y;
    RETURN_IF_ERROR(GetOffset(graph, concat_input->id, second_commom_input->id,
                              &offset_x, &offset_y, &temp_consumed_nodes));
    op_attr.offsets.push_back(int2(offset_x, offset_y));
  }

  ASSIGN_OR_RETURN(auto src0_handle,
                   model_builder->GetTensor(second_commom_input->id));
  ASSIGN_OR_RETURN(auto src1_handle,
                   model_builder->GetTensor(first_commom_input->id));
  ASSIGN_OR_RETURN(auto dst_handle,
                   model_builder->GetTensor(concat_node.outputs[0]->id));
  OperationDef op_def;
  op_def.src_tensors.push_back(src0_handle.tensor_desc);
  op_def.src_tensors.push_back(src1_handle.tensor_desc);
  op_def.dst_tensors.push_back(dst_handle.tensor_desc);
  model_builder->AddGpuOperation(
      std::vector<ValueId>({second_commom_input->id, first_commom_input->id}),
      std::vector<ValueId>({concat_node.outputs[0]->id}),
      std::make_unique<GPUOperation>(
          CreateConvPointwise(op_def,
                              model_builder->GetConvPrecision(
                                  src0_handle.tensor_desc.GetDataType()),
                              op_attr)),
      "slice_mul_reduce_concat");
  new_consumed_nodes->insert(temp_consumed_nodes.begin(),
                             temp_consumed_nodes.end());
  new_consumed_nodes->insert(concat_node.node->id);
  return absl::OkStatus();
}

}  // namespace ml_drift
