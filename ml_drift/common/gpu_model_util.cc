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

#include "ml_drift/common/gpu_model_util.h"

#include <algorithm>
#include <any>  // IWYU pragma: keep
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/selectors/operation_selector.h"
#include "ml_drift/common/selectors/special_selector.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {
namespace {

bool IsAssociativeLinkableOp(const Node& node,
                             const std::vector<Value*>& inputs,
                             const std::vector<Value*>& outputs) {
  if (inputs.size() == 1) {
    return false;
  }
  const OperationType op_type = OperationTypeFromString(node.operation.type);
  if (op_type != OperationType::kAdd && op_type != OperationType::kMul) {
    return false;
  }

  const auto dst_shape = outputs[0]->tensor.shape;
  for (int i = 0; i < inputs.size(); ++i) {
    const auto src_shape = inputs[i]->tensor.shape;
    if (dst_shape.b != src_shape.b && src_shape.b == 1) {
      return false;
    }
    if (dst_shape.h != src_shape.h && src_shape.h == 1) {
      return false;
    }
    if (dst_shape.w != src_shape.w && src_shape.w == 1) {
      return false;
    }
    if (dst_shape.c != src_shape.c && src_shape.c == 1) {
      return false;
    }
  }
  return true;
}

absl::Status CheckExternalTensorDescription(const GpuInfo& gpu_info,
                                            const TensorDescriptor& tensor_desc,
                                            const BHWC& shape) {
  if (tensor_desc.HasAxis(Axis::kDepth)) {
    return absl::InvalidArgumentError(
        "Currently no support of Depth dimension in predefined/external "
        "tensors.");
  }
  if (tensor_desc.HasAxis(Axis::kBatch) && shape.b == 1) {
    return absl::InvalidArgumentError(
        "Batch size must be not 1 if tensor_desc.HasAxis(Axis::BATCH). "
        "Shape: " +
        ToString(shape) + ", layout: " + ToString(tensor_desc.GetLayout()));
  }
  if (!tensor_desc.HasAxis(Axis::kBatch) && shape.b != 1) {
    return absl::InvalidArgumentError(
        "Batch size must be 1 if !tensor_desc.HasAxis(Axis::BATCH). "
        "Shape: " +
        ToString(shape) + ", layout: " + ToString(tensor_desc.GetLayout()));
  }
  if (tensor_desc.GetLayout() == Layout::kLinear &&
      ((shape.b > 1) + (shape.h > 1) + (shape.w > 1) + (shape.c > 1) > 1)) {
    return absl::InvalidArgumentError(
        "Linear layout can not have more than one non-linear dimension.");
  }
  if (auto s = tensor_desc.CanCreateTensorWithShape(gpu_info, shape); !s.ok()) {
    return absl::UnavailableError(absl::StrCat(
        "Current device can not allocate tensor with this shape for "
        "predefined/external descriptor: ",
        s.message()));
  }
  return absl::OkStatus();
}

// Helper class for creating descriptors for appropriate tensors from
// GraphFloat32
// Also allows to create descriptors for new tensors(not present in
// GraphFloat32)
class TensorReserver {
 public:
  TensorReserver() : next_(0) {}
  ValueId Add(const TensorDescriptor& dummy) {
    tensors_[next_] = dummy;
    return next_++;
  }
  void Add(ValueId id, const TensorDescriptor& dummy) { tensors_[id] = dummy; }
  void AddConstant(ValueId id, TensorDescriptor&& desc) {
    const_tensors_[id] = std::move(desc);
    tensors_.erase(id);
  }
  ValueId GetNewId() { return next_++; }
  void SetNext(ValueId id) { next_ = id; }
  TensorDescriptor Get(ValueId id) {
    if (auto it = tensors_.find(id); it != tensors_.end()) {
      return it->second;
    }
    TensorDescriptor result;
    const_tensors_[id].CopyWithoutData(&result);
    return result;
  }

 public:
  absl::flat_hash_map<ValueId, TensorDescriptor> tensors_;
  absl::flat_hash_map<ValueId, TensorDescriptor> const_tensors_;
  ValueId next_;
};

absl::StatusOr<TensorDescriptor> GetExternallyProvidedTensorDesc(
    const CreateGpuModelInfo& create_info, const GpuInfo& gpu_info,
    const GraphFloat32& graph, Value* value) {
  auto it_predefined = create_info.predefined.find(value->id);
  auto it_immutable_external =
      create_info.external_immutable_tensors.find(value->id);
  auto it_mutable_external =
      create_info.external_mutable_tensors.find(value->id);
  int external_categories_count = 0;
  TensorDescriptor tensor_desc;
  if (it_predefined != create_info.predefined.end()) {
    external_categories_count++;
    tensor_desc = it_predefined->second;
  }
  if (it_immutable_external != create_info.external_immutable_tensors.end()) {
    if (it_immutable_external->second == nullptr) {
      return absl::InternalError(absl::StrCat(
          "External immutable tensor is null for id: ", value->id));
    }
    external_categories_count++;
    tensor_desc = it_immutable_external->second->GetDescriptor();
  }
  if (it_mutable_external != create_info.external_mutable_tensors.end()) {
    external_categories_count++;
    tensor_desc = it_mutable_external->second;
  }
  if (external_categories_count > 1) {
    return absl::InvalidArgumentError(
        "Tensors ids from predefined / external_immutable_tensors / "
        "external_mutable_tensors should not intersect.");
  }
  if (external_categories_count == 1) {
    if (!(graph.IsGraphInput(value->id) || graph.IsGraphOutput(value->id))) {
      return absl::InvalidArgumentError(
          "Currently external can be used only for graph inputs/outputs");
    }
    ABSL_RETURN_IF_ERROR(CheckExternalTensorDescription(gpu_info, tensor_desc,
                                                        value->tensor.shape));
  }
  return tensor_desc;
}

absl::StatusOr<TensorDescriptor> GetTensorDescForValue(
    const CreateGpuModelInfo& create_info, const GpuInfo& gpu_info,
    const GraphFloat32& graph,
    const absl::flat_hash_set<ValueId>& cast_graph_outputs, Value* value) {
  // Early exit if tensor is provided externally.
  if (create_info.predefined.contains(value->id) ||
      create_info.external_immutable_tensors.contains(value->id) ||
      create_info.external_mutable_tensors.contains(value->id)) {
    return GetExternallyProvidedTensorDesc(create_info, gpu_info, graph, value);
  }

  // Initialize the tensor data type.
  DataType data_type = value->tensor.type;
  // If precision is F16 or F32_F16, use F16 for all tensors.
  // If precision is F32, use original tensor types which would enable mixed
  // precision.
  if (create_info.precision == CalculationsPrecision::kF32F16 ||
      create_info.precision == CalculationsPrecision::kF16) {
    if (value->tensor.type == DataType::kFloat32 &&
        !cast_graph_outputs.contains(value->id)) {
      data_type = DataType::kFloat16;
    }
  }

  // Initialize the tensor storage type.
  const auto shape = value->tensor.shape;
  Layout layout = shape.b == 1 ? Layout::kHWC : Layout::kBHWC;
  TensorStorageType storage_type = create_info.storage_type;
  // Attempt updating storage_type to SINGLE_TEXTURE_2D.
  if (shape.c < 4 && !gpu_info.IsApiWebGpu()) {
    bool can_use_single_texture =
        gpu_info.IsApiMetal() ||  //
        storage_type == TensorStorageType::kTexture2D ||
        storage_type == TensorStorageType::kTexture3D ||
        storage_type == TensorStorageType::kTextureArray;
    can_use_single_texture &=
        TensorDescriptor{data_type, TensorStorageType::kSingleTexture2D, layout}
            .CanCreateTensorWithShape(gpu_info, shape)
            .ok();
    if (can_use_single_texture) {
      storage_type = TensorStorageType::kSingleTexture2D;
    }
  }
  auto tensor_desc = TensorDescriptor{data_type, storage_type, layout};
  ABSL_RETURN_IF_ERROR(
      tensor_desc.UpdateToSupportedStorageType(gpu_info, shape));

  if (gpu_info.IsApiMetal() && storage_type == TensorStorageType::kTexture2D &&
      !gpu_info.apple_info.IsFamilyApple1()) {
    tensor_desc.SetUseBufferForWriteOnlyTexture2d(true);
  }
  return tensor_desc;
}

absl::Status CheckShapes(const TensorDescriptor& tensor_desc,
                         const BHWC& shape) {
  if (tensor_desc.GetBHWCShape() != shape) {
    return absl::InvalidArgumentError(
        "Shape mismatch: " + ToString(tensor_desc.GetBHWCShape()) + " vs " +
        ToString(shape));
  }
  return absl::OkStatus();
}

absl::Status ReserveGraphTensors(const CreateGpuModelInfo& create_info,
                                 const GpuInfo& gpu_info,
                                 const GraphFloat32& graph,
                                 TensorReserver* tensor_reserver) {
  // The data type of graph outputs produced by the CAST operation has to be
  // preserved.
  absl::flat_hash_set<ValueId> cast_graph_outputs;
  for (Value* output : graph.outputs()) {
    const auto node = graph.FindProducer(output->id);
    if (node &&
        OperationTypeFromString(node->operation.type) == OperationType::kCast) {
      cast_graph_outputs.insert(output->id);
    }
  }
  ValueId max_id = 0;
  for (Value* value : graph.values()) {
    ABSL_ASSIGN_OR_RETURN(auto tensor_desc,
                          GetTensorDescForValue(create_info, gpu_info, graph,
                                                cast_graph_outputs, value));
    tensor_desc.SetBHWCShape(value->tensor.shape);
    tensor_reserver->Add(value->id, tensor_desc);
    max_id = std::max(max_id, value->id);
  }
  tensor_reserver->SetNext(max_id + 1);
  std::vector<Node*> graph_nodes = graph.nodes();
  for (int i = 0; i < graph_nodes.size(); ++i) {
    const Node& node = *graph_nodes[i];
    auto op_type = OperationTypeFromString(node.operation.type);
    if (op_type == OperationType::kConstant) {
      auto attr =
          std::any_cast<ConstTensorAttributes>(node.operation.attributes);
      auto outputs = graph.FindOutputs(node.id);
      auto tensor_desc = tensor_reserver->Get(outputs[0]->id);
      if (auto tensor_f32 = std::get_if<TensorFloat32>(&attr.tensor)) {
        ABSL_RETURN_IF_ERROR(CheckShapes(tensor_desc, tensor_f32->shape));
        tensor_desc.UploadData(*tensor_f32);
      } else if (auto* tensor_float16 =
                     std::get_if<TensorFloat16>(&attr.tensor);
                 tensor_float16 != nullptr) {
        ABSL_RETURN_IF_ERROR(CheckShapes(tensor_desc, tensor_float16->shape));
        tensor_desc.UploadData(*tensor_float16);
      } else if (auto tensor_bool = std::get_if<TensorBool>(&attr.tensor)) {
        ABSL_RETURN_IF_ERROR(CheckShapes(tensor_desc, tensor_bool->shape));
        tensor_desc.UploadData(*tensor_bool);
      } else if (auto tensor_int32 = std::get_if<TensorInt32>(&attr.tensor)) {
        ABSL_RETURN_IF_ERROR(CheckShapes(tensor_desc, tensor_int32->shape));
        tensor_desc.UploadData(*tensor_int32);
      } else {
        return absl::InvalidArgumentError(
            "Constant node has unsupported tensor type");
      }
      tensor_reserver->AddConstant(outputs[0]->id, std::move(tensor_desc));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<OperationDef> GetOperationDef(const std::vector<Value*>& inputs,
                                             const std::vector<Value*>& outputs,
                                             GpuModelBuilder* model_builder) {
  OperationDef op_def;
  for (int j = 0; j < inputs.size(); ++j) {
    ABSL_ASSIGN_OR_RETURN(const auto& th,
                          model_builder->GetTensor(inputs[j]->id));
    op_def.src_tensors.push_back(th.tensor_desc);
  }
  for (int j = 0; j < outputs.size(); ++j) {
    ABSL_ASSIGN_OR_RETURN(const auto& th,
                          model_builder->GetTensor(outputs[j]->id));
    op_def.dst_tensors.push_back(th.tensor_desc);
  }
  return op_def;
}

absl::Status ConvertOperations(const GraphFloat32& graph,
                               OpSelector& op_selector,
                               GpuModelBuilder* model_builder) {
  std::set<NodeId> consumed_nodes;
  std::vector<Node*> graph_nodes = graph.nodes();
  std::map<ValueId, int>
      tensor_usages;  // keeps last index of operation that updated tensor
  for (const auto& input : graph.inputs()) {
    tensor_usages[input->id] = -1;  // so as inputs "updated" before operation
                                    // 0, we will mark them with -1
  }
  for (int node_index = 0; node_index < graph_nodes.size(); ++node_index) {
    const Node& node = *graph_nodes[node_index];
    if (consumed_nodes.find(node.id) != consumed_nodes.end()) {
      continue;
    }
    auto op_type = OperationTypeFromString(node.operation.type);
    if (op_type == OperationType::kConstant) {
      // handled in ReserveGraphTensors
      continue;
    }
    // Mapping of subgraph (set of nodes) to GPU operations. Should happen
    // before straightforward mapping.
    std::set<NodeId> new_consumed_nodes;
    if (!op_selector
             .GPUSubgraphFromGraph(graph, node.id, consumed_nodes,
                                   &new_consumed_nodes, model_builder)
             .ok()) {
      // Straightforward mapping of one graph node to GPU operations.
      auto inputs = graph.FindInputs(node.id);
      auto outputs = graph.FindOutputs(node.id);
      // Reordering of input ids and updating of temporary tensors_usage
      // struct. To have better linking we need linking tensor(latest written
      // during linear execution) on first position.
      if (IsAssociativeLinkableOp(node, inputs, outputs)) {
        int latest_written_tensor_index = 0;
        int last_usage = tensor_usages[inputs[0]->id];
        for (int j = 1; j < inputs.size(); ++j) {
          if (tensor_usages[inputs[j]->id] > last_usage) {
            last_usage = tensor_usages[inputs[j]->id];
            latest_written_tensor_index = j;
          }
        }
        std::swap(inputs[0], inputs[latest_written_tensor_index]);
      }
      new_consumed_nodes = {node.id};
      ABSL_ASSIGN_OR_RETURN(const auto& op_def,
                            GetOperationDef(inputs, outputs, model_builder));
      ABSL_RETURN_IF_ERROR(op_selector.GPUOperationFromNode(
          op_def, inputs, outputs, node, model_builder));
    }
    for (const auto& consumed_node_id : new_consumed_nodes) {
      auto outputs = graph.FindOutputs(consumed_node_id);
      for (const auto& output : outputs) {
        tensor_usages[output->id] = node_index;
      }
    }
    consumed_nodes.insert(new_consumed_nodes.begin(), new_consumed_nodes.end());
  }

  return absl::OkStatus();
}

class DefaultOpSelector : public OpSelector {
 public:
  DefaultOpSelector() = delete;
  DefaultOpSelector(const CreateGpuModelInfo& create_info,
                    const GpuInfo& gpu_info)
      : create_info_(create_info), gpu_info_(gpu_info) {}
  ~DefaultOpSelector() override = default;

  absl::Status GPUOperationFromNode(const OperationDef& op_def,
                                    const std::vector<Value*>& inputs,
                                    const std::vector<Value*>& outputs,
                                    const Node& node,
                                    GpuModelBuilder* model_builder) override {
    return ml_drift::GPUOperationFromNode(gpu_info_, op_def, create_info_,
                                          inputs, outputs, node, model_builder);
  }
  absl::Status GPUSubgraphFromGraph(const GraphFloat32& graph,
                                    NodeId first_node_id,
                                    const std::set<NodeId>& consumed_nodes,
                                    std::set<NodeId>* new_consumed_nodes,
                                    GpuModelBuilder* model_builder) override {
    return ml_drift::GPUSubgraphFromGraph(create_info_.hints, gpu_info_, graph,
                                          first_node_id, consumed_nodes,
                                          new_consumed_nodes, model_builder);
  }

 private:
  const CreateGpuModelInfo& create_info_;
  const GpuInfo& gpu_info_;
};
}  // namespace

absl::Status GraphToGpuModel(const GraphFloat32& graph,
                             const CreateGpuModelInfo& create_info,
                             const GpuInfo& gpu_info, OpSelector& op_selector,
                             std::shared_ptr<WeightsManager> weights_manager,
                             GpuModel* gpu_model) {
  if (!gpu_info.SupportsFP16() && gpu_info.IsApiOpenCl() &&
      (create_info.precision == CalculationsPrecision::kF16 ||
       create_info.precision == CalculationsPrecision::kF32F16)) {
    return absl::InvalidArgumentError(
        "CalculationsPrecision::F16/F32_F16 is not supported on this GPU(no "
        "fp16 support).");
  }
  TensorReserver tensor_reserver;
  ABSL_RETURN_IF_ERROR(
      ReserveGraphTensors(create_info, gpu_info, graph, &tensor_reserver));
  GpuModelBuilderOptions options = {
      .hints = create_info.hints,
      .storage = create_info.storage_type,
      .use_f32_accum_for_f16_convolutions =
          create_info.precision == CalculationsPrecision::kF32F16,
  };
  GpuModelBuilder model_builder(
      gpu_info, options, std::move(tensor_reserver.tensors_),
      std::move(tensor_reserver.const_tensors_), weights_manager);
  ABSL_RETURN_IF_ERROR(ConvertOperations(graph, op_selector, &model_builder));
  std::vector<std::pair<ValueId, ValueId>> input_ids_and_refs(
      graph.inputs().size());
  for (int i = 0; i < graph.inputs().size(); ++i) {
    const auto* value = graph.inputs()[i];
    input_ids_and_refs[i] = {value->id, value->tensor.ref};
  }
  std::vector<std::pair<ValueId, ValueId>> output_ids_and_refs(
      graph.outputs().size());
  for (int i = 0; i < graph.outputs().size(); ++i) {
    const auto* value = graph.outputs()[i];
    output_ids_and_refs[i] = {value->id, value->tensor.ref};
  }

  for (int i = 0; i < graph.variable_inputs().size(); ++i) {
    const auto* value = graph.variable_inputs()[i];
    input_ids_and_refs.push_back({value->id, value->tensor.ref});
    output_ids_and_refs.push_back({value->id, value->tensor.ref});
  }
  return model_builder.GetGpuModel(input_ids_and_refs, output_ids_and_refs,
                                   gpu_model);
}

absl::Status GraphToGpuModel(const GraphFloat32& graph,
                             const CreateGpuModelInfo& create_info,
                             const GpuInfo& gpu_info, GpuModel* gpu_model) {
  DefaultOpSelector default_op_selector(create_info, gpu_info);
  return GraphToGpuModel(graph, create_info, gpu_info, default_op_selector,
                         /*weights_manager=*/nullptr, gpu_model);
}

absl::Status GraphToGpuModel(const GraphFloat32& graph,
                             const CreateGpuModelInfo& create_info,
                             const GpuInfo& gpu_info, GpuModel* gpu_model,
                             OpSelector* op_selector) {
  return GraphToGpuModel(graph, create_info, gpu_info, *op_selector,
                         /*weights_manager=*/nullptr, gpu_model);
}

absl::Status GraphToGpuModelWithWeightsConversion(
    const GraphFloat32& graph, const CreateGpuModelInfo& create_info,
    const GpuInfo& gpu_info, GpuModel* gpu_model,
    GpuModel* gpu_weights_conversion_model,
    absl::flat_hash_map<ValueId, ValueId>* weights_mapping,
    std::vector<WeightsManager::UploadWeightsInfo>* upload_weights_info) {
  auto weights_manager = std::make_shared<WeightsManager>();
  DefaultOpSelector op_selector(create_info, gpu_info);
  ABSL_RETURN_IF_ERROR(GraphToGpuModel(
      graph, create_info, gpu_info, op_selector, weights_manager, gpu_model));
  return weights_manager->CreateConversionGpuModel(
      gpu_info, gpu_weights_conversion_model, weights_mapping,
      upload_weights_info);
}

absl::Status GraphToGpuModelWithWeightsConversion(
    const GraphFloat32& graph, const CreateGpuModelInfo& create_info,
    const GpuInfo& gpu_info, GpuModel* gpu_model,
    GpuModel* gpu_weights_conversion_model,
    absl::flat_hash_map<ValueId, ValueId>* weights_mapping,
    std::vector<WeightsManager::UploadWeightsInfo>* upload_weights_info,
    OpSelector* op_selector) {
  auto weights_manager = std::make_shared<WeightsManager>();
  ABSL_RETURN_IF_ERROR(GraphToGpuModel(
      graph, create_info, gpu_info, *op_selector, weights_manager, gpu_model));
  return weights_manager->CreateConversionGpuModel(
      gpu_info, gpu_weights_conversion_model, weights_mapping,
      upload_weights_info);
}

}  // namespace ml_drift
