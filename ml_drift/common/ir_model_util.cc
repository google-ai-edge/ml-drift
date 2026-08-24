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

#include "ml_drift/common/ir_model_util.h"

#include <algorithm>
#include <any>
#include <map>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/ir_model.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/selectors/operation_selector.h"
#include "ml_drift/common/selectors/special_selector.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift::ir {
namespace {

bool IsAssociativeLinkableOp(const IrOp& node,
                             const std::vector<const ir::IrTensor*>& inputs,
                             const std::vector<const ir::IrTensor*>& outputs) {
  const OperationType op_type = OperationTypeFromString(node.name);
  if (op_type != OperationType::ADD && op_type != OperationType::MUL) {
    return false;
  }
  if (inputs.size() == 1) {
    return false;
  }

  const auto dst_shape = outputs[0]->desc.GetBHWDCShape();
  for (int i = 0; i < inputs.size(); ++i) {
    const auto src_shape = inputs[i]->desc.GetBHWDCShape();
    if (dst_shape.b != src_shape.b && src_shape.b == 1) {
      return false;
    }
    if (dst_shape.h != src_shape.h && src_shape.h == 1) {
      return false;
    }
    if (dst_shape.w != src_shape.w && src_shape.w == 1) {
      return false;
    }
    if (dst_shape.d != src_shape.d && src_shape.d == 1) {
      return false;
    }
    if (dst_shape.c != src_shape.c && src_shape.c == 1) {
      return false;
    }
  }
  return true;
}

// Helper class for creating descriptors for appropriate tensors from
// IrModel.
// Also allows to create descriptors for new tensors(not present in
// IrModel)
class TensorReserver {
 public:
  TensorReserver() : next_(0) {}
  IrTensorId Add(const TensorDescriptor& dummy) {
    tensors_[next_] = dummy;
    return next_++;
  }
  void Add(IrTensorId id, const TensorDescriptor& dummy) {
    tensors_[id] = dummy;
  }
  void AddConstant(IrTensorId id, TensorDescriptor&& desc) {
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

absl::Status CheckExternalTensorDescription(const GpuInfo& gpu_info,
                                            const TensorDescriptor& tensor_desc,
                                            const BHWDC& shape) {
  if (tensor_desc.HasAxis(Axis::BATCH) && shape.b == 1) {
    return absl::InvalidArgumentError(
        "Batch size must be not 1 if tensor_desc.HasAxis(Axis::BATCH). "
        "Shape: " +
        ToString(shape) + ", layout: " + ToString(tensor_desc.GetLayout()));
  }
  if (!tensor_desc.HasAxis(Axis::BATCH) && shape.b != 1) {
    return absl::InvalidArgumentError(
        "Batch size must be 1 if !tensor_desc.HasAxis(Axis::BATCH). "
        "Shape: " +
        ToString(shape) + ", layout: " + ToString(tensor_desc.GetLayout()));
  }
  if (tensor_desc.GetLayout() == Layout::LINEAR &&
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

absl::StatusOr<TensorDescriptor> GetExternallyProvidedTensorDesc(
    const CreateGpuModelInfo& create_info, const GpuInfo& gpu_info,
    const IrModel& ir_model, IrTensor* tensor) {
  auto it_predefined = create_info.predefined.find(tensor->id);
  auto it_immutable_external =
      create_info.external_immutable_tensors.find(tensor->id);
  auto it_mutable_external =
      create_info.external_mutable_tensors.find(tensor->id);
  int external_categories_count = 0;
  TensorDescriptor tensor_desc;
  if (it_predefined != create_info.predefined.end()) {
    external_categories_count++;
    tensor_desc = it_predefined->second;
  }
  if (it_immutable_external != create_info.external_immutable_tensors.end()) {
    if (it_immutable_external->second == nullptr) {
      return absl::InternalError(absl::StrCat(
          "External immutable tensor is null for id: ", tensor->id));
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
    if (!(ir_model.IsGraphInput(tensor->id) ||
          ir_model.IsGraphOutput(tensor->id) ||
          ir_model.FindProducer(tensor->id) == nullptr)) {
      return absl::InvalidArgumentError(
          "Currently external can be used only for graph inputs/outputs");
    }
    ABSL_RETURN_IF_ERROR(CheckExternalTensorDescription(
        gpu_info, tensor_desc, tensor->desc.GetBHWDCShape()));
  }
  return tensor_desc;
}

TensorStorageType GetTensorStorageType(const CreateGpuModelInfo& create_info,
                                       const GpuInfo& gpu_info,
                                       const BHWDC& shape,
                                       const DataType& dtype,
                                       const Layout& layout) {
  TensorStorageType stype = create_info.storage_type;
  if (shape.c < 4 && !gpu_info.IsApiWebGpu()) {
    bool can_use_single_texture =
        gpu_info.IsApiMetal() ||
        stype == ::ml_drift::TensorStorageType::TEXTURE_2D ||
        stype == ::ml_drift::TensorStorageType::TEXTURE_3D ||
        stype == ::ml_drift::TensorStorageType::TEXTURE_ARRAY;
    can_use_single_texture &=
        ::ml_drift::TensorDescriptor{
            dtype, ::ml_drift::TensorStorageType::SINGLE_TEXTURE_2D, layout}
            .CanCreateTensorWithShape(gpu_info, shape)
            .ok();
    if (can_use_single_texture) {
      stype = ::ml_drift::TensorStorageType::SINGLE_TEXTURE_2D;
    }
  }
  return stype;
}

absl::StatusOr<TensorDescriptor> GetTensorDescForValue(
    const CreateGpuModelInfo& create_info, const GpuInfo& gpu_info,
    const IrModel& graph,
    const absl::flat_hash_set<ValueId>& cast_graph_outputs, IrTensor* tensor) {
  // Early exit if tensor is provided externally.
  if (create_info.predefined.contains(tensor->id) ||
      create_info.external_immutable_tensors.contains(tensor->id) ||
      create_info.external_mutable_tensors.contains(tensor->id)) {
    return GetExternallyProvidedTensorDesc(create_info, gpu_info, graph,
                                           tensor);
  }
  tensor->desc.SetStorageType(GetTensorStorageType(
      create_info, gpu_info, tensor->desc.GetBHWDCShape(),
      tensor->desc.GetDataType(), tensor->desc.GetLayout()));
  ABSL_CHECK_OK(tensor->desc.UpdateToSupportedStorageType(
      gpu_info, tensor->desc.GetBHWDCShape()))
      << "Failed to update tensor storage type.";
  if (gpu_info.IsApiMetal() &&
      tensor->desc.GetStorageType() ==
          ::ml_drift::TensorStorageType::TEXTURE_2D &&
      !gpu_info.apple_info.IsFamilyApple1()) {
    tensor->desc.SetUseBufferForWriteOnlyTexture2d(true);
  }
  // dtype fix for precision
  if (tensor->desc.GetDataType() == DataType::FLOAT32 &&
      create_info.precision != CalculationsPrecision::F32) {
    tensor->desc.SetDataType(DataType::FLOAT16);
  }
  return tensor->desc;
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
                                 const IrModel& ir_model,
                                 TensorReserver* tensor_reserver) {
  // The data type of graph outputs produced by the CAST operation has to be
  // preserved.
  absl::flat_hash_set<ValueId> cast_graph_outputs;
  for (IrTensorId output : ir_model.outputs()) {
    const IrOp* node = ir_model.FindProducer(output);
    if (node && OperationTypeFromString(node->name) == OperationType::CAST) {
      cast_graph_outputs.insert(output);
    }
  }
  IrTensorId max_id = 0;
  for (const auto& tensor : ir_model.tensors()) {
    if (tensor == nullptr) {
      continue;
    }
    // Checking if tensor is provided externally, otherwise return ir model td.
    ABSL_ASSIGN_OR_RETURN(
        auto tensor_desc,
        GetTensorDescForValue(create_info, gpu_info, ir_model,
                              cast_graph_outputs, tensor.get()));
    tensor_desc.SetBHWDCShape(tensor->desc.GetBHWDCShape());
    tensor_reserver->Add(tensor->id, tensor_desc);
    max_id = std::max(max_id, tensor->id);
  }
  tensor_reserver->SetNext(max_id + 1);
  auto& model_ops = ir_model.ops();
  // Upload constant tensors data.
  for (int i = 0; i < model_ops.size(); ++i) {
    if (model_ops[i] == nullptr) {
      continue;
    }
    const IrOp& node = *model_ops[i];
    auto op_type = OperationTypeFromString(node.name);
    if (op_type == OperationType::CONSTANT) {
      auto attr = std::any_cast<ConstTensorAttributes>(node.attr);
      std::vector<IrTensorId> outputs = node.outputs;
      auto tensor_desc = tensor_reserver->Get(outputs[0]);
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
      tensor_reserver->AddConstant(outputs[0], std::move(tensor_desc));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<OperationDef> GetOperationDef(
    const std::vector<IrTensorId>& inputs,
    const std::vector<IrTensorId>& outputs, GpuModelBuilder* model_builder) {
  OperationDef op_def;
  for (int j = 0; j < inputs.size(); ++j) {
    ABSL_ASSIGN_OR_RETURN(const auto& th, model_builder->GetTensor(inputs[j]));
    op_def.src_tensors.push_back(th.tensor_desc);
  }
  for (int j = 0; j < outputs.size(); ++j) {
    ABSL_ASSIGN_OR_RETURN(const auto& th, model_builder->GetTensor(outputs[j]));
    op_def.dst_tensors.push_back(th.tensor_desc);
  }
  return op_def;
}

class DefaultIrModelOpSelector : public IrModelOpSelector {
 public:
  DefaultIrModelOpSelector() = delete;
  DefaultIrModelOpSelector(const CreateGpuModelInfo& create_info,
                           const GpuInfo& gpu_info)
      : create_info_(create_info), gpu_info_(gpu_info) {}
  ~DefaultIrModelOpSelector() override = default;

  absl::Status GPUOperationFromNode(const OperationDef& op_def,
                                    const std::vector<const IrTensor*>& inputs,
                                    const std::vector<const IrTensor*>& outputs,
                                    const IrOp& node,
                                    GpuModelBuilder* model_builder) override {
    return ml_drift::GPUOperationFromNode(gpu_info_, op_def, create_info_,
                                          inputs, outputs, node, model_builder);
  }

  absl::Status GPUSubgraphFromIrModel(
      const ir::IrModel& ir_model, IrOpId first_op_id,
      const absl::flat_hash_set<IrOpId>& consumed_ops,
      absl::flat_hash_set<IrOpId>* new_consumed_ops,
      GpuModelBuilder* model_builder) override {
    return ml_drift::GPUSubgraphFromIrModel(create_info_.hints, gpu_info_,
                                            ir_model, first_op_id, consumed_ops,
                                            new_consumed_ops, model_builder);
  }

 private:
  const CreateGpuModelInfo& create_info_;
  const GpuInfo& gpu_info_;
};

absl::Status ConvertOperations(const IrModel& ir_model,
                               IrModelOpSelector& op_selector,
                               GpuModelBuilder* model_builder) {
  absl::flat_hash_set<IrOpId> consumed_nodes;
  auto& model_ops = ir_model.ops();
  std::map<IrTensorId, int>
      tensor_usages;  // keeps last index of operation that updated tensor
  for (const auto& input : ir_model.inputs()) {
    tensor_usages[input] = -1;  // so as inputs "updated" before operation
                                // 0, we will mark them with -1
  }
  for (int node_index = 0; node_index < model_ops.size(); ++node_index) {
    const IrOp* node = ir_model.op(node_index);
    if (node == nullptr) {
      continue;
    }
    if (consumed_nodes.find(node->id) != consumed_nodes.end()) {
      continue;
    }
    auto op_type = OperationTypeFromString(node->name);
    if (op_type == OperationType::CONSTANT) {
      // handled in ReserveGraphTensors
      continue;
    }
    // Mapping of subgraph (set of nodes) to GPU operations. Should happen
    // before straightforward mapping.
    absl::flat_hash_set<IrOpId> new_consumed_nodes;
    if (!op_selector
             .GPUSubgraphFromIrModel(ir_model, node->id, consumed_nodes,
                                     &new_consumed_nodes, model_builder)
             .ok()) {
      // Straightforward mapping of one graph node to GPU operations.
      auto input_ids = node->inputs;
      auto output_ids = node->outputs;
      std::vector<const IrTensor*> inputs;
      std::vector<const IrTensor*> outputs;
      inputs.reserve(input_ids.size());
      for (const auto& input_id : input_ids) {
        inputs.push_back(ir_model.tensor(input_id));
      }
      outputs.reserve(output_ids.size());
      for (const auto& output_id : output_ids) {
        outputs.push_back(ir_model.tensor(output_id));
      }
      // Reordering of input ids and updating of temporary tensors_usage
      // struct. To have better linking we need linking tensor(latest written
      // during linear execution) on first position.
      if (IsAssociativeLinkableOp(*node, inputs, outputs)) {
        int latest_written_tensor_index = 0;
        int last_usage = tensor_usages[inputs[0]->id];
        for (int j = 1; j < inputs.size(); ++j) {
          if (tensor_usages[inputs[j]->id] > last_usage) {
            last_usage = tensor_usages[inputs[j]->id];
            latest_written_tensor_index = j;
          }
        }
        std::swap(inputs[0], inputs[latest_written_tensor_index]);
        std::swap(input_ids[0], input_ids[latest_written_tensor_index]);
      }
      new_consumed_nodes = {node->id};
      ABSL_ASSIGN_OR_RETURN(
          const auto& op_def,
          GetOperationDef(input_ids, output_ids, model_builder));
      ABSL_RETURN_IF_ERROR(op_selector.GPUOperationFromNode(
          op_def, inputs, outputs, *node, model_builder));
    }
    for (const auto& consumed_node_id : new_consumed_nodes) {
      auto outputs = ir_model.op(consumed_node_id)->outputs;
      for (const auto& output : outputs) {
        tensor_usages[output] = node_index;
      }
    }
    consumed_nodes.insert(new_consumed_nodes.begin(), new_consumed_nodes.end());
  }

  return absl::OkStatus();
}

absl::Status IrModelToGpuModel(const IrModel& ir_model,
                               const CreateGpuModelInfo& create_info,
                               const GpuInfo& gpu_info,
                               IrModelOpSelector& op_selector,
                               std::shared_ptr<WeightsManager> weights_manager,
                               GpuModel* gpu_model) {
  if (!gpu_info.SupportsFP16() && gpu_info.IsApiOpenCl() &&
      (create_info.precision == CalculationsPrecision::F16 ||
       create_info.precision == CalculationsPrecision::F32_F16)) {
    return absl::InvalidArgumentError(
        "CalculationsPrecision::F16/F32_F16 is not supported on this GPU(no "
        "fp16 support).");
  }
  TensorReserver tensor_reserver;
  ABSL_RETURN_IF_ERROR(
      ReserveGraphTensors(create_info, gpu_info, ir_model, &tensor_reserver));
  GpuModelBuilderOptions options = {
      .hints = create_info.hints,
      .storage = create_info.storage_type,
      .use_f32_accum_for_f16_convolutions =
          create_info.precision == CalculationsPrecision::F32_F16,
  };
  GpuModelBuilder model_builder(
      gpu_info, options, std::move(tensor_reserver.tensors_),
      std::move(tensor_reserver.const_tensors_), weights_manager);
  ABSL_RETURN_IF_ERROR(
      ConvertOperations(ir_model, op_selector, &model_builder));
  std::vector<ValueId> input_ids(ir_model.inputs().begin(),
                                 ir_model.inputs().end());
  std::vector<ValueId> output_ids(ir_model.outputs().begin(),
                                  ir_model.outputs().end());
  return model_builder.GetGpuModel(input_ids, output_ids, gpu_model);
}
}  // namespace

absl::Status IrModelToGpuModel(const IrModel& ir_model,
                               const CreateGpuModelInfo& create_info,
                               const GpuInfo& gpu_info, GpuModel* gpu_model) {
  DefaultIrModelOpSelector default_op_selector(create_info, gpu_info);
  return IrModelToGpuModel(ir_model, create_info, gpu_info, default_op_selector,
                           /*weights_manager=*/nullptr, gpu_model);
}

absl::Status IrModelToGpuModel(const IrModel& ir_model,
                               const CreateGpuModelInfo& create_info,
                               const GpuInfo& gpu_info, GpuModel* gpu_model,
                               IrModelOpSelector* op_selector) {
  return IrModelToGpuModel(ir_model, create_info, gpu_info, *op_selector,
                           /*weights_manager=*/nullptr, gpu_model);
}

absl::Status IrModelToGpuModelWithWeightsConversion(
    const IrModel& ir_model, const CreateGpuModelInfo& create_info,
    const GpuInfo& gpu_info, GpuModel* gpu_model,
    GpuModel* gpu_weights_conversion_model,
    absl::flat_hash_map<ValueId, ValueId>* weights_mapping,
    std::vector<WeightsManager::UploadWeightsInfo>* upload_weights_info) {
  auto weights_manager = std::make_shared<WeightsManager>();
  DefaultIrModelOpSelector default_op_selector(create_info, gpu_info);
  ABSL_RETURN_IF_ERROR(IrModelToGpuModel(ir_model, create_info, gpu_info,
                                         default_op_selector, weights_manager,
                                         gpu_model));
  return weights_manager->CreateConversionGpuModel(
      gpu_info, gpu_weights_conversion_model, weights_mapping,
      upload_weights_info);
}

absl::Status IrModelToGpuModelWithWeightsConversion(
    const IrModel& ir_model, const CreateGpuModelInfo& create_info,
    const GpuInfo& gpu_info, GpuModel* gpu_model,
    GpuModel* gpu_weights_conversion_model,
    absl::flat_hash_map<ValueId, ValueId>* weights_mapping,
    std::vector<WeightsManager::UploadWeightsInfo>* upload_weights_info,
    IrModelOpSelector* op_selector) {
  auto weights_manager = std::make_shared<WeightsManager>();
  ABSL_RETURN_IF_ERROR(IrModelToGpuModel(ir_model, create_info, gpu_info,
                                         *op_selector, weights_manager,
                                         gpu_model));
  return weights_manager->CreateConversionGpuModel(
      gpu_info, gpu_weights_conversion_model, weights_mapping,
      upload_weights_info);
}

}  // namespace ml_drift::ir
