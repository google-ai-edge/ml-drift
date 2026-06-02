// Copyright 2024 The ML Drift Authors.
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

#include "ml_drift/common/gpu_model.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "flatbuffers/buffer.h"
#include "flatbuffers/flatbuffer_builder.h"
#include "flatbuffers/string.h"
#include "ml_drift/common/gpu_model_generated.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/serialization_base.h"
#include "ml_drift/common/task/serialization_base_generated.h"
#include "ml_drift/common/task/tensor_desc.h"

namespace ml_drift {
namespace {

flatbuffers::Offset<data::TensorDescWithId> Encode(
    const TensorDescriptor& desc, const ValueId& id,
    flatbuffers::FlatBufferBuilder* builder) {
  auto desc_fb = Encode(desc, builder);
  data::TensorDescWithIdBuilder desc_builder(*builder);
  desc_builder.add_desc(desc_fb);
  desc_builder.add_id(id);
  return desc_builder.Finish();
}

flatbuffers::Offset<data::GpuNode> Encode(
    const GpuNode& node, flatbuffers::FlatBufferBuilder* builder) {
  flatbuffers::Offset<data::GPUOperation> op_fb;
  flatbuffers::Offset<flatbuffers::String> subgraph_id_fb;
  if (node.gpu_operation) {
    op_fb = Encode(*node.gpu_operation, builder);
  } else if (!node.subgraph_id.empty()) {
    subgraph_id_fb = builder->CreateString(node.subgraph_id);
  } else {
    ABSL_LOG(ERROR) << "null gpu operation with empty subgraph id";
    return 0;
  }

  std::vector<int32_t> in_ids(node.inputs.size());
  for (int i = 0; i < in_ids.size(); ++i) {
    in_ids[i] = node.inputs[i];
  }
  std::vector<int32_t> out_ids(node.outputs.size());
  for (int i = 0; i < out_ids.size(); ++i) {
    out_ids[i] = node.outputs[i];
  }
  auto in_ids_fb = builder->CreateVector(in_ids);
  auto out_ids_fb = builder->CreateVector(out_ids);
  auto name_fb = builder->CreateString(node.name);
  data::GpuNodeBuilder node_builder(*builder);
  if (node.gpu_operation) {
    node_builder.add_gpu_op(op_fb);
  } else if (!node.subgraph_id.empty()) {
    node_builder.add_subgraph_id(subgraph_id_fb);
  }
  node_builder.add_input_ids(in_ids_fb);
  node_builder.add_output_ids(out_ids_fb);
  node_builder.add_name(name_fb);
  return node_builder.Finish();
}

absl::Status Decode(const data::GpuNode* fb_node, GpuNode* node) {
  GPUOperation op;
  if (fb_node->gpu_op()) {
    RETURN_IF_ERROR(Decode(fb_node->gpu_op(), &op));
    node->gpu_operation = std::make_unique<GPUOperation>(std::move(op));
  } else if (fb_node->subgraph_id()) {
    node->subgraph_id = std::string(fb_node->subgraph_id()->c_str(),
                                    fb_node->subgraph_id()->size());
  } else {
    return absl::InvalidArgumentError(
        "GpuNode has neither gpu_op nor subgraph_id");
  }
  for (auto in_fb : *fb_node->input_ids()) {
    node->inputs.push_back(in_fb);
  }
  for (auto out_fb : *fb_node->output_ids()) {
    node->outputs.push_back(out_fb);
  }
  node->name = std::string(fb_node->name()->c_str(), fb_node->name()->size());

  return absl::OkStatus();
}
}  // namespace

flatbuffers::Offset<data::GpuModel> Encode(
    const GpuModel& gpu_model, flatbuffers::FlatBufferBuilder* builder) {
  std::vector<int32_t> in_ids(gpu_model.input_ids_and_refs.size());
  std::vector<int64_t> in_refs(gpu_model.input_ids_and_refs.size());
  for (int i = 0; i < in_ids.size(); ++i) {
    in_ids[i] = gpu_model.input_ids_and_refs[i].first;
    in_refs[i] = gpu_model.input_ids_and_refs[i].second;
  }
  auto in_ids_fb = builder->CreateVector(in_ids);
  auto in_refs_fb = builder->CreateVector(in_refs);

  std::vector<int32_t> out_ids(gpu_model.output_ids_and_refs.size());
  std::vector<int64_t> out_refs(gpu_model.output_ids_and_refs.size());
  for (int i = 0; i < out_ids.size(); ++i) {
    out_ids[i] = gpu_model.output_ids_and_refs[i].first;
    out_refs[i] = gpu_model.output_ids_and_refs[i].second;
  }
  auto out_ids_fb = builder->CreateVector(out_ids);
  auto out_refs_fb = builder->CreateVector(out_refs);

  std::vector<flatbuffers::Offset<data::GpuNode>> nodes_fb;
  for (int i = 0; i < gpu_model.nodes.size(); ++i) {
    auto node_fb = Encode(gpu_model.nodes[i], builder);
    nodes_fb.push_back(node_fb);
  }
  auto nodes_fb_vec = builder->CreateVector(nodes_fb);

  std::vector<flatbuffers::Offset<data::TensorDescWithId>> tensors_fb;
  for (const auto& tensor : gpu_model.tensors) {
    auto tensor_fb = Encode(tensor.second, tensor.first, builder);
    tensors_fb.push_back(tensor_fb);
  }
  auto tensors_fb_vec = builder->CreateVector(tensors_fb);

  std::vector<flatbuffers::Offset<data::TensorDescWithId>> const_tensors_fb;
  for (const auto& tensor : gpu_model.const_tensors) {
    auto tensor_fb = Encode(tensor.second, tensor.first, builder);
    const_tensors_fb.push_back(tensor_fb);
  }
  auto const_tensors_fb_vec = builder->CreateVector(const_tensors_fb);

  // Encode subgraphs
  std::vector<flatbuffers::Offset<data::GpuModel>> subgraphs_fb;
  std::vector<flatbuffers::Offset<flatbuffers::String>> subgraph_ids_fb;
  for (const auto& [id, subgraph] : gpu_model.subgraphs) {
    auto subgraph_fb = Encode(subgraph, builder);
    subgraphs_fb.push_back(subgraph_fb);
    subgraph_ids_fb.push_back(builder->CreateString(id));
  }
  auto subgraphs_fb_vec = builder->CreateVector(subgraphs_fb);
  auto subgraph_ids_fb_vec = builder->CreateVector(subgraph_ids_fb);

  data::GpuModelBuilder gpu_model_builder(*builder);
  gpu_model_builder.add_nodes(nodes_fb_vec);
  gpu_model_builder.add_tensors(tensors_fb_vec);
  gpu_model_builder.add_const_tensors(const_tensors_fb_vec);
  gpu_model_builder.add_input_ids(in_ids_fb);
  gpu_model_builder.add_output_ids(out_ids_fb);
  gpu_model_builder.add_input_refs(in_refs_fb);
  gpu_model_builder.add_output_refs(out_refs_fb);
  gpu_model_builder.add_subgraphs(subgraphs_fb_vec);
  gpu_model_builder.add_subgraph_ids(subgraph_ids_fb_vec);
  return gpu_model_builder.Finish();
}

absl::Status Decode(const data::GpuModel* fb_gpu_model, GpuModel* gpu_model) {
  gpu_model->nodes.resize(fb_gpu_model->nodes()->size());
  int counter = 0;
  for (auto node_fb : *fb_gpu_model->nodes()) {
    RETURN_IF_ERROR(Decode(node_fb, &gpu_model->nodes[counter]));
    counter++;
  }

  for (const auto tensor_fb : *fb_gpu_model->tensors()) {
    TensorDescriptor desc;
    Decode(tensor_fb->desc(), &desc);
    gpu_model->tensors[tensor_fb->id()] = std::move(desc);
  }
  for (const auto tensor_fb : *fb_gpu_model->const_tensors()) {
    TensorDescriptor desc;
    Decode(tensor_fb->desc(), &desc);
    gpu_model->const_tensors[tensor_fb->id()] = std::move(desc);
  }
  for (int i = 0; i < fb_gpu_model->input_ids()->size(); ++i) {
    gpu_model->input_ids_and_refs.push_back(
        {(*fb_gpu_model->input_ids())[i], (*fb_gpu_model->input_refs())[i]});
  }
  for (int i = 0; i < fb_gpu_model->output_ids()->size(); ++i) {
    gpu_model->output_ids_and_refs.push_back(
        {(*fb_gpu_model->output_ids())[i], (*fb_gpu_model->output_refs())[i]});
  }
  if (fb_gpu_model->subgraphs() != nullptr &&
      fb_gpu_model->subgraph_ids() != nullptr) {
    if (fb_gpu_model->subgraphs()->size() !=
        fb_gpu_model->subgraph_ids()->size()) {
      return absl::DataLossError(absl::StrFormat(
          "subgraphs and subgraph_ids have different sizes. num subgraphs: %d, "
          "num subgraph_ids: %d",
          fb_gpu_model->subgraphs()->size(),
          fb_gpu_model->subgraph_ids()->size()));
    }
    for (int i = 0; i < fb_gpu_model->subgraphs()->size(); ++i) {
      const std::string subgraph_id =
          std::string((*fb_gpu_model->subgraph_ids())[i] -> c_str(),
                      (*fb_gpu_model->subgraph_ids())[i] -> size());
      RETURN_IF_ERROR(Decode((*fb_gpu_model->subgraphs())[i],
                             &gpu_model -> subgraphs[subgraph_id]));
      ;
    }
  }
  return absl::OkStatus();
}

}  // namespace ml_drift
