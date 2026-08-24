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

#include "ml_drift/common/merge_nodes.h"

#include <algorithm>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

namespace {

bool IsReady(const absl::flat_hash_set<ValueId>& ready_tensors,
             const GpuNode& node) {
  for (const ValueId in_id : node.inputs) {
    if (ready_tensors.find(in_id) == ready_tensors.end()) {
      return false;
    }
  }
  return true;
}

absl::Status MergeGpuNodes(const GpuInfo& gpu_info, GpuNode* src,
                           GpuNode* dst) {
  for (int j = 1; j < src->inputs.size(); ++j) {
    dst->inputs.push_back(src->inputs[j]);
  }
  dst->outputs[0] = src->outputs[0];
  dst->name += " -> " + src->name;
  return dst->gpu_operation->AddOperation(gpu_info, src->gpu_operation.get());
}

bool HasId(const std::vector<std::pair<ValueId, ValueId>>& output_ids_and_refs,
           ValueId id) {
  for (const auto& [out_id, ref] : output_ids_and_refs) {
    if (out_id == id) {
      return true;
    }
  }
  return false;
}

absl::Status MergeElementwiseNodes(const GpuInfo& gpu_info,
                                   GpuModel* gpu_model) {
  auto& nodes = gpu_model->nodes;
  for (int elem_root_index = 1; elem_root_index < nodes.size();
       ++elem_root_index) {
    auto& elem_root = nodes[elem_root_index];
    if (!(elem_root.inputs.size() == 1 || elem_root.inputs.size() == 2) ||
        elem_root.outputs.size() != 1 ||
        !elem_root.gpu_operation->IsLinkable()) {
      continue;
    }
    // key is elem_root input index, value is node index
    std::map<int, int> prev_nodes;
    for (int j = elem_root_index - 1; j >= 0; --j) {
      for (int k = 0; k < elem_root.inputs.size(); ++k) {
        if (elem_root.inputs[k] == nodes[j].outputs[0]) {
          prev_nodes[k] = j;
          break;
        }
      }
    }
    // TYPE_0
    //    input       input
    //      |           |
    //    elem0         |
    //      |    -->  elem
    //  elem_root       |
    //      |           |
    //    output      output
    if (prev_nodes.size() == 1) {
      if (elem_root.inputs.size() != 1) {
        continue;
      }
      const int prev_first_node_index = prev_nodes[0];
      auto& prev_node = nodes[prev_first_node_index];
      if (prev_node.inputs.size() != 1 || prev_node.outputs.size() != 1 ||
          !prev_node.gpu_operation->IsLinkable() ||
          prev_node.optional_tag != elem_root.optional_tag ||
          HasId(gpu_model->output_ids_and_refs, prev_node.outputs[0])) {
        continue;
      }
      int consumers_count = 0;
      for (const auto& node : nodes) {
        for (const auto& input : node.inputs) {
          if (input == elem_root.inputs[0]) {
            consumers_count++;
          }
        }
      }
      if (consumers_count != 1) {
        continue;
      }
      GPUOperation new_operation;
      ABSL_RETURN_IF_ERROR(FuseSimpleElemWithSimpleElem(
          gpu_info, std::move(*prev_node.gpu_operation.get()),
          std::move(*elem_root.gpu_operation.get()), &new_operation));

      GpuNode new_node;
      new_node.inputs.push_back(prev_node.inputs[0]);
      new_node.outputs.push_back(elem_root.outputs[0]);
      new_node.name = prev_node.name + " -> " + elem_root.name;
      new_node.gpu_operation =
          std::make_unique<GPUOperation>(std::move(new_operation));
      new_node.optional_tag = elem_root.optional_tag;

      nodes.erase(nodes.begin() + elem_root_index);
      nodes[prev_first_node_index] = std::move(new_node);
      elem_root_index = prev_first_node_index;
      continue;
    }

    // check TYPE_1/2/3
    if (prev_nodes.size() == 2) {
      if (elem_root.inputs.size() != 2 ||
          elem_root.gpu_operation->GetElementwiseInputsCount() != 2) {
        continue;
      }
      const int prev_first_node_index = prev_nodes[0];
      const int prev_second_node_index = prev_nodes[1];
      auto& prev_first_node = nodes[prev_first_node_index];
      auto& prev_second_node = nodes[prev_second_node_index];

      // check TYPE_1
      // TYPE_1
      //      input           input
      //     /    \             |
      //   elem0   |            |
      //     \    /      -->  elem
      //   elem_root            |
      //       |                |
      //     output           output
      if (prev_first_node.gpu_operation->IsLinkable() &&
          !prev_second_node.gpu_operation->IsLinkable() &&
          prev_second_node.outputs.size() == 1 &&
          prev_first_node.inputs.size() == 1 &&
          prev_first_node.outputs.size() == 1 &&
          prev_first_node.optional_tag == elem_root.optional_tag &&
          !HasId(gpu_model->output_ids_and_refs, prev_first_node.outputs[0])) {
        int first_node_parent_index = -1;
        for (int j = prev_first_node_index - 1; j >= 0; --j) {
          if (nodes[j].outputs[0] == prev_first_node.inputs[0]) {
            first_node_parent_index = j;
            break;
          }
        }
        if (first_node_parent_index == -1 ||
            first_node_parent_index != prev_second_node_index) {
          continue;
        }
        int consumers_count = 0;
        for (const auto& node : nodes) {
          for (const auto& input : node.inputs) {
            if (input == elem_root.inputs[0]) {
              consumers_count++;
            }
          }
        }
        if (consumers_count != 1) {
          continue;
        }

        GPUOperation new_operation;
        ABSL_RETURN_IF_ERROR(Fuse2InputElemWithSimpleElemAsFirstInput(
            gpu_info, std::move(*prev_first_node.gpu_operation.get()),
            std::move(*elem_root.gpu_operation.get()), &new_operation));

        GpuNode new_node;
        new_node.inputs.push_back(prev_first_node.inputs[0]);
        new_node.outputs.push_back(elem_root.outputs[0]);
        new_node.name = prev_first_node.name + " -> " + elem_root.name;
        new_node.gpu_operation =
            std::make_unique<GPUOperation>(std::move(new_operation));
        new_node.optional_tag = elem_root.optional_tag;

        nodes.erase(nodes.begin() + elem_root_index);
        nodes[prev_first_node_index] = std::move(new_node);
        elem_root_index = prev_first_node_index;
        continue;
      }

      // check TYPE_2
      // TYPE_2
      //      input           input
      //     /    \             |
      //    |    elem0          |
      //     \    /      -->  elem
      //   elem_root            |
      //       |                |
      //     output           output
      if (!prev_first_node.gpu_operation->IsLinkable() &&
          prev_second_node.gpu_operation->IsLinkable() &&
          prev_first_node.outputs.size() == 1 &&
          prev_second_node.inputs.size() == 1 &&
          prev_second_node.outputs.size() == 1 &&
          prev_second_node.optional_tag == elem_root.optional_tag &&
          !HasId(gpu_model->output_ids_and_refs, prev_second_node.outputs[0])) {
        int second_node_parent_index = -1;
        for (int j = prev_second_node_index - 1; j >= 0; --j) {
          if (nodes[j].outputs[0] == prev_second_node.inputs[0]) {
            second_node_parent_index = j;
            break;
          }
        }
        if (second_node_parent_index == -1 ||
            second_node_parent_index != prev_first_node_index) {
          continue;
        }
        int consumers_count = 0;
        for (const auto& node : nodes) {
          for (const auto& input : node.inputs) {
            if (input == elem_root.inputs[1]) {
              consumers_count++;
            }
          }
        }
        if (consumers_count != 1) {
          continue;
        }

        GPUOperation new_operation;
        ABSL_RETURN_IF_ERROR(Fuse2InputElemWithSimpleElemAsSecondInput(
            gpu_info, std::move(*prev_second_node.gpu_operation.get()),
            std::move(*elem_root.gpu_operation.get()), &new_operation));

        GpuNode new_node;
        new_node.inputs.push_back(prev_second_node.inputs[0]);
        new_node.outputs.push_back(elem_root.outputs[0]);
        new_node.name = prev_second_node.name + " -> " + elem_root.name;
        new_node.gpu_operation =
            std::make_unique<GPUOperation>(std::move(new_operation));
        new_node.optional_tag = elem_root.optional_tag;

        nodes.erase(nodes.begin() + elem_root_index);
        nodes[prev_second_node_index] = std::move(new_node);
        elem_root_index = prev_second_node_index;
        continue;
      }

      // check TYPE_3
      // TYPE_3
      //      input           input
      //     /    \             |
      //  elem0  elem1          |
      //     \    /      -->  elem
      //   elem_root            |
      //       |                |
      //     output           output
      if (prev_first_node.gpu_operation->IsLinkable() &&
          prev_second_node.gpu_operation->IsLinkable() &&
          prev_first_node.inputs.size() == 1 &&
          prev_first_node.outputs.size() == 1 &&
          prev_first_node.optional_tag == elem_root.optional_tag &&
          prev_second_node.inputs.size() == 1 &&
          prev_second_node.outputs.size() == 1 &&
          prev_second_node.optional_tag == elem_root.optional_tag &&
          !HasId(gpu_model->output_ids_and_refs, prev_first_node.outputs[0]) &&
          !HasId(gpu_model->output_ids_and_refs, prev_second_node.outputs[0])) {
        int first_node_parent_index = -1;
        for (int j = prev_first_node_index - 1; j >= 0; --j) {
          if (nodes[j].outputs[0] == prev_first_node.inputs[0]) {
            first_node_parent_index = j;
            break;
          }
        }
        int second_node_parent_index = -1;
        for (int j = prev_second_node_index - 1; j >= 0; --j) {
          if (nodes[j].outputs[0] == prev_second_node.inputs[0]) {
            second_node_parent_index = j;
            break;
          }
        }
        if (first_node_parent_index == -1 || second_node_parent_index == -1 ||
            first_node_parent_index != second_node_parent_index) {
          continue;
        }

        int consumers_count = 0;
        for (const auto& node : nodes) {
          for (const auto& input : node.inputs) {
            if (input == elem_root.inputs[1]) {
              consumers_count++;
            }
          }
        }
        if (consumers_count != 1) {
          continue;
        }

        consumers_count = 0;
        for (const auto& node : nodes) {
          for (const auto& input : node.inputs) {
            if (input == elem_root.inputs[0]) {
              consumers_count++;
            }
          }
        }
        if (consumers_count != 1) {
          continue;
        }

        GPUOperation new_operation;
        ABSL_RETURN_IF_ERROR(Fuse2InputElemWith2SimpleElem(
            gpu_info, std::move(*prev_first_node.gpu_operation.get()),
            std::move(*prev_second_node.gpu_operation.get()),
            std::move(*elem_root.gpu_operation.get()), &new_operation));
        GpuNode new_node;
        new_node.inputs.push_back(prev_first_node.inputs[0]);
        new_node.outputs.push_back(elem_root.outputs[0]);
        new_node.name = prev_first_node.name + " -> " + prev_second_node.name +
                        " -> " + elem_root.name;
        new_node.gpu_operation =
            std::make_unique<GPUOperation>(std::move(new_operation));
        new_node.optional_tag = elem_root.optional_tag;

        // prev_first_node_index and prev_second_node_index ordered relative to
        // elem_root inputs.
        // first_prev_node_index and second_prev_node_index ordered relative to
        // nodes.
        int first_prev_node_index =
            std::min(prev_first_node_index, prev_second_node_index);
        int second_prev_node_index =
            std::max(prev_first_node_index, prev_second_node_index);
        nodes.erase(nodes.begin() + elem_root_index);
        nodes.erase(nodes.begin() + second_prev_node_index);
        nodes[first_prev_node_index] = std::move(new_node);
        elem_root_index = first_prev_node_index - 1;
        continue;
      }
    }
  }
  return absl::OkStatus();
}

absl::Status LinkNodes(const GpuInfo& gpu_info, GpuModel* gpu_model) {
  absl::flat_hash_set<ValueId> ready_tensors;
  for (const auto& input : gpu_model->input_ids_and_refs) {
    ready_tensors.insert(input.first);
  }
  for (const auto& tensor : gpu_model->const_tensors) {
    ready_tensors.insert(tensor.first);
  }
  auto& nodes = gpu_model->nodes;
  // Maps a tensor id to a pair of (input index, node index) where that tensor
  // is found as an input.
  absl::flat_hash_map<ValueId, std::vector<std::pair<int, int>>> links;
  for (int i = 0; i < nodes.size(); ++i) {
    for (int j = 0; j < nodes[i].inputs.size(); ++j) {
      links[nodes[i].inputs[j]].push_back({j, i});
    }
  }
  std::vector<GpuNode> new_nodes;
  new_nodes.reserve(nodes.size());
  for (int i = 0; i < nodes.size(); ++i) {
    auto& node = nodes[i];
    if (!node.gpu_operation) {
      continue;
    }
    for (const auto& out_id : node.outputs) {
      ready_tensors.insert(out_id);
    }
    const int out_index = 0;
    std::vector<int> next_nodes;
    int link_index = 0;
    for (const auto& link : links[node.outputs[out_index]]) {
      if (link.second > i) {
        next_nodes.push_back(link.second);
        link_index = link.first;
      }
    }
    if (next_nodes.size() != 1 || link_index != 0 ||
        HasId(gpu_model->output_ids_and_refs, node.outputs[out_index])) {
      new_nodes.push_back(std::move(node));
      continue;
    }
    auto& linkable_node = nodes[next_nodes[0]];
    if (!linkable_node.gpu_operation ||
        !linkable_node.gpu_operation->IsLinkable() ||
        linkable_node.outputs.size() != 1 ||
        !IsReady(ready_tensors, linkable_node) ||
        node.optional_tag != linkable_node.optional_tag ||
        !node.subgraph_id.empty()) {
      new_nodes.push_back(std::move(node));
      continue;
    }
    ABSL_RETURN_IF_ERROR(MergeGpuNodes(gpu_info, &linkable_node, &node));
    // Clear the GPU operation to indicate this node has been merged.
    nodes[next_nodes[0]].gpu_operation = nullptr;
    i -= 1;
  }
  gpu_model->nodes = std::move(new_nodes);
  return absl::OkStatus();
}

absl::Status MergeReorderNodes(GpuModel* gpu_model) {
  auto& nodes = gpu_model->nodes;
  for (int i = 0; i < nodes.size(); ++i) {
    auto& first_node = nodes[i];
    if (first_node.inputs.size() != 1 || first_node.outputs.size() != 1 ||
        !first_node.gpu_operation->IsReorderOp() ||
        HasId(gpu_model->output_ids_and_refs, first_node.outputs[0])) {
      continue;
    }
    std::vector<int> next_nodes;
    for (int j = i + 1; j < nodes.size(); ++j) {
      for (int k = 0; k < nodes[j].inputs.size(); ++k) {
        if (nodes[j].inputs[k] == first_node.outputs[0]) {
          next_nodes.push_back(j);
        }
      }
    }
    if (next_nodes.size() != 1) {
      continue;
    }
    auto& second_node = nodes[next_nodes[0]];
    if (second_node.inputs.size() != 1 || second_node.outputs.size() != 1 ||
        !second_node.gpu_operation->IsReorderOp() ||
        first_node.optional_tag != second_node.optional_tag) {
      continue;
    }

    //     t0          t0
    //      |           |
    //  first_node      |
    //      |           |
    //     t1   -> second_node
    //      |           |
    // second_node      |
    //      |           |
    //     t2          t2
    const BHWC interm_shape =
        gpu_model->tensors[first_node.outputs[0]].GetBHWCShape();
    second_node.inputs[0] = first_node.inputs[0];
    second_node.name = first_node.name + " -> " + second_node.name;
    ABSL_RETURN_IF_ERROR(second_node.gpu_operation->AddReorderOperation(
        interm_shape, first_node.gpu_operation.get()));
    nodes.erase(nodes.begin() + i);
    i -= 1;
  }
  return absl::OkStatus();
}

// Serialized model will lose polymorphic properties for GpuOperations.
// Here we will retrieve some information needed for generic execution of
// GpuOperations. Specifically, BindArguments and RecalculateGridSize must be
// executed.
absl::Status ResolvePolymorphicArgs(GpuModel* gpu_model) {
  class DummySpatialTensor : public GpuSpatialTensor {
   public:
    DummySpatialTensor() = default;
    explicit DummySpatialTensor(const BHWDC& shape,
                                const TensorDescriptor& tensor_desc)
        : shape_(shape), tensor_desc_(tensor_desc) {}
    ~DummySpatialTensor() override = default;

    int Width() const override { return shape_.w; }
    int Height() const override { return shape_.h; }
    int Depth() const override { return shape_.d; }
    int Channels() const override { return shape_.c; }
    int Slices() const override { return DivideRoundUp(shape_.c, 4); }
    int Batch() const override { return shape_.b; }

    TensorDescriptor GetDescriptor() const override { return tensor_desc_; }

   private:
    BHWDC shape_;
    TensorDescriptor tensor_desc_;
  };

  for (auto& node : gpu_model->nodes) {
    std::vector<DummySpatialTensor> src_tensors(node.inputs.size());
    for (int i = 0; i < node.inputs.size(); ++i) {
      const auto& tensor_desc =
          gpu_model->const_tensors.contains(node.inputs[i])
              ? gpu_model->const_tensors[node.inputs[i]]
              : gpu_model->tensors[node.inputs[i]];
      src_tensors[i] =
          DummySpatialTensor(tensor_desc.GetBHWDCShape(), tensor_desc);
      node.gpu_operation->SetSrc(&src_tensors[i], i);
    }
    std::vector<DummySpatialTensor> dst_tensors(node.outputs.size());
    for (int i = 0; i < node.outputs.size(); ++i) {
      const auto& tensor_desc = gpu_model->tensors[node.outputs[i]];
      dst_tensors[i] =
          DummySpatialTensor(tensor_desc.GetBHWDCShape(), tensor_desc);
      node.gpu_operation->SetDst(&dst_tensors[i], i);
    }
    ABSL_RETURN_IF_ERROR(
        node.gpu_operation->BindArguments(&node.gpu_operation->args_));
    node.gpu_operation->RecalculateGridSize();
  }
  return absl::OkStatus();
}

// Expands any nodes with a non-empty `subgraph_id` to the corresponding
// subgraph nodes. For example, given the GpuModel:
//    nodes: [
//      {name: "foo", subgraph_id: ""},
//      {name: "subgraph a", subgraph_id: "a"},
//      {name: "bar", subgraph_id: ""},
//      {name: "subgraph b", subgraph_id: "b"},
//    ]
//    subgraphs: {
//      "a": [
//        {name: "a1"},
//        {name: "a2"},
//      ],
//      "b": [
//        {name: "b1"},
//        {name: "b2"},
//      ],
//    }
//
// The final model would look like:
//    nodes: [
//      {name: "foo", subgraph_id: ""},
//      {name: "a1", subgraph_id: "a"},
//      {name: "a2", subgraph_id: "a"},
//      {name: "bar", subgraph_id: ""},
//      {name: "b1", subgraph_id: "b"},
//      {name: "b2", subgraph_id: "b"},
//    ]
//    subgraphs: {
//      "a": [
//        {name: "a1"},
//        {name: "a2"},
//      ],
//      "b": [
//        {name: "b1"},
//        {name: "b2"},
//      ],
//    }
//
// The subgraphs will not have a GPUOperation set, and it is the responsibility
// of the inference context to set up the op.
void ExpandSubgraphNodes(GpuModel* gpu_model) {
  if (gpu_model->subgraphs.empty()) {
    return;
  }

  // Find the next tensor id to use for intermediate tensors.
  ValueId next_id = 0;
  for (const auto& [id, tensor] : gpu_model->tensors) {
    next_id = std::max(next_id, id);
  }
  for (const auto& [id, tensor] : gpu_model->const_tensors) {
    next_id = std::max(next_id, id);
  }

  std::vector<GpuNode> new_nodes;
  new_nodes.reserve(gpu_model->nodes.size());
  for (auto& node : gpu_model->nodes) {
    if (node.subgraph_id.empty()) {
      new_nodes.push_back(std::move(node));
      continue;
    }
    auto& subgraph = gpu_model->subgraphs[node.subgraph_id];

    // Map all inputs and outputs of the subgraph to the corresponding values of
    // the subgraph node.
    absl::flat_hash_map<ValueId, ValueId> id_map;
    for (int i = 0; i < node.inputs.size(); ++i) {
      id_map[subgraph.input_ids_and_refs[i].first] = node.inputs[i];
    }
    for (int i = 0; i < node.outputs.size(); ++i) {
      id_map[subgraph.output_ids_and_refs[i].first] = node.outputs[i];
    }
    // Replace all intermediate tensors with tensors from `gpu_model`.

    for (const auto& [id, desc] : subgraph.tensors) {
      if (!id_map.contains(id)) {
        gpu_model->tensors[++next_id] = desc;
        id_map[id] = next_id;
      }
    }
    for (const auto& [id, desc] : subgraph.const_tensors) {
      if (!id_map.contains(id)) {
        gpu_model->const_tensors[++next_id] = desc;
        id_map[id] = next_id;
      }
    }
    // Copy each subgraph node.
    for (const auto& sub_node : subgraph.nodes) {
      GpuNode new_node;
      for (ValueId id : sub_node.inputs) {
        new_node.inputs.push_back(id_map[id]);
      }
      for (ValueId id : sub_node.outputs) {
        new_node.outputs.push_back(id_map[id]);
      }
      new_node.name = sub_node.name;
      new_node.optional_tag = sub_node.optional_tag;
      new_node.optional_tag.insert(node.optional_tag.begin(),
                                   node.optional_tag.end());
      new_node.subgraph_id = node.subgraph_id;
      new_nodes.push_back(std::move(new_node));
    }
  }
  gpu_model->nodes = std::move(new_nodes);
}

}  // namespace

absl::Status AssembleCode(const GpuInfo& gpu_info, GpuModel* gpu_model) {
  for (auto& node : gpu_model->nodes) {
    ABSL_RETURN_IF_ERROR(node.gpu_operation->AssembleCode(gpu_info));
  }
  for (auto& [id, subgraph] : gpu_model->subgraphs) {
    ABSL_RETURN_IF_ERROR(AssembleCode(gpu_info, &subgraph));
  }
  return absl::OkStatus();
}

absl::Status ResolveArgs(GpuModel* gpu_model) {
  return ResolvePolymorphicArgs(gpu_model);
}

void ExpandSubgraphs(GpuModel* gpu_model) { ExpandSubgraphNodes(gpu_model); }

absl::Status MergeNodes(const GpuInfo& gpu_info, GpuModel* gpu_model) {
  ABSL_RETURN_IF_ERROR(MergeReorderNodes(gpu_model));
  ABSL_RETURN_IF_ERROR(MergeElementwiseNodes(gpu_info, gpu_model));
  ABSL_RETURN_IF_ERROR(LinkNodes(gpu_info, gpu_model));
  return absl::OkStatus();
}

}  // namespace ml_drift
