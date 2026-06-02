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

#include "ml_drift/common/gpu_model_dumper.h"

#include <algorithm>
#include <string>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"

namespace ml_drift {
namespace {

inline std::string DumpTensorDescriptor(const TensorDescriptor& desc) {
  return absl::StrCat("Shape: ", ToString(desc.GetBHWDCShape()),
                      ", Type: ", ToString(desc.GetStorageType()),
                      ", Layout: ", ToString(desc.GetLayout()),
                      ", DataType: ", ToString(desc.GetDataType()));
}

class DefaultGpuModelDumper : public GpuModelDumper {
 public:
  void Dump(const GpuModel& model) const override {
    ABSL_LOG(INFO) << "GpuModel Dump:";

    ABSL_LOG(INFO) << "Inputs:";
    for (const auto& input : model.input_ids_and_refs) {
      ABSL_LOG(INFO) << "  ID: " << input.first << ", Ref: " << input.second;
    }

    ABSL_LOG(INFO) << "Outputs:";
    for (const auto& output_pair : model.output_ids_and_refs) {
      ABSL_LOG(INFO) << "  ID: " << output_pair.first
                     << ", Ref: " << output_pair.second;
    }

    ABSL_LOG(INFO) << "Tensors:";
    std::vector<ValueId> tensor_ids;
    for (const auto& tensor : model.tensors) {
      tensor_ids.push_back(tensor.first);
    }
    std::sort(tensor_ids.begin(), tensor_ids.end());
    for (const auto& id : tensor_ids) {
      const auto& tensor = model.tensors.at(id);
      ABSL_LOG(INFO) << "  ID: " << id << " - " << DumpTensorDescriptor(tensor);
    }

    ABSL_LOG(INFO) << "Constant Tensors:";
    std::vector<ValueId> const_tensor_ids;
    for (const auto& tensor : model.const_tensors) {
      const_tensor_ids.push_back(tensor.first);
    }
    std::sort(const_tensor_ids.begin(), const_tensor_ids.end());
    for (const auto& id : const_tensor_ids) {
      const auto& tensor = model.const_tensors.at(id);
      ABSL_LOG(INFO) << "  ID: " << id << " - " << DumpTensorDescriptor(tensor);
    }

    ABSL_LOG(INFO) << "Nodes:";
    for (int i = 0; i < model.nodes.size(); ++i) {
      const auto& node = model.nodes[i];
      ABSL_LOG(INFO) << "  ID: " << i << " - " << node.name << "; in = ["
                     << absl::StrJoin(node.inputs, ", ") << "]; out = ["
                     << absl::StrJoin(node.outputs, ", ") << "]";
    }

    ABSL_LOG(INFO) << "Subgraphs:";
    for (const auto& subgraph : model.subgraphs) {
      ABSL_LOG(INFO) << "  Subgraph: " << subgraph.first;
      ABSL_LOG(INFO) << "  --- Begin Subgraph " << subgraph.first << " ---";
      Dump(subgraph.second);
      ABSL_LOG(INFO) << "  --- End Subgraph " << subgraph.first << " ---";
    }
  }
};

}  // namespace

void DumpGpuModel(const GpuModel& model, const GpuModelDumper& dumper) {
  dumper.Dump(model);
}

void DumpGpuModel(const GpuModel& model) {
  DefaultGpuModelDumper().Dump(model);
}

}  // namespace ml_drift
