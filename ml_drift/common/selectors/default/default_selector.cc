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

#include <memory>

#include "absl/strings/str_cat.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {

absl::StatusOr<std::unique_ptr<GPUOperation>> SelectDefault(
    const GpuInfo& gpu_info, const OperationDef& op_def, const Node& node) {
  return absl::UnimplementedError(
      absl::StrCat("No selector for ", node.operation.type));
}

}  // namespace ml_drift
