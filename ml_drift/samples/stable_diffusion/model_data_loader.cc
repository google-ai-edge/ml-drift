// Copyright 2023 The ML Drift Authors.
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

#include "ml_drift/samples/stable_diffusion/model_data_loader.h"

#include <fcntl.h>
#include <sys/mman.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ml_drift/common/types.h"
#include "ml_drift/samples/stable_diffusion/util.h"

namespace ml_drift {
absl::StatusOr<std::unique_ptr<ModelDataLoader>>
ModelDataLoader::CreateFromWeightsDir(const std::string& weights_dir) {
  auto model_data_loader = absl::WrapUnique(new ModelDataLoader());
  model_data_loader->SetWeightsDir(weights_dir);
  return model_data_loader;
}

absl::StatusOr<absl::Span<const half>> ModelDataLoader::GetData(
    const std::string& weights_name, int count) {
  if (!weights_dir_.empty()) {
    if (buffers_[0].size() < count) {
      buffers_[0].resize(count);
    }
    return LoadF16Ex(weights_dir_ + weights_name, count, buffers_[0].data());
  }
  return absl::FailedPreconditionError(
      "TFLite loading not supported in this target. Weights directory must be "
      "set.");
}

absl::StatusOr<std::pair<absl::Span<const half>, absl::Span<const half>>>
ModelDataLoader::GetData(const std::string& weights1_name,
                         const std::string& weights2_name, int count1,
                         int count2) {
  absl::Span<const half> data1, data2;
  if (weights_dir_.empty()) {
    return absl::FailedPreconditionError(
        "TFLite loading not supported in this target. Weights directory must "
        "be set.");
  }

  if (buffers_[0].size() < count1) {
    buffers_[0].resize(count1);
  }
  data1 = LoadF16Ex(weights_dir_ + weights1_name, count1, buffers_[0].data());

  if (buffers_[1].size() < count2) {
    buffers_[1].resize(count2);
  }
  data2 = LoadF16Ex(weights_dir_ + weights2_name, count2, buffers_[1].data());

  return std::make_pair(data1, data2);
}

}  // namespace ml_drift
