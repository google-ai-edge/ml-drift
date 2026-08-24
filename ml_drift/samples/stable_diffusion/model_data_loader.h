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

#ifndef ML_DRIFT_SAMPLES_STABLE_DIFFUSION_MODEL_DATA_LOADER_H_
#define ML_DRIFT_SAMPLES_STABLE_DIFFUSION_MODEL_DATA_LOADER_H_
#include <sys/types.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class ModelDataLoader {
 public:
  static absl::StatusOr<std::unique_ptr<ModelDataLoader>> CreateFromWeightsDir(
      const std::string& weights_dir);

  virtual ~ModelDataLoader() = default;

  virtual absl::StatusOr<absl::Span<const half>> GetData(
      const std::string& weights_name, int count);

  virtual absl::StatusOr<
      std::pair<absl::Span<const half>, absl::Span<const half>>>
  GetData(const std::string& weights1_name, const std::string& weights2_name,
          int count1, int count2);

 protected:
  ModelDataLoader() = default;
  ModelDataLoader(const ModelDataLoader&) = default;
  ModelDataLoader& operator=(const ModelDataLoader&) = default;

 private:
  std::vector<half> DeserializeF16Buffer(
      absl::Span<const uint8_t> serialized_data);

  void SetWeightsDir(const std::string& weights_dir) {
    weights_dir_ = weights_dir;
    buffers_.resize(2);
  }

  std::string weights_dir_;
  std::vector<std::vector<half>> buffers_;
};

}  // namespace ml_drift

#endif  // ML_DRIFT_SAMPLES_STABLE_DIFFUSION_MODEL_DATA_LOADER_H_
