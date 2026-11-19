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

#ifndef ML_DRIFT_SAMPLES_LLM_LLM_FILE_TENSOR_LOADER_H_
#define ML_DRIFT_SAMPLES_LLM_LLM_FILE_TENSOR_LOADER_H_

#include <string>
#include <unordered_map>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/samples/llm/llm_tensor_loader.h"

namespace ml_drift {

class LlmFileTensorLoader : public LlmTensorLoader {
 public:
  explicit LlmFileTensorLoader(const std::string& weights_directory);

  std::vector<float> LoadFloat32(const std::string& name, int size) override;

  absl::StatusOr<ml_drift::GpuSpatialTensor*> LoadWeights(
      const std::string& name, const WeightsDescription& weights_desc,
      const OHWI& shape, bool swap_dims) override;

  absl::StatusOr<Int8Weights> LoadInt8Weights(
      const std::string& tensor_name,
      const ml_drift::WeightsDescription& weights_desc, const OHWI& shape,
      bool swap_dims) override;

  absl::StatusOr<Int4Weights> LoadInt4Weights(
      const std::string& tensor_name,
      const ml_drift::WeightsDescription& weights_desc, const OHWI& shape,
      bool swap_dims) override;

  int GetQuantizationIGroupSize(
      const std::string& weights_name,
      const ml_drift::OHWI& weights_shape) const override;

  absl::StatusOr<ml_drift::GpuSpatialTensor*> LoadScale(
      const std::string& tensor_name, const ::ml_drift::OHWI& shape) override;

  absl::StatusOr<ml_drift::GpuSpatialTensor*> LoadZeroPoint(
      const std::string& zero_point_name, const std::string& scale_name,
      const ::ml_drift::OHWI& shape) override;

  bool HasTensor(const std::string& name) const override;

  int GetTensorElementSizeInBits(const std::string& tensor_name,
                                 int count) override;

 private:
  std::vector<int8_t> LoadInt8(const std::string& tensor_name, int count);
  std::vector<int8_t> LoadInt4AsInt8(const std::string& tensor_name, int count);
  std::vector<uint8_t> LoadInt4Packed(const std::string& tensor_name,
                                      int count);
  absl::StatusOr<Int4Weights> LoadInt4WeightsFast(
      const std::string& tensor_name,
      const ml_drift::WeightsDescription& weights_desc, const OHWI& shape,
      bool swap_dims);

  std::string weights_directory_;
  mutable std::unordered_map<std::string, int> group_size_cache_;
  mutable std::unordered_map<std::string, size_t> file_size_cache_;
  mutable std::unordered_map<std::string, bool> has_tensor_cache_;
};

}  // namespace ml_drift

#endif  // ML_DRIFT_SAMPLES_LLM_LLM_FILE_TENSOR_LOADER_H_
