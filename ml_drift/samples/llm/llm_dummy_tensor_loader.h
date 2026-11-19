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

#ifndef ML_DRIFT_SAMPLES_LLM_LLM_DUMMY_TENSOR_LOADER_H_
#define ML_DRIFT_SAMPLES_LLM_LLM_DUMMY_TENSOR_LOADER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/samples/llm/llm_tensor_loader.h"

namespace ml_drift {

class DummyGpuSpatialTensor : public GpuSpatialTensor {
 public:
  explicit DummyGpuSpatialTensor(const TensorDescriptor& desc) : desc_(desc) {}
  int Width() const override { return desc_.GetBHWCShape().w; }
  int Height() const override { return desc_.GetBHWCShape().h; }
  int Depth() const override { return 1; }
  int Channels() const override { return desc_.GetBHWCShape().c; }
  int Slices() const override { return (desc_.GetBHWCShape().c + 3) / 4; }
  int Batch() const override { return desc_.GetBHWCShape().b; }

  TensorDescriptor GetDescriptor() const override { return desc_; }

 private:
  TensorDescriptor desc_;
};

class LlmDummyTensorLoader : public LlmTensorLoader {
 public:
  explicit LlmDummyTensorLoader(int quantization_i_group_size = -1,
                                int default_element_size_in_bits = 4);

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

  void SetDefaultElementSizeInBits(int bits) {
    default_element_size_in_bits_ = bits;
  }
  void SetTensorElementSizeInBits(const std::string& tensor_name, int bits) {
    element_size_map_[tensor_name] = bits;
  }

 private:
  absl::StatusOr<GpuSpatialTensor*> CreateTensorInternal(
      const TensorDescriptor& td);

  absl::StatusOr<Int4Weights> LoadInt4WeightsFast(
      const std::string& tensor_name,
      const ml_drift::WeightsDescription& weights_desc, const OHWI& shape,
      bool swap_dims);

  int quantization_i_group_size_;
  int default_element_size_in_bits_ = 4;
  std::unordered_map<std::string, int> element_size_map_;
  std::vector<std::unique_ptr<DummyGpuSpatialTensor>> owned_tensors_;
};

}  // namespace ml_drift

#endif  // ML_DRIFT_SAMPLES_LLM_LLM_DUMMY_TENSOR_LOADER_H_
