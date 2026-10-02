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

#ifndef ML_DRIFT_SAMPLES_LLM_LLM_TENSOR_LOADER_H_
#define ML_DRIFT_SAMPLES_LLM_LLM_TENSOR_LOADER_H_

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/weights_layout.h"

namespace ml_drift {

class LlmTensorLoader {
 public:
  using CreateTensorFn =
      std::function<absl::StatusOr<::ml_drift::GpuSpatialTensor*>(
          const ::ml_drift::TensorDescriptor&)>;

  virtual ~LlmTensorLoader() = default;

  virtual std::vector<float> LoadFloat32(const std::string& name, int size) = 0;

  virtual absl::StatusOr<ml_drift::GpuSpatialTensor*> LoadWeights(
      const std::string& name, const WeightsDescription& weights_desc,
      const OHWI& shape, bool swap_dims) = 0;

  struct Int8Weights {
    ml_drift::GpuSpatialTensor* weights = nullptr;
    ml_drift::GpuSpatialTensor* weights_sum_i = nullptr;
  };

  virtual absl::StatusOr<Int8Weights> LoadInt8Weights(
      const std::string& name, const WeightsDescription& weights_desc,
      const OHWI& shape, bool swap_dims) = 0;

  struct Int4Weights {
    ml_drift::GpuSpatialTensor* weights = nullptr;
    ml_drift::GpuSpatialTensor* weights_sum_i = nullptr;
  };

  virtual absl::StatusOr<Int4Weights> LoadInt4Weights(
      const std::string& name, const WeightsDescription& weights_desc,
      const OHWI& shape, bool swap_dims) = 0;

  virtual void SetCreateTensorFn(CreateTensorFn create_tensor_fn) {
    create_tensor_fn_ = std::move(create_tensor_fn);
  }

  virtual void SetGpuInfo(const ::ml_drift::GpuInfo& gpu_info) {
    gpu_info_ = gpu_info;
  }
  virtual void SetDataType(::ml_drift::DataType type) { data_type_ = type; }

  virtual int GetQuantizationIGroupSize(
      const std::string& weights_name,
      const ml_drift::OHWI& weights_shape) const = 0;

  virtual absl::StatusOr<::ml_drift::GpuSpatialTensor*> LoadScale(
      const std::string& tensor_name, const ::ml_drift::OHWI& shape) = 0;

  virtual absl::StatusOr<::ml_drift::GpuSpatialTensor*> LoadZeroPoint(
      const std::string& zero_point_name, const std::string& scale_name,
      const ::ml_drift::OHWI& shape) = 0;

  virtual bool HasTensor(const std::string& name) const = 0;

  virtual int GetTensorElementSizeInBits(const std::string& tensor_name,
                                         int count) = 0;

  virtual ml_drift::DataType GetDataTypeForWeights(const std::string& name,
                                                   int count) {
    const int element_size_in_bits = GetTensorElementSizeInBits(name, count);
    if (element_size_in_bits == 32) {
      return ml_drift::DataType::kFloat32;
    } else if (element_size_in_bits == 16) {
      return ml_drift::DataType::kFloat16;
    } else if (element_size_in_bits == 8) {
      return ml_drift::DataType::kInt8;
    } else if (element_size_in_bits == 4) {
      return ml_drift::DataType::kInt4;
    }
    return ml_drift::DataType::kUnknown;
  }

  static std::unique_ptr<LlmTensorLoader> MakeCaching(
      std::unique_ptr<LlmTensorLoader> backend);

 protected:
  CreateTensorFn create_tensor_fn_;
  ::ml_drift::GpuInfo gpu_info_;
  ::ml_drift::DataType data_type_;
};

class LlmCachingTensorLoader : public LlmTensorLoader {
 public:
  explicit LlmCachingTensorLoader(std::unique_ptr<LlmTensorLoader> backend)
      : backend_(std::move(backend)) {}

  void SetCreateTensorFn(CreateTensorFn create_tensor_fn) override {
    backend_->SetCreateTensorFn(std::move(create_tensor_fn));
  }

  std::vector<float> LoadFloat32(const std::string& name, int size) override {
    std::vector<float>& cached = float_cache_[name];
    if (cached.empty()) {
      cached = backend_->LoadFloat32(name, size);
    }
    return cached;
  }

  absl::StatusOr<ml_drift::GpuSpatialTensor*> LoadWeights(
      const std::string& name, const WeightsDescription& weights_desc,
      const OHWI& shape, bool swap_dims) override {
    std::string cache_name = GetName(name, shape);
    if (auto it = cache_.find(cache_name); it != cache_.end()) {
      return it->second;
    }
    return CacheResult(
        backend_->LoadWeights(name, weights_desc, shape, swap_dims),
        cache_name);
  }

  absl::StatusOr<Int8Weights> LoadInt8Weights(
      const std::string& name, const WeightsDescription& weights_desc,
      const OHWI& shape, bool swap_dims) override {
    std::string cache_name = GetName(name, shape);
    if (auto it = int8_cache_.find(cache_name); it != int8_cache_.end()) {
      return it->second;
    }
    auto res = backend_->LoadInt8Weights(name, weights_desc, shape, swap_dims);
    if (res.ok()) {
      int8_cache_[cache_name] = *res;
    }
    return res;
  }

  absl::StatusOr<Int4Weights> LoadInt4Weights(
      const std::string& name, const WeightsDescription& weights_desc,
      const OHWI& shape, bool swap_dims) override {
    std::string cache_name = GetName(name, shape);
    if (auto it = int4_cache_.find(cache_name); it != int4_cache_.end()) {
      return it->second;
    }
    auto res = backend_->LoadInt4Weights(name, weights_desc, shape, swap_dims);
    if (res.ok()) {
      int4_cache_[cache_name] = *res;
    }
    return res;
  }

  void SetGpuInfo(const ::ml_drift::GpuInfo& gpu_info) override {
    backend_->SetGpuInfo(gpu_info);
  }
  void SetDataType(::ml_drift::DataType type) override {
    backend_->SetDataType(type);
  }
  int GetQuantizationIGroupSize(
      const std::string& weights_name,
      const ml_drift::OHWI& weights_shape) const override {
    auto it = group_size_cache_.find(weights_name);
    if (it != group_size_cache_.end()) {
      return it->second;
    }
    int group_size =
        backend_->GetQuantizationIGroupSize(weights_name, weights_shape);
    group_size_cache_[weights_name] = group_size;
    return group_size;
  }
  absl::StatusOr<::ml_drift::GpuSpatialTensor*> LoadScale(
      const std::string& tensor_name, const ::ml_drift::OHWI& shape) override {
    std::string cache_name = GetName(tensor_name, shape);
    if (auto it = cache_.find(cache_name); it != cache_.end()) {
      return it->second;
    }
    return CacheResult(backend_->LoadScale(tensor_name, shape), cache_name);
  }
  absl::StatusOr<::ml_drift::GpuSpatialTensor*> LoadZeroPoint(
      const std::string& zero_point_name, const std::string& scale_name,
      const ::ml_drift::OHWI& shape) override {
    std::string cache_name = GetName(zero_point_name, shape);
    if (auto it = cache_.find(cache_name); it != cache_.end()) {
      return it->second;
    }
    return CacheResult(
        backend_->LoadZeroPoint(zero_point_name, scale_name, shape),
        cache_name);
  }

  bool HasTensor(const std::string& name) const override {
    auto it = has_tensor_cache_.find(name);
    if (it != has_tensor_cache_.end()) {
      return it->second;
    }
    bool has = backend_->HasTensor(name);
    has_tensor_cache_[name] = has;
    return has;
  }

  int GetTensorElementSizeInBits(const std::string& tensor_name,
                                 int count) override {
    auto it = element_size_cache_.find(tensor_name);
    if (it != element_size_cache_.end()) {
      return it->second;
    }
    int size = backend_->GetTensorElementSizeInBits(tensor_name, count);
    element_size_cache_[tensor_name] = size;
    return size;
  }

 private:
  static std::string GetName(const std::string& name,
                             const ::ml_drift::OHWI& shape) {
    return absl::StrCat(name, ":", shape.o, ",", shape.h, ",", shape.w, ",",
                        shape.i);
  }
  absl::StatusOr<::ml_drift::GpuSpatialTensor*> CacheResult(
      absl::StatusOr<::ml_drift::GpuSpatialTensor*> tensor,
      const std::string& name) {
    if (tensor.ok()) {
      cache_[name] = *tensor;
    }
    return tensor;
  }

  std::unique_ptr<LlmTensorLoader> backend_;
  std::unordered_map<std::string, std::vector<float>> float_cache_;
  std::unordered_map<std::string, ml_drift::GpuSpatialTensor*> cache_;
  std::unordered_map<std::string, Int8Weights> int8_cache_;
  std::unordered_map<std::string, Int4Weights> int4_cache_;
  mutable std::unordered_map<std::string, int> group_size_cache_;
  mutable std::unordered_map<std::string, int> element_size_cache_;
  mutable std::unordered_map<std::string, bool> has_tensor_cache_;
};

inline std::unique_ptr<LlmTensorLoader> LlmTensorLoader::MakeCaching(
    std::unique_ptr<LlmTensorLoader> backend) {
  return std::make_unique<LlmCachingTensorLoader>(std::move(backend));
}

}  // namespace ml_drift

#endif  // ML_DRIFT_SAMPLES_LLM_LLM_TENSOR_LOADER_H_
