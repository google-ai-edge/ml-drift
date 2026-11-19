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

#ifndef ML_DRIFT_SAMPLES_LLM_LLM_WEIGHTS_LOADER_H_
#define ML_DRIFT_SAMPLES_LLM_LLM_WEIGHTS_LOADER_H_

#include <string>
#include <type_traits>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/samples/llm/llm_tensor_loader.h"

namespace ml_drift {

inline constexpr bool kSwapDims = false;
inline constexpr char kWeightSuffix[] = ".weight";

struct WeightsWithPrecision {
  GpuModelBuilder::Weights weights;
  DataType weight_type = DataType::kUnknown;
};

// Context object bundling dependencies required for weight loading.
// Can be constructed directly or implicitly from any BuildContext struct that
// exposes .builder, .create_info, and .loader.
struct LlmWeightsLoader {
  GpuModelBuilder& builder;
  CreateGpuModelInfo& create_info;
  LlmTensorLoader* loader;

  template <typename Ctx, typename = std::enable_if_t<!std::is_same_v<
                              std::decay_t<Ctx>, LlmWeightsLoader>>>
  /*implicit*/ LlmWeightsLoader(Ctx& ctx)
      : builder(ctx.builder),
        create_info(ctx.create_info),
        loader(ctx.loader) {}

  LlmWeightsLoader(GpuModelBuilder& b, CreateGpuModelInfo& ci,
                   LlmTensorLoader* l)
      : builder(b), create_info(ci), loader(l) {}

  GpuModelBuilder::TensorHandle AddExternalTensor(
      GpuSpatialTensor* tensor) const {
    GpuModelBuilder::TensorHandle th =
        builder.AddTensor(tensor->GetDescriptor());
    create_info.external_immutable_tensors[th.id] = tensor;
    return th;
  }

  DataType data_type() const {
    return DeduceDataTypeFromPrecision(create_info.precision);
  }

  // Loads INT4 weights with per-group scales and optional zero-points.
  GpuModelBuilder::Weights GetWeightsInt4(const std::string& w_name,
                                          const std::string& base_name,
                                          const OHWI& weights_shape) const;

  // Loads INT8 weights with per-group scales and optional zero-points.
  GpuModelBuilder::Weights GetWeightsInt8(const std::string& w_name,
                                          const std::string& base_name,
                                          const OHWI& weights_shape) const;

  // Loads floating-point weights (F16 or F32 depending on precision).
  GpuModelBuilder::Weights GetWeightsFloat(const std::string& w_name,
                                           const OHWI& weights_shape) const;

  // Loads weights with automatic precision detection (INT4, INT8, or float).
  WeightsWithPrecision GetWeights(const std::string& name,
                                  const OHWI& weights_shape) const;
};

using LlmWeightsContext = LlmWeightsLoader;

// Free function overloads that delegate to LlmWeightsLoader, enabling
// seamless calls like GetWeights(ctx, name, weights_shape).

inline GpuModelBuilder::Weights GetWeightsInt4(LlmWeightsLoader ctx,
                                               const std::string& w_name,
                                               const std::string& base_name,
                                               const OHWI& weights_shape) {
  return ctx.GetWeightsInt4(w_name, base_name, weights_shape);
}

inline GpuModelBuilder::Weights GetWeightsInt8(LlmWeightsLoader ctx,
                                               const std::string& w_name,
                                               const std::string& base_name,
                                               const OHWI& weights_shape) {
  return ctx.GetWeightsInt8(w_name, base_name, weights_shape);
}

inline GpuModelBuilder::Weights GetWeightsFloat(LlmWeightsLoader ctx,
                                                const std::string& w_name,
                                                const OHWI& weights_shape) {
  return ctx.GetWeightsFloat(w_name, weights_shape);
}

inline WeightsWithPrecision GetWeights(LlmWeightsLoader ctx,
                                       const std::string& name,
                                       const OHWI& weights_shape) {
  return ctx.GetWeights(name, weights_shape);
}

}  // namespace ml_drift

#endif  // ML_DRIFT_SAMPLES_LLM_LLM_WEIGHTS_LOADER_H_
