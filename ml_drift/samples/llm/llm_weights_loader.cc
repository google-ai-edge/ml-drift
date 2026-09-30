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

#include "ml_drift/samples/llm/llm_weights_loader.h"

#include <string>

#include "absl/log/absl_check.h"
#include "absl/strings/str_cat.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/shape.h"

namespace ml_drift {

using TensorHandle = GpuModelBuilder::TensorHandle;

GpuModelBuilder::Weights LlmWeightsLoader::GetWeightsInt4(
    const std::string& w_name, const std::string& base_name,
    const OHWI& weights_shape) const {
  auto weights_desc = builder.GetFullyConnectedInt4WeightsDesc(weights_shape);
  auto status_or_weights = loader->LoadInt4Weights(
      w_name, weights_desc, weights_shape, /*swap_dims=*/kSwapDims);
  ABSL_CHECK_OK(status_or_weights);
  auto int4_weights = status_or_weights.value();

  TensorHandle weights_th = AddExternalTensor(int4_weights.weights);
  TensorHandle* weights_sum_i_th_ptr = nullptr;
  TensorHandle weights_sum_i_th;
  if (int4_weights.weights_sum_i) {
    weights_sum_i_th = AddExternalTensor(int4_weights.weights_sum_i);
    weights_sum_i_th_ptr = &weights_sum_i_th;
  }

  const std::string scale_name =
      absl::StrCat(base_name, ".weight_quantized_scale");
  const int quantization_i_group_size =
      loader->GetQuantizationIGroupSize(w_name, weights_shape);
  const int scale_i_ch = weights_shape.i / quantization_i_group_size;
  const OHWI scale_zp_shape = OHWI(weights_shape.o, 1, 1, scale_i_ch);
  auto status_or_scale = loader->LoadScale(scale_name, scale_zp_shape);
  ABSL_CHECK_OK(status_or_scale);
  TensorHandle scale_th = AddExternalTensor(status_or_scale.value());

  TensorHandle* zp_th_ptr = nullptr;
  TensorHandle zp_th;
  const std::string zp_name = absl::StrCat(base_name, ".weight_quantized_zp");
  if (loader->HasTensor(zp_name)) {
    auto status_or_zp =
        loader->LoadZeroPoint(zp_name, scale_name, scale_zp_shape);
    if (status_or_zp.ok()) {
      zp_th = AddExternalTensor(status_or_zp.value());
      zp_th_ptr = &zp_th;
    }
  }

  return CreateExternalWeights(weights_th, weights_desc, weights_shape,
                               scale_zp_shape, &scale_th, zp_th_ptr,
                               weights_sum_i_th_ptr);
}

GpuModelBuilder::Weights LlmWeightsLoader::GetWeightsInt8(
    const std::string& w_name, const std::string& base_name,
    const OHWI& weights_shape) const {
  auto weights_desc = builder.GetFullyConnectedInt8WeightsDesc(weights_shape);
  auto status_or_weights = loader->LoadInt8Weights(
      w_name, weights_desc, weights_shape, /*swap_dims=*/kSwapDims);
  ABSL_CHECK_OK(status_or_weights);
  auto int8_weights = status_or_weights.value();

  TensorHandle weights_th = AddExternalTensor(int8_weights.weights);
  TensorHandle* weights_sum_i_th_ptr = nullptr;
  TensorHandle weights_sum_i_th;
  if (int8_weights.weights_sum_i) {
    weights_sum_i_th = AddExternalTensor(int8_weights.weights_sum_i);
    weights_sum_i_th_ptr = &weights_sum_i_th;
  }

  const std::string scale_name =
      absl::StrCat(base_name, ".weight_quantized_scale");
  const int quantization_i_group_size =
      loader->GetQuantizationIGroupSize(w_name, weights_shape);
  const int scale_i_ch = weights_shape.i / quantization_i_group_size;
  const OHWI scale_zp_shape = OHWI(weights_shape.o, 1, 1, scale_i_ch);
  auto status_or_scale = loader->LoadScale(scale_name, scale_zp_shape);
  ABSL_CHECK_OK(status_or_scale);
  TensorHandle scale_th = AddExternalTensor(status_or_scale.value());

  TensorHandle* zp_th_ptr = nullptr;
  TensorHandle zp_th;
  const std::string zp_name = absl::StrCat(base_name, ".weight_quantized_zp");
  if (loader->HasTensor(zp_name)) {
    auto status_or_zp =
        loader->LoadZeroPoint(zp_name, scale_name, scale_zp_shape);
    if (status_or_zp.ok()) {
      zp_th = AddExternalTensor(status_or_zp.value());
      zp_th_ptr = &zp_th;
    }
  }

  return CreateExternalWeights(weights_th, weights_desc, weights_shape,
                               scale_zp_shape, &scale_th, zp_th_ptr,
                               weights_sum_i_th_ptr);
}

GpuModelBuilder::Weights LlmWeightsLoader::GetWeightsFloat(
    const std::string& w_name, const OHWI& weights_shape) const {
  DataType float_type = data_type();
  auto weights_desc =
      builder.GetFullyConnectedWeightsDesc(float_type, weights_shape);
  auto status_or_weights = loader->LoadWeights(
      w_name, weights_desc, weights_shape, /*swap_dims=*/kSwapDims);
  ABSL_CHECK_OK(status_or_weights);
  TensorHandle weights_th = AddExternalTensor(status_or_weights.value());

  return CreateExternalWeights(weights_th, weights_desc, weights_shape);
}

WeightsWithPrecision LlmWeightsLoader::GetWeights(
    const std::string& name, const OHWI& weights_shape) const {
  const std::string w_name = absl::StrCat(name, kWeightSuffix);
  const int weights_count = weights_shape.DimensionsProduct();

  DataType weight_type = DataType::kUnknown;
  if (loader != nullptr) {
    weight_type = loader->GetDataTypeForWeights(w_name, weights_count);
  }

  GpuModelBuilder::Weights weights;
  if (weight_type == DataType::kInt4) {
    weights = GetWeightsInt4(w_name, name, weights_shape);
  } else if (weight_type == DataType::kInt8) {
    weights = GetWeightsInt8(w_name, name, weights_shape);
  } else {
    weights = GetWeightsFloat(w_name, weights_shape);
  }
  return {weights, weight_type};
}

}  // namespace ml_drift
