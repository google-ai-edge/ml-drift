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

#include "ml_drift/samples/llm/llm_dummy_tensor_loader.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/samples/llm/llm_tensor_loader.h"

namespace ml_drift {

LlmDummyTensorLoader::LlmDummyTensorLoader(int quantization_i_group_size,
                                           int default_element_size_in_bits)
    : quantization_i_group_size_(quantization_i_group_size),
      default_element_size_in_bits_(default_element_size_in_bits) {
  data_type_ = DataType::kFloat32;
}

int LlmDummyTensorLoader::GetTensorElementSizeInBits(
    const std::string& tensor_name, int count) {
  auto it = element_size_map_.find(tensor_name);
  if (it != element_size_map_.end()) {
    return it->second;
  }
  return default_element_size_in_bits_;
}

absl::StatusOr<GpuSpatialTensor*> LlmDummyTensorLoader::CreateTensorInternal(
    const TensorDescriptor& td) {
  if (create_tensor_fn_) {
    return create_tensor_fn_(td);
  }
  owned_tensors_.push_back(std::make_unique<DummyGpuSpatialTensor>(td));
  return owned_tensors_.back().get();
}

bool LlmDummyTensorLoader::HasTensor(const std::string& name) const {
  // Accept any name to load.
  return true;
}

std::vector<float> LlmDummyTensorLoader::LoadFloat32(const std::string& name,
                                                     int size) {
  std::vector<float> data(size);
  for (int i = 0; i < size; ++i) {
    // Generate normalized non-zero float values in range [0.01, 1.0].
    data[i] = static_cast<float>((i % 100) + 1) / 100.0f;
  }
  return data;
}

absl::StatusOr<ml_drift::GpuSpatialTensor*> LlmDummyTensorLoader::LoadScale(
    const std::string& tensor_name, const ::ml_drift::OHWI& shape) {
  ml_drift::Tensor<OHWI, DataType::kFloat32> scale;
  scale.shape = shape;
  const int count = shape.DimensionsProduct();
  scale.data.resize(count);
  for (int i = 0; i < count; ++i) {
    // Generate normalized non-zero scale factors.
    scale.data[i] = static_cast<float>((i % 50) + 1) / 100.0f;
  }

  DataType effective_data_type =
      data_type_ == DataType::kUnknown ? DataType::kFloat32 : data_type_;
  TensorDescriptor td = ml_drift::ScaleOrZeroPointToTensorDesc(
      gpu_info_, scale, effective_data_type);
  return CreateTensorInternal(td);
}

absl::StatusOr<ml_drift::GpuSpatialTensor*> LlmDummyTensorLoader::LoadZeroPoint(
    const std::string& zero_point_name, const std::string& scale_name,
    const ::ml_drift::OHWI& shape) {
  ml_drift::Tensor<OHWI, DataType::kFloat32> zero_point;
  zero_point.shape = shape;
  zero_point.data.resize(shape.DimensionsProduct(), 0.0f);

  DataType effective_data_type =
      data_type_ == DataType::kUnknown ? DataType::kFloat32 : data_type_;
  TensorDescriptor td = ml_drift::ScaleOrZeroPointToTensorDesc(
      gpu_info_, zero_point, effective_data_type);
  return CreateTensorInternal(td);
}

absl::StatusOr<ml_drift::GpuSpatialTensor*> LlmDummyTensorLoader::LoadWeights(
    const std::string& name, const WeightsDescription& weights_desc,
    const OHWI& shape, bool swap_dims) {
  auto data_weights = LoadFloat32(name, shape.DimensionsProduct());

  ml_drift::Tensor<OHWI, DataType::kFloat32> weights;
  weights.shape = shape;
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < shape.i; ++i) {
    for (int o = 0; o < shape.o; ++o) {
      if (swap_dims) {
        weights.data[shape.LinearIndex({o, 0, 0, i})] =
            data_weights[i * shape.o + o];
      } else {
        weights.data[shape.LinearIndex({o, 0, 0, i})] =
            data_weights[o * shape.i + i];
      }
    }
  }

  const int elements_count =
      GetTotalElementsCountForLayout(weights_desc, shape);
  std::vector<uint8_t> data(elements_count * SizeOf(weights_desc.type));
  RearrangeWeights(weights, weights_desc, absl::MakeSpan(data));
  TensorDescriptor weights_td(weights_desc.type, TensorStorageType::kBuffer,
                              Layout::kLinear);
  weights_td.SetBHWCShape(BHWC(1, 1, 1, elements_count));
  weights_td.SetData(std::move(data));
  return CreateTensorInternal(weights_td);
}

absl::StatusOr<LlmTensorLoader::Int8Weights>
LlmDummyTensorLoader::LoadInt8Weights(const std::string& tensor_name,
                                      const WeightsDescription& weights_desc,
                                      const OHWI& shape, bool swap_dims) {
  const int count = shape.DimensionsProduct();
  std::vector<int8_t> datai8(count);
  for (int i = 0; i < count; ++i) {
    // Generate normalized non-zero int8 values in range [1, 127].
    datai8[i] = static_cast<int8_t>((i % 127) + 1);
  }

  ml_drift::Tensor<OHWI, DataType::kInt8> weights_i8;
  weights_i8.shape = shape;
  if (swap_dims) {
    weights_i8.data.resize(weights_i8.shape.DimensionsProduct());
    for (int o = 0; o < weights_i8.shape.o; ++o) {
      for (int i = 0; i < weights_i8.shape.i; ++i) {
        weights_i8.data[weights_i8.shape.LinearIndex({o, 0, 0, i})] =
            datai8[i * weights_i8.shape.o + o];
      }
    }
  } else {
    weights_i8.data = std::move(datai8);
  }
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(int8_t));
  TensorDescriptor weights_i8_td =
      ml_drift::GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc);

  LlmTensorLoader::Int8Weights int8_weights;
  ABSL_ASSIGN_OR_RETURN(int8_weights.weights,
                        CreateTensorInternal(weights_i8_td));
  auto weights_sum_i = GetWeightsAccumulatedInputChannels(weights_i8);

  TensorDescriptor weights_sum_i_td(
      DataType::kInt32, TensorStorageType::kBuffer, Layout::kLinear);
  weights_sum_i_td.SetBHWCShape(BHWC(1, 1, 1, weights_sum_i.shape.v));
  weights_sum_i_td.UploadData(weights_sum_i.data.data());

  ABSL_ASSIGN_OR_RETURN(int8_weights.weights_sum_i,
                        CreateTensorInternal(weights_sum_i_td));
  return int8_weights;
}

absl::StatusOr<LlmTensorLoader::Int4Weights>
LlmDummyTensorLoader::LoadInt4Weights(const std::string& tensor_name,
                                      const WeightsDescription& weights_desc,
                                      const OHWI& shape, bool swap_dims) {
  if (weights_desc.layout == WeightsLayout::kOSpatialIOGroupI4O4 ||
      weights_desc.layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    return LoadInt4WeightsFast(tensor_name, weights_desc, shape, swap_dims);
  }

  const int count = shape.DimensionsProduct();
  std::vector<int8_t> datai8(count);
  for (int i = 0; i < count; ++i) {
    // Generate normalized non-zero int4 values packed as int8 in [-7, 7]
    // (excluding 0).
    int val = (i % 7) + 1;
    datai8[i] = static_cast<int8_t>(val);
  }

  ml_drift::Tensor<OHWI, DataType::kInt8> weights_i8;
  weights_i8.shape = shape;
  if (swap_dims) {
    weights_i8.data.resize(weights_i8.shape.DimensionsProduct());
    for (int o = 0; o < weights_i8.shape.o; ++o) {
      for (int i = 0; i < weights_i8.shape.i; ++i) {
        weights_i8.data[weights_i8.shape.LinearIndex({o, 0, 0, i})] =
            datai8[i * weights_i8.shape.o + o];
      }
    }
  } else {
    weights_i8.data = std::move(datai8);
  }
  TensorDescriptor weights_i4_td =
      ml_drift::GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc);

  Int4Weights int4_weights;
  ABSL_ASSIGN_OR_RETURN(int4_weights.weights,
                        CreateTensorInternal(weights_i4_td));
  auto weights_sum_i = GetWeightsAccumulatedInputChannels(weights_i8);

  TensorDescriptor weights_sum_i_td(
      DataType::kInt32, TensorStorageType::kBuffer, Layout::kLinear);
  weights_sum_i_td.SetBHWCShape(BHWC(1, 1, 1, weights_sum_i.shape.v));
  weights_sum_i_td.UploadData(weights_sum_i.data.data());

  ABSL_ASSIGN_OR_RETURN(int4_weights.weights_sum_i,
                        CreateTensorInternal(weights_sum_i_td));
  return int4_weights;
}

absl::StatusOr<LlmTensorLoader::Int4Weights>
LlmDummyTensorLoader::LoadInt4WeightsFast(
    const std::string& tensor_name, const WeightsDescription& weights_desc,
    const OHWI& shape, bool swap_dims) {
  const int count = shape.DimensionsProduct() / 2;
  std::vector<uint8_t> vector_u8(count);
  for (int i = 0; i < count; ++i) {
    // Non-zero packed nibbles
    vector_u8[i] =
        static_cast<uint8_t>(((i % 7 + 1) << 4) | ((i % 5 + 1) & 0x0F));
  }

  Tensor<OHWI, DataType::kUint8> tensor_u8;
  tensor_u8.shape = shape;
  tensor_u8.data = std::move(vector_u8);
  std::vector<uint8_t> arranged_vec_u8(tensor_u8.shape.DimensionsProduct() / 2);
  Tensor<Linear, DataType::kInt32> weights_sum_i;
  weights_sum_i.shape = Linear(shape.o);
  weights_sum_i.data.resize(shape.o);
  ABSL_RETURN_IF_ERROR(RearrangeWeightsUInt4Packed(
      tensor_u8, weights_desc, absl::MakeSpan(arranged_vec_u8),
      absl::MakeSpan(weights_sum_i.data), 8u, swap_dims));
  TensorDescriptor weights_i8_td;
  if (weights_desc.IsLinearLayout()) {
    weights_i8_td = TensorDescriptor(
        weights_desc.type, TensorStorageType::kBuffer, Layout::kLinear);
    weights_i8_td.SetBHWDCShape(
        BHWDC(1, 1, 1, 1, shape.DimensionsProduct() / 2));
  } else {
    weights_i8_td = TensorDescriptor(
        DataType::kUint16, TensorStorageType::kTexture2D, Layout::kHW);
    uint2 tex_size = Get2dResourceSize(weights_desc, shape);
    tex_size.x /= 4;  // because we store 4 uint4 as one uint16
    weights_i8_td.SetBHWDCShape(BHWDC(1, tex_size.y, tex_size.x, 1, 4));
  }

  weights_i8_td.SetData(std::move(arranged_vec_u8));
  Int4Weights int4_weights;
  ABSL_ASSIGN_OR_RETURN(int4_weights.weights,
                        CreateTensorInternal(weights_i8_td));

  TensorDescriptor weights_sum_i_td(
      DataType::kInt32, TensorStorageType::kBuffer, Layout::kLinear);
  weights_sum_i_td.SetBHWCShape(BHWC(1, 1, 1, weights_sum_i.shape.v));
  weights_sum_i_td.UploadData(weights_sum_i.data.data());

  ABSL_ASSIGN_OR_RETURN(int4_weights.weights_sum_i,
                        CreateTensorInternal(weights_sum_i_td));
  return int4_weights;
}

int LlmDummyTensorLoader::GetQuantizationIGroupSize(
    const std::string& weights_name,
    const ml_drift::OHWI& weights_shape) const {
  if (quantization_i_group_size_ > 0 &&
      weights_shape.i % quantization_i_group_size_ == 0) {
    return quantization_i_group_size_;
  }
  return weights_shape.i;
}

}  // namespace ml_drift
