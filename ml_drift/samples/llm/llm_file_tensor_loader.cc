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

#include "ml_drift/samples/llm/llm_file_tensor_loader.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
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
namespace {

size_t GetFileSize(const std::string& file_path) {
  std::ifstream ifstr(file_path, std::ios::binary);
  if (!ifstr) {
    return 0;
  }
  const auto begin = ifstr.tellg();
  ifstr.seekg(0, std::ios::end);
  const auto end = ifstr.tellg();
  return end - begin;
}

ml_drift::Tensor<OHWI, DataType::kFloat32> ReadScaleOrZeroPointInt(
    const std::string& tensor_name, const ml_drift::OHWI& shape) {
  ml_drift::Tensor<OHWI, DataType::kFloat32> scale;
  scale.shape = shape;
  scale.data.resize(shape.DimensionsProduct());
  const int scale_size = GetFileSize(tensor_name);
  if (scale_size < 4) {
    ABSL_LOG(ERROR) << "Scale file size is less than 4 bytes: " << tensor_name
                    << std::endl;
    scale.data.clear();
    return scale;
  }
  const int bytes_size = (scale_size - 4) / shape.DimensionsProduct();
  std::ifstream ifstr(tensor_name, std::ios::binary);
  if (!ifstr) {
    ABSL_LOG(ERROR) << "Cannot open file! " << tensor_name << std::endl;
    scale.data.clear();
    return scale;
  }
  int32_t group_size = 0;
  ifstr.read((char*)&group_size, sizeof(int32_t));

  if (bytes_size == 1) {
    std::vector<int8_t> result(shape.DimensionsProduct());
    ifstr.read((char*)result.data(), sizeof(int8_t) * result.size());
    for (int i = 0; i < shape.DimensionsProduct(); ++i) {
      scale.data[i] = result[i] / 1024.0f;
    }
  } else if (bytes_size == 2) {
    std::vector<int16_t> result(shape.DimensionsProduct());
    ifstr.read((char*)result.data(), sizeof(int16_t) * result.size());
    for (int i = 0; i < shape.DimensionsProduct(); ++i) {
      scale.data[i] = result[i] / 1024.0f;
    }
  } else {
    ABSL_LOG(ERROR) << "Unsupported scale file size: " << bytes_size << " "
                    << tensor_name << std::endl;
    scale.data.clear();
    return scale;
  }
  return scale;
}

std::vector<float> LoadFloat32File(const std::string& file_name, int count) {
  std::vector<float> result(count);
  std::ifstream ifstr(file_name, std::ios::binary);
  if (!ifstr) {
    ABSL_LOG(ERROR) << "Cannot open file! " << file_name << std::endl;
    return {};
  }
  ifstr.read((char*)result.data(), sizeof(float) * count);
  return result;
}

bool IsGroupedQuantization(const std::string& tensor_name,
                           const ml_drift::OHWI& shape) {
  const size_t scale_size = GetFileSize(tensor_name);
  if (scale_size < 4) {
    return false;
  }
  std::ifstream ifstr(tensor_name, std::ios::binary);
  if (!ifstr) {
    return false;
  }
  int32_t group_size = 0;
  ifstr.read(reinterpret_cast<char*>(&group_size), sizeof(int32_t));

  if (group_size > 0 && group_size < 65536) {
    const size_t remaining_bytes = scale_size - 4;
    if (remaining_bytes == shape.DimensionsProduct() ||
        remaining_bytes == shape.DimensionsProduct() * 2) {
      return true;
    }
  }
  return false;
}

ml_drift::Tensor<OHWI, DataType::kFloat32> LoadScaleOrZeroPoint(
    const std::string& tensor_name, const ml_drift::OHWI& shape) {
  if (IsGroupedQuantization(tensor_name, shape)) {
    // grouped quantization, int8/16 values
    return ReadScaleOrZeroPointInt(tensor_name, shape);
  } else {
    // linear quantization, fp32 values
    ml_drift::Tensor<OHWI, DataType::kFloat32> scale;
    scale.shape = shape;
    scale.data = LoadFloat32File(tensor_name, shape.DimensionsProduct());
    return scale;
  }
}

}  // namespace

LlmFileTensorLoader::LlmFileTensorLoader(const std::string& weights_directory)
    : weights_directory_(weights_directory) {}

std::vector<float> LlmFileTensorLoader::LoadFloat32(const std::string& name,
                                                    int size) {
  const std::string filepath = absl::StrCat(weights_directory_, name);
  std::ifstream ifs(filepath, std::ios::binary);
  if (!ifs) {
    ABSL_LOG(ERROR) << "Cannot open file! " << filepath << std::endl;
    return {};
  }
  std::vector<float> data(size);
  ifs.read(reinterpret_cast<char*>(data.data()), size * sizeof(float));
  return data;
}

absl::StatusOr<ml_drift::GpuSpatialTensor*> LlmFileTensorLoader::LoadWeights(
    const std::string& name, const WeightsDescription& weights_desc,
    const OHWI& shape, bool swap_dims) {
  auto data_weights = LoadFloat32(name, shape.DimensionsProduct());
  if (data_weights.empty()) {
    return absl::NotFoundError(absl::StrCat("Failed to load weights: ", name));
  }

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
  TensorDescriptor weights_td;
  weights_td = TensorDescriptor(weights_desc.type, TensorStorageType::kBuffer,
                                Layout::kLinear);
  weights_td.SetBHWCShape(BHWC(1, 1, 1, elements_count));
  weights_td.SetData(std::move(data));
  return create_tensor_fn_(weights_td);
}

std::vector<int8_t> LlmFileTensorLoader::LoadInt8(
    const std::string& tensor_name, int count) {
  const std::string path = absl::StrCat(weights_directory_, tensor_name);
  std::vector<int8_t> result(count);
  std::ifstream ifstr(path, std::ios::binary);
  if (!ifstr) {
    return {};
  }
  ifstr.read(reinterpret_cast<char*>(result.data()), sizeof(int8_t) * count);
  return result;
}

std::vector<int8_t> LlmFileTensorLoader::LoadInt4AsInt8(
    const std::string& tensor_name, int count) {
  const std::string path = absl::StrCat(weights_directory_, tensor_name);
  if (count % 8 != 0) {
    return {};
  }
  std::vector<int32_t> data_i32(count / 8);
  std::ifstream ifstr(path, std::ios::binary);
  if (!ifstr) {
    return {};
  }
  ifstr.read(reinterpret_cast<char*>(data_i32.data()),
             sizeof(int32_t) * data_i32.size());

  std::vector<int8_t> data_i8(count);
  for (int i = 0; i < data_i32.size(); ++i) {
    std::vector<int8_t> values(8);
    int value = data_i32[i];
    for (int j = 0; j < 8; ++j) {
      int tmp = (value + 8) & 15;
      values[j] = tmp - 8;
      value = (value >> 4);
    }
    for (int j = 0; j < 8; ++j) {
      data_i8[i * 8 + j] = values[j];
    }
  }
  return data_i8;
}

std::vector<uint8_t> LlmFileTensorLoader::LoadInt4Packed(
    const std::string& tensor_name, int count) {
  const std::string path = absl::StrCat(weights_directory_, tensor_name);
  if (count % 2 != 0) {
    return {};
  }
  std::vector<uint8_t> data_ui8(count / 2);
  std::ifstream ifstr(path, std::ios::binary);
  if (!ifstr) {
    return {};
  }
  ifstr.read(reinterpret_cast<char*>(&data_ui8[0]),
             sizeof(uint8_t) * data_ui8.size());

  return data_ui8;
}

absl::StatusOr<LlmTensorLoader::Int8Weights>
LlmFileTensorLoader::LoadInt8Weights(const std::string& tensor_name,
                                     const WeightsDescription& weights_desc,
                                     const OHWI& shape, bool swap_dims) {
  const std::vector<int8_t> datai8 =
      LoadInt8(tensor_name, shape.DimensionsProduct());
  if (datai8.empty()) {
    return absl::NotFoundError(
        absl::StrCat("Failed to load weights: ", tensor_name));
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
  ABSL_ASSIGN_OR_RETURN(int8_weights.weights, create_tensor_fn_(weights_i8_td));
  auto weights_sum_i = GetWeightsAccumulatedInputChannels(weights_i8);

  TensorDescriptor weights_sum_i_td(
      DataType::kInt32, TensorStorageType::kBuffer, Layout::kLinear);
  weights_sum_i_td.SetBHWCShape(BHWC(1, 1, 1, weights_sum_i.shape.v));
  weights_sum_i_td.UploadData(weights_sum_i.data.data());

  ABSL_ASSIGN_OR_RETURN(int8_weights.weights_sum_i,
                        create_tensor_fn_(weights_sum_i_td));
  return int8_weights;
}

absl::StatusOr<LlmTensorLoader::Int4Weights>
LlmFileTensorLoader::LoadInt4Weights(const std::string& tensor_name,
                                     const WeightsDescription& weights_desc,
                                     const OHWI& shape, bool swap_dims) {
  if (weights_desc.layout == WeightsLayout::kOSpatialIOGroupI4O4 ||
      weights_desc.layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    return LoadInt4WeightsFast(tensor_name, weights_desc, shape, swap_dims);
  }

  const std::vector<int8_t> datai8 =
      LoadInt4AsInt8(tensor_name, shape.DimensionsProduct());
  if (datai8.empty()) {
    return absl::NotFoundError(
        absl::StrCat("Failed to load weights: ", tensor_name));
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
  ABSL_ASSIGN_OR_RETURN(int4_weights.weights, create_tensor_fn_(weights_i4_td));
  auto weights_sum_i = GetWeightsAccumulatedInputChannels(weights_i8);

  TensorDescriptor weights_sum_i_td(
      DataType::kInt32, TensorStorageType::kBuffer, Layout::kLinear);
  weights_sum_i_td.SetBHWCShape(BHWC(1, 1, 1, weights_sum_i.shape.v));
  weights_sum_i_td.UploadData(weights_sum_i.data.data());

  ABSL_ASSIGN_OR_RETURN(int4_weights.weights_sum_i,
                        create_tensor_fn_(weights_sum_i_td));
  return int4_weights;
}

absl::StatusOr<LlmTensorLoader::Int4Weights>
LlmFileTensorLoader::LoadInt4WeightsFast(const std::string& tensor_name,
                                         const WeightsDescription& weights_desc,
                                         const OHWI& shape, bool swap_dims) {
  const std::vector<uint8_t> vector_u8 =
      LoadInt4Packed(tensor_name, shape.DimensionsProduct());
  if (vector_u8.empty()) {
    return absl::NotFoundError(
        absl::StrCat("Failed to load weights: ", tensor_name));
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
  ABSL_ASSIGN_OR_RETURN(int4_weights.weights, create_tensor_fn_(weights_i8_td));

  TensorDescriptor weights_sum_i_td(
      DataType::kInt32, TensorStorageType::kBuffer, Layout::kLinear);
  weights_sum_i_td.SetBHWCShape(BHWC(1, 1, 1, weights_sum_i.shape.v));
  weights_sum_i_td.UploadData(weights_sum_i.data.data());

  ABSL_ASSIGN_OR_RETURN(int4_weights.weights_sum_i,
                        create_tensor_fn_(weights_sum_i_td));
  return int4_weights;
}

int LlmFileTensorLoader::GetQuantizationIGroupSize(
    const std::string& weights_name,
    const ml_drift::OHWI& weights_shape) const {
  auto it = group_size_cache_.find(weights_name);
  if (it != group_size_cache_.end()) {
    return it->second;
  }

  const std::string base_name = weights_name.substr(0, weights_name.rfind('.'));
  const std::string scale_name =
      absl::StrCat(base_name, ".weight_quantized_scale");
  const std::string scale_path = absl::StrCat(weights_directory_, scale_name);
  const int scale_size = GetFileSize(scale_path);
  if (scale_size < 4) {
    group_size_cache_[weights_name] = weights_shape.i;
    return weights_shape.i;
  }

  // 1. Check if the scale file is a raw float32 scale file (without header).
  if (scale_size % 4 == 0) {
    const int expected_elements = scale_size / 4;
    if (expected_elements > 0 && expected_elements % weights_shape.o == 0) {
      const int expected_scale_i_ch = expected_elements / weights_shape.o;
      if (weights_shape.i % expected_scale_i_ch == 0) {
        int group_size = weights_shape.i / expected_scale_i_ch;
        group_size_cache_[weights_name] = group_size;
        return group_size;
      }
    }
  }

  // 2. Check if it has a 4-byte group size header followed by int8 or int16
  // scales.
  std::ifstream ifstr(scale_path, std::ios::binary);
  if (!ifstr) {
    group_size_cache_[weights_name] = weights_shape.i;
    return weights_shape.i;
  }
  int32_t group_size = 0;
  ifstr.read(reinterpret_cast<char*>(&group_size), sizeof(int32_t));

  if (group_size > 0 && group_size <= weights_shape.i &&
      weights_shape.i % group_size == 0) {
    const int expected_elements =
        weights_shape.o * (weights_shape.i / group_size);
    if (scale_size == 4 + expected_elements ||
        scale_size == 4 + expected_elements * 2) {
      group_size_cache_[weights_name] = group_size;
      return group_size;
    }
  }
  group_size_cache_[weights_name] = weights_shape.i;
  return weights_shape.i;
}

absl::StatusOr<ml_drift::GpuSpatialTensor*> LlmFileTensorLoader::LoadScale(
    const std::string& tensor_name, const ::ml_drift::OHWI& shape) {
  const std::string path = absl::StrCat(weights_directory_, tensor_name);
  auto scale = LoadScaleOrZeroPoint(path, shape);

  if (scale.data.empty()) {
    return absl::NotFoundError(
        absl::StrCat("Failed to load scale: ", tensor_name));
  }

  TensorDescriptor td =
      ml_drift::ScaleOrZeroPointToTensorDesc(gpu_info_, scale, data_type_);
  return create_tensor_fn_(td);
}

absl::StatusOr<ml_drift::GpuSpatialTensor*> LlmFileTensorLoader::LoadZeroPoint(
    const std::string& zero_point_name, const std::string& scale_name,
    const ::ml_drift::OHWI& shape) {
  ml_drift::Tensor<OHWI, DataType::kFloat32> zero_point;
  const std::string zp_path = absl::StrCat(weights_directory_, zero_point_name);
  const std::string scale_path = absl::StrCat(weights_directory_, scale_name);

  std::ifstream ifstr(zp_path, std::ios::binary);
  if (ifstr) {
    ifstr.close();
    zero_point = LoadScaleOrZeroPoint(zp_path, shape);
    auto scale = LoadScaleOrZeroPoint(scale_path, shape);
    for (int i = 0; i < zero_point.data.size(); ++i) {
      if (scale.data[i] == 0.0f) {
        zero_point.data[i] = 0.0f;
      } else {
        zero_point.data[i] /= scale.data[i];
      }
    }
  } else {
    zero_point.shape = shape;
    zero_point.data.resize(shape.DimensionsProduct());
  }

  TensorDescriptor td = ml_drift::ScaleOrZeroPointToTensorDesc(
      gpu_info_, zero_point, data_type_);
  return create_tensor_fn_(td);
}

bool LlmFileTensorLoader::HasTensor(const std::string& name) const {
  auto it = has_tensor_cache_.find(name);
  if (it != has_tensor_cache_.end()) {
    return it->second;
  }
  const std::string path = absl::StrCat(weights_directory_, name);
  std::ifstream ifstr(path, std::ios::binary);
  bool has = ifstr.good();
  has_tensor_cache_[name] = has;
  return has;
}

int LlmFileTensorLoader::GetTensorElementSizeInBits(
    const std::string& tensor_name, int count) {
  if (count <= 0) return 0;
  size_t file_size = 0;
  auto it = file_size_cache_.find(tensor_name);
  if (it != file_size_cache_.end()) {
    file_size = it->second;
  } else {
    file_size = GetFileSize(absl::StrCat(weights_directory_, tensor_name));
    file_size_cache_[tensor_name] = file_size;
  }
  return static_cast<int>((file_size * 8) / count);
}

}  // namespace ml_drift
