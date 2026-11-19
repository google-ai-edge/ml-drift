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

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/samples/llm/llm_tensor_loader.h"

namespace ml_drift {
namespace {

struct TensorInfo {
  std::string name;
  std::string type;
  std::vector<int> dims;
};

TensorInfo ParseLine(const std::string& line) {
  TensorInfo info;
  std::vector<std::string> tokens = absl::StrSplit(line, '.');
  if (tokens.size() < 4) return info;

  if (tokens[0] != "mdl_vars") return info;

  info.dims = {};
  std::string dims_str = tokens.back();
  std::vector<std::string> dim_tokens = absl::StrSplit(dims_str, '_');
  for (const auto& d : dim_tokens) {
    int val;
    if (absl::SimpleAtoi(d, &val)) {
      info.dims.push_back(val);
    }
  }

  info.type = tokens[tokens.size() - 2];

  std::vector<std::string> name_tokens(tokens.begin() + 1, tokens.end() - 2);
  info.name = absl::StrJoin(name_tokens, ".");

  return info;
}

TEST(LlmFileTensorLoaderTest, LoadAndPrint) {
  std::string test_dir = testing::TempDir();
  std::string layer_info_path = test_dir + "/layer_info.txt";

  // Create fake layer_info.txt file.
  std::ofstream ofs(layer_info_path);
  ofs << "mdl_vars.model.layers.9.mlp.gate_proj.weight_quantized_scale"
         ".float32.5_2\n";
  ofs << "mdl_vars.model.layers.9.mlp.gate_proj.weight.int8.5_2\n";
  ofs << "mdl_vars.model.layers.9.mlp.gate_proj.weight_int4.int4.8_1\n";
  ofs.close();

  // Create dummy files
  // float32 file: 10 floats = 40 bytes
  std::ofstream f_float(
      test_dir + "/model.layers.9.mlp.gate_proj.weight_quantized_scale",
      std::ios::binary);
  std::vector<float> float_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                   6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
  f_float.write(reinterpret_cast<const char*>(float_data.data()),
                float_data.size() * sizeof(float));
  f_float.close();

  // int8 file: 10 bytes
  std::ofstream f_int8(test_dir + "/model.layers.9.mlp.gate_proj.weight",
                       std::ios::binary);
  std::vector<int8_t> int8_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  f_int8.write(reinterpret_cast<const char*>(int8_data.data()),
               int8_data.size());
  f_int8.close();

  // int4 file: 8 elements, packed as 4 bytes if using LoadInt4Packed,
  // or 1 int32 if using LoadInt4AsInt8
  // LoadInt4AsInt8 expects count % 8 == 0 and reads count / 8 int32s.
  // Here count = 8, so it reads 1 int32.
  std::ofstream f_int4(test_dir + "/model.layers.9.mlp.gate_proj.weight_int4",
                       std::ios::binary);
  int32_t int4_packed_data = 0x12345678;  // dummy packed data
  f_int4.write(reinterpret_cast<const char*>(&int4_packed_data),
               sizeof(int32_t));
  f_int4.close();

  std::cout << "Loading from: " << layer_info_path << std::endl;
  LlmFileTensorLoader loader(test_dir + "/");

  std::vector<uint8_t> captured_data;
  loader.SetCreateTensorFn([&captured_data](const TensorDescriptor& desc)
                               -> absl::StatusOr<GpuSpatialTensor*> {
    if (desc.GetDataType() != DataType::kInt32) {
      auto span = desc.GetData();
      captured_data.assign(span.begin(), span.end());
    }
    return nullptr;
  });

  std::ifstream ifs(layer_info_path);
  ABSL_CHECK(ifs.is_open());

  std::string line;
  while (std::getline(ifs, line)) {
    // std::cout << "Line: " << line << std::endl;
    if (line.empty()) continue;
    TensorInfo info = ParseLine(line);
    std::cout << "Tensor: " << info.name << " Dims: ";
    for (int d : info.dims) std::cout << d << " ";
    std::cout << "Type: " << info.type << " Values: ";

    int size = 1;
    for (int d : info.dims) size *= d;

    if (info.type == "float32") {
      std::vector<float> vals = loader.LoadFloat32(info.name, size);
      ABSL_CHECK_GT(vals.size(), 0);

      for (int i = 0; i < std::min(5, (int)vals.size()); ++i) {
        std::cout << vals[i] << " ";
      }
    } else if (info.type == "int8") {
      WeightsDescription weights_desc;
      weights_desc.type = DataType::kUint8;  // Use UINT8 as seen in test
      weights_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
      weights_desc.output_group_size = 1;
      OHWI shape;
      if (info.dims.size() >= 2) {
        shape = OHWI(info.dims[0], 1, 1, info.dims[1]);
      } else if (info.dims.size() == 1) {
        shape = OHWI(info.dims[0], 1, 1, 1);
      }

      captured_data.clear();
      auto status =
          loader.LoadInt8Weights(info.name, weights_desc, shape, false);
      ABSL_CHECK_OK(status);

      const int8_t* vals =
          reinterpret_cast<const int8_t*>(captured_data.data());
      int n = captured_data.size() / sizeof(int8_t);
      for (int i = 0; i < std::min(5, n); ++i) {
        std::cout << (int)vals[i] << " ";
      }
    } else if (info.type == "int4") {
      WeightsDescription weights_desc;
      weights_desc.type = DataType::kUint4;  // Use UINT4 as seen in test
      weights_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
      weights_desc.output_group_size = 1;
      OHWI shape;
      if (info.dims.size() >= 2) {
        shape = OHWI(info.dims[0], 1, 1, info.dims[1]);
      } else if (info.dims.size() == 1) {
        shape = OHWI(info.dims[0], 1, 1, 1);
      }

      captured_data.clear();
      auto status =
          loader.LoadInt4Weights(info.name, weights_desc, shape, false);
      ABSL_CHECK_OK(status);

      // LoadInt4Weights with kOSpatialIOGroupI4O4 calls LoadInt4WeightsFast
      // which might store packed data.
      const uint8_t* vals = captured_data.data();
      int n = captured_data.size();
      for (int i = 0; i < std::min(5, n); ++i) {
        std::cout << (int)vals[i] << " ";
      }
    }
    std::cout << std::endl;
  }
}

TEST(LlmFileTensorLoaderTest, GetQuantizationIGroupSizeCaching) {
  std::string test_dir = testing::TempDir();
  std::string scale_file =
      test_dir + "/model.layers.0.mlp.gate_proj.weight_quantized_scale";

  // Create a float32 scale file with 10 floats (40 bytes).
  // weights_shape = OHWI(5, 1, 1, 8), expected_elements = 10, scale_i_ch = 10 /
  // 5 = 2. Group size = 8 / 2 = 4.
  {
    std::ofstream f_float(scale_file, std::ios::binary);
    std::vector<float> float_data(10, 1.0f);
    f_float.write(reinterpret_cast<const char*>(float_data.data()),
                  float_data.size() * sizeof(float));
  }

  LlmFileTensorLoader loader(test_dir + "/");
  OHWI weights_shape(5, 1, 1, 8);
  std::string weights_name = "model.layers.0.mlp.gate_proj.weight";

  int group_size =
      loader.GetQuantizationIGroupSize(weights_name, weights_shape);
  EXPECT_EQ(group_size, 4);

  // Remove the scale file from disk to verify caching prevents disk access.
  std::remove(scale_file.c_str());

  // Subsequent call should return cached value.
  int cached_group_size =
      loader.GetQuantizationIGroupSize(weights_name, weights_shape);
  EXPECT_EQ(cached_group_size, 4);
}

TEST(LlmTensorLoaderTest, CachingLoaderGetQuantizationIGroupSize) {
  std::string test_dir = testing::TempDir();
  std::string scale_file =
      test_dir + "/model.layers.1.mlp.gate_proj.weight_quantized_scale";

  // Write a scale file with 4-byte header (group_size = 16) followed by int8
  // scales. weights_shape = OHWI(4, 1, 1, 32). expected_elements = 4 * (32 /
  // 16) = 8 elements. Total bytes = 4 + 8 = 12.
  {
    std::ofstream f_scale(scale_file, std::ios::binary);
    int32_t header_group_size = 16;
    f_scale.write(reinterpret_cast<const char*>(&header_group_size),
                  sizeof(int32_t));
    std::vector<int8_t> int8_data(8, 1);
    f_scale.write(reinterpret_cast<const char*>(int8_data.data()),
                  int8_data.size());
  }

  auto backend = std::make_unique<LlmFileTensorLoader>(test_dir + "/");
  auto caching_loader = LlmTensorLoader::MakeCaching(std::move(backend));

  OHWI weights_shape(4, 1, 1, 32);
  std::string weights_name = "model.layers.1.mlp.gate_proj.weight";

  int group_size =
      caching_loader->GetQuantizationIGroupSize(weights_name, weights_shape);
  EXPECT_EQ(group_size, 16);

  // Remove the scale file from disk to verify caching loader.
  std::remove(scale_file.c_str());

  int cached_group_size =
      caching_loader->GetQuantizationIGroupSize(weights_name, weights_shape);
  EXPECT_EQ(cached_group_size, 16);
}

}  // namespace
}  // namespace ml_drift
