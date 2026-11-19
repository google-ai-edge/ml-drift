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

#include "ml_drift/common/kernels/special/dw3x3_conv8to8_dw3x3_conv8to8_add_conv8to8.h"

#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"

namespace ml_drift {
namespace {

std::string GenerateConvolutionCode(const OperationDef& op_def) {
  std::string c;
  c += R"(

#define CONV(R, SRC, i) \
  R += SRC.x * constants[i + 0]; \
  R += SRC.y * constants[i + 1]; \
  R += SRC.z * constants[i + 2]; \
  R += SRC.w * constants[i + 3];

MAIN_FUNCTION($0) {

  __constant Type* constants = args.weights.GetPtr();

  int X = get_global_id(0);
  int Y = get_global_id(1);

  if (X >= args.dst_0.Width() || Y >= args.dst_0.Height()) return;

  Type r4 = ucl::Init<Type>(0, 0, 0, 0);
  Type r5 = ucl::Init<Type>(0, 0, 0, 0);

  if (X < args.dst_0.Width() + 2) {
    Type r0 = constants[18];
    r0 += args.src_0.Read(X - 1, Y - 1, 0) * constants[0];
    r0 += args.src_0.Read(X - 0, Y - 1, 0) * constants[1];
    r0 += args.src_0.Read(X + 1, Y - 1, 0) * constants[2];
    r0 += args.src_0.Read(X - 1, Y - 0, 0) * constants[3];
    r0 += args.src_0.Read(X - 0, Y - 0, 0) * constants[4];
    r0 += args.src_0.Read(X + 1, Y - 0, 0) * constants[5];
    r0 += args.src_0.Read(X - 1, Y + 1, 0) * constants[6];
    r0 += args.src_0.Read(X - 0, Y + 1, 0) * constants[7];
    r0 += args.src_0.Read(X + 1, Y + 1, 0) * constants[8];

    Type r1 = constants[19];
    r1 += args.src_0.Read(X - 1, Y - 1, 1) * constants[9];
    r1 += args.src_0.Read(X - 0, Y - 1, 1) * constants[10];
    r1 += args.src_0.Read(X + 1, Y - 1, 1) * constants[11];
    r1 += args.src_0.Read(X - 1, Y - 0, 1) * constants[12];
    r1 += args.src_0.Read(X - 0, Y - 0, 1) * constants[13];
    r1 += args.src_0.Read(X + 1, Y - 0, 1) * constants[14];
    r1 += args.src_0.Read(X - 1, Y + 1, 1) * constants[15];
    r1 += args.src_0.Read(X - 0, Y + 1, 1) * constants[16];
    r1 += args.src_0.Read(X + 1, Y + 1, 1) * constants[17];

    CONV(r4, r0, 40);
    CONV(r5, r0, 44);

    CONV(r4, r1, 48);
    CONV(r5, r1, 52);

    r4 += constants[56];
    r5 += constants[57];
  }

  if (Y < args.dst_0.Height() + 2) {
    Type r2 = constants[38];
    r2 += args.src_1.Read(X - 1, Y - 1, 0) * constants[20];
    r2 += args.src_1.Read(X - 0, Y - 1, 0) * constants[21];
    r2 += args.src_1.Read(X + 1, Y - 1, 0) * constants[22];
    r2 += args.src_1.Read(X - 1, Y - 0, 0) * constants[23];
    r2 += args.src_1.Read(X - 0, Y - 0, 0) * constants[24];
    r2 += args.src_1.Read(X + 1, Y - 0, 0) * constants[25];
    r2 += args.src_1.Read(X - 1, Y + 1, 0) * constants[26];
    r2 += args.src_1.Read(X - 0, Y + 1, 0) * constants[27];
    r2 += args.src_1.Read(X + 1, Y + 1, 0) * constants[28];

    Type r3 = constants[39];
    r3 += args.src_1.Read(X - 1, Y - 1, 1) * constants[29];
    r3 += args.src_1.Read(X - 0, Y - 1, 1) * constants[30];
    r3 += args.src_1.Read(X + 1, Y - 1, 1) * constants[31];
    r3 += args.src_1.Read(X - 1, Y - 0, 1) * constants[32];
    r3 += args.src_1.Read(X - 0, Y - 0, 1) * constants[33];
    r3 += args.src_1.Read(X + 1, Y - 0, 1) * constants[34];
    r3 += args.src_1.Read(X - 1, Y + 1, 1) * constants[35];
    r3 += args.src_1.Read(X - 0, Y + 1, 1) * constants[36];
    r3 += args.src_1.Read(X + 1, Y + 1, 1) * constants[37];

    Type r6 = ucl::Init<Type>(0, 0, 0, 0);
    Type r7 = ucl::Init<Type>(0, 0, 0, 0);

    CONV(r6, r2, 58);
    CONV(r7, r2, 62);
    CONV(r6, r3, 66);
    CONV(r7, r3, 70);
    r6 += constants[74];
    r7 += constants[75];

    r4 += r6;
    r5 += r7;

    r4 = max(r4, ucl::Init<Type>(0.0)) + min(r4, ucl::Init<Type>(0.0)) * constants[76];
    r5 = max(r5, ucl::Init<Type>(0.0)) + min(r5, ucl::Init<Type>(0.0)) * constants[77];

    r6 = constants[78];
    r7 = constants[79];

    CONV(r6, r4, 80);
    CONV(r7, r4, 84);
    CONV(r6, r5, 88);
    CONV(r7, r5, 92);

    r6 = max(r6, ucl::Init<Type>(0.0)) + min(r6, ucl::Init<Type>(0.0)) * constants[96];
    r7 = max(r7, ucl::Init<Type>(0.0)) + min(r7, ucl::Init<Type>(0.0)) * constants[97];

    args.dst_0.Write(r6, X, Y, 0);
    args.dst_0.Write(r7, X, Y, 1);
  }
}
  )";
  absl::StrReplaceAll(
      {{"Type", ToUclDataType(op_def.dst_tensors[0].GetDataType(), 4)}}, &c);
  return c;
}

void UploadWeights(const std::vector<float>& constants, DataType data_type,
                   GPUOperation* op) {
  BufferDescriptor desc;
  desc.element_type = data_type;
  desc.element_size = 4;
  desc.memory_type = MemoryType::kConstant;
  desc.size = SizeOf(data_type) * constants.size();
  desc.data.resize(desc.size);

  if (data_type == DataType::kFloat32) {
    memcpy(desc.data.data(), constants.data(), desc.size);
  } else {
    half* gpu_data_half = reinterpret_cast<half*>(desc.data.data());
    for (int i = 0; i < constants.size(); ++i) {
      gpu_data_half[i] = constants[i];
    }
  }
  op->args_.AddObject("weights",
                      std::make_unique<BufferDescriptor>(std::move(desc)));
}
}  // namespace

GPUOperation CreateDW3x3Conv8To8DW3x3Conv8To8AddConv8To8(
    const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& dw_attr_0,
    const Convolution2DAttributes& conv8to8_0,
    const DepthwiseConvolution2DAttributes& dw_attr_1,
    const Convolution2DAttributes& conv8to8_1, const PReLUAttributes& prelu0,
    const Convolution2DAttributes& conv8to8_2, const PReLUAttributes& prelu1) {
  std::vector<float> constants;

  const auto& dw_0_weights = GetFloatWeights(dw_attr_0);
  for (int z = 0; z < 2; ++z) {
    for (int y = 0; y < 3; ++y) {
      for (int x = 0; x < 3; ++x) {
        for (int i = 0; i < 4; ++i) {
          const int f_index =
              dw_0_weights.shape.LinearIndex({0, y, x, z * 4 + i});
          constants.push_back(dw_0_weights.data[f_index]);
        }
      }
    }
  }

  for (int i = 0; i < 8; ++i) {
    constants.push_back(dw_attr_0.bias.data[i]);
  }

  const auto& dw_1_weights = GetFloatWeights(dw_attr_1);
  for (int z = 0; z < 2; ++z) {
    for (int y = 0; y < 3; ++y) {
      for (int x = 0; x < 3; ++x) {
        for (int i = 0; i < 4; ++i) {
          const int f_index =
              dw_1_weights.shape.LinearIndex({0, y, x, z * 4 + i});
          constants.push_back(dw_1_weights.data[f_index]);
        }
      }
    }
  }

  for (int i = 0; i < 8; ++i) {
    constants.push_back(dw_attr_1.bias.data[i]);
  }

  const auto& conv8to8_0_weights = GetFloatWeights(conv8to8_0);
  for (int s = 0; s < 2; ++s) {
    for (int d = 0; d < 2; ++d) {
      float4 filters[4];
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          const int src_ch = s * 4 + j;
          const int dst_ch = d * 4 + i;
          const int f_index =
              conv8to8_0_weights.shape.LinearIndex({dst_ch, 0, 0, src_ch});
          filters[i][j] = conv8to8_0_weights.data[f_index];
        }
      }
      float4 filters_new[4];
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          filters_new[i][j] = filters[j][i];
        }
      }
      for (int i = 0; i < 4; ++i) {
        constants.push_back(filters_new[i].x);
        constants.push_back(filters_new[i].y);
        constants.push_back(filters_new[i].z);
        constants.push_back(filters_new[i].w);
      }
    }
  }

  for (int i = 0; i < 8; ++i) {
    constants.push_back(conv8to8_0.bias.data[i]);
  }

  const auto& conv8to8_1_weights = GetFloatWeights(conv8to8_1);
  for (int s = 0; s < 2; ++s) {
    for (int d = 0; d < 2; ++d) {
      float4 filters[4];
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          const int src_ch = s * 4 + j;
          const int dst_ch = d * 4 + i;
          const int f_index =
              conv8to8_1_weights.shape.LinearIndex({dst_ch, 0, 0, src_ch});
          filters[i][j] = conv8to8_1_weights.data[f_index];
        }
      }
      float4 filters_new[4];
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          filters_new[i][j] = filters[j][i];
        }
      }
      for (int i = 0; i < 4; ++i) {
        constants.push_back(filters_new[i].x);
        constants.push_back(filters_new[i].y);
        constants.push_back(filters_new[i].z);
        constants.push_back(filters_new[i].w);
      }
    }
  }

  for (int i = 0; i < 8; ++i) {
    constants.push_back(conv8to8_1.bias.data[i]);
  }

  auto alpha0 = std::get_if<Tensor<Linear, DataType::kFloat32>>(&prelu0.alpha);
  for (int i = 0; i < 8; ++i) {
    constants.push_back(alpha0->data[i]);
  }

  for (int i = 0; i < 8; ++i) {
    constants.push_back(conv8to8_2.bias.data[i]);
  }

  const auto& conv8to8_2_weights = GetFloatWeights(conv8to8_2);
  for (int s = 0; s < 2; ++s) {
    for (int d = 0; d < 2; ++d) {
      float4 filters[4];
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          const int src_ch = s * 4 + j;
          const int dst_ch = d * 4 + i;
          const int f_index =
              conv8to8_2_weights.shape.LinearIndex({dst_ch, 0, 0, src_ch});
          filters[i][j] = conv8to8_2_weights.data[f_index];
        }
      }
      float4 filters_new[4];
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          filters_new[i][j] = filters[j][i];
        }
      }
      for (int i = 0; i < 4; ++i) {
        constants.push_back(filters_new[i].x);
        constants.push_back(filters_new[i].y);
        constants.push_back(filters_new[i].z);
        constants.push_back(filters_new[i].w);
      }
    }
  }

  auto alpha1 = std::get_if<Tensor<Linear, DataType::kFloat32>>(&prelu1.alpha);
  for (int i = 0; i < 8; ++i) {
    constants.push_back(alpha1->data[i]);
  }

  GPUOperation result;
  result.AddSrcTensor("src_0", definition.src_tensors[0]);
  result.AddSrcTensor("src_1", definition.src_tensors[1]);
  result.AddDstTensor("dst_0", definition.dst_tensors[0]);
  result.code_ = GenerateConvolutionCode(definition);
  result.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_ZIs1;
  UploadWeights(constants, definition.src_tensors[0].GetDataType(), &result);
  return result;
}

}  // namespace ml_drift
