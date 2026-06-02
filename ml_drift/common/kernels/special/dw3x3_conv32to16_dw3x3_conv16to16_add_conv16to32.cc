// Copyright 2024 The ML Drift Authors.
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

#include "ml_drift/common/kernels/special/dw3x3_conv32to16_dw3x3_conv16to16_add_conv16to32.h"

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
#include "ml_drift/common/task/tensor_desc.h"
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

  Type c0 = constants[80];
  Type c1 = constants[81];
  Type c2 = constants[82];
  Type c3 = constants[83];

  Type r0;

if (X < args.dst_0.Width() + 10) {
  r0 = constants[72];
  r0 += args.src_0.Read(X - 1, Y - 1, 0) * constants[0];
  r0 += args.src_0.Read(X - 0, Y - 1, 0) * constants[1];
  r0 += args.src_0.Read(X + 1, Y - 1, 0) * constants[2];
  r0 += args.src_0.Read(X - 1, Y - 0, 0) * constants[3];
  r0 += args.src_0.Read(X - 0, Y - 0, 0) * constants[4];
  r0 += args.src_0.Read(X + 1, Y - 0, 0) * constants[5];
  r0 += args.src_0.Read(X - 1, Y + 1, 0) * constants[6];
  r0 += args.src_0.Read(X - 0, Y + 1, 0) * constants[7];
  r0 += args.src_0.Read(X + 1, Y + 1, 0) * constants[8];

  CONV(c0, r0, 84);
  CONV(c1, r0, 88);
  CONV(c2, r0, 92);
  CONV(c3, r0, 96);

  r0 = constants[73];
  r0 += args.src_0.Read(X - 1, Y - 1, 1) * constants[9];
  r0 += args.src_0.Read(X - 0, Y - 1, 1) * constants[10];
  r0 += args.src_0.Read(X + 1, Y - 1, 1) * constants[11];
  r0 += args.src_0.Read(X - 1, Y - 0, 1) * constants[12];
  r0 += args.src_0.Read(X - 0, Y - 0, 1) * constants[13];
  r0 += args.src_0.Read(X + 1, Y - 0, 1) * constants[14];
  r0 += args.src_0.Read(X - 1, Y + 1, 1) * constants[15];
  r0 += args.src_0.Read(X - 0, Y + 1, 1) * constants[16];
  r0 += args.src_0.Read(X + 1, Y + 1, 1) * constants[17];

  CONV(c0, r0, 100);
  CONV(c1, r0, 104);
  CONV(c2, r0, 108);
  CONV(c3, r0, 112);
}

if (X < args.dst_0.Width() + 9) {
  r0 = constants[74];
  r0 += args.src_0.Read(X - 1, Y - 1, 2) * constants[18];
  r0 += args.src_0.Read(X - 0, Y - 1, 2) * constants[19];
  r0 += args.src_0.Read(X + 1, Y - 1, 2) * constants[20];
  r0 += args.src_0.Read(X - 1, Y - 0, 2) * constants[21];
  r0 += args.src_0.Read(X - 0, Y - 0, 2) * constants[22];
  r0 += args.src_0.Read(X + 1, Y - 0, 2) * constants[23];
  r0 += args.src_0.Read(X - 1, Y + 1, 2) * constants[24];
  r0 += args.src_0.Read(X - 0, Y + 1, 2) * constants[25];
  r0 += args.src_0.Read(X + 1, Y + 1, 2) * constants[26];

  CONV(c0, r0, 116);
  CONV(c1, r0, 120);
  CONV(c2, r0, 124);
  CONV(c3, r0, 128);

  r0 = constants[75];
  r0 += args.src_0.Read(X - 1, Y - 1, 3) * constants[27];
  r0 += args.src_0.Read(X - 0, Y - 1, 3) * constants[28];
  r0 += args.src_0.Read(X + 1, Y - 1, 3) * constants[29];
  r0 += args.src_0.Read(X - 1, Y - 0, 3) * constants[30];
  r0 += args.src_0.Read(X - 0, Y - 0, 3) * constants[31];
  r0 += args.src_0.Read(X + 1, Y - 0, 3) * constants[32];
  r0 += args.src_0.Read(X - 1, Y + 1, 3) * constants[33];
  r0 += args.src_0.Read(X - 0, Y + 1, 3) * constants[34];
  r0 += args.src_0.Read(X + 1, Y + 1, 3) * constants[35];

  CONV(c0, r0, 132);
  CONV(c1, r0, 136);
  CONV(c2, r0, 140);
  CONV(c3, r0, 144);
}

if (X < args.dst_0.Width() + 8) {
  r0 = constants[76];
  r0 += args.src_0.Read(X - 1, Y - 1, 4) * constants[36];
  r0 += args.src_0.Read(X - 0, Y - 1, 4) * constants[37];
  r0 += args.src_0.Read(X + 1, Y - 1, 4) * constants[38];
  r0 += args.src_0.Read(X - 1, Y - 0, 4) * constants[39];
  r0 += args.src_0.Read(X - 0, Y - 0, 4) * constants[40];
  r0 += args.src_0.Read(X + 1, Y - 0, 4) * constants[41];
  r0 += args.src_0.Read(X - 1, Y + 1, 4) * constants[42];
  r0 += args.src_0.Read(X - 0, Y + 1, 4) * constants[43];
  r0 += args.src_0.Read(X + 1, Y + 1, 4) * constants[44];

  CONV(c0, r0, 148);
  CONV(c1, r0, 152);
  CONV(c2, r0, 156);
  CONV(c3, r0, 160);

  r0 = constants[77];
  r0 += args.src_0.Read(X - 1, Y - 1, 5) * constants[45];
  r0 += args.src_0.Read(X - 0, Y - 1, 5) * constants[46];
  r0 += args.src_0.Read(X + 1, Y - 1, 5) * constants[47];
  r0 += args.src_0.Read(X - 1, Y - 0, 5) * constants[48];
  r0 += args.src_0.Read(X - 0, Y - 0, 5) * constants[49];
  r0 += args.src_0.Read(X + 1, Y - 0, 5) * constants[50];
  r0 += args.src_0.Read(X - 1, Y + 1, 5) * constants[51];
  r0 += args.src_0.Read(X - 0, Y + 1, 5) * constants[52];
  r0 += args.src_0.Read(X + 1, Y + 1, 5) * constants[53];

  CONV(c0, r0, 164);
  CONV(c1, r0, 168);
  CONV(c2, r0, 172);
  CONV(c3, r0, 176);
}

if (X < args.dst_0.Width() + 7) {
  r0 = constants[78];
  r0 += args.src_0.Read(X - 1, Y - 1, 6) * constants[54];
  r0 += args.src_0.Read(X - 0, Y - 1, 6) * constants[55];
  r0 += args.src_0.Read(X + 1, Y - 1, 6) * constants[56];
  r0 += args.src_0.Read(X - 1, Y - 0, 6) * constants[57];
  r0 += args.src_0.Read(X - 0, Y - 0, 6) * constants[58];
  r0 += args.src_0.Read(X + 1, Y - 0, 6) * constants[59];
  r0 += args.src_0.Read(X - 1, Y + 1, 6) * constants[60];
  r0 += args.src_0.Read(X - 0, Y + 1, 6) * constants[61];
  r0 += args.src_0.Read(X + 1, Y + 1, 6) * constants[62];

  CONV(c0, r0, 180);
  CONV(c1, r0, 184);
  CONV(c2, r0, 188);
  CONV(c3, r0, 192);

  r0 = constants[79];
  r0 += args.src_0.Read(X - 1, Y - 1, 7) * constants[63];
  r0 += args.src_0.Read(X - 0, Y - 1, 7) * constants[64];
  r0 += args.src_0.Read(X + 1, Y - 1, 7) * constants[65];
  r0 += args.src_0.Read(X - 1, Y - 0, 7) * constants[66];
  r0 += args.src_0.Read(X - 0, Y - 0, 7) * constants[67];
  r0 += args.src_0.Read(X + 1, Y - 0, 7) * constants[68];
  r0 += args.src_0.Read(X - 1, Y + 1, 7) * constants[69];
  r0 += args.src_0.Read(X - 0, Y + 1, 7) * constants[70];
  r0 += args.src_0.Read(X + 1, Y + 1, 7) * constants[71];

  CONV(c0, r0, 196);
  CONV(c1, r0, 200);
  CONV(c2, r0, 204);
  CONV(c3, r0, 208);
}

if (X < args.dst_0.Width() + 6) {
  r0 = constants[248];
  r0 += args.src_1.Read(X - 1, Y - 1, 0) * constants[212];
  r0 += args.src_1.Read(X - 0, Y - 1, 0) * constants[213];
  r0 += args.src_1.Read(X + 1, Y - 1, 0) * constants[214];
  r0 += args.src_1.Read(X - 1, Y - 0, 0) * constants[215];
  r0 += args.src_1.Read(X - 0, Y - 0, 0) * constants[216];
  r0 += args.src_1.Read(X + 1, Y - 0, 0) * constants[217];
  r0 += args.src_1.Read(X - 1, Y + 1, 0) * constants[218];
  r0 += args.src_1.Read(X - 0, Y + 1, 0) * constants[219];
  r0 += args.src_1.Read(X + 1, Y + 1, 0) * constants[220];

  CONV(c0, r0, 252);
  CONV(c1, r0, 256);
  CONV(c2, r0, 260);
  CONV(c3, r0, 264);

  r0 = constants[249];
  r0 += args.src_1.Read(X - 1, Y - 1, 1) * constants[221];
  r0 += args.src_1.Read(X - 0, Y - 1, 1) * constants[222];
  r0 += args.src_1.Read(X + 1, Y - 1, 1) * constants[223];
  r0 += args.src_1.Read(X - 1, Y - 0, 1) * constants[224];
  r0 += args.src_1.Read(X - 0, Y - 0, 1) * constants[225];
  r0 += args.src_1.Read(X + 1, Y - 0, 1) * constants[226];
  r0 += args.src_1.Read(X - 1, Y + 1, 1) * constants[227];
  r0 += args.src_1.Read(X - 0, Y + 1, 1) * constants[228];
  r0 += args.src_1.Read(X + 1, Y + 1, 1) * constants[229];

  CONV(c0, r0, 268);
  CONV(c1, r0, 272);
  CONV(c2, r0, 276);
  CONV(c3, r0, 280);
}

if (X < args.dst_0.Width() + 5) {
  r0 = constants[250];
  r0 += args.src_1.Read(X - 1, Y - 1, 2) * constants[230];
  r0 += args.src_1.Read(X - 0, Y - 1, 2) * constants[231];
  r0 += args.src_1.Read(X + 1, Y - 1, 2) * constants[232];
  r0 += args.src_1.Read(X - 1, Y - 0, 2) * constants[233];
  r0 += args.src_1.Read(X - 0, Y - 0, 2) * constants[234];
  r0 += args.src_1.Read(X + 1, Y - 0, 2) * constants[235];
  r0 += args.src_1.Read(X - 1, Y + 1, 2) * constants[236];
  r0 += args.src_1.Read(X - 0, Y + 1, 2) * constants[237];
  r0 += args.src_1.Read(X + 1, Y + 1, 2) * constants[238];

  CONV(c0, r0, 284);
  CONV(c1, r0, 288);
  CONV(c2, r0, 292);
  CONV(c3, r0, 296);

  r0 = constants[251];
  r0 += args.src_1.Read(X - 1, Y - 1, 3) * constants[239];
  r0 += args.src_1.Read(X - 0, Y - 1, 3) * constants[240];
  r0 += args.src_1.Read(X + 1, Y - 1, 3) * constants[241];
  r0 += args.src_1.Read(X - 1, Y - 0, 3) * constants[242];
  r0 += args.src_1.Read(X - 0, Y - 0, 3) * constants[243];
  r0 += args.src_1.Read(X + 1, Y - 0, 3) * constants[244];
  r0 += args.src_1.Read(X - 1, Y + 1, 3) * constants[245];
  r0 += args.src_1.Read(X - 0, Y + 1, 3) * constants[246];
  r0 += args.src_1.Read(X + 1, Y + 1, 3) * constants[247];

  CONV(c0, r0, 300);
  CONV(c1, r0, 304);
  CONV(c2, r0, 308);
  CONV(c3, r0, 312);

  c0 = max(c0, ucl::Init<Type>(0.0)) + min(c0, ucl::Init<Type>(0.0)) * constants[316];
  c1 = max(c1, ucl::Init<Type>(0.0)) + min(c1, ucl::Init<Type>(0.0)) * constants[317];
  c2 = max(c2, ucl::Init<Type>(0.0)) + min(c2, ucl::Init<Type>(0.0)) * constants[318];
  c3 = max(c3, ucl::Init<Type>(0.0)) + min(c3, ucl::Init<Type>(0.0)) * constants[319];
}

if (X < args.dst_0.Width() + 4) {
  Type f0 = constants[320];
  CONV(f0, c0, 328);
  CONV(f0, c1, 332);
  CONV(f0, c2, 336);
  CONV(f0, c3, 340);
  f0 = max(f0, ucl::Init<Type>(0.0)) + min(f0, ucl::Init<Type>(0.0)) * args.alpha1.Read(0);
  args.dst_0.Write(f0, X, Y, 0);
  Type f1 = constants[321];
  CONV(f1, c0, 344);
  CONV(f1, c1, 348);
  CONV(f1, c2, 352);
  CONV(f1, c3, 356);
  f1 = max(f1, ucl::Init<Type>(0.0)) + min(f1, ucl::Init<Type>(0.0)) * args.alpha1.Read(1);
  args.dst_0.Write(f1, X, Y, 1);
}
if (X < args.dst_0.Width() + 3) {
  Type f0 = constants[322];
  CONV(f0, c0, 360);
  CONV(f0, c1, 364);
  CONV(f0, c2, 368);
  CONV(f0, c3, 372);
  f0 = max(f0, ucl::Init<Type>(0.0)) + min(f0, ucl::Init<Type>(0.0)) * args.alpha1.Read(2);
  args.dst_0.Write(f0, X, Y, 2);
  Type f1 = constants[323];
  CONV(f1, c0, 376);
  CONV(f1, c1, 380);
  CONV(f1, c2, 384);
  CONV(f1, c3, 388);
  f1 = max(f1, ucl::Init<Type>(0.0)) + min(f1, ucl::Init<Type>(0.0)) * args.alpha1.Read(3);
  args.dst_0.Write(f1, X, Y, 3);
}
if (X < args.dst_0.Width() + 2) {
  Type f0 = constants[324];
  CONV(f0, c0, 392);
  CONV(f0, c1, 396);
  CONV(f0, c2, 400);
  CONV(f0, c3, 404);
  f0 = max(f0, ucl::Init<Type>(0.0)) + min(f0, ucl::Init<Type>(0.0)) * args.alpha1.Read(4);
  args.dst_0.Write(f0, X, Y, 4);
  Type f1 = constants[325];
  CONV(f1, c0, 408);
  CONV(f1, c1, 412);
  CONV(f1, c2, 416);
  CONV(f1, c3, 420);
  f1 = max(f1, ucl::Init<Type>(0.0)) + min(f1, ucl::Init<Type>(0.0)) * args.alpha1.Read(5);
  args.dst_0.Write(f1, X, Y, 5);
}
if (X < args.dst_0.Width() + 1) {
  Type f0 = constants[326];
  CONV(f0, c0, 424);
  CONV(f0, c1, 428);
  CONV(f0, c2, 432);
  CONV(f0, c3, 436);
  f0 = max(f0, ucl::Init<Type>(0.0)) + min(f0, ucl::Init<Type>(0.0)) * args.alpha1.Read(6);
  args.dst_0.Write(f0, X, Y, 6);
  Type f1 = constants[327];
  CONV(f1, c0, 440);
  CONV(f1, c1, 444);
  CONV(f1, c2, 448);
  CONV(f1, c3, 452);
  f1 = max(f1, ucl::Init<Type>(0.0)) + min(f1, ucl::Init<Type>(0.0)) * args.alpha1.Read(7);
  args.dst_0.Write(f1, X, Y, 7);
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
  desc.memory_type = MemoryType::CONSTANT;
  desc.size = SizeOf(data_type) * constants.size();
  desc.data.resize(desc.size);

  if (data_type == DataType::FLOAT32) {
    memcpy(desc.data.data(), constants.data(), desc.size);
  } else {
    half* gpu_data_half = reinterpret_cast<half*>(desc.data.data());
    for (int i = 0; i < constants.size(); ++i) {
      gpu_data_half[i] = constants[i];
    }
  }
  op->args_.AddObject("weights", std::make_unique<BufferDescriptor>(desc));
}
}  // namespace

GPUOperation CreateDW3x3Conv32To16DW3x3Conv16To16AddConv16To32(
    const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& dw_attr_0,
    const Convolution2DAttributes& conv32to16,
    const DepthwiseConvolution2DAttributes& dw_attr_1,
    const Convolution2DAttributes& conv16to16, const PReLUAttributes& prelu0,
    const Convolution2DAttributes& conv16to32, const PReLUAttributes& prelu1) {
  std::vector<float> constants;

  const auto& dw_0_weights = GetFloatWeights(dw_attr_0);
  for (int z = 0; z < 8; ++z) {
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

  for (int i = 0; i < 32; ++i) {
    constants.push_back(dw_attr_0.bias.data[i]);
  }

  for (int i = 0; i < 16; ++i) {
    float v = conv32to16.bias.data[i] + conv16to16.bias.data[i];
    constants.push_back(v);
  }

  const auto& conv32to16_weights = GetFloatWeights(conv32to16);
  for (int s = 0; s < 8; ++s) {
    for (int d = 0; d < 4; ++d) {
      float4 filters[4];
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          const int src_ch = s * 4 + j;
          const int dst_ch = d * 4 + i;
          const int f_index =
              conv32to16_weights.shape.LinearIndex({dst_ch, 0, 0, src_ch});
          filters[i][j] = conv32to16_weights.data[f_index];
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

  const auto& dw_1_weights = GetFloatWeights(dw_attr_1);
  for (int z = 0; z < 4; ++z) {
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

  for (int i = 0; i < 16; ++i) {
    constants.push_back(dw_attr_1.bias.data[i]);
  }

  const auto& conv16to16_weights = GetFloatWeights(conv16to16);
  for (int s = 0; s < 4; ++s) {
    for (int d = 0; d < 4; ++d) {
      float4 filters[4];
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          const int src_ch = s * 4 + j;
          const int dst_ch = d * 4 + i;
          const int f_index =
              conv16to16_weights.shape.LinearIndex({dst_ch, 0, 0, src_ch});
          filters[i][j] = conv16to16_weights.data[f_index];
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

  auto alpha0 = std::get_if<Tensor<Linear, DataType::FLOAT32>>(&prelu0.alpha);
  for (int i = 0; i < 16; ++i) {
    constants.push_back(alpha0->data[i]);
  }

  for (int i = 0; i < 32; ++i) {
    constants.push_back(conv16to32.bias.data[i]);
  }

  const auto& conv16to32_weights = GetFloatWeights(conv16to32);
  for (int d = 0; d < 8; ++d) {
    for (int s = 0; s < 4; ++s) {
      float4 filters[4];
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          const int src_ch = s * 4 + j;
          const int dst_ch = d * 4 + i;
          const int f_index =
              conv16to32_weights.shape.LinearIndex({dst_ch, 0, 0, src_ch});
          filters[i][j] = conv16to32_weights.data[f_index];
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

  auto alpha1 = std::get_if<Tensor<Linear, DataType::FLOAT32>>(&prelu1.alpha);

  GPUOperation result;
  result.AddSrcTensor("src_0", definition.src_tensors[0]);
  result.AddSrcTensor("src_1", definition.src_tensors[1]);
  result.AddDstTensor("dst_0", definition.dst_tensors[0]);
  result.code_ = GenerateConvolutionCode(definition);
  result.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_ZIs1;
  UploadWeights(constants, definition.src_tensors[0].GetDataType(), &result);
  TensorDescriptor alpha_tensor_desc = CreateConstantLinearTensorDescriptor(
      definition.src_tensors[0].GetDataType(), TensorStorageType::TEXTURE_2D,
      *alpha1);
  result.args_.AddObject("alpha1", std::make_unique<TensorDescriptor>(
                                       std::move(alpha_tensor_desc)));
  return result;
}

}  // namespace ml_drift
