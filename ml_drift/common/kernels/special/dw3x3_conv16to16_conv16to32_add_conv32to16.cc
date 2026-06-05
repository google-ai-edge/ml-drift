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

#include "ml_drift/common/kernels/special/dw3x3_conv16to16_conv16to32_add_conv32to16.h"

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

std::string GenerateConvolutionCode(bool full, const OperationDef& op_def) {
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

  Type c0 = constants[40];
  Type c1 = constants[41];
  Type c2 = constants[42];
  Type c3 = constants[43];

  Type r0;

if (X < args.dst_0.Width() + 8) {
  r0 = constants[36];
  r0 += args.src_0.Read(X - 1, Y - 1, 0) * constants[0];
  r0 += args.src_0.Read(X - 0, Y - 1, 0) * constants[1];
  r0 += args.src_0.Read(X + 1, Y - 1, 0) * constants[2];
  r0 += args.src_0.Read(X - 1, Y - 0, 0) * constants[3];
  r0 += args.src_0.Read(X - 0, Y - 0, 0) * constants[4];
  r0 += args.src_0.Read(X + 1, Y - 0, 0) * constants[5];
  r0 += args.src_0.Read(X - 1, Y + 1, 0) * constants[6];
  r0 += args.src_0.Read(X - 0, Y + 1, 0) * constants[7];
  r0 += args.src_0.Read(X + 1, Y + 1, 0) * constants[8];

  CONV(c0, r0, 44);
  CONV(c1, r0, 48);
  CONV(c2, r0, 52);
  CONV(c3, r0, 56);

  r0 = constants[37];
  r0 += args.src_0.Read(X - 1, Y - 1, 1) * constants[9];
  r0 += args.src_0.Read(X - 0, Y - 1, 1) * constants[10];
  r0 += args.src_0.Read(X + 1, Y - 1, 1) * constants[11];
  r0 += args.src_0.Read(X - 1, Y - 0, 1) * constants[12];
  r0 += args.src_0.Read(X - 0, Y - 0, 1) * constants[13];
  r0 += args.src_0.Read(X + 1, Y - 0, 1) * constants[14];
  r0 += args.src_0.Read(X - 1, Y + 1, 1) * constants[15];
  r0 += args.src_0.Read(X - 0, Y + 1, 1) * constants[16];
  r0 += args.src_0.Read(X + 1, Y + 1, 1) * constants[17];

  CONV(c0, r0, 60);
  CONV(c1, r0, 64);
  CONV(c2, r0, 68);
  CONV(c3, r0, 72);
}

if (X < args.dst_0.Width() + 7) {
  r0 = constants[38];
  r0 += args.src_0.Read(X - 1, Y - 1, 2) * constants[18];
  r0 += args.src_0.Read(X - 0, Y - 1, 2) * constants[19];
  r0 += args.src_0.Read(X + 1, Y - 1, 2) * constants[20];
  r0 += args.src_0.Read(X - 1, Y - 0, 2) * constants[21];
  r0 += args.src_0.Read(X - 0, Y - 0, 2) * constants[22];
  r0 += args.src_0.Read(X + 1, Y - 0, 2) * constants[23];
  r0 += args.src_0.Read(X - 1, Y + 1, 2) * constants[24];
  r0 += args.src_0.Read(X - 0, Y + 1, 2) * constants[25];
  r0 += args.src_0.Read(X + 1, Y + 1, 2) * constants[26];

  CONV(c0, r0, 76);
  CONV(c1, r0, 80);
  CONV(c2, r0, 84);
  CONV(c3, r0, 88);

  r0 = constants[39];
  r0 += args.src_0.Read(X - 1, Y - 1, 3) * constants[27];
  r0 += args.src_0.Read(X - 0, Y - 1, 3) * constants[28];
  r0 += args.src_0.Read(X + 1, Y - 1, 3) * constants[29];
  r0 += args.src_0.Read(X - 1, Y - 0, 3) * constants[30];
  r0 += args.src_0.Read(X - 0, Y - 0, 3) * constants[31];
  r0 += args.src_0.Read(X + 1, Y - 0, 3) * constants[32];
  r0 += args.src_0.Read(X - 1, Y + 1, 3) * constants[33];
  r0 += args.src_0.Read(X - 0, Y + 1, 3) * constants[34];
  r0 += args.src_0.Read(X + 1, Y + 1, 3) * constants[35];

  CONV(c0, r0, 92);
  CONV(c1, r0, 96);
  CONV(c2, r0, 100);
  CONV(c3, r0, 104);

  c0 = max(c0, ucl::Init<Type>(0.0)) + min(c0, ucl::Init<Type>(0.0)) * constants[108];
  c1 = max(c1, ucl::Init<Type>(0.0)) + min(c1, ucl::Init<Type>(0.0)) * constants[109];
  c2 = max(c2, ucl::Init<Type>(0.0)) + min(c2, ucl::Init<Type>(0.0)) * constants[110];
  c3 = max(c3, ucl::Init<Type>(0.0)) + min(c3, ucl::Init<Type>(0.0)) * constants[111];
}

  Type b0 = constants[256];
  Type b1 = constants[257];
  Type b2 = constants[258];
  Type b3 = constants[259];

if (X < args.dst_0.Width() + 12) {
  Type f0 = constants[112];
  f0 += args.src_1.Read(X, Y, 0);
  CONV(f0, c0, 120);
  CONV(f0, c1, 152);
  CONV(f0, c2, 184);
  CONV(f0, c3, 216);
  f0 = max(f0, ucl::Init<Type>(0.0)) + min(f0, ucl::Init<Type>(0.0)) * constants[248];
  args.dst_0.Write(f0, X, Y, 0);
  CONV(b0, f0, 260);
  CONV(b1, f0, 264);
  CONV(b2, f0, 268);
  CONV(b3, f0, 272);
}
if (X < args.dst_0.Width() + 11) {
  Type f1 = constants[113];
  f1 += args.src_1.Read(X, Y, 1);
  CONV(f1, c0, 124);
  CONV(f1, c1, 156);
  CONV(f1, c2, 188);
  CONV(f1, c3, 220);
  f1 = max(f1, ucl::Init<Type>(0.0)) + min(f1, ucl::Init<Type>(0.0)) * constants[249];
  args.dst_0.Write(f1, X, Y, 1);
  CONV(b0, f1, 276);
  CONV(b1, f1, 280);
  CONV(b2, f1, 284);
  CONV(b3, f1, 288);
}
if (X < args.dst_0.Width() + 10) {
  Type f2 = constants[114];
)";
  if (full) {
    c += "  f2 += args.src_1.Read(X, Y, 2);";
  }
  c += R"(
  CONV(f2, c0, 128);
  CONV(f2, c1, 160);
  CONV(f2, c2, 192);
  CONV(f2, c3, 224);
  f2 = max(f2, ucl::Init<Type>(0.0)) + min(f2, ucl::Init<Type>(0.0)) * constants[250];
  args.dst_0.Write(f2, X, Y, 2);
  CONV(b0, f2, 292);
  CONV(b1, f2, 296);
  CONV(b2, f2, 300);
  CONV(b3, f2, 304);
}
if (X < args.dst_0.Width() + 9) {
  Type f3 = constants[115];
  )";
  if (full) {
    c += "  f3 += args.src_1.Read(X, Y, 3);";
  }
  c += R"(
  CONV(f3, c0, 132);
  CONV(f3, c1, 164);
  CONV(f3, c2, 196);
  CONV(f3, c3, 228);
  f3 = max(f3, ucl::Init<Type>(0.0)) + min(f3, ucl::Init<Type>(0.0)) * constants[251];
  args.dst_0.Write(f3, X, Y, 3);
  CONV(b0, f3, 308);
  CONV(b1, f3, 312);
  CONV(b2, f3, 316);
  CONV(b3, f3, 320);
}
if (X < args.dst_0.Width() + 8) {
  Type f4 = constants[116];
  )";
  if (full) {
    c += "  f4 += args.src_1.Read(X, Y, 4);";
  }
  c += R"(
  CONV(f4, c0, 136);
  CONV(f4, c1, 168);
  CONV(f4, c2, 200);
  CONV(f4, c3, 232);
  f4 = max(f4, ucl::Init<Type>(0.0)) + min(f4, ucl::Init<Type>(0.0)) * constants[252];
  args.dst_0.Write(f4, X, Y, 4);
  CONV(b0, f4, 324);
  CONV(b1, f4, 328);
  CONV(b2, f4, 332);
  CONV(b3, f4, 336);
}
if (X < args.dst_0.Width() + 7) {
  Type f5 = constants[117];
  )";
  if (full) {
    c += "  f5 += args.src_1.Read(X, Y, 5);";
  }
  c += R"(
  CONV(f5, c0, 140);
  CONV(f5, c1, 172);
  CONV(f5, c2, 204);
  CONV(f5, c3, 236);
  f5 = max(f5, ucl::Init<Type>(0.0)) + min(f5, ucl::Init<Type>(0.0)) * constants[253];
  args.dst_0.Write(f5, X, Y, 5);
  CONV(b0, f5, 340);
  CONV(b1, f5, 344);
  CONV(b2, f5, 348);
  CONV(b3, f5, 352);
}
if (X < args.dst_0.Width() + 6) {
  Type f6 = constants[118];
  )";
  if (full) {
    c += "  f6 += args.src_1.Read(X, Y, 6);";
  }
  c += R"(
  CONV(f6, c0, 144);
  CONV(f6, c1, 176);
  CONV(f6, c2, 208);
  CONV(f6, c3, 240);
  f6 = max(f6, ucl::Init<Type>(0.0)) + min(f6, ucl::Init<Type>(0.0)) * constants[254];
  args.dst_0.Write(f6, X, Y, 6);
  CONV(b0, f6, 356);
  CONV(b1, f6, 360);
  CONV(b2, f6, 364);
  CONV(b3, f6, 368);
}
if (X < args.dst_0.Width() + 5) {
  Type f7 = constants[119];
  )";
  if (full) {
    c += "  f7 += args.src_1.Read(X, Y, 7);";
  }
  c += R"(
  CONV(f7, c0, 148);
  CONV(f7, c1, 180);
  CONV(f7, c2, 212);
  CONV(f7, c3, 244);
  f7 = max(f7, ucl::Init<Type>(0.0)) + min(f7, ucl::Init<Type>(0.0)) * constants[255];
  args.dst_0.Write(f7, X, Y, 7);
  CONV(b0, f7, 372);
  CONV(b1, f7, 376);
  CONV(b2, f7, 380);
  CONV(b3, f7, 384);
}
  b0 = max(b0, ucl::Init<Type>(0.0)) + min(b0, ucl::Init<Type>(0.0)) * constants[388];
  b1 = max(b1, ucl::Init<Type>(0.0)) + min(b1, ucl::Init<Type>(0.0)) * constants[389];
  b2 = max(b2, ucl::Init<Type>(0.0)) + min(b2, ucl::Init<Type>(0.0)) * constants[390];
  b3 = max(b3, ucl::Init<Type>(0.0)) + min(b3, ucl::Init<Type>(0.0)) * constants[391];

  args.dst_1.Write(b0, X, Y, 0);
  args.dst_1.Write(b1, X, Y, 1);
  args.dst_1.Write(b2, X, Y, 2);
  args.dst_1.Write(b3, X, Y, 3);
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
  op->args_.AddObject("weights",
                      std::make_unique<BufferDescriptor>(std::move(desc)));
}
}  // namespace

GPUOperation CreateDW3x3Conv16To16Conv16To32AddConv32To16(
    const OperationDef& definition,
    const DepthwiseConvolution2DAttributes& dw_attr,
    const Convolution2DAttributes& conv_0, const PReLUAttributes& prelu0,
    const Convolution2DAttributes& conv_1, const PReLUAttributes& prelu1,
    const Convolution2DAttributes& conv_2, const PReLUAttributes& prelu2,
    bool full) {
  std::vector<float> constants;
  const auto& dw_weights = GetFloatWeights(dw_attr);

  for (int z = 0; z < 4; ++z) {
    for (int y = 0; y < 3; ++y) {
      for (int x = 0; x < 3; ++x) {
        for (int i = 0; i < 4; ++i) {
          const int f_index =
              dw_weights.shape.LinearIndex({0, y, x, z * 4 + i});
          constants.push_back(dw_weights.data[f_index]);
        }
      }
    }
  }

  for (int i = 0; i < 16; ++i) {
    constants.push_back(dw_attr.bias.data[i]);
  }

  for (int i = 0; i < 16; ++i) {
    constants.push_back(conv_0.bias.data[i]);
  }

  const auto& conv_0_weights = GetFloatWeights(conv_0);
  for (int s = 0; s < 4; ++s) {
    for (int d = 0; d < 4; ++d) {
      float4 filters[4];
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          const int src_ch = s * 4 + j;
          const int dst_ch = d * 4 + i;
          const int f_index =
              conv_0_weights.shape.LinearIndex({dst_ch, 0, 0, src_ch});
          filters[i][j] = conv_0_weights.data[f_index];
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
    constants.push_back(conv_1.bias.data[i]);
  }

  const auto& conv_1_weights = GetFloatWeights(conv_1);
  for (int s = 0; s < 4; ++s) {
    for (int d = 0; d < 8; ++d) {
      float4 filters[4];
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          const int src_ch = s * 4 + j;
          const int dst_ch = d * 4 + i;
          const int f_index =
              conv_1_weights.shape.LinearIndex({dst_ch, 0, 0, src_ch});
          filters[i][j] = conv_1_weights.data[f_index];
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
  for (int i = 0; i < 32; ++i) {
    constants.push_back(alpha1->data[i]);
  }

  for (int i = 0; i < 16; ++i) {
    constants.push_back(conv_2.bias.data[i]);
  }

  const auto& conv_2_weights = GetFloatWeights(conv_2);
  for (int s = 0; s < 8; ++s) {
    for (int d = 0; d < 4; ++d) {
      float4 filters[4];
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          const int src_ch = s * 4 + j;
          const int dst_ch = d * 4 + i;
          const int f_index =
              conv_2_weights.shape.LinearIndex({dst_ch, 0, 0, src_ch});
          filters[i][j] = conv_2_weights.data[f_index];
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

  auto alpha2 = std::get_if<Tensor<Linear, DataType::FLOAT32>>(&prelu2.alpha);
  for (int i = 0; i < 16; ++i) {
    constants.push_back(alpha2->data[i]);
  }

  GPUOperation result;
  result.AddSrcTensor("src_0", definition.src_tensors[0]);
  result.AddSrcTensor("src_1", definition.src_tensors[1]);
  result.AddDstTensor("dst_0", definition.dst_tensors[0]);
  result.AddDstTensor("dst_1", definition.dst_tensors[1]);
  result.code_ = GenerateConvolutionCode(full, definition);
  result.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_ZIs1;
  UploadWeights(constants, definition.src_tensors[0].GetDataType(), &result);
  return result;
}

}  // namespace ml_drift
