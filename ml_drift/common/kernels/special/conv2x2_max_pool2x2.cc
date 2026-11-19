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

#include "ml_drift/common/kernels/special/conv2x2_max_pool2x2.h"

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

  int xc = X * 2;
  int yc = Y * 2;

  if (X >= args.dst_0.Width() || Y >= args.dst_0.Height()) return;

  Type r0, r1, r2, r3;
  r0 = args.src_0.Read(xc, yc, 0);
  r1 = args.src_0.Read(xc + 1, yc, 0);
  r2 = args.src_0.Read(xc, yc + 1, 0);
  r3 = args.src_0.Read(xc + 1, yc + 1, 0);

  Type max0 = max(max(r0, r1), max(r2, r3));
  args.dst_1.Write(max0, X, Y, 0);

  Type c0 = constants[0];
  Type c1 = constants[1];
  Type c2 = constants[2];
  Type c3 = constants[3];

  CONV(c0, r0, 4);
  CONV(c1, r0, 8);
  CONV(c2, r0, 12);
  CONV(c3, r0, 16);
  CONV(c0, r1, 20);
  CONV(c1, r1, 24);
  CONV(c2, r1, 28);
  CONV(c3, r1, 32);
  CONV(c0, r2, 36);
  CONV(c1, r2, 40);
  CONV(c2, r2, 44);
  CONV(c3, r2, 48);
  CONV(c0, r3, 52);
  CONV(c1, r3, 56);
  CONV(c2, r3, 60);
  CONV(c3, r3, 64);

  r0 = args.src_0.Read(xc, yc, 1);
  r1 = args.src_0.Read(xc + 1, yc, 1);
  r2 = args.src_0.Read(xc, yc + 1, 1);
  r3 = args.src_0.Read(xc + 1, yc + 1, 1);

  Type max1 = max(max(r0, r1), max(r2, r3));
  args.dst_1.Write(max1, X, Y, 1);

  CONV(c0, r0, 68);
  CONV(c1, r0, 72);
  CONV(c2, r0, 76);
  CONV(c3, r0, 80);
  CONV(c0, r1, 84);
  CONV(c1, r1, 88);
  CONV(c2, r1, 92);
  CONV(c3, r1, 96);
  CONV(c0, r2, 100);
  CONV(c1, r2, 104);
  CONV(c2, r2, 108);
  CONV(c3, r2, 112);
  CONV(c0, r3, 116);
  CONV(c1, r3, 120);
  CONV(c2, r3, 124);
  CONV(c3, r3, 128);

  c0 = max(c0, ucl::Init<Type>(0.0)) + min(c0, ucl::Init<Type>(0.0)) * constants[132];
  c1 = max(c1, ucl::Init<Type>(0.0)) + min(c1, ucl::Init<Type>(0.0)) * constants[133];
  c2 = max(c2, ucl::Init<Type>(0.0)) + min(c2, ucl::Init<Type>(0.0)) * constants[134];
  c3 = max(c3, ucl::Init<Type>(0.0)) + min(c3, ucl::Init<Type>(0.0)) * constants[135];

  args.dst_0.Write(c0, X, Y, 0);
  args.dst_0.Write(c1, X, Y, 1);
  args.dst_0.Write(c2, X, Y, 2);
  args.dst_0.Write(c3, X, Y, 3);
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

GPUOperation CreateConv2x2MaxPool2x2(const OperationDef& definition,
                                     const Convolution2DAttributes& conv_attr,
                                     const PReLUAttributes& prelu_attr) {
  std::vector<float> constants;

  constants.reserve(16);
  for (int i = 0; i < 16; ++i) {
    constants.push_back(conv_attr.bias.data[i]);
  }

  const auto& weights = GetFloatWeights(conv_attr);
  for (int s = 0; s < 2; ++s) {
    for (int y = 0; y < 2; ++y) {
      for (int x = 0; x < 2; ++x) {
        for (int d = 0; d < 4; ++d) {
          float4 filters[4];
          for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
              const int src_ch = s * 4 + j;
              const int dst_ch = d * 4 + i;
              const int f_index =
                  weights.shape.LinearIndex({dst_ch, y, x, src_ch});
              filters[i][j] = weights.data[f_index];
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
    }
  }

  auto alpha0 =
      std::get_if<Tensor<Linear, DataType::kFloat32>>(&prelu_attr.alpha);
  for (int i = 0; i < 16; ++i) {
    constants.push_back(alpha0->data[i]);
  }

  GPUOperation result;
  result.AddSrcTensor("src_0", definition.src_tensors[0]);
  result.AddDstTensor("dst_0", definition.dst_tensors[0]);
  result.AddDstTensor("dst_1", definition.dst_tensors[1]);
  result.code_ = GenerateConvolutionCode(definition);
  result.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_ZIs1;
  UploadWeights(constants, definition.dst_tensors[0].GetDataType(), &result);

  return result;
}

}  // namespace ml_drift
