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

#include "ml_drift/samples/stable_diffusion/util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <ios>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

constexpr int kBytesPerPixel = 3;  // red, green, & blue
constexpr int kFileHeaderSize = 14;
constexpr int kInfoHeaderSize = 40;

unsigned char* CreateBitmapFileHeader(int height, int stride) {
  const int file_size = kFileHeaderSize + kInfoHeaderSize + stride * height;
  static unsigned char file_header[] = {
      0, 0,        // signature
      0, 0, 0, 0,  // image file size in bytes
      0, 0, 0, 0,  // reserved
      0, 0, 0, 0,  // start of pixel array
  };
  file_header[0] = static_cast<unsigned char>('B');
  file_header[1] = static_cast<unsigned char>('M');
  file_header[2] = static_cast<unsigned char>(file_size);
  file_header[3] = static_cast<unsigned char>(file_size >> 8);
  file_header[4] = static_cast<unsigned char>(file_size >> 16);
  file_header[5] = static_cast<unsigned char>(file_size >> 24);
  file_header[10] =
      static_cast<unsigned char>(kFileHeaderSize + kInfoHeaderSize);
  return file_header;
}

unsigned char* CreateBitmapInfoHeader(int height, int width) {
  static unsigned char info_header[] = {
      0, 0, 0, 0,  // header size
      0, 0, 0, 0,  // image width
      0, 0, 0, 0,  // image height
      0, 0,        // number of color planes
      0, 0,        // bits per pixel
      0, 0, 0, 0,  // compression
      0, 0, 0, 0,  // image size
      0, 0, 0, 0,  // horizontal resolution
      0, 0, 0, 0,  // vertical resolution
      0, 0, 0, 0,  // colors in color table
      0, 0, 0, 0,  // important color count
  };
  info_header[0] = static_cast<unsigned char>(kInfoHeaderSize);
  info_header[4] = static_cast<unsigned char>(width);
  info_header[5] = static_cast<unsigned char>(width >> 8);
  info_header[6] = static_cast<unsigned char>(width >> 16);
  info_header[7] = static_cast<unsigned char>(width >> 24);
  info_header[8] = static_cast<unsigned char>(height);
  info_header[9] = static_cast<unsigned char>(height >> 8);
  info_header[10] = static_cast<unsigned char>(height >> 16);
  info_header[11] = static_cast<unsigned char>(height >> 24);
  info_header[12] = static_cast<unsigned char>(1);
  info_header[14] = static_cast<unsigned char>(kBytesPerPixel * 8);
  return info_header;
}

void GenerateBitmapImage(unsigned char* image, int height, int width,
                         const std::string& file_name) {
  const int width_in_bytes = width * kBytesPerPixel;
  const unsigned char padding[3] = {0, 0, 0};
  const int padding_size = (4 - (width_in_bytes) % 4) % 4;
  const int stride = width_in_bytes + padding_size;
  FILE* image_file = fopen(file_name.c_str(), "wb");
  const unsigned char* file_header = CreateBitmapFileHeader(height, stride);
  fwrite(file_header, 1, kFileHeaderSize, image_file);
  const unsigned char* info_header = CreateBitmapInfoHeader(height, width);
  fwrite(info_header, 1, kInfoHeaderSize, image_file);
  for (int i = 0; i < height; i++) {
    fwrite(image + i * width_in_bytes, kBytesPerPixel, width, image_file);
    fwrite(padding, 1, padding_size, image_file);
  }
  fclose(image_file);
}

void GenerateImage(const TensorFloat32& tensor, const std::string& file_name) {
  const int height = tensor.shape.h;
  const int width = tensor.shape.w;
  std::vector<unsigned char> image_data(height * width * 3);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      float r =
          tensor.data[tensor.shape.LinearIndex({0, height - 1 - i, j, 0})];
      float g =
          tensor.data[tensor.shape.LinearIndex({0, height - 1 - i, j, 1})];
      float b =
          tensor.data[tensor.shape.LinearIndex({0, height - 1 - i, j, 2})];
      r = std::min(std::max(r, 0.0f), 1.0f) * 255.0f;
      g = std::min(std::max(g, 0.0f), 1.0f) * 255.0f;
      b = std::min(std::max(b, 0.0f), 1.0f) * 255.0f;
      image_data[(i * width + j) * 3 + 2] = static_cast<unsigned char>(r);
      image_data[(i * width + j) * 3 + 1] = static_cast<unsigned char>(g);
      image_data[(i * width + j) * 3 + 0] = static_cast<unsigned char>(b);
    }
  }
  GenerateBitmapImage(image_data.data(), height, width, file_name);
}

void GenerateIntermediateImage(const TensorFloat32& tensor,
                               const std::string& file_name,
                               const std::string& file_folder) {
  const std::vector<float> w = {
      0.14013671875,  0.0711669921875,  -0.03271484375,    -0.11407470703125,
      0.126220703125, 0.10101318359375, 0.034515380859375, -0.1383056640625,
      0.126220703125, 0.07733154296875, 0.042633056640625, -0.177978515625};
  const std::vector<float> b = {0.423828125, 0.471923828125, 0.473876953125};
  const int height = tensor.shape.h;
  const int width = tensor.shape.w;
  std::vector<unsigned char> image_data(height * width * 3);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      const float v0 =
          tensor.data[tensor.shape.LinearIndex({0, height - 1 - i, j, 0})];
      const float v1 =
          tensor.data[tensor.shape.LinearIndex({0, height - 1 - i, j, 1})];
      const float v2 =
          tensor.data[tensor.shape.LinearIndex({0, height - 1 - i, j, 2})];
      float v3 = tensor.data[tensor.shape.LinearIndex({0, i, j, 3})];
      float r_val = v0 * w[0] + v1 * w[1] + v2 * w[2] + v3 * w[3] + b[0];
      float g_val = v0 * w[4] + v1 * w[5] + v2 * w[6] + v3 * w[7] + b[1];
      float b_val = v0 * w[8] + v1 * w[9] + v2 * w[10] + v3 * w[11] + b[2];
      r_val = std::min(std::max(r_val, 0.0f), 1.0f) * 255.0f;
      g_val = std::min(std::max(g_val, 0.0f), 1.0f) * 255.0f;
      b_val = std::min(std::max(b_val, 0.0f), 1.0f) * 255.0f;
      image_data[(i * width + j) * 3 + 2] = static_cast<unsigned char>(r_val);
      image_data[(i * width + j) * 3 + 1] = static_cast<unsigned char>(g_val);
      image_data[(i * width + j) * 3 + 0] = static_cast<unsigned char>(b_val);
    }
  }
  GenerateBitmapImage(image_data.data(), height, width, file_name);
}

TensorFloat32 ReadBMP(const std::string& file_name) {
  FILE* f = fopen(file_name.c_str(), "rb");
  std::vector<uint8_t> info(54);

  // read the 54-byte header
  fread(info.data(), sizeof(uint8_t), info.size(), f);

  // extract image height and width from header
  int width = abs(*(int*)&info[18]);
  int height = abs(*(int*)&info[22]);

  std::vector<uint8_t> data(3 * width * height);

  // read the rest of the data at once
  fread(data.data(), sizeof(uint8_t), data.size(), f);
  fclose(f);

  TensorFloat32 result;
  result.shape = BHWC(1, height, width, 3);
  result.data.resize(result.shape.DimensionsProduct());

  for (int i = 0; i < data.size(); i += 3) {
    result.data[i + 0] = data[i + 2] / 255.0f;
    result.data[i + 1] = data[i + 1] / 255.0f;
    result.data[i + 2] = data[i + 0] / 255.0f;
  }

  return result;
}

std::vector<half> LoadF16(const std::string& path, int count) {
  std::ifstream ifstr(path, std::ios::binary);
  if (!ifstr) {
    std::cout << "Cannot open file! " << path << std::endl;
    return {};
  }
  std::vector<half> result(count);
  ifstr.read(reinterpret_cast<char*>(&result[0]), sizeof(half) * count);
  return result;
}

absl::Span<const half> LoadF16Ex(const std::string& path, int count,
                                 half* buffer) {
  if (!count) return {};
  std::ifstream ifstr(path, std::ios::binary);
  if (!ifstr) {
    std::cout << "Cannot open file! " << path << std::endl;
    return {};
  }
  ifstr.read(reinterpret_cast<char*>(buffer), sizeof(half) * count);
  return absl::MakeConstSpan(buffer, count);
}

Tensor<Linear, DataType::kFloat32> CreateLinearTensor(
    absl::Span<const half> data) {
  return {
      .shape = Linear(data.size()),
      .data = internal::ConvertLinearFp16ToFp32(data),
  };
}

GPUOperation CreateTembGenerationOp(const GpuInfo& gpu_info,
                                    const TensorDescriptor& dst,
                                    const std::string& file_folder) {
  GPUOperation op;
  op.AddDstTensor("dst", dst);
  op.args_.AddFloat("index_val", 1.0f);
  Tensor<Linear, DataType::kFloat32> coeffs_tensor;
  coeffs_tensor.shape = Linear(160);
  coeffs_tensor.data.resize(coeffs_tensor.shape.DimensionsProduct());
  // th.exp(-th.log(th.tensor([10000])) * th.arange(0, 160) / 160)
  for (int i = 0; i < coeffs_tensor.shape.v; ++i) {
    float value = std::exp(-std::log(10000.0) * (i / 160.0));
    coeffs_tensor.data[i] = value;
  }
  TensorDescriptor coeffs_tensor_desc = CreateConstantLinearTensorDescriptor(
      gpu_info, DataType::kFloat32, coeffs_tensor);
  op.args_.AddObject("coeffs", std::make_unique<TensorDescriptor>(
                                   std::move(coeffs_tensor_desc)));

  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  std::string c;
  c += R"(MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.dst.Width() || Y >= args.dst.Height() || S >= args.dst.Slices()) return;
  if (S < 40) {
    args.dst::type result = ucl::Convert<args.dst::type>(cos(args.coeffs.Read(S) * args.index_val));
    args.dst.Write(result, X, Y, S);
  } else {
    args.dst::type result = ucl::Convert<args.dst::type>(sin(args.coeffs.Read(S - 40) * args.index_val));
    args.dst.Write(result, X, Y, S);
  }
})";
  op.code_ = std::move(c);
  return op;
}

GPUOperation CreateDiffusionStepOp(const TensorDescriptor& x_in,
                                   const TensorDescriptor& eta_uncond_in,
                                   const TensorDescriptor& eta_cond_in,
                                   const TensorDescriptor& dst) {
  GPUOperation op;
  op.AddSrcTensor("xIn", x_in);
  op.AddSrcTensor("etaUncondIn", eta_uncond_in);
  op.AddSrcTensor("etaCondIn", eta_cond_in);
  op.AddDstTensor("dst", dst);
  op.args_.AddHalf("guidance_scale", half(1.0f));
  op.args_.AddHalf("sqrt_alpha", half(1.0f));
  op.args_.AddHalf("sqrt_alpha_prev", half(1.0f));
  op.args_.AddHalf("sqrt_one_minus_alpha", half(1.0f));
  op.args_.AddHalf("sqrt_one_minus_alpha_prev", half(1.0f));
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  std::string c;
  c += R"(MAIN_FUNCTION($0) {
    int X = ucl::GetGlobalId<0>();
    int Y = ucl::GetGlobalId<1>();
    int S = ucl::GetGlobalId<2>();
    if (X >= args.dst.Width() || Y >= args.dst.Height() || S >= args.dst.Slices()) return;
    Type eta_cond = args.etaCondIn.Read(X, Y, S);
    Type eta_uncond = args.etaUncondIn.Read(X, Y, S);
    Type x_in = args.xIn.Read(X, Y, S);

    Type delta_cond = (eta_cond - eta_uncond) * args.guidance_scale;
    Type eta = eta_uncond + delta_cond;

    Type deltaX0 = eta * args.sqrt_one_minus_alpha;
    Type predX0Unscaled = x_in - deltaX0;
    Type predX0 = predX0Unscaled / args.sqrt_alpha;
    Type dirX = eta * args.sqrt_one_minus_alpha_prev;
    Type xPrevBase = predX0 * args.sqrt_alpha_prev;
    Type result = xPrevBase + dirX;
    args.dst.Write(result, X, Y, S);
  })";
  absl::StrReplaceAll({{"Type", ToUclDataType(dst.GetDataType(), 4)}}, &c);
  op.code_ = std::move(c);
  return op;
}

Convolution2DAttributes MakeConvAttributes(absl::Span<const half> weights,
                                           absl::Span<const half> bias,
                                           const OHWI& shape, const HW& stride,
                                           const std::string& name) {
  const int padding_h = shape.h / 2;
  const int padding_w = shape.w / 2;
  Convolution2DAttributes attr;
  attr.op_name = name;
  attr.padding.prepended = HW(padding_h, padding_w);
  attr.padding.appended = HW(padding_h, padding_w);
  attr.strides = stride;
  attr.dilations = HW(1, 1);
  auto& attr_weights = attr.weights.emplace<Tensor<OHWI, DataType::kFloat32>>();
  attr_weights.shape = shape;
  attr_weights.data = internal::ConvertOihwFp16ToOhwiFp32(weights, shape);
  if (!bias.empty()) {
    attr.bias = CreateLinearTensor(bias);
  } else {
    attr.bias.shape = Linear(attr_weights.shape.o);
    attr.bias.data.resize(attr.bias.shape.DimensionsProduct());
  }
  return attr;
}

std::pair<Convolution2DAttributes, Convolution2DAttributes> SplitAttributes(
    const Convolution2DAttributes& attributes) {
  const auto& attributes_weights = GetFloatWeights(attributes);
  const int weights_size =
      attributes_weights.shape.DimensionsProduct() / 2 * sizeof(float);
  const int biases_size = attributes.bias.data.size() / 2 * sizeof(float);
  Convolution2DAttributes attr0 = attributes;
  auto& attr0_weights =
      attr0.weights.emplace<Tensor<OHWI, DataType::kFloat32>>();
  attr0_weights.shape = attributes_weights.shape;
  attr0_weights.shape.o /= 2;
  attr0_weights.data.resize(attr0_weights.shape.DimensionsProduct() +
                            XNN_EXTRA_BYTES / sizeof(float));
  std::memcpy(attr0_weights.data.data(), attributes_weights.data.data(),
              weights_size);
  attr0.bias.shape = Linear(attr0_weights.shape.o);
  attr0.bias.data.resize(attr0.bias.shape.DimensionsProduct());
  std::memcpy(attr0.bias.data.data(), attributes.bias.data.data(), biases_size);
  Convolution2DAttributes attr1 = attributes;
  auto& attr1_weights =
      attr1.weights.emplace<Tensor<OHWI, DataType::kFloat32>>();
  attr1_weights.shape = attributes_weights.shape;
  attr1_weights.shape.o /= 2;
  attr1_weights.data.resize(attr1_weights.shape.DimensionsProduct() +
                            XNN_EXTRA_BYTES / sizeof(float));
  std::memcpy(attr1_weights.data.data(),
              attributes_weights.data.data() +
                  attributes_weights.shape.DimensionsProduct() / 2,
              weights_size);
  attr1.bias.shape = Linear(attr1_weights.shape.o);
  attr1.bias.data.resize(attr1.bias.shape.DimensionsProduct());
  std::memcpy(attr1.bias.data.data(),
              attributes.bias.data.data() + attributes.bias.data.size() / 2,
              biases_size);
  return {attr0, attr1};
}

DepthwiseConvolution2DAttributes MakeDwConvAttributes(
    absl::Span<const half> weights, absl::Span<const half> bias,
    const OHWI& shape, const HW& stride, const std::string& name) {
  const int padding_h = shape.h / 2;
  const int padding_w = shape.w / 2;
  DepthwiseConvolution2DAttributes attr;
  attr.op_name = name;
  attr.padding.prepended = HW(padding_h, padding_w);
  attr.padding.appended = HW(padding_h, padding_w);
  attr.strides = stride;
  attr.dilations = HW(1, 1);
  auto& attr_weights = attr.weights.emplace<Tensor<OHWI, DataType::kFloat32>>();
  attr_weights.shape = shape;
  attr_weights.data = internal::ConvertOihwFp16ToOhwiFp32(weights, shape);
  if (!bias.empty()) {
    attr.bias = CreateLinearTensor(bias);
  } else {
    attr.bias.shape = Linear(attr_weights.shape.o);
    attr.bias.data.resize(attr.bias.shape.DimensionsProduct());
  }
  return attr;
}

absl::StatusOr<TensorFloat32> GenerateOpenClipMaskTensor(
    const Tensor<BHWC, DataType::kInt32>& prompt_tensor) {
  if (prompt_tensor.shape.b != 1 || prompt_tensor.shape.h != 1 ||
      prompt_tensor.shape.w != 2 || prompt_tensor.shape.c != 77) {
    return absl::InvalidArgumentError(
        "The shape of the prompt tensor is expected to be [1, 1, 2, 77]");
  }
  int negative_prompt_end_index = 77, positive_prompt_end_index = 77;
  for (int i = 0; i < 77; ++i) {
    if (prompt_tensor.data[i] == 49407 && negative_prompt_end_index == 77) {
      negative_prompt_end_index = i;
    }
    if (prompt_tensor.data[i + 77] == 49407 &&
        positive_prompt_end_index == 77) {
      positive_prompt_end_index = i;
    }
  }
  TensorFloat32 mask_tensor;
  mask_tensor.shape = BHWC(1, 2, 77, 77);
  mask_tensor.data.resize(mask_tensor.shape.DimensionsProduct());
  for (int r = 0; r < 77; ++r) {
    for (int c = 0; c < 77; ++c) {
      float value1 = -65500.0f, value2 = -65500.0f;
      if (r <= negative_prompt_end_index) {
        value1 = r < c ? -65500.0f : 65500.0f;
      }
      if (r <= positive_prompt_end_index) {
        value2 = r < c ? -65500.0f : 65500.0f;
      }
      mask_tensor.data[r * 77 + c] = value1;
      mask_tensor.data[77 * 77 + r * 77 + c] = value2;
    }
  }
  return mask_tensor;
}

#if defined(__ANDROID__) && defined(__aarch64__)
// Converts 8 FP16s to 8 FP32s with a transpose.
// Read a column and write a row.
void ConvertFp16ToFp32ColumnNeon(const uint16_t* src,
                                 int src_stride,  // in elements
                                 float* dst, int width) {
  asm volatile(
      "subs        %w2, %w2, #8                  \n"  // Are there 8 rows?
      "b.lt        2f                            \n"
      "1:                                        \n"
      "ld1r        {v0.8h}, [%0], %3             \n"  // load 8 halffloats
      "ld1         {v0.h}[1], [%0], %3           \n"
      "ld1         {v0.h}[2], [%0], %3           \n"
      "ld1         {v0.h}[3], [%0], %3           \n"
      "ld1r        {v1.8h}, [%0], %3             \n"  // load 8 halffloats
      "ld1         {v1.h}[1], [%0], %3           \n"
      "ld1         {v1.h}[2], [%0], %3           \n"
      "ld1         {v1.h}[3], [%0], %3           \n"
      "subs        %w2, %w2, #8                  \n"  // 8 rows per loop
      "prfm        pldl1keep, [%0, 448]          \n"
      "fcvtl       v2.4s, v0.4h                  \n"  // 4 floats
      "fcvtl       v3.4s, v1.4h                  \n"  // 4 more floats
      "stp         q2, q3, [%1], #32             \n"  // store 8 floats
      "b.ge        1b                            \n"
      "2:                                        \n"
      "adds        %w2, %w2, #7                  \n"  // Add back 8 and compare
      "b.lt        4f                            \n"
      "3:                                        \n"
      "ldr         h0, [%0]                      \n"
      "add         %0, %0, %3                    \n"
      "subs        %w2, %w2, #1                  \n"  // 1 float per loop
      "fcvtl       v1.4s, v0.4h                  \n"  // 1 float
      "str         s1, [%1], #4                  \n"  // store 1 float
      "b.ge        3b                            \n"
      "4:                                        \n"
      : "+r"(src),                        // %0
        "+r"(dst),                        // %1
        "+r"(width)                       // %2
      : "r"((ptrdiff_t)(src_stride * 2))  // %3
      : "cc", "memory", "v0", "v1", "v2", "v3");
}

// Converts 8 FP16s to 8 FP32s.
void ConvertFp16ToFp32Neon(const uint16_t* src, float* dst, int n) {
  asm volatile(
      "subs        %w2, %w2, #8                  \n"  // Are there 8 values?
      "b.lt        2f                            \n"
      "1:                                        \n"
      "ld1         {v0.8h}, [%0], #16            \n"  // load 8 halffloats
      "subs        %w2, %w2, #8                  \n"  // 8 floats per loop
      "prfm        pldl1keep, [%0, 448]          \n"
      "fcvtl       v1.4s, v0.4h                  \n"  // 8 floats
      "fcvtl2      v2.4s, v0.8h                  \n"
      "stp         q1, q2, [%1], #32             \n"  // store 8 floats
      "b.ge        1b                            \n"
      "2:                                        \n"
      "adds        %w2, %w2, #7                  \n"  // Add back 8 and compare
      "b.lt        4f                            \n"
      "3:                                        \n"
      "ldr         h0, [%0], #2                  \n"  // load 1 halffloat
      "subs        %w2, %w2, #1                  \n"  // 1 float per loop
      "fcvtl       v1.4s, v0.4h                  \n"  // 1 float
      "str         s1, [%1], #4                  \n"  // store 1 float
      "b.ge        3b                            \n"
      "4:                                        \n"
      : "+r"(src),  // %0
        "+r"(dst),  // %1
        "+r"(n)     // %2
      :
      : "cc", "memory", "v0", "v1", "v2");
}
#endif  // defined(__ANDROID__) && defined(__aarch64__)

namespace internal {

std::vector<float> ConvertLinearFp16ToFp32(absl::Span<const half> fp16s) {
  std::vector<float> fp32s;
#if defined(__ANDROID__) && defined(__aarch64__)
  fp32s.resize(fp16s.size());
  ConvertFp16ToFp32Neon(reinterpret_cast<const uint16_t*>(fp16s.data()),
                        &fp32s[0], fp16s.size());
#else
  fp32s.reserve(fp16s.size());
  for (const half& fp16 : fp16s) fp32s.push_back(static_cast<float>(fp16));
#endif  // defined(__ANDROID__) && defined(__aarch64__)
  return fp32s;
}

std::vector<float> ConvertOihwFp16ToOhwiFp32(absl::Span<const half> fp16s,
                                             const OHWI& shape) {
  std::vector<float> fp32s;
  const int o_size = shape.o;
  const int h_size = shape.h;
  const int w_size = shape.w;
  const int i_size = shape.i;
  const int hw_size = h_size * w_size;
  const int hwi_size = h_size * w_size * i_size;
  fp32s.resize(shape.DimensionsProduct() + XNN_EXTRA_BYTES / sizeof(float));
#if defined(__ANDROID__) && defined(__aarch64__)
  float* ohwi = &fp32s[0];
  for (int o = 0; o < o_size; ++o) {
    const auto* oihw = reinterpret_cast<const uint16_t*>(&fp16s[o * hwi_size]);
    for (int h = 0; h < h_size; ++h) {
      for (int w = 0; w < w_size; ++w) {
        ConvertFp16ToFp32ColumnNeon(oihw, hw_size, ohwi, i_size);
        ++oihw;
        ohwi += i_size;
      }
    }
  }
#else
  int counter = 0;
  for (int o = 0; o < o_size; ++o) {
    for (int h = 0; h < h_size; ++h) {
      for (int w = 0; w < w_size; ++w) {
        for (int i = 0; i < i_size; ++i) {
          fp32s[counter++] = static_cast<float>(
              fp16s[o * hwi_size + i * hw_size + h * w_size + w]);
        }
      }
    }
  }
#endif  // defined(__ANDROID__) && defined(__aarch64__)
  return fp32s;
}

}  // namespace internal
}  // namespace ml_drift
