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

#include "ml_drift/samples/stable_diffusion/autoencoder_kl_builder.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/types.h"
#include "ml_drift/samples/stable_diffusion/util.h"

namespace ml_drift {

absl::Status AutoencoderKLBuilder::BuildDecoder(
    const Config& config, const GpuInfo& gpu_info,
    const CreateGpuModelInfo& create_info, int width, int height,
    GpuModel* gpu_model, GpuModelBuilder::TensorHandle* src_ptr,
    GpuModelBuilder::TensorHandle* dst_ptr, const std::string& file_folder) {
  file_folder_ = file_folder;
  builder_ = GpuModelBuilder(gpu_info, create_info.hints, create_info.precision,
                             create_info.storage_type);
  gpu_info_ = gpu_info;

  auto src_tensor =
      builder_.AddTensor(BHWC(1, height, width, 4),
                         DeduceDataTypeFromPrecision(create_info.precision));
  ABSL_ASSIGN_OR_RETURN(auto t, MakeDecoder(config, src_tensor));

  if (src_ptr) {
    *src_ptr = src_tensor;
  }
  if (dst_ptr) {
    *dst_ptr = t;
  }
  return builder_.GetGpuModel(std::vector<uint32_t>{src_tensor.id},
                              std::vector<uint32_t>{t.id}, gpu_model);
}

absl::Status AutoencoderKLBuilder::BuildEncoder(
    const Config& config, const GpuInfo& gpu_info,
    const CreateGpuModelInfo& create_info, int width, int height,
    GpuModel* gpu_model, GpuModelBuilder::TensorHandle* src_ptr,
    GpuModelBuilder::TensorHandle* dst_ptr, const std::string& file_folder) {
  file_folder_ = file_folder;
  builder_ = GpuModelBuilder(gpu_info, create_info.hints, create_info.precision,
                             create_info.storage_type);

  auto src_tensor =
      builder_.AddTensor(BHWC(1, height, width, 4),
                         DeduceDataTypeFromPrecision(create_info.precision));
  ABSL_ASSIGN_OR_RETURN(auto t, MakeEncoder(config, src_tensor));

  if (src_ptr) {
    *src_ptr = src_tensor;
  }
  if (dst_ptr) {
    *dst_ptr = t;
  }
  return builder_.GetGpuModel(std::vector<uint32_t>{src_tensor.id},
                              std::vector<uint32_t>{t.id}, gpu_model);
}

GpuModelBuilder::TensorHandle AutoencoderKLBuilder::MakeConv(
    const GpuModelBuilder::TensorHandle& src, const std::string& name,
    int out_channels, int kernel_hw, int stride_hw, bool bias) {
  auto weights_shape = OHWI(out_channels, kernel_hw, kernel_hw,
                            src.tensor_desc.GetBHWCShape().c);
  auto weights_data = LoadF16(file_folder_ + name + ".weight.bin",
                              weights_shape.DimensionsProduct());
  auto bias_data =
      bias ? LoadF16(file_folder_ + name + ".bias.bin", weights_shape.o)
           : std::vector<half>();
  return builder_.Convolution(
      src, MakeConvAttributes(weights_data, bias_data, weights_shape,
                              HW(stride_hw, stride_hw), name));
}

GpuModelBuilder::TensorHandle AutoencoderKLBuilder::MakeLinear(
    const GpuModelBuilder::TensorHandle& src, const std::string& name,
    int out_channels, bool bias) {
  return MakeConv(src, name, out_channels, 1, 1, bias);
}

GpuModelBuilder::TensorHandle AutoencoderKLBuilder::MakeGroupNorm(
    const GpuModelBuilder::TensorHandle& src, const std::string& name) {
  auto gamma = CreateLinearTensor(LoadF16(file_folder_ + name + ".weight.bin",
                                          src.tensor_desc.GetBHWCShape().c));
  auto beta = CreateLinearTensor(LoadF16(file_folder_ + name + ".bias.bin",
                                         src.tensor_desc.GetBHWCShape().c));
  return builder_.HWCGroupNorm(src, /*groups*/ 32, /*epsilon*/ 1e-5f, gamma,
                               beta);
}

GpuModelBuilder::TensorHandle AutoencoderKLBuilder::Upsample(
    const GpuModelBuilder::TensorHandle& src, bool with_conv,
    const std::string& name) {
  auto x = builder_.ResizeNearest(src, 2, false, true);
  if (with_conv) {
    x = MakeConv(x, name + ".conv", x.tensor_desc.GetBHWCShape().c, 3);
  }
  return x;
}

GpuModelBuilder::TensorHandle AutoencoderKLBuilder::PixelShuffle(
    const GpuModelBuilder::TensorHandle& src, int block_size,
    const std::string& name) {
  return builder_.DepthToSpace(src, block_size);
}

GpuModelBuilder::TensorHandle AutoencoderKLBuilder::Downsample(
    const GpuModelBuilder::TensorHandle& src, bool with_conv,
    const std::string& name) {
  if (with_conv) {
    return MakeConv(src, name + ".conv", src.tensor_desc.GetBHWCShape().c, 3,
                    2);
  } else {
    Pooling2DAttributes attr;
    attr.padding.prepended = HW(0, 0);
    attr.padding.appended = HW(0, 0);
    attr.strides = HW(2, 2);
    attr.kernel = HW(2, 2);
    attr.type = PoolingType::AVERAGE;
    return builder_.Pooling(src, attr);
  }
}

GpuModelBuilder::TensorHandle AutoencoderKLBuilder::MakeResnetBlock(
    const GpuModelBuilder::TensorHandle& src, const std::string& name,
    int out_channels) {
  auto x = src;
  x = MakeGroupNorm(x, name + ".norm1");
  x = builder_.SiLU(x);
  x = MakeConv(x, name + ".conv1", out_channels, 3);
  x = MakeGroupNorm(x, name + ".norm2");
  x = builder_.SiLU(x);
  x = MakeConv(x, name + ".conv2", out_channels, 3);
  if (src.tensor_desc.GetBHWCShape().c != out_channels) {
    auto ninShortcut = MakeConv(src, name + ".nin_shortcut", out_channels, 1);
    return builder_.Add(x, ninShortcut);
  }
  return builder_.Add(x, src);
}

absl::StatusOr<GpuModelBuilder::TensorHandle>
AutoencoderKLBuilder::MakeAttention(const GpuModelBuilder::TensorHandle& src,
                                    const std::string& name) {
  auto src_sh = src.tensor_desc.GetBHWCShape();

  auto x = src;
  x = MakeGroupNorm(x, name + ".norm");
  x = builder_.Reshape(x, BHWC(1, src_sh.b, src_sh.h * src_sh.w, src_sh.c));
  auto q = MakeLinear(x, name + ".q", src_sh.c, false);
  auto k = MakeLinear(x, name + ".k", src_sh.c, false);
  k = builder_.Multiplication(k, 1.0f / sqrt(static_cast<float>(src_sh.c)));
  k = builder_.Transpose(k, BHWC(0, 1, 3, 2));

  auto v = MakeLinear(x, name + ".v", src_sh.c, false);

  ABSL_ASSIGN_OR_RETURN(auto att,
                        builder_.BatchedMatMulSoftmaxBatchedMatMul(q, k, v));

  x = MakeLinear(att, name + ".proj_out", src_sh.c);
  x = builder_.Reshape(x, src_sh);
  return builder_.Add(x, src);
}

absl::StatusOr<GpuModelBuilder::TensorHandle> AutoencoderKLBuilder::MakeDecoder(
    const Config& config, const GpuModelBuilder::TensorHandle& src) {
  auto x = src;
  const std::string name = "first_stage_model.decoder";
  x = builder_.Multiplication(x, 1.0f / 0.18215f);
  x = MakeConv(x, "first_stage_model.post_quant_conv", 4, 1);
  x = MakeConv(x, name + ".conv_in", 512, 3);

  x = MakeResnetBlock(x, name + ".mid.block_1", 512);
  ABSL_ASSIGN_OR_RETURN(x, MakeAttention(x, name + ".mid.attn_1"));
  x = MakeResnetBlock(x, name + ".mid.block_2", 512);

  for (int i_level = config.ch_mult.size() - 1; i_level >= 0; --i_level) {
    const int block_out = config.ch * config.ch_mult[i_level];
    const std::string name_level = name + ".up." + std::to_string(i_level);
    for (int i_block = 0; i_block < config.num_res_blocks + 1; ++i_block) {
      const std::string name_block =
          name_level + ".block." + std::to_string(i_block);
      x = MakeResnetBlock(x, name_block, block_out);
    }
    if (i_level != 0) {
      x = Upsample(x, /*with_conv*/ true, name_level + ".upsample");
    }
  }

  x = MakeGroupNorm(x, name + ".norm_out");
  x = builder_.SiLU(x);
  x = MakeConv(x, name + ".conv_out", 3, 3);
  // not part of a model, range conversion [-1, -1] -> [0, 1]
  x = builder_.Add(x, 1.0f);
  return builder_.Multiplication(x, 0.5f);
}

absl::StatusOr<GpuModelBuilder::TensorHandle> AutoencoderKLBuilder::MakeEncoder(
    const Config& config, const GpuModelBuilder::TensorHandle& src) {
  auto h = src;
  const std::string name = "first_stage_model.encoder";

  // downsampling
  h = MakeConv(h, name + ".conv_in", config.ch, 3);
  for (int i_level = 0; i_level < config.ch_mult.size(); ++i_level) {
    const int block_out = config.ch * config.ch_mult[i_level];
    const std::string name_level = name + ".down." + std::to_string(i_level);
    for (int i_block = 0; i_block < config.num_res_blocks; ++i_block) {
      const std::string name_block =
          name_level + ".block." + std::to_string(i_block);
      h = MakeResnetBlock(h, name_block, block_out);
    }
    if (i_level != config.ch_mult.size() - 1) {
      h = Downsample(h, /*with_conv*/ true, name_level + ".downsample");
    }
  }

  // middle
  h = MakeResnetBlock(h, name + ".mid.block_1", 512);
  ABSL_ASSIGN_OR_RETURN(h, MakeAttention(h, name + ".mid.attn_1"));
  h = MakeResnetBlock(h, name + ".mid.block_2", 512);

  // end
  h = MakeGroupNorm(h, name + ".norm_out");
  h = builder_.SiLU(h);
  const int conv_out_ch =
      config.double_z ? 2 * config.z_channels : config.z_channels;
  h = MakeConv(h, name + ".conv_out", conv_out_ch, 3);
  return h;
}

}  // namespace ml_drift
