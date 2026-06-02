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

#include "ml_drift/common/kernels/depthwise_conv.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/strings/str_replace.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/work_group_picking.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

namespace {

bool IsSpecializedCase(int channel_multiplier) {
  return channel_multiplier == 1 || channel_multiplier == 2 ||
         channel_multiplier == 4;
}

void AppendToBack(const std::string& value, const std::string& delimiter,
                  std::string* result) {
  if (!result->empty()) {
    *result += delimiter;
  }
  *result += value;
}

std::string GetSrcValue(int channel_multiplier,
                        const std::vector<std::string>& coords,
                        const std::string& value_name) {
  std::string coords_str;
  for (const auto& coord : coords) {
    AppendToBack(coord, ", ", &coords_str);
  }
  std::string c;
  if (channel_multiplier == 1) {
    c += "    " + value_name + " = args.src_tensor.Read(" + coords_str +
         ", S);\n";
  } else if (channel_multiplier == 2) {
    c += "    {int s_layer = S / 2;\n";
    c += "    Type src = args.src_tensor.Read(" + coords_str + ", s_layer);\n";
    c += "    SType t0 = S % 2 == 0 ? src.x : src.z;\n";
    c += "    SType t1 = S % 2 == 0 ? src.y : src.w;\n";
    c += "    " + value_name + " = ucl::Init<Type>(t0, t0, t1, t1);}\n";
  } else if (channel_multiplier == 4) {
    c += "    {int s_layer = S / 4;\n";
    c += "    Type src = args.src_tensor.Read(" + coords_str + ", s_layer);\n";
    c += "    SType t0 = src.x;\n";
    c += "    int reminder = S % 4;\n";
    c += "    if (reminder == 1) t0 = src.y;\n";
    c += "    if (reminder == 2) t0 = src.z;\n";
    c += "    if (reminder == 3) t0 = src.w;\n";
    c += "    " + value_name + " = ucl::Init<Type>(t0, t0, t0, t0);}\n";
  } else {
    c += "    {int s_layer = S / args.ch_multiplier;\n";
    c += "    Type src = args.src_tensor.Read(" + coords_str + ", s_layer);\n";
    c += "    int s_offset = (S % args.ch_multiplier) * 4;\n";
    c += "    SType temp_arr[4];\n";
    c += "    temp_arr[0] = src.x;\n";
    c += "    temp_arr[1] = src.y;\n";
    c += "    temp_arr[2] = src.z;\n";
    c += "    temp_arr[3] = src.w;\n";
    c += "    src.x = temp_arr[(s_offset + 0) / args.ch_multiplier];\n";
    c += "    src.y = temp_arr[(s_offset + 1) / args.ch_multiplier];\n";
    c += "    src.z = temp_arr[(s_offset + 2) / args.ch_multiplier];\n";
    c += "    src.w = temp_arr[(s_offset + 3) / args.ch_multiplier];\n";
    c += "    " + value_name + " = src;}\n";
  }

  return c;
}

std::string GetSrcXYCheck(const GpuInfo& gpu_info,
                          const TensorDescriptor& src_desc,
                          const std::string& x_coord,
                          const std::string& y_coord) {
  std::string result;
  if (!src_desc.SupportsZeroClamp(Axis::WIDTH, gpu_info)) {
    const std::string x_check =
        x_coord + " >= 0 && " + x_coord + " < args.src_tensor.Width()";
    AppendToBack(x_check, " && ", &result);
  }
  if (!src_desc.SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
    const std::string y_check =
        y_coord + " >= 0 && " + y_coord + " < args.src_tensor.Height()";
    AppendToBack(y_check, " && ", &result);
  }
  return result;
}

bool UseBuffersForWeights(const GpuInfo& gpu_info) {
  if (gpu_info.IsApple() &&
      gpu_info.apple_info.IsFamilyOrLower(AppleInfo::Family::kApple2)) {
    return false;
  }
  return !gpu_info.SupportsImages() || gpu_info.IsMali() ||
         gpu_info.IsBroadcom() || gpu_info.IsApple() || gpu_info.IsAMD();
}
}  // namespace

DepthwiseConv::DepthwiseConv(const OperationDef& definition,
                             const DepthwiseConvParams& params)
    : params_(params) {
  if (params.UseLocalMem()) {
    work_group_size_ = params.work_group_size;
  }
}

int3 DepthwiseConv::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = dst_[0]->Height() * dst_[0]->Depth();
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

std::vector<int3> DepthwiseConv::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info) const {
  if (params_.UseLocalMem()) {
    return {work_group_size_};
  }
  return GetPossibleWorkGroups(tuning_type, gpu_info, kernel_info, grid_size_);
}

std::string DepthwiseConv::GenerateSrcUpload(const GpuInfo& gpu_info,
                                             const OperationDef& op_def) {
  int cache_size_x = params_.work_group_size.x +
                     params_.x_kernel_size * params_.x_dilation_size - 1;
  int cache_size_y = params_.work_group_size.y +
                     params_.y_kernel_size * params_.y_dilation_size - 1;
  int groups_x = DivideRoundUp(cache_size_x, params_.work_group_size.x);
  int groups_y = DivideRoundUp(cache_size_y, params_.work_group_size.y);
  std::string c;
  c += "  __local Type spatial_cache[" + std::to_string(cache_size_y) + "][" +
       std::to_string(cache_size_x) + "];\n";
  for (int gr_y = 0; gr_y < groups_y; ++gr_y) {
    std::string y_offset = std::to_string(params_.work_group_size.y * gr_y);
    std::string ys = "(y_src + " + y_offset + ")";
    std::string ly = "(ucl::GetLocalId<1>() + " + y_offset + ")";
    for (int gr_x = 0; gr_x < groups_x; ++gr_x) {
      std::string x_offset = std::to_string(params_.work_group_size.x * gr_x);
      std::string xs = "(x_src + " + x_offset + ")";
      std::string lx = "(ucl::GetLocalId<0>() + " + x_offset + ")";
      std::string value = "spatial_cache[" + ly + "][" + lx + "]";
      std::string src_value_read_instructions =
          GetSrcValue(params_.channel_multiplier, {xs, ys}, value);
      std::string check =
          GetSrcXYCheck(gpu_info, op_def.src_tensors[0], xs, ys);
      c += "  if (" + lx + " < " + std::to_string(cache_size_x) + " && " + ly +
           " < " + std::to_string(cache_size_y) + ") {\n";
      if (check.empty()) {
        c += src_value_read_instructions;
      } else {
        c += "    if (" + check + ") {\n";
        c += src_value_read_instructions;
        c += "    } else {\n";
        c += "      " + value + " = ucl::Init<Type>(0.0f);\n";
        c += "    }\n";
      }
      c += "  }\n";
    }
  }
  return c;
}

std::string DepthwiseConv::GenerateWeightsUpload(const GpuInfo& gpu_info) {
  const bool weights_are_buffer = UseBuffersForWeights(gpu_info);
  auto read_weight = [](bool weights_are_buffer, const std::string& lid,
                        int work_group_total_size) {
    if (weights_are_buffer) {
      return "args.weights.Read(S * args.kernels_total_size + " + lid + ")";
    } else {
      return "args.weights.Read(" + lid + ", S)";
    }
  };
  std::string c;
  const int work_group_total_size = params_.GetWorkGroupTotalSize();
  c += "  __local Type weights_cache[" +
       std::to_string(params_.GetKernelsTotalSize()) + "];\n";
  c += "  int linear_local_id = (ucl::GetLocalId<2>() * ucl::GetGroupSize<1>() "
       "+ "
       "ucl::GetLocalId<1>()) * "
       "ucl::GetGroupSize<0>() + ucl::GetLocalId<0>();\n";
  const int groups = params_.GetKernelsTotalSize() / work_group_total_size;
  const int reminder = params_.GetKernelsTotalSize() % work_group_total_size;
  for (int i = 0; i < groups; ++i) {
    const std::string lid =
        "linear_local_id + " + std::to_string(work_group_total_size * i);
    c += "  weights_cache[" + lid +
         "] = " + read_weight(weights_are_buffer, lid, work_group_total_size) +
         ";\n";
  }
  if (reminder != 0) {
    const std::string lid =
        "linear_local_id + " + std::to_string(work_group_total_size * groups);
    c += "  if (linear_local_id < " + std::to_string(reminder) + ") {\n";
    c += "    weights_cache[" + lid +
         "] = " + read_weight(weights_are_buffer, lid, work_group_total_size) +
         ";\n";
    c += "  }\n";
  }
  return c;
}

std::string DepthwiseConv::GenerateCode(const GpuInfo& gpu_info,
                                        const OperationDef& op_def,
                                        CalculationsPrecision precision,
                                        bool has_bias) {
  const bool weights_are_buffer = UseBuffersForWeights(gpu_info);
  const bool dynamic_weights = op_def.src_tensors.size() == 2;
  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  if (dynamic_weights) {
    AddSrcTensor("weights", op_def.src_tensors[1]);
  }
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);

  std::string c;

  const auto& src_desc = op_def.src_tensors[0];
  c += "MAIN_FUNCTION($0) {\n";
  if (src_desc.HasAxis(Axis::BATCH)) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = ucl::GetGlobalId<0>();\n";
  }
  if (src_desc.HasAxis(Axis::DEPTH)) {
    c += "  int linear_id_1 = ucl::GetGlobalId<1>();\n";
    c += "  int Y = linear_id_1 / args.dst_tensor.Depth();\n";
    c += "  int Z = linear_id_1 % args.dst_tensor.Depth();\n";
  } else {
    c += "  int Y = ucl::GetGlobalId<1>();\n";
  }
  c += "  int S = ucl::GetGlobalId<2>();\n";
  c += "  int x_src = X * args.stride_x + args.padding_x;\n";
  c += "  int y_src = Y * args.stride_y + args.padding_y;\n";
  if (src_desc.HasAxis(Axis::DEPTH)) {
    c += "  int z_src = Z * args.stride_z + args.padding_z;\n";
  }
  if (params_.use_spatial_caching) {
    c += GenerateSrcUpload(gpu_info, op_def);
  }
  if (params_.use_weights_caching) {
    c += GenerateWeightsUpload(gpu_info);
  }
  if (params_.UseLocalMem()) {
    c += "  ucl::SyncThreads<WorkGroup, Local>();\n";
  }
  c += "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || "
       "S >= args.dst_tensor.Slices()) { \n";
  c += "    return; \n";
  c += "  } \n";
  c += "  AccType r = ucl::Init<AccType>(0.0f);\n";
  if (!dynamic_weights && !params_.use_weights_caching) {
    if (weights_are_buffer) {
      c += "  int fx_c = S * args.kernels_total_size;\n";
    } else {
      c += "  int fx_c = 0;\n";
    }
  }
  std::string kernel_size_x =
      dynamic_weights ? "args.weights.Width()" : "args.kernel_size_x";
  std::string kernel_size_y =
      dynamic_weights ? "args.weights.Height()" : "args.kernel_size_y";
  std::string kernel_size_z =
      dynamic_weights ? "args.weights.Depth()" : "args.kernel_size_z";
  if (params_.UseLocalMem()) {
    kernel_size_x = std::to_string(params_.x_kernel_size);
    kernel_size_y = std::to_string(params_.y_kernel_size);
    kernel_size_z = std::to_string(params_.z_kernel_size);
  }

  std::string check;
  std::vector<std::string> coords;
  if (src_desc.HasAxis(Axis::DEPTH)) {
    c += "  for (int kz = 0; kz < " + kernel_size_z + "; ++kz) {\n";
    if (!params_.use_spatial_caching) {
      c += "    int z_c = z_src + kz * args.dilation_z;\n";
      coords.insert(coords.begin(), "z_c");
      if (!src_desc.SupportsZeroClamp(Axis::DEPTH, gpu_info)) {
        c += "    bool inside_z = z_c >= 0 && z_c < args.src_tensor.Depth();\n";
        c += "    z_c = clamp(z_c, 0, args.src_tensor.Depth() - 1);\n";
        AppendToBack("inside_z", " && ", &check);
      }
    }
  }
  if (src_desc.HasAxis(Axis::HEIGHT)) {
    c += "  for (int ky = 0; ky < " + kernel_size_y + "; ++ky) {\n";
    if (!params_.use_spatial_caching) {
      c += "    int y_c = y_src + ky * args.dilation_y;\n";
      coords.insert(coords.begin(), "y_c");
      if (!src_desc.SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
        c +=
            "    bool inside_y = y_c >= 0 && y_c < args.src_tensor.Height();\n";
        c += "    y_c = clamp(y_c, 0, args.src_tensor.Height() - 1);\n";
        AppendToBack("inside_y", " && ", &check);
      }
    }
  }
  if (src_desc.HasAxis(Axis::WIDTH)) {
    c += "  for (int kx = 0; kx < " + kernel_size_x + "; ++kx) {\n";
    if (!params_.use_spatial_caching) {
      c += "    int x_c = x_src + kx * args.dilation_x;\n";
      coords.insert(coords.begin(), "x_c");
      if (!src_desc.SupportsZeroClamp(Axis::WIDTH, gpu_info)) {
        c += "    bool inside_x = x_c >= 0 && x_c < args.src_tensor.Width();\n";
        c += "    x_c = clamp(x_c, 0, args.src_tensor.Width() - 1);\n";
        AppendToBack("inside_x", " && ", &check);
      }
    }
  }
  std::string weight_value;
  if (params_.use_weights_caching) {
    std::string weight_index = "ky";
    if (src_desc.HasAxis(Axis::DEPTH)) {
      weight_index =
          "(kz * " + std::to_string(params_.y_kernel_size) + " + ky)";
    }
    weight_value = "weights_cache[" + weight_index + " * " +
                   std::to_string(params_.x_kernel_size) + " + kx]";
  } else {
    weight_value = "f";
    if (dynamic_weights) {
      c += "    Type f = args.weights.Read(kx, ky, S);\n";
    } else {
      if (weights_are_buffer) {
        c += "    Type f = args.weights.Read(fx_c);\n";
      } else {
        c += "    Type f = args.weights.Read(fx_c, S);\n";
      }
    }
  }
  std::string src_value;
  if (params_.use_spatial_caching) {
    std::string loc_x = params_.x_dilation_size == 1
                            ? "kx"
                            : "kx * " + std::to_string(params_.x_dilation_size);
    std::string loc_y = params_.y_dilation_size == 1
                            ? "ky"
                            : "ky * " + std::to_string(params_.y_dilation_size);
    src_value = "spatial_cache[ucl::GetLocalId<1>() + " + loc_y +
                "][ucl::GetLocalId<0>() + " + loc_x + "]";
  } else {
    c += "    Type src_final;\n";
    src_value = "src_final";
    c += GetSrcValue(params_.channel_multiplier, coords, src_value);
    if (!check.empty()) {
      c += "    src_final = src_final * ucl::Convert<SType>(" + check + ");\n";
    }
  }
  c += "    r += ucl::Convert<AccType>(" + src_value + " * " + weight_value +
       ");\n";
  if (!dynamic_weights && !params_.use_weights_caching) {
    c += "    fx_c++;\n";
  }
  if (src_desc.HasAxis(Axis::WIDTH)) {
    c += "  }\n";
  }
  if (src_desc.HasAxis(Axis::HEIGHT)) {
    c += "  }\n";
  }
  if (src_desc.HasAxis(Axis::DEPTH)) {
    c += "  }\n";
  }
  c += "  Type res0 = ucl::Convert<Type>(r);\n";
  if (has_bias) {
    c += "  res0 += args.biases.Read(S);\n";
  }
  if (src_desc.HasAxis(Axis::DEPTH)) {
    c += "  args.dst_tensor.Write(res0, X, Y, Z, S);\n";
  } else {
    c += "  args.dst_tensor.Write(res0, X, Y, S);\n";
  }
  c += "}\n";
  const DataType acc_type = precision == CalculationsPrecision::F16
                                ? DataType::FLOAT16
                                : DataType::FLOAT32;
  const DataType type = op_def.src_tensors[0].GetDataType();
  absl::StrReplaceAll({{"SType", ToUclDataType(type, 1)},
                       {"Type", ToUclDataType(type, 4)},
                       {"AccType", ToUclDataType(acc_type, 4)}},
                      &c);
  return c;
}

void DepthwiseConv::UploadWeightsForDWConv2D(
    const Tensor<OHWI, DataType::FLOAT32>& weights, DataType dst_type,
    bool weights_are_buffer) {
  const int dst_channels = weights.shape.i * weights.shape.o;
  const int dst_slices = DivideRoundUp(dst_channels, 4);
  const int kernel_x = weights.shape.w;
  const int kernel_y = weights.shape.h;

  const int elements_count = kernel_x * kernel_y * dst_slices;

  std::vector<uint8_t> data(SizeOf(dst_type) * 4 * elements_count);

  if (dst_type == DataType::FLOAT32) {
    float4* ptr = reinterpret_cast<float4*>(data.data());
    RearrangeWeightsForDWConv2D(weights, absl::MakeSpan(ptr, elements_count));
  } else {  // f16
    half4* ptr = reinterpret_cast<half4*>(data.data());
    RearrangeWeightsForDWConv2D(weights, absl::MakeSpan(ptr, elements_count));
  }

  if (weights_are_buffer) {
    BufferDescriptor desc;
    desc.element_type = dst_type;
    desc.element_size = 4;
    desc.size = SizeOf(dst_type) * 4 * elements_count;
    desc.data = std::move(data);
    args_.AddObject("weights", std::make_unique<BufferDescriptor>(desc));
  } else {
    TensorDescriptor desc = CreateConstantHWVec4TensorDescriptor(
        dst_type, TensorStorageType::TEXTURE_2D, kernel_x * kernel_y,
        dst_slices, data.data());
    args_.AddObject("weights", std::make_unique<TensorDescriptor>(desc));
  }
}

DepthwiseConv CreateDepthwiseConvolution2D(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const DepthwiseConvolution2DAttributes& attr) {
  const bool weights_are_buffer = UseBuffersForWeights(gpu_info);
  DepthwiseConv::DepthwiseConvParams params;
  const auto& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  params.channel_multiplier = weights_shape.o;
  if (gpu_info.IsAMD()) {
    if (attr.strides.w == 1 && attr.strides.h == 1 && attr.dilations.w == 1 &&
        attr.dilations.h == 1 && weights_shape.w * weights_shape.h >= 10) {
      params.use_weights_caching = true;
      params.use_spatial_caching = true;
      params.x_kernel_size = weights_shape.w;
      params.y_kernel_size = weights_shape.h;
      params.x_dilation_size = attr.dilations.w;
      params.y_dilation_size = attr.dilations.h;
      params.work_group_size = int3(16, 16, 1);
    }
  }
  DepthwiseConv op(definition, params);
  op.args_.AddInt("kernel_size_x", weights_shape.w);
  op.args_.AddInt("stride_x", attr.strides.w);
  op.args_.AddInt("padding_x", -attr.padding.prepended.w);
  op.args_.AddInt("dilation_x", attr.dilations.w);
  op.args_.AddInt("kernel_size_y", weights_shape.h);
  op.args_.AddInt("stride_y", attr.strides.h);
  op.args_.AddInt("padding_y", -attr.padding.prepended.h);
  op.args_.AddInt("dilation_y", attr.dilations.h);
  op.args_.AddInt("kernels_total_size", weights_shape.w * weights_shape.h);
  if (!IsSpecializedCase(weights_shape.o)) {
    op.args_.AddInt("ch_multiplier", weights_shape.o);
  }
  bool has_bias = !attr.bias.data.empty();
  op.code_ = op.GenerateCode(gpu_info, definition, precision, has_bias);
  const DataType weights_data_type = precision == CalculationsPrecision::F32
                                         ? DataType::FLOAT32
                                         : DataType::FLOAT16;
  op.UploadWeightsForDWConv2D(GetFloatWeights(attr), weights_data_type,
                              weights_are_buffer);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  if (has_bias) {
    TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
        gpu_info, definition.src_tensors[0].GetDataType(), attr.bias);
    op.args_.AddObject("biases", std::make_unique<TensorDescriptor>(
                                     std::move(bias_tensor_desc)));
  }
  return op;
}

DepthwiseConv CreateDepthwiseConvolution2DExternalWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const DepthwiseConvolution2DAttributes& attr) {
  DepthwiseConv::DepthwiseConvParams params;
  params.channel_multiplier = 1;
  DepthwiseConv op(definition, params);
  op.args_.AddInt("stride_x", attr.strides.w);
  op.args_.AddInt("padding_x", -attr.padding.prepended.w);
  op.args_.AddInt("dilation_x", attr.dilations.w);
  op.args_.AddInt("stride_y", attr.strides.h);
  op.args_.AddInt("padding_y", -attr.padding.prepended.h);
  op.args_.AddInt("dilation_y", attr.dilations.h);
  bool has_bias = !attr.bias.data.empty();
  op.code_ = op.GenerateCode(gpu_info, definition, precision, has_bias);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  if (has_bias) {
    TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
        gpu_info, definition.src_tensors[0].GetDataType(), attr.bias);
    op.args_.AddObject("biases", std::make_unique<TensorDescriptor>(
                                     std::move(bias_tensor_desc)));
  }
  return op;
}

DepthwiseConv CreateDepthwiseConvolution3D(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision,
    const DepthwiseConvolution3DAttributes& attr) {
  const bool weights_are_buffer = UseBuffersForWeights(gpu_info);
  DepthwiseConv::DepthwiseConvParams params;
  params.channel_multiplier = attr.weights.shape.o;
  DepthwiseConv op(definition, params);
  op.args_.AddInt("kernel_size_x", attr.weights.shape.w);
  op.args_.AddInt("stride_x", attr.strides.w);
  op.args_.AddInt("padding_x", -attr.padding.prepended.w);
  op.args_.AddInt("dilation_x", attr.dilations.w);
  op.args_.AddInt("kernel_size_y", attr.weights.shape.h);
  op.args_.AddInt("stride_y", attr.strides.h);
  op.args_.AddInt("padding_y", -attr.padding.prepended.h);
  op.args_.AddInt("dilation_y", attr.dilations.h);
  op.args_.AddInt("kernel_size_z", attr.weights.shape.d);
  op.args_.AddInt("stride_z", attr.strides.d);
  op.args_.AddInt("padding_z", -attr.padding.prepended.d);
  op.args_.AddInt("dilation_z", attr.dilations.d);
  op.args_.AddInt(
      "kernels_total_size",
      attr.weights.shape.w * attr.weights.shape.h * attr.weights.shape.d);
  if (!IsSpecializedCase(attr.weights.shape.o)) {
    op.args_.AddInt("ch_multiplier", attr.weights.shape.o);
  }
  bool has_bias = !attr.bias.data.empty();
  op.code_ = op.GenerateCode(gpu_info, definition, precision, has_bias);
  const DataType weights_data_type = precision == CalculationsPrecision::F32
                                         ? DataType::FLOAT32
                                         : DataType::FLOAT16;
  op.UploadWeightsForDWConv3D(attr.weights, weights_data_type,
                              weights_are_buffer);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  if (has_bias) {
    TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
        gpu_info, definition.src_tensors[0].GetDataType(), attr.bias);
    op.args_.AddObject("biases", std::make_unique<TensorDescriptor>(
                                     std::move(bias_tensor_desc)));
  }
  return op;
}

}  // namespace ml_drift
