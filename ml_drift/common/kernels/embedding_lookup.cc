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

#include "ml_drift/common/kernels/embedding_lookup.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernels/fully_connected.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

namespace {

std::string GetCreateEmbeddingLookupCode(bool is_weights_texture,
                                         const DataType& weights_type,
                                         bool grouped_quantization,
                                         bool has_zero_point,
                                         Axis lookup_axis) {
  std::string c = R"(MAIN_FUNCTION($0) {
  int linear_xb = ucl::GetGlobalId<0>();
  int X = linear_xb / args.dst_tensor.Batch();
  int B = linear_xb % args.dst_tensor.Batch();
  args.src_tensor.SetBatchRef(B);
  args.dst_tensor.SetBatchRef(B);
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height() || S >= args.dst_tensor.Slices()) {
    return;
  }
)";
  if (lookup_axis == Axis::kChannels) {
    c += "  int index;\n";
    c += "  args.src_tensor.ReadPerChannel<int>(index, 0, 0, X);\n";
  } else if (lookup_axis == Axis::kWidth) {
    c += "  int index = args.src_tensor.Read<int>(X, 0, 0).x;\n";
  }
  c += R"(
  // Not necessary if we assume that input is correct. But this is not the case in profiling or running with dummy weights.
  index = max(index, 0);
  index = min(index, args.emb_size - 1);
  int weights_output_ch = index;
  int weights_input_slice = S;
  int weights_output_slice = weights_output_ch / 4;
  int linear_i4o4 = weights_input_slice * args.weights_output_slices + weights_output_slice;
  Type w0;
  Type w1;
  Type w2;
  Type w3;
)";
  if (is_weights_texture) {
    c += "  int o_local_id = weights_output_slice % args.o_group_size;\n";
    c += "  int o_group_id = weights_output_slice / args.o_group_size;\n";
  }
  if (weights_type == DataType::kUint8) {
    if (is_weights_texture) {
      c += "  uint4 w = args.weights.Read(o_local_id, "
           "weights_input_slice * args.o_groups + o_group_id);\n";
    } else {
      c += "  uint4 w = args.weights.Read(linear_i4o4);\n";
    }
    c += "  ucl::U32x4ToU8x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
  } else if (weights_type == DataType::kUint4) {
    if (is_weights_texture) {
      c += "  ushort4 w = args.weights.Read(o_local_id, "
           "weights_input_slice * args.o_groups + o_group_id);\n";
      c += "  ucl::U16x4ToU4x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
    } else {
      c += "  uint2 w = args.weights.Read(linear_i4o4);\n";
      c += "  ucl::U32x2ToU4x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
    }
  } else if (weights_type == DataType::kUint2) {
    if (is_weights_texture) {
      c += "  uchar4 w = args.weights.Read(o_local_id, "
           "weights_input_slice * args.o_groups + o_group_id);\n";
      c += "  ucl::U8x4ToU2x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
    } else {
      c += "  uint w = args.weights.Read(linear_i4o4);\n";
      c += "  ucl::U32x1ToU2x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
    }
  } else {
    c += "  args.weights.ReadVec16AsVec4x4<SType>(w0, w1, w2, w3, "
         "linear_i4o4);\n";
  }
  c += R"(
  Type value;
  int output_sub_ch = weights_output_ch % 4;
  if (output_sub_ch == 0) {
    value.x = w0.x;
    value.y = w1.x;
    value.z = w2.x;
    value.w = w3.x;
  } else if (output_sub_ch == 1) {
    value.x = w0.y;
    value.y = w1.y;
    value.z = w2.y;
    value.w = w3.y;
  } else if (output_sub_ch == 2) {
    value.x = w0.z;
    value.y = w1.z;
    value.z = w2.z;
    value.w = w3.z;
  } else if (output_sub_ch == 3) {
    value.x = w0.w;
    value.y = w1.w;
    value.z = w2.w;
    value.w = w3.w;
  }
)";
  if (weights_type == DataType::kUint8 || weights_type == DataType::kUint4 ||
      weights_type == DataType::kUint2) {
    c += "  SType scale;\n";
    c += "  SType zero_point = ucl::Init<SType>(0);\n";
    if (grouped_quantization) {
      c += R"(
  // weights_scale_height equals to quantization groups count
  int scale_zp_group_size = (args.weights_input_channels / args.weights_scale.Slices()) / 4;
  int src_scale_zp_group_id = weights_input_slice / scale_zp_group_size;
  Type weight_scale = args.weights_scale.Read<SType>(weights_output_slice, 0, src_scale_zp_group_id);
    )";
      if (has_zero_point) {
        c += "  Type weight_zero_point = "
             "args.weights_zero_point.Read<SType>(weights_output_slice, 0, "
             "src_scale_zp_group_id);\n";
      } else {
        c += "  Type weight_zero_point = ucl::Init<Type>(0);\n";
      }
      c += R"(
  if (index % 4 == 0) {
    scale = weight_scale.x;
    zero_point = weight_zero_point.x;
  } else if (index % 4 == 1) {
    scale = weight_scale.y;
    zero_point = weight_zero_point.y;
  } else if (index % 4 == 2) {
    scale = weight_scale.z;
    zero_point = weight_zero_point.z;
  } else {
    scale = weight_scale.w;
    zero_point = weight_zero_point.w;
  }
)";
    } else {
      c += "  args.weights_scale.ReadPerChannel<SType>(scale, index);\n";
      if (has_zero_point) {
        c += "  args.weights_zero_point.ReadPerChannel<SType>(zero_point, "
             "index);\n";
      }
    }
    // TODO: Centralize shifting logic with all quantization ops.
    std::string shift = "ucl::Init<SType>(128.0f)";
    if (weights_type == DataType::kUint4) {
      shift = "ucl::Init<SType>(8.0f)";
    } else if (weights_type == DataType::kUint2) {
      shift = "ucl::Init<SType>(2.0f)";
    }
    c += "  SType weight_bias = -scale * (" + shift + " + zero_point);\n";
    c += "  value = value * scale + weight_bias;\n";
  }
  c += R"(
  args.dst_tensor::type dst_value = ucl::Convert<args.dst_tensor::type>(value);
  args.dst_tensor.Write(dst_value, X, Y, S);
})";

  absl::StrReplaceAll({{"SType", "float"}, {"Type", "float4"}}, &c);

  return c;
}

}  // namespace

absl::StatusOr<GPUOperation> EmbeddingLookup(
    const OperationDef& op_def, const GpuInfo& gpu_info,
    const EmbeddingLookupAttributes& attr) {
  GPUOperation op;
  op.AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  op.AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  WeightsDescription weights_desc;
  if (attr.weights_type == EmbeddingLookupAttributes::WeightsType::kInt8) {
    weights_desc =
        GetFullyConnectedInt8WeightsDesc(gpu_info, attr.original_weights_shape);
  } else if (attr.weights_type ==
             EmbeddingLookupAttributes::WeightsType::kInt4) {
    weights_desc =
        GetFullyConnectedInt4WeightsDesc(gpu_info, attr.original_weights_shape);
  } else if (attr.weights_type ==
             EmbeddingLookupAttributes::WeightsType::kInt2) {
    weights_desc =
        GetFullyConnectedInt2WeightsDesc(gpu_info, attr.original_weights_shape);
  } else {
    weights_desc = GetFullyConnectedWeightsDesc(
        op_def.dst_tensors[0].GetDataType(), attr.original_weights_shape);
  }
  bool is_weights_texture =
      weights_desc.layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;
  if (is_weights_texture) {
    const int group_size = weights_desc.GetOutputGroupSize();
    op.args_.AddInt("o_group_size", group_size);
    const int dst_slices = DivideRoundUp(attr.original_weights_shape.o, 4);
    op.args_.AddInt("o_groups", DivideRoundUp(dst_slices, group_size));
  }
  TensorDescriptor dst_tensor_desc = op_def.dst_tensors[0];
  if (weights_desc.type == DataType::kUint8) {
    if (is_weights_texture) {
      const int elements_count = GetTotalElementsCountForLayout(
          weights_desc, attr.original_weights_shape);

      std::vector<uint8_t> weights_data(elements_count *
                                        SizeOf(weights_desc.type));
      Tensor<OHWI, DataType::kInt8> int8_weights =
          std::get<Tensor<OHWI, DataType::kInt8>>(attr.weights);
      RearrangeWeightsInt8AsUint8(int8_weights, weights_desc,
                                  absl::MakeSpan(weights_data), 128, 128u);
      uint2 tex_size =
          Get2dResourceSize(weights_desc, attr.original_weights_shape);
      tex_size.x /= 4;  // because we store 4 uint8 as one uint32
      TensorDescriptor desc = CreateConstantHWVec4TensorDescriptor(
          DataType::kUint32, TensorStorageType::kTexture2D, tex_size.x,
          tex_size.y, weights_data.data());
      op.args_.AddObject("weights",
                         std::make_unique<TensorDescriptor>(std::move(desc)));
    } else {
      const int flt_count = GetTotalElementsCountForLayout(
          weights_desc, attr.original_weights_shape);
      std::vector<uint8_t> weights_data(flt_count * SizeOf(weights_desc.type));
      Tensor<OHWI, DataType::kInt8> int8_weights =
          std::get<Tensor<OHWI, DataType::kInt8>>(attr.weights);
      RearrangeWeightsInt8AsUint8(int8_weights, weights_desc,
                                  absl::MakeSpan(weights_data), 128, 128u);
      BufferDescriptor desc;
      desc.element_type = DataType::kUint32;
      desc.element_size = 4;
      desc.size = SizeOf(weights_desc.type) * flt_count;
      desc.data = weights_data;
      op.args_.AddObject("weights",
                         std::make_unique<BufferDescriptor>(std::move(desc)));
    }
  } else if (weights_desc.type == DataType::kUint4) {
    if (is_weights_texture) {
      const int elements_count =
          GetTotalElementsCountForLayout(weights_desc,
                                         attr.original_weights_shape) /
          2;

      std::vector<uint8_t> weights_data(elements_count *
                                        SizeOf(weights_desc.type));
      std::vector<int32_t> weights_sum_i(attr.original_weights_shape.o);
      Tensor<OHWI, DataType::kUint8> int4_weights =
          std::get<Tensor<OHWI, DataType::kUint8>>(attr.weights);
      ABSL_RETURN_IF_ERROR(RearrangeWeightsUInt4Packed(
          int4_weights, weights_desc, absl::MakeSpan(weights_data),
          absl::MakeSpan(weights_sum_i),
          /*pad_value=*/8u, /*swap_dims=*/false));
      uint2 tex_size =
          Get2dResourceSize(weights_desc, attr.original_weights_shape);
      tex_size.x /= 4;  // because we store 4 uint4 as one uint16
      TensorDescriptor desc = CreateConstantHWVec4TensorDescriptor(
          DataType::kUint16, TensorStorageType::kTexture2D, tex_size.x,
          tex_size.y, weights_data.data());
      op.args_.AddObject("weights",
                         std::make_unique<TensorDescriptor>(std::move(desc)));
    } else {
      const int flt_count = GetTotalElementsCountForLayout(
                                weights_desc, attr.original_weights_shape) /
                            2;
      std::vector<uint8_t> weights_data(flt_count * SizeOf(weights_desc.type));
      std::vector<int32_t> weights_sum_i(attr.original_weights_shape.o);
      Tensor<OHWI, DataType::kUint8> int4_weights =
          std::get<Tensor<OHWI, DataType::kUint8>>(attr.weights);
      ABSL_RETURN_IF_ERROR(RearrangeWeightsUInt4Packed(
          int4_weights, weights_desc, absl::MakeSpan(weights_data),
          absl::MakeSpan(weights_sum_i),
          /*pad_value=*/8u, /*swap_dims=*/false));
      BufferDescriptor desc;
      desc.element_type = DataType::kUint32;
      desc.element_size = 2;
      desc.size = SizeOf(weights_desc.type) * flt_count;
      desc.data = weights_data;
      op.args_.AddObject("weights",
                         std::make_unique<BufferDescriptor>(std::move(desc)));
    }
  } else if (weights_desc.type == DataType::kUint2) {
    if (is_weights_texture) {
      const int elements_count =
          GetTotalElementsCountForLayout(weights_desc,
                                         attr.original_weights_shape) /
          4;

      std::vector<uint8_t> weights_data(elements_count *
                                        SizeOf(weights_desc.type));
      std::vector<int32_t> weights_sum_i(attr.original_weights_shape.o);
      Tensor<OHWI, DataType::kUint8> int2_weights =
          std::get<Tensor<OHWI, DataType::kUint8>>(attr.weights);
      ABSL_RETURN_IF_ERROR(RearrangeWeightsUInt2Packed(
          int2_weights, weights_desc, absl::MakeSpan(weights_data),
          absl::MakeSpan(weights_sum_i),
          /*pad_value=*/2u, /*swap_dims=*/false));
      uint2 tex_size =
          Get2dResourceSize(weights_desc, attr.original_weights_shape);
      tex_size.x /= 4;  // because we store 4 uint2 as one uint8
      TensorDescriptor desc = CreateConstantHWVec4TensorDescriptor(
          DataType::kUint8, TensorStorageType::kTexture2D, tex_size.x,
          tex_size.y, weights_data.data());
      op.args_.AddObject("weights",
                         std::make_unique<TensorDescriptor>(std::move(desc)));
    } else {
      const int flt_count = GetTotalElementsCountForLayout(
                                weights_desc, attr.original_weights_shape) /
                            4;
      std::vector<uint8_t> weights_data(flt_count * SizeOf(weights_desc.type));
      std::vector<int32_t> weights_sum_i(attr.original_weights_shape.o);
      Tensor<OHWI, DataType::kUint8> int2_weights =
          std::get<Tensor<OHWI, DataType::kUint8>>(attr.weights);
      ABSL_RETURN_IF_ERROR(RearrangeWeightsUInt2Packed(
          int2_weights, weights_desc, absl::MakeSpan(weights_data),
          absl::MakeSpan(weights_sum_i),
          /*pad_value=*/2u, /*swap_dims=*/false));
      BufferDescriptor desc;
      desc.element_type = DataType::kUint32;
      desc.element_size = 1;
      desc.size = SizeOf(weights_desc.type) * flt_count;
      desc.data = weights_data;
      op.args_.AddObject("weights",
                         std::make_unique<BufferDescriptor>(std::move(desc)));
    }
  } else {
    const int flt_count = GetTotalElementsCountForLayout(
        weights_desc, attr.original_weights_shape);
    std::vector<uint8_t> weights_data(flt_count * SizeOf(weights_desc.type));
    Tensor<OHWI, DataType::kFloat32> float32_weights =
        std::get<Tensor<OHWI, DataType::kFloat32>>(attr.weights);
    RearrangeWeights(float32_weights, weights_desc,
                     absl::MakeSpan(weights_data));
    BufferDescriptor desc;
    desc.element_type = dst_tensor_desc.GetDataType();
    desc.element_size = 16;
    desc.size = SizeOf(weights_desc.type) * flt_count;
    desc.data = weights_data;
    op.args_.AddObject("weights",
                       std::make_unique<BufferDescriptor>(std::move(desc)));
  }
  op.args_.AddInt("weights_output_slices",
                  DivideRoundUp(attr.original_weights_shape.o, 4));

  bool grouped_quantization = false;
  if (weights_desc.type == DataType::kUint8 ||
      weights_desc.type == DataType::kUint4 ||
      weights_desc.type == DataType::kUint2) {
    auto weights_scale_desc = ScaleOrZeroPointToTensorDesc(
        gpu_info, attr.weights_scale, op_def.dst_tensors[0].GetDataType());
    auto weights_zp_desc = ScaleOrZeroPointToTensorDesc(
        gpu_info, attr.weights_zero_point, op_def.dst_tensors[0].GetDataType());
    grouped_quantization = attr.weights_scale.shape.i != 1;
    op.args_.AddObject("weights_scale", std::make_unique<TensorDescriptor>(
                                            std::move(weights_scale_desc)));
    op.args_.AddObject("weights_zero_point", std::make_unique<TensorDescriptor>(
                                                 std::move(weights_zp_desc)));
    op.args_.AddInt("weights_input_channels", attr.original_weights_shape.i);
  }
  op.args_.AddInt("emb_size", attr.original_weights_shape.o);
  op.code_ = GetCreateEmbeddingLookupCode(
      is_weights_texture, weights_desc.type, grouped_quantization,
      /*has_zero_point=*/true, Axis::kChannels);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

GPUOperation EmbeddingLookupExternalWeights(
    const TensorDescriptor& src, const TensorDescriptor& dst,
    const TensorDescriptor& weights, const WeightsDescription& weights_desc,
    const OHWI& weights_shape, const TensorDescriptor* weights_scale,
    const TensorDescriptor* weights_zero_point, Axis lookup_axis) {
  GPUOperation op;
  op.AddSrcTensor("src_tensor", src);
  op.AddDstTensor("dst_tensor", dst);

  bool is_weights_texture =
      weights_desc.layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;
  if (is_weights_texture) {
    const int group_size = weights_desc.GetOutputGroupSize();
    op.args_.AddInt("o_group_size", group_size);
    const int dst_slices = DivideRoundUp(weights_shape.o, 4);
    op.args_.AddInt("o_groups", DivideRoundUp(dst_slices, group_size));
  }

  bool grouped_quantization = false;
  if (weights_desc.type == DataType::kUint8 ||
      weights_desc.type == DataType::kUint4 ||
      weights_desc.type == DataType::kUint2) {
    // quantized weights
    if (is_weights_texture) {
      op.AddSrcTensor("weights", weights);
    } else {
      BufferDescriptor desc;
      desc.element_type = DataType::kUint32;
      desc.element_size = SizeInBitsOf(weights_desc.type) / 2;
      op.AddSrcBuffer("weights", desc);
    }
    grouped_quantization = weights_scale->GetLayout() != Layout::kLinear;
    op.AddSrcTensor("weights_scale", *weights_scale);
    if (weights_zero_point) {
      op.AddSrcTensor("weights_zero_point", *weights_zero_point);
    }
    op.args_.AddInt("weights_input_channels", weights_shape.i);
  } else {
    BufferDescriptor desc;
    desc.element_type = dst.GetDataType();
    desc.element_size = 16;
    op.AddSrcBuffer("weights", desc);
  }
  op.args_.AddInt("weights_output_slices", DivideRoundUp(weights_shape.o, 4));
  op.args_.AddInt("emb_size", weights_shape.o);
  op.code_ = GetCreateEmbeddingLookupCode(
      is_weights_texture, weights_desc.type, grouped_quantization,
      weights_zero_point != nullptr, lookup_axis);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  return op;
}

// Assume src tensor's BHWC shape is [Batch, 1, 1, SequenceLength].
absl::StatusOr<GPUOperation> CreateEmbeddingLookup(
    const OperationDef& op_def, const GpuInfo& gpu_info,
    const EmbeddingLookupAttributes& attr) {
  return EmbeddingLookup(op_def, gpu_info, attr);
}

GPUOperation CreateEmbeddingLookupExternalWeights(
    const TensorDescriptor& src, const TensorDescriptor& dst,
    const TensorDescriptor& weights, const WeightsDescription& weights_desc,
    const OHWI& weights_shape, const TensorDescriptor* weights_scale,
    const TensorDescriptor* weights_zero_point, Axis lookup_axis) {
  return EmbeddingLookupExternalWeights(src, dst, weights, weights_desc,
                                        weights_shape, weights_scale,
                                        weights_zero_point, lookup_axis);
}

}  // namespace ml_drift
