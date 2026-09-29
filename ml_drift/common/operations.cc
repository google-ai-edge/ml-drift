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

#include "ml_drift/common/operations.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

bool HoldsFloatScalar(const TensorOrScalar& arg) {
  return std::holds_alternative<ScalarValue>(arg) &&
         std::holds_alternative<float>(*std::get_if<ScalarValue>(&arg));
}

const float* GetIfFloatScalar(const TensorOrScalar* arg) {
  const auto scalar = std::get_if<ScalarValue>(arg);
  return scalar ? std::get_if<float>(scalar) : nullptr;
}

bool Padding2D::operator==(const Padding2D& value) const {
  return this->prepended == value.prepended && this->appended == value.appended;
}

bool Padding2D::operator!=(const Padding2D& value) const {
  return !(*this == value);
}

Padding2D& Padding2D::operator-(const Padding2D& value) {
  prepended.h -= value.prepended.h;
  prepended.w -= value.prepended.w;
  appended.h -= value.appended.h;
  appended.w -= value.appended.w;
  return *this;
}

bool Padding3D::operator==(const Padding3D& value) const {
  return this->prepended == value.prepended && this->appended == value.appended;
}

bool Padding3D::operator!=(const Padding3D& value) const {
  return !(*this == value);
}

Padding3D& Padding3D::operator-(const Padding3D& value) {
  prepended.h -= value.prepended.h;
  prepended.w -= value.prepended.w;
  prepended.d -= value.prepended.d;
  appended.h -= value.appended.h;
  appended.w -= value.appended.w;
  appended.d -= value.appended.d;
  return *this;
}

std::string ToString(enum OperationType op) {
  switch (op) {
    case OperationType::kAbs:
      return "abs";
    case OperationType::kAdd:
      return "add";
    case OperationType::kAtan2:
      return "atan2";
    case OperationType::kBatchNormalization:
      return "batch_normalization";
    case OperationType::kBatchToSpace:
      return "batch_to_space";
    case OperationType::kBatchedMatmul:
      return "batched_matmul";
    case OperationType::kBitcast:
      return "bitcast";
    case OperationType::kBroadcastInDim:
      return "broadcast_in_dim";
    case OperationType::kCast:
      return "cast";
    case OperationType::kCeil:
      return "ceil";
    case OperationType::kConcat:
      return "concat";
    case OperationType::kConstant:
      return "const";
    case OperationType::kConvolution2D:
      return "convolution_2d";
    case OperationType::kConvolutionTransposed:
      return "convolution_transposed";
    case OperationType::kCopy:
      return "copy";
    case OperationType::kCos:
      return "cos";
    case OperationType::kCumsum:
      return "cumsum";
    case OperationType::kDepthwiseConvolution:
      return "depthwise_convolution";
    case OperationType::kDepthToSpace:
      return "depth_to_space";
    case OperationType::kDiv:
      return "div";
    case OperationType::kDotGeneral:
      return "dot_general";
    case OperationType::kDynamicUpdateSlice:
      return "dynamic_update_slice";
    case OperationType::kElu:
      return "elu";
    case OperationType::kEmbeddingLookup:
      return "embedding_lookup";
    case OperationType::kEqual:
      return "equal";
    case OperationType::kExp:
      return "exp";
    case OperationType::kFloor:
      return "floor";
    case OperationType::kFloorDiv:
      return "floor_div";
    case OperationType::kFloorMod:
      return "floor_mod";
    case OperationType::kFullyConnected:
      return "fully_connected";
    case OperationType::kFullyConnectedInt2:
      return "fully_connected_int2";
    case OperationType::kFullyConnectedInt4:
      return "fully_connected_int4";
    case OperationType::kFullyConnectedInt8:
      return "fully_connected_int8";
    case OperationType::kGather:
      return "gather";
    case OperationType::kGelu:
      return "gelu";
    case OperationType::kGeluTanhApprox:
      return "gelu_tanh_approx";
    case OperationType::kGreater:
      return "greater";
    case OperationType::kGreaterEqual:
      return "greater_equal";
    case OperationType::kGroupNorm:
      return "group_norm";
    case OperationType::kHardSwish:
      return "hard_swish";
    case OperationType::kLayerNorm:
      return "layer_norm";
    case OperationType::kLess:
      return "less";
    case OperationType::kLessEqual:
      return "less_equal";
    case OperationType::kLog:
      return "log";
    case OperationType::kLogicalNot:
      return "logical_not";
    case OperationType::kLogicalAnd:
      return "logical_and";
    case OperationType::kLogicalOr:
      return "logical_or";
    case OperationType::kLogicalXor:
      return "logical_xor";
    case OperationType::kLstm:
      return "lstm";
    case OperationType::kMaximum:
      return "maximum";
    case OperationType::kMaxIndex:
      return "max_index";
    case OperationType::kMaxUnpooling2D:
      return "max_unpooling";
    case OperationType::kMean:
      return "mean";
    case OperationType::kMeanStddevNormalization:
      return "mean_stddev_normalization";
    case OperationType::kMinimum:
      return "minimum";
    case OperationType::kMish:
      return "mish";
    case OperationType::kMod:
      return "mod";
    case OperationType::kMul:
      return "mul";
    case OperationType::kNeg:
      return "neg";
    case OperationType::kNotEqual:
      return "not_equal";
    case OperationType::kOneHot:
      return "one_hot";
    case OperationType::kPad:
      return "pad";
    case OperationType::kPooling2D:
      return "pooling_2d";
    case OperationType::kPositionalEmbedding:
      return "positional_embedding";
    case OperationType::kPow:
      return "pow";
    case OperationType::kPrelu:
      return "prelu";
    case OperationType::kQuantizeAndDequantize:
      return "quantize_and_dequantize";
    case OperationType::kReduceAll:
      return "reduce_all";
    case OperationType::kReduceAny:
      return "reduce_any";
    case OperationType::kReduceMaximum:
      return "reduce_maximum";
    case OperationType::kReduceMinimum:
      return "reduce_minimum";
    case OperationType::kReduceProduct:
      return "reduce_product";
    case OperationType::kReduceSum:
      return "reduce_sum";
    case OperationType::kRelu:
      return "relu";
    case OperationType::kRemainder:
      return "remainder";
    case OperationType::kResampler:
      return "resampler";
    case OperationType::kReshape:
      return "reshape";
    case OperationType::kResize:
      return "resize";
    case OperationType::kReverse:
      return "reverse";
    case OperationType::kRmsNorm:
      return "rms_norm";
    case OperationType::kRope:
      return "rope";
    case OperationType::kRound:
      return "round";
    case OperationType::kRsqrt:
      return "rsqrt";
    case OperationType::kScaledDotProductAttention:
      return "scaled_dot_product_attention";
    case OperationType::kSelectV2:
      return "select_v2";
    case OperationType::kShiftLeft:
      return "shift_left";
    case OperationType::kShiftRight:
      return "shift_right";
    case OperationType::kSigmoid:
      return "sigmoid";
    case OperationType::kSign:
      return "sign";
    case OperationType::kSin:
      return "sin";
    case OperationType::kSlice:
      return "slice";
    case OperationType::kSoftmax:
      return "softmax";
    case OperationType::kSpaceToBatch:
      return "space_to_batch";
    case OperationType::kSpaceToDepth:
      return "space_to_depth";
    case OperationType::kSplit:
      return "split";
    case OperationType::kSqrt:
      return "sqrt";
    case OperationType::kSquare:
      return "square";
    case OperationType::kSquaredDiff:
      return "squared_diff";
    case OperationType::kSub:
      return "subtract";
    case OperationType::kTanh:
      return "tanh";
    case OperationType::kTile:
      return "tile";
    case OperationType::kTopK:
      return "top_k";
    case OperationType::kTranspose:
      return "transpose";
    case OperationType::kUnknown:
      return "unknown_operation";
  }
}

OperationType OperationTypeFromString(const std::string& name) {
  static const auto operations =
      new absl::flat_hash_map<std::string, OperationType>({
          {"abs", OperationType::kAbs},
          {"add", OperationType::kAdd},
          {"atan2", OperationType::kAtan2},
          {"batch_normalization", OperationType::kBatchNormalization},
          {"batched_matmul", OperationType::kBatchedMatmul},
          {"bitcast", OperationType::kBitcast},
          {"broadcast_in_dim", OperationType::kBroadcastInDim},
          {"cast", OperationType::kCast},
          {"ceil", OperationType::kCeil},
          {"concat", OperationType::kConcat},
          {"const", OperationType::kConstant},
          {"convolution_2d", OperationType::kConvolution2D},
          {"convolution_transposed", OperationType::kConvolutionTransposed},
          {"copy", OperationType::kCopy},
          {"cos", OperationType::kCos},
          {"cumsum", OperationType::kCumsum},
          {"depthwise_convolution", OperationType::kDepthwiseConvolution},
          {"depth_to_space", OperationType::kDepthToSpace},
          {"div", OperationType::kDiv},
          {"dot_general", OperationType::kDotGeneral},
          {"dynamic_update_slice", OperationType::kDynamicUpdateSlice},
          {"elu", OperationType::kElu},
          {"embedding_lookup", OperationType::kEmbeddingLookup},
          {"equal", OperationType::kEqual},
          {"exp", OperationType::kExp},
          {"floor", OperationType::kFloor},
          {"floor_div", OperationType::kFloorDiv},
          {"floor_mod", OperationType::kFloorMod},
          {"fully_connected", OperationType::kFullyConnected},
          {"fully_connected_int2", OperationType::kFullyConnectedInt2},
          {"fully_connected_int4", OperationType::kFullyConnectedInt4},
          {"fully_connected_int8", OperationType::kFullyConnectedInt8},
          {"gather", OperationType::kGather},
          {"gelu", OperationType::kGelu},
          {"gelu_tanh_approx", OperationType::kGeluTanhApprox},
          {"greater", OperationType::kGreater},
          {"greater_equal", OperationType::kGreaterEqual},
          {"group_norm", OperationType::kGroupNorm},
          {"hard_swish", OperationType::kHardSwish},
          {"layer_norm", OperationType::kLayerNorm},
          {"less", OperationType::kLess},
          {"less_equal", OperationType::kLessEqual},
          {"log", OperationType::kLog},
          {"logical_and", OperationType::kLogicalAnd},
          {"logical_not", OperationType::kLogicalNot},
          {"logical_or", OperationType::kLogicalOr},
          {"logical_xor", OperationType::kLogicalXor},
          {"lstm", OperationType::kLstm},
          {"maximum", OperationType::kMaximum},
          {"max_index", OperationType::kMaxIndex},
          {"max_unpooling", OperationType::kMaxUnpooling2D},
          {"mean", OperationType::kMean},
          {"mean_stddev_normalization",
           OperationType::kMeanStddevNormalization},
          {"minimum", OperationType::kMinimum},
          {"mish", OperationType::kMish},
          {"mod", OperationType::kMod},
          {"mul", OperationType::kMul},
          {"neg", OperationType::kNeg},
          {"not_equal", OperationType::kNotEqual},
          {"one_hot", OperationType::kOneHot},
          {"pad", OperationType::kPad},
          {"pooling_2d", OperationType::kPooling2D},
          {"positional_embedding", OperationType::kPositionalEmbedding},
          {"pow", OperationType::kPow},
          {"prelu", OperationType::kPrelu},
          {"quantize_and_dequantize", OperationType::kQuantizeAndDequantize},
          {"reduce_all", OperationType::kReduceAll},
          {"reduce_any", OperationType::kReduceAny},
          {"reduce_maximum", OperationType::kReduceMaximum},
          {"reduce_minimum", OperationType::kReduceMinimum},
          {"reduce_product", OperationType::kReduceProduct},
          {"reduce_sum", OperationType::kReduceSum},
          {"relu", OperationType::kRelu},
          {"remainder", OperationType::kRemainder},
          {"resampler", OperationType::kResampler},
          {"resize", OperationType::kResize},
          {"reshape", OperationType::kReshape},
          {"reverse", OperationType::kReverse},
          {"rms_norm", OperationType::kRmsNorm},
          {"rope", OperationType::kRope},
          {"round", OperationType::kRound},
          {"rsqrt", OperationType::kRsqrt},
          {"scaled_dot_product_attention",
           OperationType::kScaledDotProductAttention},
          {"select_v2", OperationType::kSelectV2},
          {"shift_left", OperationType::kShiftLeft},
          {"shift_right", OperationType::kShiftRight},
          {"sigmoid", OperationType::kSigmoid},
          {"sign", OperationType::kSign},
          {"sin", OperationType::kSin},
          {"slice", OperationType::kSlice},
          {"softmax", OperationType::kSoftmax},
          {"space_to_depth", OperationType::kSpaceToDepth},
          {"split", OperationType::kSplit},
          {"sqrt", OperationType::kSqrt},
          {"square", OperationType::kSquare},
          {"squared_diff", OperationType::kSquaredDiff},
          {"subtract", OperationType::kSub},
          {"tanh", OperationType::kTanh},
          {"tile", OperationType::kTile},
          {"top_k", OperationType::kTopK},
          {"transpose", OperationType::kTranspose},
      });
  auto op = operations->find(name);
  return op == operations->end() ? OperationType::kUnknown : op->second;
}

namespace {

template <typename T>
T DivideRoundUp(T n, T divisor) {
  return (n - 1) / divisor + 1;
}

int32_t CalculateOutputSizeBeforeStrides(int32_t input, int32_t kernel,
                                         int32_t padding, int32_t dilation) {
  const int32_t dilated_kernel = (kernel - 1) * dilation + 1;
  return input + padding - dilated_kernel + 1;
}

template <Axis T>
int32_t CalculateOutputWithoutStrides(const BHWC& input,
                                      const Convolution2DAttributes& attr) {
  const OHWI& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  return CalculateOutputSizeBeforeStrides(
      input.get<T>(), weights_shape.get<T>(),
      attr.padding.prepended.get<T>() + attr.padding.appended.get<T>(),
      attr.dilations.get<T>());
}

template <Axis T>
int32_t CalculateOutputWithoutStrides(const BHWDC& input,
                                      const Convolution3DAttributes& attr) {
  return CalculateOutputSizeBeforeStrides(
      input.get<T>(), attr.weights.shape.get<T>(),
      attr.padding.prepended.get<T>() + attr.padding.appended.get<T>(),
      attr.dilations.get<T>());
}

template <Axis T>
int32_t CalculateOutputWithoutStrides(const BHWC& input,
                                      const Pooling2DAttributes& attr) {
  return CalculateOutputSizeBeforeStrides(
      input.get<T>(), attr.kernel.get<T>(),
      attr.padding.prepended.get<T>() + attr.padding.appended.get<T>(),
      /*dilation=*/1);
}

template <Axis T>
int32_t CalculateOutputWithoutStrides(const BHWDC& input,
                                      const Pooling3DAttributes& attr) {
  return CalculateOutputSizeBeforeStrides(
      input.get<T>(), attr.kernel.get<T>(),
      attr.padding.prepended.get<T>() + attr.padding.appended.get<T>(),
      /*dilation=*/1);
}

template <Axis T>
int32_t CalculateOutput(const BHWC& input,
                        const ConvolutionTransposedAttributes& attr) {
  return (input.get<T>() - 1) * attr.stride.get<T>() -
         (attr.padding.prepended.get<T>() + attr.padding.appended.get<T>()) +
         attr.weights.shape.get<T>() + attr.adjacent.get<T>();
}

template <Axis T>
int32_t CalculateOutput(const BHWDC& input,
                        const ConvolutionTransposed3DAttributes& attr) {
  return (input.get<T>() - 1) * attr.stride.get<T>() -
         (attr.padding.prepended.get<T>() + attr.padding.appended.get<T>()) +
         attr.weights.shape.get<T>();
}

inline int32_t StridedSize(int32_t size, int32_t stride) {
  return stride == 0 ? -1 : DivideRoundUp(size, stride);
}

template <Axis AxisT, typename AttrT>
int32_t CalculateOutput(const BHWC& input, const AttrT& attr) {
  return StridedSize(CalculateOutputWithoutStrides<AxisT>(input, attr),
                     attr.strides.template get<AxisT>());
}

template <Axis AxisT, typename AttrT>
int32_t CalculateOutput(const BHWDC& input, const AttrT& attr) {
  return StridedSize(CalculateOutputWithoutStrides<AxisT>(input, attr),
                     attr.strides.template get<AxisT>());
}

int32_t CalculateSamePadding(int32_t input, int32_t kernel, int32_t dilation,
                             int32_t stride) {
  const int32_t dilated_kernel = (kernel - 1) * dilation + 1;
  return std::max(0, dilated_kernel - (input - 1) % stride - 1);
}

// Returns a padding that should be present to make sure image size stays
// the same.
template <Axis AxisT>
int32_t CalculateSamePadding(const BHWC& input,
                             const Convolution2DAttributes& attr) {
  const OHWI& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  return CalculateSamePadding(input.get<AxisT>(), weights_shape.get<AxisT>(),
                              attr.dilations.get<AxisT>(),
                              attr.strides.get<AxisT>());
}

// Returns a padding that should be present to make sure image size stays
// the same.
template <Axis AxisT>
int32_t CalculateSamePadding(const BHWDC& input,
                             const Convolution3DAttributes& attr) {
  return CalculateSamePadding(
      input.get<AxisT>(), attr.weights.shape.get<AxisT>(),
      attr.dilations.get<AxisT>(), attr.strides.get<AxisT>());
}

template <Axis AxisT>
int32_t CalculateSamePadding(const BHWC& input,
                             const ConvolutionTransposedAttributes& attr) {
  return CalculateSamePadding(input.get<AxisT>(),
                              attr.weights.shape.get<AxisT>(),
                              /*dilation=*/1, attr.stride.get<AxisT>());
}

template <Axis AxisT>
int32_t CalculateSamePadding(const BHWDC& input,
                             const ConvolutionTransposed3DAttributes& attr) {
  return CalculateSamePadding(input.get<AxisT>(),
                              attr.weights.shape.get<AxisT>(),
                              /*dilation=*/1, attr.stride.get<AxisT>());
}

template <Axis AxisT>
int32_t CalculateSamePadding(const BHWC& input,
                             const Pooling2DAttributes& attr) {
  return CalculateSamePadding(input.get<AxisT>(), attr.kernel.get<AxisT>(),
                              /*dilation=*/1, attr.strides.get<AxisT>());
}

template <Axis AxisT>
int32_t CalculateSamePadding(const BHWDC& input,
                             const Pooling3DAttributes& attr) {
  return CalculateSamePadding(input.get<AxisT>(), attr.kernel.get<AxisT>(),
                              /*dilation=*/1, attr.strides.get<AxisT>());
}

template <Axis AxisT>
int32_t CalculateSamePadding(const BHWC& input,
                             const MaxUnpooling2DAttributes& attr) {
  return CalculateSamePadding(input.get<AxisT>(), attr.kernel.get<AxisT>(),
                              /*dilation=*/1, attr.strides.get<AxisT>());
}

template <Axis AxisT>
int32_t CalculateSamePadding(const BHWDC& input,
                             const MaxUnpooling3DAttributes& attr) {
  return CalculateSamePadding(input.get<AxisT>(), attr.kernel.get<AxisT>(),
                              /*dilation=*/1, attr.strides.get<AxisT>());
}

Padding2D MakeSamePadding(const BHWC& input,
                          const ConvolutionTransposedAttributes& attr) {
  int32_t padding_height = CalculateSamePadding<Axis::kHeight>(input, attr);
  int32_t padding_width = CalculateSamePadding<Axis::kWidth>(input, attr);
  Padding2D padding;
  padding.prepended = HW(padding_height / 2, padding_width / 2);
  padding.appended = HW(padding_height - padding_height / 2,
                        padding_width - padding_width / 2);
  return padding;
}

Padding3D MakeSamePadding(const BHWDC& input,
                          const ConvolutionTransposed3DAttributes& attr) {
  int32_t padding_height = CalculateSamePadding<Axis::kHeight>(input, attr);
  int32_t padding_width = CalculateSamePadding<Axis::kWidth>(input, attr);
  int32_t padding_depth = CalculateSamePadding<Axis::kDepth>(input, attr);
  Padding3D padding;
  padding.prepended =
      HWD(padding_height / 2, padding_width / 2, padding_depth / 2);
  padding.appended =
      HWD(padding_height - padding_height / 2,
          padding_width - padding_width / 2, padding_depth - padding_depth / 2);
  return padding;
}

// If padding depends on input, convert it into fixed padding.
template <class AttrT>
Padding2D MakeSamePadding(const BHWC& input, const AttrT& attr) {
  int32_t padding_height = CalculateSamePadding<Axis::kHeight>(input, attr);
  int32_t padding_width = CalculateSamePadding<Axis::kWidth>(input, attr);
  Padding2D padding;
  padding.prepended = HW(padding_height / 2, padding_width / 2);
  padding.appended = HW(padding_height - padding_height / 2,
                        padding_width - padding_width / 2);
  return padding;
}

// If padding depends on input, convert it into fixed padding.
template <class AttrT>
Padding3D MakeSamePadding(const BHWDC& input, const AttrT& attr) {
  int32_t padding_height = CalculateSamePadding<Axis::kHeight>(input, attr);
  int32_t padding_width = CalculateSamePadding<Axis::kWidth>(input, attr);
  int32_t padding_depth = CalculateSamePadding<Axis::kDepth>(input, attr);
  Padding3D padding;
  padding.prepended =
      HWD(padding_height / 2, padding_width / 2, padding_depth / 2);
  padding.appended =
      HWD(padding_height - padding_height / 2,
          padding_width - padding_width / 2, padding_depth - padding_depth / 2);
  return padding;
}

}  // namespace

BHWC CalculateOutputShape(const BHWC& input,
                          const MaxUnpooling2DAttributes& attr) {
  return BHWC(input.b,
              input.h * attr.strides.h - attr.padding.prepended.h -
                  attr.padding.appended.h,
              input.w * attr.strides.w - attr.padding.prepended.w -
                  attr.padding.appended.w,
              input.c);
}

BHWDC CalculateOutputShape(const BHWDC& input,
                           const MaxUnpooling3DAttributes& attr) {
  return BHWDC(input.b,
               input.h * attr.strides.h - attr.padding.prepended.h -
                   attr.padding.appended.h,
               input.w * attr.strides.w - attr.padding.prepended.w -
                   attr.padding.appended.w,
               input.d * attr.strides.d - attr.padding.prepended.d -
                   attr.padding.appended.d,
               input.c);
}

BHWC CalculateOutputShape(const BHWC& input, const Pooling2DAttributes& attr) {
  return BHWC(input.b, CalculateOutput<Axis::kHeight>(input, attr),
              CalculateOutput<Axis::kWidth>(input, attr), input.c);
}

BHWDC CalculateOutputShape(const BHWDC& input,
                           const Pooling3DAttributes& attr) {
  return BHWDC(input.b, CalculateOutput<Axis::kHeight>(input, attr),
               CalculateOutput<Axis::kWidth>(input, attr),
               CalculateOutput<Axis::kDepth>(input, attr), input.c);
}

BHWC CalculateOutputShape(const BHWC& input,
                          const Convolution2DAttributes& attr) {
  const OHWI& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  return BHWC(input.b, CalculateOutput<Axis::kHeight>(input, attr),
              CalculateOutput<Axis::kWidth>(input, attr),
              weights_shape.get<Axis::kOutputChannels>());
}

BHWDC CalculateOutputShape(const BHWDC& input,
                           const Convolution3DAttributes& attr) {
  return BHWDC(input.b, CalculateOutput<Axis::kHeight>(input, attr),
               CalculateOutput<Axis::kWidth>(input, attr),
               CalculateOutput<Axis::kDepth>(input, attr),
               attr.weights.shape.get<Axis::kOutputChannels>());
}

BHWC CalculateOutputShape(const BHWC& input,
                          const ConvolutionTransposedAttributes& attr) {
  return BHWC(input.b, CalculateOutput<Axis::kHeight>(input, attr),
              CalculateOutput<Axis::kWidth>(input, attr),
              attr.weights.shape.get<Axis::kOutputChannels>());
}

BHWDC CalculateOutputShape(const BHWDC& input,
                           const ConvolutionTransposed3DAttributes& attr) {
  return BHWDC(input.b, CalculateOutput<Axis::kHeight>(input, attr),
               CalculateOutput<Axis::kWidth>(input, attr),
               CalculateOutput<Axis::kDepth>(input, attr),
               attr.weights.shape.get<Axis::kOutputChannels>());
}

BHWC CalculateOutputShape(const BHWC& input,
                          const DepthwiseConvolution2DAttributes& attr) {
  const OHWI& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  return BHWC(input.b, CalculateOutput<Axis::kHeight>(input, attr),
              CalculateOutput<Axis::kWidth>(input, attr),
              weights_shape.get<Axis::kOutputChannels>() *
                  weights_shape.get<Axis::kInputChannels>());
}

BHWDC CalculateOutputShape(const BHWDC& input,
                           const DepthwiseConvolution3DAttributes& attr) {
  return BHWDC(input.b, CalculateOutput<Axis::kHeight>(input, attr),
               CalculateOutput<Axis::kWidth>(input, attr),
               CalculateOutput<Axis::kDepth>(input, attr),
               attr.weights.shape.get<Axis::kOutputChannels>() *
                   attr.weights.shape.get<Axis::kInputChannels>());
}

BHWC CalculateOutputShape(const BHWC& input, const SliceAttributes& attr) {
  return BHWC(StridedSize(attr.ends.b - attr.starts.b, attr.strides.b),
              StridedSize(attr.ends.h - attr.starts.h, attr.strides.h),
              StridedSize(attr.ends.w - attr.starts.w, attr.strides.w),
              StridedSize(attr.ends.c - attr.starts.c, attr.strides.c));
}

BHWDC CalculateOutputShape(const BHWDC& input, const Slice3DAttributes& attr) {
  return BHWDC(StridedSize(attr.ends.b - attr.starts.b, attr.strides.b),
               StridedSize(attr.ends.h - attr.starts.h, attr.strides.h),
               StridedSize(attr.ends.w - attr.starts.w, attr.strides.w),
               StridedSize(attr.ends.d - attr.starts.d, attr.strides.d),
               StridedSize(attr.ends.c - attr.starts.c, attr.strides.c));
}

BHWC CalculateOutputShape(const BHWC& input, const PadAttributes& attr) {
  return BHWC(attr.appended.b + attr.prepended.b + input.b,
              attr.appended.h + attr.prepended.h + input.h,
              attr.appended.w + attr.prepended.w + input.w,
              attr.appended.c + attr.prepended.c + input.c);
}

BHWDC CalculateOutputShape(const BHWDC& input, const Pad3DAttributes& attr) {
  return BHWDC(attr.appended.b + attr.prepended.b + input.b,
               attr.appended.h + attr.prepended.h + input.h,
               attr.appended.w + attr.prepended.w + input.w,
               attr.appended.d + attr.prepended.d + input.d,
               attr.appended.c + attr.prepended.c + input.c);
}

BHWC CalculateOutputShape(const BHWC& input,
                          const FullyConnectedAttributes& attr) {
  return BHWC(input.b, input.h, input.w, attr.weights.shape.o);
}

bool IsLogicalOp(OperationType op_type) {
  return op_type == OperationType::kGreater ||
         op_type == OperationType::kGreaterEqual ||
         op_type == OperationType::kLess ||
         op_type == OperationType::kLessEqual ||
         op_type == OperationType::kEqual ||
         op_type == OperationType::kNotEqual;
}

absl::Status CalculateOutputShape(const std::vector<BHWC>& input,
                                  const ConcatAttributes& attr,
                                  BHWC* output_shape) {
  BHWC new_shape = input[0];
  switch (attr.axis) {
    case Axis::kChannels:
      for (int i = 1; i < input.size(); i++) {
        if (input[i].h != new_shape.h || input[i].w != new_shape.w ||
            input[i].b != new_shape.b) {
          return absl::InvalidArgumentError(
              "Height, Width and Batch must be the same when concatenating "
              "by channels axis");
        }
        new_shape.c += input[i].c;
      }
      break;
    case Axis::kHeight:
      for (int i = 1; i < input.size(); i++) {
        if (input[i].w != new_shape.w || input[i].c != new_shape.c ||
            input[i].b != new_shape.b) {
          return absl::InvalidArgumentError(
              "Channels, Width and Batch must be the same when concatenating "
              "by height axis");
        }
        new_shape.h += input[i].h;
      }
      break;
    case Axis::kWidth:
      for (int i = 1; i < input.size(); i++) {
        if (input[i].h != new_shape.h || input[i].c != new_shape.c ||
            input[i].b != new_shape.b) {
          return absl::InvalidArgumentError(
              "Height, Channels and Batch must be the same when concatenating "
              "by width axis");
        }
        new_shape.w += input[i].w;
      }
      break;
    case Axis::kBatch:
      for (int i = 1; i < input.size(); i++) {
        if (input[i].h != new_shape.h || input[i].c != new_shape.c ||
            input[i].w != new_shape.w) {
          return absl::InvalidArgumentError(
              "Width, Height and Channels must be the same when concatenating "
              "by batch axis");
        }
        new_shape.b += input[i].b;
      }
      break;
    default:
      return absl::InvalidArgumentError("Invalid axis");
      break;
  }
  *output_shape = new_shape;
  return absl::OkStatus();
}

absl::Status CalculateOutputShape(const std::vector<BHWDC>& input,
                                  const ConcatAttributes& attr,
                                  BHWDC* output_shape) {
  BHWDC new_shape = input[0];
  switch (attr.axis) {
    case Axis::kChannels:
      for (int i = 1; i < input.size(); ++i) {
        if (input[i].h != new_shape.h || input[i].w != new_shape.w ||
            input[i].d != new_shape.d || input[i].b != new_shape.b) {
          return absl::InvalidArgumentError(
              "Height, Width, Batch and Depth must be the same when "
              "concatenating "
              "by channels axis");
        }
        new_shape.c += input[i].c;
      }
      break;
    case Axis::kHeight:
      for (int i = 1; i < input.size(); ++i) {
        if (input[i].w != new_shape.w || input[i].c != new_shape.c ||
            input[i].d != new_shape.d || input[i].b != new_shape.b) {
          return absl::InvalidArgumentError(
              "Width, Depth, Batch and Channels must be the same when "
              "concatenating "
              "by height axis");
        }
        new_shape.h += input[i].h;
      }
      break;
    case Axis::kWidth:
      for (int i = 1; i < input.size(); ++i) {
        if (input[i].h != new_shape.h || input[i].c != new_shape.c ||
            input[i].d != new_shape.d || input[i].b != new_shape.b) {
          return absl::InvalidArgumentError(
              "Height, Depth, Batch and Channels must be the same when "
              "concatenating "
              "by width axis");
        }
        new_shape.w += input[i].w;
      }
      break;
    case Axis::kDepth:
      for (int i = 1; i < input.size(); ++i) {
        if (input[i].w != new_shape.w || input[i].h != new_shape.h ||
            input[i].c != new_shape.c || input[i].b != new_shape.b) {
          return absl::InvalidArgumentError(
              "Width, Height, Batch and Channels must be the same when "
              "concatenating "
              "by depth axis");
        }
        new_shape.d += input[i].d;
      }
      break;
    case Axis::kBatch:
      for (int i = 1; i < input.size(); ++i) {
        if (input[i].w != new_shape.w || input[i].h != new_shape.h ||
            input[i].c != new_shape.c || input[i].d != new_shape.d) {
          return absl::InvalidArgumentError(
              "Width, Height, Depth and Channels must be the same when "
              "concatenating "
              "by batch axis");
        }
        new_shape.b += input[i].b;
      }
      break;
    default:
      return absl::InvalidArgumentError("Invalid axis");
  }
  *output_shape = new_shape;
  return absl::OkStatus();
}

Padding2D CalculateSamePadding(const BHWC& input,
                               const Convolution2DAttributes& attr) {
  return MakeSamePadding(input, attr);
}

Padding3D CalculateSamePadding(const BHWDC& input,
                               const Convolution3DAttributes& attr) {
  return MakeSamePadding(input, attr);
}

Padding2D CalculateSamePadding(const BHWC& input,
                               const ConvolutionTransposedAttributes& attr) {
  return MakeSamePadding(input, attr);
}

Padding3D CalculateSamePadding(const BHWDC& input,
                               const ConvolutionTransposed3DAttributes& attr) {
  return MakeSamePadding(input, attr);
}

Padding2D CalculateSamePadding(const BHWC& input,
                               const DepthwiseConvolution2DAttributes& attr) {
  return MakeSamePadding(input, attr);
}

Padding3D CalculateSamePadding(const BHWDC& input,
                               const DepthwiseConvolution3DAttributes& attr) {
  return MakeSamePadding(input, attr);
}

Padding2D CalculateSamePadding(const BHWC& input,
                               const Pooling2DAttributes& attr) {
  return MakeSamePadding(input, attr);
}

Padding3D CalculateSamePadding(const BHWDC& input,
                               const Pooling3DAttributes& attr) {
  return MakeSamePadding(input, attr);
}

Padding2D CalculateSamePadding(const BHWC& input,
                               const MaxUnpooling2DAttributes& attr) {
  return MakeSamePadding(input, attr);
}

Padding3D CalculateSamePadding(const BHWDC& input,
                               const MaxUnpooling3DAttributes& attr) {
  return MakeSamePadding(input, attr);
}

float CalculateResizeScale(int32_t input_size, int32_t output_size,
                           const Resize2DAttributes& attr) {
  return attr.align_corners && input_size > 1 && output_size > 1
             ? static_cast<float>(input_size - 1) / (output_size - 1)
             : static_cast<float>(input_size) / output_size;
}

float CalculateResizeScale(int32_t input_size, int32_t output_size,
                           const Resize3DAttributes& attr) {
  return attr.align_corners && input_size > 1 && output_size > 1
             ? static_cast<float>(input_size - 1) / (output_size - 1)
             : static_cast<float>(input_size) / output_size;
}

BHWC CalculateOutputShape(const BHWC& input, const Resize2DAttributes& attr) {
  return BHWC(input.b, attr.new_shape.h, attr.new_shape.w, input.c);
}

BHWDC CalculateOutputShape(const BHWDC& input, const Resize3DAttributes& attr) {
  return BHWDC(input.b, attr.new_shape.h, attr.new_shape.w, attr.new_shape.d,
               input.c);
}

BHWC CalculateOutputShape(const BHWC& input, const TransposeAttributes& attr) {
  return BHWC(input.get(attr.perm.b), input.get(attr.perm.h),
              input.get(attr.perm.w), input.get(attr.perm.c));
}

BHWDC CalculateOutputShape(const BHWDC& input,
                           const Transpose3DAttributes& attr) {
  return BHWDC(input.get(attr.perm.b), input.get(attr.perm.h),
               input.get(attr.perm.w), input.get(attr.perm.d),
               input.get(attr.perm.c));
}

// Returns a dequantized copy of FCInt8Attr.
// We assume that the quantization tensor shapes are well-formed.
FullyConnectedAttributes ToFloat32(const FullyConnectedInt8Attributes& qattr) {
  FullyConnectedAttributes dattr;
  dattr.weights.shape = qattr.weights.shape;
  dattr.weights.data.resize(qattr.weights.shape.DimensionsProduct() +
                            XNN_EXTRA_BYTES / sizeof(float));
  const int8_t* src = qattr.weights.Data();
  float* dst = &dattr.weights.data[0];
  const float* scales = qattr.scale.Data();
  if (qattr.scale.shape.DimensionsProduct() == 1) {  // tensor-wise
    const int zp = qattr.zero_point.Data()[0];
    const float scale = qattr.scale.Data()[0];
    for (int i = 0; i < qattr.weights.shape.DimensionsProduct(); ++i) {
      dst[i] = (src[i] - zp) * scale;
    }
  } else if (qattr.scale.shape.i == 1) {  // channel-wise
    const int* zps = qattr.zero_point.Data();
    for (int j = 0, o = 0; o < qattr.weights.shape.o; ++o) {
      const int zp = zps[o];
      const float scale = scales[o];
      for (int i = 0; i < qattr.weights.shape.i; ++i, ++j) {
        dst[j] = (src[j] - zp) * scale;
      }
    }
  } else {  // block-wise
    const int blocksize = qattr.weights.shape.i / qattr.scale.shape.i;
    for (int i = 0; i < qattr.weights.shape.DimensionsProduct(); ++i) {
      dst[i] = src[i] * scales[i / blocksize];
    }
  }
  dattr.bias = qattr.bias;
  dattr.op_name = qattr.op_name;
  return dattr;
}

Tensor<OHWI, DataType::kFloat32> DequantizeImpl(
    const OHWI& weights_ohwi_shape, const int8_t* input_data,
    const Tensor<OHWI, DataType::kFloat32>& scale,
    const Tensor<OHWI, DataType::kInt32>& zero_point, bool add_extra_bytes) {
  Tensor<OHWI, DataType::kFloat32> flt_tensor;
  flt_tensor.shape = weights_ohwi_shape;
  flt_tensor.data.resize(
      weights_ohwi_shape.DimensionsProduct() +
      (add_extra_bytes ? XNN_EXTRA_BYTES / sizeof(float) : 0));

  auto runtime_shape =
      GetTensorShape({weights_ohwi_shape.o, weights_ohwi_shape.h,
                      weights_ohwi_shape.w, weights_ohwi_shape.i});
  if (scale.size() == 1) {  // Tensor-wise quantization.
    DequantizationParams op_params;
    ABSL_QCHECK_EQ(zero_point.data.size(), 1);
    ABSL_QCHECK_EQ(scale.data.size(), 1);
    op_params.zero_point = zero_point.data[0];
    op_params.scale = scale.data[0];
    Dequantize(op_params, runtime_shape, input_data, runtime_shape,
               flt_tensor.data.data());
  } else if (scale.shape.i == 1) {  // Channel-wise quantized.
    PerChannelDequantizationParams op_params;
    std::vector<int> zero_points;
    if (zero_point.size() == scale.size()) {
      op_params.zero_point = zero_point.Data();
    } else {
      zero_points.assign(zero_point.Data(),
                         zero_point.Data() + zero_point.size());
      zero_points.resize(scale.size(), zero_point.Data()[0]);
      op_params.zero_point = zero_points.data();
    }
    op_params.scale = scale.Data();
    // We assume the first dimension is to be quantized.
    op_params.quantized_dimension = 0;
    PerChannelDequantize(op_params, runtime_shape, input_data, runtime_shape,
                         flt_tensor.data.data());
  } else {  // Block-wise quantization.
    BlockwiseDequantizationParams op_params;
    std::vector<int> zero_points;
    if (zero_point.size() == scale.size()) {
      op_params.zero_point = zero_point.Data();
    } else if (zero_point.empty()) {
      zero_points.resize(scale.size(), 0);
      op_params.zero_point = zero_points.data();
    } else {
      zero_points.assign(zero_point.Data(),
                         zero_point.Data() + zero_point.size());
      zero_points.resize(scale.size(), zero_point.Data()[0]);
      op_params.zero_point = zero_points.data();
    }
    op_params.scale = scale.Data();
    op_params.quantized_dimension = 0;
    op_params.group_size = weights_ohwi_shape.i / scale.shape.i;
    BlockwiseDequantize(op_params, runtime_shape, input_data, runtime_shape,
                        flt_tensor.data.data());
  }
  return flt_tensor;
}

FullyConnectedAttributes ToFloat32(const FullyConnectedInt4Attributes& qattr) {
  FullyConnectedAttributes dattr;
  if (std::holds_alternative<Tensor<OHWI, DataType::kInt4>>(qattr.weights)) {
    const auto& int4_weights =
        std::get<Tensor<OHWI, DataType::kInt4>>(qattr.weights);
    const size_t num_elements = int4_weights.shape.DimensionsProduct();
    auto unpacked_input_data = std::make_unique<int8_t[]>(num_elements);
    UnpackDenseInt4IntoInt8(int4_weights.Data(), num_elements,
                            unpacked_input_data.get());
    dattr.weights =
        DequantizeImpl(int4_weights.shape, unpacked_input_data.get(),
                       qattr.scale, qattr.zero_point,
                       /*add_extra_bytes=*/true);
  } else {
    const auto& int8_weights =
        std::get<Tensor<OHWI, DataType::kInt8>>(qattr.weights);
    dattr.weights = DequantizeImpl(int8_weights.shape, int8_weights.Data(),
                                   qattr.scale, qattr.zero_point,
                                   /*add_extra_bytes=*/true);
  }
  dattr.bias = qattr.bias;
  dattr.op_name = qattr.op_name;
  return dattr;
}

FullyConnectedAttributes ToFloat32(const FullyConnectedInt2Attributes& qattr) {
  FullyConnectedAttributes dattr;
  if (std::holds_alternative<Tensor<OHWI, DataType::kInt2>>(qattr.weights)) {
    const auto& int2_weights =
        std::get<Tensor<OHWI, DataType::kInt2>>(qattr.weights);
    const size_t num_elements = int2_weights.shape.DimensionsProduct();
    auto unpacked_input_data = std::make_unique<int8_t[]>(num_elements);
    UnpackDenseInt2IntoInt8(int2_weights.Data(), num_elements,
                            unpacked_input_data.get());
    dattr.weights =
        DequantizeImpl(int2_weights.shape, unpacked_input_data.get(),
                       qattr.scale, qattr.zero_point,
                       /*add_extra_bytes=*/true);
  } else {
    const auto& int8_weights =
        std::get<Tensor<OHWI, DataType::kInt8>>(qattr.weights);
    dattr.weights = DequantizeImpl(int8_weights.shape, int8_weights.Data(),
                                   qattr.scale, qattr.zero_point,
                                   /*add_extra_bytes=*/true);
  }
  dattr.bias = qattr.bias;
  dattr.op_name = qattr.op_name;
  return dattr;
}

template <DataType QuantizedT>
Tensor<OHWI, DataType::kFloat32> DequantizeTensor(
    const Tensor<OHWI, QuantizedT>& weights,
    const Tensor<OHWI, DataType::kFloat32>& scale,
    const Tensor<OHWI, DataType::kInt32>& zero_point, bool add_extra_bytes) {
  const int8_t* input_data;
  std::unique_ptr<int8_t[]> unpacked_input_data = nullptr;
  if constexpr (QuantizedT == DataType::kInt4) {
    const size_t num_elements = weights.shape.DimensionsProduct();
    unpacked_input_data = std::make_unique<int8_t[]>(num_elements);
    UnpackDenseInt4IntoInt8(weights.Data(), num_elements,
                            unpacked_input_data.get());
    input_data = unpacked_input_data.get();
  } else if constexpr (QuantizedT == DataType::kInt2) {
    const size_t num_elements = weights.shape.DimensionsProduct();
    unpacked_input_data = std::make_unique<int8_t[]>(num_elements);
    UnpackDenseInt2IntoInt8(weights.Data(), num_elements,
                            unpacked_input_data.get());
    input_data = unpacked_input_data.get();
  } else if constexpr (QuantizedT == DataType::kInt8) {
    input_data = weights.Data();
  } else {
    ABSL_LOG(FATAL) << "Unsupported quantized type: " << ToString(QuantizedT);
  }

  return DequantizeImpl(weights.shape, input_data, scale, zero_point,
                        add_extra_bytes);
}

Tensor<OHWI, DataType::kFloat32>& GetFloatWeights(
    Convolution2DAttributes& attr) {
  return std::visit(
      [&attr](auto& weights) -> Tensor<OHWI, DataType::kFloat32>& {
        using T = std::decay_t<decltype(weights)>;
        if constexpr (std::is_same_v<T, Tensor<OHWI, DataType::kFloat32>>) {
          return weights;
        } else {
          auto flt_tensor =
              DequantizeTensor(weights, attr.scale, attr.zero_point);

          // Swapping quantized tensors with dequantized tensors is not mutating
          // the attributes, because it's equivalent to store the dequantized
          // float tensor or to store the quantized tensor with scale and
          // zero_point.
          attr.weights.emplace<Tensor<OHWI, DataType::kFloat32>>(
              std::move(flt_tensor));
          attr.scale = Tensor<OHWI, DataType::kFloat32>();
          attr.zero_point = Tensor<OHWI, DataType::kInt32>();

          return std::get<Tensor<OHWI, DataType::kFloat32>>(attr.weights);
        }
      },
      attr.weights);
}

template Tensor<OHWI, DataType::kFloat32> DequantizeTensor<DataType::kInt8>(
    const Tensor<OHWI, DataType::kInt8>& weights,
    const Tensor<OHWI, DataType::kFloat32>& scale,
    const Tensor<OHWI, DataType::kInt32>& zero_point, bool add_extra_bytes);

template Tensor<OHWI, DataType::kFloat32> DequantizeTensor<DataType::kInt4>(
    const Tensor<OHWI, DataType::kInt4>& weights,
    const Tensor<OHWI, DataType::kFloat32>& scale,
    const Tensor<OHWI, DataType::kInt32>& zero_point, bool add_extra_bytes);

template Tensor<OHWI, DataType::kFloat32> DequantizeTensor<DataType::kInt2>(
    const Tensor<OHWI, DataType::kInt2>& weights,
    const Tensor<OHWI, DataType::kFloat32>& scale,
    const Tensor<OHWI, DataType::kInt32>& zero_point, bool add_extra_bytes);

bool IsConvEquivalentToFullyConnected(const Convolution2DAttributes& attr) {
  const auto& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  return weights_shape.w == 1 &&                //
         weights_shape.h == 1 &&                //
         attr.strides == HW(1, 1) &&            //
         attr.dilations == HW(1, 1) &&          //
         attr.padding.prepended == HW(0, 0) &&  //
         attr.padding.appended == HW(0, 0) &&   //
         attr.groups == 1;
}

}  // namespace ml_drift
