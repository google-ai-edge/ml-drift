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

#ifndef ML_DRIFT_COMMON_OPERATIONS_H_
#define ML_DRIFT_COMMON_OPERATIONS_H_

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

// Non exhaustive list of operations.
enum class OperationType {
  UNKNOWN = 0,
  ABS,
  ADD,
  ATAN2,
  BATCH_TO_SPACE,
  BATCH_NORMALIZATION,
  BATCHED_MATMUL,
  BITCAST,
  BROADCAST_IN_DIM,
  CAST,
  CEIL,
  CONCAT,
  CONSTANT,
  CONVOLUTION_2D,
  CONVOLUTION_TRANSPOSED,
  COPY,
  COS,
  CUMSUM,
  DEPTHWISE_CONVOLUTION,
  DEPTH_TO_SPACE,
  DIV,
  DOT_GENERAL,
  DYNAMIC_UPDATE_SLICE,
  ELU,
  EQUAL,
  EMBEDDING_LOOKUP,
  EXP,
  FLOOR,
  FLOOR_DIV,
  FLOOR_MOD,
  FULLY_CONNECTED,
  FULLY_CONNECTED_INT2,
  FULLY_CONNECTED_INT4,
  FULLY_CONNECTED_INT8,
  GATHER,
  GELU,
  GELU_TANH_APPROX,
  GREATER,
  GREATER_EQUAL,
  GROUP_NORM,
  HARD_SWISH,
  LAYER_NORM,
  LESS,
  LESS_EQUAL,
  LOG,
  LOGICAL_AND,
  LOGICAL_NOT,
  LOGICAL_OR,
  LOGICAL_XOR,
  LSTM,
  MAXIMUM,
  MAX_INDEX,
  MAX_UNPOOLING_2D,
  MEAN,
  MEAN_STDDEV_NORMALIZATION,
  MINIMUM,
  MISH,
  MOD,
  MUL,
  NEG,
  NOT_EQUAL,
  ONE_HOT,
  PAD,
  POOLING_2D,
  POSITIONAL_EMBEDDING,
  POW,
  PRELU,
  QUANTIZE_AND_DEQUANTIZE,
  REDUCE_ALL,
  REDUCE_ANY,
  REDUCE_MAXIMUM,
  REDUCE_MINIMUM,
  REDUCE_PRODUCT,
  REDUCE_SUM,
  RELU,
  REMAINDER,
  RESAMPLER,
  RESHAPE,
  RESIZE,
  REVERSE,
  RMS_NORM,
  ROPE,
  ROUND,
  RSQRT,
  SCALED_DOT_PRODUCT_ATTENTION,
  SELECT_V2,
  SHIFT_LEFT,
  SHIFT_RIGHT,
  SIGMOID,
  SIGN,
  SIN,
  SLICE,
  SOFTMAX,
  SPACE_TO_BATCH,
  SPACE_TO_DEPTH,
  SPLIT,
  SQRT,
  SQUARE,
  SQUARED_DIFF,
  SUB,
  TANH,
  TILE,
  TOP_K,
  TRANSPOSE,
};

std::string ToString(enum OperationType op);

OperationType OperationTypeFromString(const std::string& name);

typedef std::variant<float, int, unsigned int> ScalarValue;

typedef std::variant<std::monostate, Tensor<BHWC, DataType::FLOAT32>,
                     Tensor<BHWDC, DataType::FLOAT32>,
                     Tensor<Linear, DataType::FLOAT32>, ScalarValue>
    TensorOrScalar;

bool HoldsFloatScalar(const TensorOrScalar& arg);
const float* GetIfFloatScalar(const TensorOrScalar* arg);

struct Padding2D {
  Padding2D() = default;
  Padding2D& operator=(const Padding2D& value) = default;
  bool operator==(const Padding2D& value) const;
  bool operator!=(const Padding2D& value) const;
  Padding2D& operator-(const Padding2D& value);

  // Padding values for every axis (if needed), where 'prepended' defines
  // padding for the beginning of each axis and 'appended' represents end part
  // of the corresponding axis.
  HW prepended = HW(-1, -1);
  HW appended = HW(-1, -1);
};

struct Padding3D {
  Padding3D() = default;
  Padding3D& operator=(const Padding3D& value) = default;
  bool operator==(const Padding3D& value) const;
  bool operator!=(const Padding3D& value) const;
  Padding3D& operator-(const Padding3D& value);

  // Padding values for every axis (if needed), where 'prepended' defines
  // padding for the beginning of each axis and 'appended' represents end part
  // of the corresponding axis.
  HWD prepended = HWD(0, 0, 0);
  HWD appended = HWD(0, 0, 0);
};

struct Crop2D : public Padding2D {};

struct MaxIndexAttributes {
  Axis dim;
};

enum class PoolingType {
  UNDEFINED = 0,

  // average pooling
  AVERAGE = 1,

  // max pooling
  MAX = 2,
};

struct Pooling2DAttributes {
  PoolingType type = PoolingType::UNDEFINED;
  // Strides for every axis.
  HW strides = HW(-1, -1);
  HW kernel = HW(-1, -1);
  Padding2D padding;
  // NOTE(akulik): technically the number of outputs from Pooling node indicates
  // whether indices are needed or not, but I decided to keep it inside
  // attributes to simplify processing.
  bool output_indices = false;
};

struct Pooling3DAttributes {
  PoolingType type = PoolingType::UNDEFINED;
  // Strides for every axis.
  HWD strides = HWD(0, 0, 0);
  HWD kernel = HWD(0, 0, 0);
  Padding3D padding;
  // NOTE(akulik): technically the number of outputs from Pooling node indicates
  // whether indices are needed or not, but I decided to keep it inside
  // attributes to simplify processing.
  bool output_indices = false;
};

struct MaxUnpooling2DAttributes {
  // Strides for every axis.
  HW strides = HW(-1, -1);
  HW kernel = HW(-1, -1);
  Padding2D padding;
};

struct MaxUnpooling3DAttributes {
  // Strides for every axis.
  HWD strides = HWD(0, 0, 0);
  HWD kernel = HWD(0, 0, 0);
  Padding3D padding;
};

struct MeanAttributes {
  // The vector of dimensions to calculate mean along.
  std::set<Axis> dims;
};

struct ConcatAttributes {
  // Defines axis by which to concat on.
  Axis axis = Axis::UNKNOWN;
};

// @return shape of a tensor after MaxUnpooling2D operation is applied to
//         the given input.
BHWC CalculateOutputShape(const BHWC& input,
                          const MaxUnpooling2DAttributes& attr);

// @return shape of a tensor after MaxUnpooling3D operation is applied to
//         the given input.
BHWDC CalculateOutputShape(const BHWDC& input,
                           const MaxUnpooling3DAttributes& attr);

// @return shape of a tensor after Pooling2D operation is applied to the given
//         input.
BHWC CalculateOutputShape(const BHWC& input, const Pooling2DAttributes& attr);

// @return shape of a tensor after Pooling3D operation is applied to the given
//         input.
BHWDC CalculateOutputShape(const BHWDC& input, const Pooling3DAttributes& attr);

// @return shape of a tensor after Concat operation is applied to the given
//         input.
absl::Status CalculateOutputShape(const std::vector<BHWC>& input,
                                  const ConcatAttributes& attr,
                                  BHWC* output_shape);

// @return shape of a tensor after Concat operation is applied to the given
//         input.
absl::Status CalculateOutputShape(const std::vector<BHWDC>& input,
                                  const ConcatAttributes& attr,
                                  BHWDC* output_shape);

// @return padding for pooling operation to make sure output keep the same shape
// as the given input.
Padding2D CalculateSamePadding(const BHWC& input,
                               const Pooling2DAttributes& attr);

// @return padding for pooling operation to make sure output keep the same shape
// as the given input.
Padding3D CalculateSamePadding(const BHWDC& input,
                               const Pooling3DAttributes& attr);

// @return padding for max unpooling operation to make sure output keep the same
// shape as the given input.
Padding2D CalculateSamePadding(const BHWC& input,
                               const MaxUnpooling2DAttributes& attr);

// @return padding for max unpooling operation to make sure output keep the same
// shape as the given input.
Padding3D CalculateSamePadding(const BHWDC& input,
                               const MaxUnpooling3DAttributes& attr);

struct Convolution2DAttributes {
  HW strides = HW(1, 1);    // Along each axis.
  HW dilations = HW(1, 1);  // Along each axis.
  Padding2D padding;

  std::variant<Tensor<OHWI, DataType::FLOAT32>, Tensor<OHWI, DataType::INT8>,
               Tensor<OHWI, DataType::INT4>>
      weights;
  Tensor<Linear, DataType::FLOAT32> bias;  // optional
  Tensor<OHWI, DataType::FLOAT32> scale;   // optional
  Tensor<OHWI, DataType::INT32> zero_point;  // optional

  int groups = 1;  // optional, split channels dimension on equal groups
  // Restrictions:
  // src.Channels() and dst.Channels() must be divisible by groups
  // Restrictions for gpu delegates:
  //   src_group_channels = src.Channels() / groups;
  //   dst_group_channels = dst.Channels() / groups;
  //   src_group_channels and dst_group_channels must be divisible by 4
  // if groups != 1, weights will have special format
  //   weights.o = group_weights.o * groups;
  //   weights.i = group_weights.i;
  //   weights.h = group_weights.h;
  //   weights.w = group_weights.w;
  std::string op_name;  // optional field for debugging
};

bool IsConvEquivalentToFullyConnected(const Convolution2DAttributes& attr);

struct Convolution3DAttributes {
  HWD strides = HWD(0, 0, 0);    // Along each axis.
  HWD dilations = HWD(0, 0, 0);  // Along each axis.
  Padding3D padding;

  Tensor<OHWDI, DataType::FLOAT32> weights;
  Tensor<Linear, DataType::FLOAT32> bias;  // optional

  int groups = 1;  // optional, split channels dimension on equal groups
  // Restrictions:
  // src.Channels() and dst.Channels() must be divisible by groups
  // Restrictions for gpu delegates:
  //   src_group_channels = src.Channels() / groups;
  //   dst_group_channels = dst.Channels() / groups;
  //   src_group_channels and dst_group_channels must be divisible by 4
  // if groups != 1, weights will have special format
  //   weights.o = group_weights.o * groups;
  //   weights.i = group_weights.i;
  //   weights.h = group_weights.h;
  //   weights.w = group_weights.w;
  //   weights.d = group_weights.d;
  std::string op_name;  // optional field for debugging
};

// @return shape of a tensor after Convolution2D operation is applied to
//         the given input.
BHWC CalculateOutputShape(const BHWC& input,
                          const Convolution2DAttributes& attr);

// @return shape of a tensor after Convolution3D operation is applied to
//         the given input.
BHWDC CalculateOutputShape(const BHWDC& input,
                           const Convolution3DAttributes& attr);

// @return padding for convolution operation to make sure output keep the same
// shape as the given input.
Padding2D CalculateSamePadding(const BHWC& input,
                               const Convolution2DAttributes& attr);

// @return padding for convolution operation to make sure output keep the same
// shape as the given input.
Padding3D CalculateSamePadding(const BHWDC& input,
                               const Convolution3DAttributes& attr);

struct ConvolutionTransposedAttributes {
  HW stride = HW(1, 1);  // Along each axis.
  HW adjacent;           // TODO(sorokin): No op on Flow.
  Padding2D padding;

  Tensor<OHWI, DataType::FLOAT32> weights;
  Tensor<Linear, DataType::FLOAT32> bias;  // optional
  std::string op_name;                     // optional field for debugging
};

struct ConvolutionTransposed3DAttributes {
  HWD stride = HWD(0, 0, 0);  // Along each axis.
  Padding3D padding;

  Tensor<OHWDI, DataType::FLOAT32> weights;
  Tensor<Linear, DataType::FLOAT32> bias;  // optional
  std::string op_name;                     // optional field for debugging
};

Padding2D CalculateSamePadding(const BHWC& input,
                               const ConvolutionTransposedAttributes& attr);

Padding3D CalculateSamePadding(const BHWDC& input,
                               const ConvolutionTransposed3DAttributes& attr);

// @return shape of a tensor after ConvolutionTransposed operation is applied to
//         the given input.
BHWC CalculateOutputShape(const BHWC& input,
                          const ConvolutionTransposedAttributes& attr);

// @return shape of a tensor after ConvolutionTransposed3D operation is applied
// to
//         the given input.
BHWDC CalculateOutputShape(const BHWDC& input,
                           const ConvolutionTransposed3DAttributes& attr);

struct DepthwiseConvolution2DAttributes : public Convolution2DAttributes {};
struct DepthwiseConvolution3DAttributes : public Convolution3DAttributes {};

struct DotGeneralAttributes {
  // lhs_batch_axes.size() == rhs_batch_axes.size() &&
  // dim(lhs, lhs_batch_axes[i]) == dim(rhs, rhs_batch_axes[i])
  std::vector<Axis> lhs_batch_axes;  // lhs axes that are batched w/ the rhs
  std::vector<Axis> rhs_batch_axes;
  // lhs_contracting_axes.size() == rhs_contracting_axes.size() &&
  // dim(lhs, lhs_contracting_axes[i]) == dim(rhs, rhs_contracting_axes[i])
  std::vector<Axis> lhs_contracting_axes;  // Which axes are reduced
  std::vector<Axis> rhs_contracting_axes;
  // Resulting axes are the axes not batched or contracted. We need to know
  // which axes are passed through to the output. This is needed because all
  // ml_drift tensors are expanded out to 4D by ExtractTensorShape. For example,
  // consider a 2D tensor with contracting axes = {Axis::BATCH}. It will have a
  // shape of (B, 1, 1, C). Then the resulting axes are {Axis::CHANNELS}.
  std::vector<Axis> lhs_resulting_axes;
  std::vector<Axis> rhs_resulting_axes;
};

struct EmbeddingLookupAttributes {
  // This needs to be a variant if we start support float weights too.
  // TODO(b/350049081): Add support for int4 quantized weights.
  // TODO(b/351847859): Change weights to be a variant.
  std::variant<Tensor<OHWI, DataType::INT8>, Tensor<OHWI, DataType::FLOAT32>,
               Tensor<OHWI, DataType::UINT8>>
      weights;

  enum class WeightsType {
    kFloat32,
    kInt8,
    kInt4,
    kInt2,
  };
  WeightsType weights_type;
  OHWI scale_zp_shape;
  OHWI original_weights_shape;

  Tensor<OHWI, DataType::FLOAT32>
      weights_scale;  // optional field for quantization cases.
  Tensor<OHWI, DataType::FLOAT32>
      weights_zero_point;  // optional field for quantization cases.
};

// @return shape of a tensor after DepthwiseConvolution2D operation is applied
//         to the given input.
BHWC CalculateOutputShape(const BHWC& input,
                          const DepthwiseConvolution2DAttributes& attr);

// @return shape of a tensor after DepthwiseConvolution3D operation is applied
//         to the given input.
BHWDC CalculateOutputShape(const BHWDC& input,
                           const DepthwiseConvolution3DAttributes& attr);

// @return padding for depthwise convolution operation to make sure output keep
// the same shape as the given input.
Padding2D CalculateSamePadding(const BHWC& input,
                               const DepthwiseConvolution2DAttributes& attr);

// @return padding for depthwise convolution operation to make sure output keep
// the same shape as the given input.
Padding3D CalculateSamePadding(const BHWDC& input,
                               const DepthwiseConvolution3DAttributes& attr);

// f(x):= {
//   if alpha != 0: x -> min(activation_max, x)
//   else
//     if x < activation_min : x -> min(activation_min, alpha * x)
//     if x >= activation_min : x -> min(activation_max, x)
// }
//
// Examples:
//   - ReLU: activation_min = 0, activation_max = 0, alpha = 0
//   - ReLU6: activation_min = 0, activation_max = 6, alpha = 0
//   - Leaky ReLU: activation_min = 0, activation_max = 0, alpha = a
//   - ReLUN1To1: activation_min = -1, activation_max = 1, alpha = 0
struct ReLUAttributes {
  // activation_min must be < activation_max
  float activation_min = 0;

  // activation_max <= 0 mean it is not set.
  float activation_max = 0;

  // alpha must be <= 1
  float alpha = 0;
};

struct PReLUAttributes {
  // If alpha is linear, then it is sharded across CHANNELS axis, otherwise
  // full shape alpha is required.
  std::variant<Tensor<Linear, DataType::FLOAT32>,
               Tensor<HWC, DataType::FLOAT32>>
      alpha;
};

struct RoPEAttributes {
  float min_timescale = 1.0f;
  float max_timescale = 10000.0f;
  float proportion = 1.0f;  // p-RoPE proportion
};

struct ReduceAttributes {
  std::set<Axis> dims;
};

struct SoftmaxAttributes {
  Axis axis = Axis::UNKNOWN;
};

enum LstmKernelType {
  FULL = 0,
  BASIC = 1,  // Currently, only basic is supported.
};

struct LstmAttributes {
  LstmKernelType kernel_type = LstmKernelType::BASIC;
};

enum class SamplingType {
  UNKNOWN = 0,
  NEAREST = 1,
  BILINEAR = 2,
};

struct Resize2DAttributes {
  HW new_shape;

  SamplingType type = SamplingType::UNKNOWN;

  // If true, the centers of the 4 corner pixels of the input and output tensors
  // are aligned, preserving the values at the corner pixels. Defaults to false.
  bool align_corners = false;

  bool half_pixel_centers = false;
};

// TODO(b/147771327): rename to Resize3D
struct Resize3DAttributes {
  HWD new_shape;

  SamplingType type = SamplingType::NEAREST;

  // If true, the centers of the 8 corner pixels of the input and output tensors
  // are aligned, preserving the values at the corner pixels. Defaults to false.
  bool align_corners = false;

  bool half_pixel_centers = false;
};

float CalculateResizeScale(int32_t input_size, int32_t output_size,
                           const Resize2DAttributes& attr);

float CalculateResizeScale(int32_t input_size, int32_t output_size,
                           const Resize3DAttributes& attr);

// @return shape of a tensor after scale operation is applied to the given
// input.
BHWC CalculateOutputShape(const BHWC& input, const Resize2DAttributes& attr);

// @return shape of a tensor after scale operation is applied to the given
// input.
BHWDC CalculateOutputShape(const BHWDC& input, const Resize3DAttributes& attr);

struct ReverseAttributes {
  std::set<Axis> axes;
};

enum class PaddingContentType {
  ZEROS = 0,
  REFLECT = 1,
  EDGE = 2,
};

struct PadAttributes {
  PaddingContentType type = PaddingContentType::ZEROS;

  BHWC prepended;
  BHWC appended;
  float constant_values = 0;
};

// @return shape of a tensor after Pad operation is applied to the given input.
BHWC CalculateOutputShape(const BHWC& input, const PadAttributes& attr);

struct Pad3DAttributes {
  PaddingContentType type = PaddingContentType::ZEROS;

  BHWDC prepended;
  BHWDC appended;
};

// @return shape of a tensor after Pad3D operation is applied to the given
// input.
BHWDC CalculateOutputShape(const BHWDC& input, const Pad3DAttributes& attr);

struct ConstTensorAttributes {
  std::variant<TensorFloat32, TensorBool, TensorInt32, TensorFloat16> tensor;
};

// Simple slicing without advanced support for shrinking, reverse slicing etc.
struct SliceAttributes {
  // Specifies start and end dimensions for slicing.
  BHWC starts;
  BHWC ends;

  // Stride should be >= 1.
  BHWC strides;
};

// @return shape of a tensor after Slice2D operation is applied to the given
//         input.
BHWC CalculateOutputShape(const BHWC& input, const SliceAttributes& attr);

// Simple slicing without advanced support for shrinking, reverse slicing etc.
struct Slice3DAttributes {
  // Specifies start and end dimensions for slicing.
  BHWDC starts;
  BHWDC ends;

  // Stride should be >= 1.
  BHWDC strides;
};

// @return shape of a tensor after Slice3D operation is applied to the given
//         input.
BHWDC CalculateOutputShape(const BHWDC& input, const Slice3DAttributes& attr);

// Runtime check params used by FullyConnected and BatchedMatMul ops.
// See ConvRuntimeCheckDesc in common/task/gpu_operation.h for more details.
// TODO: b/475505965 - Clean up parameters as these params are too subtle for
// clients to set properly.
struct RuntimeCheckParams {
  std::optional<int> src_start_ch_index;
  std::optional<int> src_end_ch_index;
  std::optional<int> dst_start_ch_index;
  std::optional<int> dst_end_ch_index;
};

struct FullyConnectedAttributes {
  Tensor<OHWI, DataType::FLOAT32> weights;
  Tensor<Linear, DataType::FLOAT32> bias;

  // Set if the weights are an external tensor.
  struct ExternalWeightsAttributes {
    WeightsDescription desc;
    RuntimeCheckParams runtime_check;
  };
  std::optional<ExternalWeightsAttributes> external_weights;

  std::string op_name;
};

// FullyConnectedInt8Attributes can be used for external weights that only shape
// for the weights been initialized without data.
struct FullyConnectedInt8Attributes {
  Tensor<OHWI, DataType::INT8> weights;
  // Tensor-wise  quant: scale & zero_point have the shape OHWI(1, 1, 1, 1),
  // Channel-wise quant: scale & zero_point have the shape OHWI(M, 1, 1, 1),
  // Block-wise   quant: scale & zero_point have the shape OHWI(M, 1, 1, N)
  //                     though zero_point can be empty.
  Tensor<OHWI, DataType::FLOAT32> scale;
  Tensor<OHWI, DataType::INT32> zero_point;
  Tensor<Linear, DataType::FLOAT32> bias;
  std::string op_name;
};

// FullyConnectedInt4Attributes can be used for external weights that only shape
// for the weights been initialized without data.
struct FullyConnectedInt4Attributes {
  // Weights can be stored as:
  // - Tensor<OHWI, DataType::INT8>: Unpacked weights (one element per byte).
  // - Tensor<OHWI, DataType::INT4>: Packed weights (two elements per byte).
  std::variant<Tensor<OHWI, DataType::INT8>, Tensor<OHWI, DataType::INT4>>
      weights;
  // Tensor-wise  quant: scale & zero_point have the shape OHWI(1, 1, 1, 1),
  // Channel-wise quant: scale & zero_point have the shape OHWI(M, 1, 1, 1),
  // Block-wise   quant: scale & zero_point have the shape OHWI(M, 1, 1, N)
  //                     though zero_point can be empty.
  Tensor<OHWI, DataType::FLOAT32> scale;
  Tensor<OHWI, DataType::INT32> zero_point;
  Tensor<Linear, DataType::FLOAT32> bias;
  std::string op_name;
};

// FullyConnectedInt2Attributes can be used for external weights that only shape
// for the weights been initialized without data.
struct FullyConnectedInt2Attributes {
  // Weights can be stored as:
  // - Tensor<OHWI, DataType::INT8>: Unpacked weights (one element per byte).
  // - Tensor<OHWI, DataType::INT2>: Packed weights (four elements per byte).
  std::variant<Tensor<OHWI, DataType::INT8>, Tensor<OHWI, DataType::INT2>>
      weights;
  // Tensor-wise  quant: scale & zero_point have the shape OHWI(1, 1, 1, 1),
  // Channel-wise quant: scale & zero_point have the shape OHWI(M, 1, 1, 1),
  Tensor<OHWI, DataType::FLOAT32> scale;
  Tensor<OHWI, DataType::INT32> zero_point;
  Tensor<Linear, DataType::FLOAT32> bias;
  std::string op_name;
};

// Dequantize int8 weights to float32.
FullyConnectedAttributes ToFloat32(const FullyConnectedInt8Attributes& attr);

// Dequantize int4 weights to float32.
FullyConnectedAttributes ToFloat32(const FullyConnectedInt4Attributes& attr);

// Dequantize int2 weights to float32.
FullyConnectedAttributes ToFloat32(const FullyConnectedInt2Attributes& attr);

// @return shape of a tensor after FullyConnected operation is applied to
// the given input.
BHWC CalculateOutputShape(const BHWC& input,
                          const FullyConnectedAttributes& attr);

// @return shape of a tensor after Mean operation is applied to the given input.
BHWC CalculateOutputShape(const BHWC& input, const MeanAttributes& attr);

// @return shape of a tensor after Mean operation is applied to the given input.
BHWDC CalculateOutputShape(const BHWDC& input, const MeanAttributes& attr);

struct ElementwiseAttributes {
  TensorOrScalar param;
  // For elementwise operation with 2 inputs op(A, B), runtime_tensor_is_second
  // true when runtime tensor is B(on second position). this is important for
  // ops that non commutative, for example subtract.
  bool runtime_tensor_is_second = false;
};

bool IsLogicalOp(OperationType op);

struct ReshapeAttributes {
  BHWC new_shape;
};

struct Reshape3DAttributes {
  BHWDC new_shape;
};

struct TopKAttributes {
  // The number of elements to return.
  int k;
};

struct TransposeAttributes {
  // A permutation of the dimensions of input tensor
  BHWC perm;
};

// @return shape of a tensor after Transpose operation is applied to
// the given input.
BHWC CalculateOutputShape(const BHWC& input, const TransposeAttributes& attr);

struct Transpose3DAttributes {
  // A permutation of the dimensions of input tensor
  BHWDC perm;
};

// @return shape of a tensor after Transpose3D operation is applied to
// the given input.
BHWDC CalculateOutputShape(const BHWDC& input,
                           const Transpose3DAttributes& attr);

struct SpaceToDepthAttributes {
  int block_size;
};

struct SplitAttributes {
  // Defines axis by which to split.
  Axis axis = Axis::UNKNOWN;
};

// These help perform a combination of Quantize & Dequantize to adjust float
// values like quantized inference would.
struct QuantizeAndDequantizeAttributes {
  float min = 0;
  float max = 0;
  float scale = 0;
};

struct GatherAttributes {
  Axis axis = Axis::UNKNOWN;
};

struct OneHotAttributes {
  float on_value = 1;
  float off_value = 0;
};

struct SelectV2Attributes {};

struct CumsumAttributes {
  Axis axis = Axis::UNKNOWN;
};

struct GroupNormAttributes {
  int groups;
  float epsilon;
  std::optional<Tensor<Linear, DataType::FLOAT32>> gamma;
  std::optional<Tensor<Linear, DataType::FLOAT32>> beta;
};

struct ScaledDotProductAttentionAttributes {
  std::optional<float> scale;
};

struct DynamicUpdateSliceAttributes {
  int height_pos;
  int width_pos;
  int channel_pos;
};

struct LayerNormAttributes {
  float epsilon;
  std::optional<Tensor<Linear, DataType::FLOAT32>> scale;
  std::optional<Tensor<Linear, DataType::FLOAT32>> bias;
};

struct RmsNormAttributes {
  float epsilon;
  std::optional<Tensor<Linear, DataType::FLOAT32>> scale;
  std::optional<Tensor<Linear, DataType::FLOAT32>> bias;
};

struct BatchedMatMulAttributes {
  bool transpose_left = false;
  bool transpose_right = false;
  RuntimeCheckParams runtime_check;
};

// Dequantize int8 or int4 weights to a float32 tensor.
//
// `add_extra_bytes` is set as true by default, because the output tensor of
// this is usually used as the input of XNNPACK weights rearrangement operation,
// which requires extra bytes.
template <DataType QuantizedType>
Tensor<OHWI, DataType::FLOAT32> DequantizeTensor(
    const Tensor<OHWI, QuantizedType>& weights,
    const Tensor<OHWI, DataType::FLOAT32>& scale,
    const Tensor<OHWI, DataType::INT32>& zero_point,
    bool add_extra_bytes = true);

// Get the float weights tensor from Convolution2DAttributes. If the attributes
// store the weights as quantized (aka. weights tensor is INT8 or INT4 type),
// the weights will be dequantized and update the attributes to store the float
// weights.
// This function may modify the internal representation of the attributes'
// weights, but it won't change them semantically.
Tensor<OHWI, DataType::FLOAT32>& GetFloatWeights(Convolution2DAttributes& attr);

// Get the float weights tensor from Convolution2DAttributes.
inline const Tensor<OHWI, DataType::FLOAT32>& GetFloatWeights(
    const Convolution2DAttributes& attr) {
  return GetFloatWeights(const_cast<Convolution2DAttributes&>(attr));
}

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_OPERATIONS_H_
