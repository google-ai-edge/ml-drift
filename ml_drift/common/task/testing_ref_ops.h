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

#ifndef ML_DRIFT_COMMON_TASK_TESTING_REF_OPS_H_
#define ML_DRIFT_COMMON_TASK_TESTING_REF_OPS_H_

#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

struct TestingRuntimeChannels {
  static constexpr int kNumParams = 4;
  static constexpr int kSrcStartChIndex = 0;
  static constexpr int kSrcEndChIndex = 1;
  static constexpr int kDstStartChIndex = 2;
  static constexpr int kDstEndChIndex = 3;

  std::optional<int> src_start_ch = std::nullopt;
  std::optional<int> src_end_ch = std::nullopt;
  std::optional<int> dst_start_ch = std::nullopt;
  std::optional<int> dst_end_ch = std::nullopt;

  TensorInt32 GenerateTensorInt32() const;
  ConvRuntimeCheckDesc GenerateConvRuntimeCheckDesc() const;
  TestingRuntimeChannels GenerateAlignedRuntimeChannels() const;
};

TensorFloat32 AddTableReference(const std::vector<TensorFloat32>& inputs);

Tensor5DFloat32 AddTableReference(const std::vector<Tensor5DFloat32>& inputs);

TensorFloat32 PaddingReference(const PadAttributes& attr,
                               const TensorFloat32& input);

Tensor5DFloat32 PaddingReference(const Pad3DAttributes& attr,
                                 const Tensor5DFloat32& input);

TensorFloat32 BatchedMatMulReference(const TensorFloat32& left,
                                     const TensorFloat32& right);

TensorFloat32 BatchedMatMulReference(
    const TensorFloat32& left, const TensorFloat32& right,
    const TestingRuntimeChannels& runtime_channels);

TensorFloat32 ConcatReference(const ConcatAttributes& attr,
                              const std::vector<TensorFloat32>& inputs);

Tensor5DFloat32 ConcatReference(const ConcatAttributes& attr,
                                const std::vector<Tensor5DFloat32>& inputs);

TensorFloat32 ConvolutionTransposedReference(
    const ConvolutionTransposedAttributes& attr, const TensorFloat32& input);

Tensor5DFloat32 ConvolutionTransposedReference(
    const ConvolutionTransposed3DAttributes& attr,
    const Tensor5DFloat32& input);

TensorFloat32 ConvolutionReference(const Convolution2DAttributes& attr,
                                   const TensorFloat32& input);

TensorFloat32 ConvolutionReference(
    const Convolution2DAttributes& attr, const TensorFloat32& input,
    const TestingRuntimeChannels& runtime_channels);

TensorFloat32 ConvolutionReference(const Convolution2DAttributes& attr,
                                   const TensorFloat32& input,
                                   const TensorFloat32& weights);

Tensor5DFloat32 ConvolutionReference(const Convolution3DAttributes& attr,
                                     const Tensor5DFloat32& input);

TensorFloat32 DepthWiseConvolutionReference(
    const DepthwiseConvolution2DAttributes& attr, const TensorFloat32& input);

Tensor5DFloat32 DepthWiseConvolutionReference(
    const DepthwiseConvolution3DAttributes& attr, const Tensor5DFloat32& input);

TensorFloat32 FullyConnectedReference(const FullyConnectedAttributes& attr,
                                      const TensorFloat32& input);

TensorFloat32 FullyConnectedReference(
    const FullyConnectedAttributes& attr, const TensorFloat32& input,
    const TestingRuntimeChannels& runtime_channels);

TensorFloat32 FullyConnectedRefDifferentWeightsForHeight(
    Tensor<OHWI, DataType::kFloat32> weights, const TensorFloat32& src);

TensorFloat32 FullyConnectedRefDifferentWeightsForHeight(
    Tensor<OHWI, DataType::kFloat32> weights, const TensorFloat32& src,
    const TestingRuntimeChannels& runtime_channels);

TensorFloat32 FullyConnectedWeightsBatchIdsReference(
    Tensor<OHWI, DataType::kFloat32> weights, const TensorFloat32& src,
    const TensorInt32& ids);

TensorInt32 FullyConnectedReference(
    const ml_drift::Tensor<BHWC, DataType::kInt8>& src_tensor_i8,
    const ml_drift::Tensor<OHWI, DataType::kInt8>& weights_i8);

TensorInt32 FullyConnectedReference(
    const ml_drift::Tensor<BHWC, DataType::kUint8>& src_tensor_ui8,
    const ml_drift::Tensor<OHWI, DataType::kInt8>& weights_i8,
    int src_sum_scale = 0);

std::vector<TensorFloat32> LSTMReference(const TensorFloat32& active_temp,
                                         const TensorFloat32& prev_state);

TensorFloat32 MaxUnpoolingReference(const MaxUnpooling2DAttributes& attr,
                                    const TensorFloat32& input,
                                    const TensorFloat32& src_indexes);

Tensor5DFloat32 MaxUnpoolingReference(const MaxUnpooling3DAttributes& attr,
                                      const Tensor5DFloat32& input,
                                      const Tensor5DFloat32& src_indexes);

TensorFloat32 AveragePoolingReference(const Pooling2DAttributes& attr,
                                      const TensorFloat32& input);

Tensor5DFloat32 AveragePoolingReference(const Pooling3DAttributes& attr,
                                        const Tensor5DFloat32& input);

std::vector<TensorFloat32> MaxPoolingReference(const Pooling2DAttributes& attr,
                                               const TensorFloat32& input);

std::vector<Tensor5DFloat32> MaxPoolingReference(
    const Pooling3DAttributes& attr, const Tensor5DFloat32& input);

TensorFloat32 ReduceReference(const std::set<Axis>& axis_to_reduce,
                              OperationType op_type,
                              const TensorFloat32& input);

Tensor5DFloat32 ReduceReference(const std::set<Axis>& axis_to_reduce,
                                OperationType op_type,
                                const Tensor5DFloat32& input);

TensorFloat32 ReshapeReference(const ReshapeAttributes& attr,
                               const TensorFloat32& input);

Tensor5DFloat32 ReshapeReference(const Reshape3DAttributes& attr,
                                 const Tensor5DFloat32& input);

TensorFloat32 ResizeReference(const Resize2DAttributes& attr,
                              const TensorFloat32& input);

Tensor5DFloat32 ResizeReference(const Resize3DAttributes& attr,
                                const Tensor5DFloat32& input);

TensorFloat32 SoftmaxReference(const SoftmaxAttributes& attr,
                               const TensorFloat32& input);

TensorFloat32 SoftmaxReduceReference(const SoftmaxAttributes& attr,
                                     const TensorFloat32& input);

Tensor5DFloat32 SoftmaxReference(const SoftmaxAttributes& attr,
                                 const Tensor5DFloat32& input);

absl::StatusOr<TensorFloat32> SliceReference(const SliceAttributes& attr,
                                             const TensorFloat32& input);

absl::StatusOr<Tensor5DFloat32> SliceReference(const Slice3DAttributes& attr,
                                               const Tensor5DFloat32& input);

TensorFloat32 TransposeReference(const TransposeAttributes& attr,
                                 const TensorFloat32& src);

Tensor5DFloat32 TransposeReference(const Transpose3DAttributes& attr,
                                   const Tensor5DFloat32& src);

TensorFloat32 PReLUReference(const PReLUAttributes& attr,
                             const TensorFloat32& input);

Tensor5DFloat32 PReLUReference(const PReLUAttributes& attr,
                               const Tensor5DFloat32& input);

TensorFloat32 ElementwiseReference(const TensorFloat32& src,
                                   OperationType op_type);
TensorFloat32 ElementwiseReference(const TensorFloat32& src,
                                   OperationType op_type, float value);
TensorFloat32 ElementwiseReference(const TensorFloat32& src0,
                                   const TensorFloat32& src1,
                                   OperationType op_type);

TensorFloat32 RMSNormalizationReference(const TensorFloat32& src, float eps);

TensorFloat32 StatisticalTopKReference(const TensorFloat32& src,
                                       float stddev_multiplier);

TensorFloat32 Winograd3x3ForwardRef(const TensorFloat32& src_tensor,
                                    const Padding2D& padding, int tile_size);
TensorFloat32 Winograd3x3BackwardRef(TensorFloat32 src_tensor,
                                     const BHWC& dst_shape, int tile_size);

Tensor<OHWI, DataType::kFloat32> MakeWeightsFromInt8(
    const Tensor<OHWI, DataType::kInt8>& weights_i8, float weights_scale,
    float weights_zero_point);

Tensor<OHWI, DataType::kFloat32> MakeWeightsFromInt8(
    const Tensor<OHWI, DataType::kInt8>& weights_i8,
    const Tensor<OHWI, DataType::kFloat32>& weights_scale,
    const Tensor<OHWI, DataType::kFloat32>& weights_zero_point);

Tensor<OHWI, DataType::kFloat32> MakeWeightsFromInt8(
    const Tensor<OHWI, DataType::kInt8>& weights_i8,
    const Tensor<Linear, DataType::kFloat32>& weights_scale,
    const Tensor<Linear, DataType::kFloat32>* weights_zero_point = nullptr);

// group_ids shape B/H must be 1.
std::pair<TensorInt32, Tensor<Linear, DataType::kInt32>> GroupsMapReference(
    const TensorInt32& group_ids, int num_groups);

std::pair<TensorInt32, Tensor<Linear, DataType::kInt32>>
PackedGroupsMapReference(const TensorInt32& groups_map,
                         const Tensor<Linear, DataType::kInt32>& groups_sizes);

TensorFloat32 RemapToReference(const TensorFloat32& src,
                               const TensorInt32& packed_map);
TensorFloat32 RemapFromReference(const TensorFloat32& src,
                                 const TensorInt32& packed_map,
                                 int num_groups_per_element);

TensorFloat32 ConvolutionWithIds(
    const TensorFloat32& src_tensor,
    const ml_drift::Tensor<OHWI, DataType::kFloat32>& weights,
    const TensorInt32& ids);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_TASK_TESTING_REF_OPS_H_
