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

#ifndef ML_DRIFT_COMMON_GPU_MODEL_BUILDER_H_
#define ML_DRIFT_COMMON_GPU_MODEL_BUILDER_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/kernels/reduce.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/model_hints.h"
#include "ml_drift/common/op_attrs.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/tensor_handle.h"

namespace ml_drift {

class OpBase;

class WeightsManager;

struct GpuModelBuilderOptions {
  ModelHints hints;
  TensorStorageType storage = TensorStorageType::kBuffer;
  bool use_f32_accum_for_f16_convolutions = false;
};

class GpuModelBuilder {
 public:
  using ValueId = uint32_t;
  using TensorHandle = ::ml_drift::TensorHandle;

  struct Weights {
    TensorHandle weights;
    WeightsDescription desc;
    OHWI shape;
    // scale and zero_point(optional) are for quantized weights
    OHWI scale_zp_shape;
    std::optional<TensorHandle> scale = std::nullopt;
    std::optional<TensorHandle> zero_point = std::nullopt;
    std::optional<TensorHandle> sum_i = std::nullopt;  // optional for scale/zp
  };

  GpuModelBuilder() = default;
  GpuModelBuilder(const GpuInfo& gpu_info, ModelHints hints,
                  CalculationsPrecision precision, TensorStorageType storage)
      : use_f32_accum_for_f16_convolutions_(precision ==
                                            CalculationsPrecision::kF32F16),
        gpu_info_(gpu_info),
        hints_(hints),
        default_storage_(storage) {}

  GpuModelBuilder(const GpuInfo& gpu_info,
                  const GpuModelBuilderOptions& options)
      : use_f32_accum_for_f16_convolutions_(
            options.use_f32_accum_for_f16_convolutions),
        gpu_info_(gpu_info),
        hints_(options.hints),
        default_storage_(options.storage) {}

  GpuModelBuilder(
      const GpuInfo& gpu_info, GpuModelBuilderOptions options,
      absl::flat_hash_map<ValueId, TensorDescriptor>&& tensor_descs,
      absl::flat_hash_map<ValueId, TensorDescriptor>&& const_tensor_descs,
      std::shared_ptr<WeightsManager> weights_manager = nullptr)
      : GpuModelBuilder(gpu_info, options) {
    gpu_model_.tensors = std::move(tensor_descs);
    gpu_model_.const_tensors = std::move(const_tensor_descs);
    weights_manager_ = weights_manager;
    ValueId max_id = 0;
    for (const auto& [id, tensor_desc] : gpu_model_.tensors) {
      max_id = std::max(max_id, id);
    }
    for (const auto& [id, tensor_desc] : gpu_model_.const_tensors) {
      max_id = std::max(max_id, id);
    }
    id_counter_ = max_id + 1;
  }

  GpuModelBuilder(GpuModelBuilder&&) = default;
  GpuModelBuilder& operator=(GpuModelBuilder&&) = default;
  GpuModelBuilder(const GpuModelBuilder&) = delete;
  GpuModelBuilder& operator=(const GpuModelBuilder&) = delete;
  virtual ~GpuModelBuilder() = default;

  // Creates a new GpuModelBuilder with the same settings as the current one.
  GpuModelBuilder CreateBuilder() const;

  CalculationsPrecision GetConvPrecision(DataType data_type) const {
    if (data_type != DataType::kFloat16) {
      return CalculationsPrecision::kF32;
    }
    // FLOAT16
    if (use_f32_accum_for_f16_convolutions_) {
      return CalculationsPrecision::kF32F16;
    } else {
      return CalculationsPrecision::kF16;
    }
  }

  TensorHandle AddTensor(const TensorDescriptor& tensor_desc);
  std::vector<TensorHandle> AddTensors(
      const std::vector<TensorDescriptor>& tensor_descs);
  TensorHandle AddTensor(int b, int h, int w, int c,
                         TensorStorageType storage_type, DataType data_type);
  TensorHandle AddTensor(int b, int h, int w, int c, DataType data_type);
  TensorHandle AddTensor(const BHWC& shape, DataType data_type);
  TensorHandle AddTensor(int b, int h, int w, int d, int c,
                         TensorStorageType storage_type, DataType data_type);
  TensorHandle AddTensor(int b, int h, int w, int d, int c, DataType data_type);
  TensorHandle AddTensor(const BHWDC& shape, DataType data_type);

  TensorHandle AddLinearTensor(int x, DataType data_type);

  TensorHandle AddConstantTensor(TensorDescriptor&& tensor_desc);
  TensorHandle AddConstantTensor(
      const Tensor<Linear, DataType::kFloat32>& tensor, DataType data_type);
  TensorHandle AddConstantTensor(const TensorFloat32& tensor,
                                 DataType data_type);

  absl::StatusOr<TensorHandle> GetTensor(ValueId id) const;

  TensorHandle Cast(const TensorHandle& src, DataType dst_type);

  TensorHandle BitCast(const TensorHandle& src, DataType dst_type);

  TensorHandle WinoConvolution(const TensorHandle& src,
                               const Convolution2DAttributes& attr,
                               int tile_size);
  TensorHandle Convolution(const TensorHandle& src,
                           const Convolution2DAttributes& attr);
  // weights are BHWC tensor, but we use it as OHWI for this operation.
  // bias is optional.
  absl::StatusOr<TensorHandle> Convolution(const TensorHandle& src,
                                           const TensorHandle& weights,
                                           const TensorHandle* bias,
                                           const Convolution2DAttributes& attr);
  TensorHandle ConvolutionTransposed(
      const TensorHandle& src, const ConvolutionTransposedAttributes& attr);
  // weights are BHWC tensor, but we use it as OHWI for this operation.
  TensorHandle ConvolutionTransposed(
      const TensorHandle& src, const TensorHandle& weights,
      const ConvolutionTransposedAttributes& attr);
  TensorHandle EmbeddingLookup(const TensorHandle& src, const Weights& weights,
                               DataType dst_type,
                               Axis lookup_axis = Axis::kChannels);
  WeightsDescription GetFullyConnectedWeightsDesc(
      DataType data_type, const OHWI& weights_shape) const;
  absl::StatusOr<TensorHandle> FullyConnectedExternalWeights(
      const TensorHandle& src, const Weights& weights,
      const TensorHandle* biases = nullptr,
      const TensorHandle* src_exp = nullptr,
      const ConvRuntimeCheckDesc& runtime_check = {},
      const TensorHandle* runtime_check_tensor = nullptr);

  WeightsDescription GetFullyConnectedInt8WeightsDesc(
      const OHWI& weights_shape);
  TensorHandle FullyConnectedInt8ExternalWeights(const TensorHandle& src,
                                                 const Weights& weights,
                                                 const TensorHandle* biases);
  TensorHandle FullyConnectedInt8ExternalWeights(
      const TensorHandle& src, const Weights& weights,
      const TensorHandle* biases, const TensorHandle* src_exp,
      const ConvRuntimeCheckDesc& runtime_check,
      const TensorHandle* runtime_check_tensor);

  WeightsDescription GetFullyConnectedInt4WeightsDesc(
      const OHWI& weights_shape);
  TensorHandle FullyConnectedInt4ExternalWeights(const TensorHandle& src,
                                                 const Weights& weights,
                                                 const TensorHandle* biases);

  WeightsDescription GetFullyConnectedInt2WeightsDesc(
      const OHWI& weights_shape);
  TensorHandle FullyConnectedInt2ExternalWeights(const TensorHandle& src,
                                                 const Weights& weights,
                                                 const TensorHandle* biases);

  TensorHandle DepthwiseConvolution(
      const TensorHandle& src, const DepthwiseConvolution2DAttributes& attr);
  absl::StatusOr<TensorHandle> DepthwiseConvolution(
      const TensorHandle& src, const TensorHandle& weights,
      const DepthwiseConvolution2DAttributes& attr);

  TensorHandle FullyConnected(const TensorHandle& src,
                              const FullyConnectedAttributes& attr);
  TensorHandle FullyConnected(const TensorHandle& src,
                              const FullyConnectedInt8Attributes& attr);
  TensorHandle FullyConnected(const TensorHandle& src,
                              const FullyConnectedInt4Attributes& attr);
  TensorHandle FullyConnected(const TensorHandle& src,
                              const FullyConnectedInt2Attributes& attr);

  TensorHandle PositionalEmbedding(const TensorHandle& src,
                                   const TensorHandle& position);

  TensorHandle QuantizeAndDequantize(
      const TensorHandle& src, const QuantizeAndDequantizeAttributes& attr);
  TensorHandle StaticRangeQuantization(
      const TensorHandle& src, const QuantizeAndDequantizeAttributes& attr);

  TensorHandle ReLU(const TensorHandle& src, const ReLUAttributes& attr);

  std::vector<TensorHandle> RoPE(const TensorHandle& src_l,
                                 const TensorHandle& src_r,
                                 const TensorHandle& position,
                                 const RoPEAttributes& attr = {});

  TensorHandle SplitRoPEConcat(const TensorHandle& src,
                               const TensorHandle& position,
                               const RoPEAttributes& attr = {});

  TensorHandle Softmax(const TensorHandle& src,
                       const SoftmaxRuntimeCheckDesc& runtime_check = {},
                       const TensorHandle* runtime_check_tensor = nullptr);
  TensorHandle SoftmaxReduce(
      const TensorHandle& src,
      const SoftmaxRuntimeCheckDesc& runtime_check = {},
      const TensorHandle* runtime_check_tensor = nullptr);
  TensorHandle SoftmaxElementwise(const TensorHandle& src,
                                  const TensorHandle& reduced_exp);

  TensorHandle Reshape(const TensorHandle& src, const BHWC& new_shape);
  TensorHandle Reshape(const TensorHandle& src, const BHWDC& new_shape);

  TensorHandle Transpose(const TensorHandle& src, const BHWC& perm);
  TensorHandle Transpose(const TensorHandle& src, const BHWDC& perm);

  TensorHandle Add(const TensorHandle& src, float value);
  TensorHandle Add(const TensorHandle& src, double value);
  TensorHandle Add(const TensorHandle& src, int value);
  TensorHandle Add(const TensorHandle& src,
                   const Tensor<Linear, DataType::kFloat32>& value);
  TensorHandle Add(const TensorHandle& left, const TensorHandle& right);

  TensorHandle Multiplication(const TensorHandle& src, float value);
  TensorHandle Multiplication(const TensorHandle& src, double value);
  TensorHandle Multiplication(const TensorHandle& src, int value);
  TensorHandle Multiplication(const TensorHandle& src,
                              const Tensor<Linear, DataType::kFloat32>& value);
  TensorHandle Multiplication(const TensorHandle& left,
                              const TensorHandle& right);

  TensorHandle Elementwise(const TensorHandle& src, OperationType op_type);
  TensorHandle Elementwise(const TensorHandle& src, OperationType op_type,
                           float value);
  TensorHandle Elementwise(const TensorHandle& src, OperationType op_type,
                           double value);
  TensorHandle Elementwise(const TensorHandle& src, OperationType op_type,
                           int value);
  TensorHandle Elementwise(const TensorHandle& src,
                           const ElementwiseAttributes& attr,
                           OperationType op_type);
  TensorHandle Elementwise(const TensorHandle& left, const TensorHandle& right,
                           OperationType op_type);

  TensorHandle Gather(const TensorHandle& src, const TensorHandle& indices,
                      Axis axis);

  TensorHandle Tile(const TensorHandle& src, Axis axis, int tiles_count = 2);
  TensorHandle Tile(const TensorHandle& src, const BHWC& new_shape);
  TensorHandle Tile(const TensorHandle& src, const BHWDC& new_shape);

  TensorHandle Reduce(const TensorHandle& src, Reduce::Type reduce_type,
                      const std::set<Axis>& axis);
  TensorHandle Reduce(const TensorHandle& src, OperationType op_type,
                      const std::set<Axis>& axis);

  TensorHandle Elementwise(const TensorHandle& src,
                           ElementwiseDescriptor&& op_desc,
                           const std::string& name);

  void AddGpuOperation(const std::vector<TensorHandle>& srcs,
                       const std::vector<TensorHandle>& dsts,
                       std::unique_ptr<GPUOperation>&& operation,
                       const std::string& name);
  void AddGpuOperation(const std::vector<ValueId>& src_ids,
                       const std::vector<ValueId>& dst_ids,
                       std::unique_ptr<GPUOperation>&& operation,
                       const std::string& name);

  void AddGpuOperation(const std::vector<TensorHandle>& srcs,
                       const TensorHandle& dst,
                       std::unique_ptr<GPUOperation>&& operation,
                       const std::string& name);

  TensorHandle Concat(const std::vector<TensorHandle>& srcs, Axis axis);
  TensorHandle Concat(const TensorHandle& first, const TensorHandle& second,
                      Axis axis);

  void Copy(const TensorHandle& src, const TensorHandle& dst);

  TensorHandle Cumsum(const TensorHandle& src, Axis axis);

  TensorHandle Padding(const TensorHandle& src, const PadAttributes& attr);

  TensorHandle Pooling(const TensorHandle& src,
                       const Pooling2DAttributes& attr);

  TensorHandle LayerNormalization(
      const TensorHandle& src, const Tensor<Linear, DataType::kFloat32>& gamma,
      const Tensor<Linear, DataType::kFloat32>& beta, float epsilon);

  TensorHandle HWCGroupNormalization(
      const TensorHandle& src, int groups,
      const Tensor<Linear, DataType::kFloat32>& gamma,
      const Tensor<Linear, DataType::kFloat32>& beta, float epsilon);

  TensorHandle RMSNormalization(
      const TensorHandle& src, float epsilon,
      const Tensor<Linear, DataType::kFloat32>* gamma = nullptr,
      const Tensor<Linear, DataType::kFloat32>* beta = nullptr);

  TensorHandle StatisticalTopK(const TensorHandle& src,
                               float stddev_multiplier);

  TensorHandle ResizeNearest(const TensorHandle& src, int scale,
                             bool align_corners, bool half_pixel_centers);
  TensorHandle ResizeBilinear(const TensorHandle& src, const HW& new_shape,
                              bool align_corners, bool half_pixel_centers);

  TensorHandle DepthToSpace(const TensorHandle& src, int block_size);
  TensorHandle SubTensor(const TensorHandle& src, const BHWC& start,
                         const BHWC& size);
  TensorHandle StridedSlice(const TensorHandle& src,
                            const SliceAttributes& attr);
  TensorHandle StridedSlice(const TensorHandle& src,
                            const Slice3DAttributes& attr);

  TensorHandle Sampling(const TensorHandle& src_logits,
                        const TensorHandle& src_indices,
                        const TensorHandle& probabilities,
                        const TensorHandle& params_i32_handle, int top_k_index);

  // left/right/output is BHWC tensors, but they are interpreted differently for
  // this operation.
  // Left is actually a tensor of shape [1, B, M, N] that is multiplied by
  // right that is of shape [1, B, N, K] tensor and the result is stored in
  // output tensor of shape [1, B, M, K].
  // B is batch size, M is number of rows in left tensor, N is number of columns
  // in left tensor and number of rows in right tensor, K is number of columns
  // in right tensor.
  //
  //    a_tensor   b_tensor
  //    1xBxMxN    1xBxNxK
  //          \    /
  //          matmul
  //          1xBxMxK
  //            |
  //          output
  absl::StatusOr<TensorHandle> BatchedMatMul(
      const TensorHandle& left, const TensorHandle& right,
      const BatchedMatMulAttributes& attr = {},
      const TensorHandle* src_exp = nullptr,
      const ConvRuntimeCheckDesc& runtime_check = {},
      const TensorHandle* runtime_check_tensor = nullptr);

  //    softmax
  //    1xBxMxN
  //       |
  //    a_tensor   b_tensor
  //    1xBxMxN    1xBxNxK
  //          \    /
  //          matmul
  //          1xBxMxK
  //            |
  //          output
  // See BatchedMatMul for dimensions mapping.
  absl::StatusOr<TensorHandle> SoftmaxBatchedMatMul(const TensorHandle& left,
                                                    const TensorHandle& right);

  bool UseFusedSoftmaxWithBatchedMatMul(
      const TensorHandle& softmax_input) const;

  //    a_tensor   b_tensor
  //    1xBxMxN    1xBxNxK
  //          \    /
  //          matmul
  //          1xBxMxK
  //            |
  //         softmax   c_tensor
  //         1xBxMxK   1xBxKxL
  //              \     /
  //              matmul
  //              1xBxMxL
  //                |
  //              output
  // See BatchedMatMul for dimensions mapping.
  absl::StatusOr<TensorHandle> BatchedMatMulSoftmaxBatchedMatMul(
      const TensorHandle& a_tensor, const TensorHandle& b_tensor,
      const TensorHandle& c_tensor, TensorHandle* mask_tensor = nullptr);

  TensorHandle SelectV2(const TensorHandle& cond, const TensorHandle& if_tensor,
                        const TensorHandle& else_tensor);

  std::vector<TensorHandle> Split(const TensorHandle& src, Axis axis,
                                  const std::vector<int>& sizes);
  std::vector<TensorHandle> Split(const TensorHandle& src, Axis axis,
                                  int tile_size);

  TensorHandle MakeGelu(const TensorHandle& src);
  TensorHandle MakeGeluTanh(const TensorHandle& src);

  GPUOperation CreateNormalize(const TensorHandle& src,
                               const TensorHandle& mean,
                               const TensorHandle& mean_squares,
                               const TensorHandle& dst,
                               const Tensor<HWC, DataType::kFloat32>& gamma,
                               const Tensor<HWC, DataType::kFloat32>& beta,
                               float epsilon);
  TensorHandle HWCGroupNorm(const TensorHandle& src, int groups, float epsilon,
                            const Tensor<Linear, DataType::kFloat32>& gamma,
                            const Tensor<Linear, DataType::kFloat32>& beta);

  TensorHandle SiLU(const TensorHandle& src);

  // top-k works for C dimension, BxHxWxC -> BxHxWxTopK
  std::vector<TensorHandle> TopK(const TensorHandle& src, int top_k_size);

  TensorHandle GetScalarTensor(const TensorHandle& tensor, BHWC coord);

  // Need this for using existing API and building model with predefined tensor
  // descriptors(from GraphFloat32). All APIs in this class return new
  // TensorHandle, but we need to use existing TensorHandle(id and descriptor)
  // from GraphFloat32.
  // This method is kind of equivalent to Copy of for every output tensor into
  // new and then merging them.
  // Must be called right after outputs were created. Invalid to call with
  // outputs that already were used for new nodes(as inputs).
  absl::Status UpdateOutputTensors(const std::vector<TensorHandle>& outputs,
                                   const std::vector<ValueId>& new_ids);
  absl::Status UpdateOutputTensor(const TensorHandle& output, ValueId new_id);

  absl::Status GetGpuModel(const std::vector<ValueId>& input_ids,
                           const std::vector<ValueId>& output_ids,
                           GpuModel* gpu_model);
  absl::Status GetGpuModel(
      const std::vector<std::pair<ValueId, ValueId>>& input_ids_and_refs,
      const std::vector<std::pair<ValueId, ValueId>>& output_ids_and_refs,
      GpuModel* gpu_model);

  // These methods allow creating optional nodes in the graph which can be
  // turned on/off at execution time. The |src| tensor passed to
  // BeginOptionalNodes() must be the same shape as |final_tensor| passed to
  // EndOptionalNodes(), as the result from |final_tensor| will be copied back
  // to the |src| tensor.
  struct OptionalNodeContext {
    size_t start_node;
    TensorHandle src;
    int tag;
  };
  OptionalNodeContext BeginOptionalNodes(int tag, const TensorHandle& src);
  absl::Status EndOptionalNodes(OptionalNodeContext context,
                                const TensorHandle& final_tensor,
                                bool add_copy_to_src = true);

  // The below methods are for dealing with subgraphs. Subgraphs allow sharing
  // nodes from external GpuModels. Example usage:
  //
  //     if (!builder.HasSubgraph(id)) {
  //       GpuModelBuilder subgraph_builder;
  //       // Add a bunch of ops to subgraph_builder...
  //       // Then register:
  //       builder.RegisterSubgraph(std::move(subgraph_builder), id,
  //           subgraph_inputs, subgraph_outputs);
  //     }
  //     // Use registered subgraph:
  //     output = builder.Subgraph(id, inputs);

  // Checks whether a subgraph has been registered with `subgraph_id`.
  bool HasSubgraph(const std::string& subgraph_id) {
    return gpu_model_.subgraphs.contains(subgraph_id);
  }

  // Registers a subgraph with `subgraph_id`, specifying inputs and outputs.
  absl::Status RegisterSubgraph(GpuModelBuilder subgraph_builder,
                                const std::string& subgraph_id,
                                const std::vector<TensorHandle>& inputs,
                                const std::vector<TensorHandle>& outputs);

  // Creates a call to a subgraph previously registered using
  // RegisterSubgraph().
  absl::StatusOr<std::vector<TensorHandle>> Subgraph(
      const std::string& subgraph_id, const std::vector<TensorHandle>& inputs);

  std::vector<TensorHandle> WeightsConversion(
      const TensorHandle& src_weights, Layout src_layout,
      const WeightsDescription& dst_desc, const OHWI& weights_shape,
      const TensorHandle* weights_scale = nullptr,
      const TensorHandle* weights_zero_point = nullptr);
  std::vector<TensorHandle> WeightsConversion(
      const TensorHandle& src_weights, const TensorHandle* weights_scale,
      const TensorHandle* weights_zero_point,
      const WeightsDescription& src_desc, const WeightsDescription& dst_desc,
      const OHWI& weights_shape, const ConvRuntimeCheckDesc& runtime_check = {},
      const TensorHandle* runtime_check_tensor = nullptr);
  std::vector<TensorHandle> WeightsConversion(
      const Weights& weights, const WeightsDescription& dst_desc,
      const ConvRuntimeCheckDesc& runtime_check = {},
      const TensorHandle* runtime_check_tensor = nullptr);

  std::vector<TensorHandle> GetWinograd3x3WeightsFromOHWI(
      const TensorHandle& weights_ohwi, const OHWI& weights_shape,
      WeightsDescription dst_weights_desc);

  TensorHandle GetWeightsSumIFromRawOHWI(const TensorHandle& src_weights,
                                         const OHWI& weights_shape,
                                         const DataType& src_data_type);

  const GpuInfo& gpu_info() const { return gpu_info_; }
  const TensorStorageType& default_storage() const { return default_storage_; }
  void SetDefaultStorage(TensorStorageType default_storage) {
    default_storage_ = default_storage;
  }

  // Instantiates and appends a dynamically resolved operation to the graph
  // context via the global OpRegistry.
  //
  // op_name: The string name of the registered operation.
  // inputs:  Input tensor handles.
  // attrs:   Attributes for the operation. This method will validate and
  //          normalize them against the defined schema.
  // returns: Output tensor handles.
  absl::StatusOr<std::vector<TensorHandle>> AppendOp(
      std::string_view op_name, const std::vector<TensorHandle>& inputs,
      const OpAttrs& attrs = OpAttrs());
  // GetLastGpuOperation is intended for use by the operation extension
  // framework; avoid using directly in standard graph construction.
  // TODO(dlho): Remove this after killing stable diffusion OpHolder.
  std::unique_ptr<GPUOperation>& GetLastGpuOperation() {
    return gpu_model_.nodes.back().gpu_operation;
  }

 private:
  friend class OpBase;

  // Adds a source tensor to the GPUOperation and registers the tensor
  // dependency inside the GpuModel graph.
  void AddSrcTensor(GPUOperation* op, const std::string& name,
                    const TensorHandle& handle);

  // Adds a destination tensor to the GPUOperation and registers the tensor
  // output mapping inside the GpuModel graph.
  void AddDstTensor(GPUOperation* op, const std::string& name,
                    const TensorHandle& handle);

  TensorHandle BatchedMatMulSoftmaxBatchedMatMulSingleKernel(
      const TensorHandle& a_tensor, const TensorHandle& b_tensor,
      const TensorHandle& c_tensor);

  absl::StatusOr<TensorHandle> BatchedMatMulSoftmaxBatchedMatMulSeparateKernels(
      const TensorHandle& a_tensor, const TensorHandle& b_tensor,
      const TensorHandle& c_tensor, TensorHandle* mask_tensor);

  absl::StatusOr<TensorHandle> FullyConnectedExternalSpatialWeights(
      const TensorHandle& src, const TensorHandle& weights,
      const TensorHandle* biases);

  TensorHandle Mask(const TensorHandle& true_tensor, TensorHandle* mask_tensor);

  TensorHandle ConcatInternal(const std::vector<TensorHandle>& srcs, Axis axis);
  BHWC GetOutputShapeConcat(const std::vector<TensorHandle>& srcs, Axis axis);

  void Split(const TensorHandle& src, Axis axis,
             std::vector<TensorHandle>* dsts);

  TensorHandle SplitRoPEConcatInternal(const TensorHandle& src,
                                       const TensorHandle& position,
                                       const RoPEAttributes& attr);

  TensorHandle FullyConnectedInt8QuantizedWithSrcQuantization(
      const TensorHandle& src, const Weights& weights,
      const TensorHandle* biases,
      WeightsDescription* conv_weights_desc_ptr = nullptr);

  TensorHandle FullyConnectedInt4QuantizedWithSrcQuantization(
      const TensorHandle& src, const Weights& weights,
      const TensorHandle* biases);

  // src_exp - the reduced softmax tensor from src to optimize attention
  // calculations.
  // runtime_check - is a structure that contains information about runtime
  // checks, for example local and global boundary checks.
  // runtime_check_tensor - is a tensor that contains runtime check values.
  TensorHandle FullyConnectedSrcFloatExternalWeightsWithConversion(
      const TensorHandle& src, const Weights& weights,
      const TensorHandle* biases, const TensorHandle* src_exp = nullptr,
      const ConvRuntimeCheckDesc& runtime_check = {},
      const TensorHandle* runtime_check_tensor = nullptr);

  Weights GetWeights(
      const std::variant<Tensor<OHWI, DataType::kInt8>,
                         Tensor<OHWI, DataType::kInt2>>& weights);
  Weights GetWeights(
      const std::variant<Tensor<OHWI, DataType::kInt8>,
                         Tensor<OHWI, DataType::kInt4>>& weights);

  TensorHandle GetWeightsScale(const Tensor<OHWI, DataType::kFloat32>& scale,
                               DataType float_type);
  TensorHandle GetWeightsZeroPoint(
      const Tensor<OHWI, DataType::kInt32>& zero_point, DataType float_type);

  // returns 2 tensors: quantized and params(min/max/sum)
  std::vector<TensorHandle> Quantize(const TensorHandle& src,
                                     PackedType quantized_type,
                                     bool calculate_sum);

  // original top-k works for C dimension, BxHxWxC -> BxHxWxTopK
  // this one works with reshaped src-dst, B = 1, H = BxHxW, W = C / 4, C = 4;
  // [1, BxHxW, C/4, 4] -> [1, BxHxW, 1, TopK];
  std::vector<TensorHandle> TopKInternal(const TensorHandle& src,
                                         int top_k_size);

  TensorHandle ToDHWBCC4(const TensorHandle& src);

  std::vector<TensorHandle> GetWeights(
      const Tensor<OHWI, DataType::kFloat32>& weights,
      WeightsDescription weights_desc);
  std::vector<TensorHandle> GetWeights(const Convolution2DAttributes& attr,
                                       WeightsDescription weights_desc);
  std::vector<TensorHandle> GetWinograd3x3Weights(
      const Tensor<OHWI, DataType::kFloat32>& weights,
      WeightsDescription conv_weights_desc, int tile_size);

  absl::StatusOr<std::pair<int, int>> GetNodeAndIndexByOutputId(
      ValueId output_id) const;

  bool use_f32_accum_for_f16_convolutions_ = false;
  GpuInfo gpu_info_;
  ModelHints hints_;
  TensorStorageType default_storage_;
  GpuModel gpu_model_;
  uint64_t id_counter_ = 0;
  std::shared_ptr<WeightsManager> weights_manager_ = nullptr;

  struct SharedWeights {
    int weights_id;
    WeightsDescription desc;
    std::vector<TensorHandle> weights_handles;
  };
  std::vector<SharedWeights> shared_weights_;
};

GpuModelBuilder::Weights CreateExternalWeights(
    const GpuModelBuilder::TensorHandle& weights,
    const WeightsDescription& weights_desc, const OHWI& weights_shape,
    const OHWI& scale_zp_shape = OHWI(1, 1, 1, 1),
    const GpuModelBuilder::TensorHandle* scale = nullptr,
    const GpuModelBuilder::TensorHandle* zero_point = nullptr,
    const GpuModelBuilder::TensorHandle* sum_i = nullptr);

// Converts the weights to a format that can be used by the model, and updates
// the create info of the model.
class WeightsManager {
 public:
  struct UploadWeightsInfo {
    ValueId input_id;
    const void* data;
    size_t size;
  };

  enum class TargetWeightsType {
    // Weights will be converted according to WeightsDescription, and will be
    // used by CONV_2D, FULLY_CONNECTED, etc.
    kStandard,

    // This is a sum of source weights in input channels. It's used for
    // dequantizing the int4/int8 results of quantized FullyConnected.
    kWeightsSumI,

    // Weights for 3x3 CONV_2D with Winograd Convolution.
    kWinograd3x3,
  };

  struct WeightsConversionRequest {
    std::vector<ValueId> main_model_weights_ids;
    WeightsDescription weights_desc;
    OHWI weights_shape;
    DataType src_data_type;
    const void* src_data_ptr;
    TargetWeightsType dst_weights_type = TargetWeightsType::kStandard;
    absl::Span<const float> scale_data;
    absl::Span<const int> zero_point_data;
  };
  WeightsManager() = default;

  static bool IsGpuWeightsPreparationSupported(const GpuInfo& gpu_info);

  void RegisterWeightsConversion(
      const std::vector<ValueId>& main_model_weights_ids,
      const WeightsDescription& weights_desc, const OHWI& weights_shape,
      DataType src_data_type, const void* src_data_ptr,
      absl::Span<const float> scale_data = {},
      absl::Span<const int> zero_point_data = {}) {
    WeightsConversionRequest request = {
        /*main_model_weights_ids=*/main_model_weights_ids,
        /*weights_desc=*/weights_desc,
        /*weights_shape=*/weights_shape,
        /*src_data_type=*/src_data_type,
        /*src_data_ptr=*/src_data_ptr,
        /*dst_weights_type=*/TargetWeightsType::kStandard,
        /*scale_data=*/scale_data,
        /*zero_point_data=*/zero_point_data,
    };
    weights_conversion_requests_.push_back(std::move(request));
  }

  void RegisterWeightsSumIConversion(
      const std::vector<ValueId>& main_model_weights_ids,
      const OHWI& weights_shape, const DataType src_data_type,
      const void* src_data_ptr) {
    WeightsConversionRequest request = {
        /*main_model_weights_ids=*/main_model_weights_ids,
        /*weights_desc=*/{},
        /*weights_shape=*/weights_shape,
        /*src_data_type=*/src_data_type,
        /*src_data_ptr=*/src_data_ptr,
        /*dst_weights_type=*/TargetWeightsType::kWeightsSumI,
    };
    weights_conversion_requests_.push_back(std::move(request));
  }

  void RegisterWinograd3x3WeightsConversion(
      const std::vector<ValueId>& main_model_weights_ids,
      const WeightsDescription& weights_desc, const OHWI& weights_shape,
      DataType src_data_type, const void* src_data_ptr) {
    WeightsConversionRequest request = {
        /*main_model_weights_ids=*/main_model_weights_ids,
        /*weights_desc=*/weights_desc,
        /*weights_shape=*/weights_shape,
        /*src_data_type=*/src_data_type,
        /*src_data_ptr=*/src_data_ptr,
        /*dst_weights_type=*/TargetWeightsType::kWinograd3x3,
    };
    weights_conversion_requests_.push_back(std::move(request));
  }

  absl::Status CreateConversionGpuModel(
      const GpuInfo& gpu_info, GpuModel* gpu_model,
      absl::flat_hash_map<ValueId, ValueId>* io_mapping,
      std::vector<UploadWeightsInfo>* upload_weights_infos);

  absl::Status UploadWeights(const GpuInfo& gpu_info,
                             const UploadWeightsInfo& upload_weights_info,
                             GpuModelBuilder& model_builder);

  bool ShouldOffloadPreparationToGPU(const GpuInfo& gpu_info,
                                     const OHWI& weights_shape,
                                     TargetWeightsType preparation_type);

  enum ScheduleStrategy {
    // The batch size will be fixed with a default value.
    kDefaultBatch,

    // The total weight size in each batch is capped at the size of the largest
    // individual weight, though we have a minimum batch size to avoid too many
    // small batches.
    //
    // This strategy is LLM friendly, because LLM's un-embedding matrix is
    // larger than other weights.
    kBatchByMaxWeightSize,
  };

  struct WeightsPrepOperationInfo {
    ValueId main_model_weight_id;
    TensorDescriptor src_desc;
    std::vector<TensorDescriptor> dst_descs;
    std::unique_ptr<GPUOperation> gpu_operation;
    const void* data_ptr;
    size_t size;
  };

 protected:
  std::vector<WeightsConversionRequest> weights_conversion_requests_;

  // Convert the collected weights conversion requests to GPU operations.
  static std::vector<WeightsPrepOperationInfo>
  ConvertWeightsPrepRequestsToOperations(
      const GpuInfo& gpu_info,
      std::vector<WeightsConversionRequest>&& requests);

  // Separate the array of GPU operations into batches.
  static std::vector<std::vector<WeightsPrepOperationInfo>> BatchGpuOperations(
      std::vector<WeightsPrepOperationInfo>&& operations,
      ScheduleStrategy schedule_strategy, size_t total_shared_tensor_size);

 private:
  GpuModelBuilder::TensorHandle AddRawWeightTensor(
      GpuModelBuilder& model_builder, const WeightsConversionRequest& request);
};

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_GPU_MODEL_BUILDER_H_
