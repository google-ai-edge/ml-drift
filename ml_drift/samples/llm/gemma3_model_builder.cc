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

#include "ml_drift/samples/llm/gemma3_model_builder.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/kernels/reduce.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/samples/llm/llm_config.h"
#include "ml_drift/samples/llm/llm_tensor_loader.h"
#include "ml_drift/samples/llm/llm_weights_loader.h"

namespace ml_drift {
namespace {

// GEMMA 3 TRANSFORMER BUILDER

// This file demonstrates how to build a Gemma 3 Transformer model using ML
// Drift (GpuModelBuilder). For code readability and clarity, the implementation
// is organized into four logical sections:
//
//   Section 1: Build Context & Tensor Helpers
//              - Uses GpuModelBuilder::TensorHandle directly without
//              indirection
//              - Bundles shared model building state (GPU builder, loader,
//              config)
//              - Provides clean wrapper methods for common graph operations
//
//   Section 2: Custom GPU Kernels & Low-Level Hardware Operations
//              - Position IDs, Attention Masking, RoPE, KV Cache Operations
//              - Fused GELU-Tanh activation and infinity capping
//
//   Section 3: Transformer Layers & Neural Network Blocks
//              - RMSNorm, Linear, Feed-Forward (MLP), Multi-Head Attention
//              - TransformerLayer (combines attention, MLP, and residual links)
//
//   Section 4: Top-Level Model Assembly
//              - EmbedTokens, LmHead, and
//              Gemma3ModelBuilder::BuildModelInternal

using TensorHandle = GpuModelBuilder::TensorHandle;

// SECTION 1: BUILD CONTEXT & TENSOR HELPERS

// Context object bundling shared dependencies for model building,
// reducing parameter boilerplate across layer functions.
struct BuildContext {
  GpuModelBuilder& builder;
  CreateGpuModelInfo& create_info;
  LlmTensorLoader* loader;
  const LlmConfig& config;
  int sequence_size;
  TensorHandle params_i32;

  DataType data_type() const {
    return DeduceDataTypeFromPrecision(create_info.precision);
  }

  TensorHandle AddTensor(const TensorDescriptor& desc) {
    return builder.AddTensor(desc);
  }

  TensorHandle AddTensor(const BHWC& shape, DataType dt) {
    return builder.AddTensor(shape, dt);
  }

  TensorHandle AddTensor(int b, int h, int w, int c, DataType dt) {
    return builder.AddTensor(b, h, w, c, dt);
  }

  TensorHandle AddConstantTensor(
      const ml_drift::Tensor<ml_drift::Linear, DataType::kFloat32>& t,
      DataType dt) {
    return builder.AddConstantTensor(t, dt);
  }

  TensorHandle AddExternalTensor(GpuSpatialTensor* tensor) {
    TensorHandle th = builder.AddTensor(tensor->GetDescriptor());
    create_info.external_immutable_tensors[th.id] = tensor;
    return th;
  }

  TensorHandle FullyConnected(
      const TensorHandle& src, const GpuModelBuilder::Weights& weights,
      const TensorHandle* biases = nullptr,
      const TensorHandle* src_exp = nullptr,
      const ConvRuntimeCheckDesc& runtime_check = {},
      const TensorHandle* runtime_check_tensor = nullptr) {
    const TensorHandle* check_tensor = runtime_check_tensor;
    if (check_tensor == nullptr &&
        (runtime_check.src_end_ch_index >= 0 ||
         runtime_check.dst_end_ch_index >= 0 ||
         runtime_check.ring_o_offset_index.has_value() ||
         runtime_check.ring_i_offset_index.has_value())) {
      check_tensor = &params_i32;
    }
    auto status_or = builder.FullyConnectedExternalWeights(
        src, weights, biases, src_exp, runtime_check, check_tensor);
    ABSL_CHECK_OK(status_or);
    return status_or.value();
  }

  TensorHandle FullyConnectedInt4(const TensorHandle& src,
                                  const GpuModelBuilder::Weights& weights,
                                  const TensorHandle* biases = nullptr) {
    return builder.FullyConnectedInt4ExternalWeights(src, weights, biases);
  }

  TensorHandle FullyConnectedInt8(const TensorHandle& src,
                                  const GpuModelBuilder::Weights& weights,
                                  const TensorHandle* biases = nullptr) {
    return builder.FullyConnectedInt8ExternalWeights(src, weights, biases);
  }

  TensorHandle SoftmaxReduce(const TensorHandle& src,
                             const SoftmaxRuntimeCheckDesc& runtime_params,
                             const TensorHandle* params_tensor = nullptr) {
    return builder.SoftmaxReduce(src, runtime_params,
                                 params_tensor ? params_tensor : &params_i32);
  }

  TensorHandle SplitRoPEConcat(const TensorHandle& src,
                               const TensorHandle& position_ids,
                               const RoPEAttributes& rope_attr) {
    return builder.SplitRoPEConcat(src, position_ids, rope_attr);
  }

  TensorHandle EmbeddingLookup(const TensorHandle& input_ids,
                               const GpuModelBuilder::Weights& weights,
                               DataType dt, Axis axis) {
    return builder.EmbeddingLookup(input_ids, weights, dt, axis);
  }

  TensorHandle RMSNormalization(
      const TensorHandle& src, float epsilon,
      ml_drift::Tensor<ml_drift::Linear, DataType::kFloat32>* gamma) {
    return builder.RMSNormalization(src, epsilon, gamma);
  }

  TensorHandle Reduce(const TensorHandle& src, Reduce::Type type,
                      const std::set<Axis>& axes) {
    return builder.Reduce(src, type, axes);
  }

  TensorHandle Reshape(const TensorHandle& src, const BHWC& new_shape) {
    return builder.Reshape(src, new_shape);
  }

  TensorHandle Transpose(const TensorHandle& src, const BHWC& perm) {
    return builder.Transpose(src, perm);
  }

  TensorHandle Slice(const TensorHandle& src, const BHWC& offset,
                     const BHWC& slice_shape) {
    return builder.SubTensor(src, offset, slice_shape);
  }

  TensorHandle Add(const TensorHandle& a, const TensorHandle& b) {
    return builder.Add(a, b);
  }

  TensorHandle Mul(const TensorHandle& a, const TensorHandle& b) {
    return builder.Multiplication(a, b);
  }

  TensorHandle Mul(const TensorHandle& a, float scalar) {
    return builder.Multiplication(a, scalar);
  }

  void AddGpuOperation(const std::vector<TensorHandle>& inputs,
                       const std::vector<TensorHandle>& outputs,
                       std::unique_ptr<GPUOperation> operation,
                       const std::string& name = "") {
    builder.AddGpuOperation(inputs, outputs, std::move(operation), name);
  }
};

// SECTION 2: CUSTOM GPU KERNELS & LOW-LEVEL HARDWARE OPERATIONS

bool UseSmallLocalCache(const LlmConfig& config, int sequence_size) {
  if (config.sliding_window_size <= 0 ||
      config.num_local_layers_per_global <= 0) {
    return false;
  }
  return config.cache_size > config.sliding_window_size;
}

int GetLocalCacheSize(const LlmConfig& config, int sequence_size) {
  if (!UseSmallLocalCache(config, sequence_size)) {
    return config.cache_size;
  }
  const int local_cache_size = sequence_size == 1
                                   ? config.sliding_window_size
                                   : config.sliding_window_size + sequence_size;
  return std::min(config.cache_size, (local_cache_size + 3) / 4 * 4);
}

// Caps floating-point values to avoid infinity/NaN issues in FP16 mode.
TensorHandle CapInfinity(BuildContext& ctx, const TensorHandle& src) {
  if (src.tensor_desc.GetDataType() == DataType::kFloat16) {
    TensorHandle dst = ctx.AddTensor(src.tensor_desc);
    ElementwiseDescriptor op_desc;
    op_desc.args.AddHalf("min_val", static_cast<half>(-kMaxHalf));
    op_desc.args.AddHalf("max_val", static_cast<half>(kMaxHalf));
    op_desc.code = R"(
        out_value.x = clamp(in_value.x, args.min_val, args.max_val);
        out_value.y = clamp(in_value.y, args.min_val, args.max_val);
        out_value.z = clamp(in_value.z, args.min_val, args.max_val);
        out_value.w = clamp(in_value.w, args.min_val, args.max_val);
    )";
    OperationDef definition;
    definition.src_tensors.push_back(src.tensor_desc);
    definition.dst_tensors.push_back(dst.tensor_desc);
    auto cap_infinity_op = std::make_unique<GPUOperation>(CreateGpuOperation(
        definition, std::move(op_desc), dst.tensor_desc.GetBHWCShape(),
        dst.tensor_desc.GetBHWCShape()));
    ctx.AddGpuOperation({src}, {dst}, std::move(cap_infinity_op),
                        "cap_infinity");
    return dst;
  }
  return src;
}

// Generates token position IDs based on current runtime token offsets.
TensorHandle PositionIds(BuildContext& ctx) {
  TensorHandle position_ids =
      ctx.AddTensor(1, 1, ctx.sequence_size, 1, DataType::kInt32);
  GPUOperation op;
  BufferDescriptor buffer_desc;
  buffer_desc.element_type = DataType::kInt32;
  buffer_desc.element_size = 1;
  buffer_desc.memory_type = MemoryType::kGlobal;
  op.AddSrcBuffer("src", buffer_desc);
  op.AddDstTensor("dst", position_ids.tensor_desc);
  op.args_.AddInt("param_index", LlmRuntimeParams::kTokenOffsetIndex);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;
  op.code_ = R"(
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.dst.Width() || Y >= args.dst.Height() || S >= args.dst.Slices()) {
    return;
  }
  args.dst::type res_value;
  int offset = args.src.Read(args.param_index);
  res_value.x = ucl::Convert<args.dst::scalar_type>(offset + X);
  res_value.y = ucl::Convert<args.dst::scalar_type>(offset + X);
  res_value.z = ucl::Convert<args.dst::scalar_type>(offset + X);
  res_value.w = ucl::Convert<args.dst::scalar_type>(offset + X);
  args.dst.Write(res_value, X, Y, S);
}
)";
  ctx.AddGpuOperation({ctx.params_i32}, {position_ids},
                      std::make_unique<GPUOperation>(std::move(op)),
                      "fill_position_ids");
  return position_ids;
}

// Applies attention masking by setting masked positions to -30000.0f.
TensorHandle SelectMask(BuildContext& ctx, const TensorHandle& src,
                        const TensorHandle& mask) {
  TensorHandle dst = ctx.AddTensor(src.tensor_desc);
  ElementwiseDescriptor op_desc;
  // exp(-30000) underflows cleanly to exactly 0.0f.
  if (src.tensor_desc.GetDataType() == DataType::kFloat32) {
    op_desc.args.AddFloat("min_float", -30000.0f);
  } else {
    op_desc.args.AddHalf("min_float", static_cast<half>(-30000.0f));
  }
  op_desc.code = R"(
      out_value.x = in2_value.x ? args.min_float : in_value.x;
      out_value.y = in2_value.y ? args.min_float : in_value.y;
      out_value.z = in2_value.z ? args.min_float : in_value.z;
      out_value.w = in2_value.w ? args.min_float : in_value.w;
  )";
  OperationDef definition;
  definition.src_tensors.push_back(src.tensor_desc);
  definition.src_tensors.push_back(mask.tensor_desc);
  definition.dst_tensors.push_back(dst.tensor_desc);
  auto select_mask_op = std::make_unique<GPUOperation>(CreateGpuOperation(
      definition, std::move(op_desc), mask.tensor_desc.GetBHWCShape(),
      dst.tensor_desc.GetBHWCShape()));
  ctx.AddGpuOperation({src, mask}, {dst}, std::move(select_mask_op),
                      "select_mask");
  return dst;
}

// GPU kernel generating causal and sliding-window attention masks.
GPUOperation AttentionMaskOp(const TensorDescriptor& params_desc,
                             const TensorDescriptor& out_desc, int mask_width,
                             std::optional<int> sliding_window,
                             bool sliding_local_cache = false) {
  GPUOperation op;
  op.args_.AddInt("mask_width", mask_width);
  op.args_.AddInt("window_size", sliding_window.has_value()
                                     ? std::max(*sliding_window, 0)
                                     : 0);
  op.args_.AddInt("sliding_local_cache", sliding_local_cache ? 1 : 0);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  op.code_ = absl::Substitute(R"(
  MAIN_FUNCTION($$0) {
  int linear_xb = ucl::GetGlobalId<0>();
  int X = linear_xb / args.dst.Batch();
  int B = linear_xb % args.dst.Batch();
  args.dst.SetBatchRef(B);
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.dst.Width() || Y >= args.dst.Height() || S >= args.dst.Slices()) {
    return;
  }
  int token_offset;
  args.params_tensor.ReadPerChannel(token_offset, 0, 0, $0);

  args.dst::type mask_value;
  int k;
  bool visible;

  // x
  k = S * 4 + 0;
  if (args.sliding_local_cache == 1) {
    int token_index = k - args.window_size + 1;
    visible = (token_index <= X);
    if (args.window_size > 0) visible &= (token_index > X - args.window_size);
    visible &= (token_offset + token_index >= 0);
  } else {
    int token_index = k - token_offset;
    visible = (token_index <= X);
    if (args.window_size > 0) visible &= (token_index > X - args.window_size);
  }
  mask_value.x = !visible;

  // y
  k = S * 4 + 1;
  if (args.sliding_local_cache == 1) {
    int token_index = k - args.window_size + 1;
    visible = (token_index <= X);
    if (args.window_size > 0) visible &= (token_index > X - args.window_size);
    visible &= (token_offset + token_index >= 0);
  } else {
    int token_index = k - token_offset;
    visible = (token_index <= X);
    if (args.window_size > 0) visible &= (token_index > X - args.window_size);
  }
  mask_value.y = !visible;

  // z
  k = S * 4 + 2;
  if (args.sliding_local_cache == 1) {
    int token_index = k - args.window_size + 1;
    visible = (token_index <= X);
    if (args.window_size > 0) visible &= (token_index > X - args.window_size);
    visible &= (token_offset + token_index >= 0);
  } else {
    int token_index = k - token_offset;
    visible = (token_index <= X);
    if (args.window_size > 0) visible &= (token_index > X - args.window_size);
  }
  mask_value.z = !visible;

  // w
  k = S * 4 + 3;
  if (args.sliding_local_cache == 1) {
    int token_index = k - args.window_size + 1;
    visible = (token_index <= X);
    if (args.window_size > 0) visible &= (token_index > X - args.window_size);
    visible &= (token_offset + token_index >= 0);
  } else {
    int token_index = k - token_offset;
    visible = (token_index <= X);
    if (args.window_size > 0) visible &= (token_index > X - args.window_size);
  }
  mask_value.w = !visible;

  args.dst.Write(mask_value, X, Y, S);
})",
                              LlmRuntimeParams::kTokenOffsetIndex);
  op.AddSrcTensor("params_tensor", params_desc);
  op.AddDstTensor("dst", out_desc);
  return op;
}

// Builds either a global causal attention mask or a local sliding-window mask.
TensorHandle AttentionMask(BuildContext& ctx, bool is_global) {
  const bool use_small_local =
      !is_global && UseSmallLocalCache(ctx.config, ctx.sequence_size);
  const int cache_size = use_small_local
                             ? GetLocalCacheSize(ctx.config, ctx.sequence_size)
                             : ctx.config.cache_size;
  const BHWC mask_shape = BHWC(1, 1, ctx.sequence_size, cache_size);
  TensorHandle attention_mask = ctx.AddTensor(mask_shape, DataType::kBool);
  ctx.AddGpuOperation(
      {ctx.params_i32}, {attention_mask},
      std::make_unique<GPUOperation>(AttentionMaskOp(
          ctx.params_i32.tensor_desc, attention_mask.tensor_desc, cache_size,
          is_global ? std::nullopt
                    : std::optional<int>(ctx.config.sliding_window_size),
          use_small_local)),
      is_global ? "apply_attention_mask" : "apply_local_attention_mask");
  return attention_mask;
}

// --- KV Cache Tensor Layout and Shape Helpers ---

WeightsDescription GetKWeightsDescription(DataType data_type, int cache_size,
                                          int kv_heads_count,
                                          int head_dimension) {
  WeightsDescription weights_desc;
  weights_desc.type = data_type;
  weights_desc.layout = WeightsLayout::kOSpatialIOGroupO4I4;
  weights_desc.output_group_size = (cache_size + 3) / 4;
  return weights_desc;
}

OHWI GetKWeightsShape(int cache_size, int kv_heads_count, int head_dimension) {
  return OHWI(cache_size, kv_heads_count, 1, head_dimension);
}

WeightsDescription GetVWeightsDescription(DataType data_type, int cache_size,
                                          int kv_heads_count,
                                          int head_dimension) {
  WeightsDescription weights_desc;
  weights_desc.type = data_type;
  weights_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
  weights_desc.output_group_size = (head_dimension + 3) / 4;
  return weights_desc;
}

OHWI GetVWeightsShape(int cache_size, int kv_heads_count, int head_dimension) {
  return OHWI(head_dimension, kv_heads_count, 1, cache_size);
}

TensorDescriptor GetKCacheTensorDescriptor(DataType data_type, int cache_size,
                                           int kv_heads_count,
                                           int head_dimension) {
  WeightsDescription weights_desc = GetKWeightsDescription(
      data_type, cache_size, kv_heads_count, head_dimension);
  TensorDescriptor weights_td = TensorDescriptor(
      weights_desc.type, TensorStorageType::kBuffer, Layout::kLinear);
  weights_td.SetBHWCShape(
      BHWC(1, 1, 1,
           GetTotalElementsCountForLayout(
               weights_desc,
               GetKWeightsShape(cache_size, kv_heads_count, head_dimension))));
  return weights_td;
}

TensorDescriptor GetVCacheTensorDescriptor(DataType data_type, int cache_size,
                                           int kv_heads_count,
                                           int head_dimension) {
  WeightsDescription weights_desc = GetVWeightsDescription(
      data_type, cache_size, kv_heads_count, head_dimension);
  TensorDescriptor weights_td = TensorDescriptor(
      weights_desc.type, TensorStorageType::kBuffer, Layout::kLinear);
  weights_td.SetBHWCShape(
      BHWC(1, 1, 1,
           GetTotalElementsCountForLayout(
               weights_desc,
               GetVWeightsShape(cache_size, kv_heads_count, head_dimension))));
  return weights_td;
}

// Updates shared K and V cache tensors with newly computed key/value tokens.
void UpdateKVCache(BuildContext& ctx, const TensorHandle& k,
                   const TensorHandle& v, const TensorHandle& cache_k,
                   const TensorHandle& cache_v, int head_dimension,
                   int num_kv_heads, bool is_global) {
  std::string code = absl::Substitute(R"(
MAIN_FUNCTION($$0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.src_k.Width() || Y >= args.batch_size || S >= args.src_k.Slices()) {
    return;
  }
  int token_index_offset = args.params.Read($0);
  int active_tokens = args.params.Read($1);
  int token_index = token_index_offset + X;
  if (token_index >= args.global_cache_size || token_index >= active_tokens) {
    return;
  }

  token_index %= args.cache_size;

  args.src_k::type value_k = args.src_k.Read(X, Y, S);
  args.src_v::type value_v = args.src_v.Read(X, Y, S);
  {
    // cache_k - kOSpatialIOGroupO4I4 layout, O - cache_size, I - head_size
    int k_o = token_index;
    int k_o_slice = k_o / 4;
    int k_o4_index = k_o % 4;
    int k_i = S * 4;
    int k_i_slice = k_i / 4;
    int k_sp = Y;
    int i_slices = (args.head_size + 3) / 4;
    int o_slices = (args.cache_size + 3) / 4;
    int k_index = ((k_sp * i_slices + k_i_slice) * o_slices + k_o_slice) * 4 + k_o4_index;
    args.cache_k.WriteLinear(value_k, k_index);
  }
  {
    // cache_v - kOSpatialIOGroupI4O4 layout, O - head_size, I - cache_size
    int v_o = S * 4;
    int v_o_slice = v_o / 4;
    int v_i = token_index;
    int v_i_slice = v_i / 4;
    int v_i4_index = v_i % 4;
    int v_sp = Y;
    int i_slices = (args.cache_size + 3) / 4;
    int o_slices = (args.head_size + 3) / 4;
    int v_index = ((v_sp * i_slices + v_i_slice) * o_slices + v_o_slice) * 4 + v_i4_index;
    args.cache_v.WriteLinear(value_v, v_index);
  }
})",
                                      LlmRuntimeParams::kTokenOffsetIndex,
                                      LlmRuntimeParams::kActiveTokensIndex);

  class AddValuesToCacheOp : public GPUOperation {
   public:
    AddValuesToCacheOp() = default;
    int3 GetGridSize() const override {
      return int3(src_[0]->Width(), batch_size_, src_[0]->Slices());
    }
    int batch_size_ = 1;
  };

  const int cache_size = GetCacheSize(ctx.config, is_global);
  const int kv_heads_count = ctx.config.batch_size * num_kv_heads;
  AddValuesToCacheOp custom_op;
  custom_op.batch_size_ = kv_heads_count;
  custom_op.args_.AddInt("cache_size", cache_size);
  custom_op.args_.AddInt("batch_size", kv_heads_count);
  custom_op.args_.AddInt("global_cache_size", ctx.config.cache_size);
  custom_op.args_.AddInt("head_size", head_dimension);
  custom_op.AddSrcTensor("src_k", k.tensor_desc);
  custom_op.AddSrcTensor("src_v", v.tensor_desc);
  BufferDescriptor params_buffer;
  params_buffer.element_type = DataType::kInt32;
  params_buffer.element_size = 1;
  custom_op.AddSrcBuffer("params", params_buffer);
  custom_op.AddDstTensor("cache_k", cache_k.tensor_desc);
  custom_op.AddDstTensor("cache_v", cache_v.tensor_desc);
  custom_op.code_ = code;

  ctx.AddGpuOperation(
      {k, v, ctx.params_i32}, {cache_k, cache_v},
      std::make_unique<AddValuesToCacheOp>(std::move(custom_op)),
      "cache_update");
}

// Extracts a compact sliding window from global/ring K/V cache buffers for
// local layers.
void ExtractLocalCache(BuildContext& ctx, const TensorHandle& k,
                       const TensorHandle& v, const TensorHandle& k_local,
                       const TensorHandle& v_local, int head_dimension,
                       int num_kv_heads) {
  std::string code = absl::Substitute(R"(
MAIN_FUNCTION($$0) {
  int dst_cache_id = ucl::GetGlobalId<0>();
  int kv_id = ucl::GetGlobalId<1>();
  int head_dim_id = ucl::GetGlobalId<2>();
  if (dst_cache_id >= args.local_cache_size ||
      kv_id >= args.kv_heads_count ||
      head_dim_id >= args.slices_count) {
    return;
  }

  int ring_offset = args.params.Read($0);
  int src_cache_id = (dst_cache_id + ring_offset) % args.ringbuffer_size;
  {
    // K-cache: kOSpatialIOGroupO4I4 layout
    int dst_k_o = dst_cache_id;
    int dst_k_o_slice = dst_k_o / 4;
    int dst_k_o4_index = dst_k_o % 4;

    int src_k_o = src_cache_id;
    int src_k_o_slice = src_k_o / 4;
    int src_k_o4_index = src_k_o % 4;

    int k_i = head_dim_id * 4;
    int k_i_slice = k_i / 4;
    int k_sp = kv_id;
    int i_slices = (args.head_size + 3) / 4;
    int src_o_slices = (args.ringbuffer_size + 3) / 4;
    int dst_o_slices = (args.local_cache_size + 3) / 4;
    int src_index = ((k_sp * i_slices + k_i_slice) * src_o_slices + src_k_o_slice) * 4 + src_k_o4_index;
    int dst_index = ((k_sp * i_slices + k_i_slice) * dst_o_slices + dst_k_o_slice) * 4 + dst_k_o4_index;
    args.src_k::type value_k = args.src_k.Read(src_index);
    args.local_k.WriteLinear(value_k, dst_index);
  }
  {
    // V-cache: kOSpatialIOGroupI4O4 layout
    int v_o = head_dim_id * 4;
    int v_o_slice = v_o / 4;

    int dst_v_i = dst_cache_id;
    int dst_v_i_slice = dst_v_i / 4;
    int dst_v_i4_index = dst_v_i % 4;

    int src_v_i = src_cache_id;
    int src_v_i_slice = src_v_i / 4;
    int src_v_i4_index = src_v_i % 4;

    int v_sp = kv_id;
    int o_slices = (args.head_size + 3) / 4;
    int src_i_slices = (args.ringbuffer_size + 3) / 4;
    int dst_i_slices = (args.local_cache_size + 3) / 4;
    int src_index = ((v_sp * src_i_slices + src_v_i_slice) * o_slices + v_o_slice) * 4 + src_v_i4_index;
    int dst_index = ((v_sp * dst_i_slices + dst_v_i_slice) * o_slices + v_o_slice) * 4 + dst_v_i4_index;
    args.src_v::type value_v = args.src_v.Read(src_index);
    args.local_v.WriteLinear(value_v, dst_index);
  }
})",
                                      LlmRuntimeParams::kRingOffsetIndex);

  class ExtractLocalCacheOp : public GPUOperation {
   public:
    ExtractLocalCacheOp() = default;
    int3 GetGridSize() const override {
      return int3(local_cache_size_, kv_heads_count_, slices_count_);
    }
    int local_cache_size_ = 1;
    int kv_heads_count_ = 1;
    int slices_count_ = 1;
  };

  const int kv_heads_count = ctx.config.batch_size * num_kv_heads;
  const int local_cache_size = GetLocalCacheSize(ctx.config, ctx.sequence_size);
  ExtractLocalCacheOp custom_op;
  custom_op.kv_heads_count_ = kv_heads_count;
  custom_op.local_cache_size_ = local_cache_size;
  custom_op.slices_count_ = (head_dimension + 3) / 4;
  custom_op.args_.AddInt("ringbuffer_size",
                         GetCacheSize(ctx.config, /*is_global=*/false));
  custom_op.args_.AddInt("local_cache_size", custom_op.local_cache_size_);
  custom_op.args_.AddInt("window_size", ctx.config.sliding_window_size);
  custom_op.args_.AddInt("global_cache_size", ctx.config.cache_size);
  custom_op.args_.AddInt("head_size", head_dimension);
  custom_op.args_.AddInt("kv_heads_count", custom_op.kv_heads_count_);
  custom_op.args_.AddInt("slices_count", custom_op.slices_count_);
  custom_op.AddSrcTensor("src_k", k.tensor_desc);
  custom_op.AddSrcTensor("src_v", v.tensor_desc);
  BufferDescriptor params_buffer;
  params_buffer.element_type = DataType::kInt32;
  params_buffer.element_size = 1;
  custom_op.AddSrcBuffer("params", params_buffer);
  custom_op.AddDstTensor("local_k", k_local.tensor_desc);
  custom_op.AddDstTensor("local_v", v_local.tensor_desc);
  custom_op.code_ = code;

  ctx.AddGpuOperation(
      {k, v, ctx.params_i32}, {k_local, v_local},
      std::make_unique<ExtractLocalCacheOp>(std::move(custom_op)),
      "extract_local_cache");
}

// Standard Q/K/V layout transformation: reshapes and transposes heads/tokens.
void TransformQKV(BuildContext& ctx, int sequence_size, TensorHandle* q,
                  TensorHandle* k, TensorHandle* v) {
  const int batch_size = ctx.config.batch_size;
  const int head_size = ctx.config.head_dimension;
  const int heads_count = ctx.config.number_of_heads;
  const int kv_heads_count = ctx.config.number_of_kv_heads;
  const int kv_group_size = heads_count / kv_heads_count;
  if (q != nullptr) {
    *q = ctx.Reshape(*q,
                     BHWC{batch_size, sequence_size, heads_count, head_size});
    *q = ctx.Transpose(*q, BHWC(0, 2, 1, 3));
    *q = ctx.Reshape(*q, BHWC{1, batch_size * kv_heads_count,
                              sequence_size * kv_group_size, head_size});
  }
  if (k != nullptr) {
    *k = ctx.Reshape(
        *k, BHWC{batch_size, sequence_size, kv_heads_count, head_size});
    *k = ctx.Transpose(*k, BHWC(0, 2, 1, 3));
    *k = ctx.Reshape(
        *k, BHWC{1, batch_size * kv_heads_count, sequence_size, head_size});
  }
  if (v != nullptr) {
    *v = ctx.Reshape(
        *v, BHWC{batch_size, sequence_size, kv_heads_count, head_size});
    *v = ctx.Transpose(*v, BHWC(0, 2, 1, 3));
    *v = ctx.Reshape(
        *v, BHWC{1, batch_size * kv_heads_count, sequence_size, head_size});
  }
}

// Fused GPU Kernel: Reshapes Q/K/V, applies per-head RMSNorm, applies
// Rotary Position Embedding (RoPE), and writes K/V directly into the KV cache
// in a single hardware-optimized kernel.
void TransformQKVRmsNormRoPE(BuildContext& ctx, const TensorHandle& position,
                             TensorHandle* q, const TensorHandle& k,
                             const TensorHandle& v, const TensorHandle& cache_k,
                             const TensorHandle& cache_v, bool is_global,
                             float min_timescale, float max_timescale,
                             const std::string& name) {
  const bool k_eq_v = (k.id == v.id);
  const int heads_count = ctx.config.number_of_heads;
  const int head_size = ctx.config.head_dimension;
  const int batch_size = ctx.config.batch_size;
  const int kv_heads_count = ctx.config.number_of_kv_heads;
  const int kv_head_group_size = heads_count / kv_heads_count;
  const int cache_size = GetCacheSize(ctx.config, is_global);
  DataType data_type = ctx.data_type();

  TensorHandle q_dst =
      ctx.AddTensor(BHWC{1, batch_size * kv_heads_count,
                         kv_head_group_size * ctx.sequence_size, head_size},
                    q->tensor_desc.GetDataType());

  // Fuse Q scaling into Q RMSNorm.
  ml_drift::Tensor<ml_drift::Linear, DataType::kFloat32> q_gamma;
  q_gamma.shape = ml_drift::Linear(head_size);
  q_gamma.data = ctx.loader->LoadFloat32(
      absl::StrCat(name, ".q_norm", kWeightSuffix), head_size);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  for (int i = 0; i < q_gamma.data.size(); ++i) {
    q_gamma.data[i] = (1.0f + q_gamma.data[i]) * scale;
  }
  TensorHandle q_gamma_tensor = ctx.AddConstantTensor(q_gamma, data_type);

  ml_drift::Tensor<ml_drift::Linear, DataType::kFloat32> k_gamma;
  k_gamma.shape = ml_drift::Linear(head_size);
  k_gamma.data = ctx.loader->LoadFloat32(
      absl::StrCat(name, ".k_norm", kWeightSuffix), head_size);
  for (int i = 0; i < k_gamma.data.size(); ++i) {
    k_gamma.data[i] = 1.0f + k_gamma.data[i];
  }
  TensorHandle k_gamma_tensor = ctx.AddConstantTensor(k_gamma, data_type);

  std::string pow_func_name = "pow";
  std::string sin_func_name = "sin";
  std::string cos_func_name = "cos";
  if (ctx.builder.gpu_info().IsAdreno() &&
      ctx.builder.gpu_info().IsApiOpenCl()) {
    pow_func_name = "native_powr";
    sin_func_name = "native_sin";
    cos_func_name = "native_cos";
  }

  std::string code = R"(MAIN_FUNCTION($0) {
  int DST_X = ucl::GetGlobalId<0>();
  int DST_Y = ucl::GetGlobalId<1>();
  int DST_S_GLOBAL = ucl::GetGlobalId<2>();

  int LOCAL_S = ucl::GetLocalId<2>();

  int half_slices_count = args.q_dst.Slices() / 2;
  bool is_active = (DST_X < args.q_dst.Width() && DST_Y < args.q_dst.Height() && DST_S_GLOBAL < half_slices_count);

  int DST_S0 = DST_S_GLOBAL;
  int DST_S1 = DST_S_GLOBAL + half_slices_count;

  int batch_id = DST_Y / args.kv_heads_count;
  int kv_head_id = DST_Y % args.kv_heads_count;
  int kv_head_group_id = DST_X / args.sequence_size;
  int seq_id = DST_X % args.sequence_size;

  args.q_src.SetBatchRef(batch_id);
  args.k_src.SetBatchRef(batch_id);
)";

  if (!k_eq_v) {
    code += R"(  args.v_src.SetBatchRef(batch_id);
)";
  }

  code += absl::Substitute(R"(
  __local float shared_q_sum[$0];
  __local float shared_k_sum[$0];
  __local float shared_inv_std_q;
  __local float shared_inv_std_k;
)",
                           head_size / 8);

  code += R"(
  // 1. Cooperative Load & Sum of Squares for Q
  float4 q_val0 = ucl::Init<float4>(0.0f);
  float4 q_val1 = ucl::Init<float4>(0.0f);
  float local_sum_q = 0.0f;
  if (is_active) {
    int q_head_id = kv_head_id * args.kv_head_group_size + kv_head_group_id;
    int SRC_S0 = q_head_id * args.q_dst.Slices() + DST_S0;
    int SRC_S1 = q_head_id * args.q_dst.Slices() + DST_S1;
    q_val0 = ucl::Convert<float4>(args.q_src.Read(seq_id, 0, SRC_S0));
    q_val1 = ucl::Convert<float4>(args.q_src.Read(seq_id, 0, SRC_S1));
    local_sum_q = dot(q_val0, q_val0) + dot(q_val1, q_val1);
  }
  shared_q_sum[LOCAL_S] = local_sum_q;

  // Cooperative Load & Sum of Squares for K
  float4 raw_k_val0 = ucl::Init<float4>(0.0f);
  float4 raw_k_val1 = ucl::Init<float4>(0.0f);
  float4 k_val0 = ucl::Init<float4>(0.0f);
  float4 k_val1 = ucl::Init<float4>(0.0f);
  float local_sum_k = 0.0f;
  if (is_active && kv_head_group_id == 0) {
    int K_SRC_S0 = kv_head_id * args.q_dst.Slices() + DST_S0;
    int K_SRC_S1 = kv_head_id * args.q_dst.Slices() + DST_S1;
    raw_k_val0 = ucl::Convert<float4>(args.k_src.Read(seq_id, 0, K_SRC_S0));
    raw_k_val1 = ucl::Convert<float4>(args.k_src.Read(seq_id, 0, K_SRC_S1));
    k_val0 = raw_k_val0;
    k_val1 = raw_k_val1;
    local_sum_k = dot(k_val0, k_val0) + dot(k_val1, k_val1);
  }
  shared_k_sum[LOCAL_S] = local_sum_k;

  ucl::SyncThreads<WorkGroup, Local>();
)";

  code += absl::Substitute(R"(
  int reduction_size = $0;
  while (reduction_size > 1) {
    int active_thread_limit = reduction_size / 2;
    int offset = (reduction_size + 1) / 2;
    if (LOCAL_S < active_thread_limit) {
      shared_q_sum[LOCAL_S] += shared_q_sum[LOCAL_S + offset];
      shared_k_sum[LOCAL_S] += shared_k_sum[LOCAL_S + offset];
    }
    ucl::SyncThreads<WorkGroup, Local>();
    reduction_size = offset;
  }
)",
                           head_size / 8);

  code += R"(
  if (LOCAL_S == 0) {
    float inv_ch = 1.0f / ucl::Convert<float>(args.q_dst.Channels());
    shared_inv_std_q = rsqrt(shared_q_sum[0] * inv_ch + 1e-6f);
    shared_inv_std_k = rsqrt(shared_k_sum[0] * inv_ch + 1e-6f);
  }
  ucl::SyncThreads<WorkGroup, Local>();

  if (is_active) {
    float stddev_inv_q = shared_inv_std_q;
    float4 val0 = q_val0 * stddev_inv_q;
    float4 val1 = q_val1 * stddev_inv_q;

    float4 gamma0 = args.q_gamma.Read<float>(DST_S0);
    float4 gamma1 = args.q_gamma.Read<float>(DST_S1);
    val0 = val0 * gamma0;
    val1 = val1 * gamma1;

    // 2. RoPE Constants
    float4 fraction;
    float inv_q_dst_ch = 1.0f / ucl::Convert<float>(args.q_dst.Channels());
    fraction.x = 2.0f * ucl::Convert<float>(DST_S_GLOBAL * 4 + 0) * inv_q_dst_ch;
    fraction.y = 2.0f * ucl::Convert<float>(DST_S_GLOBAL * 4 + 1) * inv_q_dst_ch;
    fraction.z = 2.0f * ucl::Convert<float>(DST_S_GLOBAL * 4 + 2) * inv_q_dst_ch;
    fraction.w = 2.0f * ucl::Convert<float>(DST_S_GLOBAL * 4 + 3) * inv_q_dst_ch;
    float4 min_timescale = ucl::Init<float4>(args.min_timescale);
    float4 max_timescale = ucl::Init<float4>(args.max_timescale);
)";
  code += "    float4 timescale = min_timescale * " + pow_func_name +
          "(max_timescale / min_timescale, fraction);\n";
  code +=
      "    float4 pos_val = "
      "ucl::Init<float4>(args.position.Read<float>(seq_id, 0, 0).x);\n";
  code += "    float4 sinusoid_inp = pos_val / timescale;\n";
  code += "    Type sin_val = ucl::Convert<Type>(" + sin_func_name +
          "(sinusoid_inp));\n";
  code += "    Type cos_val = ucl::Convert<Type>(" + cos_func_name +
          "(sinusoid_inp));\n";

  code += R"(
    Type val0_t = ucl::Convert<Type>(val0);
    Type val1_t = ucl::Convert<Type>(val1);
    Type q_out0 = val0_t * cos_val - val1_t * sin_val;
    Type q_out1 = val1_t * cos_val + val0_t * sin_val;
    args.q_dst.Write(q_out0, DST_X, DST_Y, DST_S0);
    args.q_dst.Write(q_out1, DST_X, DST_Y, DST_S1);

    if (kv_head_group_id == 0) {
      float stddev_inv_k = shared_inv_std_k;
      float4 k_v0 = k_val0 * stddev_inv_k;
      float4 k_v1 = k_val1 * stddev_inv_k;
      float4 k_g0 = args.k_gamma.Read<float>(DST_S0);
      float4 k_g1 = args.k_gamma.Read<float>(DST_S1);
      k_v0 = k_v0 * k_g0;
      k_v1 = k_v1 * k_g1;
      Type k_val0_t = ucl::Convert<Type>(k_v0);
      Type k_val1_t = ucl::Convert<Type>(k_v1);
      Type out0 = k_val0_t * cos_val - k_val1_t * sin_val;
      Type out1 = k_val1_t * cos_val + k_val0_t * sin_val;
)";

  if (k_eq_v) {
    code += R"(
      Type v_out0 = ucl::Convert<Type>(raw_k_val0);
      Type v_out1 = ucl::Convert<Type>(raw_k_val1);
)";
  } else {
    code += R"(
      int V_SRC_S0 = kv_head_id * args.q_dst.Slices() + DST_S0;
      int V_SRC_S1 = kv_head_id * args.q_dst.Slices() + DST_S1;
      Type v_out0 = args.v_src.Read(seq_id, 0, V_SRC_S0);
      Type v_out1 = args.v_src.Read(seq_id, 0, V_SRC_S1);
)";
  }

  code += absl::Substitute(R"(
      int token_index_offset = args.params.Read($0);
      int active_tokens = args.params.Read($1);
      int token_index = token_index_offset + seq_id;
      if (token_index < args.global_cache_size && token_index < active_tokens) {
        token_index %= args.cache_size;
        int head_slices = args.q_dst.Slices();
        int token_chunks = (args.cache_size + 3) / 4;
        int t_chunk = token_index / 4;
        int t_rem = token_index % 4;

        int k_idx0 = ((DST_Y * head_slices + DST_S0) * token_chunks + t_chunk) * 4 + t_rem;
        int k_idx1 = ((DST_Y * head_slices + DST_S1) * token_chunks + t_chunk) * 4 + t_rem;
        args.cache_k.WriteLinear(ucl::Convert<args.cache_k::type>(out0), k_idx0);
        args.cache_k.WriteLinear(ucl::Convert<args.cache_k::type>(out1), k_idx1);

        int v_idx0 = ((DST_Y * token_chunks + t_chunk) * head_slices + DST_S0) * 4 + t_rem;
        int v_idx1 = ((DST_Y * token_chunks + t_chunk) * head_slices + DST_S1) * 4 + t_rem;
        args.cache_v.WriteLinear(ucl::Convert<args.cache_v::type>(v_out0), v_idx0);
        args.cache_v.WriteLinear(ucl::Convert<args.cache_v::type>(v_out1), v_idx1);
      }
    }
  }
})",
                           LlmRuntimeParams::kTokenOffsetIndex,
                           LlmRuntimeParams::kActiveTokensIndex);

  code = absl::StrReplaceAll(
      code, {{"Type", ToUclDataType(q_dst.tensor_desc.GetDataType(), 4)}});

  class TransformQKVRmsNormApplyRoPEOp : public GPUOperation {
   public:
    int3 GetGridSize() const override {
      const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
      const int grid_y = dst_[0]->Height() * dst_[0]->Depth();
      const int grid_z = dst_[0]->Slices() / 2;
      return int3(grid_x, grid_y, grid_z);
    }
    std::vector<int3> GetPossibleKernelWorkGroups(
        TuningType tuning_type, const GpuInfo& gpu_info,
        const KernelInfo& kernel_info) const override {
      return {work_group_size_};
    }
  };

  TransformQKVRmsNormApplyRoPEOp custom_op;
  custom_op.args_.AddInt("kv_heads_count", kv_heads_count);
  custom_op.args_.AddInt("kv_head_group_size", kv_head_group_size);
  custom_op.args_.AddInt("sequence_size", ctx.sequence_size);
  custom_op.args_.AddInt("cache_size", cache_size);
  custom_op.args_.AddInt("global_cache_size", ctx.config.cache_size);
  custom_op.args_.AddFloat("min_timescale", min_timescale);
  custom_op.args_.AddFloat("max_timescale", max_timescale);
  custom_op.work_group_size_ = int3(1, 1, head_size / 8);

  custom_op.AddSrcTensor("q_src", q->tensor_desc);
  custom_op.AddSrcTensor("k_src", k.tensor_desc);
  if (!k_eq_v) {
    custom_op.AddSrcTensor("v_src", v.tensor_desc);
  }
  custom_op.AddSrcTensor("position", position.tensor_desc);
  custom_op.AddSrcTensor("q_gamma", q_gamma_tensor.tensor_desc);
  custom_op.AddSrcTensor("k_gamma", k_gamma_tensor.tensor_desc);

  BufferDescriptor params_buffer;
  params_buffer.element_type = DataType::kInt32;
  params_buffer.element_size = 1;
  custom_op.AddSrcBuffer("params", params_buffer);

  custom_op.AddDstTensor("q_dst", q_dst.tensor_desc);
  custom_op.AddDstTensor("cache_k", cache_k.tensor_desc);
  custom_op.AddDstTensor("cache_v", cache_v.tensor_desc);

  custom_op.code_ = code;
  custom_op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  std::vector<TensorHandle> inputs = {*q, k};
  if (!k_eq_v) {
    inputs.push_back(v);
  }
  inputs.push_back(position);
  inputs.push_back(q_gamma_tensor);
  inputs.push_back(k_gamma_tensor);
  inputs.push_back(ctx.params_i32);

  std::vector<TensorHandle> outputs = {q_dst, cache_k, cache_v};

  ctx.AddGpuOperation(
      inputs, outputs,
      std::make_unique<TransformQKVRmsNormApplyRoPEOp>(std::move(custom_op)),
      "QKVRmsNormRoPE");

  *q = q_dst;
}

// Fused GPU Kernel: Computes GELU-Tanh activation and multiplies by the Up
// projection in the Feed-Forward (MLP) block.
GPUOperation CreateGeluTanhMulOp(const TensorDescriptor& gate_desc,
                                 const TensorDescriptor& up_desc,
                                 const TensorDescriptor& dst_desc) {
  ElementwiseDescriptor op_desc;
  op_desc.code = R"(
      float4 gate_f32 = ucl::Convert<float4>(in_value);
      float4 up_f32 = ucl::Convert<float4>(in2_value);
      float4 gate_sq = gate_f32 * gate_f32;
      float4 tanh_in = gate_f32 * (0.7978845608f + 0.03567740813f * gate_sq);
      float4 clamped_in = clamp(tanh_in, ucl::Init<float4>(-10.0f), ucl::Init<float4>(10.0f));
      float4 tanh_val = tanh(clamped_in);
      float4 gate_up_half = 0.5f * gate_f32 * up_f32;
      float4 res_f32 = clamp(gate_up_half * (1.0f + tanh_val), ucl::Init<float4>(-65504.0f), ucl::Init<float4>(65504.0f));
      out_value = ucl::Convert<args.dst_tensor::type>(res_f32);
  )";
  OperationDef definition;
  definition.src_tensors.push_back(gate_desc);
  definition.src_tensors.push_back(up_desc);
  definition.dst_tensors.push_back(dst_desc);
  return CreateGpuOperation(definition, std::move(op_desc),
                            dst_desc.GetBHWCShape(), dst_desc.GetBHWCShape());
}

TensorHandle GeluTanhMul(BuildContext& ctx, const TensorHandle& gate,
                         const TensorHandle& up) {
  TensorHandle dst = ctx.AddTensor(gate.tensor_desc);
  auto gelu_tanh_mul_op = std::make_unique<GPUOperation>(
      CreateGeluTanhMulOp(gate.tensor_desc, up.tensor_desc, dst.tensor_desc));
  ctx.AddGpuOperation({gate, up}, {dst}, std::move(gelu_tanh_mul_op),
                      "gelu_tanh_mul");
  return dst;
}

// SECTION 3: TRANSFORMER LAYERS & NEURAL NETWORK BLOCKS

// Root Mean Square Normalization (RMSNorm).
// Normalizes input activations across channels and scales by learned gamma
// weights. Gemma3 uses scale_shift = 1.0f by default.
TensorHandle RMSNorm(BuildContext& ctx, const TensorHandle& src,
                     const std::string& name, float scale_shift = 1.0f,
                     float multiplier = 1.0f) {
  ml_drift::Tensor<ml_drift::Linear, DataType::kFloat32> gamma;
  gamma.shape = ml_drift::Linear(src.tensor_desc.GetBHWCShape().c);
  gamma.data = ctx.loader->LoadFloat32(absl::StrCat(name, kWeightSuffix),
                                       src.tensor_desc.GetBHWCShape().c);
  for (int i = 0; i < gamma.data.size(); ++i) {
    gamma.data[i] = (scale_shift + gamma.data[i]) * multiplier;
  }
  return ctx.RMSNormalization(src, /*epsilon=*/1e-6f, &gamma);
}

// Linear (Fully Connected) projection layer with automatic precision detection.
// Automatically detects whether weights are INT4, INT8, or floating-point,
// loads the weights accordingly, and invokes the appropriate fully connected
// operation.
TensorHandle Linear(BuildContext& ctx, const TensorHandle& src,
                    const std::string& name, int out_channels,
                    const TensorHandle* biases = nullptr) {
  const int in_channels = src.tensor_desc.GetBHWCShape().c;
  const OHWI weights_shape = OHWI(out_channels, 1, 1, in_channels);
  auto [weights, weight_type] = GetWeights(ctx, name, weights_shape);

  if (weight_type == DataType::kInt4) {
    return ctx.FullyConnectedInt4(src, weights, biases);
  } else if (weight_type == DataType::kInt8) {
    return ctx.FullyConnectedInt8(src, weights, biases);
  } else {
    return ctx.FullyConnected(src, weights, biases);
  }
}

// Feed-Forward (MLP) Block:
//   1. Input RMSNorm (.pre_feedforward_layernorm)
//   2. Parallel Gate and Up projections (.mlp.gate_proj, .mlp.up_proj)
//   3. Fused GELU-Tanh activation with gating elementwise multiplication
//   4. Down projection (.mlp.down_proj) with infinity capping
//   5. Output RMSNorm (.post_feedforward_layernorm)
TensorHandle FFLayer(BuildContext& ctx, const TensorHandle& src,
                     const std::string& name) {
  TensorHandle x =
      RMSNorm(ctx, src, absl::StrCat(name, ".pre_feedforward_layernorm"));

  TensorHandle gate = Linear(ctx, x, absl::StrCat(name, ".mlp.gate_proj"),
                             ctx.config.hidden_dimension);
  TensorHandle up = Linear(ctx, x, absl::StrCat(name, ".mlp.up_proj"),
                           ctx.config.hidden_dimension);

  TensorHandle ff_hidden = GeluTanhMul(ctx, gate, up);

  TensorHandle ff_out =
      Linear(ctx, ff_hidden, absl::StrCat(name, ".mlp.down_proj"),
             ctx.config.model_dimension);
  ff_out = CapInfinity(ctx, ff_out);

  return RMSNorm(ctx, ff_out,
                 absl::StrCat(name, ".post_feedforward_layernorm"));
}

// Multi-Head Attention (MHA / GQA) Block:
//   1. Project input to Q, K, and V (.q_proj, .k_proj, .v_proj)
//   2. Apply Rotary Position Embedding (RoPE) and per-head RMSNorms
//   3. Update KV Caches with new K and V tokens
//   4. Compute scaled dot-product attention scores with causal/sliding masking
//   5. Apply softmax and weighted sum over V
//   6. Output linear projection (.o_proj)
TensorHandle SelfAttention(BuildContext& ctx, const TensorHandle& src,
                           const TensorHandle& attention_mask,
                           const TensorHandle& position_ids,
                           const std::string& name, int layer_i,
                           const TensorHandle& k_cache_in,
                           const TensorHandle& v_cache_in) {
  const int head_dimension = ctx.config.head_dimension;
  const int num_kv_heads = ctx.config.number_of_kv_heads;
  const int heads_count = ctx.config.number_of_heads;

  const int q_channels = head_dimension * heads_count;
  const int kv_channels = head_dimension * num_kv_heads;

  TensorHandle q = Linear(ctx, src, absl::StrCat(name, ".q_proj"), q_channels);
  TensorHandle k = Linear(ctx, src, absl::StrCat(name, ".k_proj"), kv_channels);
  TensorHandle v = Linear(ctx, src, absl::StrCat(name, ".v_proj"), kv_channels);

  const bool is_global =
      (layer_i + 1) % (ctx.config.num_local_layers_per_global + 1) == 0;
  float min_timescale = 1.0f;
  float max_timescale = is_global ? 1000000.0f : 10000.0f;

  if (ctx.config.head_dimension % 8 == 0) {
    // Also updates the KV cache.
    TransformQKVRmsNormRoPE(ctx, position_ids, &q, k, v, k_cache_in, v_cache_in,
                            is_global, min_timescale, max_timescale, name);
  } else {
    TransformQKV(ctx, ctx.sequence_size, &q, &k, &v);

    const float scale =
        1.0f / std::sqrt(static_cast<float>(ctx.config.head_dimension));
    q = RMSNorm(ctx, q, absl::StrCat(name, ".q_norm"),
                /*scale_shift=*/1.0f, /*multiplier=*/scale);
    k = RMSNorm(ctx, k, absl::StrCat(name, ".k_norm"));

    RoPEAttributes rope_attr;
    rope_attr.max_timescale = max_timescale;
    q = ctx.SplitRoPEConcat(q, position_ids, rope_attr);
    k = ctx.SplitRoPEConcat(k, position_ids, rope_attr);

    UpdateKVCache(ctx, k, v, k_cache_in, v_cache_in, head_dimension,
                  num_kv_heads, is_global);
  }

  TensorHandle k_cache = k_cache_in;
  TensorHandle v_cache = v_cache_in;
  const bool use_local_cache =
      !is_global && UseSmallLocalCache(ctx.config, ctx.sequence_size);
  int cache_size = use_local_cache
                       ? GetLocalCacheSize(ctx.config, ctx.sequence_size)
                       : ctx.config.cache_size;
  const int kv_heads_count = ctx.config.batch_size * num_kv_heads;
  const DataType cache_dt = ctx.data_type();

  const bool use_local_cache_without_copy =
      (ctx.builder.gpu_info().IsApple() || ctx.builder.gpu_info().IsIntel()) &&
      use_local_cache && SizeInBitsOf(cache_dt) >= 16;
  const bool use_end_checks = use_local_cache ? is_global : true;

  if (use_local_cache_without_copy) {
    cache_size = GetCacheSize(ctx.config, is_global);
  } else if (use_local_cache) {
    k_cache = ctx.AddTensor(GetKCacheTensorDescriptor(
        cache_dt, cache_size, kv_heads_count, head_dimension));
    v_cache = ctx.AddTensor(GetVCacheTensorDescriptor(
        cache_dt, cache_size, kv_heads_count, head_dimension));
    ExtractLocalCache(ctx, k_cache_in, v_cache_in, k_cache, v_cache,
                      head_dimension, num_kv_heads);
  }

  // Attention dot-product with partially fused softmax.
  WeightsDescription k_weights_desc = GetKWeightsDescription(
      cache_dt, cache_size, kv_heads_count, head_dimension);
  OHWI k_weights_shape =
      GetKWeightsShape(cache_size, kv_heads_count, head_dimension);
  ConvRuntimeCheckDesc qk_runtime_check;
  if (use_end_checks) {
    qk_runtime_check.dst_end_ch_index =
        LlmRuntimeParams::kActiveTokensAlignedIndex;
  }
  if (use_local_cache_without_copy) {
    qk_runtime_check.ring_o_offset_index = LlmRuntimeParams::kRingOffsetIndex;
    qk_runtime_check.ring_size = cache_size;
    k_weights_shape.o = GetLocalCacheSize(ctx.config, ctx.sequence_size);
  }
  const GpuModelBuilder::Weights k_external_weights =
      CreateExternalWeights(k_cache, k_weights_desc, k_weights_shape);
  TensorHandle attn =
      ctx.FullyConnected(q, k_external_weights, /*biases=*/nullptr,
                         /*src_exp=*/nullptr, qk_runtime_check);
  attn = SelectMask(ctx, attn, attention_mask);

  SoftmaxRuntimeCheckDesc softmax_runtime_params;
  if (use_end_checks) {
    softmax_runtime_params.end_ch_index =
        LlmRuntimeParams::kActiveTokensAlignedIndex;
  }
  TensorHandle att_exp = ctx.SoftmaxReduce(attn, softmax_runtime_params);

  WeightsDescription v_weights_desc = GetVWeightsDescription(
      cache_dt, cache_size, kv_heads_count, head_dimension);
  OHWI v_weights_shape =
      GetVWeightsShape(cache_size, kv_heads_count, head_dimension);
  ConvRuntimeCheckDesc v_runtime_check;
  if (use_end_checks) {
    v_runtime_check.src_end_ch_index =
        LlmRuntimeParams::kActiveTokensAlignedIndex;
  }
  if (use_local_cache_without_copy) {
    v_runtime_check.ring_i_offset_index = LlmRuntimeParams::kRingOffsetIndex;
    v_runtime_check.ring_size = cache_size;
    v_weights_shape.i = GetLocalCacheSize(ctx.config, ctx.sequence_size);
  }
  const GpuModelBuilder::Weights v_external_weights =
      CreateExternalWeights(v_cache, v_weights_desc, v_weights_shape);
  attn = ctx.FullyConnected(attn, v_external_weights, /*biases=*/nullptr,
                            &att_exp, v_runtime_check);

  // Reshape attention output for final linear projection.
  const int batch_size = ctx.config.batch_size;
  attn = ctx.Reshape(
      attn, BHWC{batch_size, heads_count, ctx.sequence_size, head_dimension});
  attn = ctx.Transpose(attn, BHWC(0, 2, 1, 3));
  attn = ctx.Reshape(attn, BHWC{batch_size, 1, ctx.sequence_size,
                                heads_count * head_dimension});

  return Linear(ctx, attn, absl::StrCat(name, ".o_proj"),
                ctx.config.model_dimension);
}

// A single Transformer layer in Gemma 3:
//   1. Attention block with RMSNorm before and after attention, with residual
//   link.
//   2. MLP (FeedForward) block with RMSNorms, with residual link.
TensorHandle TransformerLayer(BuildContext& ctx, const TensorHandle& src,
                              const TensorHandle& attention_mask,
                              const TensorHandle& position_ids,
                              const std::string& name, int layer_i,
                              const TensorHandle& k_cache_in,
                              const TensorHandle& v_cache_in) {
  TensorHandle residual = src;

  // 1. Attention Block
  TensorHandle x = RMSNorm(ctx, src, absl::StrCat(name, ".input_layernorm"));
  x = SelfAttention(ctx, x, attention_mask, position_ids,
                    absl::StrCat(name, ".self_attn"), layer_i, k_cache_in,
                    v_cache_in);
  x = RMSNorm(ctx, x, absl::StrCat(name, ".post_attention_layernorm"));
  x = ctx.Add(residual, x);

  // 2. Feed-Forward / MLP Block
  residual = x;
  TensorHandle mlp_output = FFLayer(ctx, x, name);
  x = ctx.Add(residual, mlp_output);

  return CapInfinity(ctx, x);
}

// SECTION 4: TOP-LEVEL MODEL ASSEMBLY

struct EmbeddingState {
  GpuModelBuilder::Weights weights;
  DataType weight_type = DataType::kUnknown;
};

// Loads token embeddings with automatic precision detection and applies
// scaling by sqrt(d_model).
TensorHandle EmbedTokens(BuildContext& ctx, const TensorHandle& input_ids,
                         EmbeddingState* state) {
  const std::string embed_name = "model.embed_tokens";
  const OHWI weights_shape =
      OHWI(ctx.config.vocabulary_size, 1, 1, ctx.config.model_dimension);

  auto [weights, weight_type] = GetWeights(ctx, embed_name, weights_shape);
  state->weights = weights;
  state->weight_type = weight_type;

  DataType emb_type = ctx.data_type();
  TensorHandle embeddings =
      ctx.EmbeddingLookup(input_ids, state->weights, emb_type, Axis::kWidth);

  // Scale embeddings by sqrt(model_dimension)
  return ctx.Mul(embeddings,
                 std::sqrt(static_cast<float>(ctx.config.model_dimension)));
}

// Projects final hidden states to vocabulary logits using LM head weights.
// Reuses token embedding weights if separate lm_head weights are not present.
absl::StatusOr<TensorHandle> LmHead(BuildContext& ctx,
                                    const TensorHandle& hidden_state,
                                    const EmbeddingState& emb_state) {
  if (ctx.loader->HasTensor(absl::StrCat("lm_head", kWeightSuffix))) {
    return Linear(ctx, hidden_state, "lm_head", ctx.config.vocabulary_size);
  } else {
    // Reuse loaded embedding weights for LM head.
    if (emb_state.weight_type == DataType::kInt4) {
      return ctx.FullyConnectedInt4(hidden_state, emb_state.weights,
                                    /*biases=*/nullptr);
    } else if (emb_state.weight_type == DataType::kInt8) {
      return ctx.FullyConnectedInt8(hidden_state, emb_state.weights,
                                    /*biases=*/nullptr);
    } else {
      return ctx.FullyConnected(hidden_state, emb_state.weights,
                                /*biases=*/nullptr);
    }
  }
}

}  // namespace

Gemma3ModelBuilder::Gemma3ModelBuilder(
    const LlmConfig& config, const GpuInfo& gpu_info,
    CreateGpuModelInfo& create_info,
    std::unique_ptr<LlmTensorLoader> tensor_loader)
    : config_(config),
      gpu_info_(gpu_info),
      create_info_(create_info),
      tensor_loader_(std::move(tensor_loader)),
      model_builder_(gpu_info, create_info.hints, create_info.precision,
                     create_info.storage_type) {
  if (tensor_loader_) {
    tensor_loader_->SetGpuInfo(gpu_info);
    tensor_loader_->SetDataType(
        DeduceDataTypeFromPrecision(create_info.precision));
  }
}

void Gemma3ModelBuilder::CreateKVCache(
    std::vector<GpuModelBuilder::TensorHandle>* k_caches_in_handles,
    std::vector<GpuModelBuilder::TensorHandle>* v_caches_in_handles) {
  DataType cache_dt = DeduceDataTypeFromPrecision(create_info_.precision);
  const int head_dimension = config_.head_dimension;
  const int num_kv_heads = config_.number_of_kv_heads;
  const int kv_heads_count = config_.batch_size * num_kv_heads;
  k_caches_in_handles->resize(config_.stack_size);
  v_caches_in_handles->resize(config_.stack_size);

  for (int i = 0; i < config_.stack_size; ++i) {
    const bool is_global =
        (i + 1) % (config_.num_local_layers_per_global + 1) == 0;
    const int cache_size = GetCacheSize(config_, is_global);
    auto k_desc = GetKCacheTensorDescriptor(cache_dt, cache_size,
                                            kv_heads_count, head_dimension);
    auto v_desc = GetVCacheTensorDescriptor(cache_dt, cache_size,
                                            kv_heads_count, head_dimension);

    (*k_caches_in_handles)[i] = model_builder_.AddTensor(k_desc);
    (*v_caches_in_handles)[i] = model_builder_.AddTensor(v_desc);
  }
}

absl::Status Gemma3ModelBuilder::BuildModelInternal(
    int sequence_size, GpuModel* gpu_model,
    const GpuModelBuilder::TensorHandle& params_i32_handle,
    const std::vector<GpuModelBuilder::TensorHandle>& k_caches_in_handles,
    const std::vector<GpuModelBuilder::TensorHandle>& v_caches_in_handles,
    GpuModelBuilder::TensorHandle* input_handle,
    GpuModelBuilder::TensorHandle* output_handle) {
  BuildContext ctx{model_builder_, create_info_,  tensor_loader_.get(),
                   config_,        sequence_size, params_i32_handle};

  TensorHandle input_ids =
      ctx.AddTensor(config_.batch_size, 1, sequence_size, 1, DataType::kInt32);
  *input_handle = input_ids;

  // 1. Token Embeddings
  EmbeddingState emb_state;
  TensorHandle x = EmbedTokens(ctx, input_ids, &emb_state);

  // 2. Positional and Attention Masks
  TensorHandle position_ids = PositionIds(ctx);
  TensorHandle global_attention_mask = AttentionMask(ctx, /*is_global=*/true);
  TensorHandle local_attention_mask = AttentionMask(ctx, /*is_global=*/false);

  // 3. Transformer Layers Stack
  for (int i = 0; i < config_.stack_size; ++i) {
    const bool is_global =
        (i + 1) % (config_.num_local_layers_per_global + 1) == 0;
    const TensorHandle& attention_mask =
        is_global ? global_attention_mask : local_attention_mask;
    x = TransformerLayer(ctx, x, attention_mask, position_ids,
                         absl::StrCat("model.layers.", i), i,
                         k_caches_in_handles[i], v_caches_in_handles[i]);
  }

  // 4. Final Normalization
  x = RMSNorm(ctx, x, "model.norm");

  // Slice out the last token hidden state before the LM head projection during
  // prefill to avoid projecting all prompt tokens to vocabulary logits.
  if (sequence_size > 1) {
    x = ctx.Slice(x, BHWC(0, 0, sequence_size - 1, 0),
                  BHWC(config_.batch_size, 1, 1, config_.model_dimension));
  }

  // 5. LM Head Output Projection
  auto status_or_logits = LmHead(ctx, x, emb_state);
  ABSL_RETURN_IF_ERROR(status_or_logits.status());
  *output_handle = status_or_logits.value();

  // 6. Compile GPU Model
  std::vector<ValueId> input_ids_vec = {input_handle->id, params_i32_handle.id};
  std::vector<ValueId> output_ids_vec = {output_handle->id};
  for (int i = 0; i < config_.stack_size; ++i) {
    input_ids_vec.push_back(k_caches_in_handles[i].id);
    output_ids_vec.push_back(k_caches_in_handles[i].id);
    input_ids_vec.push_back(v_caches_in_handles[i].id);
    output_ids_vec.push_back(v_caches_in_handles[i].id);
  }

  ABSL_RETURN_IF_ERROR(
      model_builder_.GetGpuModel(input_ids_vec, output_ids_vec, gpu_model));

  // Ensure descriptors for shared external inputs (runtime params and
  // KV caches) are present in the produced GpuModel's tensors map.
  gpu_model->tensors[params_i32_handle.id] = params_i32_handle.tensor_desc;
  for (int i = 0; i < config_.stack_size; ++i) {
    gpu_model->tensors[k_caches_in_handles[i].id] =
        k_caches_in_handles[i].tensor_desc;
    gpu_model->tensors[v_caches_in_handles[i].id] =
        v_caches_in_handles[i].tensor_desc;
  }

  return absl::OkStatus();
}

absl::Status Gemma3ModelBuilder::Build(
    GpuModel* prefill_model, GpuModel* decode_model,
    GpuModelBuilder::TensorHandle* input_handle,
    GpuModelBuilder::TensorHandle* decode_input_handle,
    GpuModelBuilder::TensorHandle* params_i32_handle,
    GpuModelBuilder::TensorHandle* output_handle,
    GpuModelBuilder::TensorHandle* decode_output_handle,
    std::vector<GpuModelBuilder::TensorHandle>* k_caches_in_handles,
    std::vector<GpuModelBuilder::TensorHandle>* v_caches_in_handles) {
  // Create shared KV cache tensors
  CreateKVCache(k_caches_in_handles, v_caches_in_handles);

  TensorDescriptor params_td = TensorDescriptor{
      DataType::kInt32, TensorStorageType::kBuffer, Layout::kHWC};
  params_td.SetBHWCShape(BHWC(1, 1, 1, LlmRuntimeParams::kTotalParamsCount));
  *params_i32_handle = model_builder_.AddTensor(params_td);

  // Build prefill model
  ABSL_RETURN_IF_ERROR(BuildModelInternal(
      config_.sequence_size, prefill_model, *params_i32_handle,
      *k_caches_in_handles, *v_caches_in_handles, input_handle, output_handle));

  // Build decode model
  ABSL_RETURN_IF_ERROR(BuildModelInternal(
      1, decode_model, *params_i32_handle, *k_caches_in_handles,
      *v_caches_in_handles, decode_input_handle, decode_output_handle));

  return absl::OkStatus();
}

absl::Status Gemma3ModelBuilder::BuildPostProcessGreedy(
    GpuModel* gpu_model, GpuModelBuilder::TensorHandle* input_logits_handle,
    GpuModelBuilder::TensorHandle* output_token_handle) {
  BuildContext ctx{
      model_builder_,      create_info_, tensor_loader_.get(), config_, 1,
      *input_logits_handle};
  DataType input_type = ctx.data_type();

  // Input is expected to be (B, 1, 1, Vocab)
  TensorHandle logits = ctx.AddTensor(
      BHWC{config_.batch_size, 1, 1, config_.vocabulary_size}, input_type);
  *input_logits_handle = logits;

  // ArgMax via Reduce across vocabulary dimension
  TensorHandle output_token =
      ctx.Reduce(logits, Reduce::Type::kMaximumIndex, {Axis::kChannels});
  *output_token_handle = output_token;

  return model_builder_.GetGpuModel({input_logits_handle->id},
                                    {output_token_handle->id}, gpu_model);
}

}  // namespace ml_drift
