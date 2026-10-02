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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/functional/bind_front.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/model_hints.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/common/task/profiling_info.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"
#include "ml_drift/samples/llm/gemma3_model_builder.h"
#include "ml_drift/samples/llm/gemma4_model_builder.h"
#include "ml_drift/samples/llm/llm_config.h"
#include "ml_drift/samples/llm/llm_file_tensor_loader.h"
#include "ml_drift/samples/llm/llm_tensor_loader.h"
#include "ml_drift/samples/llm/qwen3_model_builder.h"

#if defined(ML_DRIFT_LLM_PROFILE_METAL) ||                          \
    (defined(__APPLE__) && !defined(ML_DRIFT_LLM_PROFILE_WEBGPU) && \
     !defined(ML_DRIFT_LLM_PROFILE_OPENCL))
#define ML_DRIFT_USE_METAL_BACKEND 1
#import <Metal/Metal.h>

#include "ml_drift/metal/common.h"
#include "ml_drift/metal/environment.h"
#include "ml_drift/metal/inference_context.h"
#include "ml_drift/metal/metal_spatial_tensor.h"
#elif defined(ML_DRIFT_LLM_PROFILE_OPENCL)
#define ML_DRIFT_USE_OPENCL_BACKEND 1
#include "ml_drift/cl/cl_command_queue.h"
#include "ml_drift/cl/environment.h"
#include "ml_drift/cl/inference_context.h"
#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/cl/tensor.h"
#else
#define ML_DRIFT_USE_WEBGPU_BACKEND 1
#include "ml_drift/webgpu/environment.h"
#include "ml_drift/webgpu/inference_context.h"
#include "ml_drift/webgpu/spatial_tensor.h"
#include "ml_drift/webgpu/webgpu_headers.h"
#endif

// NOLINTBEGIN
ABSL_FLAG(std::string, model, "gemma4:12b",
          "Model name to profile: gemma4:12b, gemma3:270m, gemma3:1b, "
          "qwen3:0.6b, qwen3:1.7b, qwen3:8b.");
ABSL_FLAG(std::string, weights_path, "",
          "Optional path to extracted model weights directory. If empty, "
          "synthetic GPU weights are allocated for fast profiling.");
ABSL_FLAG(int, weight_bits, 4,
          "Weight bit-width when using synthetic weights (4, 8, 16, or 32).");
ABSL_FLAG(int, quantization_group_size, 32,
          "Quantization input group size for synthetic int4/int8 weights "
          "(e.g. 32, 64, 128, or -1 for per-channel).");
ABSL_FLAG(bool, include_zero_point, false,
          "Whether synthetic quantized weights include a zero-point tensor.");
ABSL_FLAG(bool, reuse_synthetic_weights, false,
          "Reuse backing GPU buffers for identical synthetic weight tensor "
          "descriptors to reduce GPU memory footprint when profiling large "
          "models on memory-constrained devices.");
ABSL_FLAG(int, num_layers, 0,
          "Override number of transformer layers (stack_size). 0 uses the "
          "model's full layer count (e.g. 48 for gemma4:12b; use 6 for one "
          "full 5-local + 1-global layer cycle).");
ABSL_FLAG(int, prefill_seq_len, 64,
          "Sequence length for the prefill graph (default: 64).");
ABSL_FLAG(int, max_seq_len, 2048,
          "Maximum context / KV cache length (default: 2048).");
ABSL_FLAG(int, token_offset, 128,
          "Simulated token offset in the KV cache during decode profiling "
          "(default: 128).");
ABSL_FLAG(std::string, phase, "all",
          "Which phase(s) to profile: 'all', 'decode', 'prefill', or "
          "'greedy'.");
ABSL_FLAG(bool, use_fp32, false,
          "Force FP32 activations/calculations instead of FP16.");
ABSL_FLAG(bool, print_per_dispatch, false,
          "Print the full per-node dispatch trace from ProfilingInfo in "
          "addition to the aggregated op and LLM component tables.");
ABSL_FLAG(int, end_to_end_iters, 10,
          "Number of end-to-end graph executions to measure wall-clock "
          "latency after per-kernel profiling (0 to skip).");
// NOLINTEND

namespace ml_drift {
namespace {

// Fast synthetic weight loader that constructs exact TensorDescriptors matching
// real weight layouts and allocates GPU buffers/textures directly without
// spending CPU time rearranging billions of dummy elements on the host.
class LlmFastSyntheticTensorLoader : public LlmTensorLoader {
 public:
  LlmFastSyntheticTensorLoader(int weight_bits, int quantization_group_size,
                               bool include_zero_point, ModelType model_type)
      : weight_bits_(weight_bits),
        quantization_group_size_(quantization_group_size),
        include_zero_point_(include_zero_point),
        model_type_(model_type) {}

  std::vector<float> LoadFloat32(const std::string& name, int size) override {
    if (absl::StrContains(name, "layer_scalar")) {
      return std::vector<float>(size, 0.95f);
    }
    return std::vector<float>(size, 0.02f);
  }

  absl::StatusOr<GpuSpatialTensor*> LoadWeights(
      const std::string& name, const WeightsDescription& weights_desc,
      const OHWI& shape, bool swap_dims) override {
    const int elements_count =
        GetTotalElementsCountForLayout(weights_desc, shape);
    TensorDescriptor weights_td(weights_desc.type, TensorStorageType::kBuffer,
                                Layout::kLinear);
    weights_td.SetBHWCShape(BHWC(1, 1, 1, elements_count));
    return create_tensor_fn_(weights_td);
  }

  absl::StatusOr<Int8Weights> LoadInt8Weights(
      const std::string& name, const WeightsDescription& weights_desc,
      const OHWI& shape, bool swap_dims) override {
    TensorDescriptor weights_td = BuildQuantizedWeightsDescriptor(
        weights_desc, shape, /*bits_per_element=*/8);
    Int8Weights int8_weights;
    ABSL_ASSIGN_OR_RETURN(int8_weights.weights, create_tensor_fn_(weights_td));

    TensorDescriptor sum_i_td(DataType::kInt32, TensorStorageType::kBuffer,
                              Layout::kLinear);
    sum_i_td.SetBHWCShape(BHWC(1, 1, 1, shape.o));
    ABSL_ASSIGN_OR_RETURN(int8_weights.weights_sum_i,
                          create_tensor_fn_(sum_i_td));
    return int8_weights;
  }

  absl::StatusOr<Int4Weights> LoadInt4Weights(
      const std::string& name, const WeightsDescription& weights_desc,
      const OHWI& shape, bool swap_dims) override {
    TensorDescriptor weights_td = BuildQuantizedWeightsDescriptor(
        weights_desc, shape, /*bits_per_element=*/4);
    Int4Weights int4_weights;
    ABSL_ASSIGN_OR_RETURN(int4_weights.weights, create_tensor_fn_(weights_td));

    TensorDescriptor sum_i_td(DataType::kInt32, TensorStorageType::kBuffer,
                              Layout::kLinear);
    sum_i_td.SetBHWCShape(BHWC(1, 1, 1, shape.o));
    ABSL_ASSIGN_OR_RETURN(int4_weights.weights_sum_i,
                          create_tensor_fn_(sum_i_td));
    return int4_weights;
  }

  int GetQuantizationIGroupSize(const std::string& weights_name,
                                const OHWI& weights_shape) const override {
    if (quantization_group_size_ > 0 &&
        weights_shape.i % quantization_group_size_ == 0) {
      return quantization_group_size_;
    }
    return weights_shape.i;
  }

  absl::StatusOr<GpuSpatialTensor*> LoadScale(const std::string& tensor_name,
                                              const OHWI& shape) override {
    Tensor<OHWI, DataType::kFloat32> scale;
    scale.shape = shape;
    scale.data.resize(shape.DimensionsProduct(), 0.02f);
    DataType effective_dt =
        data_type_ == DataType::kUnknown ? DataType::kFloat32 : data_type_;
    TensorDescriptor td =
        ScaleOrZeroPointToTensorDesc(gpu_info_, scale, effective_dt);
    return create_tensor_fn_(td);
  }

  absl::StatusOr<GpuSpatialTensor*> LoadZeroPoint(
      const std::string& zero_point_name, const std::string& scale_name,
      const OHWI& shape) override {
    Tensor<OHWI, DataType::kFloat32> zero_point;
    zero_point.shape = shape;
    zero_point.data.resize(shape.DimensionsProduct(), 0.0f);
    DataType effective_dt =
        data_type_ == DataType::kUnknown ? DataType::kFloat32 : data_type_;
    TensorDescriptor td =
        ScaleOrZeroPointToTensorDesc(gpu_info_, zero_point, effective_dt);
    return create_tensor_fn_(td);
  }

  bool HasTensor(const std::string& name) const override {
    if (absl::EndsWith(name, ".weight_quantized_zp")) {
      return include_zero_point_;
    }
    if (name == "lm_head.weight") {
      // Gemma 3 and Gemma 4 tie embed_tokens and lm_head weights; Qwen 3 uses a
      // separate lm_head weight tensor.
      return model_type_ == ModelType::kQwen3;
    }
    return true;
  }

  int GetTensorElementSizeInBits(const std::string& tensor_name,
                                 int count) override {
    return weight_bits_;
  }

 private:
  static TensorDescriptor BuildQuantizedWeightsDescriptor(
      const WeightsDescription& weights_desc, const OHWI& shape,
      int bits_per_element) {
    const int elements_per_byte = 8 / bits_per_element;
    if (weights_desc.IsLinearLayout()) {
      const size_t total_elements =
          GetTotalElementsCountForLayout(weights_desc, shape);
      const size_t num_bytes = DivideRoundUp(total_elements, elements_per_byte);
      TensorDescriptor td(DataType::kUint8, TensorStorageType::kBuffer,
                          Layout::kLinear);
      td.SetBHWDCShape(BHWDC(1, 1, 1, 1, static_cast<int>(num_bytes)));
      return td;
    }

    DataType texture_type =
        bits_per_element == 4 ? DataType::kUint16 : DataType::kUint32;
    TensorDescriptor td(texture_type, TensorStorageType::kTexture2D,
                        Layout::kHW);
    uint2 tex_size = Get2dResourceSize(weights_desc, shape);
    tex_size.x /= 4;
    td.SetBHWDCShape(BHWDC(1, tex_size.y, tex_size.x, 1, 4));
    return td;
  }

  int weight_bits_;
  int quantization_group_size_;
  bool include_zero_point_;
  ModelType model_type_;
};

// Formats a Shape vector (e.g. dispatch.inshape or dispatch.outshape) into a
// compact human-readable string like "[1,1,1,3840]".
std::string FormatShapes(const std::vector<Shape>& shapes, int max_shapes = 2) {
  if (shapes.empty()) {
    return "[]";
  }
  std::string out;
  const int limit = std::min<int>(shapes.size(), max_shapes);
  for (int i = 0; i < limit; ++i) {
    if (i > 0) {
      absl::StrAppend(&out, ",");
    }
    absl::StrAppend(&out, "[");
    for (size_t d = 0; d < shapes[i].dimensions.size(); ++d) {
      if (d > 0) {
        absl::StrAppend(&out, ",");
      }
      absl::StrAppend(&out, shapes[i].dimensions[d]);
    }
    absl::StrAppend(&out, "]");
  }
  if (static_cast<int>(shapes.size()) > limit) {
    absl::StrAppend(&out, ",...");
  }
  return out;
}

struct LlmDispatchClassification {
  std::string category;
  std::string role;
};

// Infers the high-level LLM component category and specific architectural role
// for a GPU dispatch based on its operation label, tensor shapes, and
// LlmConfig.
LlmDispatchClassification ClassifyDispatch(
    const ProfilingInfo::DispatchInfo& dispatch, const LlmConfig& config) {
  const std::string& label = dispatch.label;
  const int d_model = config.model_dimension;
  const int d_ff = config.hidden_dimension;
  const int vocab = config.vocabulary_size;

  const int local_head_dim = config.head_dimension;
  const int global_head_dim =
      config.global_head_dimension.value_or(config.head_dimension);
  const int local_kv_heads = config.number_of_kv_heads;
  const int global_kv_heads =
      config.global_number_of_kv_heads.value_or(config.number_of_kv_heads);

  const int q_local_ch = config.number_of_heads * local_head_dim;
  const int kv_local_ch = local_kv_heads * local_head_dim;
  const int q_global_ch = config.number_of_heads * global_head_dim;
  const int kv_global_ch = global_kv_heads * global_head_dim;

  const int in_h =
      (!dispatch.inshape.empty() && dispatch.inshape[0].dimensions.size() >= 4)
          ? dispatch.inshape[0].dimensions[1]
          : 0;
  const int in_c =
      (!dispatch.inshape.empty() && dispatch.inshape[0].dimensions.size() >= 4)
          ? dispatch.inshape[0].dimensions[3]
          : 0;
  const int out_h = (!dispatch.outshape.empty() &&
                     dispatch.outshape[0].dimensions.size() >= 4)
                        ? dispatch.outshape[0].dimensions[1]
                        : 0;
  const int out_c = (!dispatch.outshape.empty() &&
                     dispatch.outshape[0].dimensions.size() >= 4)
                        ? dispatch.outshape[0].dimensions[3]
                        : 0;

  if (absl::StartsWith(label, "embedding_lookup")) {
    return {"Embeddings & LM Head", "EmbedTokens Lookup"};
  }
  if (absl::StartsWith(label, "strided_slice")) {
    return {"Embeddings & LM Head", "Prefill Last-Token Slice"};
  }
  if (absl::StartsWith(label, "fill_position_ids")) {
    return {"Mask & Position Setup", "PositionIds"};
  }
  if (absl::StartsWith(label, "apply_local_attention_mask")) {
    return {"Mask & Position Setup", "Local AttentionMask Build"};
  }
  if (absl::StartsWith(label, "apply_attention_mask")) {
    return {"Mask & Position Setup", "AttentionMask Build"};
  }
  if (absl::StartsWith(label, "weights_convert")) {
    return {"Weight Dequant / Conversion", label};
  }
  if (absl::StartsWith(label, "QKVRmsNormRoPE")) {
    if (config.global_head_dimension.has_value() &&
        (out_c == global_head_dim || in_c == q_global_ch)) {
      return {"RoPE & KV Cache", "QKVRmsNormRoPE (Global)"};
    }
    return {"RoPE & KV Cache", "QKVRmsNormRoPE (Local)"};
  }
  if (absl::StartsWith(label, "rope") ||
      absl::StartsWith(label, "split_rope_concat")) {
    return {"RoPE & KV Cache", "RoPE"};
  }
  if (absl::StartsWith(label, "extract_local_cache") ||
      absl::StartsWith(label, "cache_update")) {
    return {"RoPE & KV Cache", label};
  }
  if (absl::StartsWith(label, "select_mask")) {
    return {"Attention Core (QK^T, Softmax, SV)", "Attention Mask Select"};
  }
  if (absl::StartsWith(label, "softmax_reduce") ||
      absl::StartsWith(label, "softmax")) {
    return {"Attention Core (QK^T, Softmax, SV)", "Attention Softmax"};
  }
  if (absl::StrContains(label, "+ src softmax")) {
    if (config.global_head_dimension.has_value() && out_c == global_head_dim) {
      return {"Attention Core (QK^T, Softmax, SV)",
              "Attention Score*V (Global)"};
    }
    return {"Attention Core (QK^T, Softmax, SV)", "Attention Score*V (Local)"};
  }
  if (absl::StartsWith(label, "gelu_tanh_mul") ||
      absl::StartsWith(label, "silu_mul")) {
    return {"MLP / FeedForward", "MLP Gated Activation"};
  }
  if (absl::StartsWith(label, "rms_normalization")) {
    return {"RMSNorm & Residuals", "RMSNorm"};
  }
  if (absl::StartsWith(label, "quantize_and_gather")) {
    return {"Linear Projections (Quantize Input)", "Dynamic Input Quantize"};
  }
  if (absl::StartsWith(label, "fc1x1") ||
      absl::StartsWith(label, "fully_connected") ||
      absl::StartsWith(label, "convolution")) {
    if (absl::StrContains(label, "select_mask") ||
        (in_h == config.number_of_heads && out_h == config.number_of_heads)) {
      if (config.global_head_dimension.has_value() && in_c == global_head_dim) {
        return {"Attention Core (QK^T, Softmax, SV)",
                "Attention Q*K^T (Global)"};
      }
      return {"Attention Core (QK^T, Softmax, SV)", "Attention Q*K^T (Local)"};
    }
    if (out_c == vocab) {
      return {"Embeddings & LM Head", "LM Head Projection"};
    }
    if (in_c == d_model && out_c == d_ff) {
      return {"MLP / FeedForward", "MLP gate_proj / up_proj"};
    }
    if (in_c == d_ff && out_c == d_model) {
      return {"MLP / FeedForward", "MLP down_proj"};
    }
    if (in_c == d_model) {
      if (config.global_head_dimension.has_value() && out_c == q_global_ch &&
          q_global_ch != q_local_ch) {
        return {"Attention Projections (QKV/O)", "Attn q_proj (Global)"};
      }
      if (config.global_head_dimension.has_value() && out_c == kv_global_ch &&
          kv_global_ch != kv_local_ch) {
        return {"Attention Projections (QKV/O)", "Attn k_proj (Global, K=V)"};
      }
      if (out_c == q_local_ch && in_h == 1 && out_h == 1) {
        if (q_local_ch == d_model) {
          return {"Attention Projections (QKV/O)", "Attn q_proj / o_proj"};
        }
        return {"Attention Projections (QKV/O)", "Attn q_proj (Local)"};
      }
      if (out_c == kv_local_ch) {
        return {"Attention Projections (QKV/O)",
                "Attn k_proj / v_proj (Local)"};
      }
      return {"Attention Projections (QKV/O)", "Attn QKV/O Projection"};
    }
    if (out_c == d_model) {
      if (config.global_head_dimension.has_value() && in_c == q_global_ch &&
          q_global_ch != q_local_ch) {
        return {"Attention Projections (QKV/O)", "Attn o_proj (Global)"};
      }
      return {"Attention Projections (QKV/O)", "Attn o_proj (Local)"};
    }
    return {"Other Linear / MatMul", label};
  }
  if (absl::StartsWith(label, "reshape") ||
      absl::StartsWith(label, "transpose") ||
      absl::StartsWith(label, "split") || absl::StartsWith(label, "concat") ||
      absl::StartsWith(label, "sub_tensor")) {
    return {"Data Movement (Reshape/Transpose/Slice)", label};
  }
  if (label == "add -> mul") {
    return {"RMSNorm & Residuals", "Residual Add + Layer Scalar"};
  }
  if (label == "add" || absl::StartsWith(label, "add ") ||
      absl::StartsWith(label, "elementwise")) {
    return {"RMSNorm & Residuals", "Residual Add"};
  }
  if (absl::StartsWith(label, "reduce")) {
    return {"Post-Processing", "ArgMax Reduce"};
  }
  return {"Other Operations", label};
}

struct AggregatedGroupStats {
  std::string category;
  std::string role;
  std::string op_name;
  std::string shape_signature;
  int count = 0;
  double total_ms = 0.0;
  uint64_t total_flops = 0;
  uint64_t total_bytes = 0;
};

struct CategorySummaryStats {
  std::string category;
  int count = 0;
  double total_ms = 0.0;
  uint64_t total_flops = 0;
  uint64_t total_bytes = 0;
};

// Prints the LLM-specific breakdown tables (by Role + Op + Shape, and by
// high-level LLM architectural category), sorted by descending GPU time.
void PrintLlmBreakdownReport(absl::string_view phase_name,
                             const LlmConfig& config,
                             const ProfilingInfo& profiling_info,
                             bool print_per_dispatch) {
  const double total_ms =
      absl::ToDoubleMilliseconds(profiling_info.GetTotalTime());
  std::cout << "\n============================================================="
               "===================\n";
  std::cout << absl::StreamFormat(
      "LLM PERFORMANCE BREAKDOWN: %s (layers=%d, seq_len=%d, cache_size=%d)\n",
      phase_name, config.stack_size, config.sequence_size, config.cache_size);
  std::cout << "==============================================================="
               "=================\n";

  if (print_per_dispatch) {
    std::cout << "\n--- Detailed Per-Dispatch Report ---\n";
    std::cout << profiling_info.GetDetailedReport() << "\n";
  } else {
    // Still print the concise per-op-prefix summary from ProfilingInfo.
    std::cout << "\n--- Summary by Low-Level Op Prefix ---\n";
    absl::flat_hash_map<std::string, double> op_prefix_ms;
    absl::flat_hash_map<std::string, int> op_prefix_count;
    for (const auto& d : profiling_info.dispatches) {
      std::string prefix = d.label.substr(0, d.label.find(' '));
      op_prefix_ms[prefix] += absl::ToDoubleMilliseconds(d.duration);
      op_prefix_count[prefix] += 1;
    }
    std::vector<std::pair<std::string, double>> sorted_prefixes(
        op_prefix_ms.begin(), op_prefix_ms.end());
    absl::c_sort(sorted_prefixes, [](const auto& a, const auto& b) {
      return a.second > b.second;
    });
    std::cout << absl::StreamFormat("%-32s %8s %12s %8s\n", "Op Prefix",
                                    "Count", "Total (ms)", "% Time");
    std::cout << std::string(64, '-') << "\n";
    for (const auto& [prefix, ms] : sorted_prefixes) {
      const double pct = total_ms > 0.0 ? (100.0 * ms / total_ms) : 0.0;
      std::cout << absl::StreamFormat("%-32s %8d %12.3f %7.2f%%\n", prefix,
                                      op_prefix_count[prefix], ms, pct);
    }
  }

  // Group dispatches by (Role, Op Label, Input->Output Shape).
  absl::flat_hash_map<std::string, AggregatedGroupStats> groups;
  absl::flat_hash_map<std::string, CategorySummaryStats> categories;

  uint64_t grand_total_flops = 0;
  uint64_t grand_total_bytes = 0;

  for (const auto& d : profiling_info.dispatches) {
    LlmDispatchClassification cls = ClassifyDispatch(d, config);
    std::string shape_sig = absl::StrCat(FormatShapes(d.inshape, 1), " -> ",
                                         FormatShapes(d.outshape, 1));
    std::string key = absl::StrCat(cls.role, "|", d.label, "|", shape_sig);

    const double ms = absl::ToDoubleMilliseconds(d.duration);
    const uint64_t bytes = d.read_mem_size + d.write_mem_size;

    AggregatedGroupStats& g = groups[key];
    if (g.count == 0) {
      g.category = cls.category;
      g.role = cls.role;
      g.op_name = d.label;
      g.shape_signature = shape_sig;
    }
    g.count += 1;
    g.total_ms += ms;
    g.total_flops += d.flops;
    g.total_bytes += bytes;

    CategorySummaryStats& c = categories[cls.category];
    if (c.count == 0) {
      c.category = cls.category;
    }
    c.count += 1;
    c.total_ms += ms;
    c.total_flops += d.flops;
    c.total_bytes += bytes;

    grand_total_flops += d.flops;
    grand_total_bytes += bytes;
  }

  std::vector<AggregatedGroupStats> sorted_groups;
  sorted_groups.reserve(groups.size());
  for (const auto& [_, group] : groups) {
    sorted_groups.push_back(group);
  }
  absl::c_sort(sorted_groups, [](const AggregatedGroupStats& a,
                                 const AggregatedGroupStats& b) {
    return a.total_ms > b.total_ms;
  });

  std::vector<CategorySummaryStats> sorted_categories;
  sorted_categories.reserve(categories.size());
  for (const auto& [_, cat] : categories) {
    sorted_categories.push_back(cat);
  }
  absl::c_sort(sorted_categories, [](const CategorySummaryStats& a,
                                     const CategorySummaryStats& b) {
    return a.total_ms > b.total_ms;
  });

  // Table 1: High-Level LLM Component Category Summary.
  std::cout << "\n--- Table 1: High-Level LLM Component Summary ---\n";
  std::cout << absl::StreamFormat("%-38s %7s %11s %8s %10s %10s\n",
                                  "LLM Component Category", "Count",
                                  "Total (ms)", "% Time", "GB/s", "GFLOP/s");
  std::cout << std::string(90, '-') << "\n";
  for (const CategorySummaryStats& c : sorted_categories) {
    const double pct = total_ms > 0.0 ? (100.0 * c.total_ms / total_ms) : 0.0;
    const double sec = c.total_ms / 1000.0;
    const double gbps =
        sec > 0.0 ? (static_cast<double>(c.total_bytes) / 1e9 / sec) : 0.0;
    const double gflops =
        sec > 0.0 ? (static_cast<double>(c.total_flops) / 1e9 / sec) : 0.0;
    std::cout << absl::StreamFormat("%-38s %7d %11.3f %7.2f%% %10.1f %10.1f\n",
                                    c.category, c.count, c.total_ms, pct, gbps,
                                    gflops);
  }
  std::cout << std::string(90, '-') << "\n";
  const double total_sec = total_ms / 1000.0;
  const double avg_gbps =
      total_sec > 0.0
          ? (static_cast<double>(grand_total_bytes) / 1e9 / total_sec)
          : 0.0;
  const double avg_gflops =
      total_sec > 0.0
          ? (static_cast<double>(grand_total_flops) / 1e9 / total_sec)
          : 0.0;
  std::cout << absl::StreamFormat(
      "%-38s %7d %11.3f %7.2f%% %10.1f %10.1f\n",
      "TOTAL (Ideal Sum of Dispatches)",
      static_cast<int>(profiling_info.dispatches.size()), total_ms, 100.0,
      avg_gbps, avg_gflops);

  // Table 2: Detailed Breakdown by LLM Role + Operation + Shape.
  std::cout << "\n--- Table 2: Breakdown by LLM Role, Operation & Shape ---\n";
  std::cout << absl::StreamFormat(
      "%-28s %-34s %-33s %6s %10s %9s %7s %9s %9s\n", "LLM Role", "Operation",
      "In[0] -> Out[0] (BHWC)", "Count", "Total(ms)", "Avg(ms)", "%Time",
      "GB/s", "GFLOP/s");
  std::cout << std::string(154, '-') << "\n";
  for (const AggregatedGroupStats& g : sorted_groups) {
    const double avg_ms = g.count > 0 ? (g.total_ms / g.count) : 0.0;
    const double pct = total_ms > 0.0 ? (100.0 * g.total_ms / total_ms) : 0.0;
    const double sec = g.total_ms / 1000.0;
    const double gbps =
        sec > 0.0 ? (static_cast<double>(g.total_bytes) / 1e9 / sec) : 0.0;
    const double gflops =
        sec > 0.0 ? (static_cast<double>(g.total_flops) / 1e9 / sec) : 0.0;

    std::string role_col = g.role.substr(0, 27);
    std::string op_col = g.op_name.substr(0, 33);
    std::string shape_col = g.shape_signature.substr(0, 32);
    std::cout << absl::StreamFormat(
        "%-28s %-34s %-33s %6d %10.3f %9.3f %6.2f%% %9.1f %9.1f\n", role_col,
        op_col, shape_col, g.count, g.total_ms, avg_ms, pct, gbps, gflops);
  }
  std::cout << std::string(154, '-') << "\n";
}

// Populates the 4-element int32 runtime parameter array for prefill or decode.
void FillRuntimeParamsData(const LlmConfig& config, int token_offset,
                           int active_tokens, int data_out[4]) {
  const int ch_alignment = ConvRuntimeCheckDesc::kChannelsAlignment;
  data_out[LlmRuntimeParams::kTokenOffsetIndex] = token_offset;
  data_out[LlmRuntimeParams::kActiveTokensIndex] = active_tokens;
  data_out[LlmRuntimeParams::kActiveTokensAlignedIndex] =
      std::min(config.cache_size, AlignByN(active_tokens, ch_alignment));
  data_out[LlmRuntimeParams::kRingOffsetIndex] =
      token_offset + GetRingOffset(config);
}

// Dispatches Build() and BuildPostProcessGreedy() on the selected model
// builder (Gemma4ModelBuilder, Gemma3ModelBuilder, or Qwen3ModelBuilder).
absl::Status BuildLlmGraphs(
    ModelType model_type, const LlmConfig& config, const GpuInfo& gpu_info,
    CreateGpuModelInfo& create_info,
    std::unique_ptr<LlmTensorLoader> tensor_loader, GpuModel* prefill_model,
    GpuModel* decode_model, GpuModel* greedy_model,
    GpuModelBuilder::TensorHandle* input_handle,
    GpuModelBuilder::TensorHandle* decode_input_handle,
    GpuModelBuilder::TensorHandle* params_i32_handle,
    GpuModelBuilder::TensorHandle* output_logits_handle,
    GpuModelBuilder::TensorHandle* decode_output_logits_handle,
    std::vector<GpuModelBuilder::TensorHandle>* k_caches_in_handles,
    std::vector<GpuModelBuilder::TensorHandle>* v_caches_in_handles,
    GpuModelBuilder::TensorHandle* greedy_input_logits_handle,
    GpuModelBuilder::TensorHandle* greedy_output_token_handle) {
  switch (model_type) {
    case ModelType::kGemma4: {
      Gemma4ModelBuilder builder(config, gpu_info, create_info,
                                 std::move(tensor_loader));
      ABSL_RETURN_IF_ERROR(builder.Build(
          prefill_model, decode_model, input_handle, decode_input_handle,
          params_i32_handle, output_logits_handle, decode_output_logits_handle,
          k_caches_in_handles, v_caches_in_handles));
      Gemma4ModelBuilder greedy_builder(config, gpu_info, create_info, nullptr);
      ABSL_RETURN_IF_ERROR(greedy_builder.BuildPostProcessGreedy(
          greedy_model, greedy_input_logits_handle,
          greedy_output_token_handle));
      return absl::OkStatus();
    }
    case ModelType::kGemma3: {
      Gemma3ModelBuilder builder(config, gpu_info, create_info,
                                 std::move(tensor_loader));
      ABSL_RETURN_IF_ERROR(builder.Build(
          prefill_model, decode_model, input_handle, decode_input_handle,
          params_i32_handle, output_logits_handle, decode_output_logits_handle,
          k_caches_in_handles, v_caches_in_handles));
      Gemma3ModelBuilder greedy_builder(config, gpu_info, create_info, nullptr);
      ABSL_RETURN_IF_ERROR(greedy_builder.BuildPostProcessGreedy(
          greedy_model, greedy_input_logits_handle,
          greedy_output_token_handle));
      return absl::OkStatus();
    }
    case ModelType::kQwen3: {
      Qwen3ModelBuilder builder(config, gpu_info, create_info,
                                std::move(tensor_loader));
      ABSL_RETURN_IF_ERROR(builder.Build(
          prefill_model, decode_model, input_handle, decode_input_handle,
          params_i32_handle, output_logits_handle, decode_output_logits_handle,
          k_caches_in_handles, v_caches_in_handles));
      Qwen3ModelBuilder greedy_builder(config, gpu_info, create_info, nullptr);
      ABSL_RETURN_IF_ERROR(greedy_builder.BuildPostProcessGreedy(
          greedy_model, greedy_input_logits_handle,
          greedy_output_token_handle));
      return absl::OkStatus();
    }
  }
  return absl::InvalidArgumentError("Unsupported ModelType");
}

// Computes a descriptor signature string for optional synthetic weight buffer
// reuse across identical layers.
std::string GetDescriptorReuseKey(const TensorDescriptor& desc) {
  const BHWDC shape = desc.GetBHWDCShape();
  return absl::StrCat(ToString(desc.GetDataType()), ":",
                      ToString(desc.GetStorageType()), ":", shape.b, ",",
                      shape.h, ",", shape.w, ",", shape.d, ",", shape.c);
}

#if defined(ML_DRIFT_USE_METAL_BACKEND)

absl::StatusOr<GpuSpatialTensor*> CreateMetalWeightTensor(
    metal::Environment* env, bool reuse_tensors,
    std::vector<std::unique_ptr<metal::MetalSpatialTensor>>* model_tensors,
    absl::flat_hash_map<std::string, GpuSpatialTensor*>* reuse_map,
    const TensorDescriptor& tensor_desc) {
  if (reuse_tensors && tensor_desc.GetData().empty()) {
    std::string key = GetDescriptorReuseKey(tensor_desc);
    auto it = reuse_map->find(key);
    if (it != reuse_map->end()) {
      return it->second;
    }
    metal::MetalSpatialTensor& tensor = *model_tensors->emplace_back(
        std::make_unique<metal::MetalSpatialTensor>());
    ABSL_RETURN_IF_ERROR(
        tensor.CreateFromDescriptor(tensor_desc, env->device()));
    (*reuse_map)[key] = &tensor;
    return &tensor;
  }
  metal::MetalSpatialTensor& tensor = *model_tensors->emplace_back(
      std::make_unique<metal::MetalSpatialTensor>());
  ABSL_RETURN_IF_ERROR(tensor.CreateFromDescriptor(tensor_desc, env->device()));
  return &tensor;
}

#elif defined(ML_DRIFT_USE_OPENCL_BACKEND)

absl::StatusOr<GpuSpatialTensor*> CreateOpenClWeightTensor(
    cl::Environment* env, bool reuse_tensors,
    std::vector<std::unique_ptr<cl::Tensor>>* model_tensors,
    absl::flat_hash_map<std::string, GpuSpatialTensor*>* reuse_map,
    const TensorDescriptor& tensor_desc) {
  if (reuse_tensors && tensor_desc.GetData().empty()) {
    std::string key = GetDescriptorReuseKey(tensor_desc);
    auto it = reuse_map->find(key);
    if (it != reuse_map->end()) {
      return it->second;
    }
    cl::Tensor& tensor =
        *model_tensors->emplace_back(std::make_unique<cl::Tensor>());
    ABSL_RETURN_IF_ERROR(
        tensor.CreateFromDescriptor(tensor_desc, env->context()));
    (*reuse_map)[key] = &tensor;
    return &tensor;
  }
  cl::Tensor& tensor =
      *model_tensors->emplace_back(std::make_unique<cl::Tensor>());
  ABSL_RETURN_IF_ERROR(
      tensor.CreateFromDescriptor(tensor_desc, env->context()));
  ABSL_RETURN_IF_ERROR(env->queue()->WaitForCompletion());
  return &tensor;
}

#else  // ML_DRIFT_USE_WEBGPU_BACKEND

absl::StatusOr<GpuSpatialTensor*> CreateWebGpuWeightTensor(
    webgpu::Environment* env, bool reuse_tensors,
    std::vector<std::unique_ptr<webgpu::SpatialTensor>>* model_tensors,
    absl::flat_hash_map<std::string, GpuSpatialTensor*>* reuse_map,
    const TensorDescriptor& tensor_desc) {
  if (reuse_tensors && tensor_desc.GetData().empty()) {
    std::string key = GetDescriptorReuseKey(tensor_desc);
    auto it = reuse_map->find(key);
    if (it != reuse_map->end()) {
      return it->second;
    }
    webgpu::SpatialTensor& tensor =
        *model_tensors->emplace_back(std::make_unique<webgpu::SpatialTensor>());
    ABSL_RETURN_IF_ERROR(
        tensor.CreateFromDescriptor(env->device(), tensor_desc));
    (*reuse_map)[key] = &tensor;
    return &tensor;
  }
  webgpu::SpatialTensor& tensor =
      *model_tensors->emplace_back(std::make_unique<webgpu::SpatialTensor>());
  ABSL_RETURN_IF_ERROR(tensor.CreateFromDescriptor(env->device(), tensor_desc));
  return &tensor;
}

#endif

absl::Status RunLlmPerformanceProfiling() {
  const std::string model_name = absl::GetFlag(FLAGS_model);
  std::string weights_path = absl::GetFlag(FLAGS_weights_path);
  if (!weights_path.empty() && weights_path.back() != '/') {
    weights_path.push_back('/');
  }
  const int weight_bits = absl::GetFlag(FLAGS_weight_bits);
  const int quant_group_size = absl::GetFlag(FLAGS_quantization_group_size);
  const bool include_zero_point = absl::GetFlag(FLAGS_include_zero_point);
  const bool reuse_synthetic = absl::GetFlag(FLAGS_reuse_synthetic_weights);
  const int num_layers_override = absl::GetFlag(FLAGS_num_layers);
  const int prefill_seq_len = absl::GetFlag(FLAGS_prefill_seq_len);
  const int max_seq_len = absl::GetFlag(FLAGS_max_seq_len);
  const int token_offset = absl::GetFlag(FLAGS_token_offset);
  const std::string phase = absl::GetFlag(FLAGS_phase);
  const bool use_fp32_flag = absl::GetFlag(FLAGS_use_fp32);
  const bool print_per_dispatch = absl::GetFlag(FLAGS_print_per_dispatch);
  const int end_to_end_iters = absl::GetFlag(FLAGS_end_to_end_iters);

  if (weight_bits != 4 && weight_bits != 8 && weight_bits != 16 &&
      weight_bits != 32) {
    return absl::InvalidArgumentError(absl::StrCat(
        "--weight_bits must be 4, 8, 16, or 32; got ", weight_bits));
  }

  absl::StatusOr<ModelInfo> model_info = GetModelInfo(model_name);
  if (!model_info.ok()) {
    return model_info.status();
  }
  LlmConfig config = model_info->config;
  const ModelType model_type = model_info->type;
  const bool use_fp32 = use_fp32_flag || model_info->force_fp32;

  constexpr int kMaxPrefillChunkSize = 1024;
  config.sequence_size = std::min(prefill_seq_len, kMaxPrefillChunkSize);
  config.cache_size = max_seq_len;
  if (num_layers_override > 0) {
    config.stack_size = num_layers_override;
  }

  const bool run_prefill = (phase == "all" || phase == "prefill");
  const bool run_decode = (phase == "all" || phase == "decode");
  const bool run_greedy = (phase == "all" || phase == "greedy");
  if (!run_prefill && !run_decode && !run_greedy) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid --phase='", phase,
                     "'. Expected 'all', 'decode', 'prefill', "
                     "or 'greedy'."));
  }

#if defined(ML_DRIFT_USE_METAL_BACKEND)
  auto env = std::make_unique<metal::Environment>();
  id<MTLCommandQueue> command_queue = [env->device() newCommandQueue];
  const GpuInfo& gpu_info = env->GetInfo();
  const absl::string_view backend_name = "Metal";
  const std::string gpu_device_name =
      env->device().name ? [env->device().name UTF8String] : "Unknown";
  std::vector<std::unique_ptr<metal::MetalSpatialTensor>> model_tensors;
#elif defined(ML_DRIFT_USE_OPENCL_BACKEND)
  ABSL_RETURN_IF_ERROR(cl::LoadOpenCL());
  auto env = std::make_unique<cl::Environment>();
  ABSL_RETURN_IF_ERROR(cl::CreateEnvironment(env.get()));
  const GpuInfo& gpu_info = env->GetDevicePtr()->GetInfo();
  const absl::string_view backend_name = "OpenCL";
  const std::string gpu_device_name = gpu_info.opencl_info.device_name;
  std::vector<std::unique_ptr<cl::Tensor>> model_tensors;
#else
  auto env = std::make_unique<webgpu::Environment>();
  env->RequestExtension("timestamp_query");
  env->RequestExtension("shader_f16");
  ABSL_RETURN_IF_ERROR(env->Initialize());
  const GpuInfo& gpu_info = env->GetInfo();
  const absl::string_view backend_name = "WebGPU";
  wgpu::AdapterInfo adapter_info = {};
  env->device().GetAdapterInfo(&adapter_info);
  const std::string gpu_device_name =
      adapter_info.device.length > 0
          ? std::string(absl::string_view(adapter_info.device.data,
                                          adapter_info.device.length))
          : env->GetPlatformDescription();
  std::vector<std::unique_ptr<webgpu::SpatialTensor>> model_tensors;
#endif

  std::cout << "==============================================================="
               "=================\n";
  std::cout << "ML Drift LLM Performance Profiler\n";
  std::cout << "  Backend        : " << backend_name << "\n";
  std::cout << "  GPU Device     : " << gpu_device_name << "\n";
  std::cout << "  Model          : " << model_name
            << " (layers=" << config.stack_size
            << ", d_model=" << config.model_dimension
            << ", d_ff=" << config.hidden_dimension
            << ", vocab=" << config.vocabulary_size << ")\n";
  std::cout << "  Precision      : " << (use_fp32 ? "FP32" : "FP16") << "\n";
  if (weights_path.empty()) {
    std::cout << "  Weights        : Synthetic (int" << weight_bits
              << ", group_size=" << quant_group_size
              << ", reuse_buffers=" << (reuse_synthetic ? "true" : "false")
              << ")\n";
  } else {
    std::cout << "  Weights        : File (" << weights_path << ")\n";
  }
  std::cout << "  Prefill SeqLen : " << config.sequence_size << "\n";
  std::cout << "  Max KV Cache   : " << config.cache_size
            << " (decode token_offset=" << token_offset << ")\n";
  std::cout << "==============================================================="
               "=================\n";

  CreateGpuModelInfo create_info;
  create_info.precision =
      use_fp32 ? CalculationsPrecision::kF32 : CalculationsPrecision::kF16;
#if defined(ML_DRIFT_USE_METAL_BACKEND)
  create_info.storage_type = metal::GetFastestStorageType(gpu_info);
#elif defined(ML_DRIFT_USE_OPENCL_BACKEND)
  create_info.storage_type = cl::GetFastestStorageType(gpu_info);
#else
  create_info.storage_type = webgpu::GetFastestStorageType(gpu_info);
#endif
  create_info.hints.Add(ModelHints::kFastTuning);
  create_info.hints.Add(ModelHints::kDisallow8bitConvs);
  create_info.hints.Add(ModelHints::kEnableHostMappedPointer);

  std::unique_ptr<LlmTensorLoader> base_loader;
  if (weights_path.empty()) {
    base_loader = std::make_unique<LlmFastSyntheticTensorLoader>(
        weight_bits, quant_group_size, include_zero_point, model_type);
  } else {
    base_loader = std::make_unique<LlmFileTensorLoader>(weights_path);
  }
  std::unique_ptr<LlmTensorLoader> tensor_loader =
      LlmTensorLoader::MakeCaching(std::move(base_loader));

  absl::flat_hash_map<std::string, GpuSpatialTensor*> reuse_map;
#if defined(ML_DRIFT_USE_METAL_BACKEND)
  tensor_loader->SetCreateTensorFn(absl::bind_front(
      &CreateMetalWeightTensor, env.get(),
      weights_path.empty() && reuse_synthetic, &model_tensors, &reuse_map));
#elif defined(ML_DRIFT_USE_OPENCL_BACKEND)
  tensor_loader->SetCreateTensorFn(absl::bind_front(
      &CreateOpenClWeightTensor, env.get(),
      weights_path.empty() && reuse_synthetic, &model_tensors, &reuse_map));
#else
  tensor_loader->SetCreateTensorFn(absl::bind_front(
      &CreateWebGpuWeightTensor, env.get(),
      weights_path.empty() && reuse_synthetic, &model_tensors, &reuse_map));
#endif

  const absl::Time build_start = absl::Now();
  GpuModel prefill_model;
  GpuModel decode_model;
  GpuModel greedy_model;
  GpuModelBuilder::TensorHandle input_handle;
  GpuModelBuilder::TensorHandle decode_input_handle;
  GpuModelBuilder::TensorHandle params_i32_handle;
  GpuModelBuilder::TensorHandle output_logits_handle;
  GpuModelBuilder::TensorHandle decode_output_logits_handle;
  std::vector<GpuModelBuilder::TensorHandle> k_caches_in_handles;
  std::vector<GpuModelBuilder::TensorHandle> v_caches_in_handles;
  GpuModelBuilder::TensorHandle greedy_input_logits_handle;
  GpuModelBuilder::TensorHandle greedy_output_token_handle;

  ABSL_RETURN_IF_ERROR(BuildLlmGraphs(
      model_type, config, gpu_info, create_info, std::move(tensor_loader),
      &prefill_model, &decode_model, &greedy_model, &input_handle,
      &decode_input_handle, &params_i32_handle, &output_logits_handle,
      &decode_output_logits_handle, &k_caches_in_handles, &v_caches_in_handles,
      &greedy_input_logits_handle, &greedy_output_token_handle));

  uint64_t external_weights_bytes = 0;
  for (const auto& t : model_tensors) {
    external_weights_bytes += t->GetMemorySizeInBytes();
  }
  std::cout << absl::StreamFormat(
      "Built GpuModels in %.1f ms (prefill nodes=%d, decode nodes=%d, "
      "allocated weight tensors=%d [%.2f MB])\n",
      absl::ToDoubleMilliseconds(absl::Now() - build_start),
      static_cast<int>(prefill_model.nodes.size()),
      static_cast<int>(decode_model.nodes.size()),
      static_cast<int>(model_tensors.size()),
      static_cast<double>(external_weights_bytes) / (1024.0 * 1024.0));

  // Allocate I/O and KV cache tensors on the GPU.
#if defined(ML_DRIFT_USE_METAL_BACKEND)
  using BackendTensor = metal::MetalSpatialTensor;
  using BackendContext = metal::InferenceContext;
  auto alloc_tensor = [&](const TensorDescriptor& desc,
                          std::unique_ptr<BackendTensor>* out) -> absl::Status {
    *out = std::make_unique<BackendTensor>();
    return (*out)->CreateFromDescriptor(desc, env->device());
  };
  auto get_intermediate_bytes = [](const BackendContext& ctx) -> uint64_t {
    return ctx.GetIntermediateTensorsSize();
  };
  auto get_const_bytes = [](const BackendContext& ctx) -> uint64_t {
    return ctx.GetConstantTensorsSize();
  };
#elif defined(ML_DRIFT_USE_OPENCL_BACKEND)
  using BackendTensor = cl::Tensor;
  using BackendContext = cl::InferenceContext;
  auto alloc_tensor = [&](const TensorDescriptor& desc,
                          std::unique_ptr<BackendTensor>* out) -> absl::Status {
    *out = std::make_unique<BackendTensor>();
    return (*out)->CreateFromDescriptor(desc, env->context());
  };
  auto get_intermediate_bytes = [](const BackendContext& ctx) -> uint64_t {
    return ctx.GetSizeOfMemoryAllocatedForIntermediateTensors();
  };
  auto get_const_bytes = [](const BackendContext& ctx) -> uint64_t {
    return ctx.GetConstantTensorsSize();
  };
#else
  using BackendTensor = webgpu::SpatialTensor;
  using BackendContext = webgpu::InferenceContext;
  auto alloc_tensor = [&](const TensorDescriptor& desc,
                          std::unique_ptr<BackendTensor>* out) -> absl::Status {
    *out = std::make_unique<BackendTensor>();
    return (*out)->CreateFromDescriptor(env->device(), desc);
  };
  auto get_intermediate_bytes = [](const BackendContext& ctx) -> uint64_t {
    return ctx.GetSizeOfMemoryAllocatedForIntermediateTensors();
  };
  auto get_const_bytes = [](const BackendContext& ctx) -> uint64_t {
    return ctx.GetSizeOfMemoryAllocatedForConstTensors();
  };
#endif

  std::unique_ptr<BackendTensor> input_tokens;
  std::unique_ptr<BackendTensor> decode_input_tokens;
  std::unique_ptr<BackendTensor> params_i32_tensor;
  std::unique_ptr<BackendTensor> output_logits;
  std::unique_ptr<BackendTensor> decode_output_logits;
  ABSL_RETURN_IF_ERROR(alloc_tensor(input_handle.tensor_desc, &input_tokens));
  ABSL_RETURN_IF_ERROR(
      alloc_tensor(decode_input_handle.tensor_desc, &decode_input_tokens));
  ABSL_RETURN_IF_ERROR(
      alloc_tensor(params_i32_handle.tensor_desc, &params_i32_tensor));
  ABSL_RETURN_IF_ERROR(
      alloc_tensor(output_logits_handle.tensor_desc, &output_logits));
  ABSL_RETURN_IF_ERROR(alloc_tensor(decode_output_logits_handle.tensor_desc,
                                    &decode_output_logits));

  const int num_kv = k_caches_in_handles.size();
  std::vector<std::unique_ptr<BackendTensor>> kv_cache_tensors(num_kv * 2);
  uint64_t kv_cache_bytes = 0;
  for (int i = 0; i < num_kv; ++i) {
    ABSL_RETURN_IF_ERROR(
        alloc_tensor(k_caches_in_handles[i].tensor_desc, &kv_cache_tensors[i]));
    ABSL_RETURN_IF_ERROR(alloc_tensor(v_caches_in_handles[i].tensor_desc,
                                      &kv_cache_tensors[i + num_kv]));
    kv_cache_bytes += kv_cache_tensors[i]->GetMemorySizeInBytes() +
                      kv_cache_tensors[i + num_kv]->GetMemorySizeInBytes();
  }
  std::cout << absl::StreamFormat(
      "Allocated KV Cache (%d K + %d V tensors): %.2f MB\n", num_kv, num_kv,
      static_cast<double>(kv_cache_bytes) / (1024.0 * 1024.0));

  auto write_params = [&](int offset, int active) -> absl::Status {
    int params_data[LlmRuntimeParams::kTotalParamsCount] = {0};
    FillRuntimeParamsData(config, offset, active, params_data);
#if defined(ML_DRIFT_USE_METAL_BACKEND)
    return params_i32_tensor->WriteData(command_queue, params_data,
                                        /*wait_for_completion=*/true);
#elif defined(ML_DRIFT_USE_OPENCL_BACKEND)
    return env->queue()->EnqueueWriteBuffer(params_i32_tensor->GetMemoryPtr(),
                                            sizeof(params_data), params_data,
                                            /*async=*/false);
#else
    return params_i32_tensor->WriteData(env->queue(), params_data);
#endif
  };

  // 1. Profile Prefill Phase
  if (run_prefill) {
    CreateGpuModelInfo prefill_create_info = create_info;
    prefill_create_info.external_immutable_tensors[input_handle.id] =
        input_tokens.get();
    prefill_create_info.external_immutable_tensors[params_i32_handle.id] =
        params_i32_tensor.get();
    prefill_create_info.external_immutable_tensors[output_logits_handle.id] =
        output_logits.get();
    for (int i = 0; i < num_kv; ++i) {
      prefill_create_info
          .external_immutable_tensors[k_caches_in_handles[i].id] =
          kv_cache_tensors[i].get();
      prefill_create_info
          .external_immutable_tensors[v_caches_in_handles[i].id] =
          kv_cache_tensors[i + num_kv].get();
    }

    const absl::Time init_start = absl::Now();
    BackendContext prefill_context;
#if defined(ML_DRIFT_USE_METAL_BACKEND)
    ABSL_RETURN_IF_ERROR(prefill_context.InitFromGpuModel(
        prefill_create_info, &prefill_model, env.get(), nullptr));
#elif defined(ML_DRIFT_USE_OPENCL_BACKEND)
    ABSL_RETURN_IF_ERROR(prefill_context.InitFromGpuModel(
        prefill_create_info, &prefill_model, env.get(), nullptr));
#else
    ABSL_RETURN_IF_ERROR(prefill_context.InitFromGpuModel(
        *env, prefill_create_info, &prefill_model));
#endif
    std::cout << absl::StreamFormat(
        "\n[Prefill] Compiled InferenceContext in %.1f ms (intermediate "
        "tensors: %.2f MB, const tensors: %.2f MB)\n",
        absl::ToDoubleMilliseconds(absl::Now() - init_start),
        static_cast<double>(get_intermediate_bytes(prefill_context)) /
            (1024.0 * 1024.0),
        static_cast<double>(get_const_bytes(prefill_context)) /
            (1024.0 * 1024.0));

    ABSL_RETURN_IF_ERROR(
        write_params(/*offset=*/0, /*active=*/config.sequence_size));

    ProfilingInfo prefill_profile;
#if defined(ML_DRIFT_USE_METAL_BACKEND)
    prefill_context.Profile(env->device(), &prefill_profile);
#elif defined(ML_DRIFT_USE_OPENCL_BACKEND)
    ABSL_RETURN_IF_ERROR(
        prefill_context.Profile(env->profiling_queue(), &prefill_profile));
#else
    ABSL_RETURN_IF_ERROR(prefill_context.Profile(*env, &prefill_profile));
#endif

    LlmConfig prefill_cfg = config;
    PrintLlmBreakdownReport("PREFILL", prefill_cfg, prefill_profile,
                            print_per_dispatch);

    if (end_to_end_iters > 0) {
      const absl::Time e2e_start = absl::Now();
      for (int iter = 0; iter < end_to_end_iters; ++iter) {
#if defined(ML_DRIFT_USE_METAL_BACKEND)
        @autoreleasepool {
          id<MTLCommandBuffer> cb = [command_queue commandBuffer];
          prefill_context.EncodeWithCommandBuffer(cb, /*tasks_per_encoder=*/8);
          [cb commit];
          [cb waitUntilCompleted];
        }
#elif defined(ML_DRIFT_USE_OPENCL_BACKEND)
        ABSL_RETURN_IF_ERROR(prefill_context.AddToQueue(env->queue()));
        ABSL_RETURN_IF_ERROR(env->queue()->WaitForCompletion());
#else
        ABSL_RETURN_IF_ERROR(prefill_context.Execute(*env));
#endif
      }
      const double avg_e2e_ms =
          absl::ToDoubleMilliseconds(absl::Now() - e2e_start) /
          end_to_end_iters;
      const double toks_per_sec =
          avg_e2e_ms > 0.0 ? (1000.0 * config.sequence_size / avg_e2e_ms) : 0.0;
      std::cout << absl::StreamFormat(
          "[Prefill End-to-End] %d iterations: %.3f ms/chunk (%.1f tokens/sec "
          "at seq_len=%d)\n",
          end_to_end_iters, avg_e2e_ms, toks_per_sec, config.sequence_size);
    }
  }

  // 2. Profile Decode Phase
  if (run_decode) {
    CreateGpuModelInfo decode_create_info = create_info;
    decode_create_info.external_immutable_tensors[decode_input_handle.id] =
        decode_input_tokens.get();
    decode_create_info.external_immutable_tensors[params_i32_handle.id] =
        params_i32_tensor.get();
    decode_create_info
        .external_immutable_tensors[decode_output_logits_handle.id] =
        decode_output_logits.get();
    for (int i = 0; i < num_kv; ++i) {
      decode_create_info.external_immutable_tensors[k_caches_in_handles[i].id] =
          kv_cache_tensors[i].get();
      decode_create_info.external_immutable_tensors[v_caches_in_handles[i].id] =
          kv_cache_tensors[i + num_kv].get();
    }

    const absl::Time init_start = absl::Now();
    BackendContext decode_context;
#if defined(ML_DRIFT_USE_METAL_BACKEND)
    ABSL_RETURN_IF_ERROR(decode_context.InitFromGpuModel(
        decode_create_info, &decode_model, env.get(), nullptr));
#elif defined(ML_DRIFT_USE_OPENCL_BACKEND)
    ABSL_RETURN_IF_ERROR(decode_context.InitFromGpuModel(
        decode_create_info, &decode_model, env.get(), nullptr));
#else
    ABSL_RETURN_IF_ERROR(decode_context.InitFromGpuModel(
        *env, decode_create_info, &decode_model));
#endif
    std::cout << absl::StreamFormat(
        "\n[Decode] Compiled InferenceContext in %.1f ms (intermediate "
        "tensors: %.2f MB, const tensors: %.2f MB)\n",
        absl::ToDoubleMilliseconds(absl::Now() - init_start),
        static_cast<double>(get_intermediate_bytes(decode_context)) /
            (1024.0 * 1024.0),
        static_cast<double>(get_const_bytes(decode_context)) /
            (1024.0 * 1024.0));

    const int clamped_offset =
        std::clamp(token_offset, 0, config.cache_size - 1);
    ABSL_RETURN_IF_ERROR(
        write_params(/*offset=*/clamped_offset, /*active=*/clamped_offset + 1));

    ProfilingInfo decode_profile;
#if defined(ML_DRIFT_USE_METAL_BACKEND)
    decode_context.Profile(env->device(), &decode_profile);
#elif defined(ML_DRIFT_USE_OPENCL_BACKEND)
    ABSL_RETURN_IF_ERROR(
        decode_context.Profile(env->profiling_queue(), &decode_profile));
#else
    ABSL_RETURN_IF_ERROR(decode_context.Profile(*env, &decode_profile));
#endif

    LlmConfig decode_cfg = config;
    decode_cfg.sequence_size = 1;
    PrintLlmBreakdownReport("DECODE", decode_cfg, decode_profile,
                            print_per_dispatch);

    if (end_to_end_iters > 0) {
      const absl::Time e2e_start = absl::Now();
      for (int iter = 0; iter < end_to_end_iters; ++iter) {
#if defined(ML_DRIFT_USE_METAL_BACKEND)
        @autoreleasepool {
          id<MTLCommandBuffer> cb = [command_queue commandBuffer];
          id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
          decode_context.EncodeWithEncoder(encoder);
          [encoder endEncoding];
          [cb commit];
          [cb waitUntilCompleted];
        }
#elif defined(ML_DRIFT_USE_OPENCL_BACKEND)
        ABSL_RETURN_IF_ERROR(decode_context.AddToQueue(env->queue()));
        ABSL_RETURN_IF_ERROR(env->queue()->WaitForCompletion());
#else
        ABSL_RETURN_IF_ERROR(decode_context.Execute(*env));
#endif
      }
      const double avg_e2e_ms =
          absl::ToDoubleMilliseconds(absl::Now() - e2e_start) /
          end_to_end_iters;
      const double toks_per_sec =
          avg_e2e_ms > 0.0 ? (1000.0 / avg_e2e_ms) : 0.0;
      std::cout << absl::StreamFormat(
          "[Decode End-to-End] %d iterations: %.3f ms/token (%.1f tokens/sec "
          "at token_offset=%d)\n",
          end_to_end_iters, avg_e2e_ms, toks_per_sec, clamped_offset);
    }
  }

  // 3. Profile Greedy Post-Process Phase
  if (run_greedy) {
    CreateGpuModelInfo greedy_create_info = create_info;
    greedy_create_info.external_mutable_tensors.clear();
    greedy_create_info.external_immutable_tensors.clear();
    greedy_create_info
        .external_immutable_tensors[greedy_input_logits_handle.id] =
        decode_output_logits.get();
    greedy_create_info
        .external_immutable_tensors[greedy_output_token_handle.id] =
        decode_input_tokens.get();

    BackendContext greedy_context;
#if defined(ML_DRIFT_USE_METAL_BACKEND)
    ABSL_RETURN_IF_ERROR(greedy_context.InitFromGpuModel(
        greedy_create_info, &greedy_model, env.get(), nullptr));
#elif defined(ML_DRIFT_USE_OPENCL_BACKEND)
    ABSL_RETURN_IF_ERROR(greedy_context.InitFromGpuModel(
        greedy_create_info, &greedy_model, env.get(), nullptr));
#else
    ABSL_RETURN_IF_ERROR(greedy_context.InitFromGpuModel(
        *env, greedy_create_info, &greedy_model));
#endif

    ProfilingInfo greedy_profile;
#if defined(ML_DRIFT_USE_METAL_BACKEND)
    greedy_context.Profile(env->device(), &greedy_profile);
#elif defined(ML_DRIFT_USE_OPENCL_BACKEND)
    ABSL_RETURN_IF_ERROR(
        greedy_context.Profile(env->profiling_queue(), &greedy_profile));
#else
    ABSL_RETURN_IF_ERROR(greedy_context.Profile(*env, &greedy_profile));
#endif

    LlmConfig greedy_cfg = config;
    greedy_cfg.sequence_size = 1;
    PrintLlmBreakdownReport("GREEDY POST-PROCESS (ArgMax)", greedy_cfg,
                            greedy_profile, print_per_dispatch);
  }

  return absl::OkStatus();
}

}  // namespace
}  // namespace ml_drift

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::Status status = ml_drift::RunLlmPerformanceProfiling();
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Profiling failed: " << status;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
