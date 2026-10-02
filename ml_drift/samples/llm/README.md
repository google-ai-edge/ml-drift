# ML Drift LLM Samples & Performance Profiler

This folder contains sample runners, model builders, weight extraction scripts,
and an op-level GPU performance profiler for running Large Language Models
(LLMs) using ML Drift across **Metal**, **WebGPU**, and **OpenCL**.

**Supported Models**: `gemma4:12b`, `gemma3:1b`, `gemma3:270m`, `qwen3:0.6b`,
`qwen3:1.7b`, `qwen3:8b`.

### Contents

*   **`llm_config.h`**: C++ header defining `LlmConfig` and model presets.
*   **`gemma4_model_builder.h/.cc`**: Builds Gemma 4 prefill, decode, and greedy
    graphs using ML Drift `GpuModelBuilder`.
*   **`gemma3_model_builder.h/.cc`**: Builds Gemma 3 prefill, decode, and greedy
    graphs using ML Drift `GpuModelBuilder`.
*   **`qwen3_model_builder.h/.cc`**: Builds Qwen 3 prefill, decode, and greedy
    graphs using ML Drift `GpuModelBuilder`.
*   **`llm_runner.h/.cc`**: Shared runner interface, tokenizer wrapper, CLI
    flags, and generation loop.
*   **`llm_metal.cc` / `llm_webgpu.cc` / `llm_opencl.cc`**: Backend-specific
    implementations of `LlmModelRunner` for Metal, WebGPU, and OpenCL.
*   **`llm_performance_profiling.cc`**: Op-level and LLM-component GPU
    performance profiler across Metal, WebGPU, and OpenCL (supports both fast
    synthetic weights and extracted weights).
*   **`llm_tensor_loader.h` / `llm_file_tensor_loader.h/.cc` /
    `llm_dummy_tensor_loader.h/.cc`**: Tensor loader interfaces and
    implementations for loading binary weight files or dummy/synthetic tensors.
*   **`llm_weights_loader.h/.cc`**: Helper utilities for loading, managing
    precision, and registering weights with `GpuModelBuilder`.
*   **`extract_weights_hf.py`**: Python script to extract and quantize weights
    from Hugging Face `.safetensors` models into binary files usable by
    ML Drift model runners.
*   **`BUILD`**: bazel build file for all libraries, tests, runners, and
    profiling binaries.

<br>

---

## Sample LLM Model Runner (`llm_runner`)

The sample LLM runner executes end-to-end text generation (or prefill/decode
throughput benchmarking) using extracted Hugging Face weights and a tokenizer.

> Note: This basic ML Drift LLM runner is designed for clarity and educational
> purposes; it prioritizes simplicity and code readability over advanced
> runtime scheduling features.

### Preparing Model Weights (`extract_weights_hf`)

The LLM sample focuses on running Q4_0 quantized versions of the supported
models using 4-bit weights and block size 32. Weights are downloaded from
Hugging Face as unquantized float weights, and running `extract_weights_hf.py`
converts and quantizes them into separate binary tensor files for ML Drift.

1.  **Download float model weights and tokenizer from Hugging Face**:
    *   [Gemma 3 270M](https://huggingface.co/google/gemma-3-270m-it/tree/main)
    *   [Gemma 3 1B QAT](https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-unquantized/tree/main)
    *   [Gemma 4 12B](https://huggingface.co/google/gemma-4-12B-it/tree/main)
    *   [Qwen 3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/main)
    *   [Qwen 3 1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/tree/main)
    *   [Qwen 3 8B](https://huggingface.co/Qwen/Qwen3-8B/tree/main)

2.  **Run `extract_weights_hf` to generate Q4_0 quantized binary weight files**:

    ```bash
    bazel run -c opt //third_party/ml_drift/samples/llm:extract_weights_hf -- \
      --model_path=/tmp/model.safetensors \
      --output_dir=/tmp/extracted \
      --quantize \
      --embedding_quant_bits=4 \
      --attention_quant_bits=4 \
      --feedforward_quant_bits=4 \
      --block_size=32
    ```

    *   `--model_path` can be either a single `.safetensors` file or a directory
        containing sharded `.safetensors` files.
    *   For a slight performance boost, use channel-wise quantization via
        `--block_size=0 --optimize-scale=True`.

3.  **Copy the tokenizer file** (`tokenizer.model` or
    `tokenizer.json`) into `/tmp/extracted/` (or pass its path
    via `--tokenizer_path`).

### Build Targets & Backends

| Target | Backend | Platform | Description |
| :--- | :--- | :--- | :--- |
| `//third_party/ml_drift/samples/llm:llm_runner_metal` | **Metal** | macOS | Native Metal LLM runner binary |
| `//third_party/ml_drift/samples/llm:llm_runner_webgpu` | **WebGPU** (Dawn/Vulkan/Metal) | Linux & macOS | WebGPU LLM runner binary |
| `//third_party/ml_drift/samples/llm:llm_runner_opencl` | **OpenCL** | Linux | OpenCL LLM runner binary |
| `//third_party/ml_drift/samples/llm:extract_weights_hf` | **Python (CPU)** | Linux & macOS | Extracts and quantizes Hugging Face `.safetensors` weights |

### Runner Command-Line Flags

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--model` | `"none"` | Required. Model to run: `gemma4:12b`, `gemma3:1b`, `gemma3:270m`, `qwen3:0.6b`, `qwen3:1.7b`, or `qwen3:8b`. |
| `--weights_path` | `""` | Required. Path to the directory containing extracted weight files (and tokenizer by default). |
| `--tokenizer_path` | `""` | Optional path to the tokenizer file (`tokenizer.model` or `tokenizer.json`). Defaults to `<weights_path>/tokenizer.model` (Gemma) or `<weights_path>/tokenizer.json` (Qwen). |
| `--prompt` | `""` | Input text prompt to feed into the model (required unless `--prompt_file` or `--benchmark` is set). |
| `--prompt_file` | `""` | Path to a text file containing the input prompt (overrides `--prompt`). |
| `--max_gen_tokens` | `512` | Maximum number of output tokens to generate during interactive decoding. |
| `--context_window_size` | `0` | Context window / KV cache size (`cache_size` in `LlmConfig`). If `0`, automatically set to `(# input tokens + max_gen_tokens)` rounded up to a multiple of 256. |
| `--benchmark` | `false` | If `true`, overrides `--prompt` and runs synthetic token prefill and decode throughput benchmarks with timing output. |
| `--benchmark_prefill_tokens` | `512` | Number of input tokens to prefill when `--benchmark=true`. |
| `--benchmark_decode_tokens` | `128` | Number of output tokens to decode when `--benchmark=true`. |

### Usage Examples

#### 1. Run Text Generation on macOS (Metal)

```bash
bazel run -c opt //third_party/ml_drift/samples/llm:llm_runner_metal -- \
  --model=gemma4:12b \
  --weights_path=/tmp/extracted/ \
  --prompt="Write a haiku about coffee." \
  --max_gen_tokens=128
```

#### 2. Run Text Generation on Linux or macOS (WebGPU / OpenCL)

```bash
# WebGPU backend:
bazel run -c opt //third_party/ml_drift/samples/llm:llm_runner_webgpu -- \
  --model=gemma3:1b \
  --weights_path=/tmp/extracted/ \
  --prompt="Write a haiku about coffee."

# OpenCL backend:
bazel run -c opt //third_party/ml_drift/samples/llm:llm_runner_opencl -- \
  --model=qwen3:0.6b \
  --weights_path=/tmp/extracted/ \
  --prompt="Write a haiku about coffee."
```

#### 3. Run from a Prompt File with an Explicit Context Window

```bash
bazel run -c opt //third_party/ml_drift/samples/llm:llm_runner_metal -- \
  --model=gemma4:12b \
  --weights_path=/tmp/extracted/ \
  --prompt_file=/tmp/long_prompt.txt \
  --context_window_size=4096 \
  --max_gen_tokens=256
```

#### 4. Run End-to-End Prefill & Decode Throughput Benchmark

Pass `--benchmark=true` to measure prefill and decode speed over a fixed number
of tokens without needing a text prompt:

```bash
bazel run -c opt //third_party/ml_drift/samples/llm:llm_runner_metal -- \
  --model=gemma4:12b \
  --weights_path=/tmp/extracted/ \
  --benchmark=true \
  --benchmark_prefill_tokens=512 \
  --benchmark_decode_tokens=128
```

<br>

---

## Performance Profiling (`llm_performance_profiling`)

`llm_performance_profiling` is an op-level and architectural-component GPU
performance profiler for the LLM graphs (`prefill`, `decode`, and `greedy`
post-processing) built using `*_model_builder.cc` files.

Use this tool to identify which operations, shader dispatches, or transformer
sub-blocks (e.g., `fc1x1` quantized projections, `QKVRmsNormRoPE`, local vs.
global attention `Q*K^T` / `Score*V`, KV cache updates, or LM Head) dominate
execution time so you know where to optimize next. By default, it allocates
synthetic quantized GPU weights in milliseconds so you can profile any model
configuration without downloading or extracting multi-gigabyte weight files.

### Build Targets & Backends

| Target | Backend | Platform | Description |
| :--- | :--- | :--- | :--- |
| `//third_party/ml_drift/samples/llm:llm_performance_profiling` | **Metal** (macOS) / **WebGPU** (Linux) | macOS & Linux | Default alias that automatically selects Metal on macOS and WebGPU on Linux |
| `//third_party/ml_drift/samples/llm:llm_performance_profiling_metal` | **Metal** | macOS | Explicit native Metal backend target |
| `//third_party/ml_drift/samples/llm:llm_performance_profiling_webgpu` | **WebGPU** (Dawn/Vulkan/Metal) | Linux & macOS | Explicit WebGPU backend target |
| `//third_party/ml_drift/samples/llm:llm_performance_profiling_opencl` | **OpenCL** | Linux | Explicit OpenCL backend target |

### Profiler Command-Line Flags

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--model` | `"gemma4:12b"` | Model to profile: `gemma4:12b`, `gemma3:270m`, `gemma3:1b`, `qwen3:0.6b`, `qwen3:1.7b`, or `qwen3:8b`. |
| `--phase` | `"all"` | Which inference graph(s) to profile: `all`, `decode`, `prefill`, or `greedy`. |
| `--num_layers` | `0` | Override transformer layer count (`stack_size`). `0` uses the model's full layer count (`48` for `gemma4:12b`). Tip: use `--num_layers=6` on `gemma4:12b` to profile one complete 5-local + 1-global layer cycle in seconds. |
| `--prefill_seq_len` | `64` | Sequence length for the prefill graph (clamped to `1024`). |
| `--max_seq_len` | `2048` | Maximum KV cache context length (`config.cache_size`). |
| `--token_offset` | `128` | Simulated token position in the KV cache during `decode` profiling. |
| `--weights_path` | `""` | Path to extracted weights directory. If empty, fast synthetic GPU weights are allocated without disk I/O. |
| `--weight_bits` | `4` | Synthetic weight bit-width: `4` (int4), `8` (int8), `16` (fp16), or `32` (fp32). |
| `--quantization_group_size` | `32` | Input channel quantization block size for synthetic `int4`/`int8` weights (`32`, `64`, `128`, or `-1` for per-channel). |
| `--include_zero_point` | `false` | Whether synthetic quantized weights include asymmetric zero-point tensors. |
| `--reuse_synthetic_weights` | `false` | Reuse backing GPU buffers across layers with identical weight descriptors to reduce VRAM footprint on memory-constrained devices. |
| `--use_fp32` | `false` | Force FP32 activations and calculations instead of FP16. |
| `--print_per_dispatch` | `false` | Print the full node-by-node `ProfilingInfo::GetDetailedReport()` trace before the aggregated tables. |
| `--end_to_end_iters` | `10` | Number of full-graph executions to measure wall-clock latency after per-kernel profiling (`0` to skip). |

### Usage Examples

#### 1. Profile Gemma 4 (12B) Decode & Prefill on macOS (Metal)

To profile the full 48-layer `gemma4:12b` model with `int4` synthetic weights on
macOS via Metal:

```bash
bazel run -c opt //third_party/ml_drift/samples/llm:llm_performance_profiling -- \
  --model=gemma4:12b \
  --weight_bits=4 \
  --quantization_group_size=32 \
  --prefill_seq_len=64 \
  --max_seq_len=2048
```

> Tip: If your device has limited unified memory / VRAM, pass
> `--reuse_synthetic_weights=true` so identical layer weight descriptors share
> backing GPU buffers.

#### 2. Fast Iteration: Profile One 6-Layer Cycle of Gemma 4 (`--num_layers=6`)

Gemma 4 repeats a 6-layer pattern (5 local sliding-window attention layers with
`head_dim=256, kv_heads=8` followed by 1 global full-attention layer with
`head_dim=512, kv_heads=2, K=V`). Profiling 6 layers compiles and runs 8x faster
while capturing the exact per-layer kernel mix:

```bash
bazel run -c opt //third_party/ml_drift/samples/llm:llm_performance_profiling -- \
  --model=gemma4:12b \
  --num_layers=6 \
  --phase=decode
```

#### 3. Profile with Real Extracted Weights (`--weights_path`)

To profile using real converted weights on disk instead of synthetic weights:

```bash
bazel run -c opt //third_party/ml_drift/samples/llm:llm_performance_profiling -- \
  --model=gemma4:12b \
  --weights_path=/tmp/extracted/ \
  --phase=all
```

#### 4. Compare Decode Attention Scaling Across Context Lengths

To inspect how global attention (`Attention Q*K^T (Global)`,
`Attention Softmax`, and `Attention Score*V (Global)`) scales as the KV cache
fills up, increase `--max_seq_len` and `--token_offset`:

```bash
bazel run -c opt //third_party/ml_drift/samples/llm:llm_performance_profiling -- \
  --model=gemma4:12b \
  --num_layers=6 \
  --phase=decode \
  --max_seq_len=8192 \
  --token_offset=4096
```

#### 5. Compare Models (`gemma3:1b` or `qwen3:8b`) or Quantization (`int4` vs. `int8`)

```bash
# Profile Qwen3 8B with int8 weights:
bazel run -c opt //third_party/ml_drift/samples/llm:llm_performance_profiling -- \
  --model=qwen3:8b \
  --num_layers=8 \
  --weight_bits=8 \
  --phase=decode

# Profile Gemma 3 1B with full per-dispatch trace enabled:
bazel run -c opt //third_party/ml_drift/samples/llm:llm_performance_profiling -- \
  --model=gemma3:1b \
  --phase=decode \
  --print_per_dispatch=true
```

#### 6. Run on Linux with OpenCL Instead of WebGPU

```bash
bazel run -c opt //third_party/ml_drift/samples/llm:llm_performance_profiling_opencl -- \
  --model=gemma4:12b \
  --num_layers=6 \
  --phase=decode
```

### Interpreting the Output Tables

For _each_ profiled phase (`PREFILL`, `DECODE`, `GREEDY POST-PROCESS`), the tool
prints:

1.  **Summary by Low-Level Op Prefix**: Raw ML Drift kernel prefixes
    (e.g., `fc1x1_int4_weights`, `convolution`, `extract_local_cache`,
    `QKVRmsNormRoPE`, `rms_normalization`, `gelu_tanh_mul`, `softmax`) ranked by
    GPU time.
2.  **Table 1 — High-Level LLM Component Summary**:
    *   **`MLP / FeedForward`**: `gate_proj`, `up_proj`, `down_proj`, and gated
        activation (`gelu_tanh_mul` / `silu_mul`).
    *   **`Attention Projections (QKV/O)`**: `q_proj`, `k_proj`, `v_proj`, and
        `o_proj` linear projections, separated into `(Local)` and `(Global)` in
        Table 2.
    *   **`Attention Core (QK^T, Softmax, SV)`**: Batched matmuls for attention
        logits and value aggregation plus attention mask selection and softmax.
    *   **`Embeddings & LM Head`**: Token embedding lookup (`embedding_lookup`),
        prefill last-token slice (`strided_slice`), and final vocabulary
        projection (`LM Head Projection`).
    *   **`RoPE & KV Cache`**: Fused `QKVRmsNormRoPE` (or `rope`),
        `cache_update`, and `extract_local_cache`.
    *   **`Weight Dequant / Conversion`**: Weight dequantization/format
        conversion kernels (e.g., `weights_convert_uint4_to_float16`) when the
        backend's batched prefill convolution does not dequantize `int4` weights
        inline.
    *   **`Linear Projections (Quantize Input)`**: Dynamic activation
        quantization (`quantize_and_gather`) preceding quantized `fc1x1` kernels
        when dynamic quantization is active on the target GPU.
3.  **Table 2 — Breakdown by LLM Role, Operation & Shape**:
    *   Lists every distinct `(LLM Role, Operation, In[0] -> Out[0] BHWC Shape)`
        tuple, how many times it executed across the model (`Count`), total and
        average execution time (`Total(ms)`, `Avg(ms)`), percentage of total GPU
        time (`%Time`), and achieved memory bandwidth (`GB/s`) and compute rate
        (`GFLOP/s`).
    *   For memory-bound decode kernels (`sequence_size=1`), compare the
        `GB/s` column against your GPU's peak memory bandwidth to spot kernels
        with low memory utilization.
    *   For compute-bound prefill kernels (`sequence_size=64+`), compare the
        `GFLOP/s` column across `fc1x1` and `convolution` variants.
