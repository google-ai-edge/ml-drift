# Sample ML-Drift LLM Model Runner

This folder contains files for running LLM models using ML-Drift
Metal, WebGPU or OpenCL.

Models supported: Gemma3-1B, Gemma3-270M, Gemma4-12B,
Qwen3-0.6B, Qwen3-1.7B, Qwen3-8B

Note: This basic ML Drift LLM implementation is for educational purposes rather
than benchmarking. This sample prioritizes simplicity and code readability over
performance, and lacks advanced features.

## Contents

*   **llm_config.h**: C++ header for model configurations.
*   **llm_runner.h**: Shared C++ header for the runner interface, tokenizer,
    and helper functions.
*   **llm_runner.cc**: Generic C++ main entrypoint file that defines the
    CLI flags and runner loop.
    * Build targets: `llm_metal_runner, llm_opencl_runner, llm_webgpu_runner`
*   **llm_metal.cc**: Backend-specific implementation of the runner for Metal.
*   **llm_opencl.cc**: Backend-specific implementation of the runner for OpenCL.
*   **llm_webgpu.cc**: Backend-specific implementation of the runner for WebGPU.
*   **gemma3_model_builder.h/cc**: C++ library for building Gemma3 models using
    ML-Drift `GpuModelBuilder`.
*   **gemma4_model_builder.h/cc**: C++ library for building Gemma4 models using
    ML-Drift `GpuModelBuilder`.
*   **qwen3_model_builder.h/cc**: C++ library for building Qwen3 models using
    ML-Drift `GpuModelBuilder`.
*   **llm_file_tensor_loader.h/cc**: C++ implementation of `LlmTensorLoader`
    for loading weight, bias, and scale tensors from binary files on disk.
*   **llm_weights_loader.h/cc**: C++ helper utilities for loading, managing
    precision, and registering weights with `GpuModelBuilder`.
*   **extract_weights_hf.py**: Python script to extract weights from Hugging
    Face models (.safetensors) into a format usable by ML Drift model runners.
    Saves all tensors as separate binary files in F32 precision, int8 or int4.
*   **BUILD**: Build file for the project.

## Getting Started

The LLM sample focuses on running Q4_0 quantized versions of the supported
models using all 4-bit weights and block size 32. Weights are downloaded from
Hugging Face as unquantized float weights, and running `extract_weights_hf.py`
is a required step to perform ML-Drift's special quantization. Even though
running with float32 weights is supported, it is not shown here because memory
and performance suffer significantly without quantization.

*   Download float model weights and tokenizer from huggingface.com:
    *   [Gemma3 270m](https://huggingface.co/google/gemma-3-270m-it/tree/main)
    *   [Gemma3 1B QAT](https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-unquantized/tree/main)
    *   [Gemma4 12B](https://huggingface.co/google/gemma-4-12B-it/tree/main)
    *   [Qwen3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/main)
    *   [Qwen3 1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/tree/main)
    *   [Qwen3 8B](https://huggingface.co/Qwen/Qwen3-8B/tree/main)

*   Run extract_weights_hf.py to get binary weight files with Q4_0 quantization
    (all 4-bit weights and block size 32):
    *   `blaze run -c opt
        //third_party/ml_drift/samples/llm:extract_weights_hf --
        --model_path=/tmp/model.safetensors --output_dir=/tmp/extracted
        --quantize --embedding_quant_bits 4 --attention_quant_bits 4
        --feedforward_quant_bits 4 --block_size 32`
    *   Note: `--model_path` can be either a single `.safetensors` file or a
        directory containing multiple `.safetensors` files.
    *   For a slight performance boost, use channelwise quantization, via:
        `--block_size 0 --optimize-scale True`

*   Copy `tokenizer.model` or `tokenizer.json` into `output_dir` also

*   Run *llm_runner_webgpu* (_metal and _opencl backends also available)
    *   `blaze run -c opt
        //third_party/ml_drift/samples/llm:llm_runner_webgpu --
        --weights_path=/tmp/extracted/
        --prompt="Write a haiku about coffee."
        --model=<model string>`

*   Supported runner flags:
    *   `--model` (model string) must be one of:
         "gemma3:1b", "gemma3:270m", "gemma4:12b", "qwen3:0.6b", "qwen3:1.7b",
         "qwen3:8b"
    *   `--prompt` or `--prompt_file` to load prompt
    *   `--max_gen_tokens` controls maximum number of output tokens
    *   `--weights_path` set to extracted weights folder (+ tokenizer)
    *   `--tokenizer_path` (optional) path to tokenizer file (defaults to
        `weights_path`/tokenizer.json or tokenizer.model)
