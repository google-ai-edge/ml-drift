# Sample ML-Drift SD1.5 Runner
This folder contains files for running Stable Diffusion 1.5 using ML-Drift.

## Instructions

Example for running SD 1.5 sample for WebGPU (OpenCL and Metal backends are also
available). For the following instructions, the working directory will be
`/tmp/sd_1_5` for simplicity.

### Data prep

- Download and extract [bpe_simple_vocab_16e6.txt](https://huggingface.co/OpenGVLab/ViCLIP-B-16-hf/blob/main/bpe_simple_vocab_16e6.txt.gz)

  - copy to `/tmp/sd_1_5/`

- Download [v1-5-pruned-emaonly-fp16] (https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/blob/main/v1-5-pruned-emaonly-fp16.safetensors) weights (safetensors)

- Extract safetensors weights via `extract_weights.py`

  - `bazel run //third_party/ml_drift/samples/stable_diffusion:extract_weights -- --model_path=/tmp/v1-5-pruned-emaonly.safetensors --output_dir=/tmp/sd_1_5/`

### Build and Run

- Build sample (choose a backend)

  - `bazel build -c opt //third_party/ml_drift/samples/stable_diffusion:sd_gpu_webgpu`
  - `bazel build -c opt //third_party/ml_drift/samples/stable_diffusion:sd_gpu_metal`
  - `bazel build -c opt //third_party/ml_drift/samples/stable_diffusion:sd_gpu_opencl`

- Run sample (example with WebGPU)

  - Copy resulting binary `sd_gpu_webgpu` into `/tmp/sd_1_5`

  - (Check: weights, vocab, binary files are all in `/tmp/sd_1_5`)

  - `./sd_gpu_webgpu /tmp/sd_1_5`

  - Output will be saved to `/tmp/sd_1_5/result.bmp`
