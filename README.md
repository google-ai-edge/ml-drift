# ML Drift

![ML Drift Logo](ml_drift.png)

Performant & Universal On-device GPU Compute.

## Overview

ML Drift is a high-performance, cross-platform GPU-accelerated inference
engine for on-device machine learning intelligence. It is designed to enable
efficient execution of demanding AI/ML workloads, including large generative
models, across a wide range of devices such as mobile phones (Android & iOS),
web browsers, and servers. The engine focuses on minimizing latency and power
consumption, empowering developers to bring real-time intelligence to edge
devices while addressing critical needs like user privacy and offline
functionality.

## Key Features

* **High Performance:** Achieves an order-of-magnitude performance
  improvement relative to existing open-source GPU inference engines through
  highly optimized kernels and runtime.
* **Cross-Platform:** Seamlessly runs on various operating systems and
  hardware, including mobile, desktop, and servers.
* **Multiple GPU API Backends:** Comprehensive support for OpenCL, Metal,
  WebGPU (via Dawn), and OpenGL ES 3.1+.
* **Unified Compute Language (UCL):** Write GPU kernels once in an
  abstraction layer over device-specific shading languages (GLSL, MSL, WGSL)
  to be compiled dynamically for different backends.
* **Broad Operator Support:** Highly tuned implementations of common ML
  operations.
* **Dynamic Code Generation:** Shaders are generated at runtime, tailored to
  the specific model, inputs, and GPU for maximum efficiency.
* **Advanced Memory Management:** Efficiently reuses GPU memory to run large
  models with limited resources using advanced allocation algorithms.
* **Large Generative Model Capabilities:** Specially designed to run large
  workloads, features stage-aware execution (prefill vs. decode) and supports
  optimizations like FP16/INT8/INT4 quantization.
* **Custom Workloads:** ML Drift's `GpuModel` graphs can be hand-tailored to
  fit any ML model.

## Target Use Cases

* On-device inference for mobile applications (Android/iOS).
* In-browser ML acceleration using WebGPU.
* Accelerating ML models on various GPU-equipped edge devices.
* Running large language models (LLMs) and generative AI efficiently on-device.

## Architecture Highlights

* **Model Abstraction:** Uses a central, backend-agnostic in-memory
  representation of a machine learning model's compute graph called
  `GpuModel`.
* **Unified Compute Language (UCL):** An abstraction layer over
  device-specific shading languages (GLSL, MSL, WGSL, etc.), enabling
  write-once, run-anywhere GPU kernels.
* **GpuOperation:** Represents a single computation step, containing the UCL
  shader code and data references.
* **Tensor Virtualization:** A technique decoupling logical tensor views from
  physical GPU storage, allowing dynamic mapping of indices to memory (e.g.,
  textures) during code generation without runtime overhead.
* **Runtime Optimization:** Performs graph transformations, operator fusion
  to reduce kernel launch overhead, and layout adjustments.
* **Backend Interface:** Abstracts GPU API specifics, allowing the core
  engine to be platform-agnostic.

## Supported Backends

* OpenCL (Primary for Android)
* Metal (Apple devices)
* WebGPU (Web browsers, cross-platform via Dawn)
* OpenGL ES 3.1+

## Key Optimizations

* **Dynamic Kernel Generation:** Optimizes shaders at runtime.
* **Memory Reuse:** Uses algorithms like `GREEDY_BY_SIZE` to minimize the
  GPU memory footprint for intermediate tensors.
* **Precision & Quantization:** Leverages FP16/INT8/INT4 quantization for
  significant speed and memory gains.
* **Weights Rearrangement:** Optimizes weight layouts for specific GPU
  access patterns and data locality.
* **Workgroup Tuning:** Adapts GPU thread distribution to specific hardware
  architectures.
* **LLM-Specific Optimizations:** Efficient KV cache management and
  specialized handling of prefill vs. decode stages.

## Directory Structure

* `api/`: Public API headers.
* `common/`: Core components, UCL kernels, `GpuModel`, and platform-neutral
  logic.
* `cl/`, `gl/`, `metal/`, `webgpu/`: Backend-specific implementations.
* `samples/`: Example usage and demos.

## Getting Started

See "Hello world" examples using ML Drift's
[OpenCL](docs/hello_world_cl.md) and
[WebGPU](docs/hello_world_webgpu.md) apis.

Check out the [samples](ml_drift/samples/) directory for more examples.

## Status

ML Drift is actively developed and is the successor to the GPU delegate in
TensorFlow Lite.

See the [contributing](CONTRIBUTING.md) page if you are interested in
improving ML Drift.
