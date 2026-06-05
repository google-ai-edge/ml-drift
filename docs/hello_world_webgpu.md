# WebGPU Example

<!-- linter off -->

## Environment setup on Linux or macOS

Step 1. Build Dawn library on Linux or macOS, and set up ML Drift's WORKSPACE:

```bash
sh third_party/dawn/build_libdawn.sh
```

Step 2. Download and install the latest version of Bazelisk from the [releases page](https://github.com/bazelbuild/bazelisk/releases). Or, install the latest bazelisk from your package manager (`apt-get`/ `brew`/ etc)

## Run tests on Linux or macOS

Step 1. Build one of the unit tests:

```bash
bazelisk build -c opt ml_drift/common/kernels/tests:add_test_webgpu
```

A successful build will display a message similar to this:

```bash
Target //ml_drift/common/kernels/tests:add_test_webgpu up-to-date:
  bazel-bin/ml_drift/common/kernels/tests/add_test_webgpu
INFO: Elapsed time: 29.057s, Critical Path: 28.33s
INFO: Build completed successfully, 62 total actions
```

Step 2. Run the test on Linux or macOS:

```bash
bazel-bin/ml_drift/common/kernels/tests/add_test_webgpu
```

A successful run will display the following in the terminal:

```bash
[==========] Running 96 tests from 2 test suites.
[----------] Global test environment set-up.
[----------] 42 tests from AddTypedTestSuite/AddTypedTest
[ RUN      ] AddTypedTestSuite/AddTypedTest.TwoEqualTensors/TensorStorageTypeBUFFER
[       OK ] AddTypedTestSuite/AddTypedTest.TwoEqualTensors/TensorStorageTypeBUFFER (578 ms)
[ RUN      ] AddTypedTestSuite/AddTypedTest.TwoEqualTensors/TensorStorageTypeTEXTURE_ARRAY
[       OK ] AddTypedTestSuite/AddTypedTest.TwoEqualTensors/TensorStorageTypeTEXTURE_ARRAY (419 ms)
[ RUN      ] AddTypedTestSuite/AddTypedTest.TwoEqualTensors/TensorStorageTypeTEXTURE_2D
[       OK ] AddTypedTestSuite/AddTypedTest.TwoEqualTensors/TensorStorageTypeTEXTURE_2D (224 ms)
[ RUN      ] AddTypedTestSuite/AddTypedTest.TwoEqualTensors/TensorStorageTypeTEXTURE_3D
[       OK ] AddTypedTestSuite/AddTypedTest.TwoEqualTensors/TensorStorageTypeTEXTURE_3D (290 ms)
...
```

## Run Performance Profiling on Linux or macOS

Step 1. To run a performance test, you'll need a TFLite model file. For example, you can download the MediaPipe object detection model from [here](
https://github.com/google-ai-edge/mediapipe/blob/f96eadd6df64a5f9a31918d6319e51847497641a/mediapipe/models/ssdlite_object_detection.tflite).

Step 2. Build the performance profiling binary:

```bash
bazelisk build -c opt ml_drift/webgpu/testing:performance_profiling
```

A successful build will display a message similar to this:

```bash
Target //ml_drift/webgpu/testing:performance_profiling up-to-date:
  bazel-bin/ml_drift/webgpu/testing/performance_profiling
INFO: Elapsed time: 273.071s, Critical Path: 226.35s
INFO: Build completed successfully, 1552 total actions
```

Step 3. Run the binary with a TFLite model

```bash
bazel-bin/ml_drift/webgpu/testing/performance_profiling --model_path ${HOME}/Downloads/ssdlite_object_detection.tflite
```

After running the script, you'll see detailed profiling information displayed in the terminal.

```bash
Precision: CalculationsPrecision::F16
Storage type: TensorStorageType::TEXTURE_2D
GraphFloat32 from flatbuffer initialization time: 0.839917 ms.
GPU model from GraphFloat32 initialization time: 3.20133 ms.
Inference context from GPU model initialization time: 2569.11 ms.
Per kernel timing(103 kernels):
  convolution_2d 0 -> relu 1; 0.133391 ms; 11.447432 Gb/s; 165.816595 Gflops
  depthwise_convolution 2 -> relu 3; 0.053654 ms; 28.444661 Gb/s; 137.413159 Gflops
  convolution_2d 4 -> add 5; 0.060040 ms; 38.130155 Gb/s; 218.308704 Gflops
  convolution_2d 6 -> relu 7; 0.143461 ms; 37.247930 Gb/s; 548.186176 Gflops
  depthwise_convolution 8 -> relu 9; 0.128195 ms; 44.649346 Gb/s; 86.268407 Gflops
  ...
  ...
  concat 171; 0.065724 ms; 10.606528 Gb/s
  reshape 172; 0.020528 ms; 1.476487 Gb/s
--------------------
Accumulated time per operation type:
  concat(x2) - 0.094251 ms, 7.717891 Gb/s
  convolution_2d(x52) - 1.790015 ms, 13.670818 Gb/s, 317.000570 Gflops
  depthwise_convolution(x33) - 0.863383 ms, 19.502018 Gb/s, 67.271354 Gflops
  fully_connected(x3) - 0.049274 ms, 1.515406 Gb/s
  reshape(x13) - 0.244570 ms, 3.093954 Gb/s
--------------------
Ideal total time: 3.041492
--------------------
Total Gflops for model: ~0.625517
Average Gflops per sec: ~205.661114
Average Gb/s per sec: ~14.094218
--------------------

Memory for intermediate tensors - 14.257 MB
Memory for constant tensors - 2.9898 MB
Total tensors memory(const + intermediate) - 17.2467 MB
Model time(GPU throughput) - 11.4383ms
...
```
