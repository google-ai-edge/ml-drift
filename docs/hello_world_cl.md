# OpenCL on Android Example

<!-- linter off -->

## Environment setup on Linux or macOS

Step 2. Install Android SDK and NDK, and set the `ANDROID_SDK_HOME` and `ANDROID_NDK_HOME` env variables according to the script output:

```bash
wget -P /tmp https://raw.githubusercontent.com/google-ai-edge/mediapipe/refs/heads/master/setup_android_sdk_and_ndk.sh
bash /tmp/setup_android_sdk_and_ndk.sh

export ANDROID_SDK_HOME=<see the script output>
export ANDROID_NDK_HOME=<see the script output>
```

Note: Only the NDK is actually used for building the core ML Drift library (SDK can be skipped for this example).

Step 3. Download and install the latest version of Bazelisk from the [releases page](https://github.com/bazelbuild/bazelisk/releases). Or, install the latest bazelisk from your package manager (`apt-get`/ `brew`/ etc)

## Running Unit Tests

Step 1. Build one of the unit tests:

```bash
bazelisk build -c opt --config=android_arm64 ml_drift/common/kernels/tests:add_test_opencl
```

A successful build will display a message similar to this:

```bash
Target //ml_drift/common/kernels/tests:add_test_opencl up-to-date:
  bazel-bin/ml_drift/common/kernels/tests/add_test_opencl
INFO: Elapsed time: 81.789s, Critical Path: 63.56s
INFO: Build completed successfully, 916 total actions
```

Step 2. Push the test binary to your Android device:

```bash
adb push bazel-bin/ml_drift/common/kernels/tests/add_test_opencl /data/local/tmp
```

Step 3. Run the test on Android:

```bash
adb shell /data/local/tmp/add_test_opencl
```

A successful run will display the following in the terminal:

```bash
Running main() from gmock_main.cc
[==========] Running 96 tests from 2 test suites.
[----------] Global test environment set-up.
[----------] 42 tests from AddTypedTestSuite/AddTypedTest
[ RUN      ] AddTypedTestSuite/AddTypedTest.TwoEqualTensors/TensorStorageTypeBUFFER
[       OK ] AddTypedTestSuite/AddTypedTest.TwoEqualTensors/TensorStorageTypeBUFFER (134 ms)
[ RUN      ] AddTypedTestSuite/AddTypedTest.TwoEqualTensors/TensorStorageTypeTEXTURE_ARRAY
[       OK ] AddTypedTestSuite/AddTypedTest.TwoEqualTensors/TensorStorageTypeTEXTURE_ARRAY (25 ms)
[ RUN      ] AddTypedTestSuite/AddTypedTest.TwoEqualTensors/TensorStorageTypeTEXTURE_2D
[       OK ] AddTypedTestSuite/AddTypedTest.TwoEqualTensors/TensorStorageTypeTEXTURE_2D (18 ms)
[ RUN      ] AddTypedTestSuite/AddTypedTest.TwoEqualTensors/TensorStorageTypeTEXTURE_3D
[       OK ] AddTypedTestSuite/AddTypedTest.TwoEqualTensors/TensorStorageTypeTEXTURE_3D (15 ms)
...
```

## Run Performance Profiling

Step 1. To run a performance test, you'll need a TFLite model file. For example, you can download the MediaPipe face landmark model from [here](
https://github.com/google-ai-edge/mediapipe/blob/f96eadd6df64a5f9a31918d6319e51847497641a/mediapipe/models/face_landmark.tflite).

Step 2. Run the script with a TFLite model:

```bash
sh ml_drift/cl/testing/run_performance_profiling.sh --model_path ${HOME}/Downloads/face_landmark.tflite
```

After running the script, you'll see detailed profiling information displayed in the terminal.

```bash
Precision: CalculationsPrecision::F16
Storage type: TensorStorageType::TEXTURE_2D
flatbuffer initialization time: 0.075156 ms.
INFO: Initialized TensorFlow Lite runtime.
VERBOSE: Replacing 97 out of 97 node(s) with delegate (unknown) node, yielding 1 partitions for the whole graph.
GraphFloat32 from flatbuffer initialization time: 2.33271 ms.
TransformsForGpuModel time: 0.041562 ms.
GPU model from GraphFloat32 initialization time: 9.28182 ms.
Inference context from GPU model initialization time: 462.668 ms.
Per kernel timing(45 kernels):
  convolution_2d 0 -> prelu 1; 0.012032 ms; 45.726452 Gb/s; 661.787234 Gflops
  dw_conv2d->conv1x1 -> add 4 -> prelu 5; 0.012032 ms; 68.551286 Gb/s; 612.765957 Gflops
  dw_conv2d->conv1x1 -> add 8 -> prelu 9; 0.007936 ms; 103.932595 Gb/s; 929.032258 Gflops
  dw_conv2d->conv1x1; 0.005888 ms; 70.193378 Gb/s; 513.391304 Gflops
  pooling_2d 11; 0.004096 ms; 83.819032 Gb/s
  add 14 -> prelu 15; 0.003840 ms; 89.422489 Gb/s
  ...
  ...
  convolution_2d 96; 0.029952 ms; 25.338243 Gb/s; 27.000000 Gflops
--------------------
Accumulated time per operation type:
  add(x1) - 0.003840 ms, 89.422489 Gb/s
  convolution_2d(x18) - 0.270592 ms, 9.456714 Gb/s, 137.590587 Gflops
  depthwise_convolution(x13) - 0.056576 ms, 14.182652 Gb/s, 49.846154 Gflops
  dw_conv2d->conv1x1(x7) - 0.060160 ms, 48.214768 Gb/s, 497.125532 Gflops
  pooling_2d(x6) - 0.021760 ms, 31.193828 Gb/s
--------------------
Ideal total time: 0.412928
--------------------
Total Gflops for model: ~0.069958
Average Gflops per sec: ~169.419560
Average Gb/s per sec: ~17.640047
--------------------

Memory for intermediate tensors - 0.632812 MB
Memory for constant tensors - 1.15335 MB
Total tensors memory(const + intermediate) - 1.78616 MB
Total time - 0.41622ms
...
```
