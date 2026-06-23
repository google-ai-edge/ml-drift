# OpenCL on Android Example

<!-- linter off -->

## Environment setup on Linux or macOS

Step 1. Install Android SDK and NDK, and set the `ANDROID_SDK_HOME` and `ANDROID_NDK_HOME` env variables according to the script output:

```bash
wget -P /tmp https://raw.githubusercontent.com/google-ai-edge/mediapipe/refs/heads/master/setup_android_sdk_and_ndk.sh
bash /tmp/setup_android_sdk_and_ndk.sh

export ANDROID_SDK_HOME=<see the script output>
export ANDROID_NDK_HOME=<see the script output>
```

Note: Only the NDK is actually used for building the core ML Drift library (SDK can be skipped for this example).

Step 2. Download and install the latest version of Bazelisk from the [releases page](https://github.com/bazelbuild/bazelisk/releases). Or, install the latest bazelisk from your package manager (`apt-get`/ `brew`/ etc)

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
