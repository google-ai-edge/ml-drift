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
