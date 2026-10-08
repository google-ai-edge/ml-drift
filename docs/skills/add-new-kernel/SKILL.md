---
name: mldrift-add-new-kernel
description: >-
  Generates code for a new GPU Kernel in ml_drift using UCL syntax.
  Use when the user wants to add or implement a new operation/kernel in ml_drift (e.g., SquaredDifference, Maximum).
  Creates the .h, .cc files, generates test files (test.cc, test_util.h, test_util.cc), and updates BUILD rules.
  Don't use for generic C++ development outside of ml_drift.
license: Apache-2.0 (full text in the repository's top-level LICENSE file)
---
<!-- disableFinding(LINE_OVER_80) -->

# ML Drift Add New Kernel

This skill guides you to implement a new ML Drift GPU Kernel using ML Drift's internal layouts and the internal Unified Compute Language (UCL) so that the kernel can run across multiple backends natively.

## Instructions

When asked to add a new GPU kernel (e.g., "squared_difference") to ml_drift, strictly follow these 5 steps in order:

### 1. Create the Header (`common/kernels/{op_name}.h`)
Create the header file with standard include guards (`ML_DRIFT_COMMON_KERNELS_{OP_NAME}_H_`).

*   **Licensing**: If generating new files in a user's repository, do NOT hardcode ML Drift's Apache copyright headers. Instead, adopt the existing license header style of the user's surrounding files, or leave it blank.
*   **Namespace**: Always place the code under `namespace ml_drift { ... }`.
*   **Factory Function**: Declare a factory function that returns a `GPUOperation`.
    *   Example: `GPUOperation CreateSquaredDifference(const OperationDef& definition, ...);`

### 2. Create the Implementation (`common/kernels/{op_name}.cc`)
Create the implementation file. It must define the factory function and inject UCL shader code.

*   **Factory Setup**:
    *   Instantiate `GPUOperation op;`.
    *   Add input/output tensors using `op.AddSrcTensor()` and `op.AddDstTensor()`.
    *   Register any custom parameters with primitive data types using `op.args_.AddFloat()`, `op.args_.AddInt()`, etc.
    *   *Multiple Inputs Tip*: If the operation has two inputs, add them with distinct names: `op.AddSrcTensor("src_tensor_0", ...);` and `op.AddSrcTensor("src_tensor_1", ...);`.
*   **Grid Assignment**:
    *   Explicitly set the `grid_size_` based on the tensor shape using `int3` and `DivideRoundUp`.
    *   Example: `op.grid_size_ = int3(DivideRoundUp(shape.w, 8), DivideRoundUp(shape.h, 4), shape.c);`
*   **UCL Code Generation**:
    *   The `op.code_` must be a string containing UCL code.
    *   Wrap the actual shader inside `MAIN_FUNCTION($0) { ... }`.

**CRITICAL: Tensor Layout and Storage Type Guide**
When designing the shader logic and declaring tests, you MUST work with ML Drift's layout rules:

*   **Logical Layouts**:
    *   Inputs, outputs, and intermediate tensors typically use `LINEAR`, `HWC`, or `BHWC`. Weights use `OHWI` or `IHWO`.
    *   Use `Layout::HWC` (or `BHWC`) for variables read using `X, Y, S` coordinates in UCL.
    *   `Layout::LINEAR` is only for 1D vectors and does not support three-parameter reads like `.Read(X, Y, S)`.
*   **Slices & Padding**:
    *   Tensors are split into 4-channel slices stored sequentially in memory.
    *   If the number of channels is not divisible by 4, it is zero-padded.
*   **Physical Coords Translation**:
    *   The shader `Read`/`Write` calls (e.g., `args.src_tensor.Read(X, Y, S)`) use logical coordinates.
    *   ML Drift automatically translates these to physical coordinates based on the underlying hardware storage type.
    *   For `BHWC` Textures, the arguments translate to `(X * batch_size + B, Y * num_slices + S)`.
    *   Therefore, you do NOT need to manually compute physical memory offsets in UCL; just pass logical coordinates `(B, X, Y, S)` for BHWC, `(X, Y, S)` for HWC, and `(index)` for LINEAR directly.

**CRITICAL: UCL Syntax Guide**
When generating the shader code string, you MUST use the following UCL semantics:

*   **Tensor/Arg Names**: The names accessed via `args.NAME` in the shader MUST exactly match the string names registered in the C++ file.
*   **Thread IDs**: Retrieve global coordinates using `ucl::GetGlobalId<0>()` (usually X), and local workgroup coordinates using `ucl::GetLocalId<0>()`.
*   **Local Memory & Synchronization**: Declare shared memory arrays with `__local` (e.g., `__local int hist[SIZE];`). Synchronize threads in a workgroup using `ucl::SyncThreads<WorkGroup, Local>();`.
*   **Macro Injection**: For constants (like array sizes or bucket counts), do NOT pass them as runtime arguments. Instead, inject them as macros into the shader string in C++ using `absl::StrReplaceAll({{"NUM_BUCKETS", std::to_string(num)}}, &code);`.
*   **Initialization**: Vectors MUST be initialized via `ucl::Init`. E.g., `int4 result = ucl::Init<int4>(0);`.
*   **Type Casting**: Use `ucl::Convert<TargetType>(value)`.
*   **Tensor Properties**: Read tensor dimensions via `args.dst_tensor.Width()`, `args.dst_tensor.Height()`, `args.dst_tensor.Slices()`.
*   **Tensor I/O**:
    *   Read: `args.src_tensor.Read(X, Y, S)`
    *   Write: `args.dst_tensor.Write(value, X, Y, S)`
    *   Linear read/write: `args.src_tensor.ReadPerChannel(value, index)`, `args.dst_tensor.WriteLinear(value, DST_S)`
*   **Type deduction**: Use `args.src_tensor::type` and `args.src_tensor::zero_value` to create and initialize generic variables.

### 3. Create Tests (`common/kernels/tests/`)
Do not bundle tests into a single file. You must create three separate files and implement actual test logic using `TestExecutionEnvironment` and `TensorDescriptor`.

1.  `tests/{op_name}_test_util.h`: Declare `Run*Test(TestExecutionEnvironment& env, ...)` parameterized test helpers.
2.  `tests/{op_name}_test_util.cc`: Implement setup, `UploadDataRaw`, `ExecuteGPUOperation`, and `DownloadData` for verification.
3.  `tests/{op_name}_test.cc`: The actual googletest cases calling the util functions.

**Implementation Guide for Tests:**

*   **Test Parameterization**:
    *   If the kernel supports variations (e.g., multiple data types like `float32`/`float16`, or varying sizes), use GTest value parameterization (`TEST_P` and `INSTANTIATE_TEST_SUITE_P`) rather than `TYPED_TEST`.
    *   For an example, refer to `tests/add_test.cc`. This allows you to cleanly execute all test cases across supported configurations.
*   **Required Test Cases** in `{op_name}_test.cc`:
    *   `SingleElementTest`: A trivial `1x1x1x1` shape test.
        *   *Note: For **ALL** subsequent tests, you MUST use multidimensional shapes, e.g., `BHWC(1, 2, 3, 4)` to ensure UCL coordinate traversal is fully exercised.*
    *   `BasicTest`: Multidimensional shape with normal positive numbers.
    *   `PaddingTest`: Crucial for ML Drift! Since the channel dimension is zero-padded to multiples of 4, test with `C` that is NOT a multiple of 4 (e.g., `C=3` or `C=5`) to ensure bounds are handled cleanly.
    *   `NegativeValuesTest`: Multidimensional shape with negative numbers, to ensure sign preservation and logic hold up.
    *   `ZerosTest`: Multidimensional shape with zeros, to catch div-by-zero or initialization errors.

### 4. Update `BUILD` Rules
You must update both the kernel and test BUILD files:

1.  **`common/kernels/BUILD`**: Append a new `cc_library` target for your operation.
    *   The target name should match the op_name.
    *   `srcs = ["{op_name}.cc"]` and `hdrs = ["{op_name}.h"]`
2.  **`common/kernels/tests/BUILD`**: Register the new test target using `mld_kernel_test(name = "{op_name}_test")` and `cc_library(name = "{op_name}_test_util")`.

### 5. Test and Verify
The `mld_kernel_test` macro generates test targets for multiple GPU backends (e.g., `_opencl`, `_opengl`, `_metal`). You MUST test your kernel on all available backends.

