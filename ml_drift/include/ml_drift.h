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

#ifndef ML_DRIFT_INCLUDE_ML_DRIFT_H_
#define ML_DRIFT_INCLUDE_ML_DRIFT_H_

// Common headers for all APIs.

#include "absl/status/status.h"              // IWYU pragma: export
#include "absl/status/statusor.h"            // IWYU pragma: export
#include "ml_drift/common/gpu_info.h"        // IWYU pragma: export
#include "ml_drift/common/gpu_model.h"       // IWYU pragma: export
#include "ml_drift/common/gpu_model_util.h"  // IWYU pragma: export
#include "ml_drift/common/model.h"           // IWYU pragma: export
#include "ml_drift/common/model_hints.h"     // IWYU pragma: export
#include "ml_drift/common/precision.h"       // IWYU pragma: export
#include "ml_drift/common/shape.h"           // IWYU pragma: export
#include "ml_drift/common/task/tensor_desc.h"  // IWYU pragma: export
#include "ml_drift/common/tensor.h"  // IWYU pragma: export

#if defined(ML_DRIFT_API_MTL)

// Example usage:
//
//   ml_drift::metal::Environment env;
//
//   ml_drift::GpuModel gpu_model;
//   ABSL_QCHECK_OK(LoadTfLite(model, &gpu_model));
//
//   ml_drift::metal::InferenceContext context;
//   ml_drift::CreateGpuModelInfo ci;
//   ABSL_QCHECK_OK(context.InitFromGpuModel(ci, &gpu_model, &env);
//
//   // write input
//   context.SetInputTensor(...);
//
//   // run inference
//   id<MTLCommandQueue> command_queue = [env.device() newCommandQueue];
//   id<MTLCommandBuffer> cmd_buffer = [command_queue commandBuffer];
//   id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
//   context.EncodeWithEncoder(encoder);
//   [encoder endEncoding];
//   [cmd_buffer commit];
//   [cmd_buffer waitUntilCompleted];
//
//   // read output
//   context.GetOutputTensor(...)
//
// $ bazel build --define ml_drift_api=mtl --config darwin_arm64 -c opt <target>

#include "ml_drift/metal/metal_api.h"  // IWYU pragma: export

#elif defined(ML_DRIFT_API_CL)

// Example usage:
//
//   ABSL_QCHECK_OK(ml_drift::cl::LoadOpenCL());
//
//   ml_drift::cl::Environment env;
//   ABSL_QCHECK_OK(ml_drift::cl::CreateEnvironment(&env));
//
//   ml_drift::GpuModel gpu_model;
//   ABSL_QCHECK_OK(LoadTfLiteModelFile(model_file, &gpu_model));
//
//   ml_drift::cl::InferenceContext context;
//   ml_drift::CreateGpuModelInfo ci;
//   ABSL_QCHECK_OK(context.InitFromGpuModel(ci, &gpu_model, &env));
//
//   // write input
//   context.SetInputTensor(..., env.queue());
//
//   // run inference
//   ABSL_QCHECK_OK(context.AddToQueue(env.queue()));
//   ABSL_QCHECK_OK(ml_drift::cl::WaitUntilCompleted(env.queue()));
//
//   // read output
//   context.GetOutputTensor(...)
//
// $ bazel build --define ml_drift_api=cl --config android_arm64 -c opt <target>

#include "ml_drift/cl/cl_api_util.h"        // IWYU pragma: export
#include "ml_drift/cl/cl_command_buffer.h"  // IWYU pragma: export
#include "ml_drift/cl/environment.h"        // IWYU pragma: export
#include "ml_drift/cl/inference_context.h"  // IWYU pragma: export
#include "ml_drift/cl/opencl_wrapper.h"     // IWYU pragma: export
#include "ml_drift/cl/tensor.h"             // IWYU pragma: export

#elif defined(ML_DRIFT_API_WGPU)

// Example usage:
//
//   ml_drift::webgpu::Environment env;  // uses default backend
//   ABSL_QCHECK_OK(env.Initialize());
//
//   ml_drift::GpuModel gpu_model;
//   ABSL_QCHECK_OK(LoadTfLiteModelFile(model_file, &gpu_model));
//
//   ml_drift::webgpu::InferenceContext context;
//   ml_drift::CreateGpuModelInfo ci;
//   ABSL_QCHECK_OK(context.InitFromGpuModel(ci, &gpu_model, &env));
//
//   // write input
//   context.SetInputTensor(env, ...);
//
//   // run inference
//   ABSL_QCHECK_OK(context.AddToQueue(env));
//   ABSL_QCHECK_OK(
//       ml_drift::webgpu::WaitUntilCompleted(env.queue(), env.device()));
//
//   // read output
//   context.GetOutputTensor(...)
//
// $ bazel build --define ml_drift_api=wgpu -c opt <target>

#include "ml_drift/webgpu/execution_environment.h"  // IWYU pragma: export
#include "ml_drift/webgpu/inference_context.h"  // IWYU pragma: export
#include "ml_drift/webgpu/registry.h"         // IWYU pragma: export
#include "ml_drift/webgpu/webgpu_api_util.h"  // IWYU pragma: export
#include "ml_drift/webgpu/webgpu_headers.h"   // IWYU pragma: export

#else
// #warning "Invalid ML_DRIFT_API"
#endif

#endif  // ML_DRIFT_INCLUDE_ML_DRIFT_H_
