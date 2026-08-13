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

#ifndef ML_DRIFT_WEBGPU_INSTANCE_H_
#define ML_DRIFT_WEBGPU_INSTANCE_H_

#include "ml_drift/common/status.h"
#include "absl/time/time.h"
#include "ml_drift/webgpu/webgpu_headers.h"

namespace ml_drift {
namespace webgpu {

// Converts wgpu::WaitStatus into absl::Status.
absl::Status VerifyWaitStatus(wgpu::WaitStatus status);

// Provides entry points into a cached global wgpu::Instance. The Instance
// object caches the adapter list and can be very expensive to recreate. We only
// want to create this once per process.
class Instance {
 public:
  struct WebGpuFlushCallback {
    // A custom callback function invoked during synchronous buffer readback.
    void (*callback)(void* user_data) = nullptr;
    // An optional opaque pointer passed directly into `callback` when invoked.
    void* user_data = nullptr;
  };

  // Returns a cached global instance. The instance object caches the adapter
  // list and can be very expensive to recreate. We only want to create this
  // once per process.
  static const wgpu::Instance& Get();
  static wgpu::Instance Get(const wgpu::Device& device);

  // Sets the instance to be returned by Get(). When ML-Drift is duplicated in
  // multiple shared libraries, a singleton instance must be set for each shared
  // library to use it.
  //
  // Ownership & Ref-Counting:
  // Storing `instance` increments Dawn's reference count (+1). Callers do NOT
  // need to keep their original handle alive after calling Set(); ML-Drift
  // retains its own reference for process/module duration.
  //
  // WARNING: Overwriting a global singleton introduces risks of data races and
  // undefined behavior. This function is NOT thread-safe and should NOT be used
  // by new clients before refactoring the singleton initialization. It must
  // only be called sequentially during initialization before any concurrent
  // threads invoke Get(), when there is no race condition.
  // TODO(crbug.com/524317888) - Remove this pattern and refactor singleton
  // initialization.
  static absl::Status Set(const wgpu::Instance& instance);

  static absl::Status Wait(wgpu::Future future, absl::Duration timeout);
  static absl::Status Wait(const wgpu::Device& device, wgpu::Future future,
                           absl::Duration timeout);
  static void ProcessEvents();

  // Sets an optional callback (`WebGpuFlushCallback`) to be invoked during
  // synchronous WebGPU buffer readback for a specific WebGPU device.
  //
  // When to set this callback:
  // - Required when ML-Drift runs over an inter-process or client-server WebGPU
  //   transport layer, such as Chromium's Renderer process
  //   (`WebNNLiteRTInRenderer`) using Dawn Wire over Mojo IPC.
  // - When running directly on native GPU drivers (e.g., inside Chrome's GPU
  //   process or standalone benchmark tools), this callback is not needed and
  //   should remain `nullptr`.
  //
  // When it is used:
  // - Invoked by `MaybeRunFlushCallback()` inside `ReadDataFromMappableBuffer`
  //   while polling a `wgpu::Future` in a synchronous wait loop.
  // - In out-of-process architectures, the callback flushes outbound wire
  //   commands across the IPC channel (`dawn_control_client_->Flush()`) and
  //   pumps inbound IPC messages on the thread message loop (`base::RunLoop`)
  //   so that the GPU process's readback response is delivered to Dawn.
  static void SetFlushCallback(WGPUDevice device,
                               const WebGpuFlushCallback* callback);
  // Releases any flush callback registered for `device` from the global
  // registry. This should be called when the WebGPU environment or device is
  // being destroyed.
  static void ReleaseFlushCallback(WGPUDevice device);
  // Invokes the flush callback associated with `device`. This is called
  // during synchronous CPU readback loops (`ReadDataFromMappableBuffer`) before
  // calling `WaitAny()` to prevent deadlocks over IPC channels.
  static void MaybeRunFlushCallback(WGPUDevice device);
};

}  // namespace webgpu
}  // namespace ml_drift

#endif  // ML_DRIFT_WEBGPU_INSTANCE_H_
