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

#include "absl/status/status.h"
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
  // Returns a cached global instance. The instance object caches the adapter
  // list and can be very expensive to recreate. We only want to create this
  // once per process.
  static const wgpu::Instance& Get();

  // Sets the instance to be returned by Get(). When ML-Drift is duplicated in
  // multiple shared libraries, a singleton instance must be set for each shared
  // library to use it.
  // Note that it is not thread-safe on purpose assuming it must be called
  // before any call to Get() in each shared library.
  static absl::Status Set(const wgpu::Instance& instance);

  static absl::Status Wait(wgpu::Future future, absl::Duration timeout);
  static void ProcessEvents();
};

}  // namespace webgpu
}  // namespace ml_drift

#endif  // ML_DRIFT_WEBGPU_INSTANCE_H_
