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

#ifndef ML_DRIFT_METAL_METAL_DEBUGGER_UTIL_H_
#define ML_DRIFT_METAL_METAL_DEBUGGER_UTIL_H_

#import <Metal/Metal.h>

#include <string>

#include "absl/status/status.h"

namespace ml_drift {
namespace metal {

// To use Metal Debugger for performance analysis, use this utility to capture a
// Metal workload programmatically. The API will generate a `.gputrace` file and
// it can be opened in Xcode's Metal Debugger.
//
// Usage Example:
//
//   // 1. Start the capture before the commands you want to profile.
//   auto status = ml_drift::metal::StartMetalCapture(
//       device, "/tmp/my_model_capture.gputrace");
//
//   // 2. Dispatch compute commands, encode buffers, and commit.
//   // (Metal captures the encoded commands; you do not need to wait for GPU
//   execution). [command_buffer commit];
//
//   // 3. Stop the capture to finalize the trace file.
//   ml_drift::metal::StopMetalCapture();
//
// Scope: This API captures commands only within `MTLCommandBuffer` objects that
// you create after the capture starts and commit before the capture stops.
//
// Note: You must run your executable with the environment variable
// `MTL_CAPTURE_ENABLED=1` for this API to work.
absl::Status StartMetalCapture(id<MTLDevice> device,
                               const std::string& output_file_path);
void StopMetalCapture();

}  // namespace metal
}  // namespace ml_drift

#endif  // ML_DRIFT_METAL_METAL_DEBUGGER_UTIL_H_
