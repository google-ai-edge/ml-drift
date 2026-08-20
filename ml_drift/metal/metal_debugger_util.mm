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

#import "third_party/ml_drift/metal/metal_debugger_util.h"

namespace ml_drift {
namespace metal {

absl::Status StartMetalCapture(id<MTLDevice> device, const std::string& output_file_path) {
  @autoreleasepool {
    MTLCaptureManager* captureManager = [MTLCaptureManager sharedCaptureManager];
    if (![captureManager supportsDestination:MTLCaptureDestinationGPUTraceDocument]) {
      return absl::FailedPreconditionError(
          "Capturing to a GPU trace file isn't supported (did you forget MTL_CAPTURE_ENABLED=1?).");
    }
    if ([captureManager isCapturing]) {
      return absl::AlreadyExistsError("Metal capture is already in progress.");
    }

    MTLCaptureDescriptor* descriptor = [[MTLCaptureDescriptor alloc] init];
    descriptor.captureObject = device;
    descriptor.destination = MTLCaptureDestinationGPUTraceDocument;

    NSString* ns_trace_path = [NSString stringWithUTF8String:output_file_path.c_str()];
    descriptor.outputURL = [NSURL fileURLWithPath:ns_trace_path];

    NSError* error = nil;
    if (![captureManager startCaptureWithDescriptor:descriptor error:&error]) {
      std::string error_msg = error ? [[error localizedDescription] UTF8String] : "Unknown error";
      return absl::InternalError("Failed to start Metal capture: " + error_msg);
    }
    return absl::OkStatus();
  }
}

void StopMetalCapture() {
  @autoreleasepool {
    [[MTLCaptureManager sharedCaptureManager] stopCapture];
  }
}

}  // namespace metal
}  // namespace ml_drift
