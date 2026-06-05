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

#ifndef ML_DRIFT_METAL_ENVIRONMENT_H_
#define ML_DRIFT_METAL_ENVIRONMENT_H_

#import <Metal/Metal.h>

#include <string>
#include <vector>

#include "ml_drift/common/gpu_info.h"

namespace ml_drift {
namespace metal {

// A wrapper around metal device
class Environment {
 public:
  Environment();
  Environment(id<MTLDevice> device);

  Environment(Environment&& device) = default;
  Environment& operator=(Environment&& device) = default;
  Environment(const Environment&) = delete;
  Environment& operator=(const Environment&) = delete;

  ~Environment() = default;

  id<MTLDevice> device() const { return device_; }

  const GpuInfo& GetInfo() const { return info_; }

  bool IsLanguageVersion2orHigher() const;

 private:
  id<MTLDevice> device_ = nullptr;
  GpuInfo info_;
};

}  // namespace metal
}  // namespace ml_drift

#endif  // ML_DRIFT_METAL_ENVIRONMENT_H_
