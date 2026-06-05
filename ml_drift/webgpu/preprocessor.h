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

#ifndef ML_DRIFT_WEBGPU_PREPROCESSOR_H_
#define ML_DRIFT_WEBGPU_PREPROCESSOR_H_

#include <string>
#include <vector>

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/status.h"

namespace ml_drift {
namespace webgpu {

struct ExtensionsInfo {
  // extensions with keyword "enable"
  std::vector<std::string> enable_extensions;
  // extensions with keyword "requires"
  std::vector<std::string> language_extensions;
};
absl::Status ConvertToWGSL(const WebGpuInfo& webgpu_info, std::string* code,
                           ExtensionsInfo* extensions_info);

}  // namespace webgpu
}  // namespace ml_drift

#endif  // ML_DRIFT_WEBGPU_PREPROCESSOR_H_
