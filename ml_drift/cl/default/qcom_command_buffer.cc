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

#include <cstddef>
#include <memory>

#include "ml_drift/cl/cl_context.h"
#include "ml_drift/cl/cl_device.h"
#include "ml_drift/cl/command_buffer.h"
#include "ml_drift/common/status.h"

namespace ml_drift {
namespace cl {

bool IsQcomCommandBufferSupported(const CLDevice& device) { return false; }

size_t GetQcomCommandBufferMaxSize(const CLDevice& device) { return 0; }

absl::StatusOr<std::unique_ptr<CommandBuffer>> CreateQcomCommandBuffer(
    const CLDevice& device, const CLContext& context) {
  return absl::UnimplementedError("Not supported");
}

}  // namespace cl
}  // namespace ml_drift
