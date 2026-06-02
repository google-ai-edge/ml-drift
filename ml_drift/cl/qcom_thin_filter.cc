// Copyright 2024 The ML Drift Authors.
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

#include "ml_drift/cl/qcom_thin_filter.h"

#include <utility>

#include "absl/status/status.h"
#include "ml_drift/cl/cl_context.h"
#include "ml_drift/cl/gpu_object.h"
#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/cl/util.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/qcom_thin_filter_desc.h"

namespace ml_drift {
namespace cl {

QcomThinFilter::QcomThinFilter(QcomThinFilter&& filter)
    : filter_(filter.filter_) {
  filter.filter_ = nullptr;
}

QcomThinFilter& QcomThinFilter::operator=(QcomThinFilter&& filter) {
  if (this != &filter) {
    Release();
    std::swap(filter_, filter.filter_);
  }
  return *this;
}

absl::Status QcomThinFilter::CreateFromDescriptor(
    const QcomThinFilterDescriptor& desc, CLContext* context) {
  return CreateQcomConvolutionFilter(context->context(), desc.kernel_size_x,
                                     desc.kernel_size_y, &filter_,
                                     desc.data.data());
}

absl::Status QcomThinFilter::GetGPUResources(
    const GPUObjectDescriptor* obj_ptr,
    GPUResourcesWithValue* resources) const {
  const auto* filter_desc =
      dynamic_cast<const QcomThinFilterDescriptor*>(obj_ptr);
  if (!filter_desc) {
    return absl::InvalidArgumentError(
        "Expected QcomThinFilterDescriptor on input.");
  }
  resources->custom_memories.push_back({"filter", filter_});
  return absl::OkStatus();
}

QcomThinFilter::~QcomThinFilter() { Release(); }

void QcomThinFilter::Release() {
  if (filter_) {
    clReleaseMemObject(filter_);
    filter_ = nullptr;
  }
}

}  // namespace cl
}  // namespace ml_drift
