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

#include "ml_drift/webgpu/instance.h"

#include <vector>

#include "absl/debugging/leak_check.h"
#include "ml_drift/common/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "ml_drift/webgpu/webgpu_headers.h"

namespace ml_drift {
namespace webgpu {
namespace {

const wgpu::Instance* g_instance = nullptr;

wgpu::Instance CreateInstance() {
  wgpu::InstanceDescriptor instance_desc = {};
  static const auto kTimedWaitAny = wgpu::InstanceFeatureName::TimedWaitAny;
  instance_desc.requiredFeatureCount = 1;
  instance_desc.requiredFeatures = &kTimedWaitAny;

#ifndef __EMSCRIPTEN__
  std::vector<const char*> enabled_toggles;
  enabled_toggles.push_back("allow_unsafe_apis");

  wgpu::DawnTogglesDescriptor toggles_desc;
  toggles_desc.enabledToggles = enabled_toggles.data();
  toggles_desc.enabledToggleCount = enabled_toggles.size();
  instance_desc.nextInChain = &toggles_desc;
#endif  // __EMSCRIPTEN__

  return wgpu::CreateInstance(&instance_desc);
}

const wgpu::Instance* GetInstaceCreatedIfNecessary() {
  if (!g_instance) {
    g_instance = absl::IgnoreLeak(new wgpu::Instance(CreateInstance()));
  }
  return g_instance;
}

}  // namespace

absl::Status VerifyWaitStatus(wgpu::WaitStatus status) {
  switch (status) {
    case wgpu::WaitStatus::Success:
      return absl::OkStatus();
    case wgpu::WaitStatus::TimedOut:
      return absl::InternalError("wgpu::WaitStatus::TimedOut");
    case wgpu::WaitStatus::Error:
      return absl::InternalError("wgpu::WaitStatus::Error");
  }
}

const wgpu::Instance& Instance::Get() {
  static const auto* instance = GetInstaceCreatedIfNecessary();
  return *instance;
}

absl::Status Instance::Set(const wgpu::Instance& instance) {
  if (g_instance) {
    return absl::AlreadyExistsError("wgpu::Instance already set");
  }
  g_instance = &instance;
  return absl::OkStatus();
}

absl::Status Instance::Wait(wgpu::Future future, absl::Duration timeout) {
  wgpu::FutureWaitInfo wait_info = {};
  wait_info.future = future;
  wgpu::WaitStatus status =
      Get().WaitAny(1u, &wait_info, absl::ToInt64Nanoseconds(timeout));
  if (status == wgpu::WaitStatus::TimedOut) {
    return absl::DeadlineExceededError(
        absl::StrCat("Timed out waiting for future: ", timeout));
  }
  return VerifyWaitStatus(status);
}

void Instance::ProcessEvents() { Get().ProcessEvents(); }

}  // namespace webgpu
}  // namespace ml_drift
