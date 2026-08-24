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

#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/debugging/leak_check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "ml_drift/webgpu/webgpu_headers.h"

namespace ml_drift {
namespace webgpu {
namespace {

const wgpu::Instance* g_instance = nullptr;

// A thread-safe, reference-counted registry mapping hardware device handles
// (`WGPUDevice`) to their corresponding out-of-process flush callbacks.
//
// In client-server architectures (such as client using Dawn Wire IPC), separate
// WebGPU environments or execution contexts may operate over independent wire
// channels or hardware handles. Keying by device guarantees that synchronous
// CPU buffer readback loops (`MaybeRunFlushCallback`) pump wire messages and
// events on the specific IPC transport layer associated with that exact device
// handle without colliding with or deadlocking neighboring GPU execution
// channels.
//
// Multiple LiteRT subgraphs or model instances may share the same underlying
// WebGPU device handle. Maintaining a reference count (`ref_count`) ensures the
// callback remains actively registered as long as at least one dependent model
// or environment is alive, and is cleanly erased when all consumers release it
// upon destruction.
struct FlushRegistryEntry {
  Instance::WebGpuFlushCallback cb;
  int ref_count = 0;
};

absl::Mutex g_flush_mutex(absl::kConstInit);
absl::flat_hash_map<WGPUDevice, FlushRegistryEntry>& GetFlushMap() // NOLINT
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(g_flush_mutex) {
  static auto* map = new absl::flat_hash_map<WGPUDevice, FlushRegistryEntry>();
  return *map;
}

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

absl::Status Instance::Wait(wgpu::Future future, absl::Duration timeout) {
  return Wait(nullptr, future, timeout);
}

wgpu::Instance Instance::Get(const wgpu::Device& device) {
#if !defined(__EMSCRIPTEN__)
  if (device && device.GetAdapter() && device.GetAdapter().GetInstance()) {
    return device.GetAdapter().GetInstance();
  }
#endif  // !defined(__EMSCRIPTEN__)
  // If failed to get the instance from the device, return the singleton
  // instance.
  return Get();
}

absl::Status Instance::Set(const wgpu::Instance& instance) {
  if (!g_instance) {
    g_instance = absl::IgnoreLeak(new wgpu::Instance(instance));
  } else {
    *const_cast<wgpu::Instance*>(g_instance) = instance;
  }
  return absl::OkStatus();
}

absl::Status Instance::Wait(const wgpu::Device& device, wgpu::Future future,
                            absl::Duration timeout) {
  wgpu::FutureWaitInfo wait_info = {};
  wait_info.future = future;
  wgpu::WaitStatus status =
      Get(device).WaitAny(1u, &wait_info, absl::ToInt64Nanoseconds(timeout));
  if (status == wgpu::WaitStatus::TimedOut) {
    return absl::DeadlineExceededError(
        absl::StrCat("Timed out waiting for future: ", timeout));
  }
  return VerifyWaitStatus(status);
}

void Instance::ProcessEvents() { Get().ProcessEvents(); }

void Instance::SetFlushCallback(WGPUDevice device,
                                const WebGpuFlushCallback* callback) {
  if (callback && callback->callback) {
    absl::MutexLock lock(g_flush_mutex);
    auto& entry = GetFlushMap()[device];
    entry.cb = *callback;
    entry.ref_count++;
  }
}

void Instance::ReleaseFlushCallback(WGPUDevice device) {
  absl::MutexLock lock(g_flush_mutex);
  auto& map = GetFlushMap();
  auto it = map.find(device);
  if (it != map.end() && --it->second.ref_count <= 0) {
    map.erase(it);
  }
}

void Instance::MaybeRunFlushCallback(WGPUDevice device) {
  Instance::WebGpuFlushCallback cb{nullptr, nullptr};
  {
    absl::MutexLock lock(g_flush_mutex);
    const auto& map = GetFlushMap();
    auto it = map.find(device);
    if (it != map.end()) {
      cb = it->second.cb;
    }
  }
  if (cb.callback) {
    cb.callback(cb.user_data);
  }
}

}  // namespace webgpu
}  // namespace ml_drift
