#!/bin/bash
# Copyright 2025 The ML Drift Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Install webgpu
source third_party/dawn/cached_dawn.sh

# Build all non-kernel webgpu tests
bazel build -c opt --test_output=errors ml_drift/webgpu/...

# Build webgpu sample
bazel build -c opt --config=android_arm64 ml_drift/samples/stable_diffusion:sd_gpu_webgpu

# Build all webgpu kernel tests.
bazel build -c opt --test_output=errors ml_drift/common/kernels/tests:all_tests_webgpu
