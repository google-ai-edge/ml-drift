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

source third_party/dawn/build_libdawn.sh
set_chromium_version

DAWN_TARBALL="${KOKORO_GFILE_DIR}/webgpu-dawn-binaries.tar.gz"
CACHED_CHROMIUM_VERSION=""

# Try to grab cached dawn binary if it exists
if [ -f "$DAWN_TARBALL" ]; then
  echo "Found cached Dawn tarball at $DAWN_TARBALL"
  mkdir -p /tmp/dawn
  tar -xf "$DAWN_TARBALL" -C /tmp/dawn
  # Read the chromium version if extraction succeeded
  if [ -f "/tmp/dawn/webgpu-dawn-binaries/chromium_version.txt" ]; then
     CACHED_CHROMIUM_VERSION=$(cat /tmp/dawn/webgpu-dawn-binaries/chromium_version.txt)
  fi
else
  echo "No cached Dawn tarball found."
fi

echo "Cached chromium version: ${CACHED_CHROMIUM_VERSION}"

# Check chromium version
# if match, add to module file, return
# if not (or if cache was missing), call build_libdawn
if [[ -n "${CACHED_CHROMIUM_VERSION}" && "${CACHED_CHROMIUM_VERSION}" == "${CHROMIUM_VERSION}" ]]; then
  echo "Cached chromium version matches current dawn version. Using cached dawn binary."
  module_file="$(pwd -P)"/MODULE.bazel
  echo "dawn_ext = use_extension('//third_party/dawn:dawn.bzl', 'local_non_bazel_repo')" >> $module_file
  if [ "$(uname)" = "Darwin" ]; then
    echo "dawn_ext.local_repo(repo_name = \"dawn\", build_file = \"//third_party/dawn:dawn_macos.BUILD\", path = \"/tmp/dawn/webgpu-dawn-binaries/out/latest\")" >> $module_file
  elif [ "$(expr substr $(uname -s) 1 5)" = "Linux" ]; then
    echo "dawn_ext.local_repo(repo_name = \"dawn\", build_file = \"//third_party/dawn:dawn_linux.BUILD\", path = \"/tmp/dawn/webgpu-dawn-binaries/out/latest\")" >> $module_file
  fi
  echo "use_repo(dawn_ext, 'dawn')" >> $module_file
else
  if [ -z "${CACHED_CHROMIUM_VERSION}" ]; then
     echo "Cached dawn version not found. Building dawn binaries."
  else
     echo "Cached dawn version ($CACHED_CHROMIUM_VERSION) does not match current ($CHROMIUM_VERSION). Rebuilding."
  fi

  rm -rf /tmp/dawn
  ./third_party/dawn/build_libdawn.sh
fi
