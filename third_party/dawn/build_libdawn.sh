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

set_chromium_version() {
  export CHROMIUM_VERSION=147.0.7714.2
  echo $CHROMIUM_VERSION
}
export CMAKE_VERSION=3.23.5

# If we source this script, don't run it. We would do this if we only want
# the chromium_version.
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  return 0
fi

set -ex

if [ "$(uname)" = "Darwin" ]; then
  if ! cmake --version 2>/dev/null | grep -q ${CMAKE_VERSION}; then
    brew install cmake
  fi
elif [ "$(expr substr $(uname -s) 1 5)" = "Linux" ]; then
  sudo apt-get -qq update || true
  sudo apt-get -qq install -y --no-install-recommends wget make \
    libc++-dev libxrandr-dev libxinerama-dev libxcursor-dev mesa-common-dev \
    libx11-xcb-dev pkg-config libx11-dev libxi-dev libxext-dev
  if ! ${HOME}/.local/bin/cmake --version 2>/dev/null | grep -q ${CMAKE_VERSION}; then
    (
      # Install cmake locally to hardcode version
      mkdir -p $HOME/.local
      cd $(mktemp -d)
      wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh
      chmod +x cmake-${CMAKE_VERSION}-Linux-x86_64.sh
      ./cmake-${CMAKE_VERSION}-Linux-x86_64.sh --skip-license --prefix=$HOME/.local --exclude-subdir
    )
  fi
fi

module_file="$(pwd -P)"/MODULE.bazel
mkdir -p /tmp/dawn
cd /tmp/dawn
git clone https://github.com/jspanchu/webgpu-dawn-binaries.git

rm webgpu-dawn-binaries/chromium_version.txt
touch webgpu-dawn-binaries/chromium_version.txt

set_chromium_version
if [ "$(uname)" = "Darwin" ]; then
  echo $CHROMIUM_VERSION | tr -d '\n' >> webgpu-dawn-binaries/chromium_version.txt
elif [ "$(expr substr $(uname -s) 1 5)" = "Linux" ]; then
  echo -n $CHROMIUM_VERSION >> webgpu-dawn-binaries/chromium_version.txt
fi

mkdir -p webgpu-dawn-binaries/out/latest/include
cd webgpu-dawn-binaries/out/latest

if [ "$(uname)" = "Darwin" ]; then
  cmake ../..
  make -j 32
elif [ "$(expr substr $(uname -s) 1 5)" = "Linux" ]; then
  ${HOME}/.local/bin/cmake ../..
  /usr/bin/make -j 32
fi

cp -R _deps/dawn-src/include/dawn include
cp -R _deps/dawn-build/gen/include/dawn include
cp -R _deps/dawn-src/include/webgpu include
cp -R _deps/dawn-build/gen/include/webgpu include

echo "dawn_ext = use_extension('//third_party/dawn:dawn.bzl', 'local_non_bazel_repo')" >> $module_file
if [ "$(uname)" = "Darwin" ]; then
  echo "dawn_ext.local_repo(repo_name = \"dawn\", build_file = \"//third_party/dawn:dawn_macos.BUILD\", path = \"/tmp/dawn/webgpu-dawn-binaries/out/latest\")" >> $module_file
elif [ "$(expr substr $(uname -s) 1 5)" = "Linux" ]; then
  echo "dawn_ext.local_repo(repo_name = \"dawn\", build_file = \"//third_party/dawn:dawn_linux.BUILD\", path = \"/tmp/dawn/webgpu-dawn-binaries/out/latest\")" >> $module_file
fi
echo "use_repo(dawn_ext, 'dawn')" >> $module_file
