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

# Navigate to the head directory of ml_drift.
cd "$(dirname "$0")"  # go to ml_drift/kokoro/android_cmake/
cd ../..  # go to ml_drift/

source kokoro/common.sh

export MLD_HOME=$(pwd)

# Android SDK and NDK root directory workaround. For details see
# https://github.com/bazelbuild/bazel/issues/714#issuecomment-166735874
export ANDROID_DEV_HOME="/android"
mkdir -p "${ANDROID_DEV_HOME}"

# Install Android sdk
export ANDROID_SDK_FILENAME="tools_r25.2.5-linux.zip"
export ANDROID_SDK_URL="https://dl.google.com/android/repository/${ANDROID_SDK_FILENAME}"
export ANDROID_API_LEVEL=23
export ANDROID_NDK_API_LEVEL=21
export ANDROID_BUILD_TOOLS_VERSION=28.0.0
export ANDROID_SDK_HOME="${ANDROID_DEV_HOME}/sdk"
export PATH="${PATH}:${ANDROID_SDK_HOME}/tools:${ANDROID_SDK_HOME}/platform-tools"
cd ${ANDROID_DEV_HOME}
wget -q ${ANDROID_SDK_URL}
unzip -q ${ANDROID_SDK_FILENAME} -d android-sdk-linux
rm ${ANDROID_SDK_FILENAME}
bash -c "ln -s ${ANDROID_DEV_HOME}/android-sdk-* ${ANDROID_SDK_HOME}"

# Install Android NDK
export ANDROID_NDK_FILENAME="android-ndk-r28b-linux.zip"
export ANDROID_NDK_URL="https://dl.google.com/android/repository/${ANDROID_NDK_FILENAME}"
export ANDROID_NDK_HOME=${ANDROID_DEV_HOME}/ndk
export PATH=${PATH}:${ANDROID_NDK_HOME}
cd ${ANDROID_DEV_HOME}
wget -q ${ANDROID_NDK_URL}
unzip -q ${ANDROID_NDK_FILENAME} -d ${ANDROID_DEV_HOME}
rm ${ANDROID_NDK_FILENAME}
bash -c "ln -s ${ANDROID_DEV_HOME}/android-ndk-* ${ANDROID_NDK_HOME}"

chmod -R go=u ${ANDROID_DEV_HOME}
cd ${MLD_HOME}

# Install dependencies for CMake.
echo "Installing CMake..."
sudo apt-get update -y
sudo apt-get install -y cmake ninja-build
# Verify installation
cmake --version
ninja --version

# Build all opencl targets with CMake.
cmake ml_drift/ --preset android-arm64-release-static
cmake --build build/android-release-static/
