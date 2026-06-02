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

set -ev

# Package installs
apt-get -qq update
apt-get -qq install -y --no-install-recommends \
    autoconf \
    automake \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libcurl4-openssl-dev \
    libtool \
    libssl-dev \
    openjdk-8-jdk \
    openjdk-8-jre-headless \
    pkg-config \
    python-setuptools \
    python3-virtualenv \
    python3-dev \
    python3-setuptools \
    rename \
    rsync \
    sudo \
    swig \
    unzip \
    vim \
    wget \
    zip \
    zlib1g-dev \
    > /dev/null 2>&1

# Kokoro network proxy certificate handling.
if [[ -f /var/cache/proxy.crt ]]; then
  cp /var/cache/proxy.crt /usr/local/share/ca-certificates/kokoro-proxy.crt
fi

# Update CA certificates to include any newly added certificates.
update-ca-certificates -f > /dev/null 2>&1

# Export environment variables for tools to use the correct trust store.
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Install cmake. Used only to compile tflite.
(
  mkdir -p "$HOME/.local"
  cd "$(mktemp -d)"
  wget --no-verbose https://github.com/Kitware/CMake/releases/download/v3.23.5/cmake-3.23.5-Linux-x86_64.sh
  chmod +x cmake-3.23.5-Linux-x86_64.sh
  ./cmake-3.23.5-Linux-x86_64.sh --skip-license --prefix="$HOME/.local" --exclude-subdir
)
if [[ ! ":$PATH:" =~ :"$HOME"/.local/bin/?: ]]; then
  PATH="$HOME/.local/bin:$PATH"
fi

apt-get clean
export TF_PYTHON_VERSION=3.11

# Install bazel
mkdir -p "$HOME/bin"
wget --no-verbose -O "$HOME/bin/bazel" \
    "https://github.com/bazelbuild/bazelisk/releases/download/v1.27.0/bazelisk-linux-amd64"
chmod u+x "$HOME/bin/bazel"
if [[ ! ":$PATH:" =~ :"$HOME"/bin/?: ]]; then
  PATH="$HOME/bin:$PATH"
fi
mkdir -p /tmpfs/bazel_output
export TEST_TMPDIR=/tmpfs/bazel_output
which bazel
bazel version
date
