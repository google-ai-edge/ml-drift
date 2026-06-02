# Copyright 2025 The ML Drift Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(ExternalProject)
ExternalProject_Add(
    tensorflow
    GIT_REPOSITORY "https://github.com/tensorflow/tensorflow.git"
    GIT_TAG "v2.19.1"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
)

ExternalProject_Get_Property(tensorflow source_dir)
message(STATUS "TensorFlow source_dir: ${source_dir}")

add_library(tflite_schema INTERFACE)
target_include_directories(tflite_schema INTERFACE "${source_dir}")
add_dependencies(tflite_schema tensorflow)
