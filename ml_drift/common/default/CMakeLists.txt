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

add_library(mld_common_status_matchers INTERFACE
    status_matchers.h
)
target_include_directories(mld_common_status_matchers INTERFACE $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_common_status_matchers INTERFACE
    mld_common_status
    absl::status
    gtest
    gmock
)
