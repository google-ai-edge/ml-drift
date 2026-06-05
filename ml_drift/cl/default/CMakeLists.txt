# Copyright 2026 The ML Drift Authors.
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

add_library(mld_cl_default_util util.cc)
target_include_directories(mld_cl_default_util PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_cl_default_util PUBLIC
  mld_cl_util_types
  mld_common_gpu_info
  mld_common_status
  mld_cl_opencl_wrapper
  absl::status
)

add_library(mld_cl_default_qcom_command_buffer qcom_command_buffer.cc)
target_include_directories(mld_cl_default_qcom_command_buffer PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_cl_default_qcom_command_buffer PUBLIC
  mld_cl_cl_context
  mld_cl_cl_device
  mld_cl_command_buffer
  mld_common_status
)
