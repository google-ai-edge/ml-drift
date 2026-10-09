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

add_library(mld_samples_stable_diffusion_ops_temb_generation
    temb_generation.cc
)
target_include_directories(mld_samples_stable_diffusion_ops_temb_generation PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_samples_stable_diffusion_ops_temb_generation
  PUBLIC
    absl::status
    absl::status_macros
    flatbuffers
    mld_common_op_api
    mld_common_shape
    mld_common_task_tensor_desc
    mld_common_tensor
)

add_library(mld_samples_stable_diffusion_ops_diffusion_step
    diffusion_step.cc
)
target_include_directories(mld_samples_stable_diffusion_ops_diffusion_step PUBLIC $<BUILD_INTERFACE:${ML_DRIFT_PARENT_DIR}>)
target_link_libraries(mld_samples_stable_diffusion_ops_diffusion_step
  PUBLIC
    absl::status
    absl::strings
    flatbuffers
    mld_common_op_api
)
