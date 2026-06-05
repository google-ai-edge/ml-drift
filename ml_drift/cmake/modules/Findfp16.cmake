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

include(fp16)
if(fp16_POPULATED)
  set(_FP16_LIBRARY_NAMES
    fp16
  )
  set(_FP16_LIBRARIES ${_FP16_LIBRARY_NAMES})
  foreach(_LIBRARY ${_FP16_LIBRARY_NAMES})
    list(APPEND _FP16_LIBRARIES "FP16::${LIBRARY}")
  endforeach()
  set(FP16_LIBRARIES ${FP16_LIBRARIES} CACHE STRING "FP16 libs")
endif()
