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

set(XNNPACK_BUILD_TESTS OFF CACHE BOOL "")
set(XNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "")

include(xnnpack)
if(xnnpack_POPULATED)
  set(_XNNPACK_LIBRARY_NAMES
    XNNPACK
  )
  set(_XNNPACK_LIBRARIES ${_XNNPACK_LIBRARY_NAMES})
  foreach(_LIBRARY ${_XNNPACK_LIBRARY_NAMES})
    list(APPEND _XNNPACK_LIBRARIES "XNNPACK::${LIBRARY}")
  endforeach()
  set(XNNPACK_LIBRARIES ${XNNPACK_LIBRARIES} CACHE STRING "XNNPACK libs")
endif()
