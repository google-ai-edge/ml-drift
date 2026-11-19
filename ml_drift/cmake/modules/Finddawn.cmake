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

include(dawn)
if(DAWN_FOUND)
  set(_DAWN_LIBRARY_NAMES
    dawn
  )
  set(_DAWN_LIBRARIES ${_DAWN_LIBRARY_NAMES})
  foreach(_LIBRARY ${_DAWN_LIBRARY_NAMES})
    list(APPEND _DAWN_LIBRARIES "Dawn::${LIBRARY}")
  endforeach()
  set(DAWN_LIBRARIES ${DAWN_LIBRARIES} CACHE STRING "Dawn libs")
  set(DAWN_INCLUDE_DIRS ${DAWN_INCLUDE_DIR} CACHE STRING "Dawn include dirs")
endif()
