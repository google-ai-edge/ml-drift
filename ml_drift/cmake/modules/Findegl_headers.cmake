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

include(egl_headers)
if(egl_headers_POPULATED)
  set(_EGL_HEADERS_LIBRARY_NAMES
    EGL
  )
  set(_EGL_HEADERS_LIBRARIES ${_EGL_HEADERS_LIBRARY_NAMES})
  foreach(_LIBRARY ${_EGL_HEADERS_LIBRARY_NAMES})
    list(APPEND _EGL_HEADERS_LIBRARIES "egl_headers::${LIBRARY}")
  endforeach()
  set(EGL_HEADERS_LIBRARIES ${EGL_HEADERS_LIBRARIES} CACHE STRING "egl_headers libs")
endif()
