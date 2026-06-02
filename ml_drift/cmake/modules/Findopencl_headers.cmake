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

# grpc uses find_package in CONFIG mode for this package, so override the
# system installation and build from source instead.
include(opencl_headers)
if(opencl_headers_POPULATED)
  set(_OPENCL_HEADERS_LIBRARY_NAMES
    OpenCL
  )
  set(_OPENCL_HEADERS_LIBRARIES ${_OPENCL_HEADERS_LIBRARY_NAMES})
  foreach(_LIBRARY ${_OPENCL_HEADERS_LIBRARY_NAMES})
    list(APPEND _OPENCL_HEADERS_LIBRARIES "opencl_headers::${LIBRARY}")
  endforeach()
  set(OPENCL_HEADERS_LIBRARIES ${OPENCL_HEADERS_LIBRARIES} CACHE STRING "opencl_headers libs")
endif()
