// Copyright 2026 The ML Drift Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ML_DRIFT_GL_GL_CALL_H_
#define ML_DRIFT_GL_GL_CALL_H_

#include <string>
#include <type_traits>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/gl/gl_errors.h"

namespace ml_drift {
namespace gl {

// Primary purpose of this file is to provide useful macro for calling GL
// functions and checking errors. It also attaches a context to status in case
// of a GL error.
//
// Use ML_DRIFT_CALL_GL as follows:
//
//   For GL functions with a return value:
//     Before:
//       GLint result = glFunc(...);
//       ABSL_RETURN_IF_ERROR(GetOpenGlErrors());
//     After:
//       GLint result;
//       ABSL_RETURN_IF_ERROR(ML_DRIFT_CALL_GL(glFunc, &result, ...));
//
//   For GL functions without a return value:
//     Before:
//       glFunc(...);
//       ABSL_RETURN_IF_ERROR(GetOpenGlErrors());
//     After:
//       ABSL_RETURN_IF_ERROR(ML_DRIFT_CALL_GL(glFunc, ...));

namespace gl_call_internal {

// For GL functions with a return value.
template <typename T>
struct Caller {
  template <typename F, typename ErrorF, typename... Params>
  absl::Status operator()(const std::string& context, F func, ErrorF error_func,
                          T* result, Params&&... params) {
    *result = func(std::forward<Params>(params)...);
    const auto status = error_func();
    if (status.ok()) return absl::OkStatus();
    return absl::Status(status.code(),
                        std::string(status.message()) + ": " + context);
  }
};

// For GL functions without a return value.
template<>
struct Caller<void> {
  template <typename F, typename ErrorF, typename... Params>
  absl::Status operator()(const std::string& context, F func, ErrorF error_func,
                          Params&&... params) {
    func(std::forward<Params>(params)...);
    const auto status = error_func();
    if (status.ok()) return absl::OkStatus();
    return absl::Status(status.code(),
                        std::string(status.message()) + ": " + context);
  }
};

template <typename F, typename ErrorF, typename ResultT, typename... ParamsT>
absl::Status CallAndCheckError(const std::string& context, F func,
                               ErrorF error_func, ResultT* result,
                               ParamsT&&... params) {
  return Caller<ResultT>()(context, func, error_func, result,
                           std::forward<ParamsT>(params)...);
}

template <typename F, typename ErrorF, typename... Params>
absl::Status CallAndCheckError(const std::string& context, F func,
                               ErrorF error_func, Params&&... params) {
  return Caller<void>()(context, func, error_func,
                        std::forward<Params>(params)...);
}

}  // namespace gl_call_internal

// XX_STRINGIFY is a helper macro to effectively apply # operator to an
// arbitrary value.
#define ML_DRIFT_INTERNAL_STRINGIFY_HELPER(x) #x
#define ML_DRIFT_INTERNAL_STRINGIFY(x) ML_DRIFT_INTERNAL_STRINGIFY_HELPER(x)
#define ML_DRIFT_FILE_LINE __FILE__ ":" ML_DRIFT_INTERNAL_STRINGIFY(__LINE__)

#define ML_DRIFT_CALL_GL(method, ...)                  \
  ::ml_drift::gl::gl_call_internal::CallAndCheckError( \
      #method " in " ML_DRIFT_FILE_LINE, method,       \
      ::ml_drift::gl::GetOpenGlErrors, __VA_ARGS__)

#define ML_DRIFT_CALL_EGL(method, ...)                 \
  ::ml_drift::gl::gl_call_internal::CallAndCheckError( \
      #method " in " ML_DRIFT_FILE_LINE, method,       \
      ::ml_drift::gl::GetEglError, __VA_ARGS__)

}  // namespace gl
}  // namespace ml_drift

#endif  // ML_DRIFT_GL_GL_CALL_H_
