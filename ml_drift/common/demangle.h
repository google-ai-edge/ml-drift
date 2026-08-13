// Copyright 2026 The ML Drift Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ML_DRIFT_COMMON_DEMANGLE_H_
#define ML_DRIFT_COMMON_DEMANGLE_H_

#ifndef ML_DRIFT_HAS_CXA_DEMANGLE
// We only support some compilers that support __cxa_demangle.
#if defined(__ANDROID__) && (defined(__i386__) || defined(__x86_64__))
#define ML_DRIFT_HAS_CXA_DEMANGLE 0
#elif (__GNUC__ >= 4 || (__GNUC__ >= 3 && __GNUC_MINOR__ >= 4)) && \
    !defined(__mips__)
#define ML_DRIFT_HAS_CXA_DEMANGLE 1
#elif defined(__clang__) && !defined(_MSC_VER)
#define ML_DRIFT_HAS_CXA_DEMANGLE 1
#else
#define ML_DRIFT_HAS_CXA_DEMANGLE 0
#endif
#endif

#include <cstdlib>
#include <string>

#if ML_DRIFT_HAS_CXA_DEMANGLE
#include <cxxabi.h>  // IWYU pragma: keep
#endif

namespace ml_drift {

// Demangle a mangled symbol name and return the demangled name.
// If 'mangled' isn't mangled in the first place, this function
// simply returns 'mangled' as is.
//
// This function is used for demangling mangled symbol names such as
// '_Z3bazifdPv'.  It uses abi::__cxa_demangle() if your compiler has
// the API.  Otherwise, this function simply returns 'mangled' as is.
//
// Currently, we support only GCC 3.4.x or later for the following
// reasons.
//
// - GCC 2.95.3 doesn't have cxxabi.h
// - GCC 3.3.5 and ICC 9.0 have a bug.  Their abi::__cxa_demangle()
//   returns junk values for non-mangled symbol names (ex. function
//   names in C linkage).  For example,
//     abi::__cxa_demangle("main", 0,  0, &status)
//   returns "unsigned long" and the status code is 0 (successful).
//
// Also,
//
//  - MIPS is not supported because abi::__cxa_demangle() is not defined.
//  - Android x86 is not supported because STLs don't define __cxa_demangle
//
// Prefer using DrishtiTypeStringOrDemangled<T>() when possible (defined
// in type_map.h).
inline std::string Demangle(const char* mangled) {
  int status = 0;
  char* demangled = nullptr;
#if ML_DRIFT_HAS_CXA_DEMANGLE
  demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
#endif
  std::string out;
  if (status == 0 && demangled != nullptr) {  // Demangling succeeded.
    out.append(demangled);
    free(demangled);
  } else {
    out.append(mangled);
  }
  return out;
}

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_DEMANGLE_H_
