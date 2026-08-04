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

#ifndef ML_DRIFT_COMMON_TENSOR_HANDLE_H_
#define ML_DRIFT_COMMON_TENSOR_HANDLE_H_

#include <cstdint>
#include <string>

#include "absl/strings/str_cat.h"
#include "ml_drift/common/task/tensor_desc.h"

namespace ml_drift {

using ValueId = uint32_t;

struct TensorHandle {
  TensorDescriptor tensor_desc;
  ValueId id;

  std::string ShapeToString() const {
    return absl::StrCat(
        tensor_desc.GetBHWCShape().b, "x", tensor_desc.GetBHWCShape().h, "x",
        tensor_desc.GetBHWCShape().w, "x", tensor_desc.GetBHWCShape().c);
  }
};

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_TENSOR_HANDLE_H_
