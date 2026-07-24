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

#include "ml_drift_delegate/tflite/custom_parsers.h"

#include <memory>

#include "absl/strings/string_view.h"
#include "ml_drift_delegate/tflite/operation_parser.h"
#include "ml_drift_delegate/tflite/unimplemented_operation_parser.h"

namespace litert::ml_drift {

std::unique_ptr<TFLiteOperationParser>
NewCustomOperationParser(absl::string_view op_name) {
  return std::make_unique<UnimplementedOperationParser>(op_name);
}

}  // namespace litert::ml_drift
