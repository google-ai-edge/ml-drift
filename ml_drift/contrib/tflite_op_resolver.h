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

#ifndef ML_DRIFT_CONTRIB_TFLITE_OP_RESOLVER_H_
#define ML_DRIFT_CONTRIB_TFLITE_OP_RESOLVER_H_

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/op_resolver.h"

namespace ml_drift {
namespace contrib {

// Registers the custom TfLite operations avoiding implementation.
class TfLiteOpResolver
    : public tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates {
 public:
  TfLiteOpResolver();
};

void RegisterCustomOps(::tflite::MutableOpResolver* resolver);

}  // namespace contrib
}  // namespace ml_drift

#endif  // ML_DRIFT_CONTRIB_TFLITE_OP_RESOLVER_H_
