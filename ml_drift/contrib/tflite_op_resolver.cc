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

#include "ml_drift/contrib/tflite_op_resolver.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/op_resolver.h"

namespace ml_drift {
namespace contrib {

namespace {

// The 'name' parameter must point to a C string with static storage duration.
TfLiteRegistration* RegisterButNotImplement(const char* name) {
  static TfLiteRegistration reg = {
      /*.init=*/nullptr,
      /*.free=*/nullptr,
      /*.prepare=*/nullptr,
      /*.invoke=*/nullptr,
      /*.profiling_string=*/nullptr,
      /*.builtin_code=*/0,
      /*.custom_name=*/name,
  };
  return &reg;
}

}  // namespace

void RegisterCustomOps(::tflite::MutableOpResolver* resolver) {
  resolver->AddCustom("custom_call.GroupNorm",
                      RegisterButNotImplement("custom_call.GroupNorm"));
  resolver->AddCustom("custom_call.LayerNorm",
                      RegisterButNotImplement("custom_call.LayerNorm"));
  resolver->AddCustom("custom_call.RmsNorm",
                      RegisterButNotImplement("custom_call.RmsNorm"));
  resolver->AddCustom("custom_call.PixelShuffle",
                      RegisterButNotImplement("custom_call.PixelShuffle"));
  // TODO: Decide on which namespace to keep.
  resolver->AddCustom(
      "odml.scaled_dot_product_attention",
      RegisterButNotImplement("odml.scaled_dot_product_attention"));
  resolver->AddCustom(
      "custom_call.scaled_dot_product_attention",
      RegisterButNotImplement("custom_call.scaled_dot_product_attention"));
  resolver->AddCustom(
      "custom_call.absolute_positional_embedding",
      RegisterButNotImplement("custom_call.absolute_positional_embedding"));
  resolver->AddCustom(
      "custom_call.rotary_positional_embedding",
      RegisterButNotImplement("custom_call.rotary_positional_embedding"));
  resolver->AddCustom("cros-mtk-pre-compile",
                      RegisterButNotImplement("cros-mtk-pre-compile"));
}

TfLiteOpResolver::TfLiteOpResolver() {
  ::ml_drift::contrib::RegisterCustomOps(this);
}

}  // namespace contrib
}  // namespace ml_drift
