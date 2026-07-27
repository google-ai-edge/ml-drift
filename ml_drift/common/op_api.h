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

#ifndef ML_DRIFT_COMMON_OP_API_H_
#define ML_DRIFT_COMMON_OP_API_H_

// This header is used to include the core portions of the ML Drift operation
// framework. The comments that follow describe the main classes within
// the framework and how they interact.
//
// OpBase: A class which clients subclass to define GPU operation logic. It
// defines the attributes it expects and implements a Build method to append
// nodes into a GpuModelBuilder graph.
//
// OpRegistry: A global repository that keeps track of available operations.
// Operations can be registered in the file they are defined via the
// MLD_REGISTER_OP macro.
//
// Attrs: A data container for operation parameters (e.g., scalars, booleans)
// configured at initialization time. Attributes defined for an operation are
// validated before OpBase::Build is called.
//
// Example usage:
// namespace custom_ns {
// class MyCustomOp : public ::ml_drift::OpBase {
//  public:
//   MLD_DECLARE_OP_ATTRS(
//       MLD_OP_ATTR(int, int_arg, 3),
//       MLD_OP_ATTR(float, float_arg, 1.0f)
//   );
//
//   absl::Status Build(
//       ::ml_drift::GpuModelBuilder& graph,
//       const std::vector<::ml_drift::GpuModelBuilder::TensorHandle>& inputs,
//       const ::ml_drift::Attrs& attrs,
//       std::vector<::ml_drift::GpuModelBuilder::TensorHandle>& outputs)
//       const override {
//     // Implementation here...
//     return absl::OkStatus();
//   }
// };
//
// // Registration must happen in a namespace context where the generated
// // variable name is valid (i.e. without colons like custom_ns::MyCustomOp).
// MLD_REGISTER_OP(MyCustomOp);
// }  // namespace custom_ns

#include "ml_drift/common/op_base.h"      // IWYU pragma: export
#include "ml_drift/common/op_registry.h"  // IWYU pragma: export

#endif  // ML_DRIFT_COMMON_OP_API_H_
