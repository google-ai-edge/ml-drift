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

#ifndef ML_DRIFT_COMMON_OP_REGISTRY_H_
#define ML_DRIFT_COMMON_OP_REGISTRY_H_

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "ml_drift/common/op_base.h"

namespace ml_drift {

using OpFactory = std::function<std::unique_ptr<OpBaseImpl>()>;

// Global registry for ML Drift operations. See `op_api.h` for the full
// operation authoring guide.
class OpRegistry {
 public:
  static OpRegistry& Global();

  // Registers an operation factory.
  void Register(std::string_view name, OpFactory factory);

  // Creates an instance of an operation by name. Returns nullptr if not found.
  std::unique_ptr<OpBaseImpl> Create(std::string_view name) const;

  std::vector<std::string> GetRegisteredNames() const;

 private:
  OpRegistry() = default;

  absl::flat_hash_map<std::string, OpFactory> registry_;
};

// Statically registers an operation deriving from OpBase<OpType> (see
// `op_api.h` and `op_base.h`). Place this macro at namespace scope in the .cc
// file where the operation is implemented, after the class definition. It also
// checks the operation's contract at compile time so mistakes are reported on
// this line:
// - OpType must derive from OpBase<OpType>. OpBase<X> dispatches Build() to X,
//   so deriving from OpBase of another type would silently run that type's
//   Build().
// - OpType must declare Build() with the expected signature.
#define MLD_REGISTER_OP(OpType)                                                \
  static_assert(std::is_base_of_v<::ml_drift::OpBase<OpType>, OpType>,         \
                #OpType " must derive from ::ml_drift::OpBase<" #OpType ">");  \
  static_assert(::ml_drift::internal::HasExpectedBuild<OpType>::value, #OpType \
                " must declare a public member: absl::Status "                 \
                "Build(GpuModelBuilder&, const std::vector<TensorHandle>&, "   \
                "Attrs&, std::vector<TensorHandle>&) const. (If the op uses "  \
                "MLD_OP_ATTRS, the macro must appear before Build.)");         \
  [[maybe_unused]] static bool reg_##OpType = []() {                           \
    ::ml_drift::OpRegistry::Global().Register(                                 \
        #OpType, []() { return std::make_unique<OpType>(); });                 \
    return true;                                                               \
  }();

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_OP_REGISTRY_H_
