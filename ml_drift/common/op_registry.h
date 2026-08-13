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
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "ml_drift/common/op_base.h"

namespace ml_drift {

using OpFactory = std::function<std::unique_ptr<OpBase>()>;

// A global registry for ML Drift operations.
class OpRegistry {
 public:
  static OpRegistry& Global();

  // Registers an operation factory and its attribute specifications.
  void Register(std::string_view name, OpFactory factory,
                const std::vector<AttrSpec>& attr_specs);

  // Creates an instance of an operation by name. Returns nullptr if not found.
  std::unique_ptr<OpBase> Create(std::string_view name) const;

  // Retrieves the expected attributes for a registered operation.
  absl::StatusOr<std::vector<AttrSpec>> GetAttrSpecs(
      std::string_view name) const;

  std::vector<std::string> GetRegisteredNames() const;

 private:
  OpRegistry() = default;

  struct Entry {
    OpFactory factory;
    std::vector<AttrSpec> attr_specs;
  };

  absl::flat_hash_map<std::string, Entry> registry_;
};

// Macro to statically register an operation extending OpBase (see op_base.h).
// Add this macro to the .cc file where the operation is implemented.
#define MLD_REGISTER_OP(OpType)                               \
  [[maybe_unused]] static bool reg_##OpType = []() {          \
    ::ml_drift::OpRegistry::Global().Register(                \
        #OpType, []() { return std::make_unique<OpType>(); }, \
        OpType().GetAttrSpecs());                             \
    return true;                                              \
  }();

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_OP_REGISTRY_H_
