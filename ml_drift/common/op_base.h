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

#ifndef ML_DRIFT_COMMON_OP_BASE_H_
#define ML_DRIFT_COMMON_OP_BASE_H_

#include <any>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/tensor_handle.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class Arguments;
class GpuModelBuilder;

// Map-based attributes system for operation parameters.
// This container holds uniforms and configuration for ops at graph
// construction time.
class Attrs {
 public:
  Attrs() = default;
  Attrs(std::initializer_list<std::pair<std::string, std::any>> init)
      : map_(init) {}

  auto find(const std::string& key) const { return map_.find(key); }

  template <typename T>
  absl::StatusOr<T> Get(const std::string& key) const {
    auto it = map_.find(key);
    if (it == map_.end()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Missing required attribute: ", key));
    }

    const T* val_ptr = std::any_cast<T>(&it->second);
    if (val_ptr == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("Type mismatch for attribute: ", key));
    }
    return *val_ptr;
  }

  bool contains(const std::string& key) const { return map_.contains(key); }

  const std::any& at(const std::string& key) const { return map_.at(key); }

  std::any& operator[](const std::string& key) { return map_[key]; }

  void insert(const std::pair<std::string, std::any>& val) { map_.insert(val); }

  auto begin() const { return map_.begin(); }
  auto end() const { return map_.end(); }

 private:
  absl::flat_hash_map<std::string, std::any> map_;
};

// Schema definition for a single attribute, defining its required type
// and fallback default value if omitted.
struct AttrSpec {
  std::string name;
  const std::type_info* type_info;
  std::any default_value;
};

// Macro for defining attribute specifications for an operation.
#define MLD_DECLARE_OP_ATTRS(...)                                 \
  std::vector<ml_drift::AttrSpec> GetAttrSpecs() const override { \
    return {__VA_ARGS__};                                         \
  }

// Macro for defining a single attribute specification for an operation.
// Attributes of type {int32_t, float, half, bool} are automatically bound to
// the operation arguments, except for those beginning with an underscore.
#define MLD_OP_ATTR(type, name, default_val)   \
  ml_drift::AttrSpec {                         \
    #name, &typeid(type), type { default_val } \
  }

// [Internal] Binds attributes matching standard primitive types to the given
// GPUOperation argument list, making them accessible to the shader.
inline void BindAutoAttrs(GPUOperation* op, const Attrs& attrs,
                          const std::vector<AttrSpec>& specs) {
  for (const auto& spec : specs) {
    if (!spec.name.empty() && spec.name[0] != '_') {
      auto it = attrs.find(spec.name);
      if (it != attrs.end()) {
        const auto& attr_var = it->second;
        if (auto* ptr = std::any_cast<int32_t>(&attr_var)) {
          op->args_.AddInt(spec.name, *ptr);
        } else if (auto* ptr = std::any_cast<float>(&attr_var)) {
          op->args_.AddFloat(spec.name, *ptr);
        } else if (auto* ptr = std::any_cast<half>(&attr_var)) {
          op->args_.AddHalf(spec.name, *ptr);
        } else if (auto* ptr = std::any_cast<bool>(&attr_var)) {
          op->args_.AddInt(spec.name, *ptr ? 1 : 0);
        }
      }
    }
  }
}

// [Internal] Validates the provided attributes against their specifications.
// Populates missing attributes with their default values and checks for
// type mismatch errors.
inline absl::Status ValidateAndNormalizeAttrs(
    const std::vector<AttrSpec>& specs, Attrs& attrs) {
  for (const auto& spec : specs) {
    if (!attrs.contains(spec.name)) {
      attrs.insert({spec.name, spec.default_value});
    } else {
      if (attrs.at(spec.name).type() != *spec.type_info) {
        return absl::InvalidArgumentError(
            absl::StrCat("Type mismatch for attribute: ", spec.name));
      }
    }
  }
  return absl::OkStatus();
}

// Base interface for operations. Inherit from this class to register an
// operation in ML Drift (see op_api.h).
class OpBase {
 public:
  virtual ~OpBase() = default;

  // Defines the expected attributes and their default values.
  // By default, assumes an operation requires no attributes.
  //
  // Note: It is recommended to use the MLD_DECLARE_OP_ATTRS macro in the
  // derived class to define the attribute specifications.
  virtual std::vector<AttrSpec> GetAttrSpecs() const { return {}; }

  // Constructs the operation in the provided GpuModelBuilder graph.
  // inputs:  The tensor handles and shapes for the operation inputs, to be
  //          validated by the operation.
  // attrs:   The attributes for the operation. At the time of calling Build,
  //          the attributes are validated and normalized.
  // outputs: The output tensor handles and shapes, to be populated by the
  //          operation.
  virtual absl::Status Build(GpuModelBuilder& graph,
                             const std::vector<TensorHandle>& inputs,
                             const Attrs& attrs,
                             std::vector<TensorHandle>& outputs) const = 0;
};

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_OP_BASE_H_
