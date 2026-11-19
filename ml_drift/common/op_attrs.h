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

#ifndef ML_DRIFT_COMMON_OP_ATTRS_H_
#define ML_DRIFT_COMMON_OP_ATTRS_H_

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

class OpBaseImpl;
template <typename Derived>
class OpBase;

namespace internal {

template <typename T>
inline constexpr bool kIsShaderBindable =
    std::is_same_v<T, int32_t> || std::is_same_v<T, int> ||
    std::is_same_v<T, uint32_t> || std::is_same_v<T, unsigned int> ||
    std::is_same_v<T, float> || std::is_same_v<T, half> ||
    std::is_same_v<T, bool>;

}  // namespace internal

// Container and base class for operation attributes.
// Initialized as a key-value map at `AppendOp` call sites, and serves as the
// base class for the strongly-typed nested `Attrs` struct declared via
// `MLD_OP_ATTRS`. See `op_api.h` for the full attribute and shader
// auto-binding guide.
class OpAttrs {
 public:
  OpAttrs() = default;
  OpAttrs(std::initializer_list<std::pair<std::string, std::any>> init)
      : map_(init) {}

 protected:
  template <typename T, bool kBindToShader>
  T RegisterAttr(std::string_view name, T* member_ptr, T default_val) {
    static_assert(std::is_copy_assignable_v<T>,
                  "Attribute type must be copy-assignable.");
    void (*write_fn)(GPUOperation*, std::string_view, const void*) = nullptr;
    if constexpr (kBindToShader) {
      static_assert(internal::kIsShaderBindable<T>,
                    "Type does not support automatic shader binding; use "
                    "MLD_OP_ATTR_NO_BIND instead.");
      write_fn = &WriteShaderArg<T>;
    }
    fields_.push_back(Field{name, member_ptr, &ParseAttr<T>, write_fn});
    return default_val;
  }

 private:
  friend class OpBaseImpl;
  template <typename Derived>
  friend class OpBase;

  struct Field {
    std::string_view name;
    void* member;
    absl::Status (*parse)(void* dst, const std::any& src, std::string_view key);
    void (*write)(GPUOperation* op, std::string_view name, const void* src);
  };

  template <typename T>
  static absl::Status ParseAttr(void* dst, const std::any& src,
                                std::string_view key) {
    if (const auto* p = std::any_cast<T>(&src)) {
      *static_cast<T*>(dst) = *p;
      return absl::OkStatus();
    }
    return absl::InvalidArgumentError(
        absl::StrCat("Type mismatch for attribute: ", key));
  }

  template <typename T>
  static void WriteShaderArg(GPUOperation* op, std::string_view arg_name,
                             const void* src) {
    const T& val = *static_cast<const T*>(src);
    const std::string key(arg_name);
    if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int>) {
      op->args_.AddInt(key, val);
    } else if constexpr (std::is_same_v<T, uint32_t> ||
                         std::is_same_v<T, unsigned int>) {
      op->args_.AddUint(key, val);
    } else if constexpr (std::is_same_v<T, float>) {
      op->args_.AddFloat(key, val);
    } else if constexpr (std::is_same_v<T, half>) {
      op->args_.AddHalf(key, val);
    } else if constexpr (std::is_same_v<T, bool>) {
      op->args_.AddInt(key, val ? 1 : 0);
    }
  }

  void AttachTo(GPUOperation* op) const { attached_ops_.push_back(op); }

  void BindShaderArgs() const {
    for (GPUOperation* op : attached_ops_) {
      for (const Field& field : fields_) {
        if (field.write != nullptr) {
          field.write(op, field.name, field.member);
        }
      }
    }
  }

  // Validates and populates registered MLD_OP_ATTR fields from the key-value
  // map passed at the AppendOp call site.
  absl::Status Parse(const OpAttrs& raw_attrs) {
    std::size_t matched_count = 0;
    for (std::size_t i = 0; i < fields_.size(); ++i) {
      const Field& field = fields_[i];
      if (field.name.empty()) {
        return absl::InternalError("Attribute key must not be empty");
      }
      for (std::size_t j = 0; j < i; ++j) {
        if (fields_[j].name == field.name) {
          return absl::InternalError(
              absl::StrCat("Duplicate attribute key: ", field.name));
        }
      }
      if (auto it = raw_attrs.map_.find(field.name);
          it != raw_attrs.map_.end()) {
        absl::Status status = field.parse(field.member, it->second, field.name);
        if (!status.ok()) {
          return status;
        }
        ++matched_count;
      }
    }
    if (matched_count != raw_attrs.map_.size()) {
      for (const auto& [key, unused_value] : raw_attrs.map_) {
        const bool declared =
            std::any_of(fields_.begin(), fields_.end(),
                        [&key](const Field& f) { return f.name == key; });
        if (!declared) {
          std::vector<std::string_view> names;
          names.reserve(fields_.size());
          for (const Field& f : fields_) {
            names.push_back(f.name);
          }
          return absl::InvalidArgumentError(absl::StrCat(
              "unknown attribute \"", key, "\"; declared attributes: [",
              absl::StrJoin(names, ", "), "]"));
        }
      }
    }
    return absl::OkStatus();
  }

  absl::flat_hash_map<std::string, std::any> map_;
  std::vector<Field> fields_;
  mutable std::vector<GPUOperation*> attached_ops_;
};

// Declares the nested, strongly-typed `Attrs` struct of an operation. Invoke
// once inside a class deriving from `OpBase<Self>`, before `Build()`. See
// `op_api.h` for the full attribute and shader auto-binding guide.
//
//   MLD_OP_ATTRS(
//       MLD_OP_ATTR(int, width, 1);                     // -> args.width
//       MLD_OP_ATTR(float, scale, 1.0f);                // -> args.scale
//       MLD_OP_ATTR_NO_BIND(Axis, axis, Axis::kWidth);  // not auto-bound
//   )
#define MLD_OP_ATTRS(...)                     \
  struct Attrs : public ::ml_drift::OpAttrs { \
    Attrs() = default;                        \
    Attrs(const Attrs&) = delete;             \
    Attrs& operator=(const Attrs&) = delete;  \
    __VA_ARGS__                               \
  };

// Defines a caller-configurable attribute on `Attrs` that is automatically
// bound to `args.<name>` when `Build()` returns OK. `type` must support shader
// auto-binding (currently scalars: int/int32_t, unsigned int/uint32_t, float,
// half, bool).
#define MLD_OP_ATTR(type, name, default_val)                    \
  type name = this->RegisterAttr<type, /*kBindToShader=*/true>( \
      #name, &this->name, type{default_val})

// Defines a caller-configurable attribute on `Attrs` without automatic shader
// binding. Any copy-assignable C++ `type` is allowed.
#define MLD_OP_ATTR_NO_BIND(type, name, default_val)             \
  type name = this->RegisterAttr<type, /*kBindToShader=*/false>( \
      #name, &this->name, type{default_val})

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_OP_ATTRS_H_
