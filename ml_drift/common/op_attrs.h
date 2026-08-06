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

#ifndef ML_DRIFT_COMMON_ATTRS_H_
#define ML_DRIFT_COMMON_ATTRS_H_

#include <any>
#include <initializer_list>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace ml_drift {

// Map-based attributes system for operation parameters.
// This container holds uniforms and configuration for ops at graph
// construction time.
class OpAttrs {
 public:
  OpAttrs() = default;
  OpAttrs(std::initializer_list<std::pair<std::string, std::any>> init)
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

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_ATTRS_H_
