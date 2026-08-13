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

#include "ml_drift/common/op_registry.h"

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ml_drift/common/op_base.h"

namespace ml_drift {

OpRegistry& OpRegistry::Global() {
  static OpRegistry* instance = new OpRegistry();
  return *instance;
}

void OpRegistry::Register(std::string_view name, OpFactory factory,
                          const std::vector<AttrSpec>& attr_specs) {
  registry_[name] = {std::move(factory), attr_specs};
}

absl::StatusOr<std::vector<AttrSpec>> OpRegistry::GetAttrSpecs(
    std::string_view name) const {
  auto it = registry_.find(name);
  if (it == registry_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Operation not found in registry: ", name));
  }
  return it->second.attr_specs;
}

std::unique_ptr<OpBase> OpRegistry::Create(std::string_view name) const {
  auto it = registry_.find(name);
  return it != registry_.end() ? it->second.factory() : nullptr;
}

std::vector<std::string> OpRegistry::GetRegisteredNames() const {
  std::vector<std::string> names;
  names.reserve(registry_.size());
  for (const auto& [name, entry] : registry_) {
    names.push_back(name);
  }
  std::sort(names.begin(), names.end());
  return names;
}

}  // namespace ml_drift
