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

#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "ml_drift/common/demangle.h"
#include "ml_drift/common/op_attrs.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/tensor_handle.h"

namespace ml_drift {

class GpuModelBuilder;

template <typename Derived>
class OpBase;

// Non-templated base interface used by OpRegistry and GpuModelBuilder.
class OpBaseImpl {
 public:
  virtual ~OpBaseImpl() = default;

  virtual absl::Status InvokeBuild(
      GpuModelBuilder& graph, const std::vector<TensorHandle>& inputs,
      const OpAttrs& raw_attrs, std::vector<TensorHandle>& outputs) const = 0;

 protected:
  // Appends a new operation to the provided GpuModelBuilder graph. This helper:
  // - Instantiates the underlying GPUOperation
  // - Infers the operation name
  // - Attaches `attrs` so auto-bound shader arguments are populated when
  //   Build() succeeds
  // - Stages it in the graph's pending execution context
  //
  // graph: Model builder graph to append the new operation to.
  // attrs: Configuration attributes for the operation.
  // returns: Pointer to the newly allocated and staged GPUOperation.
  template <typename BuilderType>
  GPUOperation* AppendNewOp(BuilderType& graph, const OpAttrs& attrs) const {
    auto op_ptr = std::make_unique<GPUOperation>();
    auto* raw_op = op_ptr.get();

    std::string op_name = ml_drift::Demangle(typeid(*this).name());

    attrs.AttachTo(raw_op);

    graph.AddGpuOperation(std::vector<ValueId>{}, std::vector<ValueId>{},
                          std::move(op_ptr), op_name);

    return raw_op;
  }
  template <typename BuilderType>
  void AddSrcTensor(BuilderType& graph, GPUOperation* op, std::string_view name,
                    const TensorHandle& handle) const {
    graph.AddSrcTensor(op, name, handle);
  }

  template <typename BuilderType>
  void AddSrcBuffer(BuilderType& graph, GPUOperation* op, std::string_view name,
                    const TensorHandle& handle, int element_size = 1) const {
    graph.AddSrcBuffer(op, name, handle, element_size);
  }

  template <typename BuilderType>
  void AddDstTensor(BuilderType& graph, GPUOperation* op, std::string_view name,
                    const TensorHandle& handle) const {
    graph.AddDstTensor(op, name, handle);
  }

  template <typename BuilderType>
  void AddDstBuffer(BuilderType& graph, GPUOperation* op, std::string_view name,
                    const TensorHandle& handle, int element_size = 1) const {
    graph.AddDstBuffer(op, name, handle, element_size);
  }

 private:
  template <typename Derived>
  friend class OpBase;

  OpBaseImpl() = default;
};

namespace internal {

// True iff `Derived` declares exactly
//   absl::Status Build(GpuModelBuilder&, const std::vector<TensorHandle>&,
//                      Derived::Attrs&, std::vector<TensorHandle>&) const;
// as an accessible member. Used by MLD_REGISTER_OP to report a wrong Build
// signature at the registration line.
template <typename Derived, typename = void>
struct HasExpectedBuild : std::false_type {};

template <typename Derived>
struct HasExpectedBuild<Derived, std::void_t<decltype(&Derived::Build)>>
    : std::is_same<decltype(&Derived::Build),
                   absl::Status (Derived::*)(
                       GpuModelBuilder&, const std::vector<TensorHandle>&,
                       typename Derived::Attrs&, std::vector<TensorHandle>&)
                       const> {};

}  // namespace internal

// Base class for operations, used via CRTP. Derive as
// `class MyOp : public OpBase<MyOp>`, optionally declare attributes with
// MLD_OP_ATTRS(...), implement Build() below, and register the op with
// MLD_REGISTER_OP(MyOp). See op_api.h for a complete example.
//
//   absl::Status Build(GpuModelBuilder& graph,
//                      const std::vector<TensorHandle>& inputs,
//                      Attrs& attrs,
//                      std::vector<TensorHandle>& outputs) const;
//
// See `op_api.h` for the full operation authoring guide, attribute rules, and
// usage example.
template <typename Derived>
class OpBase : public OpBaseImpl {
 public:
  // Attribute type for ops that do not use MLD_OP_ATTRS. A nested
  // `struct Attrs` declared in Derived hides this alias.
  using Attrs = OpAttrs;

  absl::Status InvokeBuild(GpuModelBuilder& graph,
                           const std::vector<TensorHandle>& inputs,
                           const OpAttrs& raw_attrs,
                           std::vector<TensorHandle>& outputs) const final {
    // This body is instantiated once Derived is complete (when its vtable is
    // emitted in MLD_REGISTER_OP), so this names Derived's nested struct if
    // one exists.
    using DerivedAttrs = typename Derived::Attrs;
    DerivedAttrs attrs;
    absl::Status status = attrs.Parse(raw_attrs);
    if (!status.ok()) {
      return absl::Status(
          status.code(),
          absl::StrCat(ml_drift::Demangle(typeid(Derived).name()), ": ",
                       status.message()));
    }
    status =
        static_cast<const Derived*>(this)->Build(graph, inputs, attrs, outputs);
    if (status.ok()) {
      attrs.BindShaderArgs();
    }
    return status;
  }
};

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_OP_BASE_H_
