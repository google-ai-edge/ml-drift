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

// Umbrella header and authoring guide for the ML Drift operation framework.
//
// Core Components
// ---------------
// - OpBase<Self> (op_base.h): CRTP base class for custom GPU operations.
//   Subclasses implement `Build()` to stage one or more kernels onto a
//   `GpuModelBuilder` graph via helpers such as `AppendNewOp`:
//     absl::Status Build(GpuModelBuilder& graph,
//                        const std::vector<TensorHandle>& inputs,
//                        Attrs& attrs,
//                        std::vector<TensorHandle>& outputs) const;
//
// - OpAttrs / MLD_OP_ATTRS (op_attrs.h): Strongly-typed graph-construction
//   parameters declared inside an op class, with automatic shader binding for
//   supported types. Ops without attributes omit `MLD_OP_ATTRS` (`Attrs` then
//   aliases the empty `OpAttrs`).
//
// - OpRegistry / MLD_REGISTER_OP (op_registry.h): Global operation registry.
//   Place `MLD_REGISTER_OP(MyOp)` at namespace scope in the `.cc` file after
//   the class definition; it also verifies `Build()` at compile time.
//
// - GpuModelBuilder::AppendOp (gpu_model_builder.h): Caller entry point that
//   instantiates a registered op, parses its attributes, and runs `Build()`:
//     graph.AppendOp("MyOp", {x}, {{"scale", 0.5f}, {"axis", Axis::kWidth}});
//
// Attributes & Shader Auto-Binding
// --------------------------------
// Declare attributes once inside the op class (before `Build()`) using
// `MLD_OP_ATTRS(...)` and `MLD_OP_ATTR(T, name, default_val)`:
//
//   Declaration                        Caller key   Shader `args.<name>`
//   ---------------------------------  -----------  -------------------------
//   MLD_OP_ATTR(T, name, default)      "name"       auto-bound (bindable T)
//   MLD_OP_ATTR(T, _name, default)     "name"       not bound  (any C++ T)
//
// - Shader auto-binding: Passing `attrs` to `AppendNewOp` attaches it to that
//   kernel. Un-prefixed attributes are automatically bound to `args.<name>` in
//   the shader. Currently supported auto-bound types `T` are scalars:
//     - `int`/`int32_t`
//     - `unsigned int`/`uint32_t`
//     - `float`
//     - `half`
//     - `bool` (uploaded as `int` 0/1)
// - Host-only (`_` prefix): Prefix `name` with `_` to opt out of shader
//   binding while keeping the `"name"` key for `AppendOp` callers. Types
//   without shader auto-bind support must be prefixed with `_`.
// - Validation: Omitted keys take their declared defaults. Unknown
//   keys or type mismatches fail `AppendOp` with `InvalidArgumentError` before
//   `Build()` runs.
// - Mutable attributes: `attrs` can be modified in `Build()`; any edits to
//   auto-bound fields take effect because final values are bound to
//   `args.<name>` upon successful return of `Build()`.
//
// Lifecycle of an AppendOp Call
// -----------------------------
//   AppendOp(name, inputs, {key: value, ...})
//     -> OpRegistry instantiates MyOp
//     -> MyOp::Attrs is constructed and parsed from the caller's map
//     -> MyOp::Build(graph, inputs, attrs, outputs) runs
//     -> auto-bound fields on `attrs` are written to attached GPUOperations
//     -> outputs are returned to the caller
//
// Example Usage
// -------------
// #include "ml_drift/common/op_api.h"
// #include "ml_drift/common/shape.h"
//
// namespace custom_ns {
// class MyCustomOp : public ::ml_drift::OpBase<MyCustomOp> {
//  public:
//   MLD_OP_ATTRS(
//       MLD_OP_ATTR(float, scale, 1.0f);  // "scale" -> shader args.scale
//       MLD_OP_ATTR(::ml_drift::Axis, _axis,
//                   ::ml_drift::Axis::kWidth);  // "axis" -> host-only
//   )
//
//   absl::Status Build(
//       ::ml_drift::GpuModelBuilder& graph,
//       const std::vector<::ml_drift::GpuModelBuilder::TensorHandle>& inputs,
//       Attrs& attrs,
//       std::vector<::ml_drift::GpuModelBuilder::TensorHandle>& outputs)
//       const {
//     std::string read_coords;
//     if (attrs._axis == ::ml_drift::Axis::kWidth) {
//       read_coords = "args.dst.Width() - 1 - X, Y, S";
//     } else if (attrs._axis == ::ml_drift::Axis::kHeight) {
//       read_coords = "X, args.dst.Height() - 1 - Y, S";
//     } else {
//       return absl::UnimplementedError("Unsupported axis.");
//     }
//
//     auto* op = AppendNewOp(graph, attrs);
//     AddSrcTensor(graph, op, "src", inputs[0]);
//     outputs = graph.AddTensors({inputs[0].tensor_desc});
//     AddDstTensor(graph, op, "dst", outputs[0]);
//
//     const auto& dst_shape = outputs[0].tensor_desc.GetBHWCShape();
//     op->SetGridSize(int3(dst_shape.w * dst_shape.b, dst_shape.h,
//                          (dst_shape.c + 3) / 4));
//     op->code_ =
//         "MAIN_FUNCTION($0) {\n"
//         "  int X = ucl::GetGlobalId<0>();\n"
//         "  int Y = ucl::GetGlobalId<1>();\n"
//         "  int S = ucl::GetGlobalId<2>();\n"
//         "  if (X >= args.dst.Width() || Y >= args.dst.Height() ||\n"
//         "      S >= args.dst.Slices()) return;\n"
//         "  args.dst::type val = args.src.Read(" + read_coords + ");\n"
//         "  val *= ucl::Convert<args.dst::scalar_type>(args.scale);\n"
//         "  args.dst.Write(val, X, Y, S);\n"
//         "}";
//     return absl::OkStatus();
//   }
// };
//
// // Registration must happen in a namespace context where the generated
// // variable name is valid (i.e. without colons like custom_ns::MyCustomOp).
// MLD_REGISTER_OP(MyCustomOp);
// }  // namespace custom_ns

#include "absl/status/status.h"         // IWYU pragma: export
#include "ml_drift/common/data_type.h"  // IWYU pragma: export
#include "ml_drift/common/gpu_model_builder.h"  // IWYU pragma: export
#include "ml_drift/common/op_attrs.h"     // IWYU pragma: export
#include "ml_drift/common/op_base.h"      // IWYU pragma: export
#include "ml_drift/common/op_registry.h"  // IWYU pragma: export
#include "ml_drift/common/task/gpu_operation.h"  // IWYU pragma: export
#include "ml_drift/common/types.h"  // IWYU pragma: export

#endif  // ML_DRIFT_COMMON_OP_API_H_
