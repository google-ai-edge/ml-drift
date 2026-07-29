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

#ifndef ML_DRIFT_COMMON_IR_MODEL_H_
#define ML_DRIFT_COMMON_IR_MODEL_H_

#include <any>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/tensor_desc.h"

// Defines an intermediate representation (IR) for a computational graph.
// This IR is used internally for graph construction and manipulation and is not
// intended for direct use by end-users.

namespace ml_drift::ir {

using IrOpId = size_t;
using IrTensorId = size_t;

// Represents an instance of an operation (a graph node).
struct IrOp {
  explicit IrOp(IrOpId id) : id(id) {}
  std::string str() const;

  IrOpId id;
  std::string name;
  std::any attr;
  std::vector<IrTensorId> inputs;
  std::vector<IrTensorId> outputs;
};

struct IrQuantParams {
  float min = 0.f;
  float max = 0.f;
  float scale = 0.f;
};

// Describes the origin of a tensor's backing bytes. Used for shared constants,
// whose bytes are shared with the caller rather than owned by the graph.
struct BufferSource {
  // True if the tensor is a shared constant.
  bool is_shared = false;
  // Global buffer id identifying the shared buffer in the shared-memory
  // manager. Only meaningful when is_shared is true.
  int64_t global_id = -1;
  // If true, the shared constant must be dequantized to float before being
  // shared. Set by op converters for ops that cannot consume quantized shared
  // weights (e.g. Convolution2D). Only meaningful when is_shared is true.
  bool dequant_forced = false;
  // If true, the shared constant must be materialized with LINEAR layout by the
  // shared-memory manager. Set by op converters for shared bias tensors (parity
  // with GraphFloat32). Only meaningful when is_shared is true.
  bool force_linear_layout = false;
};

// Represents a tensor of data that flows along the graph's edges.
struct IrTensor {
  explicit IrTensor(IrTensorId id) : id(id) {}
  std::string str() const;

  IrTensorId id;
  TensorDescriptor desc;  // shape, layout, dtype
  BufferSource buffer_source;
  std::optional<IrQuantParams> quant_params;
  std::optional<IrOpId> producer;
  absl::flat_hash_set<IrOpId> consumers;
};

// A container and factory for building the computational graph.
// This class owns the IrOp and IrTensor objects and manages their lifetimes.
class IrModel {
 public:
  // Accessors/mutators for ops_.
  // May contain nullptr entries (tombstones) for deleted objects. Callers must
  // check for null before dereferencing.

  // Creates a new IrOp and returns a pointer to it.
  IrOp* add_op() {
    return ops_.emplace_back(std::make_unique<IrOp>(ops_.size())).get();
  }

  // Returns the vector of all IrOps (including tombstones).
  const std::vector<std::unique_ptr<IrOp>>& ops() const { return ops_; }

  // Returns the IrOp with the given ID, or nullptr if the ID is out of bounds
  // or is a tombstone.
  const IrOp* op(IrOpId id) const {
    return id < ops_.size() ? ops_.at(id).get() : nullptr;
  }

  // Accessors/mutators for tensors_.
  // May contain nullptr entries (tombstones) for deleted objects. Callers must
  // check for null before dereferencing.

  // Creates a new IrTensor and returns a pointer to it.
  // Storage type is set during IrModel->GpuModel conversion.
  template <::ml_drift::Layout L>
  IrTensor* add_tensor(const ::ml_drift::DataType& dtype,
                       const ::ml_drift::StrongShape<L>& shape) {
    IrTensorId id = tensors_.size();
    tensors_.push_back(std::make_unique<IrTensor>(id));

    ::ml_drift::BHWDC bhwdc = ToBHWDC(shape);
    const bool has_b = bhwdc.b > 1;
    const bool has_d = bhwdc.d > 1;
    ::ml_drift::Layout layout;
    if (has_b && has_d) {
      layout = ::ml_drift::Layout::BHWDC;
    } else if (has_b) {
      layout = ::ml_drift::Layout::BHWC;
    } else if (has_d) {
      layout = ::ml_drift::Layout::HWDC;
    } else {
      layout = ::ml_drift::Layout::HWC;
    }

    // Storage type is set during IrModel->GpuModel conversion
    tensors_[id]->desc =
        ::ml_drift::TensorDescriptor(dtype, /*storage_type=*/{}, layout);
    tensors_[id]->desc.SetBHWDCShape(bhwdc);
    return tensors_[id].get();
  }

  // Clones an existing tensor descriptor to create a new tensor.
  IrTensor* add_tensor(const ::ml_drift::TensorDescriptor& desc);

  // Returns the vector of all IrTensors (including tombstones).
  const std::vector<std::unique_ptr<IrTensor>>& tensors() const {
    return tensors_;
  }

  // Returns the IrTensor with the given ID, or nullptr if the ID is out of
  // bounds or is a tombstone.
  const IrTensor* tensor(IrTensorId id) const {
    return id < tensors_.size() ? tensors_.at(id).get() : nullptr;
  }

  // Mutable accessor for op converters that enrich tensor metadata in place
  // (e.g. shape, buffer_source), mirroring GraphFloat32::GetValue. Prefer the
  // const `tensor()` for reads; topology changes must go through
  // SetProducer/AddConsumer/ReplaceInput rather than mutating here.
  IrTensor* GetMutableTensor(IrTensorId id) {
    return id < tensors_.size() ? tensors_.at(id).get() : nullptr;
  }

  // Other mutators & accessors.
  void add_input(IrTensorId input) { inputs_.push_back(input); }
  const std::vector<IrTensorId>& inputs() const { return inputs_; }
  void add_output(IrTensorId output) { outputs_.push_back(output); }
  const std::vector<IrTensorId>& outputs() const { return outputs_; }

  // Sets the producer for a given tensor. A tensor can only have one producer.
  // This method also updates the producer IrOp's `outputs` list.
  void SetProducer(IrTensorId tensor_id, IrOpId op_id);

  // Adds a consumer for a given tensor. A tensor can have multiple consumers.
  // This method also updates the consumer IrOp's `inputs` list.
  void AddConsumer(IrTensorId tensor_id, IrOpId op_id);

  // Finds the producer for a given tensor.
  IrOp* FindProducer(IrTensorId tensor_id) const;

  // Finds all consumers for a given tensor.
  std::vector<IrOp*> FindConsumers(IrTensorId tensor_id) const;

  bool IsGraphInput(IrTensorId tensor_id) const;
  bool IsGraphOutput(IrTensorId tensor_id) const;

  // Graph manipulation helpers

  // Replaces an input tensor for a given op with a new tensor.
  absl::Status ReplaceInput(IrOpId op_id, IrTensorId old_tensor_id,
                            IrTensorId new_tensor_id);

  // Clears quantization parameters for a given tensor.
  void ResetQuantParams(IrTensorId tensor_id);

  // Removes an op that has exactly one input and one output.
  // It automatically routes consumers/producers to bypass the operation.
  // By default, the output tensor is deleted. However, if the output tensor
  // is a graph output, the input tensor is deleted instead.
  // Returns OkStatus if the operation was successfully removed.
  absl::Status RemoveSimpleOp(IrOpId op_id);

  // Returns a mutable pointer to the attribute of the specified op, if it
  // exists and the type matches. Otherwise returns nullptr. This should be used
  // safely, ensuring topology conditions are verified before mutation.
  template <typename T>
  T* GetMutableAttr(IrOpId op_id) {
    if (op_id < ops_.size() && ops_[op_id] != nullptr) {
      return std::any_cast<T>(&ops_[op_id]->attr);
    }
    return nullptr;
  }

 private:
  // A nullptr entry acts as a tombstone: the object was deleted, but the entry
  // is kept to reserve the ID and prevent its immediate reuse.
  std::vector<std::unique_ptr<IrOp>> ops_;
  std::vector<std::unique_ptr<IrTensor>> tensors_;

  // Tensors defining the model's external interface (e.g., inputs and outputs).
  std::vector<IrTensorId> inputs_;
  std::vector<IrTensorId> outputs_;
  std::vector<IrTensorId> variable_inputs_;

  // Helper to expand StrongShape to BHWDC.
  template <::ml_drift::Layout L>
  static ::ml_drift::BHWDC ToBHWDC(const ::ml_drift::StrongShape<L>& shape) {
    auto pad = [](int32_t val) { return val < 0 ? 1 : val; };
    return ::ml_drift::BHWDC(pad(shape.get(::ml_drift::Axis::BATCH)),
                             pad(shape.get(::ml_drift::Axis::HEIGHT)),
                             pad(shape.get(::ml_drift::Axis::WIDTH)),
                             pad(shape.get(::ml_drift::Axis::DEPTH)),
                             pad(std::max(shape.get(::ml_drift::Axis::CHANNELS),
                                          shape.get(::ml_drift::Axis::VALUE))));
  }
};

typedef absl::flat_hash_map<int, IrTensorId> TensorMap;

}  // namespace ml_drift::ir

#endif  // ML_DRIFT_COMMON_IR_MODEL_H_
