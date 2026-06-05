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

#include "ml_drift/common/kernels/special/thin_pointwise_fuser.h"

#include <algorithm>
#include <any>
#include <cstdint>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/flops_util.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/kernels/elementwise.h"
#include "ml_drift/common/kernels/prelu.h"
#include "ml_drift/common/kernels/relu.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/compiler_options.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/util.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace {
// Multiply-Accumulate
std::string MAC(const GpuInfo& gpu_info, const std::string& accum,
                const std::string& a, const std::string& b) {
  const bool use_fma = gpu_info.IsAMD() && gpu_info.IsApiOpenCl();
  if (use_fma) {
    return accum + " = fma(" + a + ", " + b + ", " + accum + ")";
  } else {
    return accum + " += " + a + " * " + b;
  }
}

bool IsConvKernelXis1(const Convolution2DAttributes& conv_attr) {
  const OHWI& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, conv_attr.weights);
  return weights_shape.w == 1 && conv_attr.dilations.w == 1 &&
         conv_attr.strides.w == 1 && conv_attr.padding.prepended.w == 0 &&
         conv_attr.padding.appended.w == 0;
}
bool IsConvKernelYis1(const Convolution2DAttributes& conv_attr) {
  const OHWI& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, conv_attr.weights);
  return weights_shape.h == 1 && conv_attr.dilations.h == 1 &&
         conv_attr.strides.h == 1 && conv_attr.padding.prepended.h == 0 &&
         conv_attr.padding.appended.h == 0;
}

bool IsConv1x1(const Convolution2DAttributes& conv_attr) {
  return IsConvKernelXis1(conv_attr) && IsConvKernelYis1(conv_attr);
}

int GetConvWeightsCount(const Convolution2DAttributes& attr) {
  const OHWI& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  int conv_src_ch_aligned = AlignByN(weights_shape.i, 4);
  int conv_dst_ch_aligned = AlignByN(weights_shape.o, 4);
  return conv_dst_ch_aligned + conv_src_ch_aligned * conv_dst_ch_aligned *
                                   weights_shape.w * weights_shape.h;
}

int GetConvWeightsSize(const Convolution2DAttributes& attr,
                       DataType data_type) {
  return GetConvWeightsCount(attr) * SizeOf(data_type);
}

int GetDepthwiseConvWeightsCount(const DepthwiseConvolution2DAttributes& attr) {
  const OHWI& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  int dw_dst_ch_aligned = AlignByN(weights_shape.i, 4);
  return dw_dst_ch_aligned +
         dw_dst_ch_aligned * weights_shape.h * weights_shape.w;
}

int GetDepthwiseConvWeightsSize(const DepthwiseConvolution2DAttributes& attr,
                                DataType data_type) {
  return GetDepthwiseConvWeightsCount(attr) * SizeOf(data_type);
}

bool IsElementwiseOneInput(const OperationType& op_type) {
  return op_type == OperationType::ABS || op_type == OperationType::COPY ||
         op_type == OperationType::COS || op_type == OperationType::ELU ||
         op_type == OperationType::EXP || op_type == OperationType::GELU ||
         op_type == OperationType::HARD_SWISH ||
         op_type == OperationType::LOG || op_type == OperationType::NEG ||
         op_type == OperationType::RSQRT || op_type == OperationType::SIGMOID ||
         op_type == OperationType::SIN || op_type == OperationType::SQRT ||
         op_type == OperationType::SQUARE || op_type == OperationType::TANH;
}
}  // namespace

class ThinPointwiseFuser {
 public:
  void Init(const GraphFloat32* graph, const GpuModelBuilder* model_builder,
            DataType data_type, const std::set<NodeId>& consumed_nodes);
  absl::Status Finalize(const GpuInfo& gpu_info,
                        GpuModelBuilder* model_builder);

  absl::Status ReserveNode(const GpuInfo& gpu_info, Node* node);

  std::set<NodeId> GetFusedNodes() const {
    std::set<NodeId> fused_nodes;
    for (const auto& node : nodes_) {
      fused_nodes.insert(node->id);
    }
    return fused_nodes;
  }

 private:
  bool IsNodeSupported(const GpuInfo& gpu_info, Node* node) const;
  bool IsElementwiseNode(Node* node) const;
  bool IsConvNode(Node* node) const;
  bool IsDwConvNode(Node* node) const;
  uint64_t GetNodeFlops(Node* node) const;
  // node_index for std::vector<Node*> nodes_
  absl::Status AddNode(const GpuInfo& gpu_info, int node_index);
  void AddElementwiseNode(ElementwiseDescriptor&& op_desc);
  void AddConv1x1Node(const GpuInfo& gpu_info,
                      const Convolution2DAttributes& attr, bool last_op);
  void AddConv2dNode(const GpuInfo& gpu_info,
                     const Convolution2DAttributes& attr);
  void AddReluNode(const ReLUAttributes& attr);
  void AddPreluNode(const PReLUAttributes& attr);
  absl::Status AddAddNode(ValueId add_new_input_id);
  void AddElementwiseOneInputNode(const GpuInfo& gpu_info,
                                  const OperationType& op_type);
  void AddDepthwiseConvNode(const GpuInfo& gpu_info,
                            const DepthwiseConvolution2DAttributes& attr);
  void AddConv1x1Data(const Convolution2DAttributes& conv_attr);
  void AddConv2dData(const Convolution2DAttributes& conv_attr);
  void AddDepthwiseConvData(const DepthwiseConvolution2DAttributes& dw_attr);
  void CreateConstantsGpuBuffer(const GpuInfo& gpu_info);
  std::vector<Node*> nodes_;
  OperationDef op_def_;
  CalculationsPrecision precision_;
  DataType data_type_;
  std::vector<Value*> inputs_;
  Arguments args_;
  std::string code_;
  std::vector<std::string> outputs_;
  std::vector<float> gpu_data_;
  int weights_counter_ = 0;
  int buffer_size_ = 0;
  std::string op_name_;
  int link_counter_ = 0;
  int convs_count_ = 0;
  const GraphFloat32* graph_;
  const GpuModelBuilder* model_builder_;
  const std::set<NodeId>* consumed_nodes_;
};

void ThinPointwiseFuser::AddDepthwiseConvData(
    const DepthwiseConvolution2DAttributes& dw_attr) {
  const OHWI& dw_weights_shape =
      std::visit([](const auto& w) { return w.shape; }, dw_attr.weights);
  const int dst_slices = DivideRoundUp(dw_weights_shape.i, 4);
  const int dw_weights_count = GetDepthwiseConvWeightsCount(dw_attr);
  gpu_data_.reserve(gpu_data_.size() + dw_weights_count);
  // dw bias loading
  for (int i = 0; i < dst_slices * 4; ++i) {
    if (i < dw_attr.bias.shape.v) {
      gpu_data_.push_back(dw_attr.bias.data[i]);
    } else {
      gpu_data_.push_back(0.0f);
    }
  }
  const auto& dw_weights_data = GetFloatWeights(dw_attr).data;
  // dw weights loading
  for (int d = 0; d < dst_slices; ++d) {
    for (int y = 0; y < dw_weights_shape.h; ++y) {
      for (int x = 0; x < dw_weights_shape.w; ++x) {
        for (int i = 0; i < 4; ++i) {
          const int d_ch = d * 4 + i;
          if (d_ch < dw_weights_shape.i) {
            const int f_index = dw_weights_shape.LinearIndex({0, y, x, d_ch});
            gpu_data_.push_back(dw_weights_data[f_index]);
          } else {
            gpu_data_.push_back(0.0f);
          }
        }
      }
    }
  }
}

void ThinPointwiseFuser::AddConv1x1Data(
    const Convolution2DAttributes& conv_attr) {
  const OHWI& conv_weights_shape =
      std::visit([](const auto& w) { return w.shape; }, conv_attr.weights);
  const int src_slices = DivideRoundUp(conv_weights_shape.i, 4);
  const int dst_slices = DivideRoundUp(conv_weights_shape.o, 4);
  const int weights_count = GetConvWeightsCount(conv_attr);
  gpu_data_.reserve(gpu_data_.size() + weights_count);
  // conv bias loading
  for (int i = 0; i < dst_slices * 4; ++i) {
    if (i < conv_attr.bias.shape.v) {
      gpu_data_.push_back(conv_attr.bias.data[i]);
    } else {
      gpu_data_.push_back(0.0f);
    }
  }
  // conv weights loading
  const auto& conv_weights_data = GetFloatWeights(conv_attr).data;
  for (int d = 0; d < dst_slices; ++d) {
    for (int s = 0; s < src_slices; ++s) {
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          const int s_ch = s * 4 + j;
          const int d_ch = d * 4 + i;
          if (s_ch < conv_weights_shape.i && d_ch < conv_weights_shape.o) {
            const int f_index =
                conv_weights_shape.LinearIndex({d_ch, 0, 0, s_ch});
            gpu_data_.push_back(conv_weights_data[f_index]);
          } else {
            gpu_data_.push_back(0.0f);
          }
        }
      }
    }
  }
}

void ThinPointwiseFuser::AddConv2dData(
    const Convolution2DAttributes& conv_attr) {
  const OHWI& conv_weights_shape =
      std::visit([](const auto& w) { return w.shape; }, conv_attr.weights);
  const int src_slices = DivideRoundUp(conv_weights_shape.i, 4);
  const int dst_slices = DivideRoundUp(conv_weights_shape.o, 4);
  const int weights_count = GetConvWeightsCount(conv_attr);
  gpu_data_.reserve(gpu_data_.size() + weights_count);
  // conv bias loading
  for (int i = 0; i < dst_slices * 4; ++i) {
    if (i < conv_attr.bias.shape.v) {
      gpu_data_.push_back(conv_attr.bias.data[i]);
    } else {
      gpu_data_.push_back(0.0f);
    }
  }
  // conv weights loading
  const auto& conv_weights_data = GetFloatWeights(conv_attr).data;
  for (int s = 0; s < src_slices; ++s) {
    for (int ky = 0; ky < conv_weights_shape.h; ++ky) {
      for (int kx = 0; kx < conv_weights_shape.w; ++kx) {
        for (int d = 0; d < dst_slices; ++d) {
          for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < 4; ++i) {
              const int s_ch = s * 4 + j;
              const int d_ch = d * 4 + i;
              if (s_ch < conv_weights_shape.i && d_ch < conv_weights_shape.o) {
                const int f_index =
                    conv_weights_shape.LinearIndex({d_ch, ky, kx, s_ch});
                gpu_data_.push_back(conv_weights_data[f_index]);
              } else {
                gpu_data_.push_back(0.0f);
              }
            }
          }
        }
      }
    }
  }
}

void ThinPointwiseFuser::CreateConstantsGpuBuffer(const GpuInfo& gpu_info) {
  BufferDescriptor desc;
  desc.element_type = data_type_;
  desc.element_size = 4;
  desc.memory_type =
      gpu_info.IsMali() || gpu_info.IsBroadcom() || gpu_info.IsAMD()
          ? MemoryType::GLOBAL
          : MemoryType::CONSTANT;
  if (gpu_info.IsApiVulkan()) {
    desc.memory_type = MemoryType::GLOBAL;
  }
  desc.size = SizeOf(data_type_) * gpu_data_.size();
  desc.data.resize(desc.size);

  if (data_type_ == DataType::FLOAT32) {
    memcpy(desc.data.data(), gpu_data_.data(), desc.size);
  } else {
    half* gpu_data_half = reinterpret_cast<half*>(desc.data.data());
    for (int i = 0; i < gpu_data_.size(); ++i) {
      gpu_data_half[i] = gpu_data_[i];
    }
  }
  args_.AddObject("constants",
                  std::make_unique<BufferDescriptor>(std::move(desc)));
}

void ThinPointwiseFuser::Init(const GraphFloat32* graph,
                              const GpuModelBuilder* model_builder,
                              DataType data_type,
                              const std::set<NodeId>& consumed_nodes) {
  data_type_ = data_type;
  precision_ = model_builder->GetConvPrecision(data_type_);
  graph_ = graph;
  model_builder_ = model_builder;
  consumed_nodes_ = &consumed_nodes;
  weights_counter_ = 0;
}

bool ThinPointwiseFuser::IsNodeSupported(const GpuInfo& gpu_info,
                                         Node* node) const {
  if (!node) {
    return false;
  }
  auto op_type = OperationTypeFromString(node->operation.type);
  if (op_type == OperationType::ADD) {
    if (nodes_.empty()) {
      return false;
    }
    auto add_inputs = graph_->FindInputs(node->id);
    if (add_inputs.size() != 2) {
      return false;
    }
    auto last_node_outputs = graph_->FindOutputs(nodes_.back()->id);
    Value* add_new_input = add_inputs[0]->id != last_node_outputs[0]->id
                               ? add_inputs[0]
                               : add_inputs[1];

    const auto prev_shape = last_node_outputs[0]->tensor.shape;
    const auto add_new_shape = add_new_input->tensor.shape;
    if (prev_shape != add_new_shape) {
      return false;
    }
    Node* producer = graph_->FindProducer(add_new_input->id);
    if (!producer) {
      // add_new_input_id is global input without producer
      return true;
    }
    if (consumed_nodes_->find(producer->id) == consumed_nodes_->end()) {
      return false;
    }
    return true;
  }
  if (op_type == OperationType::RELU || op_type == OperationType::PRELU) {
    return !nodes_.empty();
  } else if (IsElementwiseOneInput(op_type)) {
    return !nodes_.empty();
  } else if (op_type == OperationType::DEPTHWISE_CONVOLUTION) {
    if (!nodes_.empty()) {
      return false;
    }
    auto inputs = graph_->FindInputs(node->id);
    if (inputs.size() != 1) {
      return false;
    }
    DepthwiseConvolution2DAttributes* dw_attr =
        std::any_cast<DepthwiseConvolution2DAttributes>(
            &node->operation.attributes);
    const auto dw_shape =
        std::visit([](const auto& w) { return w.shape; }, dw_attr->weights);
    bool good_dw = dw_shape.o == 1;
    if (!good_dw) {
      return false;
    }
    if (gpu_info.IsApple() || gpu_info.IsIntel()) {
      return dw_shape.i <= 16 &&
             dw_shape.i * dw_shape.h * dw_shape.w <= 3 * 3 * 16;
    } else if (gpu_info.IsMali()) {
      const bool kNeedExplicitClampToZero =
          !op_def_.src_tensors[0].SupportsZeroClamp(Axis::WIDTH, gpu_info) ||
          !op_def_.src_tensors[0].SupportsZeroClamp(Axis::HEIGHT, gpu_info);
      if (precision_ == CalculationsPrecision::F16 &&
          !kNeedExplicitClampToZero) {
        const int kMaxChannels = 16;
        return dw_shape.i <= kMaxChannels &&
               dw_shape.i * dw_shape.h * dw_shape.w <= 3 * 3 * kMaxChannels;
      } else {
        return false;
      }
    } else {
      if (precision_ == CalculationsPrecision::F16) {
        return dw_shape.i <= 32 &&
               dw_shape.i * dw_shape.h * dw_shape.w <= 3 * 3 * 32;
      } else {
        return dw_shape.i <= 16 &&
               dw_shape.i * dw_shape.h * dw_shape.w <= 3 * 3 * 16;
      }
    }
  } else if (op_type == OperationType::CONVOLUTION_2D) {
    auto inputs = graph_->FindInputs(node->id);
    if (inputs.size() != 1) {
      return false;
    }
    Convolution2DAttributes* conv_attr =
        std::any_cast<Convolution2DAttributes>(&node->operation.attributes);
    if (conv_attr->groups != 1) {
      return false;
    }
    const bool is_1x1_conv = IsConv1x1(*conv_attr);
    if (!is_1x1_conv && !nodes_.empty()) {
      return false;
    }
    const int weights_size = GetConvWeightsSize(*conv_attr, data_type_);
    int max_convs_count = 1;
    int max_buffer_size = 1024;
    if (gpu_info.IsAdreno() && gpu_info.IsApiOpenCl()) {
      max_convs_count = 3;
      max_buffer_size = 1024 * 3;
    } else if (gpu_info.IsApple() && gpu_info.apple_info.IsBionic()) {
      max_convs_count = 3;
      max_buffer_size = 1024 * 2;
    } else if (gpu_info.IsMali() && !gpu_info.mali_info.IsBifrost()) {
      max_convs_count = 3;
      max_buffer_size = 1024 * 3;
    } else if (gpu_info.IsPowerVR()) {
      max_convs_count = 3;
      max_buffer_size = 1024 * 3;
    } else if (gpu_info.IsNvidia()) {
      max_convs_count = 3;
      max_buffer_size = 1024 * 3;
    } else if (gpu_info.IsIntel()) {
      max_convs_count = 3;
      max_buffer_size = 1024 * 2;
    }
    if (convs_count_ >= max_convs_count ||
        buffer_size_ + weights_size > max_buffer_size) {
      return false;
    }
    const auto conv_shape =
        std::visit([](const auto& w) { return w.shape; }, conv_attr->weights);
    const int kernel_size = conv_shape.i * conv_shape.w * conv_shape.h;
    if (gpu_info.IsApple() || gpu_info.IsIntel()) {
      if (precision_ == CalculationsPrecision::F16) {
        return conv_shape.o <= 16 && kernel_size * conv_shape.o <= 16 * 16;
      } else {
        return conv_shape.o <= 8 && kernel_size * conv_shape.o <= 8 * 16;
      }
    } else if (gpu_info.IsMali()) {
      const bool kNeedExplicitClampToZero =
          !is_1x1_conv &&
          (!op_def_.src_tensors[0].SupportsZeroClamp(Axis::WIDTH, gpu_info) ||
           !op_def_.src_tensors[0].SupportsZeroClamp(Axis::HEIGHT, gpu_info));
      if (precision_ == CalculationsPrecision::F16 &&
          !kNeedExplicitClampToZero) {
        const int kMaxChannels = gpu_info.mali_info.IsBifrost() ? 16 : 32;
        return conv_shape.o <= kMaxChannels &&
               kernel_size * conv_shape.o <= kMaxChannels * kMaxChannels;
      } else {
        return false;
      }
    } else {
      if (precision_ == CalculationsPrecision::F16) {
        return conv_shape.o <= 32 && kernel_size * conv_shape.o <= 32 * 32;
      } else {
        return conv_shape.o <= 32 && kernel_size * conv_shape.o <= 16 * 32;
      }
    }
  } else {
    return false;
  }
}

absl::Status ThinPointwiseFuser::ReserveNode(const GpuInfo& gpu_info,
                                             Node* node) {
  if (nodes_.empty()) {
    auto inputs = graph_->FindInputs(node->id);
    ASSIGN_OR_RETURN(auto handle, model_builder_->GetTensor(inputs[0]->id));
    op_def_.src_tensors.push_back(handle.tensor_desc);
  }
  if (!IsNodeSupported(gpu_info, node)) {
    return absl::InternalError("Node is not supported");
  }
  nodes_.push_back(node);
  if (IsConvNode(node)) {
    convs_count_++;
    Convolution2DAttributes* conv_attr =
        std::any_cast<Convolution2DAttributes>(&node->operation.attributes);
    buffer_size_ += GetConvWeightsSize(*conv_attr, data_type_);
  }
  if (IsDwConvNode(node)) {
    DepthwiseConvolution2DAttributes* dw_attr =
        std::any_cast<DepthwiseConvolution2DAttributes>(
            &node->operation.attributes);
    buffer_size_ += GetDepthwiseConvWeightsSize(*dw_attr, data_type_);
  }
  return absl::OkStatus();
}

uint64_t ThinPointwiseFuser::GetNodeFlops(Node* node) const {
  auto op_type = OperationTypeFromString(node->operation.type);
  auto output_shape = graph_->FindOutputs(node->id)[0]->tensor.shape;
  if (op_type == OperationType::DEPTHWISE_CONVOLUTION) {
    DepthwiseConvolution2DAttributes* attr =
        std::any_cast<DepthwiseConvolution2DAttributes>(
            &node->operation.attributes);
    return GetDepthwiseConvolutionFlops(
        output_shape,
        std::visit([](const auto& w) { return w.shape; }, attr->weights));
  } else if (op_type == OperationType::CONVOLUTION_2D) {
    Convolution2DAttributes* attr =
        std::any_cast<Convolution2DAttributes>(&node->operation.attributes);
    return GetConvolutionFlops(
        output_shape,
        std::visit([](const auto& w) { return w.shape; }, attr->weights));
  }
  return 0;
}

absl::Status ThinPointwiseFuser::AddNode(const GpuInfo& gpu_info,
                                         int node_index) {
  Node* node = nodes_[node_index];
  auto op_type = OperationTypeFromString(node->operation.type);
  if (op_type == OperationType::RELU) {
    ReLUAttributes* attr =
        std::any_cast<ReLUAttributes>(&node->operation.attributes);
    AddReluNode(*attr);
  } else if (op_type == OperationType::PRELU) {
    PReLUAttributes* attr =
        std::any_cast<PReLUAttributes>(&node->operation.attributes);
    AddPreluNode(*attr);
  } else if (op_type == OperationType::ADD) {
    Node* prev_node = nodes_[node_index - 1];
    auto add_inputs = graph_->FindInputs(node->id);
    auto prev_node_outputs = graph_->FindOutputs(prev_node->id);
    Value* add_new_input = add_inputs[0]->id != prev_node_outputs[0]->id
                               ? add_inputs[0]
                               : add_inputs[1];
    inputs_.push_back(add_new_input);
    RETURN_IF_ERROR(AddAddNode(add_new_input->id));
  } else if (IsElementwiseOneInput(op_type)) {
    AddElementwiseOneInputNode(gpu_info, op_type);
  } else if (op_type == OperationType::DEPTHWISE_CONVOLUTION) {
    DepthwiseConvolution2DAttributes* attr =
        std::any_cast<DepthwiseConvolution2DAttributes>(
            &node->operation.attributes);
    AddDepthwiseConvNode(gpu_info, *attr);
  } else if (op_type == OperationType::CONVOLUTION_2D) {
    Convolution2DAttributes* attr =
        std::any_cast<Convolution2DAttributes>(&node->operation.attributes);
    if (IsConv1x1(*attr) && node_index != 0) {
      AddConv1x1Node(gpu_info, *attr, node_index == nodes_.size() - 1);
    } else {
      AddConv2dNode(gpu_info, *attr);
    }
  }
  return absl::OkStatus();
}

bool ThinPointwiseFuser::IsElementwiseNode(Node* node) const {
  auto op_type = OperationTypeFromString(node->operation.type);
  return op_type == OperationType::RELU || op_type == OperationType::PRELU ||
         op_type == OperationType::ADD || IsElementwiseOneInput(op_type);
}

bool ThinPointwiseFuser::IsConvNode(Node* node) const {
  auto op_type = OperationTypeFromString(node->operation.type);
  return op_type == OperationType::CONVOLUTION_2D;
}

bool ThinPointwiseFuser::IsDwConvNode(Node* node) const {
  auto op_type = OperationTypeFromString(node->operation.type);
  return op_type == OperationType::DEPTHWISE_CONVOLUTION;
}

void ThinPointwiseFuser::AddDepthwiseConvNode(
    const GpuInfo& gpu_info, const DepthwiseConvolution2DAttributes& attr) {
  AddDepthwiseConvData(attr);
  op_name_ += "dw_conv2d";
  args_.AddInt("stride_x", attr.strides.w);
  args_.AddInt("padding_x", -attr.padding.prepended.w);
  args_.AddInt("dilation_x", attr.dilations.w);
  args_.AddInt("stride_y", attr.strides.h);
  args_.AddInt("padding_y", -attr.padding.prepended.h);
  args_.AddInt("dilation_y", attr.dilations.h);

  const auto& src_desc = op_def_.src_tensors[0];
  const auto& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  int intermediate_depth = DivideRoundUp(weights_shape.i, 4);
  for (int d = 0; d < intermediate_depth; ++d) {
    outputs_.push_back("dw_res_" + std::to_string(d));
    code_ += "  Type " + outputs_[d] + " = args.constants.Read(" +
             std::to_string(weights_counter_++) + ");\n";
  }
  code_ += "  int x_offseted = X * args.stride_x + args.padding_x;\n";
  code_ += "  int y_offseted = Y * args.stride_y + args.padding_y;\n";
  code_ += "  int x_c, y_c;\n";

  auto generate_check = [&]() {
    std::string check;
    const std::vector<Axis> axes{Axis::WIDTH, Axis::HEIGHT, Axis::DEPTH};
    const std::vector<std::string> names{"x_in", "y_in", "z_in"};
    for (int i = 0; i < axes.size(); ++i) {
      const auto& axis = axes[i];
      if (src_desc.HasAxis(axis) &&
          !src_desc.SupportsZeroClamp(axis, gpu_info)) {
        if (!check.empty()) {
          check += " && ";
        }
        check += names[i];
      }
    }
    return check;
  };
  const std::string check = generate_check();
  if (!src_desc.SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
    code_ += "  bool y_in;\n";
  }
  if (!src_desc.SupportsZeroClamp(Axis::WIDTH, gpu_info)) {
    code_ += "  bool x_in;\n";
  }

  const std::string postfixes[] = {".x", ".xy", ".xyz", ""};
  code_ += "  Type src;\n";
  for (int d = 0; d < intermediate_depth; ++d) {
    const int src_ch_count = std::min(4, weights_shape.i - d * 4);
    const std::string s_postfix = postfixes[src_ch_count - 1];
    for (int ky = 0; ky < weights_shape.h; ++ky) {
      code_ += "  y_c = y_offseted + " + std::to_string(ky) +
               " * args.dilation_y;\n";
      if (!src_desc.SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
        code_ += "  y_in = y_c >= 0 && y_c < args.src_tensor.Height();\n";
        code_ += "  y_c = clamp(y_c, 0, args.src_tensor.Height() - 1);\n";
      }
      for (int kx = 0; kx < weights_shape.w; ++kx) {
        code_ += "  x_c = x_offseted + " + std::to_string(kx) +
                 " * args.dilation_x;\n";
        if (!src_desc.SupportsZeroClamp(Axis::WIDTH, gpu_info)) {
          code_ += "  x_in = x_c >= 0 && x_c < args.src_tensor.Width();\n";
          code_ += "  x_c = clamp(x_c, 0, args.src_tensor.Width() - 1);\n";
        }
        std::string multiplier =
            check.empty() ? "" : " * ucl::Convert<SType>(" + check + ")";
        code_ += "  src" + s_postfix + " = args.src_tensor.Read(x_c, y_c, " +
                 std::to_string(d) + ")" + s_postfix + multiplier + ";\n";
        code_ += "  " +
                 MAC(gpu_info, outputs_[d] + s_postfix, "src" + s_postfix,
                     "args.constants.Read(" +
                         std::to_string(weights_counter_++) + ")" + s_postfix) +
                 ";\n";
      }
    }
  }
}

void ThinPointwiseFuser::AddElementwiseNode(ElementwiseDescriptor&& op_desc) {
  std::string unique_postfix = absl::StrCat("_link_internal", link_counter_);
  link_counter_++;
  op_desc.args.RenameArgs(unique_postfix, &op_desc.code);
  auto status = args_.Merge(std::move(op_desc.args), unique_postfix);
  for (int i = 0; i < outputs_.size(); ++i) {
    const std::string elementwise_new_code =
        absl::StrReplaceAll(op_desc.code, {{"in_value", outputs_[i]},
                                           {"out_value", outputs_[i]},
                                           {"X_COORD", "X"},
                                           {"Y_COORD", "Y"},
                                           {"S_COORD", std::to_string(i)},
                                           {"B_COORD", "B"}});
    code_ += "  {  " + elementwise_new_code + "  }\n";
  }
}

void ThinPointwiseFuser::AddReluNode(const ReLUAttributes& attr) {
  ElementwiseDescriptor op_desc =
      CreateReLU(attr, op_def_.src_tensors[0].GetDataType());
  AddElementwiseNode(std::move(op_desc));
}

void ThinPointwiseFuser::AddPreluNode(const PReLUAttributes& attr) {
  ElementwiseDescriptor op_desc = CreatePReLU(attr, op_def_.dst_tensors[0]);
  AddElementwiseNode(std::move(op_desc));
}

absl::Status ThinPointwiseFuser::AddAddNode(ValueId add_new_input_id) {
  op_name_ += "->add";
  const std::string tensor_name =
      absl::StrCat("src_tensor", op_def_.src_tensors.size());
  ASSIGN_OR_RETURN(auto handle, model_builder_->GetTensor(add_new_input_id));
  op_def_.src_tensors.push_back(handle.tensor_desc);
  for (int i = 0; i < outputs_.size(); ++i) {
    code_ += "  if (" + std::to_string(i) + " < args." + tensor_name +
             ".Slices()) {\n" + outputs_[i] + " += args." + tensor_name +
             ".Read(X, Y, " + std::to_string(i) + ");\n}\n";
  }
  return absl::OkStatus();
}

void ThinPointwiseFuser::AddElementwiseOneInputNode(
    const GpuInfo& gpu_info, const OperationType& op_type) {
  ElementwiseDescriptor op_desc = CreateElementwiseOneInput(
      gpu_info, op_type, op_def_.src_tensors[0].GetDataType());
  AddElementwiseNode(std::move(op_desc));
}

void ThinPointwiseFuser::AddConv1x1Node(const GpuInfo& gpu_info,
                                        const Convolution2DAttributes& attr,
                                        bool last_op) {
  AddConv1x1Data(attr);
  op_name_ += "->conv1x1";
  const auto& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  const int src_slices = DivideRoundUp(weights_shape.i, 4);
  const int dst_slices = DivideRoundUp(weights_shape.o, 4);
  std::vector<std::string> inputs = outputs_;
  outputs_.resize(dst_slices);
  std::string link = "_link_" + std::to_string(link_counter_);
  link_counter_++;
  for (int d = 0; d < dst_slices; ++d) {
    std::string dst = "conv_res_" + std::to_string(d) + link;
    outputs_[d] = dst;
    code_ += "  Type " + outputs_[d] + " = args.constants.Read(" +
             std::to_string(weights_counter_++) + ");\n";
  }
  for (int d = 0; d < dst_slices; ++d) {
    std::string dst = outputs_[d];
    for (int s = 0; s < src_slices; ++s) {
      std::string src = inputs[s];
      const std::string c0 =
          "args.constants.Read(" + std::to_string(weights_counter_++) + ")";
      const std::string c1 =
          "args.constants.Read(" + std::to_string(weights_counter_++) + ")";
      const std::string c2 =
          "args.constants.Read(" + std::to_string(weights_counter_++) + ")";
      const std::string c3 =
          "args.constants.Read(" + std::to_string(weights_counter_++) + ")";
      code_ += "  " + MAC(gpu_info, dst, c0, src + ".x") + ";\n";
      code_ += "  " + MAC(gpu_info, dst, c1, src + ".y") + ";\n";
      code_ += "  " + MAC(gpu_info, dst, c2, src + ".z") + ";\n";
      code_ += "  " + MAC(gpu_info, dst, c3, src + ".w") + ";\n";
    }
    if (last_op) {
      code_ += "  if(" + std::to_string(d) + " < args.dst_tensor.Slices()) {\n";
      code_ += "    args.dst_tensor.Write(" + dst + ", X, Y, " +
               std::to_string(d) + ");\n";
      code_ += "  }\n";
    }
  }
}

void ThinPointwiseFuser::AddConv2dNode(const GpuInfo& gpu_info,
                                       const Convolution2DAttributes& attr) {
  AddConv2dData(attr);
  if (IsConv1x1(attr)) {
    op_name_ += "conv1x1";
  } else {
    op_name_ += "conv2d";
  }
  const auto& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  const int src_slices = DivideRoundUp(weights_shape.i, 4);
  const int dst_slices = DivideRoundUp(weights_shape.o, 4);
  args_.AddInt("stride_x", attr.strides.w);
  args_.AddInt("padding_x", -attr.padding.prepended.w);
  args_.AddInt("dilation_x", attr.dilations.w);
  args_.AddInt("stride_y", attr.strides.h);
  args_.AddInt("padding_y", -attr.padding.prepended.h);
  args_.AddInt("dilation_y", attr.dilations.h);

  const auto& src_desc = op_def_.src_tensors[0];
  for (int d = 0; d < dst_slices; ++d) {
    outputs_.push_back("conv_res_" + std::to_string(d));
    code_ += "  Type " + outputs_[d] + " = args.constants.Read(" +
             std::to_string(weights_counter_++) + ");\n";
  }
  std::string x_base_coord = "X";
  if (attr.strides.w != 1 || attr.padding.prepended.w != 0) {
    code_ += "  int x_offseted = X * args.stride_x + args.padding_x;\n";
    x_base_coord = "x_offseted";
  }
  std::string y_base_coord = "Y";
  if (attr.strides.h != 1 || attr.padding.prepended.h != 0) {
    code_ += "  int y_offseted = Y * args.stride_y + args.padding_y;\n";
    y_base_coord = "y_offseted";
  }
  if (!IsConvKernelXis1(attr)) {
    code_ += "  int x_c;\n";
  }
  if (!IsConvKernelYis1(attr)) {
    code_ += "  int y_c;\n";
  }

  auto generate_check = [&]() {
    std::string check;
    const std::vector<Axis> axes{Axis::WIDTH, Axis::HEIGHT, Axis::DEPTH};
    const std::vector<std::string> names{"x_in", "y_in", "z_in"};
    for (int i = 0; i < axes.size(); ++i) {
      const auto& axis = axes[i];
      if (src_desc.HasAxis(axis) &&
          !src_desc.SupportsZeroClamp(axis, gpu_info)) {
        if (!check.empty()) {
          check += " && ";
        }
        check += names[i];
      }
    }
    return check;
  };
  const std::string check = generate_check();
  if (!src_desc.SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
    code_ += "  bool y_in;\n";
  }
  if (!src_desc.SupportsZeroClamp(Axis::WIDTH, gpu_info)) {
    code_ += "  bool x_in;\n";
  }

  const std::string postfixes[] = {".x", ".xy", ".xyz", ""};
  code_ += "  Type src;\n";
  for (int s = 0; s < src_slices; ++s) {
    for (int ky = 0; ky < weights_shape.h; ++ky) {
      std::string y_coord = "Y";
      if (!IsConvKernelYis1(attr)) {
        y_coord = "y_c";
        code_ += "  y_c = " + y_base_coord + " + " + std::to_string(ky) +
                 " * args.dilation_y;\n";
        if (!src_desc.SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
          code_ += "  y_in = y_c >= 0 && y_c < args.src_tensor.Height();\n";
          code_ += "  y_c = clamp(y_c, 0, args.src_tensor.Height() - 1);\n";
        }
      }
      for (int kx = 0; kx < weights_shape.w; ++kx) {
        std::string x_coord = "X";
        if (!IsConvKernelXis1(attr)) {
          x_coord = "x_c";
          code_ += "  x_c = " + x_base_coord + " + " + std::to_string(kx) +
                   " * args.dilation_x;\n";
          if (!src_desc.SupportsZeroClamp(Axis::WIDTH, gpu_info)) {
            code_ += "  x_in = x_c >= 0 && x_c < args.src_tensor.Width();\n";
            code_ += "  x_c = clamp(x_c, 0, args.src_tensor.Width() - 1);\n";
          }
        }
        std::string multiplier;
        if (!IsConv1x1(attr) && !check.empty()) {
          multiplier = " * ucl::Convert<SType>(" + check + ")";
        }
        code_ += "  src = args.src_tensor.Read(" + x_coord + ", " + y_coord +
                 ", " + std::to_string(s) + ")" + multiplier + ";\n";
        for (int d = 0; d < dst_slices; ++d) {
          std::string src = "src";
          const std::string c0 =
              "args.constants.Read(" + std::to_string(weights_counter_++) + ")";
          const std::string c1 =
              "args.constants.Read(" + std::to_string(weights_counter_++) + ")";
          const std::string c2 =
              "args.constants.Read(" + std::to_string(weights_counter_++) + ")";
          const std::string c3 =
              "args.constants.Read(" + std::to_string(weights_counter_++) + ")";
          code_ += "  " + MAC(gpu_info, outputs_[d], c0, "src.x") + ";\n";
          code_ += "  " + MAC(gpu_info, outputs_[d], c1, "src.y") + ";\n";
          code_ += "  " + MAC(gpu_info, outputs_[d], c2, "src.z") + ";\n";
          code_ += "  " + MAC(gpu_info, outputs_[d], c3, "src.w") + ";\n";
        }
      }
    }
  }
}

absl::Status ThinPointwiseFuser::Finalize(const GpuInfo& gpu_info,
                                          GpuModelBuilder* model_builder) {
  while (!nodes_.empty() && IsElementwiseNode(nodes_.back())) {
    nodes_.pop_back();
  }
  int non_elementwise_nodes_count = 0;
  for (const auto& node : nodes_) {
    if (!IsElementwiseNode(node)) {
      non_elementwise_nodes_count += 1;
    }
  }
  if (non_elementwise_nodes_count <= 1) {
    return absl::UnavailableError("Not found suitable sequence.");
  }
  inputs_ = graph_->FindInputs(nodes_.front()->id);
  auto last_node_outputs = graph_->FindOutputs(nodes_.back()->id);
  auto handle = model_builder_->GetTensor(last_node_outputs[0]->id);
  op_def_.dst_tensors.push_back(handle.value().tensor_desc);

  code_ = "MAIN_FUNCTION($0) {\n";
  if (op_def_.src_tensors[0].HasAxis(Axis::BATCH)) {
    code_ += "  int linear_id = ucl::GetGlobalId<0>();\n";
    code_ += "  int X = linear_id / args.dst_tensor.Batch();\n";
    code_ += "  int B = linear_id % args.dst_tensor.Batch();\n";
    code_ += "  args.dst_tensor.SetBatchRef(B);\n";
    code_ += "  args.src_tensor.SetBatchRef(B);\n";
  } else {
    code_ += "  int X = ucl::GetGlobalId<0>();\n";
  }
  code_ += "  int Y = ucl::GetGlobalId<1>();\n";
  code_ +=
      "  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height()) { "
      "\n";
  code_ += "    return; \n";
  code_ += "  } \n";

  for (int i = 0; i < nodes_.size(); ++i) {
    RETURN_IF_ERROR(AddNode(gpu_info, i));
  }
  code_ += "}\n";
  const DataType type = op_def_.src_tensors[0].GetDataType();
  absl::StrReplaceAll(
      {{"SType", ToUclDataType(type, 1)}, {"Type", ToUclDataType(type, 4)}},
      &code_);

  if (gpu_info.IsMali()) {
    const BHWC dst_shape = last_node_outputs[0]->tensor.shape;
    const int dst_slices = DivideRoundUp(dst_shape.c, 4);
    const int task_size = dst_shape.b * dst_shape.h * dst_shape.w * dst_slices;
    const int block_size =
        GetRecommendedBlockSizeForConv(gpu_info, precision_, task_size);
    if (block_size < 4 && dst_slices >= 2) {
      return absl::UnimplementedError("No performant implementation.");
    }
    if (block_size < 2 && dst_slices >= 4) {
      return absl::UnimplementedError("No performant implementation.");
    }
  }
  CreateConstantsGpuBuffer(gpu_info);
  GPUOperation operation;
  operation.args_ = std::move(args_);
  operation.AddSrcTensor("src_tensor", op_def_.src_tensors[0]);
  for (int i = 1; i < op_def_.src_tensors.size(); ++i) {
    const std::string tensor_name = absl::StrCat("src_tensor", i);
    operation.AddSrcTensor(tensor_name, op_def_.src_tensors[i]);
  }
  operation.AddDstTensor("dst_tensor", op_def_.dst_tensors[0]);
  operation.code_ = code_;
  operation.flops_ = 0;
  for (const auto& node : nodes_) {
    operation.flops_ += GetNodeFlops(node);
  }
  operation.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_ZIs1;
  if (gpu_info.IsMali() || gpu_info.IsPowerVR()) {
    operation.compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
  std::vector<ValueId> src_ids(inputs_.size());
  for (int i = 0; i < inputs_.size(); ++i) {
    src_ids[i] = inputs_[i]->id;
  }
  std::vector<ValueId> dst_ids(last_node_outputs.size());
  for (int i = 0; i < last_node_outputs.size(); ++i) {
    dst_ids[i] = last_node_outputs[i]->id;
  }
  model_builder->AddGpuOperation(
      src_ids, dst_ids, std::make_unique<GPUOperation>(std::move(operation)),
      op_name_);
  return absl::OkStatus();
}

Node* GetNextLinearNode(const GraphFloat32& graph, NodeId current_node) {
  auto outputs = graph.FindOutputs(current_node);
  if (outputs.size() != 1) {
    return nullptr;
  }
  auto consumers = graph.FindConsumers(outputs[0]->id);
  if (consumers.size() != 1) {
    return nullptr;
  }
  return consumers[0];
}

absl::Status TryThinPointwiseFuser(const GpuInfo& gpu_info,
                                   const GraphFloat32& graph,
                                   NodeId first_node_id,
                                   const std::set<NodeId>& consumed_nodes,
                                   std::set<NodeId>* new_consumed_nodes,
                                   GpuModelBuilder* model_builder) {
  if (!(gpu_info.IsAdreno() || gpu_info.IsNvidia() || gpu_info.IsMali() ||
        gpu_info.IsApple() || gpu_info.IsAMD() || gpu_info.IsIntel() ||
        gpu_info.IsPowerVR())) {
    return absl::NotFoundError("ThinPointwiseFuser not suitable.");
  }
  if (gpu_info.IsMali() && gpu_info.mali_info.IsMidgard()) {
    return absl::NotFoundError("ThinPointwiseFuser not suitable.");
  }
  if (gpu_info.IsPowerVR() &&
      gpu_info.powervr_info.gpu_version < PowerVRGpu::kCXT) {
    return absl::NotFoundError("ThinPointwiseFuser not suitable.");
  }

  auto* node = graph.GetNode(first_node_id);
  if (node == nullptr ||
      consumed_nodes.find(node->id) != consumed_nodes.end()) {
    return absl::NotFoundError("ThinPointwiseFuser not suitable.");
  }
  auto node_inputs = graph.FindInputs(node->id);
  if (node_inputs.empty()) {
    return absl::NotFoundError("ThinPointwiseFuser not suitable.");
  }

  ASSIGN_OR_RETURN(auto tensor_handle,
                   model_builder->GetTensor(node_inputs[0]->id));

  ThinPointwiseFuser fuser;
  fuser.Init(&graph, model_builder, tensor_handle.tensor_desc.GetDataType(),
             consumed_nodes);
  while (fuser.ReserveNode(gpu_info, node).ok()) {
    node = GetNextLinearNode(graph, node->id);
    if (node == nullptr ||
        consumed_nodes.find(node->id) != consumed_nodes.end()) {
      break;
    }
  }
  RETURN_IF_ERROR(fuser.Finalize(gpu_info, model_builder));
  const auto fused_nodes = fuser.GetFusedNodes();
  new_consumed_nodes->insert(fused_nodes.begin(), fused_nodes.end());
  return absl::OkStatus();
}

}  // namespace ml_drift
