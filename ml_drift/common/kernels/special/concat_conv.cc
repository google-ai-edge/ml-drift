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

#include "ml_drift/common/kernels/special/concat_conv.h"

#include <any>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/flops_util.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/compiler_options.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/work_group_picking.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace {
std::string GetCode(const int2& block_size, DataType type, bool dummy_checks) {
  std::string c;
  c += "  MAIN_FUNCTION($0) {\n";
  c += "  int X = ucl::GetGlobalId<0>() * " + std::to_string(block_size.x) +
       ";\n";
  c += "  int Y = ucl::GetGlobalId<1>() * " + std::to_string(block_size.y) +
       ";\n";
  for (int by = 0; by < block_size.y; ++by) {
    for (int bx = 0; bx < block_size.x; ++bx) {
      c += "  Type r_x" + std::to_string(bx) + "_y" + std::to_string(by) +
           " = ucl::Init<Type>(0.0f);\n";
    }
  }
  c += "  int src_x, src_y;\n";
  for (int bx = 0; bx < block_size.x + 2; ++bx) {
    c += "  Type src" + std::to_string(bx) + ";\n";
  }
  for (int by = 0; by < block_size.y + 2; ++by) {
    c += "  src_y = Y - 1 + " + std::to_string(by) + ";\n";
    if (dummy_checks) {
      c += "  if (src_y >= " + std::to_string(by - 10) + ") {\n";
    }
    for (int bx = 0; bx < block_size.x + 2; ++bx) {
      c += "  src_x = X - 1 + " + std::to_string(bx) + ";\n";
      c += "  src" + std::to_string(bx) +
           ".x = args.src_tensor0.Read(src_x, src_y, 0).x;\n";
      c += "  src" + std::to_string(bx) +
           ".y = args.src_tensor1.Read(src_x, src_y, 0).x;\n";
      c += "  src" + std::to_string(bx) +
           ".z = args.src_tensor2.Read(src_x, src_y, 0).x;\n";
    }
    for (int bx = 0; bx < block_size.x; ++bx) {
      for (int ky = 0; ky < 3; ++ky) {
        int dst_idy = by - ky;
        if (dst_idy < 0 || dst_idy >= block_size.y) {
          continue;
        }
        for (int kx = 0; kx < 3; ++kx) {
          const std::string rname =
              "r_x" + std::to_string(bx) + "_y" + std::to_string(dst_idy);
          const std::string sname = "src" + std::to_string(bx + kx);
          const int w_id = ky * 3 + kx;
          c += "  " + rname + " += " + sname + ".x * args.weights.Read(" +
               std::to_string(w_id * 3 + 0) + ");\n";
          c += "  " + rname + " += " + sname + ".y * args.weights.Read(" +
               std::to_string(w_id * 3 + 1) + ");\n";
          c += "  " + rname + " += " + sname + ".z * args.weights.Read(" +
               std::to_string(w_id * 3 + 2) + ");\n";
        }
      }
    }
    if (dummy_checks) {
      c += "  }\n";
    }
  }
  c += "  Type bias_val = args.bias.Read(0);\n";
  for (int by = 0; by < block_size.y; ++by) {
    for (int bx = 0; bx < block_size.x; ++bx) {
      const std::string rname =
          "r_x" + std::to_string(bx) + "_y" + std::to_string(by);
      const std::string xname = "X + " + std::to_string(bx);
      const std::string yname = "Y + " + std::to_string(by);
      c += "  if (" + xname + " < args.dst_tensor.Width() && " + yname +
           " < args.dst_tensor.Height()) {\n";
      c += "    " + rname + " += bias_val;\n";
      c += "    args.dst_tensor.Write(" + rname + ", " + xname + ", " + yname +
           ", 0);\n";
      c += "  }\n";
    }
  }
  c += "}\n";
  return absl::StrReplaceAll(c, {{"Type", ToUclDataType(type, 4)}});
}

void AddConstantsGpuBuffer(const GpuInfo& gpu_info, DataType data_type,
                           const std::vector<float>& weights, Arguments* args) {
  BufferDescriptor desc;
  desc.element_type = data_type;
  desc.element_size = 4;
  desc.memory_type =
      gpu_info.IsMali() || gpu_info.IsBroadcom() || gpu_info.IsAMD()
          ? MemoryType::GLOBAL
          : MemoryType::CONSTANT;
  desc.size = SizeOf(data_type) * weights.size();
  desc.data.resize(desc.size);

  if (data_type == DataType::FLOAT32) {
    memcpy(desc.data.data(), weights.data(), desc.size);
  } else {
    half* gpu_data_half = reinterpret_cast<half*>(desc.data.data());
    for (int i = 0; i < weights.size(); ++i) {
      gpu_data_half[i] = weights[i];
    }
  }
  args->AddObject("weights",
                  std::make_unique<BufferDescriptor>(std::move(desc)));
}

}  // namespace

class ConcatConv : public GPUOperation {
 public:
  explicit ConcatConv(int2 block_size) : block_size_(block_size) {}
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override {
    return GetPossibleWorkGroups(tuning_type, gpu_info, kernel_info,
                                 grid_size_);
  }
  int3 GetGridSize() const override {
    int grid_x = DivideRoundUp(dst_[0]->Width(), block_size_.x);
    int grid_y = DivideRoundUp(dst_[0]->Height(), block_size_.y);
    return int3(grid_x, grid_y, 1);
  }

  // Move only
  ConcatConv(ConcatConv&& kernel) = default;
  ConcatConv& operator=(ConcatConv&& kernel) = default;
  ConcatConv(const ConcatConv&) = delete;
  ConcatConv& operator=(const ConcatConv&) = delete;

 private:
  int2 block_size_;
};

bool IsConcatConvRecommended(const GpuInfo& gpu_info) {
  const bool is_apple_recommended = gpu_info.IsApple();
  const bool is_adreno_recommended =
      gpu_info.IsAdreno() &&
      gpu_info.adreno_info.generation >= AdrenoInfo::Generation::kGen6;
  const bool is_amd_recommended = gpu_info.IsAMD();
  const bool is_mali_recommended =
      gpu_info.IsMali() &&
      gpu_info.mali_info.generation >= MaliInfo::Gen::kValhallV2;
  const bool is_powervr_recommended = gpu_info.IsPowerVR();
  return is_apple_recommended || is_adreno_recommended || is_amd_recommended ||
         is_mali_recommended || is_powervr_recommended;
}

absl::Status TryConcatConv(const GpuInfo& gpu_info, const GraphFloat32& graph,
                           NodeId first_node_id,
                           const std::set<NodeId>& consumed_nodes,
                           std::set<NodeId>* new_consumed_nodes,
                           GpuModelBuilder* model_builder) {
  if (!IsConcatConvRecommended(gpu_info)) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }
  auto* concat_node = graph.GetNode(first_node_id);
  if (concat_node == nullptr) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }
  if (OperationTypeFromString(concat_node->operation.type) !=
      OperationType::CONCAT) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }
  auto concat_inputs = graph.FindInputs(concat_node->id);
  if (concat_inputs.size() != 3 || concat_inputs[0]->tensor.shape.c != 1 ||
      concat_inputs[1]->tensor.shape.c != 1 ||
      concat_inputs[2]->tensor.shape.c != 1) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }
  ASSIGN_OR_RETURN(auto src0_handle,
                   model_builder->GetTensor(concat_inputs[0]->id));
  ASSIGN_OR_RETURN(auto src1_handle,
                   model_builder->GetTensor(concat_inputs[1]->id));
  ASSIGN_OR_RETURN(auto src2_handle,
                   model_builder->GetTensor(concat_inputs[2]->id));
  const auto& src0_td = src0_handle.tensor_desc;
  const auto& src1_td = src1_handle.tensor_desc;
  const auto& src2_td = src2_handle.tensor_desc;
  if (!src0_td.SupportsZeroClamp(Axis::WIDTH, gpu_info) ||
      !src0_td.SupportsZeroClamp(Axis::HEIGHT, gpu_info) ||
      !src1_td.SupportsZeroClamp(Axis::WIDTH, gpu_info) ||
      !src1_td.SupportsZeroClamp(Axis::HEIGHT, gpu_info) ||
      !src2_td.SupportsZeroClamp(Axis::WIDTH, gpu_info) ||
      !src2_td.SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }
  auto concat_output = graph.FindOutputs(concat_node->id)[0];
  auto concat_consumers = graph.FindConsumers(concat_output->id);
  if (concat_consumers.size() != 1) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }
  auto* conv_node = concat_consumers[0];
  if (conv_node == nullptr) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }
  if (OperationTypeFromString(conv_node->operation.type) !=
      OperationType::CONVOLUTION_2D) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }
  auto conv_inputs = graph.FindInputs(conv_node->id);
  auto conv_output = graph.FindOutputs(conv_node->id)[0];

  OperationDef op_def;
  op_def.src_tensors.push_back(src0_td);
  op_def.src_tensors.push_back(src1_td);
  op_def.src_tensors.push_back(src2_td);
  ASSIGN_OR_RETURN(auto dst_handle, model_builder->GetTensor(conv_output->id));
  op_def.dst_tensors.push_back(dst_handle.tensor_desc);

  auto concat_attr =
      std::any_cast<ConcatAttributes>(concat_node->operation.attributes);
  if (concat_attr.axis != Axis::CHANNELS) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }

  const auto& conv_attr =
      std::any_cast<Convolution2DAttributes&>(conv_node->operation.attributes);
  const auto& conv_weights = GetFloatWeights(conv_attr);

  if (conv_weights.shape != OHWI(4, 3, 3, 3) || conv_attr.strides != HW(1, 1) ||
      conv_attr.dilations != HW(1, 1) ||
      conv_attr.padding.prepended != HW(1, 1) ||
      conv_attr.padding.appended != HW(1, 1)) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }

  const DataType data_type = op_def.src_tensors[0].GetDataType();
  const int2 block_size = int2(2, 2);
  ConcatConv operation(block_size);
  std::vector<float> weights_reordered(4 * 3 * 3 * 3);
  for (int i = 0; i < 3; ++i) {
    for (int ky = 0; ky < 3; ++ky) {
      for (int kx = 0; kx < 3; ++kx) {
        for (int o = 0; o < 4; ++o) {
          int w_index = conv_weights.shape.LinearIndex({o, ky, kx, i});
          float w_val = conv_weights.data[w_index];
          weights_reordered[((ky * 3 + kx) * 3 + i) * 4 + o] = w_val;
        }
      }
    }
  }
  AddConstantsGpuBuffer(gpu_info, data_type, weights_reordered,
                        &operation.args_);
  TensorDescriptor bias_tensor_desc =
      CreateConstantLinearTensorDescriptor(gpu_info, data_type, conv_attr.bias);
  operation.args_.AddObject(
      "bias", std::make_unique<TensorDescriptor>(std::move(bias_tensor_desc)));
  operation.AddSrcTensor("src_tensor0", op_def.src_tensors[0]);
  operation.AddSrcTensor("src_tensor1", op_def.src_tensors[1]);
  operation.AddSrcTensor("src_tensor2", op_def.src_tensors[2]);
  operation.AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  // on some devices we need to add dummy checks for better performance.
  const bool dummy_checks = gpu_info.IsMali();
  operation.code_ = GetCode(block_size, data_type, dummy_checks);
  operation.flops_ =
      GetConvolutionFlops(conv_output->tensor.shape, conv_weights.shape);
  if (gpu_info.IsMali() || gpu_info.IsPowerVR()) {
    operation.compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
  const std::string op_name = absl::StrCat("convolution_2d (+ concat input) ",
                                           concat_node->id, " ", conv_node->id);
  model_builder->AddGpuOperation(
      std::vector<ValueId>(
          {concat_inputs[0]->id, concat_inputs[1]->id, concat_inputs[2]->id}),
      std::vector<ValueId>({conv_output->id}),
      std::make_unique<ConcatConv>(std::move(operation)), op_name);

  new_consumed_nodes->insert(concat_node->id);
  new_consumed_nodes->insert(conv_node->id);
  return absl::OkStatus();
}

}  // namespace ml_drift
