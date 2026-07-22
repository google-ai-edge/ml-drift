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

#include "ml_drift/common/task/testing_util.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace {
// MAD is multiply-add operation and GPU mostly do MAD as one elementary op.
int GetMADSAmount(const OHWDI& shape) {
  return shape.w * shape.h * shape.d * shape.i + 1 /*bias add*/;
}

int GetMADSAmount(const OHWI& shape) {
  return shape.w * shape.h * shape.i + 1 /*bias add*/;
}

// for ConvolutionTransposed amount of mads can be different for different
// pixels.
int GetMaxMADSAmount(const ConvolutionTransposedAttributes& attr) {
  const int max_kernel_w = DivideRoundUp(attr.weights.shape.w, attr.stride.w);
  const int max_kernel_h = DivideRoundUp(attr.weights.shape.h, attr.stride.h);
  const int num_slices = DivideRoundUp(attr.weights.shape.i, 4);
  return max_kernel_w * max_kernel_h * num_slices * 4 + 1 /*bias add*/;
}

int GetMaxMADSAmount(const ConvolutionTransposed3DAttributes& attr) {
  const int max_kernel_w = DivideRoundUp(attr.weights.shape.w, attr.stride.w);
  const int max_kernel_h = DivideRoundUp(attr.weights.shape.h, attr.stride.h);
  const int max_kernel_d = DivideRoundUp(attr.weights.shape.d, attr.stride.d);
  const int num_slices = DivideRoundUp(attr.weights.shape.i, 4);
  return max_kernel_w * max_kernel_h * max_kernel_d * num_slices * 4 +
         1 /*bias add*/;
}
}  // namespace

bool TestExecutionEnvironment::IsStorageSupported(
    TensorStorageType storage_type, DataType data_type) const {
  std::vector<TensorStorageType> supported_storages =
      GetSupportedStorages(data_type);
  if (std::find(supported_storages.begin(), supported_storages.end(),
                storage_type) == supported_storages.end()) {
    return false;
  }
  return true;
}

absl::Status TestExecutionEnvironment::ExecuteGPUOperation(
    const std::vector<TensorDescriptor*>& src_cpu,
    const std::vector<TensorDescriptor*>& dst_cpu,
    std::unique_ptr<GPUOperation>&& operation) {
  for (int i = 0; i < src_cpu.size(); ++i) {
    auto src_shape = src_cpu[i]->GetBHWDCShape();
    TensorDescriptor tensor_desc;
    ABSL_RETURN_IF_ERROR(operation->GetTensorDescriptor(
        operation->GetSrcTensorsNames()[i], &tensor_desc));
    if (src_shape.b != 1 && !tensor_desc.HasAxis(Axis::BATCH)) {
      return absl::InvalidArgumentError(
          "Layout doesn't have Batch dimension, but shape.b != 1");
    }
  }

  for (int i = 0; i < dst_cpu.size(); ++i) {
    auto dst_shape = dst_cpu[i]->GetBHWDCShape();
    TensorDescriptor tensor_desc;
    ABSL_RETURN_IF_ERROR(operation->GetTensorDescriptor(
        operation->GetDstTensorsNames()[i], &tensor_desc));
    if (dst_shape.b != 1 && !tensor_desc.HasAxis(Axis::BATCH)) {
      return absl::InvalidArgumentError(
          "Layout doesn't have Batch dimension, but shape.b != 1");
    }
    if (dst_cpu[i]->GetDataType() == DataType::UNKNOWN) {
      *dst_cpu[i] = tensor_desc;
    }
    dst_cpu[i]->SetBHWDCShape(dst_shape);
  }
  ABSL_RETURN_IF_ERROR(operation->AssembleCode(GetGpuInfo()));
  return ExecuteGpuOperationInternal(src_cpu, dst_cpu, std::move(operation));
}

template <typename DstTensorType>
absl::Status TestExecutionEnvironment::ExecuteGPUOperation(
    const std::vector<TensorFloat32>& src_cpu,
    std::unique_ptr<GPUOperation>&& operation,
    const std::vector<BHWC>& dst_sizes,
    const std::vector<DstTensorType*>& dst_cpu) {
  std::vector<TensorDescriptor> src_cpu_descs(src_cpu.size());
  std::vector<TensorDescriptor*> src_cpu_desc_ptrs(src_cpu.size());
  for (int i = 0; i < src_cpu.size(); ++i) {
    ABSL_RETURN_IF_ERROR(operation->GetTensorDescriptor(
        operation->GetSrcTensorsNames()[i], &src_cpu_descs[i]));
    src_cpu_descs[i].UploadData(src_cpu[i]);
    src_cpu_desc_ptrs[i] = &src_cpu_descs[i];
  }
  std::vector<TensorDescriptor> dst_cpu_descs(dst_cpu.size());
  std::vector<TensorDescriptor*> dst_cpu_desc_ptrs(dst_cpu.size());
  for (int i = 0; i < dst_cpu.size(); ++i) {
    ABSL_RETURN_IF_ERROR(operation->GetTensorDescriptor(
        operation->GetDstTensorsNames()[i], &dst_cpu_descs[i]));
    dst_cpu_descs[i].SetBHWCShape(dst_sizes[i]);
    dst_cpu_desc_ptrs[i] = &dst_cpu_descs[i];
  }

  ABSL_RETURN_IF_ERROR(ExecuteGPUOperation(src_cpu_desc_ptrs, dst_cpu_desc_ptrs,
                                           std::move(operation)));

  for (int i = 0; i < dst_cpu.size(); ++i) {
    dst_cpu_descs[i].DownloadData(dst_cpu[i]);
  }
  return absl::OkStatus();
}

template absl::Status TestExecutionEnvironment::ExecuteGPUOperation(
    const std::vector<TensorFloat32>& src_cpu,
    std::unique_ptr<GPUOperation>&& operation,
    const std::vector<BHWC>& dst_sizes,
    const std::vector<TensorFloat32*>& dst_cpu);

template absl::Status TestExecutionEnvironment::ExecuteGPUOperation(
    const std::vector<TensorFloat32>& src_cpu,
    std::unique_ptr<GPUOperation>&& operation,
    const std::vector<BHWC>& dst_sizes,
    const std::vector<TensorInt32*>& dst_cpu);

absl::Status TestExecutionEnvironment::ExecuteGPUOperation(
    const std::vector<Tensor5DFloat32>& src_cpu,
    std::unique_ptr<GPUOperation>&& operation,
    const std::vector<BHWDC>& dst_sizes,
    const std::vector<Tensor5DFloat32*>& dst_cpu) {
  std::vector<TensorDescriptor> src_cpu_descs(src_cpu.size());
  std::vector<TensorDescriptor*> src_cpu_desc_ptrs(src_cpu.size());
  for (int i = 0; i < src_cpu.size(); ++i) {
    ABSL_RETURN_IF_ERROR(operation->GetTensorDescriptor(
        operation->GetSrcTensorsNames()[i], &src_cpu_descs[i]));
    src_cpu_descs[i].UploadData(src_cpu[i]);
    src_cpu_desc_ptrs[i] = &src_cpu_descs[i];
  }
  std::vector<TensorDescriptor> dst_cpu_descs(dst_cpu.size());
  std::vector<TensorDescriptor*> dst_cpu_desc_ptrs(dst_cpu.size());
  for (int i = 0; i < dst_cpu.size(); ++i) {
    ABSL_RETURN_IF_ERROR(operation->GetTensorDescriptor(
        operation->GetDstTensorsNames()[i], &dst_cpu_descs[i]));
    dst_cpu_descs[i].SetBHWDCShape(dst_sizes[i]);
    dst_cpu_desc_ptrs[i] = &dst_cpu_descs[i];
  }

  ABSL_RETURN_IF_ERROR(ExecuteGPUOperation(src_cpu_desc_ptrs, dst_cpu_desc_ptrs,
                                           std::move(operation)));

  for (int i = 0; i < dst_cpu.size(); ++i) {
    dst_cpu_descs[i].DownloadData(dst_cpu[i]);
  }
  return absl::OkStatus();
}

std::string ToString(
    const std::tuple<CalculationsPrecision, TensorStorageType>& info) {
  // Test names can only contain alphanumeric
  // characters and underscores. ToString(storage_type)
  // = "TensorStorageType::x" ToString(precision) =
  // "CalculationsPrecision::x" We get rid of the
  // colons with StrReplaceAll.
  const std::string name = absl::StrCat(ToString(std::get<0>(info)), "_",
                                        ToString(std::get<1>(info)));
  return absl::StrReplaceAll(name, {{":", ""}});
}

std::string ToString(const std::tuple<DataType, TensorStorageType>& info) {
  // Test names can only contain alphanumeric
  // characters and underscores. ToString(storage_type)
  // = "TensorStorageType::x". We get rid of the
  // colons with StrReplaceAll.
  const std::string name = absl::StrCat(ToString(std::get<0>(info)), "_",
                                        ToString(std::get<1>(info)));
  return absl::StrReplaceAll(name, {{":", ""}});
}

template <typename DstTensorType>
absl::Status TestExecutionEnvironment::ExecuteGpuModel(
    const std::vector<TensorFloat32>& src_cpu,
    const std::vector<DstTensorType*>& dst_cpu, GpuModel* gpu_model) {
  for (int i = 0; i < gpu_model->input_ids_and_refs.size(); ++i) {
    gpu_model->tensors[gpu_model->input_ids_and_refs[i].first].UploadData(
        src_cpu[i]);
  }

  for (int k = 0; k < gpu_model->nodes.size(); ++k) {
    auto& gpu_node = gpu_model->nodes[k];
    std::vector<TensorDescriptor*> src_descs(gpu_node.inputs.size());
    for (int i = 0; i < gpu_node.inputs.size(); ++i) {
      if (gpu_model->const_tensors.find(gpu_node.inputs[i]) !=
          gpu_model->const_tensors.end()) {
        src_descs[i] = &gpu_model->const_tensors[gpu_node.inputs[i]];
      } else {
        src_descs[i] = &gpu_model->tensors[gpu_node.inputs[i]];
      }
    }

    std::vector<TensorDescriptor*> dst_descs(gpu_node.outputs.size());
    for (int i = 0; i < gpu_node.outputs.size(); ++i) {
      dst_descs[i] = &gpu_model->tensors[gpu_node.outputs[i]];
    }

    ABSL_RETURN_IF_ERROR(ExecuteGpuOperationInternal(
        src_descs, dst_descs, std::move(gpu_model->nodes[k].gpu_operation)));
  }
  for (int i = 0; i < gpu_model->output_ids_and_refs.size(); ++i) {
    gpu_model->tensors[gpu_model->output_ids_and_refs[i].first].DownloadData(
        dst_cpu[i]);
  }
  return absl::OkStatus();
}

template absl::Status TestExecutionEnvironment::ExecuteGpuModel(
    const std::vector<TensorFloat32>& src_cpu,
    const std::vector<Tensor<BHWC, DataType::FLOAT32>*>& dst_cpu,
    GpuModel* gpu_model);

template absl::Status TestExecutionEnvironment::ExecuteGpuModel(
    const std::vector<TensorFloat32>& src_cpu,
    const std::vector<Tensor<BHWC, DataType::INT32>*>& dst_cpu,
    GpuModel* gpu_model);

template <>
void GenerateData(absl::BitGenRef gen, float* data, int size) {
  for (int i = 0; i < size; ++i) {
    data[i] = absl::Uniform<float>(gen, -1.0f, 1.0f);
  }
}

template <>
void GenerateData(absl::BitGenRef gen, int32_t* data, int size) {
  for (int i = 0; i < size; ++i) {
    data[i] = absl::Uniform<int32_t>(gen, -128, 127);
  }
}

template <>
void GenerateData(absl::BitGenRef gen, int16_t* data, int size) {
  for (int i = 0; i < size; ++i) {
    data[i] = absl::Uniform<int16_t>(gen, -128, 127);
  }
}

template <>
void GenerateData(absl::BitGenRef gen, int8_t* data, int size) {
  for (int i = 0; i < size; ++i) {
    data[i] = absl::Uniform<int8_t>(gen, -128, 127);
  }
}

template <>
void GenerateData(absl::BitGenRef gen, uint32_t* data, int size) {
  for (int i = 0; i < size; ++i) {
    data[i] = absl::Uniform<uint32_t>(gen, 0, 255);
  }
}

template <>
void GenerateData(absl::BitGenRef gen, uint16_t* data, int size) {
  for (int i = 0; i < size; ++i) {
    data[i] = absl::Uniform<uint16_t>(gen, 0, 255);
  }
}

template <>
void GenerateData(absl::BitGenRef gen, uint8_t* data, int size) {
  for (int i = 0; i < size; ++i) {
    data[i] = absl::Uniform<uint8_t>(gen, 0, 255);
  }
}

template <>
void GenerateData(absl::BitGenRef gen, bool* data, int size) {
  for (int i = 0; i < size; ++i) {
    data[i] = absl::Bernoulli(gen, 0.5);
  }
}

float GetEpsilon(CalculationsPrecision precision, const GpuInfo& gpu_info,
                 const Convolution2DAttributes& attr) {
  return GetEpsilon(precision, gpu_info) *
         GetMADSAmount(
             std::visit([](const auto& w) { return w.shape; }, attr.weights));
}

float GetEpsilon(CalculationsPrecision precision, const GpuInfo& gpu_info,
                 const Convolution3DAttributes& attr) {
  return GetEpsilon(precision, gpu_info) * GetMADSAmount(attr.weights.shape);
}

float GetEpsilon(CalculationsPrecision precision, const GpuInfo& gpu_info,
                 const FullyConnectedAttributes& attr) {
  return GetEpsilon(precision, gpu_info) * GetMADSAmount(attr.weights.shape);
}

float GetEpsilon(CalculationsPrecision precision, const GpuInfo& gpu_info,
                 const DepthwiseConvolution2DAttributes& attr) {
  const auto& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  return GetEpsilon(precision, gpu_info) *
         (weights_shape.h * weights_shape.w + 1);
}

float GetEpsilon(CalculationsPrecision precision, const GpuInfo& gpu_info,
                 const DepthwiseConvolution3DAttributes& attr) {
  return GetEpsilon(precision, gpu_info) *
         (attr.weights.shape.h * attr.weights.shape.w * attr.weights.shape.d +
          1);
}

float GetEpsilon(CalculationsPrecision precision, const GpuInfo& gpu_info,
                 const ConvolutionTransposedAttributes& attr) {
  return GetEpsilon(precision, gpu_info) * GetMaxMADSAmount(attr);
}

float GetEpsilon(CalculationsPrecision precision, const GpuInfo& gpu_info,
                 const ConvolutionTransposed3DAttributes& attr) {
  return GetEpsilon(precision, gpu_info) * GetMaxMADSAmount(attr);
}

TensorInt32 GenerateGroupIds(const BHWC& size, int num_groups) {
  const int num_active_groups = size.c;
  TensorInt32 group_ids;
  group_ids.shape = size;
  group_ids.data.resize(group_ids.shape.DimensionsProduct());
  absl::BitGen gen;
  for (int b = 0; b < size.b; ++b) {
    for (int h = 0; h < size.h; ++h) {
      for (int w = 0; w < size.w; ++w) {
        int count = 0;
        for (int i = 0; i < num_groups && count < num_active_groups; ++i) {
          if (absl::Uniform(gen, 0.0, 1.0) <
              static_cast<double>(num_active_groups - count) /
                  (num_groups - i)) {
            group_ids.data[group_ids.shape.LinearIndex({b, h, w, count})] = i;
            count++;
          }
        }
      }
    }
  }
  return group_ids;
}

}  // namespace ml_drift
