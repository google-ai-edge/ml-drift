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

#include "ml_drift/cl/testing/cl_test.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "ml_drift/cl/cl_operation.h"
#include "ml_drift/cl/environment.h"
#include "ml_drift/cl/tensor.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/profiling_info.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {
namespace cl {

absl::Status ClExecutionEnvironment::Init() { return CreateEnvironment(&env_); }

std::vector<DataType> ClExecutionEnvironment::GetSupportedDataTypes() const {
  std::vector<DataType> data_types = {DataType::FLOAT32};
  if (env_.GetDevicePtr()->GetInfo().SupportsFP16()) {
    data_types.push_back(DataType::FLOAT16);
  }
  return data_types;
}

std::vector<TensorStorageType> ClExecutionEnvironment::GetSupportedStorages(
    DataType data_type) const {
  if (data_type == DataType::FLOAT16 &&
      !env_.GetDevicePtr()->GetInfo().SupportsFP16()) {
    return {};
  }
  return env_.GetSupportedStorages();
}

const GpuInfo& ClExecutionEnvironment::GetGpuInfo() const {
  return env_.GetDevicePtr()->GetInfo();
}

absl::Status ClExecutionEnvironment::ExecuteGpuOperationInternal(
    const std::vector<TensorDescriptor*>& src_cpu,
    const std::vector<TensorDescriptor*>& dst_cpu,
    std::unique_ptr<GPUOperation>&& operation) {
  ClOperation cl_op;
  cl_op.Init(std::move(operation));
  ABSL_RETURN_IF_ERROR(cl_op.Compile(env_.GetDevicePtr(), &env_.context(),
                                     env_.program_cache()));
  std::vector<Tensor> src(src_cpu.size());
  for (int i = 0; i < src_cpu.size(); ++i) {
    ABSL_RETURN_IF_ERROR(
        src[i].CreateFromDescriptor(*src_cpu[i], env_.context()));
    ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(i, &src[i]));
  }

  std::vector<Tensor> dst(dst_cpu.size());
  for (int i = 0; i < dst_cpu.size(); ++i) {
    ABSL_RETURN_IF_ERROR(
        dst[i].CreateFromDescriptor(*dst_cpu[i], env_.context()));
    ABSL_RETURN_IF_ERROR(cl_op.SetDstTensor(i, &dst[i]));
  }
  ABSL_RETURN_IF_ERROR(cl_op.UpdateParams());
  ABSL_RETURN_IF_ERROR(cl_op.AddToQueue(env_.queue()));
  ABSL_RETURN_IF_ERROR(env_.queue()->WaitForCompletion());

  for (int i = 0; i < dst_cpu.size(); ++i) {
    ABSL_RETURN_IF_ERROR(dst[i].ToDescriptor(dst_cpu[i], env_.queue()));
  }
  return absl::OkStatus();
}

absl::StatusOr<ProfilingInfo> ClExecutionEnvironment::GetGpuOperationTimeMs(
    const std::vector<TensorFloat32>& src_cpu,
    std::unique_ptr<GPUOperation>&& operation,
    const std::vector<BHWC>& dst_sizes, int num_repeats) {
  std::vector<TensorDescriptor> src_cpu_descs(src_cpu.size());
  for (int i = 0; i < src_cpu.size(); ++i) {
    ABSL_RETURN_IF_ERROR(operation->GetTensorDescriptor(
        operation->GetSrcTensorsNames()[i], &src_cpu_descs[i]));
    src_cpu_descs[i].UploadData(src_cpu[i]);
  }
  std::vector<TensorDescriptor> dst_cpu_descs(dst_sizes.size());
  for (int i = 0; i < dst_sizes.size(); ++i) {
    ABSL_RETURN_IF_ERROR(operation->GetTensorDescriptor(
        operation->GetDstTensorsNames()[i], &dst_cpu_descs[i]));
    dst_cpu_descs[i].SetBHWCShape(dst_sizes[i]);
  }

  ABSL_RETURN_IF_ERROR(operation->AssembleCode(GetGpuInfo()));
  ClOperation cl_op;
  cl_op.Init(std::move(operation));
  ABSL_RETURN_IF_ERROR(cl_op.Compile(env_.GetDevicePtr(), &env_.context(),
                                     env_.program_cache()));
  std::vector<Tensor> src(src_cpu.size());
  for (int i = 0; i < src_cpu.size(); ++i) {
    ABSL_RETURN_IF_ERROR(
        src[i].CreateFromDescriptor(src_cpu_descs[i], env_.context()));
    ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(i, &src[i]));
  }

  std::vector<Tensor> dst(dst_sizes.size());
  for (int i = 0; i < dst_sizes.size(); ++i) {
    ABSL_RETURN_IF_ERROR(
        dst[i].CreateFromDescriptor(dst_cpu_descs[i], env_.context()));
    ABSL_RETURN_IF_ERROR(cl_op.SetDstTensor(i, &dst[i]));
  }
  ABSL_RETURN_IF_ERROR(cl_op.UpdateParams());

  ABSL_RETURN_IF_ERROR(cl_op.Tune(TuningType::kExhaustive, GetGpuInfo(),
                                  env_.profiling_queue()));

  ProfilingInfo result;
  result.dispatches.resize(num_repeats);
  for (int i = 0; i < num_repeats; ++i) {
    ABSL_ASSIGN_OR_RETURN(
        result.dispatches[i].duration,
        cl_op.GetOpTime(GetGpuInfo(), env_.queue(), env_.profiling_queue()));
    result.dispatches[i].label = "test_op";

    uint64_t read_size = 0;
    for (const auto& src_tensor : src) {
      read_size += src_tensor.GetMemorySizeInBytes();
    }
    read_size += cl_op.GetConstArgsSize();
    uint64_t write_size = 0;
    for (const auto& dst_tensor : dst) {
      write_size += dst_tensor.GetMemorySizeInBytes();
    }
    result.dispatches[i].flops = cl_op.GetFlopsCount();
    result.dispatches[i].read_mem_size = read_size;
    result.dispatches[i].write_mem_size = write_size;
  }

  return result;
}

}  // namespace cl
}  // namespace ml_drift
