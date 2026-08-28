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

#ifndef ML_DRIFT_COMMON_TASK_TESTING_UTIL_H_
#define ML_DRIFT_COMMON_TASK_TESTING_UTIL_H_

#include <stdint.h>

#include <array>
#include <initializer_list>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/log/absl_log.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/profiling_info.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {
class TestExecutionEnvironment {
 public:
  TestExecutionEnvironment() = default;
  virtual ~TestExecutionEnvironment() = default;

  virtual std::vector<DataType> GetSupportedDataTypes() const = 0;
  virtual std::vector<TensorStorageType> GetSupportedStorages(
      DataType data_type) const = 0;

  virtual const GpuInfo& GetGpuInfo() const = 0;

  std::string SkipTestMessage() const {
    return "Not supported. Skipping test.";
  }

  // Check if the environment supports the given storage type and data type.
  bool IsStorageSupported(TensorStorageType storage_type,
                          DataType data_type) const;

  absl::Status ExecuteGPUOperation(
      const std::vector<TensorDescriptor*>& src_cpu,
      const std::vector<TensorDescriptor*>& dst_cpu,
      std::unique_ptr<GPUOperation>&& operation);

  template <typename DstTensorType>
  absl::Status ExecuteGpuModel(const std::vector<TensorFloat32>& src_cpu,
                               const std::vector<DstTensorType*>& dst_cpu,
                               GpuModel* gpu_model);

  template <typename DstTensorType>
  absl::Status ExecuteGPUOperation(const std::vector<TensorFloat32>& src_cpu,
                                   std::unique_ptr<GPUOperation>&& operation,
                                   const std::vector<BHWC>& dst_sizes,
                                   const std::vector<DstTensorType*>& dst_cpu);

  absl::Status ExecuteGPUOperation(
      const std::vector<TensorFloat32>& src_cpu,
      std::unique_ptr<GPUOperation>&& operation,
      const std::vector<BHWC>& dst_sizes,
      const std::initializer_list<TensorFloat32*>& dst_cpu) {
    return ExecuteGPUOperation(src_cpu, std::move(operation), dst_sizes,
                               std::vector<TensorFloat32*>(dst_cpu));
  }

  absl::Status ExecuteGPUOperation(
      const std::vector<Tensor5DFloat32>& src_cpu,
      std::unique_ptr<GPUOperation>&& operation,
      const std::vector<BHWDC>& dst_sizes,
      const std::vector<Tensor5DFloat32*>& dst_cpu);

  absl::Status ExecuteGPUOperation(const TensorFloat32& src_cpu,
                                   std::unique_ptr<GPUOperation>&& operation,
                                   const BHWC& dst_size,
                                   TensorFloat32* result) {
    return ExecuteGPUOperation(std::vector<TensorFloat32>{src_cpu},
                               std::move(operation), dst_size, result);
  }

  absl::Status ExecuteGPUOperation(const Tensor5DFloat32& src_cpu,
                                   std::unique_ptr<GPUOperation>&& operation,
                                   const BHWDC& dst_size,
                                   Tensor5DFloat32* result) {
    return ExecuteGPUOperation(std::vector<Tensor5DFloat32>{src_cpu},
                               std::move(operation), dst_size, result);
  }

  absl::Status ExecuteGPUOperation(const std::vector<TensorFloat32>& src_cpu,
                                   std::unique_ptr<GPUOperation>&& operation,
                                   const BHWC& dst_size,
                                   TensorFloat32* result) {
    return ExecuteGPUOperation(
        std::vector<TensorFloat32>{src_cpu}, std::move(operation),
        std::vector<BHWC>{dst_size}, std::vector<TensorFloat32*>{result});
  }

  absl::Status ExecuteGPUOperation(const std::vector<Tensor5DFloat32>& src_cpu,
                                   std::unique_ptr<GPUOperation>&& operation,
                                   const BHWDC& dst_size,
                                   Tensor5DFloat32* result) {
    return ExecuteGPUOperation(
        std::vector<Tensor5DFloat32>{src_cpu}, std::move(operation),
        std::vector<BHWDC>{dst_size}, std::vector<Tensor5DFloat32*>{result});
  }

  virtual absl::StatusOr<ProfilingInfo> GetGpuOperationTimeMs(
      const std::vector<TensorFloat32>& src_cpu,
      std::unique_ptr<GPUOperation>&& operation,
      const std::vector<BHWC>& dst_sizes, int num_repeats) {
    return absl::UnimplementedError("Not implemented");
  };

 protected:
  virtual absl::Status ExecuteGpuOperationInternal(
      const std::vector<TensorDescriptor*>& src_cpu,
      const std::vector<TensorDescriptor*>& dst_cpu,
      std::unique_ptr<GPUOperation>&& operation) = 0;
};

constexpr std::array<DataType, 2> GetFloatTypes() {
  return {DataType::FLOAT16, DataType::FLOAT32};
}

class FloatTest : public testing::Test,
                  public testing::WithParamInterface<
                      std::tuple<CalculationsPrecision, TensorStorageType>> {
  // Convenience accessor methods
 protected:
  CalculationsPrecision precision() const { return std::get<0>(GetParam()); }
  TensorStorageType storage() const { return std::get<1>(GetParam()); }
};
std::string ToString(
    const std::tuple<CalculationsPrecision, TensorStorageType>& info);

class DataTypeTest : public testing::Test,
                     public testing::WithParamInterface<
                         std::tuple<DataType, TensorStorageType>> {
  // Convenience accessor methods
 protected:
  DataType data_type() const { return std::get<0>(GetParam()); }
  TensorStorageType storage() const { return std::get<1>(GetParam()); }
};
std::string ToString(const std::tuple<DataType, TensorStorageType>& info);

inline std::string GetShapeName(const BHWC& shape) {
  return absl::StrCat(shape.b, "_", shape.h, "_", shape.w, "_", shape.c);
}

inline std::string GetShapeName(const OHWI& shape) {
  return absl::StrCat(shape.o, "_", shape.h, "_", shape.w, "_", shape.i);
}

// GenerateData fcns can take a number of generators.
// Use std::mt19937 for deterministic generation.
// Use absl::BitGen for random generation.
template <typename T>
void GenerateData(absl::BitGenRef gen, T* data, int size) {
  ABSL_LOG(FATAL) << "Unsupported tensor type";
}

template <>
void GenerateData(absl::BitGenRef gen, float* data, int size);
template <>
void GenerateData(absl::BitGenRef gen, int32_t* data, int size);
template <>
void GenerateData(absl::BitGenRef gen, int16_t* data, int size);
template <>
void GenerateData(absl::BitGenRef gen, int8_t* data, int size);
template <>
void GenerateData(absl::BitGenRef gen, uint32_t* data, int size);
template <>
void GenerateData(absl::BitGenRef gen, uint16_t* data, int size);
template <>
void GenerateData(absl::BitGenRef gen, uint8_t* data, int size);
template <>
void GenerateData(absl::BitGenRef gen, bool* data, int size);

template <DataType DataTypeT = DataType::FLOAT32, typename ShapeT>
ml_drift::Tensor<ShapeT, DataTypeT> MakeSyntheticTensor(
    const ShapeT& shape) {
  ml_drift::Tensor<ShapeT, DataTypeT> tensor;
  tensor.shape = shape;
  tensor.data.resize(shape.DimensionsProduct());
  absl::BitGen gen;
  GenerateData(gen, tensor.data.data(), tensor.data.size());
  return tensor;
}

template <typename ShapeT>
ml_drift::Tensor<ShapeT, DataType::FLOAT32> MakeZeroTensor(
    const ShapeT& shape) {
  ml_drift::Tensor<ShapeT, DataType::FLOAT32> tensor;
  tensor.shape = shape;
  tensor.data = std::vector<float>(shape.DimensionsProduct(), 0);
  return tensor;
}

float GetEpsilon(CalculationsPrecision precision, const GpuInfo& gpu_info,
                 const Convolution2DAttributes& attr);

float GetEpsilon(CalculationsPrecision precision, const GpuInfo& gpu_info,
                 const Convolution3DAttributes& attr);

float GetEpsilon(CalculationsPrecision precision, const GpuInfo& gpu_info,
                 const FullyConnectedAttributes& attr);

float GetEpsilon(CalculationsPrecision precision, const GpuInfo& gpu_info,
                 const DepthwiseConvolution2DAttributes& attr);

float GetEpsilon(CalculationsPrecision precision, const GpuInfo& gpu_info,
                 const DepthwiseConvolution3DAttributes& attr);

float GetEpsilon(CalculationsPrecision precision, const GpuInfo& gpu_info,
                 const ConvolutionTransposedAttributes& attr);

float GetEpsilon(CalculationsPrecision precision, const GpuInfo& gpu_info,
                 const ConvolutionTransposed3DAttributes& attr);

// Generates group ids for a tensor of shape BHWC where the last dimension is
// the number of active groups. The ids are generated such that each active
// group has a unique id and the ids are sorted in ascending order. Ids range
// from 0 to num_groups - 1.
TensorInt32 GenerateGroupIds(const BHWC& size, int num_groups);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_TASK_TESTING_UTIL_H_
