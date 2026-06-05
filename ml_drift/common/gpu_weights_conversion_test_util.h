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

#ifndef ML_DRIFT_COMMON_GPU_WEIGHTS_CONVERSION_TEST_UTIL_H_
#define ML_DRIFT_COMMON_GPU_WEIGHTS_CONVERSION_TEST_UTIL_H_


#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

inline void InitWithSinValues(std::vector<float>& data) {
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = std::sin(i * 0.1f);
  }
}

// This function first sets up the quantization parameters, int8 weights,
// and then calculates the floating point dequantized weights to achieve the
// complete match of the weights values in the float and int8 execution
// scenarios.
void InitializeQuantizedWeights(
    int input_channels, int output_channels,
    std::vector<int8_t>& weights_data_int8, std::vector<float>& scale_data,
    std::vector<int32_t>& zero_point_data,
    std::vector<float>* absl_nullable weights_data_float32_ptr = nullptr,
    DataType quantization_type = DataType::INT8);

void InitializeWeightsSumIData(std::vector<int8_t>& weights_data,
                               std::vector<int32_t>& reference_output,
                               const OHWI& weights_shape,
                               DataType input_data_type);

absl::StatusOr<GraphFloat32> CreateConvGraph(const BHWC& input_shape,
                                             const BHWC& output_shape,
                                             int kernel_size);


template <DataType QuantizedT>
absl::StatusOr<GraphFloat32> CreateQuantizedConvGraph(const BHWC& input_shape,
                                                      const BHWC& output_shape,
                                                      int kernel_size) {
  Convolution2DAttributes conv_attr;
  int padding_size = (kernel_size - 1) / 2;
  conv_attr.padding.prepended = HW(padding_size, padding_size);
  conv_attr.padding.appended = HW(padding_size, padding_size);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  auto weights_shape =
      OHWI(output_shape.c, kernel_size, kernel_size, input_shape.c);
  auto& conv_weights = conv_attr.weights.emplace<Tensor<OHWI, QuantizedT>>();
  conv_weights.shape = weights_shape;
  conv_attr.scale.shape = OHWI(output_shape.c, 1, 1, 1);
  conv_attr.zero_point.shape = OHWI(output_shape.c, 1, 1, 1);

  InitializeQuantizedWeights(input_shape.c * kernel_size * kernel_size,
                             output_shape.c, conv_weights.data,
                             conv_attr.scale.data, conv_attr.zero_point.data,
                             /*weights_data_float32_ptr=*/nullptr, QuantizedT);

  GraphFloat32 graph;
  auto input = graph.NewValue();
  input->tensor.type = DataType::FLOAT32;
  input->tensor.shape = input_shape;
  auto conv_node = graph.NewNode();
  conv_node->operation.type = ToString(OperationType::CONVOLUTION_2D);
  conv_node->operation.attributes = std::move(conv_attr);
  graph.AddConsumer(conv_node->id, input->id);
  Value* conv_output = nullptr;
  RETURN_IF_ERROR(AddOutput(&graph, conv_node, &conv_output));
  conv_output->tensor.type = DataType::FLOAT32;
  conv_output->tensor.shape = output_shape;
  return graph;
}

class FCInt8TestGraph {
 public:
  FCInt8TestGraph(const BHWC& input_shape, const BHWC& output_shape)
      : input_shape_(input_shape), output_shape_(output_shape) {
    InitializeQuantizedWeights(input_shape.c, output_shape.c,
                               weights_data_int8_, scale_data_,
                               zero_point_data_, &weights_data_float32_);
  }

  absl::StatusOr<GraphFloat32> CreateFCFloat32Graph();
  absl::StatusOr<GraphFloat32> CreateFCInt8Graph();

 private:
  BHWC input_shape_;
  BHWC output_shape_;
  std::vector<int8_t> weights_data_int8_;
  std::vector<float> scale_data_;
  std::vector<int32_t> zero_point_data_;
  std::vector<float> weights_data_float32_;
};

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_GPU_WEIGHTS_CONVERSION_TEST_UTIL_H_
