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

#include "ml_drift/common/gpu_weights_conversion_test_util.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {

// This function first sets up the quantization parameters, int8 weights,
// and then calculates the floating point dequantized weights to achieve the
// complete match of the weights values in the float and int8 execution
// scenarios.
void InitializeQuantizedWeights(
    int input_channels, int output_channels,
    std::vector<int8_t>& weights_data_int8, std::vector<float>& scale_data,
    std::vector<int32_t>& zero_point_data,
    std::vector<float>* absl_nullable weights_data_float32_ptr,
    DataType quantization_type) {
  int weights_size = input_channels * output_channels;
  scale_data.resize(output_channels);
  zero_point_data.resize(output_channels);

  for (int i = 0; i < output_channels; ++i) {
    scale_data[i] = sin(i) * 1.1f + 2.2f;
    zero_point_data[i] = i;
  }

  if (quantization_type == DataType::INT8) {
    weights_data_int8.resize(weights_size + XNN_EXTRA_BYTES / sizeof(int8_t));
    for (int i = 0; i < weights_size; ++i) {
      weights_data_int8[i] = i % 256 - 128;
    }
  } else {
    const int kNumElementsPerInt8 =
        (SizeInBitsOf(DataType::INT8) / SizeInBitsOf(DataType::INT4));
    ABSL_CHECK_EQ(input_channels % kNumElementsPerInt8, 0)
        << "Only an even number of input channels is supported.";
    const int weights_data_size = weights_size / kNumElementsPerInt8;
    weights_data_int8.resize(weights_data_size +
                             XNN_EXTRA_BYTES / sizeof(int8_t));
    for (int i = 0; i < output_channels; ++i) {
      for (int j = 0; j < input_channels; j += kNumElementsPerInt8) {
        int8_t lower_int4 = (i * input_channels + j) % 16 - 8;
        int8_t upper_int4 = (i * input_channels + j + 1) % 16 - 8;
        weights_data_int8[(i * input_channels + j) / kNumElementsPerInt8] =
            (upper_int4 << 4) | (lower_int4 & 0x0F);
      }
    }
  }

  // Get Float32 weights via dequantizing the Int8 weights.
  if (weights_data_float32_ptr != nullptr) {
    weights_data_float32_ptr->resize(weights_size +
                                     XNN_EXTRA_BYTES / sizeof(float));
    for (int o = 0; o < output_channels; ++o) {
      float scale = scale_data[o];
      int32_t zero_point = zero_point_data[o];
      for (int i = 0; i < input_channels; ++i) {
        int8_t src;
        if (quantization_type == DataType::INT8) {
          src = weights_data_int8[o * input_channels + i];
        } else {
          const int kNumElementsPerInt8 =
              (SizeInBitsOf(DataType::INT8) / SizeInBitsOf(DataType::INT4));
          int8_t source =
              weights_data_int8[(o * input_channels + i) / kNumElementsPerInt8];
          if (i % 2 == 0) {
            src = source & 0x0F;
          } else {
            src = source >> 4;
          }
        }
        float dequant = scale * (src - zero_point);
        weights_data_float32_ptr->at(o * input_channels + i) = dequant;
      }
    }
  }
}

void InitializeWeightsSumIData(std::vector<int8_t>& weights_data,
                               std::vector<int32_t>& reference_output,
                               const OHWI& weights_shape,
                               const DataType input_data_type) {
  reference_output.resize(weights_shape.o);
  if (input_data_type == DataType::INT8) {
    weights_data.resize(weights_shape.DimensionsProduct() +
                        XNN_EXTRA_BYTES / sizeof(int8_t));
    for (int i = 0; i < weights_shape.o; ++i) {
      int sum = 0;
      for (int j = 0; j < weights_shape.i; ++j) {
        int8_t value = i % 256 - 128;
        sum += value;
        weights_data[i * weights_shape.i + j] = value;
      }
      reference_output[i] = sum;
    }
  } else if (input_data_type == DataType::INT4) {
    const int kNumElementsPerInt8 =
        (SizeInBitsOf(DataType::INT8) / SizeInBitsOf(input_data_type));
    ABSL_CHECK_EQ(weights_shape.i % kNumElementsPerInt8, 0)
        << "Only an even number of input channels is supported.";
    const int weights_data_size =
        weights_shape.DimensionsProduct() / kNumElementsPerInt8;
    weights_data.resize(weights_data_size + XNN_EXTRA_BYTES / sizeof(int8_t));
    for (int i = 0; i < weights_shape.o; ++i) {
      int sum = 0;
      for (int j = 0; j < weights_shape.i; j += kNumElementsPerInt8) {
        int8_t lower_int4 = (i * weights_shape.i + j) % 16 - 8;
        int8_t upper_int4 = (i * weights_shape.i + j + 1) % 16 - 8;
        sum += lower_int4 + upper_int4;
        weights_data[(i * weights_shape.i + j) / kNumElementsPerInt8] =
            (upper_int4 << 4) | (lower_int4 & 0x0F);
      }
      reference_output[i] = sum;
    }
  } else {
    ABSL_LOG(FATAL) << absl::StrCat("Unsupported input data type: ",
                                       ToString(input_data_type));
  }
}

absl::StatusOr<GraphFloat32> CreateConvGraph(const BHWC& input_shape,
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
  auto& conv_weights =
      conv_attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
  conv_weights.shape = weights_shape;
  conv_weights.data.resize(weights_shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  // TODO: b/410586700 - Remove the following workaround once the bug is fixed,
  // and only set one field for the weights.
  // The weights.spanned_data field is designed to be a view of an external
  // memory, so the unit tests should maintain a memory for it. At the same
  // time, computing without WeightsManager (for result reference) requires
  // allocating memory for the weights.data. As a result, we have the
  // spanned_data be a view onto the allocated weights.data, rather than
  // maintaining a duplicate copy.
  conv_weights.spanned_data = absl::MakeSpan(conv_weights.data);
  InitWithSinValues(conv_weights.data);
  conv_attr.bias.shape = Linear(output_shape.c);
  conv_attr.bias.data.resize(conv_attr.bias.shape.DimensionsProduct());
  InitWithSinValues(conv_attr.bias.data);

  GraphFloat32 graph;
  auto input = graph.NewValue();
  input->tensor.type = DataType::FLOAT32;
  input->tensor.shape = input_shape;
  auto conv_node = graph.NewNode();
  conv_node->operation.type = ToString(OperationType::CONVOLUTION_2D);
  conv_node->operation.attributes = std::move(conv_attr);
  graph.AddConsumer(conv_node->id, input->id);
  Value* conv_output = nullptr;
  ABSL_RETURN_IF_ERROR(AddOutput(&graph, conv_node, &conv_output));
  conv_output->tensor.type = DataType::FLOAT32;
  conv_output->tensor.shape = output_shape;
  return graph;
}

absl::StatusOr<GraphFloat32> FCInt8TestGraph::CreateFCFloat32Graph() {
    FullyConnectedAttributes fc_attr;
    fc_attr.weights.shape = OHWI(output_shape_.c, 1, 1, input_shape_.c);
    fc_attr.weights.data.resize(weights_data_float32_.size());
    memcpy(fc_attr.weights.data.data(), weights_data_float32_.data(),
           weights_data_float32_.size() * sizeof(float));

    fc_attr.bias.shape = Linear(output_shape_.c);
    fc_attr.bias.data.resize(output_shape_.c);
    InitWithSinValues(fc_attr.bias.data);

    GraphFloat32 graph;
    auto input = graph.NewValue();
    input->tensor.type = DataType::FLOAT32;
    input->tensor.shape = input_shape_;
    auto fc_node = graph.NewNode();
    fc_node->operation.type = ToString(OperationType::FULLY_CONNECTED);
    fc_node->operation.attributes = fc_attr;
    graph.AddConsumer(fc_node->id, input->id);
    Value* fc_output = nullptr;
    ABSL_RETURN_IF_ERROR(AddOutput(&graph, fc_node, &fc_output));
    fc_output->tensor.type = DataType::FLOAT32;
    fc_output->tensor.shape = output_shape_;
    return graph;
  }

  absl::StatusOr<GraphFloat32> FCInt8TestGraph::CreateFCInt8Graph() {
    FullyConnectedInt8Attributes fc_attr;
    fc_attr.weights.shape = OHWI(output_shape_.c, 1, 1, input_shape_.c);
    fc_attr.weights.spanned_data = absl::MakeSpan(weights_data_int8_);
    fc_attr.scale.shape = OHWI(output_shape_.c, 1, 1, 1);
    fc_attr.scale.spanned_data = absl::MakeSpan(scale_data_);
    fc_attr.zero_point.shape = OHWI(output_shape_.c, 1, 1, 1);
    fc_attr.zero_point.spanned_data = absl::MakeSpan(zero_point_data_);
    fc_attr.bias.shape = Linear(output_shape_.c);
    fc_attr.bias.data.resize(output_shape_.c);
    InitWithSinValues(fc_attr.bias.data);

    GraphFloat32 graph;
    auto input = graph.NewValue();
    input->tensor.type = DataType::FLOAT32;
    input->tensor.shape = input_shape_;
    auto fc_node = graph.NewNode();
    fc_node->operation.type = ToString(OperationType::FULLY_CONNECTED_INT8);
    fc_node->operation.attributes = fc_attr;
    graph.AddConsumer(fc_node->id, input->id);
    Value* fc_output = nullptr;
    ABSL_RETURN_IF_ERROR(AddOutput(&graph, fc_node, &fc_output));
    fc_output->tensor.type = DataType::FLOAT32;
    fc_output->tensor.shape = output_shape_;
    return graph;
  }

}  // namespace ml_drift
