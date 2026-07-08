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

#include "ml_drift/metal/perf_util.h"

#import <Metal/Metal.h>

#include <iomanip>
#include <iostream>
#include <memory>
#include <utility>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/random/random.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/flops_util.h"
#include "ml_drift/common/kernels/conv_apple_mpp.h"
#include "ml_drift/common/kernels/conv_generic.h"
#include "ml_drift/common/kernels/conv_wave_matrix.h"
#include "ml_drift/common/kernels/conv_wave_memory.h"
#include "ml_drift/common/kernels/depthwise_conv_tiled.h"
#include "ml_drift/common/kernels/elementwise.h"
#include "ml_drift/common/kernels/fully_connected.h"
#include "ml_drift/common/kernels/quantize_and_dequantize.h"
#include "ml_drift/common/kernels/softmax.h"
#include "ml_drift/common/kernels/special/conv_softmax_conv.h"
#include "ml_drift/common/kernels/winograd.h"
#include "ml_drift/common/task/testing_ref_ops.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/metal/compute_task.h"
#include "ml_drift/metal/environment.h"
#include "ml_drift/metal/metal_spatial_tensor.h"

namespace ml_drift {
namespace metal {
namespace {
absl::Status TestConvolutionPerformance(const Convolution2DAttributes& attr, const BHWC& src_shape,
                                        const OperationDef& op_def,
                                        CalculationsPrecision precision) {
  const auto dst_shape = CalculateOutputShape(src_shape, attr);

  Environment env;
  std::unique_ptr<GPUOperation> conv;
  conv = std::make_unique<ConvGeneric>(
      CreateConvGeneric(env.GetInfo(), op_def, precision, attr, &dst_shape));

  // conv = std::make_unique<ConvAppleMPP>(
  //    CreateConvAppleMPP(op_def.src_tensors[0], op_def.dst_tensors[0],
  //                       std::get<Tensor<OHWI, DataType::FLOAT32>>(attr.weights)));

  // conv = std::make_unique<ConvWaveMemory>(
  //    CreateConvWaveMemory(env.GetInfo(), op_def, precision, attr, &dst_shape));

  // conv = std::make_unique<ConvWaveMatrix>(
  //     CreateConvWaveMatrix(op_def, precision, dst_shape, attr, env.GetInfo()));

  // {
  //   FullyConnectedAttributes fc_attr;
  //   fc_attr.weights = attr.weights;
  //   fc_attr.bias = attr.bias;
  //   conv = std::make_unique<FullyConnected>(
  //       CreateFullyConnected(env.GetInfo(), op_def, precision, fc_attr, &dst_shape));
  // }

  MetalSpatialTensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(src_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  descriptor_with_shape.SetBHWCShape(dst_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &dst));

  const auto w_shape =
      std::visit([](const auto& weights) -> const auto& { return weights.shape; }, attr.weights);

  std::cout << "Src size(HWC) - " << src_shape.h << "x" << src_shape.w << "x" << src_shape.c
            << std::endl;
  std::cout << "Dst size(HWC) - " << dst_shape.h << "x" << dst_shape.w << "x" << dst_shape.c
            << std::endl;
  std::cout << "Convolution attributes : " << std::endl;
  std::cout << "  Kernel size - " << w_shape.w << "x" << w_shape.h << std::endl;
  std::cout << "  Stride - " << attr.strides.w << "x" << attr.strides.h << std::endl;
  std::cout << "  Dilation - " << attr.dilations.w << "x" << attr.dilations.h << std::endl;
  std::cout << "  Padding (prepended) - " << attr.padding.prepended.w << "x"
            << attr.padding.prepended.h << ", (appended) - " << attr.padding.appended.w << "x"
            << attr.padding.appended.h << std::endl;

  RETURN_IF_ERROR(conv->AssembleCode(env.GetInfo()));

  const int float_size = precision == CalculationsPrecision::F32 ? 4 : 2;
  const int64_t flops_per_element = w_shape.i * w_shape.h * w_shape.w * 2;
  const int64_t dst_elements = dst.Width() * dst.Height() * dst.Channels();
  const int64_t flops_count = dst_elements * flops_per_element;
  const double gflops_count = flops_count * 1e-9;

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const int64_t dst_elements_alignedx4 = dst.Width() * dst.Height() * dst.Slices() * 4;
  const int64_t src_elements_alignedx4 = src.Width() * src.Height() * src.Slices() * 4;
  const double dst_gbytes = dst_elements_alignedx4 * float_size / kGByte;
  const double src_gbytes = src_elements_alignedx4 * float_size / kGByte;
  const double weight_gbytes = w_shape.DimensionsProduct() * float_size / kGByte;
  const double bias_gbytes = w_shape.o * float_size / kGByte;

  ComputeTask gpu_task;
  gpu_task.Init(std::move(conv));
  RETURN_IF_ERROR(gpu_task.Compile(&env));
  gpu_task.SetSrcTensor(&src, 0);
  gpu_task.SetDstTensor(&dst, 0);
  RETURN_IF_ERROR(gpu_task.UpdateParams());

  id<MTLCommandQueue> command_queue = [env.device() newCommandQueue];
  for (int i = 0; i < 10; ++i) {
    absl::Duration gpu_task_time = gpu_task.GetTaskTime(command_queue);
    double time_ms = absl::ToDoubleMilliseconds(gpu_task_time);

    const double fps = 1000.0 / time_ms;
    const double gflops_real = fps * gflops_count;
    const double gbyte_read = fps * (src_gbytes + weight_gbytes + bias_gbytes);
    const double gbyte_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(3) << i << "  Time - " << time_ms
              << std::setprecision(2) << "(ms), GFlops - " << gflops_real << ", Bandwidth - "
              << gbyte_read + gbyte_write << "(GB/s), Read - " << gbyte_read << "(GB/s), Write - "
              << gbyte_write << "(GB/s)" << std::endl;
  }

  return absl::OkStatus();
}

absl::Status TestConvSoftmaxConvPerformance(const BHWC& dst_shape, int src_ch0, int src_ch1,
                                            const OperationDef& op_def,
                                            CalculationsPrecision precision) {
  Environment device;

  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights0;
  weights0.shape = OHWI(src_ch1, 1, dst_shape.h, src_ch0);
  weights0.data.resize(weights0.shape.DimensionsProduct() + XNN_EXTRA_BYTES / sizeof(float));
  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights1;
  weights1.shape = OHWI(dst_shape.c, 1, dst_shape.h, src_ch1);
  weights1.data.resize(weights1.shape.DimensionsProduct() + XNN_EXTRA_BYTES / sizeof(float));

  ConvSoftmaxConv::WeightsDesc weights_desc;
  weights_desc.constant = true;
  weights_desc.weights0 = &weights0;
  weights_desc.weights1 = &weights1;
  std::unique_ptr<GPUOperation> conv = std::make_unique<ConvSoftmaxConv>(
      CreateConvSoftmaxConv(device.GetInfo(), op_def, precision, weights_desc));

  MetalSpatialTensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  const BHWC src_shape = BHWC(dst_shape.b, dst_shape.h, dst_shape.w, src_ch0);
  descriptor_with_shape.SetBHWCShape(src_shape);
  RETURN_IF_ERROR(CreateTensor(device.device(), descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  descriptor_with_shape.SetBHWCShape(dst_shape);
  RETURN_IF_ERROR(CreateTensor(device.device(), descriptor_with_shape, &dst));

  std::cout << "Src size(HWC) - " << src_shape.h << "x" << src_shape.w << "x" << src_shape.c
            << std::endl;
  std::cout << "Dst size(HWC) - " << dst_shape.h << "x" << dst_shape.w << "x" << dst_shape.c
            << std::endl;
  std::cout << "Attention attributes : " << std::endl;

  RETURN_IF_ERROR(conv->AssembleCode(device.GetInfo()));

  const int float_size = precision == CalculationsPrecision::F32 ? 4 : 2;
  const int64_t conv0_flops_per_element = weights0.shape.i * 2;
  const int64_t conv0_dst_elements = dst.Width() * dst.Height() * weights0.shape.o;
  const int64_t conv0_flops_count = conv0_dst_elements * conv0_flops_per_element;
  const int64_t conv1_flops_per_element = weights1.shape.i * 2;
  const int64_t conv1_dst_elements = dst.Width() * dst.Height() * weights1.shape.o;
  const int64_t conv1_flops_count = conv1_dst_elements * conv1_flops_per_element;
  const int64_t flops_count = conv0_flops_count + conv1_flops_count;
  const double gflops_count = flops_count * 1e-9;

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const int64_t dst_elements_alignedx4 = dst.Width() * dst.Height() * dst.Slices() * 4;
  const int64_t src_elements_alignedx4 = src.Width() * src.Height() * src.Slices() * 4;
  const double dst_gbytes = dst_elements_alignedx4 * float_size / kGByte;
  const double src_gbytes = src_elements_alignedx4 * float_size / kGByte;
  const double weight_gbytes =
      (weights0.shape.DimensionsProduct() + weights1.shape.DimensionsProduct()) * float_size /
      kGByte;

  ComputeTask gpu_task;
  gpu_task.Init(std::move(conv));
  RETURN_IF_ERROR(gpu_task.Compile(&device));
  gpu_task.SetSrcTensor(&src, 0);
  gpu_task.SetDstTensor(&dst, 0);
  RETURN_IF_ERROR(gpu_task.UpdateParams());

  id<MTLCommandQueue> command_queue = [device.device() newCommandQueue];
  for (int i = 0; i < 5; ++i) {
    absl::Duration gpu_task_time = gpu_task.GetTaskTime(command_queue);
    double time_ms = absl::ToDoubleMilliseconds(gpu_task_time);
    const double fps = 1000.0 / time_ms;
    const double gflops_real = fps * gflops_count;
    const double gbyte_read = fps * (src_gbytes + weight_gbytes);
    const double gbyte_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(3) << "  Time - " << time_ms
              << std::setprecision(2) << "(ms), GFlops - " << gflops_real << ", Bandwidth - "
              << gbyte_read + gbyte_write << "(GB/s), Read - " << gbyte_read << "(GB/s), Write - "
              << gbyte_write << "(GB/s)" << std::endl;
  }

  return absl::OkStatus();
}

absl::Status TestDepthwiseConvPerformance(const DepthwiseConvolution2DAttributes& attr,
                                          const BHWC& src_shape, const OperationDef& op_def,
                                          CalculationsPrecision precision) {
  const auto dst_shape = CalculateOutputShape(src_shape, attr);

  Environment env;
  std::unique_ptr<GPUOperation> conv =
      CreateDepthwiseConvTiled(env.GetInfo(), op_def, precision, attr);

  MetalSpatialTensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(src_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  descriptor_with_shape.SetBHWCShape(dst_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &dst));

  const auto w_shape =
      std::visit([](const auto& weights) -> const auto& { return weights.shape; }, attr.weights);

  std::cout << "Src size(HWC) - " << src_shape.h << "x" << src_shape.w << "x" << src_shape.c
            << std::endl;
  std::cout << "Dst size(HWC) - " << dst_shape.h << "x" << dst_shape.w << "x" << dst_shape.c
            << std::endl;
  std::cout << "DepthwiseConvolution attributes : " << std::endl;
  std::cout << "  Kernel size - " << w_shape.w << "x" << w_shape.h << std::endl;
  std::cout << "  Stride - " << attr.strides.w << "x" << attr.strides.h << std::endl;
  std::cout << "  Dilation - " << attr.dilations.w << "x" << attr.dilations.h << std::endl;
  std::cout << "  Padding (prepended) - " << attr.padding.prepended.w << "x"
            << attr.padding.prepended.h << ", (appended) - " << attr.padding.appended.w << "x"
            << attr.padding.appended.h << std::endl;

  RETURN_IF_ERROR(conv->AssembleCode(env.GetInfo()));

  const int64_t flops_count = GetDepthwiseConvolutionFlops(dst_shape, w_shape);

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const int64_t dst_elements_alignedx4 = dst.Width() * dst.Height() * dst.Slices() * 4;
  const int64_t src_elements_alignedx4 = src.Width() * src.Height() * src.Slices() * 4;
  const int float_size = precision == CalculationsPrecision::F32 ? 4 : 2;
  const double dst_gbytes = dst_elements_alignedx4 * float_size / kGByte;
  const double src_gbytes = src_elements_alignedx4 * float_size / kGByte;
  const double weight_gbytes = w_shape.DimensionsProduct() * float_size / kGByte;
  const double bias_gbytes = w_shape.o * float_size / kGByte;

  ComputeTask gpu_task;
  gpu_task.Init(std::move(conv));
  RETURN_IF_ERROR(gpu_task.Compile(&env));
  gpu_task.SetSrcTensor(&src, 0);
  gpu_task.SetDstTensor(&dst, 0);
  RETURN_IF_ERROR(gpu_task.UpdateParams());

  id<MTLCommandQueue> command_queue = [env.device() newCommandQueue];
  for (int i = 0; i < 10; ++i) {
    absl::Duration gpu_task_time = gpu_task.GetTaskTime(command_queue);
    double time_ms = absl::ToDoubleMilliseconds(gpu_task_time);

    const double fps = 1000.0 / time_ms;
    const double gflops_real = fps * flops_count * 1e-9;
    const double gbyte_read = fps * (src_gbytes + weight_gbytes + bias_gbytes);
    const double gbyte_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(3) << i << "  Time - " << time_ms
              << std::setprecision(2) << "(ms), GFlops - " << gflops_real << ", Bandwidth - "
              << gbyte_read + gbyte_write << "(GB/s), Read - " << gbyte_read << "(GB/s), Write - "
              << gbyte_write << "(GB/s)" << std::endl;
  }

  return absl::OkStatus();
}

}  // namespace

absl::Status ConvolutionPerfTest(CalculationsPrecision precision, const BHWC& src_shape,
                                 int dst_channels, const HW& kernel_size) {
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(kernel_size.h / 2, kernel_size.w / 2);
  attr.padding.appended = HW(kernel_size.h / 2, kernel_size.w / 2);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto& attr_weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(dst_channels, kernel_size.h, kernel_size.w, src_shape.c);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias.shape = Linear(dst_channels);
  attr.bias.data.resize(attr.bias.shape.DimensionsProduct());

  OperationDef op_def;
  auto data_type = DeduceDataTypeFromPrecision(precision);
  Layout layout = src_shape.b == 1 ? Layout::HWC : Layout::BHWC;
  op_def.src_tensors.push_back({data_type, TensorStorageType::BUFFER, layout});
  op_def.dst_tensors.push_back({data_type, TensorStorageType::BUFFER, layout});

  RETURN_IF_ERROR(TestConvolutionPerformance(attr, src_shape, op_def, precision));

  return absl::OkStatus();
}

absl::Status ConvolutionInt8PerfTest(const BHWC& src_shape, int dst_channels) {
  const bool dequantize = true;
  const bool batched_weights = false;
  const int weights_h = batched_weights ? src_shape.h : 1;
  const DataType float_type = DataType::FLOAT16;

  Environment env;

  BHWC dst_shape = src_shape;
  dst_shape.c = dst_channels;

  Layout layout = src_shape.b == 1 ? Layout::HWC : Layout::BHWC;
  TensorDescriptor src_tensor_desc{DataType::INT8, TensorStorageType::BUFFER, layout,
                                   TensorDescriptor::PhysicalLayout1D::kDHWBCC4};
  TensorDescriptor dst_tensor_desc{DataType::INT32, TensorStorageType::BUFFER, layout};

  ml_drift::Tensor<OHWI, DataType::INT8> weights;
  weights.shape = OHWI(dst_channels, weights_h, 1, src_shape.c);
  weights.data.resize(weights.shape.DimensionsProduct() + XNN_EXTRA_BYTES / sizeof(int8_t));

  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_scales;
  weights_scales.shape = OHWI(dst_channels, weights_h, 1, 1);
  weights_scales.data.resize(weights_scales.shape.DimensionsProduct());
  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_zero_point;
  weights_zero_point.shape = OHWI(dst_channels, weights_h, 1, 1);
  weights_zero_point.data.resize(weights_scales.shape.DimensionsProduct(), 0.0f);

  PackedType quantized_type = PackedType::kInt8C4;
  std::unique_ptr<GPUOperation> conv = std::make_unique<ConvAppleMPP>(
      CreateConvAppleMPPInt8(src_tensor_desc, dst_tensor_desc, weights));

  TensorDescriptor src_params_td;
  TensorDescriptor weights_scale_td;
  TensorDescriptor weights_zero_point_td;
  TensorDescriptor weights_sum_i_td;
  if (dequantize) {
    ml_drift::TensorFloat32 src_params;
    src_params.shape = BHWC(src_shape.b, src_shape.h, src_shape.w, 3);
    src_params.data.resize(src_params.shape.DimensionsProduct());
    src_params_td = TensorDescriptor{DataType::FLOAT32, TensorStorageType::BUFFER, layout};
    src_params_td.UploadData(src_params);

    weights_scale_td = ScaleOrZeroPointToTensorDesc(env.GetInfo(), weights_scales, float_type);
    weights_zero_point_td =
        ScaleOrZeroPointToTensorDesc(env.GetInfo(), weights_zero_point, float_type);

    auto weights_sum_i = GetWeightsAccumulatedInputChannels(weights);
    weights_sum_i_td = CreateConstantLinearTensorDescriptor(env.GetInfo(), weights_sum_i);

    TensorDescriptor dequant_dst = {float_type, dst_tensor_desc.GetStorageType(),
                                    dst_tensor_desc.GetLayout()};
    auto dequant_op = CreateDequantization(weights.shape, env.GetInfo(), dst_tensor_desc,
                                           dequant_dst, src_params_td, weights_sum_i_td,
                                           weights_scale_td, &weights_zero_point_td);
    dst_tensor_desc = dequant_dst;
    RETURN_IF_ERROR(conv->AddOperation(env.GetInfo(), &dequant_op));
  }

  MetalSpatialTensor src, dst;
  TensorDescriptor descriptor_with_shape = src_tensor_desc;
  descriptor_with_shape.SetBHWCShape(GetShapeForPackedType(src_shape, quantized_type));
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &src));
  descriptor_with_shape = dst_tensor_desc;
  descriptor_with_shape.SetBHWCShape(dst_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &dst));

  MetalSpatialTensor src_params_tensor;
  MetalSpatialTensor weights_scale_tensor;
  MetalSpatialTensor weights_zero_point_tensor;
  MetalSpatialTensor weights_sum_i_tensor;
  if (dequantize) {
    RETURN_IF_ERROR(CreateTensor(env.device(), src_params_td, &src_params_tensor));
    RETURN_IF_ERROR(CreateTensor(env.device(), weights_scale_td, &weights_scale_tensor));
    RETURN_IF_ERROR(CreateTensor(env.device(), weights_zero_point_td, &weights_zero_point_tensor));
    RETURN_IF_ERROR(CreateTensor(env.device(), weights_sum_i_td, &weights_sum_i_tensor));
  }

  const auto w_shape = weights.shape;

  std::cout << "Src size(HWC) - " << src_shape.h << "x" << src_shape.w << "x" << src_shape.c
            << std::endl;
  std::cout << "Dst size(HWC) - " << dst_shape.h << "x" << dst_shape.w << "x" << dst_shape.c
            << std::endl;
  RETURN_IF_ERROR(conv->AssembleCode(env.GetInfo()));

  const int64_t flops_per_element = w_shape.i * 2;
  const int64_t dst_elements = dst.Width() * dst.Height() * dst.Channels();
  const int64_t flops_count = dst_elements * flops_per_element;
  const double gflops_count = flops_count * 1e-9;

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const double src_gbytes = src.GetMemorySizeInBytes() / kGByte;
  const double dst_gbytes = dst.GetMemorySizeInBytes() / kGByte;
  const double weight_gbytes = weights.shape.DimensionsProduct() * sizeof(int8_t) / kGByte;

  ComputeTask gpu_task;
  gpu_task.Init(std::move(conv));
  RETURN_IF_ERROR(gpu_task.Compile(&env));
  gpu_task.SetSrcTensor(&src, 0);
  if (dequantize) {
    gpu_task.SetSrcTensor(&src_params_tensor, 1);
    gpu_task.SetSrcTensor(&weights_sum_i_tensor, 2);
    gpu_task.SetSrcTensor(&weights_scale_tensor, 3);
    gpu_task.SetSrcTensor(&weights_zero_point_tensor, 4);
  }
  gpu_task.SetDstTensor(&dst, 0);
  RETURN_IF_ERROR(gpu_task.UpdateParams());

  id<MTLCommandQueue> command_queue = [env.device() newCommandQueue];
  for (int i = 0; i < 10; ++i) {
    absl::Duration gpu_task_time = gpu_task.GetTaskTime(command_queue);
    double time_ms = absl::ToDoubleMilliseconds(gpu_task_time);
    const double fps = 1000.0 / time_ms;
    const double gflops_real = fps * gflops_count;
    const double gbyte_read = fps * (src_gbytes + weight_gbytes);
    const double gbyte_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(3) << "  Time - " << time_ms
              << std::setprecision(2) << "(ms), Giops - " << gflops_real << ", Bandwidth - "
              << gbyte_read + gbyte_write << "(GB/s), Read - " << gbyte_read << "(GB/s), Write - "
              << gbyte_write << "(GB/s)" << std::endl;
  }

  return absl::OkStatus();
}

absl::Status ConvolutionSf16Wi4BatchedPerfTest(const BHWC& src_shape, int dst_channels,
                                               OHWI scale_zp_shape) {
  Environment env;
  const DataType float_type = DataType::FLOAT16;
  const bool use_zero_point = true;

  BHWC dst_shape = src_shape;
  dst_shape.c = dst_channels;

  const int src_channels = src_shape.c;

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i4;
  weights_i4.shape = OHWI(dst_channels, scale_zp_shape.h, 1, src_channels);
  weights_i4.data.resize(weights_i4.shape.DimensionsProduct());
  for (int i = 0; i < weights_i4.data.size(); ++i) {
    weights_i4.data[i] = (i % 15) - 7;
  }

  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_scales;
  weights_scales.shape = scale_zp_shape;
  weights_scales.data.resize(weights_scales.shape.DimensionsProduct());
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] = 1.0f / 8.0f;
  }

  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_zero_point;
  weights_zero_point.shape = weights_scales.shape;
  weights_zero_point.data.resize(weights_scales.shape.DimensionsProduct(), 0.0f);

  Layout layout = src_shape.b == 1 ? Layout::HWC : Layout::BHWC;
  TensorDescriptor src_tensor_desc{float_type, TensorStorageType::BUFFER, layout,
                                   TensorDescriptor::PhysicalLayout1D::kDHWBCC4};
  TensorDescriptor dst_tensor_desc{float_type, TensorStorageType::BUFFER, layout};

  WeightsDescription weights_desc =
      GetFullyConnectedInt4WeightsDesc(env.GetInfo(), weights_i4.shape);

  auto scale_desc = ScaleOrZeroPointToTensorDesc(env.GetInfo(), weights_scales, float_type);
  auto zp_desc = ScaleOrZeroPointToTensorDesc(env.GetInfo(), weights_zero_point, float_type);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_i4.shape;
  external_weights.scale_zp_shape = weights_scales.shape;
  external_weights.scale = &scale_desc;
  if (use_zero_point) {
    external_weights.zero_point = &zp_desc;
  }

  std::unique_ptr<GPUOperation> conv;
  const auto& gpu_info = env.GetInfo();
  if (SupportsConvAppleMPP(gpu_info, external_weights)) {
    auto conv_apple_mpp =
        CreateConvAppleMPPExternalWeights(src_tensor_desc, dst_tensor_desc, external_weights,
                                          /*bias=*/nullptr,
                                          /*src_exp=*/nullptr,
                                          /*different_weights_for_height=*/true);
    conv = std::make_unique<ConvAppleMPP>(std::move(conv_apple_mpp));
  } else if (SupportsConvWaveMatrix(gpu_info, CalculationsPrecision::F16, external_weights)) {
    OperationDef conv_def;
    conv_def.src_tensors.push_back(src_tensor_desc);
    conv_def.dst_tensors.push_back(dst_tensor_desc);
    auto conv_wave_matrix = CreateConvWaveMatrixExternalWeights(
        conv_def, CalculationsPrecision::F16, dst_shape, external_weights, gpu_info,
        /*bias=*/nullptr, /*src_exp=*/nullptr,
        /*different_weights_for_height=*/true);
    conv = std::make_unique<ConvWaveMatrix>(std::move(conv_wave_matrix));
  } else {
    return absl::UnimplementedError("no supported conv");
  }

  TensorDescriptor weights_i4_td = GetTensorDescriptorForWeightsLayout(weights_i4, weights_desc);

  MetalSpatialTensor src, dst;
  TensorDescriptor descriptor_with_shape = src_tensor_desc;
  descriptor_with_shape.SetBHWCShape(src_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &src));

  descriptor_with_shape = dst_tensor_desc;
  descriptor_with_shape.SetBHWCShape(dst_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &dst));

  MetalSpatialTensor weights_i4_tensor;
  RETURN_IF_ERROR(CreateTensor(env.device(), weights_i4_td, &weights_i4_tensor));

  MetalSpatialTensor weights_scale_tensor;
  RETURN_IF_ERROR(CreateTensor(env.device(), scale_desc, &weights_scale_tensor));

  MetalSpatialTensor weights_zero_point_tensor;
  RETURN_IF_ERROR(CreateTensor(env.device(), zp_desc, &weights_zero_point_tensor));

  RETURN_IF_ERROR(conv->AssembleCode(env.GetInfo()));

  const int64_t flops_per_element = weights_i4.shape.i * 2;
  const int64_t dst_elements = dst.Width() * dst.Height() * dst.Channels();
  const int64_t flops_count = dst_elements * flops_per_element;
  const double gflops_count = flops_count * 1e-9;

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const double src_gbytes = src.GetMemorySizeInBytes() / kGByte;
  const double dst_gbytes = dst.GetMemorySizeInBytes() / kGByte;
  const double weight_gbytes = (weights_i4.shape.DimensionsProduct() * 0.5) / kGByte;

  ComputeTask gpu_task;
  gpu_task.Init(std::move(conv));
  RETURN_IF_ERROR(gpu_task.Compile(&env));
  gpu_task.SetSrcTensor(&src, 0);
  gpu_task.SetSrcTensor(&weights_i4_tensor, 1);
  gpu_task.SetSrcTensor(&weights_scale_tensor, 2);
  if (use_zero_point) {
    gpu_task.SetSrcTensor(&weights_zero_point_tensor, 3);
  }
  gpu_task.SetDstTensor(&dst, 0);
  RETURN_IF_ERROR(gpu_task.UpdateParams());

  id<MTLCommandQueue> command_queue = [env.device() newCommandQueue];
  for (int i = 0; i < 10; ++i) {
    absl::Duration gpu_task_time = gpu_task.GetTaskTime(command_queue);
    double time_ms = absl::ToDoubleMilliseconds(gpu_task_time);
    const double fps = 1000.0 / time_ms;
    const double gflops_real = fps * gflops_count;
    const double gbyte_read = fps * (src_gbytes + weight_gbytes);
    const double gbyte_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(3) << "  Time - " << time_ms
              << std::setprecision(2) << "(ms), Giops - " << gflops_real << ", Bandwidth - "
              << gbyte_read + gbyte_write << "(GB/s), Read - " << gbyte_read << "(GB/s), Write - "
              << gbyte_write << "(GB/s)" << std::endl;
  }

  return absl::OkStatus();
}

absl::Status ConvolutionSi8Wi4PerfTest(const BHWC& src_shape, int dst_channels) {
  Environment env;
  const bool dequantize = false;
  const bool batched_weights = true;
  const int weights_h = batched_weights ? src_shape.h : 1;
  const DataType float_type = DataType::FLOAT16;
  const DataType src_type = DataType::INT8;
  const DataType dst_type = DataType::INT32;

  BHWC dst_shape = src_shape;
  dst_shape.c = dst_channels;

  const int src_channels = src_shape.c;

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i4;
  weights_i4.shape = OHWI(dst_channels, weights_h, 1, src_channels);
  weights_i4.data.resize(weights_i4.shape.DimensionsProduct());
  for (int i = 0; i < weights_i4.data.size(); ++i) {
    weights_i4.data[i] = (i % 15) - 7;
  }

  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_scales;
  weights_scales.shape = OHWI(dst_channels, weights_h, 1, 1);
  weights_scales.data.resize(weights_scales.shape.DimensionsProduct());
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] = 1.0f / 8.0f;
  }

  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_zero_point;
  weights_zero_point.shape = weights_scales.shape;
  weights_zero_point.data.resize(weights_scales.shape.DimensionsProduct(), 0.0f);

  Layout layout = src_shape.b == 1 ? Layout::HWC : Layout::BHWC;
  TensorDescriptor src_tensor_desc{src_type, TensorStorageType::BUFFER, layout,
                                   TensorDescriptor::PhysicalLayout1D::kDHWBCC4};
  TensorDescriptor dst_tensor_desc{dst_type, TensorStorageType::BUFFER, layout};

  WeightsDescription weights_desc =
      GetFullyConnectedInt4WeightsDesc(env.GetInfo(), weights_i4.shape);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_i4.shape;
  external_weights.scale = nullptr;
  external_weights.zero_point = nullptr;

  if (!SupportsConvAppleMPP(env.GetInfo(), external_weights)) {
    return absl::UnimplementedError("no supported conv");
  }

  auto conv_apple_mpp = CreateConvAppleMPPInt8(src_tensor_desc, dst_tensor_desc, external_weights);

  std::unique_ptr<GPUOperation> conv = std::make_unique<ConvAppleMPP>(std::move(conv_apple_mpp));

  TensorDescriptor src_params_td;
  TensorDescriptor weights_scale_td;
  TensorDescriptor weights_zero_point_td;
  TensorDescriptor weights_sum_i_td;
  if (dequantize) {
    ml_drift::TensorFloat32 src_params;
    src_params.shape = BHWC(src_shape.b, src_shape.h, src_shape.w, 3);
    src_params.data.resize(src_params.shape.DimensionsProduct());
    src_params_td = TensorDescriptor{DataType::FLOAT32, TensorStorageType::BUFFER, layout};
    src_params_td.UploadData(src_params);

    weights_scale_td = ScaleOrZeroPointToTensorDesc(env.GetInfo(), weights_scales, float_type);
    weights_zero_point_td =
        ScaleOrZeroPointToTensorDesc(env.GetInfo(), weights_zero_point, float_type);

    auto weights_sum_i = GetWeightsAccumulatedInputChannels(weights_i4);
    weights_sum_i_td = CreateConstantLinearTensorDescriptor(env.GetInfo(), weights_sum_i);

    TensorDescriptor dequant_dst = {float_type, dst_tensor_desc.GetStorageType(),
                                    dst_tensor_desc.GetLayout()};
    auto dequant_op = CreateDequantization(weights_i4.shape, env.GetInfo(), dst_tensor_desc,
                                           dequant_dst, src_params_td, weights_sum_i_td,
                                           weights_scale_td, &weights_zero_point_td);
    dst_tensor_desc = dequant_dst;
    RETURN_IF_ERROR(conv->AddOperation(env.GetInfo(), &dequant_op));
  }

  TensorDescriptor weights_i4_td = GetTensorDescriptorForWeightsLayout(weights_i4, weights_desc);

  MetalSpatialTensor src, dst;
  TensorDescriptor descriptor_with_shape = src_tensor_desc;
  descriptor_with_shape.SetBHWCShape(src_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &src));

  descriptor_with_shape = dst_tensor_desc;
  descriptor_with_shape.SetBHWCShape(dst_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &dst));

  MetalSpatialTensor weights_i4_tensor;
  RETURN_IF_ERROR(CreateTensor(env.device(), weights_i4_td, &weights_i4_tensor));

  MetalSpatialTensor src_params_tensor;
  MetalSpatialTensor weights_scale_tensor;
  MetalSpatialTensor weights_zero_point_tensor;
  MetalSpatialTensor weights_sum_i_tensor;
  if (dequantize) {
    RETURN_IF_ERROR(CreateTensor(env.device(), src_params_td, &src_params_tensor));
    RETURN_IF_ERROR(CreateTensor(env.device(), weights_scale_td, &weights_scale_tensor));
    RETURN_IF_ERROR(CreateTensor(env.device(), weights_zero_point_td, &weights_zero_point_tensor));
    RETURN_IF_ERROR(CreateTensor(env.device(), weights_sum_i_td, &weights_sum_i_tensor));
  }
  RETURN_IF_ERROR(conv->AssembleCode(env.GetInfo()));

  const int64_t flops_per_element = weights_i4.shape.i * 2;
  const int64_t dst_elements = dst.Width() * dst.Height() * dst.Channels();
  const int64_t flops_count = dst_elements * flops_per_element;
  const double gflops_count = flops_count * 1e-9;

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const double src_gbytes = src.GetMemorySizeInBytes() / kGByte;
  const double dst_gbytes = dst.GetMemorySizeInBytes() / kGByte;
  const double weight_gbytes = (weights_i4.shape.DimensionsProduct() * 0.5) / kGByte;

  ComputeTask gpu_task;
  gpu_task.Init(std::move(conv));
  RETURN_IF_ERROR(gpu_task.Compile(&env));
  gpu_task.SetSrcTensor(&src, 0);
  gpu_task.SetSrcTensor(&weights_i4_tensor, 1);
  if (dequantize) {
    gpu_task.SetSrcTensor(&src_params_tensor, 2);
    gpu_task.SetSrcTensor(&weights_sum_i_tensor, 3);
    gpu_task.SetSrcTensor(&weights_scale_tensor, 4);
    gpu_task.SetSrcTensor(&weights_zero_point_tensor, 5);
  }
  gpu_task.SetDstTensor(&dst, 0);
  RETURN_IF_ERROR(gpu_task.UpdateParams());

  id<MTLCommandQueue> command_queue = [env.device() newCommandQueue];
  for (int i = 0; i < 10; ++i) {
    absl::Duration gpu_task_time = gpu_task.GetTaskTime(command_queue);
    double time_ms = absl::ToDoubleMilliseconds(gpu_task_time);
    const double fps = 1000.0 / time_ms;
    const double gflops_real = fps * gflops_count;
    const double gbyte_read = fps * (src_gbytes + weight_gbytes);
    const double gbyte_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(3) << "  Time - " << time_ms
              << std::setprecision(2) << "(ms), Giops - " << gflops_real << ", Bandwidth - "
              << gbyte_read + gbyte_write << "(GB/s), Read - " << gbyte_read << "(GB/s), Write - "
              << gbyte_write << "(GB/s)" << std::endl;
  }

  return absl::OkStatus();
}

absl::Status ConvMoEPerfTest(int seq_size, int src_channels, int dst_channels, int num_experts,
                             int num_active_experts, DataType weights_type) {
  Environment env;
  const auto& gpu_info = env.GetInfo();

  const BHWC src_shape = BHWC(1, 1, seq_size * num_active_experts, src_channels);
  OperationDef op_def;
  auto data_type = DataType::FLOAT16;
  Layout layout = Layout::HWC;
  auto storage_type = TensorStorageType::BUFFER;
  TensorDescriptor src_tensor_desc{data_type, storage_type, layout};

  auto dst_shape = src_shape;
  dst_shape.c = dst_channels;
  TensorDescriptor dst_tensor_desc{data_type, storage_type, layout};

  op_def.src_tensors.push_back(src_tensor_desc);
  op_def.dst_tensors.push_back(dst_tensor_desc);

  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_f32;
  weights_f32.shape = OHWI(dst_channels, num_experts, 1, src_shape.c);
  weights_f32.data.resize(weights_f32.shape.DimensionsProduct() + XNN_EXTRA_BYTES / sizeof(float));

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(dst_channels, num_experts, 1, src_shape.c);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct() + XNN_EXTRA_BYTES / sizeof(uint8_t));

  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_scale;
  weights_scale.shape = OHWI(dst_channels, num_experts, 1, 1);
  weights_scale.data.resize(weights_scale.shape.DimensionsProduct(), 1.0f);
  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_zp;
  weights_zp.shape = OHWI(dst_channels, num_experts, 1, 1);
  weights_zp.data.resize(weights_zp.shape.DimensionsProduct(), 0.0f);

  WeightsDescription weights_desc;
  std::vector<TensorDescriptor> weights_gpu;
  if (weights_type == DataType::FLOAT16 || weights_type == DataType::FLOAT32) {
    weights_desc.type = weights_type;
    weights_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
    weights_desc.output_group_size = DivideRoundUp(weights_f32.shape.o, 4);
    weights_gpu = GetTensorDescriptorsForWeightsLayout(weights_f32, weights_desc);
  } else if (weights_type == DataType::INT8) {
    weights_desc = GetFullyConnectedInt8WeightsDesc(gpu_info, weights_i8.shape);
    weights_gpu.push_back(GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc));
  } else if (weights_type == DataType::INT4) {
    weights_desc = GetFullyConnectedInt4WeightsDesc(gpu_info, weights_i8.shape);
    weights_gpu.push_back(GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc));
  } else if (weights_type == DataType::INT2) {
    weights_desc = GetFullyConnectedInt2WeightsDesc(gpu_info, weights_i8.shape);
    weights_gpu.push_back(GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc));
  }

  std::vector<MetalSpatialTensor> weights_tensors(weights_gpu.size());
  for (int i = 0; i < weights_gpu.size(); ++i) {
    RETURN_IF_ERROR(CreateTensor(env.device(), weights_gpu[i], &weights_tensors[i]));
  }

  TensorDescriptor scale_desc = ScaleOrZeroPointToTensorDesc(gpu_info, weights_scale, data_type);
  TensorDescriptor zp_desc = ScaleOrZeroPointToTensorDesc(gpu_info, weights_zp, data_type);

  const bool is_quantized = SizeInBitsOf(weights_type) < 16;

  MetalSpatialTensor scale_tensor;
  MetalSpatialTensor zp_tensor;
  if (is_quantized) {
    RETURN_IF_ERROR(CreateTensor(env.device(), scale_desc, &scale_tensor));
    RETURN_IF_ERROR(CreateTensor(env.device(), zp_desc, &zp_tensor));
  }

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_f32.shape;
  if (is_quantized) {
    external_weights.scale_zp_shape = weights_scale.shape;
    external_weights.scale = &scale_desc;
    external_weights.zero_point = &zp_desc;
  }

  std::unique_ptr<GPUOperation> conv;
  ConvRuntimeCheckDesc::PackedGroups packed_groups;
  packed_groups.params_offset = 0;
  packed_groups.num_groups = num_experts;
  packed_groups.max_group_size = seq_size;
  ConvRuntimeCheckDesc runtime_check;
  runtime_check.packed_groups = packed_groups;
  if (SupportsConvAppleMPP(gpu_info, external_weights)) {
    auto conv_apple_mpp = CreateConvAppleMPPExternalWeights(
        op_def.src_tensors[0], op_def.dst_tensors[0], external_weights,
        /*bias=*/nullptr,
        /*src_exp=*/nullptr, /*different_weights_for_height=*/true, runtime_check);
    conv = std::make_unique<ConvAppleMPP>(std::move(conv_apple_mpp));
  } else if (SupportsConvWaveMatrix(gpu_info, CalculationsPrecision::F16, external_weights)) {
    auto conv_wave_matrix = CreateConvWaveMatrixExternalWeights(
        op_def, CalculationsPrecision::F16, dst_shape, external_weights, gpu_info,
        /*bias=*/nullptr,
        /*src_exp=*/nullptr, /*different_weights_for_height=*/true, runtime_check);
    conv = std::make_unique<ConvWaveMatrix>(std::move(conv_wave_matrix));
  } else {
    ASSIGN_OR_RETURN(auto conv_fc, CreateFullyConnectedExternalWeights(
                                       gpu_info, CalculationsPrecision::F16, op_def.src_tensors[0],
                                       op_def.dst_tensors[0], external_weights, /*bias=*/nullptr,
                                       &dst_shape, /*src_exp=*/nullptr, runtime_check));
    conv = std::make_unique<FullyConnected>(std::move(conv_fc));
  }

  MetalSpatialTensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(src_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  descriptor_with_shape.SetBHWCShape(dst_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &dst));

  TensorInt32 active_expert_ids =
      GenerateGroupIds(BHWC(1, 1, seq_size, num_active_experts), num_experts);

  auto [groups_map, groups_sizes] = GroupsMapReference(active_expert_ids, num_experts);
  auto [packed_groups_map, groups_offsets] = PackedGroupsMapReference(groups_map, groups_sizes);

  std::vector<int32_t> runtime_params_cpu(num_experts * 2, 0);
  for (int i = 0; i < num_experts; ++i) {
    runtime_params_cpu[i] = groups_sizes.data[i];
    runtime_params_cpu[num_experts + i] = groups_offsets.data[i];
  }
  MetalSpatialTensor runtime_params;
  TensorDescriptor runtime_params_td(DataType::INT32, TensorStorageType::BUFFER, Layout::LINEAR);
  runtime_params_td.SetBHWCShape(BHWC(1, 1, 1, num_experts * 2));
  runtime_params_td.UploadData(runtime_params_cpu.data());
  RETURN_IF_ERROR(CreateTensor(env.device(), runtime_params_td, &runtime_params));

  std::cout << "Src size(BHWC) - " << src_shape.b << "x" << src_shape.h << "x" << src_shape.w << "x"
            << src_shape.c << std::endl;
  std::cout << "Dst size(BHWC) - " << dst_shape.b << "x" << dst_shape.h << "x" << dst_shape.w << "x"
            << dst_shape.c << std::endl;

  RETURN_IF_ERROR(conv->AssembleCode(gpu_info));

  const int64_t flops_count = 2.0 * dst_shape.DimensionsProduct() * weights_f32.shape.i;
  const double gflops_count = flops_count * 1e-9;

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const double src_gbytes = src.GetMemorySizeInBytes() / kGByte;
  const double dst_gbytes = dst.GetMemorySizeInBytes() / kGByte;
  const double weight_element_size = SizeInBitsOf(weights_type) / 8.0;
  const double weight_gbytes = weights_f32.shape.DimensionsProduct() * weight_element_size / kGByte;
  const double scale_gbytes = scale_tensor.GetMemorySizeInBytes() / kGByte;

  ComputeTask gpu_task;
  gpu_task.Init(std::move(conv));
  RETURN_IF_ERROR(gpu_task.Compile(&env));
  int index = 0;
  gpu_task.SetSrcTensor(&src, index++);
  for (int i = 0; i < weights_tensors.size(); ++i) {
    gpu_task.SetSrcTensor(&weights_tensors[i], index++);
  }
  if (is_quantized) {
    gpu_task.SetSrcTensor(&scale_tensor, index++);
    gpu_task.SetSrcTensor(&zp_tensor, index++);
  }
  gpu_task.SetSrcTensor(&runtime_params, index++);
  gpu_task.SetDstTensor(&dst, 0);
  RETURN_IF_ERROR(gpu_task.UpdateParams());

  id<MTLCommandQueue> command_queue = [env.device() newCommandQueue];
  for (int i = 0; i < 10; ++i) {
    absl::Duration gpu_task_time = gpu_task.GetTaskTime(command_queue);
    double time_ms = absl::ToDoubleMilliseconds(gpu_task_time);
    const double fps = 1000.0 / time_ms;
    const double gflops_real = fps * gflops_count;
    double gbytes_read = src_gbytes + weight_gbytes;
    if (is_quantized) {
      gbytes_read += scale_gbytes * 2.0;
    }
    const double gbyte_read = fps * gbytes_read;
    const double gbyte_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(3) << "  Time - " << time_ms
              << std::setprecision(2) << "(ms), GFlops - " << gflops_real << ", Bandwidth - "
              << gbyte_read + gbyte_write << "(GB/s), Read - " << gbyte_read << "(GB/s), Write - "
              << gbyte_write << "(GB/s)" << std::endl;
  }

  return absl::OkStatus();
}

absl::Status ConvSoftmaxConvPerfTest() {
  const auto precision = CalculationsPrecision::F16;
  OperationDef op_def;
  auto data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back({data_type, TensorStorageType::BUFFER, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, TensorStorageType::BUFFER, Layout::HWC});

  RETURN_IF_ERROR(
      TestConvSoftmaxConvPerformance(BHWC(1, 16, 4096, 40), 40, 4096, op_def, precision));

  return absl::OkStatus();
}

absl::Status FullyConnectedOptimalWGSize(CalculationsPrecision precision, DataType weights_type,
                                         const BHWC& src_shape, int dst_channels,
                                         OHWI scale_zp_shape) {
  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = OHWI(dst_channels, scale_zp_shape.h, 1, src_shape.c);
  weights.data.resize(weights.shape.DimensionsProduct() + XNN_EXTRA_BYTES / sizeof(float));

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(dst_channels, scale_zp_shape.h, 1, src_shape.c);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct() + XNN_EXTRA_BYTES / sizeof(uint8_t));

  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_scale;
  weights_scale.shape = scale_zp_shape;
  weights_scale.data.resize(weights_scale.shape.DimensionsProduct(), 1.0f);
  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_zp;
  weights_zp.shape = scale_zp_shape;
  weights_zp.data.resize(weights_zp.shape.DimensionsProduct(), 0.0f);

  Environment env;
  const GpuInfo& gpu_info = env.GetInfo();

  OperationDef op_def;
  auto data_type = DeduceDataTypeFromPrecision(precision);
  Layout layout = src_shape.b == 1 ? Layout::HWC : Layout::BHWC;
  op_def.src_tensors.push_back({data_type, TensorStorageType::BUFFER, layout});
  op_def.dst_tensors.push_back({data_type, TensorStorageType::BUFFER, layout});

  auto dst_shape = src_shape;
  dst_shape.c = dst_channels;

  id<MTLCommandQueue> command_queue = [env.device() newCommandQueue];

  MetalSpatialTensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(src_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  descriptor_with_shape.SetBHWCShape(dst_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &dst));

  const auto w_shape = weights.shape;

  std::cout << "Src size(HWC) - " << src_shape.h << "x" << src_shape.w << "x" << src_shape.c
            << std::endl;
  std::cout << "Dst size(HWC) - " << dst_shape.h << "x" << dst_shape.w << "x" << dst_shape.c
            << std::endl;

  double element_size = precision == CalculationsPrecision::F32 ? 4.0 : 2.0;
  int64_t flops_per_element = w_shape.i * w_shape.h * w_shape.w * 2;
  if (scale_zp_shape.h > 1) {
    flops_per_element /= scale_zp_shape.h;
  }
  const int64_t dst_elements = dst.Width() * dst.Height() * dst.Channels();
  const int64_t flops_count = dst_elements * flops_per_element;
  const double gflops_count = flops_count * 1e-9;

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const int64_t dst_elements_alignedx4 = dst.Width() * dst.Height() * dst.Slices() * 4;
  const int64_t src_elements_alignedx4 = src.Width() * src.Height() * src.Slices() * 4;
  const double dst_gbytes = dst_elements_alignedx4 * element_size / kGByte;
  const double src_gbytes = src_elements_alignedx4 * element_size / kGByte;
  const double weight_element_size = SizeInBitsOf(weights_type) / 8.0;
  const double weight_gbytes = weights.shape.DimensionsProduct() * weight_element_size / kGByte;
  const double bias_gbytes = weights.shape.o * element_size / kGByte;
  const double scale_gbytes = scale_zp_shape.DimensionsProduct() * element_size / kGByte;

  WeightsDescription weights_desc;
  std::vector<TensorDescriptor> weights_gpu;
  if (weights_type == DataType::FLOAT16 || weights_type == DataType::FLOAT32) {
    weights_desc.type = DeduceDataTypeFromPrecision(precision);
    weights_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
    weights_desc.output_group_size = DivideRoundUp(weights.shape.o, 4);
    weights_gpu = GetTensorDescriptorsForWeightsLayout(weights, weights_desc);
  } else if (weights_type == DataType::INT8) {
    weights_desc = GetFullyConnectedInt8WeightsDesc(gpu_info, weights_i8.shape);
    weights_gpu.push_back(GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc));
  } else if (weights_type == DataType::INT4) {
    weights_desc = GetFullyConnectedInt4WeightsDesc(gpu_info, weights_i8.shape);
    weights_gpu.push_back(GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc));
  } else if (weights_type == DataType::INT2) {
    weights_desc = GetFullyConnectedInt2WeightsDesc(gpu_info, weights_i8.shape);
    weights_gpu.push_back(GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc));
  }

  std::vector<MetalSpatialTensor> weights_tensors(weights_gpu.size());
  for (int i = 0; i < weights_gpu.size(); ++i) {
    RETURN_IF_ERROR(CreateTensor(env.device(), weights_gpu[i], &weights_tensors[i]));
  }

  TensorDescriptor scale_desc = ScaleOrZeroPointToTensorDesc(gpu_info, weights_scale, data_type);
  TensorDescriptor zp_desc = ScaleOrZeroPointToTensorDesc(gpu_info, weights_zp, data_type);

  const bool is_quantized = SizeInBitsOf(weights_type) < 16;

  MetalSpatialTensor scale_tensor;
  MetalSpatialTensor zp_tensor;
  if (is_quantized) {
    RETURN_IF_ERROR(CreateTensor(env.device(), scale_desc, &scale_tensor));
    RETURN_IF_ERROR(CreateTensor(env.device(), zp_desc, &zp_tensor));
  }

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights.shape;
  if (is_quantized) {
    external_weights.scale_zp_shape = scale_zp_shape;
    external_weights.scale = &scale_desc;
    external_weights.zero_point = &zp_desc;
  }

  std::vector<int3> wg_sizes;
  for (int y = 1; y <= 256; y *= 2) {
    for (int x = 256; x >= 1; x /= 2) {
      if (x * y >= 16 && x * y <= 256) {
        wg_sizes.push_back({x, y, 1});
      }
    }
  }
  std::vector<double> time_ms(wg_sizes.size());

  for (int i = 0; i < wg_sizes.size(); ++i) {
    ASSIGN_OR_RETURN(auto operation, CreateFullyConnectedExternalWeights(
                                         gpu_info, precision, op_def.src_tensors[0],
                                         op_def.dst_tensors[0], external_weights,
                                         /*bias=*/nullptr, &dst_shape, /*src_exp=*/nullptr,
                                         /*runtime_check=*/{}, &wg_sizes[i]));
    std::unique_ptr<GPUOperation> conv = std::make_unique<FullyConnected>(std::move(operation));
    RETURN_IF_ERROR(conv->AssembleCode(env.GetInfo()));

    ComputeTask gpu_kernel;
    gpu_kernel.Init(std::move(conv));
    RETURN_IF_ERROR(gpu_kernel.Compile(&env));
    int index = 0;
    gpu_kernel.SetSrcTensor(&src, index++);
    for (int i = 0; i < weights_tensors.size(); ++i) {
      gpu_kernel.SetSrcTensor(&weights_tensors[i], index++);
    }
    if (is_quantized) {
      gpu_kernel.SetSrcTensor(&scale_tensor, index++);
      gpu_kernel.SetSrcTensor(&zp_tensor, index++);
    }
    gpu_kernel.SetDstTensor(&dst, 0);
    RETURN_IF_ERROR(gpu_kernel.UpdateParams());

    time_ms[i] = absl::ToDoubleMilliseconds(gpu_kernel.GetTaskTime(command_queue));
    std::cout << "WG size: " << wg_sizes[i].x << "x" << wg_sizes[i].y << " - time: " << time_ms[i]
              << std::endl;
  }

  double min_time_ms = 10000.0;
  int3 optimal_wg_size;
  for (int i = 0; i < wg_sizes.size(); ++i) {
    if (time_ms[i] < min_time_ms) {
      min_time_ms = time_ms[i];
      optimal_wg_size = wg_sizes[i];
    }
  }

  const double fps = 1000.0 / min_time_ms;
  const double gflops_real = fps * gflops_count;
  double gbytes_read = src_gbytes + weight_gbytes + bias_gbytes;
  if (is_quantized) {
    gbytes_read += scale_gbytes * 2.0;
  }
  double gbs_read = fps * gbytes_read;
  const double gbs_write = fps * dst_gbytes;
  std::cout << std::endl
            << "Optimal WG size: " << optimal_wg_size.x << "x" << optimal_wg_size.y << std::fixed
            << std::setprecision(4) << " Time - " << min_time_ms << "(ms), GFlops - " << gflops_real
            << ", Bandwidth - " << gbs_read + gbs_write << "(GB/s), Read - " << gbs_read
            << "(GB/s), Write - " << gbs_write << "(GB/s)" << std::endl
            << std::endl;

  double sub_optimal_threshold = 5.0;  // in percents
  std::cout << std::fixed << std::setprecision(1) << "WG sizes close to optimal ("
            << sub_optimal_threshold << "%): " << std::setprecision(4) << std::endl;
  for (int i = 0; i < wg_sizes.size(); ++i) {
    const auto& wg_size = wg_sizes[i];
    if (time_ms[i] < min_time_ms * (1.0 + sub_optimal_threshold / 100.0)) {
      double scale = min_time_ms / time_ms[i];
      std::cout << "WG size: " << wg_size.x << "x" << wg_size.y << " - time: " << time_ms[i]
                << ", Bandwidth - " << (gbs_read + gbs_write) * scale << "(GB/s)" << std::endl;
    }
  }

  return absl::OkStatus();
}

absl::Status FullyConnectedPerfTest(CalculationsPrecision precision, DataType weights_type,
                                    const BHWC& src_shape, int dst_channels, OHWI scale_zp_shape) {
  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = OHWI(dst_channels, scale_zp_shape.h, 1, src_shape.c);
  weights.data.resize(weights.shape.DimensionsProduct() + XNN_EXTRA_BYTES / sizeof(float));

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(dst_channels, scale_zp_shape.h, 1, src_shape.c);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct() + XNN_EXTRA_BYTES / sizeof(uint8_t));

  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_scale;
  weights_scale.shape = scale_zp_shape;
  weights_scale.data.resize(weights_scale.shape.DimensionsProduct(), 1.0f);
  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_zp;
  weights_zp.shape = scale_zp_shape;
  weights_zp.data.resize(weights_zp.shape.DimensionsProduct(), 0.0f);

  Environment env;
  const GpuInfo& gpu_info = env.GetInfo();

  OperationDef op_def;
  auto data_type = DeduceDataTypeFromPrecision(precision);
  Layout layout = src_shape.b == 1 ? Layout::HWC : Layout::BHWC;
  auto storage_type = TensorStorageType::BUFFER;
  op_def.src_tensors.push_back({data_type, storage_type, layout});
  op_def.dst_tensors.push_back({data_type, storage_type, layout});

  auto dst_shape = src_shape;
  dst_shape.c = dst_channels;

  MetalSpatialTensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(src_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  descriptor_with_shape.SetBHWCShape(dst_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &dst));

  const auto w_shape = weights.shape;

  std::cout << "Src size(HWC) - " << src_shape.h << "x" << src_shape.w << "x" << src_shape.c
            << std::endl;
  std::cout << "Dst size(HWC) - " << dst_shape.h << "x" << dst_shape.w << "x" << dst_shape.c
            << std::endl;

  double element_size = precision == CalculationsPrecision::F32 ? 4.0 : 2.0;
  int64_t flops_per_element = w_shape.i * w_shape.h * w_shape.w * 2;
  if (scale_zp_shape.h > 1) {
    flops_per_element /= scale_zp_shape.h;
  }
  const int64_t dst_elements = dst.Width() * dst.Height() * dst.Channels();
  const int64_t flops_count = dst_elements * flops_per_element;
  const double gflops_count = flops_count * 1e-9;

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const int64_t dst_elements_alignedx4 = dst.Width() * dst.Height() * dst.Slices() * 4;
  const int64_t src_elements_alignedx4 = src.Width() * src.Height() * src.Slices() * 4;
  const double dst_gbytes = dst_elements_alignedx4 * element_size / kGByte;
  const double src_gbytes = src_elements_alignedx4 * element_size / kGByte;
  const double weight_element_size = SizeInBitsOf(weights_type) / 8.0;
  const double weight_gbytes = weights.shape.DimensionsProduct() * weight_element_size / kGByte;
  const double scale_gbytes = scale_zp_shape.DimensionsProduct() * element_size / kGByte;

  std::cout << "Weight size: " << weight_gbytes * 1024.0 << " MB" << std::endl;

  WeightsDescription weights_desc;
  std::vector<TensorDescriptor> weights_gpu;
  if (weights_type == DataType::FLOAT16 || weights_type == DataType::FLOAT32) {
    weights_desc.type = DeduceDataTypeFromPrecision(precision);
    weights_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
    weights_desc.output_group_size = DivideRoundUp(weights.shape.o, 4);
    weights_gpu = GetTensorDescriptorsForWeightsLayout(weights, weights_desc);
  } else if (weights_type == DataType::INT8) {
    weights_desc = GetFullyConnectedInt8WeightsDesc(gpu_info, weights_i8.shape);
    weights_gpu.push_back(GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc));
  } else if (weights_type == DataType::INT4) {
    weights_desc = GetFullyConnectedInt4WeightsDesc(gpu_info, weights_i8.shape);
    weights_gpu.push_back(GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc));
  } else if (weights_type == DataType::INT2) {
    weights_desc = GetFullyConnectedInt2WeightsDesc(gpu_info, weights_i8.shape);
    weights_gpu.push_back(GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc));
  }

  std::vector<MetalSpatialTensor> weights_tensors(weights_gpu.size());
  for (int i = 0; i < weights_gpu.size(); ++i) {
    RETURN_IF_ERROR(CreateTensor(env.device(), weights_gpu[i], &weights_tensors[i]));
  }

  TensorDescriptor scale_desc = ScaleOrZeroPointToTensorDesc(gpu_info, weights_scale, data_type);
  TensorDescriptor zp_desc = ScaleOrZeroPointToTensorDesc(gpu_info, weights_zp, data_type);

  const bool is_quantized = SizeInBitsOf(weights_type) < 16;

  MetalSpatialTensor scale_tensor;
  MetalSpatialTensor zp_tensor;
  if (is_quantized) {
    RETURN_IF_ERROR(CreateTensor(env.device(), scale_desc, &scale_tensor));
    RETURN_IF_ERROR(CreateTensor(env.device(), zp_desc, &zp_tensor));
  }

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights.shape;
  if (is_quantized) {
    external_weights.scale_zp_shape = scale_zp_shape;
    external_weights.scale = &scale_desc;
    external_weights.zero_point = &zp_desc;
  }

  ASSIGN_OR_RETURN(auto operation,
                   CreateFullyConnectedExternalWeights(gpu_info, precision, op_def.src_tensors[0],
                                                       op_def.dst_tensors[0], external_weights,
                                                       /*bias=*/nullptr, &dst_shape));
  std::unique_ptr<GPUOperation> conv = std::make_unique<FullyConnected>(std::move(operation));
  RETURN_IF_ERROR(conv->AssembleCode(gpu_info));

  ComputeTask gpu_task;
  gpu_task.Init(std::move(conv));
  RETURN_IF_ERROR(gpu_task.Compile(&env));
  gpu_task.SetSrcTensor(&src, 0);
  int index = 0;
  gpu_task.SetSrcTensor(&src, index++);
  for (int i = 0; i < weights_tensors.size(); ++i) {
    gpu_task.SetSrcTensor(&weights_tensors[i], index++);
  }
  if (is_quantized) {
    gpu_task.SetSrcTensor(&scale_tensor, index++);
    gpu_task.SetSrcTensor(&zp_tensor, index++);
  }
  gpu_task.SetDstTensor(&dst, 0);
  RETURN_IF_ERROR(gpu_task.UpdateParams());

  id<MTLCommandQueue> command_queue = [env.device() newCommandQueue];
  for (int i = 0; i < 10; ++i) {
    absl::Duration gpu_task_time = gpu_task.GetTaskTime(command_queue);
    double time_ms = absl::ToDoubleMilliseconds(gpu_task_time);
    const double fps = 1000.0 / time_ms;
    const double gflops_real = fps * gflops_count;
    double gbytes_read = src_gbytes + weight_gbytes;
    if (is_quantized) {
      gbytes_read += scale_gbytes * 2.0;
    }
    double gbs_read = fps * gbytes_read;
    const double gbs_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(4) << " Time - " << time_ms << "(ms), GFlops - "
              << gflops_real << ", Bandwidth - " << gbs_read + gbs_write << "(GB/s), Read - "
              << gbs_read << "(GB/s), Write - " << gbs_write << "(GB/s)" << std::endl;
  }

  return absl::OkStatus();
}

absl::Status FullyConnectedInt4Sparse2x4PerfTest(CalculationsPrecision precision,
                                                 const BHWC& src_shape, int dst_channels) {
  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_scale;
  weights_scale.shape = OHWI(dst_channels, 1, 1, 1);
  weights_scale.data.resize(weights_scale.shape.DimensionsProduct(), 1.0f);
  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_zp;
  weights_zp.shape = OHWI(dst_channels, 1, 1, 1);
  weights_zp.data.resize(weights_zp.shape.DimensionsProduct(), 0.0f);
  ml_drift::Tensor<Linear, DataType::FLOAT32> bias;
  bias.shape = Linear(dst_channels);
  bias.data.resize(bias.shape.DimensionsProduct());

  const OHWI weights_shape_dense = OHWI(dst_channels, 1, 1, src_shape.c);
  const OHWI weights_shape_sparse = OHWI(dst_channels, 1, 1, src_shape.c / 2);

  ml_drift::Tensor<OHWI, DataType::INT8> weights;
  weights.shape = weights_shape_sparse;
  weights.data.resize(weights.shape.DimensionsProduct());
  ml_drift::Tensor<OHWI, DataType::UINT8> weights_indices;
  weights_indices.shape = weights_shape_sparse;
  weights_indices.data.resize(weights_indices.shape.DimensionsProduct());

  Environment env;

  OperationDef op_def;
  auto data_type = DeduceDataTypeFromPrecision(precision);
  Layout layout = src_shape.b == 1 ? Layout::HWC : Layout::BHWC;
  auto storage_type = TensorStorageType::BUFFER;
  op_def.src_tensors.push_back({data_type, storage_type, layout});
  op_def.dst_tensors.push_back({data_type, storage_type, layout});

  auto dst_shape = src_shape;
  dst_shape.c = dst_channels;

  MetalSpatialTensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(src_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  descriptor_with_shape.SetBHWCShape(dst_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &dst));

  std::cout << "Src size(HWC) - " << src_shape.h << "x" << src_shape.w << "x" << src_shape.c
            << std::endl;
  std::cout << "Dst size(HWC) - " << dst_shape.h << "x" << dst_shape.w << "x" << dst_shape.c
            << std::endl;

  double element_size = precision == CalculationsPrecision::F32 ? 4.0 : 2.0;
  double weight_element_size = (0.5 + 0.25) * 0.5;
  const int64_t flops_per_element = weights_shape_dense.i * 2;
  const int64_t dst_elements = dst.Width() * dst.Height() * dst.Channels();
  const int64_t flops_count = dst_elements * flops_per_element;
  const double gflops_count = flops_count * 1e-9;

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const int64_t dst_elements_alignedx4 = dst.Width() * dst.Height() * dst.Slices() * 4;
  const int64_t src_elements_alignedx4 = src.Width() * src.Height() * src.Slices() * 4;
  const double dst_gbytes = dst_elements_alignedx4 * element_size / kGByte;
  const double src_gbytes = src_elements_alignedx4 * element_size / kGByte;
  const double weight_gbytes =
      weights_shape_dense.DimensionsProduct() * weight_element_size / kGByte;
  const double bias_gbytes = weights_shape_dense.o * element_size / kGByte;
  const double scale_gbytes = weights_scale.shape.DimensionsProduct() * element_size / kGByte;

  std::unique_ptr<GPUOperation> conv = std::make_unique<FullyConnected>(
      CreateFullyConnectedInt4Sparse2x4(env.GetInfo(), op_def, precision, weights, weights_indices,
                                        weights_scale, weights_zp, bias, &dst_shape));
  RETURN_IF_ERROR(conv->AssembleCode(env.GetInfo()));

  ComputeTask gpu_task;
  gpu_task.Init(std::move(conv));
  RETURN_IF_ERROR(gpu_task.Compile(&env));
  gpu_task.SetSrcTensor(&src, 0);
  gpu_task.SetDstTensor(&dst, 0);
  RETURN_IF_ERROR(gpu_task.UpdateParams());

  id<MTLCommandQueue> command_queue = [env.device() newCommandQueue];
  for (int i = 0; i < 10; ++i) {
    absl::Duration gpu_task_time = gpu_task.GetTaskTime(command_queue);
    double time_ms = absl::ToDoubleMilliseconds(gpu_task_time);
    const double fps = 1000.0 / time_ms;
    const double gflops_real = fps * gflops_count;
    double gbytes_read = src_gbytes + weight_gbytes + bias_gbytes;
    gbytes_read += scale_gbytes * 2.0;
    double gbs_read = fps * gbytes_read;
    const double gbs_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(4) << " Time - " << time_ms << "(ms), GFlops - "
              << gflops_real << ", Bandwidth - " << gbs_read + gbs_write << "(GB/s), Read - "
              << gbs_read << "(GB/s), Write - " << gbs_write << "(GB/s)" << std::endl;
  }

  return absl::OkStatus();
}

absl::Status AddScalarTest(const BHWC& shape, const DataType& data_type) {
  Environment env;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, TensorStorageType::BUFFER, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, TensorStorageType::BUFFER, Layout::HWC});
  ElementwiseAttributes attr;
  attr.param = 0.5f;
  auto add = std::make_unique<GPUOperation>(
      CreateElementwise(env.GetInfo(), op_def, OperationType::ADD, attr));

  MetalSpatialTensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  descriptor_with_shape.SetBHWCShape(shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &dst));

  RETURN_IF_ERROR(add->AssembleCode(env.GetInfo()));

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const int64_t dst_elements_alignedx4 = dst.Width() * dst.Height() * dst.Slices() * 4;
  const int64_t src_elements_alignedx4 = src.Width() * src.Height() * src.Slices() * 4;
  const double dst_gbytes = dst_elements_alignedx4 * SizeOf(data_type) / kGByte;
  const double src_gbytes = src_elements_alignedx4 * SizeOf(data_type) / kGByte;

  std::cout << "Size(HWC) - " << shape.h << "x" << shape.w << "x" << shape.c << std::endl;
  std::cout << "Size MB - " << src_gbytes * 1024.0 << std::endl;

  ComputeTask gpu_task;
  gpu_task.Init(std::move(add));
  RETURN_IF_ERROR(gpu_task.Compile(&env));
  gpu_task.SetSrcTensor(&src, 0);
  gpu_task.SetDstTensor(&dst, 0);
  RETURN_IF_ERROR(gpu_task.UpdateParams());

  id<MTLCommandQueue> command_queue = [env.device() newCommandQueue];
  for (int i = 0; i < 10; ++i) {
    absl::Duration gpu_task_time = gpu_task.GetTaskTime(command_queue);
    double time_ms = absl::ToDoubleMilliseconds(gpu_task_time);

    const double fps = 1000.0 / time_ms;
    const double gbyte_read = fps * src_gbytes;
    const double gbyte_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(3) << i << "  Time - " << time_ms
              << std::setprecision(2) << "(ms), Bandwidth - " << gbyte_read + gbyte_write
              << "(GB/s), Read - " << gbyte_read << "(GB/s), Write - " << gbyte_write << "(GB/s)"
              << std::endl;
  }

  return absl::OkStatus();
}

absl::Status WinogradForwardTest(const BHWC& src_shape, const DataType& data_type) {
  Environment env;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, TensorStorageType::BUFFER, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, TensorStorageType::BUFFER, Layout::HWC});
  Padding2D padding;
  padding.prepended = HW(1, 1);
  padding.appended = HW(1, 1);
  const int kTileSize = 6;
  auto operation = std::make_unique<Winograd3x3TiledXForward>(
      CreateWinograd3x3TiledXForward(env.GetInfo(), op_def, padding, kTileSize));
  // auto operation = std::make_unique<Winograd4x4To36>(CreateWinograd4x4To36(op_def, padding,
  // env.GetInfo()));

  BHWC dst_shape = src_shape;
  const int tile_size_outer = kTileSize;
  const int tile_size_inner = tile_size_outer - 2;
  const int tiles_x = DivideRoundUp(src_shape.w, tile_size_inner);
  const int tiles_y = DivideRoundUp(src_shape.h, tile_size_inner);
  dst_shape.h = tile_size_outer * tile_size_outer;
  dst_shape.w = tiles_x * tiles_y;

  MetalSpatialTensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(src_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  descriptor_with_shape.SetBHWCShape(dst_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &dst));

  RETURN_IF_ERROR(operation->AssembleCode(env.GetInfo()));

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const int64_t dst_elements_alignedx4 = dst.Width() * dst.Height() * dst.Slices() * 4;
  const int64_t src_elements_alignedx4 = src.Width() * src.Height() * src.Slices() * 4;
  const double dst_gbytes = dst_elements_alignedx4 * SizeOf(data_type) / kGByte;
  const double src_gbytes = src_elements_alignedx4 * SizeOf(data_type) / kGByte;

  std::cout << "Src size(HWC) - " << src_shape.h << "x" << src_shape.w << "x" << src_shape.c
            << std::endl;
  std::cout << "Size MB - " << src_gbytes * 1024.0 << std::endl;
  std::cout << "Dst size(HWC) - " << dst_shape.h << "x" << dst_shape.w << "x" << dst_shape.c
            << std::endl;
  std::cout << "Size MB - " << dst_gbytes * 1024.0 << std::endl;

  ComputeTask gpu_task;
  gpu_task.Init(std::move(operation));
  RETURN_IF_ERROR(gpu_task.Compile(&env));
  gpu_task.SetSrcTensor(&src, 0);
  gpu_task.SetDstTensor(&dst, 0);
  RETURN_IF_ERROR(gpu_task.UpdateParams());

  id<MTLCommandQueue> command_queue = [env.device() newCommandQueue];
  for (int i = 0; i < 10; ++i) {
    absl::Duration gpu_task_time = gpu_task.GetTaskTime(command_queue);
    double time_ms = absl::ToDoubleMilliseconds(gpu_task_time);

    const double fps = 1000.0 / time_ms;
    const double gbyte_read = fps * src_gbytes;
    const double gbyte_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(3) << i << "  Time - " << time_ms
              << std::setprecision(2) << "(ms), Bandwidth - " << gbyte_read + gbyte_write
              << "(GB/s), Read - " << gbyte_read << "(GB/s), Write - " << gbyte_write << "(GB/s)"
              << std::endl;
  }
  return absl::OkStatus();
}

absl::Status WinogradBackwardTest(const BHWC& dst_shape, const DataType& data_type) {
  Environment env;
  OperationDef op_def;
  op_def.src_tensors.push_back({data_type, TensorStorageType::BUFFER, Layout::HWC});
  op_def.dst_tensors.push_back({data_type, TensorStorageType::BUFFER, Layout::HWC});
  Tensor<Linear, DataType::FLOAT32> biases;
  biases.shape = Linear(dst_shape.c);
  biases.data.resize(biases.shape.DimensionsProduct());
  for (int i = 0; i < biases.data.size(); ++i) {
    biases.data[i] = 0.0f;
  }
  const int kTileSize = 6;
  auto operation = std::make_unique<Winograd3x3TiledXBackward>(
      CreateWinograd3x3TiledXBackward(env.GetInfo(), op_def, biases, kTileSize));
  // auto operation = std::make_unique<Winograd36To4x4>(CreateWinograd36To4x4(op_def, biases));

  BHWC src_shape = dst_shape;
  const int tile_size_outer = kTileSize;
  const int tile_size_inner = tile_size_outer - 2;
  const int tiles_x = DivideRoundUp(dst_shape.w, tile_size_inner);
  const int tiles_y = DivideRoundUp(dst_shape.h, tile_size_inner);
  src_shape.h = tile_size_outer * tile_size_outer;
  src_shape.w = tiles_x * tiles_y;

  MetalSpatialTensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(src_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  descriptor_with_shape.SetBHWCShape(dst_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &dst));

  RETURN_IF_ERROR(operation->AssembleCode(env.GetInfo()));

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const int64_t dst_elements_alignedx4 = dst.Width() * dst.Height() * dst.Slices() * 4;
  const int64_t src_elements_alignedx4 = src.Width() * src.Height() * src.Slices() * 4;
  const double dst_gbytes = dst_elements_alignedx4 * SizeOf(data_type) / kGByte;
  const double src_gbytes = src_elements_alignedx4 * SizeOf(data_type) / kGByte;

  std::cout << "Dst size(HWC) - " << src_shape.h << "x" << src_shape.w << "x" << src_shape.c
            << std::endl;
  std::cout << "Size MB - " << src_gbytes * 1024.0 << std::endl;
  std::cout << "Src size(HWC) - " << dst_shape.h << "x" << dst_shape.w << "x" << dst_shape.c
            << std::endl;
  std::cout << "Size MB - " << dst_gbytes * 1024.0 << std::endl;

  ComputeTask gpu_task;
  gpu_task.Init(std::move(operation));
  RETURN_IF_ERROR(gpu_task.Compile(&env));
  gpu_task.SetSrcTensor(&src, 0);
  gpu_task.SetDstTensor(&dst, 0);
  RETURN_IF_ERROR(gpu_task.UpdateParams());

  id<MTLCommandQueue> command_queue = [env.device() newCommandQueue];
  for (int i = 0; i < 10; ++i) {
    absl::Duration gpu_task_time = gpu_task.GetTaskTime(command_queue);
    double time_ms = absl::ToDoubleMilliseconds(gpu_task_time);

    const double fps = 1000.0 / time_ms;
    const double gbyte_read = fps * src_gbytes;
    const double gbyte_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(3) << i << "  Time - " << time_ms
              << std::setprecision(2) << "(ms), Bandwidth - " << gbyte_read + gbyte_write
              << "(GB/s), Read - " << gbyte_read << "(GB/s), Write - " << gbyte_write << "(GB/s)"
              << std::endl;
  }
  return absl::OkStatus();
}

absl::Status DepthwiseConvPerfTest(CalculationsPrecision precision, const BHWC& src_shape,
                                   const HW& kernel_size, const HW& strides, const HW& dilation) {
  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(kernel_size.h / 2, kernel_size.w / 2);
  attr.padding.appended = HW(kernel_size.h / 2, kernel_size.w / 2);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto& attr_weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(1, kernel_size.h, kernel_size.w, src_shape.c);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct());
  attr.bias.shape = Linear(src_shape.c);
  attr.bias.data.resize(attr.bias.shape.DimensionsProduct());

  OperationDef op_def;
  auto data_type = DeduceDataTypeFromPrecision(precision);
  Layout layout = src_shape.b == 1 ? Layout::HWC : Layout::BHWC;
  op_def.src_tensors.push_back({data_type, TensorStorageType::BUFFER, layout});
  op_def.dst_tensors.push_back({data_type, TensorStorageType::BUFFER, layout});

  RETURN_IF_ERROR(TestDepthwiseConvPerformance(attr, src_shape, op_def, precision));

  return absl::OkStatus();
}

absl::Status QuantizationPerfTest(const BHWC& src_shape, DataType src_type, PackedType dst_type,
                                  bool calculate_sum) {
  Environment env;

  Layout layout = src_shape.b == 1 ? Layout::HWC : Layout::BHWC;
  TensorDescriptor src_tensor_desc{src_type, TensorStorageType::BUFFER, layout};
  TensorDescriptor dst_tensor_desc{ToSpatialTensorType(dst_type), TensorStorageType::BUFFER,
                                   layout};
  TensorDescriptor& params_tensor_desc = src_tensor_desc;
  BHWC params_shape = src_shape;
  params_shape.c = calculate_sum ? 3 : 2;
  BHWC dst_shape = GetShapeForPackedType(src_shape, dst_type);

  OperationDef op_def;
  op_def.src_tensors.push_back(src_tensor_desc);
  op_def.dst_tensors.push_back(dst_tensor_desc);
  op_def.dst_tensors.push_back(params_tensor_desc);
  auto quant_op = CreateQuantization(op_def, dst_type, env.GetInfo(), src_shape, calculate_sum);

  MetalSpatialTensor src, dst, params;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(src_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  descriptor_with_shape.SetBHWCShape(dst_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &dst));
  descriptor_with_shape = op_def.dst_tensors[1];
  descriptor_with_shape.SetBHWCShape(params_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &params));

  std::cout << "Src size(HWC) - " << src_shape.h << "x" << src_shape.w << "x" << src_shape.c
            << std::endl;
  std::cout << "Dst size(HWC) - " << dst_shape.h << "x" << dst_shape.w << "x" << dst_shape.c
            << std::endl;

  RETURN_IF_ERROR(quant_op->AssembleCode(env.GetInfo()));

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const double dst_gbytes = (dst.GetMemorySizeInBytes() + params.GetMemorySizeInBytes()) / kGByte;
  const double src_gbytes = src.GetMemorySizeInBytes() / kGByte;

  ComputeTask gpu_task;
  gpu_task.Init(std::move(quant_op));
  RETURN_IF_ERROR(gpu_task.Compile(&env));
  gpu_task.SetSrcTensor(&src, 0);
  gpu_task.SetDstTensor(&dst, 0);
  gpu_task.SetDstTensor(&params, 1);
  RETURN_IF_ERROR(gpu_task.UpdateParams());

  id<MTLCommandQueue> command_queue = [env.device() newCommandQueue];
  for (int i = 0; i < 5; ++i) {
    absl::Duration gpu_task_time = gpu_task.GetTaskTime(command_queue);
    double time_ms = absl::ToDoubleMilliseconds(gpu_task_time);
    const double fps = 1000.0 / time_ms;
    const double gbyte_read = fps * src_gbytes;
    const double gbyte_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(3) << "  Time - " << time_ms
              << std::setprecision(2) << "(ms), Bandwidth - " << gbyte_read + gbyte_write
              << "(GB/s), Read - " << gbyte_read << "(GB/s), Write - " << gbyte_write << "(GB/s)"
              << std::endl;
  }

  return absl::OkStatus();
}

absl::Status SoftmaxPerfTest(const BHWC& shape, bool reduce_only) {
  Environment env;
  const auto precision = CalculationsPrecision::F16;
  OperationDef op_def;
  auto data_type = DeduceDataTypeFromPrecision(precision);
  auto storage_type = TensorStorageType::BUFFER;
  Layout layout = shape.b == 1 ? Layout::HWC : Layout::BHWC;
  TensorDescriptor tensor_desc{data_type, storage_type, layout};
  RETURN_IF_ERROR(tensor_desc.UpdateToSupportedStorageType(env.GetInfo(), shape));
  op_def.src_tensors.push_back(tensor_desc);
  op_def.dst_tensors.push_back(tensor_desc);

  std::unique_ptr<GPUOperation> softmax;
  if (reduce_only) {
    // softmax = std::make_unique<Softmax1x1>(CreateSoftmax1x1Reduce(
    //    op_def, env.GetInfo(), shape));

    softmax = std::make_unique<Softmax>(CreateSoftmaxReduce(op_def, env.GetInfo(), shape));
  } else {
    // softmax = std::make_unique<Softmax1x1>(
    //    CreateSoftmax1x1(op_def, env.GetInfo(), shape));

    softmax = std::make_unique<Softmax>(CreateSoftmax(op_def, env.GetInfo(), shape));
  }

  MetalSpatialTensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  BHWC dst_shape = shape;
  if (reduce_only) {
    dst_shape.c = 2;
  }
  descriptor_with_shape.SetBHWCShape(dst_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &dst));

  std::cout << "Src shape - " << ToString(shape) << std::endl;
  std::cout << "Dst shape - " << ToString(dst_shape) << std::endl;

  RETURN_IF_ERROR(softmax->AssembleCode(env.GetInfo()));

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const double src_gbytes = src.GetMemorySizeInBytes() / kGByte;
  const double dst_gbytes = dst.GetMemorySizeInBytes() / kGByte;

  ComputeTask gpu_task;
  gpu_task.Init(std::move(softmax));
  RETURN_IF_ERROR(gpu_task.Compile(&env));
  gpu_task.SetSrcTensor(&src, 0);
  gpu_task.SetDstTensor(&dst, 0);
  RETURN_IF_ERROR(gpu_task.UpdateParams());

  id<MTLCommandQueue> command_queue = [env.device() newCommandQueue];
  for (int i = 0; i < 5; ++i) {
    absl::Duration gpu_task_time = gpu_task.GetTaskTime(command_queue);
    const auto time_ms = absl::ToDoubleMilliseconds(gpu_task_time);
    const double fps = 1000.0 / time_ms;
    const double gbyte_read = fps * (src_gbytes);
    const double gbyte_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(3) << "  Time - " << time_ms
              << std::setprecision(3) << "(ms), Bandwidth - " << gbyte_read + gbyte_write
              << "(GB/s), Read - " << gbyte_read << "(GB/s), Write - " << gbyte_write << "(GB/s)"
              << std::endl;
  }

  return absl::OkStatus();
}
}  // namespace metal
}  // namespace ml_drift
