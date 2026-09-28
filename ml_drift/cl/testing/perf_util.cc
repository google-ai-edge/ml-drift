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

#include "ml_drift/cl/testing/perf_util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <iostream>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "ml_drift/cl/cl_operation.h"
#include "ml_drift/cl/environment.h"
#include "ml_drift/cl/tensor.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/flops_util.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernels/conv_constants.h"
#include "ml_drift/common/kernels/conv_generic.h"
#include "ml_drift/common/kernels/conv_wave_matrix_mali.h"
#include "ml_drift/common/kernels/conv_wave_memory.h"
#include "ml_drift/common/kernels/depthwise_conv.h"
#include "ml_drift/common/kernels/depthwise_conv_3x3.h"
#include "ml_drift/common/kernels/depthwise_conv_tiled.h"
#include "ml_drift/common/kernels/depthwise_conv_wave_memory.h"
#include "ml_drift/common/kernels/fully_connected.h"
#include "ml_drift/common/kernels/fully_connected_oi.h"
#include "ml_drift/common/kernels/quantize_and_dequantize.h"
#include "ml_drift/common/kernels/softmax.h"
#include "ml_drift/common/kernels/special/conv_softmax_conv.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/selectors/convolution_selector.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/testing_ref_ops.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"
#include "ml_drift/common/winograd_util.h"

namespace ml_drift {
namespace cl {
// For Bandwidth parameter we have rough estimation, because we don't use in
// calculations different levels of caches(we don't know values for this
// parameters and even if we knew it, it would be very complex and not universal
// for different kernels)
absl::Status TestConvolutionPerformance(const Convolution2DAttributes& attr,
                                        const BHWC& src_shape,
                                        const OperationDef& op_def,
                                        CalculationsPrecision precision,
                                        Environment* env) {
  const auto dst_shape = CalculateOutputShape(src_shape, attr);
  const GpuInfo& gpu_info = env->GetDevicePtr()->GetInfo();

  std::unique_ptr<GPUOperation> conv;
  if (IsConvWaveMemorySupported(gpu_info)) {
    conv = std::make_unique<ConvWaveMemory>(
        CreateConvWaveMemory(gpu_info, op_def, precision, attr, &dst_shape));
  } else if (/* DISABLES CODE */ (true)) {
    conv = std::make_unique<ConvGeneric>(
        CreateConvGeneric(gpu_info, op_def, precision, attr, &dst_shape));
  } else {
    conv = std::make_unique<ConvConstants>(
        CreateConvConstants(gpu_info, op_def, precision, attr));
  }

  //{
  //  FullyConnectedAttributes fc_attr;
  //  fc_attr.weights = attr.weights;
  //  fc_attr.bias = attr.bias;
  //  conv = std::make_unique<FullyConnected>(CreateFullyConnected(
  //    gpu_info, op_def, precision, fc_attr, &dst_shape));
  //}

  Tensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(src_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env->context(), descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  descriptor_with_shape.SetBHWCShape(dst_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env->context(), descriptor_with_shape, &dst));

  const auto w_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);

  std::cout << "Src size(BHWC) - " << src_shape.b << "x" << src_shape.h << "x"
            << src_shape.w << "x" << src_shape.c << std::endl;
  std::cout << "Dst size(BHWC) - " << dst_shape.b << "x" << dst_shape.h << "x"
            << dst_shape.w << "x" << dst_shape.c << std::endl;
  std::cout << "Convolution attributes : " << std::endl;
  std::cout << "  Kernel size - " << w_shape.w << "x" << w_shape.h << std::endl;
  std::cout << "  Stride - " << attr.strides.w << "x" << attr.strides.h
            << std::endl;
  std::cout << "  Dilation - " << attr.dilations.w << "x" << attr.dilations.h
            << std::endl;
  std::cout << "  Padding (prepended) - " << attr.padding.prepended.w << "x"
            << attr.padding.prepended.h << ", (appended) - "
            << attr.padding.appended.w << "x" << attr.padding.appended.h
            << std::endl;

  ABSL_RETURN_IF_ERROR(conv->AssembleCode(gpu_info));

  const int float_size = precision == CalculationsPrecision::kF32 ? 4 : 2;
  const int64_t flops_count = GetConvolutionFlops(dst_shape, w_shape);
  const double gflops_count = flops_count * 1e-9;

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const double dst_gbytes = dst.GetMemorySizeInBytes() / kGByte;
  const double src_gbytes = src.GetMemorySizeInBytes() / kGByte;
  const double weight_gbytes =
      w_shape.DimensionsProduct() * float_size / kGByte;
  const double bias_gbytes = w_shape.o * float_size / kGByte;

  ClOperation cl_op;
  cl_op.Init(std::move(conv));
  ABSL_RETURN_IF_ERROR(cl_op.Compile(env->GetDevicePtr(), &env->context(),
                                     env->program_cache()));
  ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(0, &src));
  ABSL_RETURN_IF_ERROR(cl_op.SetDstTensor(0, &dst));
  ABSL_RETURN_IF_ERROR(cl_op.UpdateParams());
  ABSL_RETURN_IF_ERROR(cl_op.Tune(TuningType::kExhaustive,
                                  env->device().GetInfo(),
                                  env->profiling_queue()));

  for (int i = 0; i < 5; ++i) {
    ABSL_ASSIGN_OR_RETURN(auto duration,
                          cl_op.GetOpTime(env->device().GetInfo(), env->queue(),
                                          env->profiling_queue()));
    const auto time_ms = absl::ToDoubleMilliseconds(duration);
    const double fps = 1000.0 / time_ms;
    const double gflops_real = fps * gflops_count;
    const double gbyte_read = fps * (src_gbytes + weight_gbytes + bias_gbytes);
    const double gbyte_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(3) << "  Time - " << time_ms
              << std::setprecision(2) << "(ms), GFlops - " << gflops_real
              << ", Bandwidth - " << gbyte_read + gbyte_write
              << "(GB/s), Read - " << gbyte_read << "(GB/s), Write - "
              << gbyte_write << "(GB/s)" << std::endl;
  }

  return absl::OkStatus();
}

absl::Status TestConvSoftmaxConvPerformance(const BHWC& dst_shape, int src_ch0,
                                            int src_ch1,
                                            const OperationDef& op_def,
                                            CalculationsPrecision precision,
                                            Environment* env) {
  const GpuInfo& gpu_info = env->GetDevicePtr()->GetInfo();

  ml_drift::Tensor<OHWI, DataType::kFloat32> weights0;
  weights0.shape = OHWI(src_ch1, 1, dst_shape.h, src_ch0);
  weights0.data.resize(weights0.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  ml_drift::Tensor<OHWI, DataType::kFloat32> weights1;
  weights1.shape = OHWI(dst_shape.c, 1, dst_shape.h, src_ch1);
  weights1.data.resize(weights1.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));

  ConvSoftmaxConv::WeightsDesc weights_desc;
  weights_desc.constant = true;
  weights_desc.weights0 = &weights0;
  weights_desc.weights1 = &weights1;
  std::unique_ptr<GPUOperation> conv = std::make_unique<ConvSoftmaxConv>(
      CreateConvSoftmaxConv(gpu_info, op_def, precision, weights_desc));

  Tensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  const BHWC src_shape = BHWC(dst_shape.b, dst_shape.h, dst_shape.w, src_ch0);
  descriptor_with_shape.SetBHWCShape(src_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env->context(), descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  descriptor_with_shape.SetBHWCShape(dst_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env->context(), descriptor_with_shape, &dst));

  std::cout << "Src size(HWC) - " << src_shape.h << "x" << src_shape.w << "x"
            << src_shape.c << std::endl;
  std::cout << "Dst size(HWC) - " << dst_shape.h << "x" << dst_shape.w << "x"
            << dst_shape.c << std::endl;
  std::cout << "Attention attributes : " << std::endl;

  ABSL_RETURN_IF_ERROR(conv->AssembleCode(gpu_info));

  const int float_size = precision == CalculationsPrecision::kF32 ? 4 : 2;
  const int64_t conv0_flops_per_element = weights0.shape.i * 2;
  const int64_t conv0_dst_elements =
      dst.Width() * dst.Height() * weights0.shape.o;
  const int64_t conv0_flops_count =
      conv0_dst_elements * conv0_flops_per_element;
  const int64_t conv1_flops_per_element = weights1.shape.i * 2;
  const int64_t conv1_dst_elements =
      dst.Width() * dst.Height() * weights1.shape.o;
  const int64_t conv1_flops_count =
      conv1_dst_elements * conv1_flops_per_element;
  const int64_t flops_count = conv0_flops_count + conv1_flops_count;
  const double gflops_count = flops_count * 1e-9;

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const int64_t dst_elements_alignedx4 =
      dst.Width() * dst.Height() * dst.Slices() * 4;
  const int64_t src_elements_alignedx4 =
      src.Width() * src.Height() * src.Slices() * 4;
  const double dst_gbytes = dst_elements_alignedx4 * float_size / kGByte;
  const double src_gbytes = src_elements_alignedx4 * float_size / kGByte;
  const double weight_gbytes = (weights0.shape.DimensionsProduct() +
                                weights1.shape.DimensionsProduct()) *
                               float_size / kGByte;

  ClOperation cl_op;
  cl_op.Init(std::move(conv));
  ABSL_RETURN_IF_ERROR(cl_op.Compile(env->GetDevicePtr(), &env->context(),
                                     env->program_cache()));
  ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(0, &src));
  ABSL_RETURN_IF_ERROR(cl_op.SetDstTensor(0, &dst));
  ABSL_RETURN_IF_ERROR(cl_op.UpdateParams());
  ABSL_RETURN_IF_ERROR(cl_op.Tune(TuningType::kExhaustive,
                                  env->device().GetInfo(),
                                  env->profiling_queue()));

  for (int i = 0; i < 5; ++i) {
    ABSL_ASSIGN_OR_RETURN(auto duration,
                          cl_op.GetOpTime(env->device().GetInfo(), env->queue(),
                                          env->profiling_queue()));
    const auto time_ms = absl::ToDoubleMilliseconds(duration);
    const double fps = 1000.0 / time_ms;
    const double gflops_real = fps * gflops_count;
    const double gbyte_read = fps * (src_gbytes + weight_gbytes);
    const double gbyte_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(3) << "  Time - " << time_ms
              << std::setprecision(2) << "(ms), GFlops - " << gflops_real
              << ", Bandwidth - " << gbyte_read + gbyte_write
              << "(GB/s), Read - " << gbyte_read << "(GB/s), Write - "
              << gbyte_write << "(GB/s)" << std::endl;
  }

  return absl::OkStatus();
}

absl::Status TestDepthwiseConvPerformance(
    const DepthwiseConvolution2DAttributes& attr, const BHWC& src_shape,
    const OperationDef& op_def, CalculationsPrecision precision,
    Environment* env) {
  const GpuInfo& gpu_info = env->GetDevicePtr()->GetInfo();

  const auto dst_shape = CalculateOutputShape(src_shape, attr);

  std::unique_ptr<GPUOperation> conv;
  bool use_depthwise_conv_3x3 = false;
  bool use_depthwise_conv_tiled = true;
  bool use_depthwise_conv_base = false;
  bool use_depthwise_conv_wave_memory = false;
  if (use_depthwise_conv_3x3) {
    conv = std::make_unique<DepthwiseConv3x3>(
        CreateDepthwiseConv3x3(gpu_info, op_def, precision, attr));
  }
  if (use_depthwise_conv_tiled) {
    conv = CreateDepthwiseConvTiled(gpu_info, op_def, precision, attr);
  }
  if (use_depthwise_conv_base) {
    conv = std::make_unique<DepthwiseConv>(
        CreateDepthwiseConvolution2D(gpu_info, op_def, precision, attr));
  }
  if (use_depthwise_conv_wave_memory) {
    conv = std::make_unique<DepthwiseConvWaveMemory>(
        CreateDepthwiseConvWaveMemory(gpu_info, op_def, precision, attr));
  }

  Tensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(src_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env->context(), descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  descriptor_with_shape.SetBHWCShape(dst_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env->context(), descriptor_with_shape, &dst));

  const auto w_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);

  std::cout << "Src size(HWC) - " << src_shape.h << "x" << src_shape.w << "x"
            << src_shape.c << std::endl;
  std::cout << "Dst size(HWC) - " << dst_shape.h << "x" << dst_shape.w << "x"
            << dst_shape.c << std::endl;
  std::cout << "DepthwiseConvolution attributes : " << std::endl;
  std::cout << "  Kernel size - " << w_shape.w << "x" << w_shape.h << std::endl;
  std::cout << "  Stride - " << attr.strides.w << "x" << attr.strides.h
            << std::endl;
  std::cout << "  Dilation - " << attr.dilations.w << "x" << attr.dilations.h
            << std::endl;
  std::cout << "  Padding (prepended) - " << attr.padding.prepended.w << "x"
            << attr.padding.prepended.h << ", (appended) - "
            << attr.padding.appended.w << "x" << attr.padding.appended.h
            << std::endl;

  ABSL_RETURN_IF_ERROR(conv->AssembleCode(gpu_info));

  const int64_t flops_count = GetDepthwiseConvolutionFlops(dst_shape, w_shape);

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const int64_t dst_elements_alignedx4 =
      dst.Width() * dst.Height() * dst.Slices() * 4;
  const int64_t src_elements_alignedx4 =
      src.Width() * src.Height() * src.Slices() * 4;
  const int float_size = precision == CalculationsPrecision::kF32 ? 4 : 2;
  const double dst_gbytes = dst_elements_alignedx4 * float_size / kGByte;
  const double src_gbytes = src_elements_alignedx4 * float_size / kGByte;
  const double weight_gbytes =
      w_shape.DimensionsProduct() * float_size / kGByte;
  const double bias_gbytes = w_shape.o * float_size / kGByte;

  ClOperation cl_op;
  cl_op.Init(std::move(conv));
  ABSL_RETURN_IF_ERROR(cl_op.Compile(env->GetDevicePtr(), &env->context(),
                                     env->program_cache()));
  ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(0, &src));
  ABSL_RETURN_IF_ERROR(cl_op.SetDstTensor(0, &dst));
  ABSL_RETURN_IF_ERROR(cl_op.UpdateParams());
  ABSL_RETURN_IF_ERROR(cl_op.Tune(TuningType::kExhaustive,
                                  env->device().GetInfo(),
                                  env->profiling_queue()));

  for (int i = 0; i < 5; ++i) {
    ABSL_ASSIGN_OR_RETURN(auto duration,
                          cl_op.GetOpTime(env->device().GetInfo(), env->queue(),
                                          env->profiling_queue()));
    const auto time_ms = absl::ToDoubleMilliseconds(duration);
    const double fps = 1000.0 / time_ms;
    const double gflops_real = fps * flops_count * 1e-9;
    const double gbyte_read = fps * (src_gbytes + weight_gbytes + bias_gbytes);
    const double gbyte_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(3) << "  Time - " << time_ms
              << std::setprecision(2) << "(ms), GFlops - " << gflops_real
              << ", Bandwidth - " << gbyte_read + gbyte_write
              << "(GB/s), Read - " << gbyte_read << "(GB/s), Write - "
              << gbyte_write << "(GB/s)" << std::endl;
  }

  return absl::OkStatus();
}

absl::Status ConvolutionPerfTest(CalculationsPrecision precision,
                                 const BHWC& src_shape, int dst_channels,
                                 const HW& kernel_size, const HW& strides,
                                 const HW& dilations) {
  Environment env;
  ABSL_RETURN_IF_ERROR(CreateEnvironment(&env));

  Convolution2DAttributes attr;
  attr.padding.prepended = HW(kernel_size.h / 2, kernel_size.w / 2);
  attr.padding.appended = HW(kernel_size.h / 2, kernel_size.w / 2);
  attr.strides = strides;
  attr.dilations = dilations;
  auto& attr_weights =
      attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::kFloat32>>();
  attr_weights.shape =
      OHWI(dst_channels, kernel_size.h, kernel_size.w, src_shape.c);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias.shape = Linear(dst_channels);
  attr.bias.data.resize(attr.bias.shape.DimensionsProduct());

  OperationDef op_def;
  auto data_type = DeduceDataTypeFromPrecision(precision);
  Layout layout = src_shape.b == 1 ? Layout::kHWC : Layout::kBHWC;
  auto storage_type = GetFastestStorageType(env.device().GetInfo());
  TensorDescriptor src_tensor_desc{data_type, storage_type, layout};
  ABSL_RETURN_IF_ERROR(src_tensor_desc.UpdateToSupportedStorageType(
      env.device().GetInfo(), src_shape));

  const auto dst_shape = CalculateOutputShape(src_shape, attr);
  TensorDescriptor dst_tensor_desc{data_type, storage_type, layout};
  ABSL_RETURN_IF_ERROR(dst_tensor_desc.UpdateToSupportedStorageType(
      env.device().GetInfo(), dst_shape));

  op_def.src_tensors.push_back(src_tensor_desc);
  op_def.dst_tensors.push_back(dst_tensor_desc);

  ABSL_RETURN_IF_ERROR(
      TestConvolutionPerformance(attr, src_shape, op_def, precision, &env));

  return absl::OkStatus();
}

absl::Status ConvolutionWinogradPerfTest(CalculationsPrecision precision,
                                         const BHWC& src_shape,
                                         int dst_channels) {
  Environment env;
  ABSL_RETURN_IF_ERROR(CreateEnvironment(&env));

  Convolution2DAttributes attr;
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(1, 1);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto& attr_weights =
      attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::kFloat32>>();
  attr_weights.shape = OHWI(dst_channels, 3, 3, src_shape.c);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct());
  attr.bias.shape = Linear(dst_channels);
  attr.bias.data.resize(attr.bias.shape.DimensionsProduct());

  OperationDef op_def;
  auto data_type = DeduceDataTypeFromPrecision(precision);
  Layout layout = src_shape.b == 1 ? Layout::kHWC : Layout::kBHWC;
  auto storage_type = GetFastestStorageType(env.device().GetInfo());
  TensorDescriptor src_tensor_desc{data_type, storage_type, layout};
  ABSL_RETURN_IF_ERROR(src_tensor_desc.UpdateToSupportedStorageType(
      env.device().GetInfo(), src_shape));

  auto dst_shape = src_shape;
  dst_shape.c = dst_channels;
  TensorDescriptor dst_tensor_desc{data_type, storage_type, layout};
  ABSL_RETURN_IF_ERROR(dst_tensor_desc.UpdateToSupportedStorageType(
      env.device().GetInfo(), dst_shape));

  op_def.src_tensors.push_back(src_tensor_desc);
  op_def.dst_tensors.push_back(dst_tensor_desc);

  const GpuInfo& gpu_info = env.GetDevicePtr()->GetInfo();

  ml_drift::Tensor<OHWI, DataType::kFloat32> wino_weights;
  RearrangeWeightsToWinograd3x3TileNxN(attr_weights, &wino_weights, 6);

  Convolution2DAttributes wino_attr;
  wino_attr.padding.prepended = HW(0, 0);
  wino_attr.padding.appended = HW(0, 0);
  wino_attr.strides = HW(1, 1);
  wino_attr.dilations = HW(1, 1);
  auto& wino_attr_weights =
      wino_attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::kFloat32>>();
  wino_attr_weights.shape = wino_weights.shape;
  std::unique_ptr<GPUOperation> conv;
  WeightsDescription weights_desc;
  if (IsConvWaveMemorySupported(gpu_info) &&
      env.device().GetInfo().IsAdreno()) {
    auto conv_wave_memory = CreateConvWaveMemoryExternalWeights(
        gpu_info, op_def, precision, wino_attr,
        /*bias=*/nullptr,
        /*dst_shape=*/nullptr,
        /*src_exp=*/nullptr, /*different_weights_for_height=*/true);
    weights_desc = conv_wave_memory.GetWeightsDescription();
    conv = std::make_unique<ConvWaveMemory>(std::move(conv_wave_memory));
  } else {
    auto conv_generic = CreateConvGenericExternalWeights(
        gpu_info, op_def, precision, wino_attr,
        /*bias=*/nullptr,
        /*dst_shape=*/nullptr,
        /*src_exp=*/nullptr, /*different_weights_for_height=*/true);
    weights_desc = conv_generic.GetWeightsDescription();
    conv = std::make_unique<ConvGeneric>(std::move(conv_generic));
  }

  std::vector<TensorDescriptor> weights_gpu_descs =
      GetTensorDescriptorsForWeightsLayout(wino_weights, weights_desc);

  Tensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(src_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  descriptor_with_shape.SetBHWCShape(dst_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), descriptor_with_shape, &dst));
  std::vector<Tensor> weights(weights_gpu_descs.size());
  for (int i = 0; i < weights_gpu_descs.size(); ++i) {
    ABSL_RETURN_IF_ERROR(
        CreateTensor(env.context(), weights_gpu_descs[i], &weights[i]));
  }

  std::cout << "Src size(BHWC) - " << src_shape.b << "x" << src_shape.h << "x"
            << src_shape.w << "x" << src_shape.c << std::endl;
  std::cout << "Dst size(BHWC) - " << dst_shape.b << "x" << dst_shape.h << "x"
            << dst_shape.w << "x" << dst_shape.c << std::endl;

  ABSL_RETURN_IF_ERROR(conv->AssembleCode(gpu_info));

  const int float_size = precision == CalculationsPrecision::kF32 ? 4 : 2;
  const int64_t flops_count =
      2.0 * dst_shape.DimensionsProduct() * attr_weights.shape.i;
  const double gflops_count = flops_count * 1e-9;

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const double dst_gbytes = dst.GetMemorySizeInBytes() / kGByte;
  const double src_gbytes = src.GetMemorySizeInBytes() / kGByte;
  const double weight_gbytes =
      36 * attr_weights.shape.o * attr_weights.shape.i * float_size / kGByte;
  const double bias_gbytes = attr_weights.shape.o * float_size / kGByte;

  ClOperation cl_op;
  cl_op.Init(std::move(conv));
  ABSL_RETURN_IF_ERROR(
      cl_op.Compile(env.GetDevicePtr(), &env.context(), env.program_cache()));
  ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(0, &src));
  for (int i = 0; i < weights.size(); ++i) {
    ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(i + 1, &weights[i]));
  }
  ABSL_RETURN_IF_ERROR(cl_op.SetDstTensor(0, &dst));
  ABSL_RETURN_IF_ERROR(cl_op.UpdateParams());
  ABSL_RETURN_IF_ERROR(cl_op.Tune(
      TuningType::kExhaustive, env.device().GetInfo(), env.profiling_queue()));

  for (int i = 0; i < 5; ++i) {
    ABSL_ASSIGN_OR_RETURN(auto duration,
                          cl_op.GetOpTime(env.device().GetInfo(), env.queue(),
                                          env.profiling_queue()));
    const auto time_ms = absl::ToDoubleMilliseconds(duration);
    const double fps = 1000.0 / time_ms;
    const double gflops_real = fps * gflops_count;
    const double gbyte_read = fps * (src_gbytes + weight_gbytes + bias_gbytes);
    const double gbyte_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(3) << "  Time - " << time_ms
              << std::setprecision(2) << "(ms), GFlops - " << gflops_real
              << ", Bandwidth - " << gbyte_read + gbyte_write
              << "(GB/s), Read - " << gbyte_read << "(GB/s), Write - "
              << gbyte_write << "(GB/s)" << std::endl;
  }

  return absl::OkStatus();
}

TensorDescriptor GetTensorDescriptor(PackedType quantized_type,
                                     TensorStorageType storage_type,
                                     Layout layout, const BHWC& src_shape,
                                     const GpuInfo& gpu_info) {
  TensorDescriptor src_tensor_desc{ToSpatialTensorType(quantized_type),
                                   storage_type, layout};
  BHWC src_packed_shape = GetShapeForPackedType(src_shape, quantized_type);
  if (!src_tensor_desc.UpdateToSupportedStorageType(gpu_info, src_packed_shape)
           .ok()) {
    std::cout << "Failed to find supported storage type." << std::endl;
  }
  return src_tensor_desc;
}

absl::Status ConvolutionInt8PerfTest(const BHWC& src_shape, int dst_channels) {
  Environment env;
  ABSL_RETURN_IF_ERROR(CreateEnvironment(&env));

  BHWC dst_shape;
  {
    Convolution2DAttributes attr;
    attr.padding.prepended = HW(0, 0);
    attr.padding.appended = HW(0, 0);
    attr.strides = HW(1, 1);
    attr.dilations = HW(1, 1);
    auto& weights =
        attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::kFloat32>>();
    weights.shape = OHWI(dst_channels, 1, 1, src_shape.c);
    attr.bias.shape = Linear(dst_channels);
    dst_shape = CalculateOutputShape(src_shape, attr);
  }

  const bool dequantize = true;
  const auto precision = CalculationsPrecision::kF16;
  OperationDef op_def;
  auto dst_data_type = UseUint8MathForInt8Weights(env.device().GetInfo())
                           ? DataType::kUint32
                           : DataType::kInt32;
  Layout layout = src_shape.b == 1 ? Layout::kHWC : Layout::kBHWC;
  auto src_storage_type = GetFastestStorageType(env.device().GetInfo());
  auto dst_storage_type = src_storage_type;

  TensorDescriptor dst_tensor_desc{dst_data_type, dst_storage_type, layout};
  ABSL_RETURN_IF_ERROR(dst_tensor_desc.UpdateToSupportedStorageType(
      env.device().GetInfo(), dst_shape));

  op_def.dst_tensors.push_back(dst_tensor_desc);

  ml_drift::Tensor<OHWI, DataType::kInt8> weights;
  weights.shape = OHWI(dst_channels, 1, 1, src_shape.c);
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(int8_t));
  ml_drift::Tensor<Linear, DataType::kFloat32> weights_scale;
  weights_scale.shape = Linear(dst_channels);
  weights_scale.data.resize(weights_scale.shape.DimensionsProduct());
  ml_drift::Tensor<Linear, DataType::kFloat32> weights_zero_point;
  weights_zero_point.shape = Linear(dst_channels);
  weights_zero_point.data.resize(weights_zero_point.shape.DimensionsProduct());

  const GpuInfo& gpu_info = env.GetDevicePtr()->GetInfo();

  PackedType quantized_type = PackedType::kUnknown;
  std::unique_ptr<GPUOperation> conv;
  if (SupportsConvWaveMatrixMaliInt8(gpu_info, src_shape)) {
    std::cout << "ConvWaveMatrixMali" << std::endl;
    quantized_type = PackedType::kInt8W4C4;
    op_def.src_tensors.push_back(GetTensorDescriptor(
        quantized_type, src_storage_type, layout, src_shape, gpu_info));
    conv = std::make_unique<ConvWaveMatrixMali>(
        ConvWaveMatrixMali(gpu_info, op_def, weights));
  } else if (SupportsConvWaveMemoryInt8(gpu_info)) {
    std::cout << "ConvWaveMemory" << std::endl;
    quantized_type = GetConvWaveMemoryInt8SrcType(gpu_info, src_shape);
    op_def.src_tensors.push_back(GetTensorDescriptor(
        quantized_type, src_storage_type, layout, src_shape, gpu_info));
    conv = std::make_unique<ConvWaveMemory>(
        CreateConvWaveMemoryInt8(gpu_info, op_def, weights, &dst_shape));
  } else if (SupportsConvGenericInt8(gpu_info)) {
    std::cout << "ConvGeneric" << std::endl;
    quantized_type = GetConvGenericInt8SrcType(gpu_info, src_shape);
    op_def.src_tensors.push_back(GetTensorDescriptor(
        quantized_type, src_storage_type, layout, src_shape, gpu_info));
    conv = std::make_unique<ConvGeneric>(CreateConvGenericInt8(
        gpu_info, op_def, quantized_type, weights, &dst_shape));
  } else {
    return absl::InvalidArgumentError("no supported int8 conv");
  }

  TensorDescriptor src_params_td;
  TensorDescriptor weights_scale_td;
  TensorDescriptor weights_zero_point_td;
  TensorDescriptor weights_sum_i_td;
  if (dequantize) {
    ml_drift::TensorFloat32 src_params;
    src_params.shape = BHWC(src_shape.b, src_shape.h, src_shape.w, 3);
    src_params.data.resize(src_params.shape.DimensionsProduct());
    src_params_td =
        TensorDescriptor{DataType::kFloat32, src_storage_type, layout};
    src_params_td.UploadData(src_params);

    weights_scale_td = CreateConstantLinearTensorDescriptor(
        gpu_info, DeduceDataTypeFromPrecision(precision), weights_scale);
    weights_zero_point_td = CreateConstantLinearTensorDescriptor(
        gpu_info, DeduceDataTypeFromPrecision(precision), weights_zero_point);

    auto weights_sum_i = GetWeightsAccumulatedInputChannels(weights);
    weights_sum_i_td =
        CreateConstantLinearTensorDescriptor(gpu_info, weights_sum_i);

    TensorDescriptor dequant_dst = {DeduceDataTypeFromPrecision(precision),
                                    op_def.dst_tensors[0].GetStorageType(),
                                    op_def.dst_tensors[0].GetLayout()};
    auto dequant_op =
        CreateDequantization(weights.shape, gpu_info, op_def.dst_tensors[0],
                             dequant_dst, src_params_td, weights_sum_i_td,
                             weights_scale_td, &weights_zero_point_td);
    ABSL_RETURN_IF_ERROR(conv->AddOperation(gpu_info, &dequant_op));
  }

  Tensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(
      GetShapeForPackedType(src_shape, quantized_type));
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), descriptor_with_shape, &src));
  descriptor_with_shape = dst_tensor_desc;
  descriptor_with_shape.SetBHWCShape(dst_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), descriptor_with_shape, &dst));

  Tensor src_params_tensor;
  Tensor weights_scale_tensor;
  Tensor weights_zero_point_tensor;
  Tensor weights_sum_i_tensor;
  if (dequantize) {
    ABSL_RETURN_IF_ERROR(
        CreateTensor(env.context(), src_params_td, &src_params_tensor));
    ABSL_RETURN_IF_ERROR(
        CreateTensor(env.context(), weights_scale_td, &weights_scale_tensor));
    ABSL_RETURN_IF_ERROR(CreateTensor(env.context(), weights_zero_point_td,
                                      &weights_zero_point_tensor));
    ABSL_RETURN_IF_ERROR(
        CreateTensor(env.context(), weights_sum_i_td, &weights_sum_i_tensor));
  }

  const auto w_shape = weights.shape;

  std::cout << "Src size(HWC) - " << src_shape.h << "x" << src_shape.w << "x"
            << src_shape.c << std::endl;
  std::cout << "Dst size(HWC) - " << dst_shape.h << "x" << dst_shape.w << "x"
            << dst_shape.c << std::endl;

  ABSL_RETURN_IF_ERROR(conv->AssembleCode(gpu_info));

  const int dst_element_size = SizeOf(dst_tensor_desc.GetDataType());
  const int64_t flops_per_element = w_shape.i * w_shape.h * w_shape.w * 2;
  const int64_t dst_elements = dst.Width() * dst.Height() * dst.Channels();
  const int64_t flops_count = dst_elements * flops_per_element;
  const double gflops_count = flops_count * 1e-9;

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const double src_gbytes = src.GetMemorySizeInBytes() / kGByte;
  const double dst_gbytes = dst.GetMemorySizeInBytes() / kGByte;
  const double weight_gbytes =
      weights.shape.DimensionsProduct() * sizeof(int8_t) / kGByte;
  const double bias_gbytes = weights.shape.o * dst_element_size / kGByte;
  const double scale_gbytes =
      weights_scale.shape.DimensionsProduct() * dst_element_size / kGByte;

  ClOperation cl_op;
  cl_op.Init(std::move(conv));
  ABSL_RETURN_IF_ERROR(
      cl_op.Compile(env.GetDevicePtr(), &env.context(), env.program_cache()));
  ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(0, &src));
  if (dequantize) {
    ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(1, &src_params_tensor));
    ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(2, &weights_sum_i_tensor));
    ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(3, &weights_scale_tensor));
    ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(4, &weights_zero_point_tensor));
  }
  ABSL_RETURN_IF_ERROR(cl_op.SetDstTensor(0, &dst));
  ABSL_RETURN_IF_ERROR(cl_op.UpdateParams());
  ABSL_RETURN_IF_ERROR(cl_op.Tune(
      TuningType::kExhaustive, env.device().GetInfo(), env.profiling_queue()));

  for (int i = 0; i < 5; ++i) {
    ABSL_ASSIGN_OR_RETURN(auto duration,
                          cl_op.GetOpTime(env.device().GetInfo(), env.queue(),
                                          env.profiling_queue()));
    const auto time_ms = absl::ToDoubleMilliseconds(duration);
    const double fps = 1000.0 / time_ms;
    const double gflops_real = fps * gflops_count;
    const double gbyte_read =
        fps * (src_gbytes + weight_gbytes + bias_gbytes + scale_gbytes * 2.0);
    const double gbyte_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(3) << "  Time - " << time_ms
              << std::setprecision(2) << "(ms), GFlops - " << gflops_real
              << ", Bandwidth - " << gbyte_read + gbyte_write
              << "(GB/s), Read - " << gbyte_read << "(GB/s), Write - "
              << gbyte_write << "(GB/s)" << std::endl;
  }

  return absl::OkStatus();
}

absl::Status ConvolutionInt8GroupedPerfTest(const BHWC& src_shape,
                                            int dst_channels, int group_size) {
  Environment env;
  ABSL_RETURN_IF_ERROR(CreateEnvironment(&env));

  BHWC dst_shape;
  {
    Convolution2DAttributes attr;
    attr.padding.prepended = HW(0, 0);
    attr.padding.appended = HW(0, 0);
    attr.strides = HW(1, 1);
    attr.dilations = HW(1, 1);
    auto& weights =
        attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::kFloat32>>();
    weights.shape = OHWI(dst_channels, 1, 1, src_shape.c);
    attr.bias.shape = Linear(dst_channels);
    dst_shape = CalculateOutputShape(src_shape, attr);
  }

  const auto precision = CalculationsPrecision::kF16;
  OperationDef op_def;
  auto quantized_type =
      GetConvolutionInt8SrcType(env.device().GetInfo(), src_shape);
  auto dst_data_type = UseUint8MathForInt8Weights(env.device().GetInfo())
                           ? DataType::kUint32
                           : DataType::kInt32;
  Layout layout = src_shape.b == 1 ? Layout::kHWC : Layout::kBHWC;
  auto storage_type = GetFastestStorageType(env.device().GetInfo());
  TensorDescriptor src_tensor_desc{ToSpatialTensorType(quantized_type),
                                   storage_type, layout};
  BHWC src_packed_shape = GetShapeForPackedType(src_shape, quantized_type);
  ABSL_RETURN_IF_ERROR(src_tensor_desc.UpdateToSupportedStorageType(
      env.device().GetInfo(), src_packed_shape));

  TensorDescriptor dst_tensor_desc{dst_data_type, storage_type, layout};
  ABSL_RETURN_IF_ERROR(dst_tensor_desc.UpdateToSupportedStorageType(
      env.device().GetInfo(), dst_shape));

  op_def.src_tensors.push_back(src_tensor_desc);
  op_def.dst_tensors.push_back(dst_tensor_desc);

  ml_drift::Tensor<OHWI, DataType::kInt8> weights;
  weights.shape = OHWI(dst_channels, 1, 1, src_shape.c);
  weights.data.resize(weights.shape.DimensionsProduct());
  ml_drift::Tensor<Linear, DataType::kFloat32> weights_scale;
  weights_scale.shape = Linear(dst_channels);
  weights_scale.data.resize(weights_scale.shape.DimensionsProduct());
  ml_drift::Tensor<Linear, DataType::kFloat32> weights_zero_point;
  weights_zero_point.shape = Linear(dst_channels);
  weights_zero_point.data.resize(weights_zero_point.shape.DimensionsProduct());

  const int quant_i_groups = weights.shape.i / group_size;
  ml_drift::TensorFloat32 src_params;
  src_params.shape =
      BHWC(src_shape.b, src_shape.h, src_shape.w, quant_i_groups * 4);
  src_params.data.resize(src_params.shape.DimensionsProduct());
  TensorDescriptor src_params_td{DataType::kFloat32, storage_type, layout};
  src_params_td.UploadData(src_params);

  {
    const GpuInfo& gpu_info = env.device().GetInfo();

    Tensor src_params_tensor;
    ABSL_RETURN_IF_ERROR(
        CreateTensor(env.context(), src_params_td, &src_params_tensor));

    std::unique_ptr<GPUOperation> conv;
    if (SupportsConvWaveMemoryInt8(gpu_info)) {
      std::cout << "ConvWaveMemory" << std::endl;
      conv = std::make_unique<ConvWaveMemory>(CreateConvWaveMemoryInt8Grouped(
          gpu_info, op_def, weights, group_size,
          src_params_tensor.GetDescriptor(), &dst_shape));
    } else {
      return absl::InvalidArgumentError("no supported int8 conv");
    }

    TensorDescriptor weights_scale_td = CreateConstantLinearTensorDescriptor(
        gpu_info, DeduceDataTypeFromPrecision(precision), weights_scale);
    TensorDescriptor weights_zero_point_td =
        CreateConstantLinearTensorDescriptor(
            gpu_info, DeduceDataTypeFromPrecision(precision),
            weights_zero_point);

    auto weights_sum_i = GetWeightsAccumulatedInputChannels(weights);
    auto weights_sum_i_td =
        CreateConstantLinearTensorDescriptor(gpu_info, weights_sum_i);

    TensorDescriptor dequant_dst = {DeduceDataTypeFromPrecision(precision),
                                    op_def.dst_tensors[0].GetStorageType(),
                                    op_def.dst_tensors[0].GetLayout()};
    auto dequant_op =
        CreateDequantization(weights.shape, gpu_info, op_def.dst_tensors[0],
                             dequant_dst, src_params_td, weights_sum_i_td,
                             weights_scale_td, &weights_zero_point_td);
    ABSL_RETURN_IF_ERROR(conv->AddOperation(gpu_info, &dequant_op));

    Tensor src, dst;
    TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
    descriptor_with_shape.SetBHWCShape(src_packed_shape);
    ABSL_RETURN_IF_ERROR(
        CreateTensor(env.context(), descriptor_with_shape, &src));
    descriptor_with_shape = dequant_dst;
    descriptor_with_shape.SetBHWCShape(dst_shape);
    ABSL_RETURN_IF_ERROR(
        CreateTensor(env.context(), descriptor_with_shape, &dst));

    Tensor weights_scale_tensor;
    ABSL_RETURN_IF_ERROR(
        CreateTensor(env.context(), weights_scale_td, &weights_scale_tensor));
    Tensor weights_zero_point_tensor;
    ABSL_RETURN_IF_ERROR(CreateTensor(env.context(), weights_zero_point_td,
                                      &weights_zero_point_tensor));
    Tensor weights_sum_i_tensor;
    ABSL_RETURN_IF_ERROR(
        CreateTensor(env.context(), weights_sum_i_td, &weights_sum_i_tensor));

    const auto w_shape = weights.shape;

    std::cout << "Src size(HWC) - " << src_shape.h << "x" << src_shape.w << "x"
              << src_shape.c << std::endl;
    std::cout << "Dst size(HWC) - " << dst_shape.h << "x" << dst_shape.w << "x"
              << dst_shape.c << std::endl;

    ABSL_RETURN_IF_ERROR(conv->AssembleCode(gpu_info));

    const int dst_element_size = SizeOf(dequant_dst.GetDataType());
    const int64_t flops_per_element = w_shape.i * w_shape.h * w_shape.w * 2;
    const int64_t dst_elements = dst.Width() * dst.Height() * dst.Channels();
    const int64_t flops_count = dst_elements * flops_per_element;
    const double gflops_count = flops_count * 1e-9;

    const double kGByte = 1024.0 * 1024.0 * 1024.0;
    const double src_gbytes = src.GetMemorySizeInBytes() / kGByte;
    const double dst_gbytes = dst.GetMemorySizeInBytes() / kGByte;
    const double weight_gbytes =
        weights.shape.DimensionsProduct() * sizeof(int8_t) / kGByte;
    const double bias_gbytes = weights.shape.o * dst_element_size / kGByte;
    const double scale_gbytes =
        weights_scale.shape.DimensionsProduct() * dst_element_size / kGByte;

    ClOperation cl_op;
    cl_op.Init(std::move(conv));
    ABSL_RETURN_IF_ERROR(
        cl_op.Compile(env.GetDevicePtr(), &env.context(), env.program_cache()));
    ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(0, &src));
    ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(1, &src_params_tensor));
    ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(2, &src_params_tensor));
    ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(3, &weights_sum_i_tensor));
    ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(4, &weights_scale_tensor));
    ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(5, &weights_zero_point_tensor));
    ABSL_RETURN_IF_ERROR(cl_op.SetDstTensor(0, &dst));
    ABSL_RETURN_IF_ERROR(cl_op.UpdateParams());
    ABSL_RETURN_IF_ERROR(cl_op.Tune(TuningType::kExhaustive,
                                    env.device().GetInfo(),
                                    env.profiling_queue()));

    for (int i = 0; i < 5; ++i) {
      ABSL_ASSIGN_OR_RETURN(auto duration,
                            cl_op.GetOpTime(env.device().GetInfo(), env.queue(),
                                            env.profiling_queue()));
      const auto time_ms = absl::ToDoubleMilliseconds(duration);
      const double fps = 1000.0 / time_ms;
      const double gflops_real = fps * gflops_count;
      const double gbyte_read =
          fps * (src_gbytes + weight_gbytes + bias_gbytes + scale_gbytes * 2.0);
      const double gbyte_write = fps * dst_gbytes;
      std::cout << std::fixed << std::setprecision(3) << "  Time - " << time_ms
                << std::setprecision(2) << "(ms), GFlops - " << gflops_real
                << ", Bandwidth - " << gbyte_read + gbyte_write
                << "(GB/s), Read - " << gbyte_read << "(GB/s), Write - "
                << gbyte_write << "(GB/s)" << std::endl;
    }
  }

  return absl::OkStatus();
}

absl::Status ConvolutionSf16Wi4BatchedPerfTest(const BHWC& src_shape,
                                               int dst_channels,
                                               OHWI scale_zp_shape) {
  Environment env;
  ABSL_RETURN_IF_ERROR(CreateEnvironment(&env));
  const GpuInfo& gpu_info = env.device().GetInfo();
  const bool use_zero_point = true;

  const DataType float_type = DataType::kFloat16;

  BHWC dst_shape = src_shape;
  dst_shape.c = dst_channels;

  const int src_channels = src_shape.c;

  ml_drift::Tensor<OHWI, DataType::kInt8> weights_i4;
  weights_i4.shape = OHWI(dst_channels, scale_zp_shape.h, 1, src_channels);
  weights_i4.data.resize(weights_i4.shape.DimensionsProduct());
  for (int i = 0; i < weights_i4.data.size(); ++i) {
    weights_i4.data[i] = (i % 15) - 7;
  }

  ml_drift::Tensor<OHWI, DataType::kFloat32> weights_scales;
  weights_scales.shape = scale_zp_shape;
  weights_scales.data.resize(weights_scales.shape.DimensionsProduct());
  for (int i = 0; i < weights_scales.data.size(); ++i) {
    weights_scales.data[i] = 1.0f / 8.0f;
  }

  ml_drift::Tensor<OHWI, DataType::kFloat32> weights_zero_point;
  weights_zero_point.shape = weights_scales.shape;
  weights_zero_point.data.resize(weights_scales.shape.DimensionsProduct(),
                                 0.0f);

  Layout layout = src_shape.b == 1 ? Layout::kHWC : Layout::kBHWC;
  auto storage_type = GetFastestStorageType(gpu_info);
  TensorDescriptor src_tensor_desc{float_type, storage_type, layout};
  TensorDescriptor dst_tensor_desc{float_type, storage_type, layout};

  WeightsDescription weights_desc;
  weights_desc.type = DataType::kUint4;
  weights_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
  weights_desc.output_group_size = DivideRoundUp(weights_i4.shape.o, 4);

  auto scale_desc =
      ScaleOrZeroPointToTensorDesc(gpu_info, weights_scales, float_type);
  auto zp_desc =
      ScaleOrZeroPointToTensorDesc(gpu_info, weights_zero_point, float_type);

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_i4.shape;
  external_weights.scale_zp_shape = weights_scales.shape;
  external_weights.scale = &scale_desc;
  if (use_zero_point) {
    external_weights.zero_point = &zp_desc;
  }

  std::unique_ptr<GPUOperation> conv;
  if (SupportsConvGeneric(gpu_info, CalculationsPrecision::kF16,
                          external_weights)) {
    OperationDef conv_def;
    conv_def.src_tensors.push_back(src_tensor_desc);
    conv_def.dst_tensors.push_back(dst_tensor_desc);
    auto conv_generic = CreateConvGenericExternalWeights(
        gpu_info, conv_def, CalculationsPrecision::kF16, external_weights,
        /*bias=*/nullptr, &dst_shape,
        /*src_exp=*/nullptr,
        /*different_weights_for_height=*/true);
    conv = std::make_unique<ConvGeneric>(std::move(conv_generic));
  } else {
    return absl::UnimplementedError("no supported conv");
  }

  TensorDescriptor weights_i4_td =
      GetTensorDescriptorForWeightsLayout(weights_i4, weights_desc);

  Tensor src, dst;
  TensorDescriptor descriptor_with_shape = src_tensor_desc;
  descriptor_with_shape.SetBHWCShape(src_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), descriptor_with_shape, &src));

  descriptor_with_shape = dst_tensor_desc;
  descriptor_with_shape.SetBHWCShape(dst_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), descriptor_with_shape, &dst));

  Tensor weights_i4_tensor;
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), weights_i4_td, &weights_i4_tensor));

  Tensor weights_scale_tensor;
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), scale_desc, &weights_scale_tensor));

  Tensor weights_zero_point_tensor;
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), zp_desc, &weights_zero_point_tensor));

  conv->SetSrc(&src);
  conv->SetSrc(&weights_i4_tensor);
  conv->SetSrc(&weights_scale_tensor);
  if (use_zero_point) {
    conv->SetSrc(&weights_zero_point_tensor);
  }
  conv->SetDst(&dst);

  ABSL_RETURN_IF_ERROR(conv->AssembleCode(gpu_info));

  const int64_t flops_per_element = weights_i4.shape.i * 2;
  const int64_t dst_elements = dst.Width() * dst.Height() * dst.Channels();
  const int64_t flops_count = dst_elements * flops_per_element;
  const double gflops_count = flops_count * 1e-9;

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const double src_gbytes = src.GetMemorySizeInBytes() / kGByte;
  const double dst_gbytes = dst.GetMemorySizeInBytes() / kGByte;
  const double weight_gbytes =
      (weights_i4.shape.DimensionsProduct() * 0.5) / kGByte;

  ClOperation cl_op;
  cl_op.Init(std::move(conv));
  ABSL_RETURN_IF_ERROR(
      cl_op.Compile(env.GetDevicePtr(), &env.context(), env.program_cache()));
  ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(0, &src));
  ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(1, &weights_i4_tensor));
  ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(2, &weights_scale_tensor));
  if (use_zero_point) {
    ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(3, &weights_zero_point_tensor));
  }
  ABSL_RETURN_IF_ERROR(cl_op.SetDstTensor(0, &dst));
  ABSL_RETURN_IF_ERROR(cl_op.UpdateParams());
  ABSL_RETURN_IF_ERROR(cl_op.Tune(
      TuningType::kExhaustive, env.device().GetInfo(), env.profiling_queue()));

  for (int i = 0; i < 5; ++i) {
    ABSL_ASSIGN_OR_RETURN(auto duration,
                          cl_op.GetOpTime(env.device().GetInfo(), env.queue(),
                                          env.profiling_queue()));
    const auto time_ms = absl::ToDoubleMilliseconds(duration);
    const double fps = 1000.0 / time_ms;
    const double gflops_real = fps * gflops_count;
    const double gbyte_read = fps * (src_gbytes + weight_gbytes);
    const double gbyte_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(3) << "  Time - " << time_ms
              << std::setprecision(2) << "(ms), GFlops - " << gflops_real
              << ", Bandwidth - " << gbyte_read + gbyte_write
              << "(GB/s), Read - " << gbyte_read << "(GB/s), Write - "
              << gbyte_write << "(GB/s)" << std::endl;
  }

  return absl::OkStatus();
}

absl::Status ConvolutionInt4PerfTest(const BHWC& src_shape, int dst_channels) {
  Environment env;
  ABSL_RETURN_IF_ERROR(CreateEnvironment(&env));
  const GpuInfo& gpu_info = env.device().GetInfo();

  BHWC dst_shape;
  {
    Convolution2DAttributes attr;
    attr.padding.prepended = HW(0, 0);
    attr.padding.appended = HW(0, 0);
    attr.strides = HW(1, 1);
    attr.dilations = HW(1, 1);
    auto& weights =
        attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::kFloat32>>();
    weights.shape = OHWI(dst_channels, 1, 1, src_shape.c);
    attr.bias.shape = Linear(dst_channels);
    dst_shape = CalculateOutputShape(src_shape, attr);
  }

  const auto precision = CalculationsPrecision::kF16;
  OperationDef op_def;
  auto quantized_type =
      GetConvGenericInt4SrcType(env.device().GetInfo(), src_shape);
  Layout layout = src_shape.b == 1 ? Layout::kHWC : Layout::kBHWC;
  auto storage_type = GetFastestStorageType(env.device().GetInfo());
  TensorDescriptor src_tensor_desc{ToSpatialTensorType(quantized_type),
                                   storage_type, layout};
  BHWC src_packed_shape = GetShapeForPackedType(src_shape, quantized_type);
  ABSL_RETURN_IF_ERROR(src_tensor_desc.UpdateToSupportedStorageType(
      env.device().GetInfo(), src_packed_shape));

  TensorDescriptor dst_tensor_desc{DataType::kInt32, storage_type, layout};
  ABSL_RETURN_IF_ERROR(dst_tensor_desc.UpdateToSupportedStorageType(
      env.device().GetInfo(), dst_shape));

  op_def.src_tensors.push_back(src_tensor_desc);
  op_def.dst_tensors.push_back(dst_tensor_desc);

  ml_drift::Tensor<OHWI, DataType::kInt8> weights;
  weights.shape = OHWI(dst_channels, 1, 1, src_shape.c);
  weights.data.resize(weights.shape.DimensionsProduct());
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(int8_t));
  ml_drift::Tensor<Linear, DataType::kFloat32> weights_scale;
  weights_scale.shape = Linear(dst_channels);
  weights_scale.data.resize(weights_scale.shape.DimensionsProduct());
  ml_drift::Tensor<Linear, DataType::kFloat32> weights_zero_point;
  weights_zero_point.shape = Linear(dst_channels);
  weights_zero_point.data.resize(weights_zero_point.shape.DimensionsProduct());

  std::unique_ptr<GPUOperation> conv;
  if (SupportsConvGenericInt4(gpu_info, src_shape)) {
    std::cout << "ConvGeneric" << std::endl;
    conv = std::make_unique<ConvGeneric>(
        CreateConvGenericInt4(gpu_info, op_def, weights, &dst_shape));
  } else {
    return absl::InvalidArgumentError("no supported int4 conv");
  }

  ml_drift::TensorFloat32 src_params;
  src_params.shape = BHWC(src_shape.b, src_shape.h, src_shape.w, 3);
  src_params.data.resize(src_params.shape.DimensionsProduct());
  TensorDescriptor src_params_td{DataType::kFloat32, storage_type, layout};
  src_params_td.UploadData(src_params);

  TensorDescriptor weights_scale_td = CreateConstantLinearTensorDescriptor(
      gpu_info, DeduceDataTypeFromPrecision(precision), weights_scale);
  TensorDescriptor weights_zero_point_td = CreateConstantLinearTensorDescriptor(
      gpu_info, DeduceDataTypeFromPrecision(precision), weights_zero_point);

  auto weights_sum_i = GetWeightsAccumulatedInputChannels(weights);
  auto weights_sum_i_td =
      CreateConstantLinearTensorDescriptor(gpu_info, weights_sum_i);

  TensorDescriptor dequant_dst = {DeduceDataTypeFromPrecision(precision),
                                  op_def.dst_tensors[0].GetStorageType(),
                                  op_def.dst_tensors[0].GetLayout()};
  auto dequant_op =
      CreateDequantization(weights.shape, gpu_info, op_def.dst_tensors[0],
                           dequant_dst, src_params_td, weights_sum_i_td,
                           weights_scale_td, &weights_zero_point_td);
  ABSL_RETURN_IF_ERROR(conv->AddOperation(gpu_info, &dequant_op));

  Tensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(src_packed_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), descriptor_with_shape, &src));
  descriptor_with_shape = dequant_dst;
  descriptor_with_shape.SetBHWCShape(dst_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), descriptor_with_shape, &dst));

  Tensor src_params_tensor;
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), src_params_td, &src_params_tensor));
  Tensor weights_scale_tensor;
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), weights_scale_td, &weights_scale_tensor));
  Tensor weights_zero_point_tensor;
  ABSL_RETURN_IF_ERROR(CreateTensor(env.context(), weights_zero_point_td,
                                    &weights_zero_point_tensor));
  Tensor weights_sum_i_tensor;
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), weights_sum_i_td, &weights_sum_i_tensor));

  const auto w_shape = weights.shape;

  std::cout << "Src size(HWC) - " << src_shape.h << "x" << src_shape.w << "x"
            << src_shape.c << std::endl;
  std::cout << "Dst size(HWC) - " << dst_shape.h << "x" << dst_shape.w << "x"
            << dst_shape.c << std::endl;

  ABSL_RETURN_IF_ERROR(conv->AssembleCode(gpu_info));

  const int dst_element_size = SizeOf(dequant_dst.GetDataType());
  const int64_t flops_per_element = w_shape.i * w_shape.h * w_shape.w * 2;
  const int64_t dst_elements = dst.Width() * dst.Height() * dst.Channels();
  const int64_t flops_count = dst_elements * flops_per_element;
  const double gflops_count = flops_count * 1e-9;

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const double src_gbytes = src.GetMemorySizeInBytes() / kGByte;
  const double dst_gbytes = dst.GetMemorySizeInBytes() / kGByte;
  const double weight_gbytes =
      weights.shape.DimensionsProduct() * sizeof(int8_t) / 2 / kGByte;
  const double bias_gbytes = weights.shape.o * dst_element_size / kGByte;
  const double scale_gbytes =
      weights_scale.shape.DimensionsProduct() * dst_element_size / kGByte;

  ClOperation cl_op;
  cl_op.Init(std::move(conv));
  ABSL_RETURN_IF_ERROR(
      cl_op.Compile(env.GetDevicePtr(), &env.context(), env.program_cache()));
  ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(0, &src));
  ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(1, &src_params_tensor));
  ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(2, &weights_sum_i_tensor));
  ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(3, &weights_scale_tensor));
  ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(4, &weights_zero_point_tensor));
  ABSL_RETURN_IF_ERROR(cl_op.SetDstTensor(0, &dst));
  ABSL_RETURN_IF_ERROR(cl_op.UpdateParams());
  ABSL_RETURN_IF_ERROR(cl_op.Tune(
      TuningType::kExhaustive, env.device().GetInfo(), env.profiling_queue()));

  for (int i = 0; i < 5; ++i) {
    ABSL_ASSIGN_OR_RETURN(auto duration,
                          cl_op.GetOpTime(env.device().GetInfo(), env.queue(),
                                          env.profiling_queue()));
    const auto time_ms = absl::ToDoubleMilliseconds(duration);
    const double fps = 1000.0 / time_ms;
    const double gflops_real = fps * gflops_count;
    const double gbyte_read =
        fps * (src_gbytes + weight_gbytes + bias_gbytes + scale_gbytes * 2.0);
    const double gbyte_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(3) << "  Time - " << time_ms
              << std::setprecision(2) << "(ms), GFlops - " << gflops_real
              << ", Bandwidth - " << gbyte_read + gbyte_write
              << "(GB/s), Read - " << gbyte_read << "(GB/s), Write - "
              << gbyte_write << "(GB/s)" << std::endl;
  }

  return absl::OkStatus();
}

absl::Status ConvMoEPerfTest(int seq_size, int src_channels, int dst_channels,
                             int num_experts, int num_active_experts,
                             DataType weights_type) {
  Environment env;
  ABSL_RETURN_IF_ERROR(CreateEnvironment(&env));
  const auto& gpu_info = env.device().GetInfo();

  const BHWC src_shape =
      BHWC(1, 1, seq_size * num_active_experts, src_channels);
  OperationDef op_def;
  auto data_type = DataType::kFloat16;
  Layout layout = Layout::kHWC;
  auto storage_type = GetFastestStorageType(env.device().GetInfo());
  TensorDescriptor src_tensor_desc{data_type, storage_type, layout};
  ABSL_RETURN_IF_ERROR(src_tensor_desc.UpdateToSupportedStorageType(
      env.device().GetInfo(), src_shape));

  auto dst_shape = src_shape;
  dst_shape.c = dst_channels;
  TensorDescriptor dst_tensor_desc{data_type, storage_type, layout};
  ABSL_RETURN_IF_ERROR(dst_tensor_desc.UpdateToSupportedStorageType(
      env.device().GetInfo(), dst_shape));

  op_def.src_tensors.push_back(src_tensor_desc);
  op_def.dst_tensors.push_back(dst_tensor_desc);

  ml_drift::Tensor<OHWI, DataType::kFloat32> weights_f32;
  weights_f32.shape = OHWI(dst_channels, num_experts, 1, src_shape.c);
  weights_f32.data.resize(weights_f32.shape.DimensionsProduct() +
                          XNN_EXTRA_BYTES / sizeof(float));

  ml_drift::Tensor<OHWI, DataType::kInt8> weights_i8;
  weights_i8.shape = OHWI(dst_channels, num_experts, 1, src_shape.c);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(uint8_t));

  ml_drift::Tensor<OHWI, DataType::kFloat32> weights_scale;
  weights_scale.shape = OHWI(dst_channels, num_experts, 1, 1);
  weights_scale.data.resize(weights_scale.shape.DimensionsProduct(), 1.0f);
  ml_drift::Tensor<OHWI, DataType::kFloat32> weights_zp;
  weights_zp.shape = OHWI(dst_channels, num_experts, 1, 1);
  weights_zp.data.resize(weights_zp.shape.DimensionsProduct(), 0.0f);

  WeightsDescription weights_desc;
  if (weights_type == DataType::kFloat16 ||
      weights_type == DataType::kFloat32) {
    weights_desc.type = weights_type;
    weights_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
    weights_desc.output_group_size = DivideRoundUp(weights_f32.shape.o, 4);
  } else if (weights_type == DataType::kInt8) {
    weights_desc = GetFullyConnectedInt8WeightsDesc(gpu_info, weights_i8.shape);
  } else if (weights_type == DataType::kInt4) {
    weights_desc = GetFullyConnectedInt4WeightsDesc(gpu_info, weights_i8.shape);
  } else if (weights_type == DataType::kInt2) {
    weights_desc = GetFullyConnectedInt2WeightsDesc(gpu_info, weights_i8.shape);
  }

  TensorDescriptor scale_desc =
      ScaleOrZeroPointToTensorDesc(gpu_info, weights_scale, data_type);
  TensorDescriptor zp_desc =
      ScaleOrZeroPointToTensorDesc(gpu_info, weights_zp, data_type);

  const bool is_quantized = SizeInBitsOf(weights_type) < 16;

  Tensor scale_tensor;
  Tensor zp_tensor;
  if (is_quantized) {
    ABSL_RETURN_IF_ERROR(
        CreateTensor(env.context(), scale_desc, &scale_tensor));
    ABSL_RETURN_IF_ERROR(CreateTensor(env.context(), zp_desc, &zp_tensor));
  }

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights_f32.shape;
  if (is_quantized) {
    external_weights.scale_zp_shape = weights_scale.shape;
    external_weights.scale = &scale_desc;
    external_weights.zero_point = &zp_desc;
  }

  Convolution2DAttributes conv_attr;
  conv_attr.padding.prepended = HW(0, 0);
  conv_attr.padding.appended = HW(0, 0);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  auto& conv_attr_weights =
      conv_attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::kFloat32>>();
  conv_attr_weights.shape = weights_f32.shape;

  std::unique_ptr<GPUOperation> conv;
  ConvRuntimeCheckDesc::PackedGroups packed_groups;
  packed_groups.params_offset = 0;
  packed_groups.num_groups = num_experts;
  packed_groups.max_group_size = seq_size;
  ConvRuntimeCheckDesc runtime_check;
  runtime_check.packed_groups = packed_groups;
  if (SupportsConvGeneric(gpu_info, CalculationsPrecision::kF16,
                          external_weights)) {
    auto conv_generic = CreateConvGenericExternalWeights(
        gpu_info, op_def, CalculationsPrecision::kF16, external_weights,
        /*bias=*/nullptr, &dst_shape,
        /*src_exp=*/nullptr, /*different_weights_for_height=*/true,
        runtime_check);
    conv = std::make_unique<ConvGeneric>(std::move(conv_generic));
  } else if (!is_quantized && IsConvWaveMemorySupported(gpu_info) &&
             gpu_info.IsAdreno()) {
    auto conv_wave_memory = CreateConvWaveMemoryExternalWeights(
        gpu_info, op_def, CalculationsPrecision::kF16, conv_attr,
        /*bias=*/nullptr, &dst_shape,
        /*src_exp=*/nullptr, /*different_weights_for_height=*/true,
        runtime_check);
    weights_desc = conv_wave_memory.GetWeightsDescription();
    conv = std::make_unique<ConvWaveMemory>(std::move(conv_wave_memory));
  } else {
    ABSL_ASSIGN_OR_RETURN(
        auto conv_fc,
        CreateFullyConnectedExternalWeights(
            gpu_info, CalculationsPrecision::kF16, op_def.src_tensors[0],
            op_def.dst_tensors[0], external_weights, /*bias=*/nullptr,
            &dst_shape, /*src_exp=*/nullptr, runtime_check));
    conv = std::make_unique<FullyConnected>(std::move(conv_fc));
  }

  Tensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(src_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  descriptor_with_shape.SetBHWCShape(dst_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), descriptor_with_shape, &dst));

  std::vector<TensorDescriptor> weights_gpu;
  if (!is_quantized) {
    weights_gpu =
        GetTensorDescriptorsForWeightsLayout(weights_f32, weights_desc);
  } else {
    weights_gpu.push_back(
        GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc));
  }

  std::vector<Tensor> weights_tensors(weights_gpu.size());
  for (int i = 0; i < weights_gpu.size(); ++i) {
    ABSL_RETURN_IF_ERROR(
        CreateTensor(env.context(), weights_gpu[i], &weights_tensors[i]));
  }

  TensorInt32 active_expert_ids =
      GenerateGroupIds(BHWC(1, 1, seq_size, num_active_experts), num_experts);

  auto [groups_map, groups_sizes] =
      GroupsMapReference(active_expert_ids, num_experts);
  auto [packed_groups_map, groups_offsets] =
      PackedGroupsMapReference(groups_map, groups_sizes);

  std::vector<int32_t> runtime_params_cpu(num_experts * 2, 0);
  for (int i = 0; i < num_experts; ++i) {
    runtime_params_cpu[i] = groups_sizes.data[i];
    runtime_params_cpu[num_experts + i] = groups_offsets.data[i];
  }
  Tensor runtime_params;
  TensorDescriptor runtime_params_td(
      DataType::kInt32, TensorStorageType::kBuffer, Layout::kLinear);
  runtime_params_td.SetBHWCShape(BHWC(1, 1, 1, num_experts * 2));
  runtime_params_td.UploadData(runtime_params_cpu.data());
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), runtime_params_td, &runtime_params));

  std::cout << "Src size(BHWC) - " << src_shape.b << "x" << src_shape.h << "x"
            << src_shape.w << "x" << src_shape.c << std::endl;
  std::cout << "Dst size(BHWC) - " << dst_shape.b << "x" << dst_shape.h << "x"
            << dst_shape.w << "x" << dst_shape.c << std::endl;

  ABSL_RETURN_IF_ERROR(conv->AssembleCode(gpu_info));

  const int64_t flops_count =
      2.0 * dst_shape.DimensionsProduct() * weights_f32.shape.i;
  const double gflops_count = flops_count * 1e-9;

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const double src_gbytes = src.GetMemorySizeInBytes() / kGByte;
  const double dst_gbytes = dst.GetMemorySizeInBytes() / kGByte;
  const double weight_element_size = SizeInBitsOf(weights_type) / 8.0;
  const double weight_gbytes =
      weights_f32.shape.DimensionsProduct() * weight_element_size / kGByte;
  const double scale_gbytes = scale_tensor.GetMemorySizeInBytes() / kGByte;

  ClOperation cl_op;
  cl_op.Init(std::move(conv));
  ABSL_RETURN_IF_ERROR(
      cl_op.Compile(env.GetDevicePtr(), &env.context(), env.program_cache()));
  int index = 0;
  ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(index++, &src));
  for (int i = 0; i < weights_tensors.size(); ++i) {
    ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(index++, &weights_tensors[i]));
  }
  if (is_quantized) {
    ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(index++, &scale_tensor));
    ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(index++, &zp_tensor));
  }
  ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(index++, &runtime_params));
  ABSL_RETURN_IF_ERROR(cl_op.SetDstTensor(0, &dst));
  ABSL_RETURN_IF_ERROR(cl_op.UpdateParams());
  ABSL_RETURN_IF_ERROR(cl_op.Tune(
      TuningType::kExhaustive, env.device().GetInfo(), env.profiling_queue()));

  for (int i = 0; i < 5; ++i) {
    ABSL_ASSIGN_OR_RETURN(auto duration,
                          cl_op.GetOpTime(env.device().GetInfo(), env.queue(),
                                          env.profiling_queue()));
    const auto time_ms = absl::ToDoubleMilliseconds(duration);
    const double fps = 1000.0 / time_ms;
    const double gflops_real = fps * gflops_count;
    double gbytes_read = src_gbytes + weight_gbytes;
    if (is_quantized) {
      gbytes_read += scale_gbytes * 2.0;
    }
    const double gbyte_read = fps * gbytes_read;
    const double gbyte_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(3) << "  Time - " << time_ms
              << std::setprecision(2) << "(ms), GFlops - " << gflops_real
              << ", Bandwidth - " << gbyte_read + gbyte_write
              << "(GB/s), Read - " << gbyte_read << "(GB/s), Write - "
              << gbyte_write << "(GB/s)" << std::endl;
  }

  return absl::OkStatus();
}

absl::Status SoftmaxPerfTest(const BHWC& shape, bool reduce_only) {
  Environment env;
  ABSL_RETURN_IF_ERROR(CreateEnvironment(&env));
  const auto precision = CalculationsPrecision::kF16;
  OperationDef op_def;
  auto data_type = DeduceDataTypeFromPrecision(precision);
  auto storage_type = GetFastestStorageType(env.device().GetInfo());
  Layout layout = shape.b == 1 ? Layout::kHWC : Layout::kBHWC;
  TensorDescriptor tensor_desc{data_type, storage_type, layout};
  ABSL_RETURN_IF_ERROR(
      tensor_desc.UpdateToSupportedStorageType(env.device().GetInfo(), shape));
  op_def.src_tensors.push_back(tensor_desc);
  op_def.dst_tensors.push_back(tensor_desc);

  const GpuInfo& gpu_info = env.GetDevicePtr()->GetInfo();

  std::unique_ptr<GPUOperation> softmax;
  if (reduce_only) {
    // softmax = std::make_unique<Softmax1x1>(CreateSoftmax1x1Reduce(
    //    op_def, gpu_info, shape));

    softmax =
        std::make_unique<Softmax>(CreateSoftmaxReduce(op_def, gpu_info, shape));
  } else {
    // softmax = std::make_unique<Softmax1x1>(
    //    CreateSoftmax1x1(op_def, gpu_info, shape));

    softmax = std::make_unique<Softmax>(CreateSoftmax(op_def, gpu_info, shape));
  }

  Tensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  BHWC dst_shape = shape;
  if (reduce_only) {
    dst_shape.c = 2;
  }
  descriptor_with_shape.SetBHWCShape(dst_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), descriptor_with_shape, &dst));

  std::cout << "Src shape - " << ToString(shape) << std::endl;
  std::cout << "Dst shape - " << ToString(dst_shape) << std::endl;

  ABSL_RETURN_IF_ERROR(softmax->AssembleCode(gpu_info));

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const double src_gbytes = src.GetMemorySizeInBytes() / kGByte;
  const double dst_gbytes = dst.GetMemorySizeInBytes() / kGByte;

  ClOperation cl_op;
  cl_op.Init(std::move(softmax));
  ABSL_RETURN_IF_ERROR(
      cl_op.Compile(env.GetDevicePtr(), &env.context(), env.program_cache()));
  ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(0, &src));
  ABSL_RETURN_IF_ERROR(cl_op.SetDstTensor(0, &dst));
  ABSL_RETURN_IF_ERROR(cl_op.UpdateParams());
  ABSL_RETURN_IF_ERROR(cl_op.Tune(
      TuningType::kExhaustive, env.device().GetInfo(), env.profiling_queue()));

  for (int i = 0; i < 5; ++i) {
    ABSL_ASSIGN_OR_RETURN(auto duration,
                          cl_op.GetOpTime(env.device().GetInfo(), env.queue(),
                                          env.profiling_queue()));
    const auto time_ms = absl::ToDoubleMilliseconds(duration);
    const double fps = 1000.0 / time_ms;
    const double gbyte_read = fps * (src_gbytes);
    const double gbyte_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(3) << "  Time - " << time_ms
              << std::setprecision(3) << "(ms), Bandwidth - "
              << gbyte_read + gbyte_write << "(GB/s), Read - " << gbyte_read
              << "(GB/s), Write - " << gbyte_write << "(GB/s)" << std::endl;
  }

  return absl::OkStatus();
}

absl::Status ConvSoftmaxConvPerfTest() {
  Environment env;
  ABSL_RETURN_IF_ERROR(CreateEnvironment(&env));

  const auto precision = CalculationsPrecision::kF16;
  OperationDef op_def;
  auto data_type = DeduceDataTypeFromPrecision(precision);
  op_def.src_tensors.push_back(
      {data_type, TensorStorageType::kTexture2D, Layout::kHWC});
  op_def.dst_tensors.push_back(
      {data_type, TensorStorageType::kTexture2D, Layout::kHWC});

  ABSL_RETURN_IF_ERROR(TestConvSoftmaxConvPerformance(
      BHWC(1, 16, 4096, 40), 40, 4096, op_def, precision, &env));

  return absl::OkStatus();
}

absl::Status FullyConnectedOptimalWGSize(CalculationsPrecision precision,
                                         DataType weights_type,
                                         const BHWC& src_shape,
                                         int dst_channels, OHWI scale_zp_shape,
                                         bool sparse_2x4) {
  const bool use_zero_point = false;
  const OHWI weights_shape(dst_channels, scale_zp_shape.h, 1,
                           src_shape.c / (sparse_2x4 ? 2 : 1));
  ml_drift::Tensor<OHWI, DataType::kFloat32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  // Initialize weights with non uniform values. Uniform values can be optimized
  // in textures and show too good performance???
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = i;
  }

  ml_drift::Tensor<OHWI, DataType::kInt8> weights_i8;
  weights_i8.shape = weights_shape;
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(uint8_t));
  for (int i = 0; i < weights_i8.data.size(); ++i) {
    weights_i8.data[i] = (i % 256) - 128;
  }

  ml_drift::Tensor<OHWI, DataType::kFloat32> weights_scale;
  weights_scale.shape = scale_zp_shape;
  weights_scale.data.resize(weights_scale.shape.DimensionsProduct(), 1.0f);
  for (int i = 0; i < weights_scale.data.size(); ++i) {
    weights_scale.data[i] = std::sin(i);
  }
  ml_drift::Tensor<OHWI, DataType::kFloat32> weights_zp;
  weights_zp.shape = scale_zp_shape;
  weights_zp.data.resize(weights_zp.shape.DimensionsProduct(), 0.0f);
  for (int i = 0; i < weights_zp.data.size(); ++i) {
    weights_zp.data[i] = std::cos(i);
  }

  Environment env;
  ABSL_RETURN_IF_ERROR(CreateEnvironment(&env));
  const GpuInfo& gpu_info = env.GetDevicePtr()->GetInfo();

  OperationDef op_def;
  auto data_type = DeduceDataTypeFromPrecision(precision);
  Layout layout = src_shape.b == 1 ? Layout::kHWC : Layout::kBHWC;
  auto storage_type = GetFastestStorageType(env.device().GetInfo());
  op_def.src_tensors.push_back({data_type, storage_type, layout});
  ABSL_RETURN_IF_ERROR(op_def.src_tensors[0].UpdateToSupportedStorageType(
      env.device().GetInfo(), src_shape));
  op_def.dst_tensors.push_back({data_type, storage_type, layout});
  auto dst_shape = src_shape;
  dst_shape.c = dst_channels;
  ABSL_RETURN_IF_ERROR(op_def.dst_tensors[0].UpdateToSupportedStorageType(
      env.device().GetInfo(), dst_shape));

  Tensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(src_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  descriptor_with_shape.SetBHWCShape(dst_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), descriptor_with_shape, &dst));

  const auto w_shape = weights.shape;

  std::cout << "Src - " << src_shape.h << "x" << src_shape.w << "x"
            << src_shape.c << ", "
            << ToString(op_def.src_tensors[0].GetStorageType()) << std::endl;
  std::cout << "Dst - " << dst_shape.h << "x" << dst_shape.w << "x"
            << dst_shape.c << ", "
            << ToString(op_def.dst_tensors[0].GetStorageType()) << std::endl;

  double element_size = precision == CalculationsPrecision::kF32 ? 4.0 : 2.0;
  int64_t flops_per_element = w_shape.i * w_shape.h * w_shape.w * 2;
  if (scale_zp_shape.h > 1) {
    flops_per_element /= scale_zp_shape.h;
  }
  const int64_t dst_elements = dst.Width() * dst.Height() * dst.Channels();
  const int64_t flops_count = dst_elements * flops_per_element;
  const double gflops_count = flops_count * 1e-9;

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const int64_t dst_elements_alignedx4 =
      dst.Width() * dst.Height() * dst.Slices() * 4;
  const int64_t src_elements_alignedx4 =
      src.Width() * src.Height() * src.Slices() * 4;
  const double dst_gbytes = dst_elements_alignedx4 * element_size / kGByte;
  const double src_gbytes = src_elements_alignedx4 * element_size / kGByte;
  double weight_element_size = SizeInBitsOf(weights_type) / 8.0;
  if (sparse_2x4) {
    weight_element_size += 0.25;
  }
  const double weight_gbytes =
      weights.shape.DimensionsProduct() * weight_element_size / kGByte;
  const double bias_gbytes = weights.shape.o * element_size / kGByte;
  const double scale_gbytes =
      scale_zp_shape.DimensionsProduct() * element_size / kGByte;

  std::vector<int3> wg_sizes;
  const int max_wg_size = 256;
  for (int y = 1; y <= max_wg_size; y *= 2) {
    for (int x = max_wg_size; x >= 1; x /= 2) {
      if (x * y >= 16 && x * y <= max_wg_size) {
        wg_sizes.push_back({x, y, 1});
      }
    }
  }
  std::vector<double> time_ms(wg_sizes.size());

  WeightsDescription weights_desc;
  std::vector<TensorDescriptor> weights_gpu;
  if (sparse_2x4) {
    weights_desc.type = DataType::kUint4;
    weights_desc.layout = WeightsLayout::kCustomGroups;
    weights_desc.group_sizes = {
        {Axis::kOutputChannels, 4},
        {Axis::kInputChannels, 2},
        {Axis::kOutputChannels, DivideRoundUp(weights_i8.shape.o, 4)},
        {Axis::kInputChannels, 0},
        {Axis::kOutputChannels, 0},
    };
    ml_drift::Tensor<OHWI, DataType::kUint8> weights_indices;
    weights_indices.shape = weights_i8.shape;
    weights_indices.data.resize(weights_indices.shape.DimensionsProduct() +
                                XNN_EXTRA_BYTES / sizeof(uint8_t));

    const int elements_count =
        GetTotalElementsCountForLayout(weights_desc, weights_i8.shape);
    {
      std::vector<uint8_t> weights_data(elements_count / 2);
      RearrangeWeightsInt8AsUint4(weights_i8, weights_desc,
                                  absl::MakeSpan(weights_data),
                                  /*shift_value=*/8, /*pad_value=*/8u);

      TensorDescriptor weights_gpu_desc(
          DataType::kUint32, TensorStorageType::kBuffer, Layout::kLinear);
      weights_gpu_desc.SetBHWCShape(BHWC(1, 1, 1, weights_data.size()));
      weights_gpu_desc.UploadDataRaw(absl::MakeConstSpan(weights_data));
      weights_gpu.push_back(std::move(weights_gpu_desc));
    }
    {
      WeightsDescription indices_desc = weights_desc;
      indices_desc.type = DataType::kUint2;
      std::vector<uint8_t> weights_indices_data(elements_count / 4);
      RearrangeWeightsUint2(weights_indices, indices_desc,
                            absl::MakeSpan(weights_indices_data));

      TensorDescriptor weights_indices_desc(
          DataType::kUint32, TensorStorageType::kBuffer, Layout::kLinear);
      weights_indices_desc.SetBHWCShape(
          BHWC(1, 1, 1, weights_indices_data.size()));
      weights_indices_desc.UploadDataRaw(
          absl::MakeConstSpan(weights_indices_data));
      weights_gpu.push_back(std::move(weights_indices_desc));
    }
  } else if (weights_type == DataType::kFloat16 ||
             weights_type == DataType::kFloat32) {
    weights_desc.type = DeduceDataTypeFromPrecision(precision);
    weights_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
    weights_desc.output_group_size = DivideRoundUp(weights.shape.o, 4);
    weights_gpu = GetTensorDescriptorsForWeightsLayout(weights, weights_desc);
  } else if (weights_type == DataType::kInt8) {
    weights_desc = GetFullyConnectedInt8WeightsDesc(gpu_info, weights_i8.shape);
    weights_gpu.push_back(
        GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc));
  } else if (weights_type == DataType::kInt4) {
    weights_desc = GetFullyConnectedInt4WeightsDesc(gpu_info, weights_i8.shape);
    weights_gpu.push_back(
        GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc));
  } else if (weights_type == DataType::kInt2) {
    weights_desc = GetFullyConnectedInt2WeightsDesc(gpu_info, weights_i8.shape);
    weights_gpu.push_back(
        GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc));
  }

  std::vector<Tensor> weights_tensors(weights_gpu.size());
  for (int i = 0; i < weights_gpu.size(); ++i) {
    ABSL_RETURN_IF_ERROR(
        CreateTensor(env.context(), weights_gpu[i], &weights_tensors[i]));
  }

  TensorDescriptor scale_desc =
      ScaleOrZeroPointToTensorDesc(gpu_info, weights_scale, data_type);
  TensorDescriptor zp_desc =
      ScaleOrZeroPointToTensorDesc(gpu_info, weights_zp, data_type);

  const bool is_quantized = SizeInBitsOf(weights_type) <= 8;

  Tensor scale_tensor;
  Tensor zp_tensor;
  if (is_quantized) {
    ABSL_RETURN_IF_ERROR(
        CreateTensor(env.context(), scale_desc, &scale_tensor));
    if (use_zero_point) {
      ABSL_RETURN_IF_ERROR(CreateTensor(env.context(), zp_desc, &zp_tensor));
    }
  }

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights.shape;
  if (is_quantized) {
    external_weights.scale_zp_shape = scale_zp_shape;
    external_weights.scale = &scale_desc;
    if (use_zero_point) {
      external_weights.zero_point = &zp_desc;
    }
  }

  double gbytes_read = src_gbytes + weight_gbytes + bias_gbytes;
  if (is_quantized) {
    gbytes_read += scale_gbytes;
    if (use_zero_point) {
      gbytes_read += scale_gbytes;
    }
  }

  const bool kUseMultipleWeights = false;
  const int kMultiplier = 32;
  std::vector<Tensor> weights_tensors_multiple(weights_gpu.size() *
                                               kMultiplier);
  std::vector<Tensor> scale_tensors_multiple(kMultiplier);
  std::vector<Tensor> zp_tensors_multiple(kMultiplier);
  if (kUseMultipleWeights) {
    std::cout << "Weights total size: " << weight_gbytes * kMultiplier * 1024.0
              << " MB" << std::endl;
    for (int k = 0; k < kMultiplier; ++k) {
      for (int i = 0; i < weights_gpu.size(); ++i) {
        ABSL_RETURN_IF_ERROR(CreateTensor(
            env.context(), weights_gpu[i],
            &weights_tensors_multiple[k * weights_gpu.size() + i]));
      }
      if (is_quantized) {
        ABSL_RETURN_IF_ERROR(CreateTensor(env.context(), scale_desc,
                                          &scale_tensors_multiple[k]));
        if (use_zero_point) {
          ABSL_RETURN_IF_ERROR(
              CreateTensor(env.context(), zp_desc, &zp_tensors_multiple[k]));
        }
      }
    }
  }
  for (int i = 0; i < wg_sizes.size(); ++i) {
    std::unique_ptr<GPUOperation> conv;
    if (sparse_2x4) {
      conv = std::make_unique<FullyConnected>(CreateFullyConnectedInt4Sparse2x4(
          gpu_info, precision, op_def.src_tensors[0], op_def.dst_tensors[0],
          external_weights, /*bias=*/nullptr, &dst_shape, &wg_sizes[i]));
    } else {
      ABSL_ASSIGN_OR_RETURN(
          auto operation,
          CreateFullyConnectedExternalWeights(
              gpu_info, precision, op_def.src_tensors[0], op_def.dst_tensors[0],
              external_weights, /*bias=*/nullptr, &dst_shape,
              /*src_exp=*/nullptr,
              /*runtime_check=*/{}, &wg_sizes[i]));
      conv = std::make_unique<FullyConnected>(std::move(operation));
    }
    ABSL_RETURN_IF_ERROR(conv->AssembleCode(gpu_info));

    ClOperation cl_op;
    cl_op.Init(std::move(conv));
    ABSL_RETURN_IF_ERROR(
        cl_op.Compile(env.GetDevicePtr(), &env.context(), env.program_cache()));
    int index = 0;
    ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(index++, &src));
    for (int i = 0; i < weights_tensors.size(); ++i) {
      ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(index++, &weights_tensors[i]));
    }
    if (is_quantized) {
      ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(index++, &scale_tensor));
      if (use_zero_point) {
        ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(index++, &zp_tensor));
      }
    }
    ABSL_RETURN_IF_ERROR(cl_op.SetDstTensor(0, &dst));
    ABSL_RETURN_IF_ERROR(cl_op.UpdateParams());
    ABSL_RETURN_IF_ERROR(cl_op.Tune(TuningType::kExhaustive,
                                    env.device().GetInfo(),
                                    env.profiling_queue()));

    ABSL_ASSIGN_OR_RETURN(auto duration,
                          cl_op.GetOpTime(env.device().GetInfo(), env.queue(),
                                          env.profiling_queue()));
    time_ms[i] = absl::ToDoubleMilliseconds(duration);

    if (kUseMultipleWeights) {
      int inferences = 200.0 / time_ms[i];
      inferences = AlignByN(inferences, kMultiplier);

      const auto start = absl::Now();
      for (int j = 0; j < inferences; ++j) {
        int w_id = j % kMultiplier;
        int index = 1;
        for (int k = 0; k < weights_gpu.size(); ++k) {
          ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(
              index++,
              &weights_tensors_multiple[w_id * weights_gpu.size() + k]));
        }
        if (is_quantized) {
          ABSL_RETURN_IF_ERROR(
              cl_op.SetSrcTensor(index++, &scale_tensors_multiple[w_id]));
          if (use_zero_point) {
            ABSL_RETURN_IF_ERROR(
                cl_op.SetSrcTensor(index++, &zp_tensors_multiple[w_id]));
          }
        }
        ABSL_RETURN_IF_ERROR(cl_op.UpdateParams());
        ABSL_RETURN_IF_ERROR(cl_op.AddToQueue(env.queue()));
      }
      ABSL_RETURN_IF_ERROR(env.queue()->WaitForCompletion());
      const auto end = absl::Now();
      double time_ms_local =
          static_cast<double>((end - start) / absl::Nanoseconds(1)) /
          inferences * 1e-6;
      time_ms[i] = time_ms_local;
    }

    const double fps = 1000.0 / time_ms[i];
    const double gflops_real = fps * gflops_count;
    const double gbs_read = fps * gbytes_read;
    const double gbs_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(4)
              << "WG size: " << wg_sizes[i].x << "x" << wg_sizes[i].y
              << " time - " << time_ms[i] << "(ms), GFlops - " << gflops_real
              << ", Bandwidth - " << gbs_read + gbs_write << "(GB/s), Read - "
              << gbs_read << "(GB/s), Write - " << gbs_write << "(GB/s)"
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
  const double gbs_read = fps * gbytes_read;
  const double gbs_write = fps * dst_gbytes;
  std::cout << std::endl
            << "Optimal WG size: " << optimal_wg_size.x << "x"
            << optimal_wg_size.y << std::fixed << std::setprecision(4)
            << " Time - " << min_time_ms << "(ms), GFlops - " << gflops_real
            << ", Bandwidth - " << gbs_read + gbs_write << "(GB/s), Read - "
            << gbs_read << "(GB/s), Write - " << gbs_write << "(GB/s)"
            << std::endl
            << std::endl;

  double sub_optimal_threshold = 5.0;  // in percents
  std::cout << std::fixed << std::setprecision(1)
            << "WG sizes close to optimal (" << sub_optimal_threshold
            << "%): " << std::setprecision(4) << std::endl;
  for (int i = 0; i < wg_sizes.size(); ++i) {
    const auto& wg_size = wg_sizes[i];
    if (time_ms[i] < min_time_ms * (1.0 + sub_optimal_threshold / 100.0)) {
      double scale = min_time_ms / time_ms[i];
      std::cout << "WG size: " << wg_size.x << "x" << wg_size.y
                << " - time: " << time_ms[i] << ", Bandwidth - "
                << (gbs_read + gbs_write) * scale << "(GB/s)" << std::endl;
    }
  }

  return absl::OkStatus();
}

absl::Status FullyConnectedPerfTest(CalculationsPrecision precision,
                                    DataType weights_type,
                                    const BHWC& src_shape, int dst_channels,
                                    OHWI scale_zp_shape, bool sparse_2x4) {
  const bool use_zero_point = false;
  const OHWI weights_shape(dst_channels, scale_zp_shape.h, 1,
                           src_shape.c / (sparse_2x4 ? 2 : 1));
  ml_drift::Tensor<OHWI, DataType::kFloat32> weights;
  weights.shape = weights_shape;
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  // Initialize weights with non uniform values. Uniform values can be optimized
  // in textures and show too good performance???
  for (int i = 0; i < weights.data.size(); ++i) {
    weights.data[i] = i;
  }

  ml_drift::Tensor<OHWI, DataType::kInt8> weights_i8;
  weights_i8.shape = weights_shape;
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(uint8_t));
  for (int i = 0; i < weights_i8.data.size(); ++i) {
    weights_i8.data[i] = (i % 256) - 128;
  }

  ml_drift::Tensor<OHWI, DataType::kFloat32> weights_scale;
  weights_scale.shape = scale_zp_shape;
  weights_scale.data.resize(weights_scale.shape.DimensionsProduct(), 1.0f);
  for (int i = 0; i < weights_scale.data.size(); ++i) {
    weights_scale.data[i] = std::sin(i);
  }
  ml_drift::Tensor<OHWI, DataType::kFloat32> weights_zp;
  weights_zp.shape = scale_zp_shape;
  weights_zp.data.resize(weights_zp.shape.DimensionsProduct(), 0.0f);
  for (int i = 0; i < weights_zp.data.size(); ++i) {
    weights_zp.data[i] = std::cos(i);
  }

  Environment env;
  ABSL_RETURN_IF_ERROR(CreateEnvironment(&env));
  const GpuInfo& gpu_info = env.GetDevicePtr()->GetInfo();

  OperationDef op_def;
  auto data_type = DeduceDataTypeFromPrecision(precision);
  Layout layout = src_shape.b == 1 ? Layout::kHWC : Layout::kBHWC;
  auto storage_type = GetFastestStorageType(env.device().GetInfo());
  op_def.src_tensors.push_back({data_type, storage_type, layout});
  ABSL_RETURN_IF_ERROR(op_def.src_tensors[0].UpdateToSupportedStorageType(
      env.device().GetInfo(), src_shape));
  op_def.dst_tensors.push_back({data_type, storage_type, layout});
  auto dst_shape = src_shape;
  dst_shape.c = dst_channels;
  ABSL_RETURN_IF_ERROR(op_def.dst_tensors[0].UpdateToSupportedStorageType(
      env.device().GetInfo(), dst_shape));

  Tensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(src_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  descriptor_with_shape.SetBHWCShape(dst_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), descriptor_with_shape, &dst));

  const auto w_shape = weights.shape;

  std::cout << "Src - " << src_shape.h << "x" << src_shape.w << "x"
            << src_shape.c << ", "
            << ToString(op_def.src_tensors[0].GetStorageType()) << std::endl;
  std::cout << "Dst - " << dst_shape.h << "x" << dst_shape.w << "x"
            << dst_shape.c << ", "
            << ToString(op_def.dst_tensors[0].GetStorageType()) << std::endl;

  double element_size = precision == CalculationsPrecision::kF32 ? 4.0 : 2.0;
  int64_t flops_per_element = w_shape.i * w_shape.h * w_shape.w * 2;
  if (scale_zp_shape.h > 1) {
    flops_per_element /= scale_zp_shape.h;
  }
  const int64_t dst_elements = dst.Width() * dst.Height() * dst.Channels();
  const int64_t flops_count = dst_elements * flops_per_element;
  const double gflops_count = flops_count * 1e-9;

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const int64_t dst_elements_alignedx4 =
      dst.Width() * dst.Height() * dst.Slices() * 4;
  const int64_t src_elements_alignedx4 =
      src.Width() * src.Height() * src.Slices() * 4;
  const double dst_gbytes = dst_elements_alignedx4 * element_size / kGByte;
  const double src_gbytes = src_elements_alignedx4 * element_size / kGByte;
  double weight_element_size = SizeInBitsOf(weights_type) / 8.0;
  if (sparse_2x4) {
    weight_element_size += 0.25;
  }
  const double weight_gbytes =
      weights.shape.DimensionsProduct() * weight_element_size / kGByte;
  const double scale_gbytes =
      scale_zp_shape.DimensionsProduct() * element_size / kGByte;

  std::cout << "Weight size: " << weight_gbytes * 1024.0 << " MB" << std::endl;

  WeightsDescription weights_desc;
  std::vector<TensorDescriptor> weights_gpu;
  if (sparse_2x4) {
    weights_desc.type = DataType::kUint4;
    weights_desc.layout = WeightsLayout::kCustomGroups;
    weights_desc.group_sizes = {
        {Axis::kOutputChannels, 4},
        {Axis::kInputChannels, 2},
        {Axis::kOutputChannels, DivideRoundUp(weights_i8.shape.o, 4)},
        {Axis::kInputChannels, 0},
        {Axis::kOutputChannels, 0},
    };
    ml_drift::Tensor<OHWI, DataType::kUint8> weights_indices;
    weights_indices.shape = weights_i8.shape;
    weights_indices.data.resize(weights_indices.shape.DimensionsProduct() +
                                XNN_EXTRA_BYTES / sizeof(uint8_t));

    const int elements_count =
        GetTotalElementsCountForLayout(weights_desc, weights_i8.shape);
    {
      std::vector<uint8_t> weights_data(elements_count / 2);
      RearrangeWeightsInt8AsUint4(weights_i8, weights_desc,
                                  absl::MakeSpan(weights_data),
                                  /*shift_value=*/8, /*pad_value=*/8u);

      TensorDescriptor weights_gpu_desc(
          DataType::kUint32, TensorStorageType::kBuffer, Layout::kLinear);
      weights_gpu_desc.SetBHWCShape(BHWC(1, 1, 1, weights_data.size()));
      weights_gpu_desc.UploadDataRaw(absl::MakeConstSpan(weights_data));
      weights_gpu.push_back(std::move(weights_gpu_desc));
    }
    {
      WeightsDescription indices_desc = weights_desc;
      indices_desc.type = DataType::kUint2;
      std::vector<uint8_t> weights_indices_data(elements_count / 4);
      RearrangeWeightsUint2(weights_indices, indices_desc,
                            absl::MakeSpan(weights_indices_data));

      TensorDescriptor weights_indices_desc(
          DataType::kUint32, TensorStorageType::kBuffer, Layout::kLinear);
      weights_indices_desc.SetBHWCShape(
          BHWC(1, 1, 1, weights_indices_data.size()));
      weights_indices_desc.UploadDataRaw(
          absl::MakeConstSpan(weights_indices_data));
      weights_gpu.push_back(std::move(weights_indices_desc));
    }
  } else if (weights_type == DataType::kFloat16 ||
             weights_type == DataType::kFloat32) {
    weights_desc.type = DeduceDataTypeFromPrecision(precision);
    weights_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
    weights_desc.output_group_size = DivideRoundUp(weights.shape.o, 4);
    weights_gpu = GetTensorDescriptorsForWeightsLayout(weights, weights_desc);
  } else if (weights_type == DataType::kInt8) {
    weights_desc = GetFullyConnectedInt8WeightsDesc(gpu_info, weights_i8.shape);
    weights_gpu.push_back(
        GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc));
  } else if (weights_type == DataType::kInt4) {
    weights_desc = GetFullyConnectedInt4WeightsDesc(gpu_info, weights_i8.shape);
    weights_gpu.push_back(
        GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc));
  } else if (weights_type == DataType::kInt2) {
    weights_desc = GetFullyConnectedInt2WeightsDesc(gpu_info, weights_i8.shape);
    weights_gpu.push_back(
        GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc));
  }

  std::vector<Tensor> weights_tensors(weights_gpu.size());
  for (int i = 0; i < weights_gpu.size(); ++i) {
    ABSL_RETURN_IF_ERROR(
        CreateTensor(env.context(), weights_gpu[i], &weights_tensors[i]));
  }

  TensorDescriptor scale_desc =
      ScaleOrZeroPointToTensorDesc(gpu_info, weights_scale, data_type);
  TensorDescriptor zp_desc =
      ScaleOrZeroPointToTensorDesc(gpu_info, weights_zp, data_type);

  const bool is_quantized = SizeInBitsOf(weights_type) <= 8;

  Tensor scale_tensor;
  Tensor zp_tensor;
  if (is_quantized) {
    ABSL_RETURN_IF_ERROR(
        CreateTensor(env.context(), scale_desc, &scale_tensor));
    ABSL_RETURN_IF_ERROR(CreateTensor(env.context(), zp_desc, &zp_tensor));
  }

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights.shape;
  if (is_quantized) {
    external_weights.scale_zp_shape = scale_zp_shape;
    external_weights.scale = &scale_desc;
    if (use_zero_point) {
      external_weights.zero_point = &zp_desc;
    }
  }

  std::unique_ptr<GPUOperation> conv;
  if (sparse_2x4) {
    conv = std::make_unique<FullyConnected>(CreateFullyConnectedInt4Sparse2x4(
        gpu_info, precision, op_def.src_tensors[0], op_def.dst_tensors[0],
        external_weights, /*bias=*/nullptr, &dst_shape));
  } else {
    ABSL_ASSIGN_OR_RETURN(
        auto operation,
        CreateFullyConnectedExternalWeights(
            gpu_info, precision, op_def.src_tensors[0], op_def.dst_tensors[0],
            external_weights, /*bias=*/nullptr, &dst_shape));
    conv = std::make_unique<FullyConnected>(std::move(operation));
  }
  ABSL_RETURN_IF_ERROR(conv->AssembleCode(gpu_info));

  ClOperation cl_op;
  cl_op.Init(std::move(conv));
  ABSL_RETURN_IF_ERROR(
      cl_op.Compile(env.GetDevicePtr(), &env.context(), env.program_cache()));
  int index = 0;
  ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(index++, &src));
  for (int i = 0; i < weights_tensors.size(); ++i) {
    ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(index++, &weights_tensors[i]));
  }
  if (is_quantized) {
    ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(index++, &scale_tensor));
    if (use_zero_point) {
      ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(index++, &zp_tensor));
    }
  }
  ABSL_RETURN_IF_ERROR(cl_op.SetDstTensor(0, &dst));
  ABSL_RETURN_IF_ERROR(cl_op.UpdateParams());
  ABSL_RETURN_IF_ERROR(cl_op.Tune(
      TuningType::kExhaustive, env.device().GetInfo(), env.profiling_queue()));

  double gbytes_read = src_gbytes + weight_gbytes;
  if (is_quantized) {
    gbytes_read += scale_gbytes;
    if (use_zero_point) {
      gbytes_read += scale_gbytes;
    }
  }

  absl::Duration min_duration = absl::InfiniteDuration();
  for (int i = 0; i < 10; ++i) {
    ABSL_ASSIGN_OR_RETURN(auto duration,
                          cl_op.GetOpTime(env.device().GetInfo(), env.queue(),
                                          env.profiling_queue()));
    double time_ms = absl::ToDoubleMilliseconds(duration);
    const double fps = 1000.0 / time_ms;
    const double gflops_real = fps * gflops_count;
    const double gbs_read = fps * gbytes_read;
    const double gbs_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(4) << " Time - " << time_ms
              << "(ms), GFlops - " << gflops_real << ", Bandwidth - "
              << gbs_read + gbs_write << "(GB/s), Read - " << gbs_read
              << "(GB/s), Write - " << gbs_write << "(GB/s)" << std::endl;
    min_duration = std::min(min_duration, duration);
  }

  // For testing weights reading from global memory. Often tensors can be cached
  // into some cache levels. This can lead to better bandwidth in test above
  // than theoretical or when reading from global memory. With this test we want
  // to test the weights reading from global memory.
  const bool kUseMultipleWeights = false;
  if (kUseMultipleWeights) {
    const int kMultiplier = 32;
    int inferences = 1000.0 / absl::ToDoubleMilliseconds(min_duration);
    inferences = AlignByN(inferences, kMultiplier);
    std::vector<Tensor> weights_tensors_multiple(weights_gpu.size() *
                                                 kMultiplier);
    std::vector<Tensor> scale_tensors_multiple(kMultiplier);
    std::vector<Tensor> zp_tensors_multiple(kMultiplier);
    for (int k = 0; k < kMultiplier; ++k) {
      for (int i = 0; i < weights_gpu.size(); ++i) {
        ABSL_RETURN_IF_ERROR(CreateTensor(
            env.context(), weights_gpu[i],
            &weights_tensors_multiple[k * weights_gpu.size() + i]));
      }
      if (is_quantized) {
        ABSL_RETURN_IF_ERROR(CreateTensor(env.context(), scale_desc,
                                          &scale_tensors_multiple[k]));
        if (use_zero_point) {
          ABSL_RETURN_IF_ERROR(
              CreateTensor(env.context(), zp_desc, &zp_tensors_multiple[k]));
        }
      }
    }
    std::cout << "Weight total size: " << weight_gbytes * kMultiplier * 1024.0
              << " MB" << std::endl;
    for (int i = 0; i < 10; ++i) {
      const auto start = absl::Now();
      for (int j = 0; j < inferences; ++j) {
        int w_id = j % kMultiplier;
        int index = 1;
        for (int k = 0; k < weights_gpu.size(); ++k) {
          ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(
              index++,
              &weights_tensors_multiple[w_id * weights_gpu.size() + k]));
        }
        if (is_quantized) {
          ABSL_RETURN_IF_ERROR(
              cl_op.SetSrcTensor(index++, &scale_tensors_multiple[w_id]));
          if (use_zero_point) {
            ABSL_RETURN_IF_ERROR(
                cl_op.SetSrcTensor(index++, &zp_tensors_multiple[w_id]));
          }
        }
        ABSL_RETURN_IF_ERROR(cl_op.UpdateParams());
        ABSL_RETURN_IF_ERROR(cl_op.AddToQueue(env.queue()));
      }
      ABSL_RETURN_IF_ERROR(env.queue()->WaitForCompletion());
      const auto end = absl::Now();
      double time_ms =
          static_cast<double>((end - start) / absl::Nanoseconds(1)) /
          inferences * 1e-6;
      const double fps = 1000.0 / time_ms;
      const double gflops_real = fps * gflops_count;
      double gbs_read = fps * gbytes_read;
      const double gbs_write = fps * dst_gbytes;
      std::cout << std::fixed << std::setprecision(4) << " Time - " << time_ms
                << "(ms), GFlops - " << gflops_real << ", Bandwidth - "
                << gbs_read + gbs_write << "(GB/s), Read - " << gbs_read
                << "(GB/s), Write - " << gbs_write << "(GB/s)" << std::endl;
    }
  }

  return absl::OkStatus();
}

absl::Status FullyConnectedWeightsBatchIdsPerfTest(
    CalculationsPrecision precision, DataType weights_type,
    const BHWC& src_shape, int dst_channels, int batch_size,
    int active_ids_size, OHWI scale_zp_shape) {
  const bool use_zero_point = false;
  ml_drift::Tensor<OHWI, DataType::kFloat32> weights;
  weights.shape = OHWI(dst_channels, batch_size, 1, src_shape.c);
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));

  ml_drift::Tensor<OHWI, DataType::kInt8> weights_i8;
  weights_i8.shape = OHWI(dst_channels, batch_size, 1, src_shape.c);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(uint8_t));

  ml_drift::Tensor<OHWI, DataType::kFloat32> weights_scale;
  weights_scale.shape = scale_zp_shape;
  weights_scale.data.resize(weights_scale.shape.DimensionsProduct(), 1.0f);
  ml_drift::Tensor<OHWI, DataType::kFloat32> weights_zp;
  weights_zp.shape = scale_zp_shape;
  weights_zp.data.resize(weights_zp.shape.DimensionsProduct(), 0.0f);

  Environment env;
  ABSL_RETURN_IF_ERROR(CreateEnvironment(&env));
  const GpuInfo& gpu_info = env.GetDevicePtr()->GetInfo();

  OperationDef op_def;
  auto data_type = DeduceDataTypeFromPrecision(precision);
  Layout layout = src_shape.b == 1 ? Layout::kHWC : Layout::kBHWC;
  auto storage_type = GetFastestStorageType(env.device().GetInfo());
  op_def.src_tensors.push_back({data_type, storage_type, layout});
  op_def.dst_tensors.push_back({data_type, storage_type, layout});

  auto dst_shape = src_shape;
  dst_shape.c = dst_channels;
  dst_shape.h = active_ids_size;

  Tensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(src_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  descriptor_with_shape.SetBHWCShape(dst_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), descriptor_with_shape, &dst));

  ml_drift::Tensor<BHWC, DataType::kInt32> ids_data;
  ids_data.shape = BHWC(1, 1, 1, active_ids_size);
  ids_data.data.resize(ids_data.shape.DimensionsProduct(), 0);
  for (int i = 0; i < active_ids_size; ++i) {
    ids_data.data[i] = i;  // rand() % batch_size;
  }
  TensorDescriptor ids_desc =
      TensorDescriptor(DataType::kInt32, storage_type, Layout::kHWC);
  TensorDescriptor ids_desc_with_data = ids_desc;
  ids_desc_with_data.SetBHWCShape(ids_data.shape);
  ids_desc_with_data.UploadData(ids_data);
  Tensor ids;
  ABSL_RETURN_IF_ERROR(CreateTensor(env.context(), ids_desc_with_data, &ids));

  const auto w_shape = weights.shape;

  std::cout << "Src size(HWC) - " << src_shape.h << "x" << src_shape.w << "x"
            << src_shape.c << std::endl;
  std::cout << "Dst size(HWC) - " << dst_shape.h << "x" << dst_shape.w << "x"
            << dst_shape.c << std::endl;

  double element_size = precision == CalculationsPrecision::kF32 ? 4.0 : 2.0;
  const int64_t flops_per_element = w_shape.i * 2;
  const int64_t dst_elements = dst.Width() * dst.Height() * dst.Channels();
  const int64_t flops_count = dst_elements * flops_per_element;
  const double gflops_count = flops_count * 1e-9;

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const int64_t dst_elements_alignedx4 =
      dst.Width() * dst.Height() * dst.Slices() * 4;
  const int64_t src_elements_alignedx4 =
      src.Width() * src.Height() * src.Slices() * 4;
  const double dst_gbytes = dst_elements_alignedx4 * element_size / kGByte;
  const double src_gbytes = src_elements_alignedx4 * element_size / kGByte;
  const double weight_element_size = SizeInBitsOf(weights_type) / 8.0;
  const double weight_gbytes = weights.shape.DimensionsProduct() / batch_size *
                               active_ids_size * weight_element_size / kGByte;
  const double scale_gbytes = scale_zp_shape.DimensionsProduct() / batch_size *
                              active_ids_size * element_size / kGByte;

  WeightsDescription weights_desc;
  std::vector<TensorDescriptor> weights_gpu;
  if (weights_type == DataType::kFloat16 ||
      weights_type == DataType::kFloat32) {
    weights_desc.type = DeduceDataTypeFromPrecision(precision);
    weights_desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
    // weights_desc.layout =
    //    WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4;
    weights_desc.output_group_size = DivideRoundUp(weights.shape.o, 4);

    weights_gpu = GetTensorDescriptorsForWeightsLayout(weights, weights_desc);
  } else if (weights_type == DataType::kInt8) {
    weights_desc = GetFullyConnectedInt8WeightsDesc(gpu_info, weights_i8.shape);
    weights_gpu.push_back(
        GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc));
  } else if (weights_type == DataType::kInt4) {
    weights_desc = GetFullyConnectedInt4WeightsDesc(gpu_info, weights_i8.shape);
    weights_gpu.push_back(
        GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc));
  } else if (weights_type == DataType::kInt2) {
    weights_desc = GetFullyConnectedInt2WeightsDesc(gpu_info, weights_i8.shape);
    weights_gpu.push_back(
        GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc));
  }

  std::vector<Tensor> weights_tensors(weights_gpu.size());
  for (int i = 0; i < weights_gpu.size(); ++i) {
    ABSL_RETURN_IF_ERROR(
        CreateTensor(env.context(), weights_gpu[i], &weights_tensors[i]));
  }

  TensorDescriptor scale_desc =
      ScaleOrZeroPointToTensorDesc(gpu_info, weights_scale, data_type);
  TensorDescriptor zp_desc =
      ScaleOrZeroPointToTensorDesc(gpu_info, weights_zp, data_type);

  const bool is_quantized =
      weights_type != DataType::kFloat16 && weights_type != DataType::kFloat32;

  Tensor scale_tensor;
  Tensor zp_tensor;
  if (is_quantized) {
    ABSL_RETURN_IF_ERROR(
        CreateTensor(env.context(), scale_desc, &scale_tensor));
    ABSL_RETURN_IF_ERROR(CreateTensor(env.context(), zp_desc, &zp_tensor));
  }

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights.shape;
  if (is_quantized) {
    external_weights.scale_zp_shape = scale_zp_shape;
    external_weights.scale = &scale_desc;
    if (use_zero_point) {
      external_weights.zero_point = &zp_desc;
    }
  }

  ABSL_ASSIGN_OR_RETURN(
      auto operation,
      CreateFullyConnectedWeightsBatchIds(
          gpu_info, precision, op_def.src_tensors[0], ids_desc,
          op_def.dst_tensors[0], external_weights, nullptr, &dst_shape));
  std::unique_ptr<GPUOperation> conv =
      std::make_unique<FullyConnected>(std::move(operation));
  ABSL_RETURN_IF_ERROR(conv->AssembleCode(gpu_info));

  ClOperation cl_op;
  cl_op.Init(std::move(conv));
  ABSL_RETURN_IF_ERROR(
      cl_op.Compile(env.GetDevicePtr(), &env.context(), env.program_cache()));
  int index = 0;
  ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(index++, &src));
  ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(index++, &ids));
  for (int i = 0; i < weights_tensors.size(); ++i) {
    ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(index++, &weights_tensors[i]));
  }
  if (is_quantized) {
    ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(index++, &scale_tensor));
    if (use_zero_point) {
      ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(index++, &zp_tensor));
    }
  }
  ABSL_RETURN_IF_ERROR(cl_op.SetDstTensor(0, &dst));
  ABSL_RETURN_IF_ERROR(cl_op.UpdateParams());
  ABSL_RETURN_IF_ERROR(cl_op.Tune(
      TuningType::kExhaustive, env.device().GetInfo(), env.profiling_queue()));

  double gbytes_read = src_gbytes + weight_gbytes;
  if (is_quantized) {
    gbytes_read += scale_gbytes;
    if (use_zero_point) {
      gbytes_read += scale_gbytes;
    }
  }

  for (int i = 0; i < 10; ++i) {
    ABSL_ASSIGN_OR_RETURN(auto duration,
                          cl_op.GetOpTime(env.device().GetInfo(), env.queue(),
                                          env.profiling_queue()));
    double time_ms = absl::ToDoubleMilliseconds(duration);
    const double fps = 1000.0 / time_ms;
    const double gflops_real = fps * gflops_count;
    double gbs_read = fps * gbytes_read;
    const double gbs_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(4) << " Time - " << time_ms
              << "(ms), GFlops - " << gflops_real << ", Bandwidth - "
              << gbs_read + gbs_write << "(GB/s), Read - " << gbs_read
              << "(GB/s), Write - " << gbs_write << "(GB/s)" << std::endl;
  }

  return absl::OkStatus();
}

absl::Status FullyConnectedOIPerfTest(const BHWC& src_shape, int dst_channels,
                                      int src_group_size, DataType weights_type,
                                      bool sparse_2x4) {
  Environment env;
  ABSL_RETURN_IF_ERROR(CreateEnvironment(&env));

  if (src_group_size <= 0) {
    src_group_size = src_shape.c;
  }
  const bool use_zero_point = false;
  const OHWI weights_shape(dst_channels, 1, 1,
                           src_shape.c / (sparse_2x4 ? 2 : 1));
  FullyConnectedAttributes attr;
  attr.weights = MakeSyntheticTensor(weights_shape);
  attr.weights.data.resize(attr.weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias = MakeZeroTensor(Linear(dst_channels));

  FullyConnectedInt8Attributes attr_i8;
  attr_i8.weights.shape = weights_shape;
  attr_i8.weights.data.resize(attr_i8.weights.shape.DimensionsProduct() +
                              XNN_EXTRA_BYTES / sizeof(uint8_t));
  attr_i8.bias = MakeZeroTensor(Linear(dst_channels));

  OHWI scale_zp_shape(dst_channels, 1, 1, src_shape.c / src_group_size);
  auto weights_scales = MakeSyntheticTensor(scale_zp_shape);
  auto weights_zero_point = MakeSyntheticTensor(scale_zp_shape);

  OperationDef op_def;
  auto data_type = DataType::kFloat16;
  Layout layout = src_shape.b == 1 ? Layout::kHWC : Layout::kBHWC;
  auto storage_type = GetFastestStorageType(env.device().GetInfo());
  if (storage_type != TensorStorageType::kBuffer) {
    storage_type = TensorStorageType::kImageBuffer;
  }
  op_def.src_tensors.push_back({data_type, storage_type, layout});
  op_def.dst_tensors.push_back({data_type, storage_type, layout});

  const auto dst_shape = CalculateOutputShape(src_shape, attr);

  Tensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(src_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  descriptor_with_shape.SetBHWCShape(dst_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), descriptor_with_shape, &dst));

  std::cout << "Src size(HWC) - " << src_shape.h << "x" << src_shape.w << "x"
            << src_shape.c << std::endl;
  std::cout << "Dst size(HWC) - " << dst_shape.h << "x" << dst_shape.w << "x"
            << dst_shape.c << std::endl;

  double element_size = SizeOf(data_type);
  double weight_element_size = element_size;
  if (weights_type == DataType::kInt8) {
    weight_element_size = 1.0;
  } else if (weights_type == DataType::kInt4) {
    weight_element_size = 0.5;
    if (sparse_2x4) {
      weight_element_size = 0.75;
    }
  } else if (weights_type == DataType::kInt2) {
    weight_element_size = 0.25;
  }
  const int64_t flops_per_element = src_shape.c * 2;
  const int64_t dst_elements = dst.Width() * dst.Height() * dst.Channels();
  const int64_t flops_count = dst_elements * flops_per_element;
  const double gflops_count = flops_count * 1e-9;

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const int64_t dst_elements_alignedx4 =
      dst.Width() * dst.Height() * dst.Slices() * 4;
  const int64_t src_elements_alignedx4 =
      src.Width() * src.Height() * src.Slices() * 4;
  const double dst_gbytes = dst_elements_alignedx4 * element_size / kGByte;
  const double src_gbytes = src_elements_alignedx4 * element_size / kGByte;
  const double weight_gbytes =
      weights_shape.DimensionsProduct() * weight_element_size / kGByte;
  const double scale_gbytes =
      weights_scales.shape.DimensionsProduct() * element_size / kGByte;

  const GpuInfo& gpu_info = env.device().GetInfo();

  DataType type = op_def.src_tensors[0].GetDataType();
  auto scale_desc =
      ScaleOrZeroPointToTensorDesc(gpu_info, weights_scales, type);
  auto zero_point_desc =
      ScaleOrZeroPointToTensorDesc(gpu_info, weights_zero_point, type);

  ExternalWeights external_weights;
  external_weights.desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
  external_weights.desc.output_group_size = 1;
  external_weights.shape = weights_shape;
  if (SizeInBitsOf(weights_type) <= 8) {
    external_weights.scale_zp_shape = weights_scales.shape;
    external_weights.scale = &scale_desc;
    if (use_zero_point) {
      external_weights.zero_point = &zero_point_desc;
    }
  }

  std::unique_ptr<GPUOperation> conv;
  TensorDescriptor weights_desc;
  TensorDescriptor weights_indices_desc;

  const bool buffer_weights = true;

  if (weights_type == DataType::kInt8) {
    external_weights.desc.type = DataType::kUint8;
    TensorDescriptor weights_desc_buffer = GetTensorDescriptorForWeightsLayout(
        attr_i8.weights, external_weights.desc);

    if (buffer_weights) {
      weights_desc = weights_desc_buffer;
    } else {
      int width = DivideRoundUp(attr_i8.weights.shape.i, 4);
      int height = DivideRoundUp(attr_i8.weights.shape.o, 4);
      weights_desc = CreateConstantHWVec4TensorDescriptor(
          DataType::kUint32, TensorStorageType::kTexture2D, width, height,
          weights_desc_buffer.GetData().data());
      external_weights.desc.layout =
          WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4;
    }

  } else if (weights_type == DataType::kInt4) {
    if (sparse_2x4) {
      external_weights.desc.type = DataType::kUint4;
      ml_drift::Tensor<OHWI, DataType::kUint8> weights_indices;
      weights_indices.shape = weights_shape;
      weights_indices.data.resize(weights_indices.shape.DimensionsProduct() +
                                  XNN_EXTRA_BYTES / sizeof(uint8_t));

      const int elements_count = GetTotalElementsCountForLayout(
          external_weights.desc, attr_i8.weights.shape);
      TensorDescriptor weights_desc;
      {
        std::vector<uint8_t> weights_data(elements_count / 2);
        RearrangeWeightsInt8AsUint4(attr_i8.weights, external_weights.desc,
                                    absl::MakeSpan(weights_data),
                                    /*shift_value=*/8, /*pad_value=*/8u);

        weights_desc = TensorDescriptor(
            DataType::kUint32, TensorStorageType::kBuffer, Layout::kLinear);
        weights_desc.SetBHWCShape(BHWC(1, 1, 1, weights_data.size()));
        weights_desc.UploadDataRaw(absl::MakeConstSpan(weights_data));
      }

      TensorDescriptor weights_indices_desc;
      {
        WeightsDescription indices_desc = external_weights.desc;
        indices_desc.type = DataType::kUint2;
        std::vector<uint8_t> weights_indices_data(elements_count / 4);
        RearrangeWeightsUint2(weights_indices, indices_desc,
                              absl::MakeSpan(weights_indices_data));

        weights_indices_desc = TensorDescriptor(
            DataType::kUint32, TensorStorageType::kBuffer, Layout::kLinear);
        weights_indices_desc.SetBHWCShape(
            BHWC(1, 1, 1, weights_indices_data.size()));
        weights_indices_desc.UploadDataRaw(
            absl::MakeConstSpan(weights_indices_data));
      }
    } else {
      external_weights.desc.type = DataType::kUint4;
      TensorDescriptor weights_desc_buffer =
          GetTensorDescriptorForWeightsLayout(attr_i8.weights,
                                              external_weights.desc);
      if (buffer_weights) {
        weights_desc = weights_desc_buffer;
      } else {
        int width = DivideRoundUp(attr_i8.weights.shape.i, 4);
        int height = DivideRoundUp(attr_i8.weights.shape.o, 4);
        weights_desc = CreateConstantHWVec4TensorDescriptor(
            DataType::kUint16, TensorStorageType::kTexture2D, width, height,
            weights_desc_buffer.GetData().data());
        external_weights.desc.layout =
            WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4;
      }
    }
  } else if (weights_type == DataType::kInt2) {
    external_weights.desc.type = DataType::kUint2;
    TensorDescriptor weights_desc_buffer = GetTensorDescriptorForWeightsLayout(
        attr_i8.weights, external_weights.desc);

    if (buffer_weights) {
      weights_desc = weights_desc_buffer;
    } else {
      int width = DivideRoundUp(attr_i8.weights.shape.i, 4);
      int height = DivideRoundUp(attr_i8.weights.shape.o, 4);
      weights_desc = CreateConstantHWVec4TensorDescriptor(
          DataType::kUint8, TensorStorageType::kTexture2D, width, height,
          weights_desc_buffer.GetData().data());
      external_weights.desc.layout =
          WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4;
    }

  } else {
    external_weights.desc.type = data_type;
    auto weights_descs = GetTensorDescriptorsForWeightsLayout(
        attr.weights, external_weights.desc);
    weights_desc = weights_descs[0];
  }
  if (sparse_2x4) {
    conv =
        std::make_unique<FullyConnectedOI>(CreateFullyConnectedOIInt4Sparse2x4(
            gpu_info, CalculationsPrecision::kF16, op_def.src_tensors[0],
            op_def.dst_tensors[0], external_weights, /*bias=*/nullptr,
            &dst_shape));
  } else {
    conv = std::make_unique<FullyConnectedOI>(CreateFullyConnectedOI(
        gpu_info, CalculationsPrecision::kF16, op_def.src_tensors[0],
        op_def.dst_tensors[0], external_weights, /*bias=*/nullptr, &dst_shape));
  }
  ABSL_RETURN_IF_ERROR(conv->AssembleCode(gpu_info));

  Tensor weights_gpu_tensor;
  Tensor weights_indices_gpu_tensor;
  Tensor scale_gpu_tensor;
  Tensor zero_point_gpu_tensor;

  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), weights_desc, &weights_gpu_tensor));
  if (sparse_2x4) {
    ABSL_RETURN_IF_ERROR(CreateTensor(env.context(), weights_indices_desc,
                                      &weights_indices_gpu_tensor));
  }
  if (SizeInBitsOf(weights_type) <= 8) {
    ABSL_RETURN_IF_ERROR(
        CreateTensor(env.context(), scale_desc, &scale_gpu_tensor));
    if (use_zero_point) {
      ABSL_RETURN_IF_ERROR(
          CreateTensor(env.context(), zero_point_desc, &zero_point_gpu_tensor));
    }
  }

  ClOperation cl_op;
  cl_op.Init(std::move(conv));
  ABSL_RETURN_IF_ERROR(
      cl_op.Compile(env.GetDevicePtr(), &env.context(), env.program_cache()));
  int src_id = 0;
  ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(src_id++, &src));
  ABSL_RETURN_IF_ERROR(cl_op.SetDstTensor(0, &dst));
  ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(src_id++, &weights_gpu_tensor));
  if (sparse_2x4) {
    ABSL_RETURN_IF_ERROR(
        cl_op.SetSrcTensor(src_id++, &weights_indices_gpu_tensor));
  }
  if (SizeInBitsOf(weights_type) <= 8) {
    ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(src_id++, &scale_gpu_tensor));
    if (use_zero_point) {
      ABSL_RETURN_IF_ERROR(
          cl_op.SetSrcTensor(src_id++, &zero_point_gpu_tensor));
    }
  }
  ABSL_RETURN_IF_ERROR(cl_op.UpdateParams());
  ABSL_RETURN_IF_ERROR(
      cl_op.Tune(TuningType::kExhaustive, gpu_info, env.profiling_queue()));

  double gbytes_read = src_gbytes + weight_gbytes;
  if (SizeInBitsOf(weights_type) <= 8) {
    gbytes_read += scale_gbytes;
    if (use_zero_point) {
      gbytes_read += scale_gbytes;
    }
  }

  absl::Duration min_duration = absl::InfiniteDuration();
  for (int i = 0; i < 10; ++i) {
    ABSL_ASSIGN_OR_RETURN(
        auto duration,
        cl_op.GetOpTime(gpu_info, env.queue(), env.profiling_queue()));
    min_duration = std::min(min_duration, duration);
    double time_ms = absl::ToDoubleMilliseconds(duration);
    const double fps = 1000.0 / time_ms;
    const double gflops_real = fps * gflops_count;
    const double gbs_read = fps * gbytes_read;
    const double gbs_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(4) << " Time - " << time_ms
              << "(ms), GFlops - " << gflops_real << ", Bandwidth - "
              << gbs_read + gbs_write << "(GB/s), Read - " << gbs_read
              << "(GB/s), Write - " << gbs_write << "(GB/s)" << std::endl;
  }

  const bool is_quantized = SizeInBitsOf(weights_type) <= 8;
  const bool kUseMultipleWeights = false;
  if (kUseMultipleWeights) {
    const int kMultiplier = 32;
    int inferences = 1000.0 / absl::ToDoubleMilliseconds(min_duration);
    inferences = AlignByN(inferences, kMultiplier);
    std::vector<Tensor> weights_tensors_multiple(kMultiplier);
    std::vector<Tensor> weights_indices_tensors_multiple(kMultiplier);
    std::vector<Tensor> scale_tensors_multiple(kMultiplier);
    std::vector<Tensor> zp_tensors_multiple(kMultiplier);
    for (int k = 0; k < kMultiplier; ++k) {
      ABSL_RETURN_IF_ERROR(CreateTensor(env.context(), weights_desc,
                                        &weights_tensors_multiple[k]));
      if (sparse_2x4) {
        ABSL_RETURN_IF_ERROR(
            CreateTensor(env.context(), weights_indices_desc,
                         &weights_indices_tensors_multiple[k]));
      }
      if (is_quantized) {
        ABSL_RETURN_IF_ERROR(CreateTensor(env.context(), scale_desc,
                                          &scale_tensors_multiple[k]));
        if (use_zero_point) {
          ABSL_RETURN_IF_ERROR(CreateTensor(env.context(), zero_point_desc,
                                            &zp_tensors_multiple[k]));
        }
      }
    }
    std::cout << "Weight total size: " << weight_gbytes * kMultiplier * 1024.0
              << " MB" << std::endl;
    for (int i = 0; i < 10; ++i) {
      const auto start = absl::Now();
      for (int j = 0; j < inferences; ++j) {
        int w_id = j % kMultiplier;
        int index = 1;
        ABSL_RETURN_IF_ERROR(
            cl_op.SetSrcTensor(index++, &weights_tensors_multiple[w_id]));
        if (sparse_2x4) {
          ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(
              index++, &weights_indices_tensors_multiple[w_id]));
        }
        if (is_quantized) {
          ABSL_RETURN_IF_ERROR(
              cl_op.SetSrcTensor(index++, &scale_tensors_multiple[w_id]));
          if (use_zero_point) {
            ABSL_RETURN_IF_ERROR(
                cl_op.SetSrcTensor(index++, &zp_tensors_multiple[w_id]));
          }
        }
        ABSL_RETURN_IF_ERROR(cl_op.UpdateParams());
        ABSL_RETURN_IF_ERROR(cl_op.AddToQueue(env.queue()));
      }
      ABSL_RETURN_IF_ERROR(env.queue()->WaitForCompletion());
      const auto end = absl::Now();
      double time_ms =
          static_cast<double>((end - start) / absl::Nanoseconds(1)) /
          inferences * 1e-6;
      const double fps = 1000.0 / time_ms;
      const double gflops_real = fps * gflops_count;
      double gbs_read = fps * gbytes_read;
      const double gbs_write = fps * dst_gbytes;
      std::cout << std::fixed << std::setprecision(4) << " Time - " << time_ms
                << "(ms), GFlops - " << gflops_real << ", Bandwidth - "
                << gbs_read + gbs_write << "(GB/s), Read - " << gbs_read
                << "(GB/s), Write - " << gbs_write << "(GB/s)" << std::endl;
    }
  }
  return absl::OkStatus();
}

absl::Status FullyConnectedOIWeightsBatchIdsPerfTest(
    CalculationsPrecision precision, DataType weights_type,
    const BHWC& src_shape, int dst_channels, int batch_size,
    int active_ids_size, OHWI scale_zp_shape) {
  Environment env;
  ABSL_RETURN_IF_ERROR(CreateEnvironment(&env));

  const bool use_zero_point = false;
  ml_drift::Tensor<OHWI, DataType::kFloat32> weights;
  weights.shape = OHWI(dst_channels, batch_size, 1, src_shape.c);
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));

  ml_drift::Tensor<OHWI, DataType::kInt8> weights_i8;
  weights_i8.shape = OHWI(dst_channels, batch_size, 1, src_shape.c);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(uint8_t));

  ml_drift::Tensor<OHWI, DataType::kFloat32> weights_scale;
  weights_scale.shape = scale_zp_shape;
  weights_scale.data.resize(weights_scale.shape.DimensionsProduct(), 1.0f);
  ml_drift::Tensor<OHWI, DataType::kFloat32> weights_zp;
  weights_zp.shape = scale_zp_shape;
  weights_zp.data.resize(weights_zp.shape.DimensionsProduct(), 0.0f);

  const GpuInfo& gpu_info = env.device().GetInfo();

  OperationDef op_def;
  auto data_type = DeduceDataTypeFromPrecision(precision);
  Layout layout = src_shape.b == 1 ? Layout::kHWC : Layout::kBHWC;
  auto storage_type = GetFastestStorageType(env.device().GetInfo());
  op_def.src_tensors.push_back({data_type, storage_type, layout});
  op_def.dst_tensors.push_back({data_type, storage_type, layout});

  auto dst_shape = src_shape;
  dst_shape.c = dst_channels;
  dst_shape.h = active_ids_size;

  Tensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(src_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  descriptor_with_shape.SetBHWCShape(dst_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), descriptor_with_shape, &dst));

  ml_drift::Tensor<BHWC, DataType::kInt32> ids_data;
  ids_data.shape = BHWC(1, 1, 1, active_ids_size);
  ids_data.data.resize(ids_data.shape.DimensionsProduct(), 0);
  for (int i = 0; i < active_ids_size; ++i) {
    ids_data.data[i] = i;  // rand() % batch_size;
  }
  TensorDescriptor ids_desc =
      TensorDescriptor(DataType::kInt32, storage_type, Layout::kHWC);
  TensorDescriptor ids_desc_with_data = ids_desc;
  ids_desc_with_data.SetBHWCShape(ids_data.shape);
  ids_desc_with_data.UploadData(ids_data);
  Tensor ids;
  ABSL_RETURN_IF_ERROR(CreateTensor(env.context(), ids_desc_with_data, &ids));

  const auto w_shape = weights.shape;

  std::cout << "Src size(HWC) - " << src_shape.h << "x" << src_shape.w << "x"
            << src_shape.c << std::endl;
  std::cout << "Dst size(HWC) - " << dst_shape.h << "x" << dst_shape.w << "x"
            << dst_shape.c << std::endl;

  double element_size = precision == CalculationsPrecision::kF32 ? 4.0 : 2.0;
  const int64_t flops_per_element = w_shape.i * 2;
  const int64_t dst_elements = dst.Width() * dst.Height() * dst.Channels();
  const int64_t flops_count = dst_elements * flops_per_element;
  const double gflops_count = flops_count * 1e-9;

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const int64_t dst_elements_alignedx4 =
      dst.Width() * dst.Height() * dst.Slices() * 4;
  const int64_t src_elements_alignedx4 =
      src.Width() * src.Height() * src.Slices() * 4;
  const double dst_gbytes = dst_elements_alignedx4 * element_size / kGByte;
  const double src_gbytes = src_elements_alignedx4 * element_size / kGByte;
  const double weight_element_size = SizeInBitsOf(weights_type) / 8.0;
  const double weight_gbytes = weights.shape.DimensionsProduct() / batch_size *
                               active_ids_size * weight_element_size / kGByte;
  const double scale_gbytes = scale_zp_shape.DimensionsProduct() / batch_size *
                              active_ids_size * element_size / kGByte;

  ExternalWeights external_weights;
  external_weights.desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
  external_weights.desc.output_group_size = 1;
  external_weights.shape = weights.shape;

  TensorDescriptor weights_desc;

  if (weights_type == DataType::kInt8) {
    external_weights.desc.type = DataType::kUint8;
    TensorDescriptor weights_desc_buffer =
        GetTensorDescriptorForWeightsLayout(weights_i8, external_weights.desc);
    int width = DivideRoundUp(weights_i8.shape.i, 4);
    int height = DivideRoundUp(weights_i8.shape.o, 4);
    weights_desc = CreateConstantHWVec4TensorDescriptor(
        DataType::kUint32, TensorStorageType::kTexture2D, width, height,
        weights_desc_buffer.GetData().data());

    external_weights.desc.layout =
        WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4;
  } else if (weights_type == DataType::kInt4) {
    external_weights.desc.type = DataType::kUint4;
    TensorDescriptor weights_desc_buffer =
        GetTensorDescriptorForWeightsLayout(weights_i8, external_weights.desc);
    int width = DivideRoundUp(weights_i8.shape.i, 4);
    int height = DivideRoundUp(weights_i8.shape.o, 4);
    weights_desc = CreateConstantHWVec4TensorDescriptor(
        DataType::kUint16, TensorStorageType::kTexture2D, width, height,
        weights_desc_buffer.GetData().data());
    external_weights.desc.layout =
        WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4;
  } else if (weights_type == DataType::kInt2) {
    external_weights.desc.type = DataType::kUint2;
    TensorDescriptor weights_desc_buffer =
        GetTensorDescriptorForWeightsLayout(weights_i8, external_weights.desc);
    int width = DivideRoundUp(weights_i8.shape.i, 4);
    int height = DivideRoundUp(weights_i8.shape.o, 4);
    weights_desc = CreateConstantHWVec4TensorDescriptor(
        DataType::kUint8, TensorStorageType::kTexture2D, width, height,
        weights_desc_buffer.GetData().data());

    external_weights.desc.layout =
        WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4;
  } else {
    external_weights.desc.type = data_type;
    auto weights_descs =
        GetTensorDescriptorsForWeightsLayout(weights, external_weights.desc);
    weights_desc = weights_descs[0];
  }

  Tensor weights_gpu_tensor;
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), weights_desc, &weights_gpu_tensor));

  TensorDescriptor scale_desc =
      ScaleOrZeroPointToTensorDesc(gpu_info, weights_scale, data_type);
  TensorDescriptor zp_desc =
      ScaleOrZeroPointToTensorDesc(gpu_info, weights_zp, data_type);

  const bool is_quantized =
      weights_type != DataType::kFloat16 && weights_type != DataType::kFloat32;

  Tensor scale_tensor;
  Tensor zp_tensor;
  if (is_quantized) {
    ABSL_RETURN_IF_ERROR(
        CreateTensor(env.context(), scale_desc, &scale_tensor));
    ABSL_RETURN_IF_ERROR(CreateTensor(env.context(), zp_desc, &zp_tensor));
  }

  if (is_quantized) {
    external_weights.scale_zp_shape = scale_zp_shape;
    external_weights.scale = &scale_desc;
    if (use_zero_point) {
      external_weights.zero_point = &zp_desc;
    }
  }

  auto operation = CreateFullyConnectedOIWeightsBatchIds(
      gpu_info, precision, op_def.src_tensors[0], ids_desc,
      op_def.dst_tensors[0], external_weights, nullptr, &dst_shape);
  std::unique_ptr<GPUOperation> conv =
      std::make_unique<FullyConnectedOI>(std::move(operation));
  ABSL_RETURN_IF_ERROR(conv->AssembleCode(gpu_info));

  ClOperation cl_op;
  cl_op.Init(std::move(conv));
  ABSL_RETURN_IF_ERROR(
      cl_op.Compile(env.GetDevicePtr(), &env.context(), env.program_cache()));
  int index = 0;
  ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(index++, &src));
  ABSL_RETURN_IF_ERROR(cl_op.SetDstTensor(0, &dst));
  ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(index++, &ids));
  ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(index++, &weights_gpu_tensor));
  if (is_quantized) {
    ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(index++, &scale_tensor));
    if (use_zero_point) {
      ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(index++, &zp_tensor));
    }
  }
  ABSL_RETURN_IF_ERROR(cl_op.UpdateParams());
  ABSL_RETURN_IF_ERROR(cl_op.Tune(
      TuningType::kExhaustive, env.device().GetInfo(), env.profiling_queue()));

  double gbytes_read = src_gbytes + weight_gbytes;
  if (is_quantized) {
    gbytes_read += scale_gbytes;
    if (use_zero_point) {
      gbytes_read += scale_gbytes;
    }
  }

  for (int i = 0; i < 10; ++i) {
    ABSL_ASSIGN_OR_RETURN(auto duration,
                          cl_op.GetOpTime(env.device().GetInfo(), env.queue(),
                                          env.profiling_queue()));
    double time_ms = absl::ToDoubleMilliseconds(duration);
    const double fps = 1000.0 / time_ms;
    const double gflops_real = fps * gflops_count;
    double gbs_read = fps * gbytes_read;
    const double gbs_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(4) << " Time - " << time_ms
              << "(ms), GFlops - " << gflops_real << ", Bandwidth - "
              << gbs_read + gbs_write << "(GB/s), Read - " << gbs_read
              << "(GB/s), Write - " << gbs_write << "(GB/s)" << std::endl;
  }

  return absl::OkStatus();
}

absl::Status DepthwiseConvPerfTest(const BHWC& src_shape, const HW& kernel_size,
                                   const HW& strides, const HW& dilation) {
  Environment env;
  ABSL_RETURN_IF_ERROR(CreateEnvironment(&env));

  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(kernel_size.h / 2, kernel_size.w / 2);
  attr.padding.appended = HW(kernel_size.h / 2, kernel_size.w / 2);
  attr.strides = strides;
  attr.dilations = dilation;
  auto& attr_weights =
      attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::kFloat32>>();
  attr_weights.shape = OHWI(1, kernel_size.h, kernel_size.w, src_shape.c);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct());
  attr.bias.shape = Linear(src_shape.c);
  attr.bias.data.resize(attr.bias.shape.DimensionsProduct());

  const auto precision = CalculationsPrecision::kF16;
  OperationDef op_def;
  auto data_type = DeduceDataTypeFromPrecision(precision);
  Layout layout = src_shape.b == 1 ? Layout::kHWC : Layout::kBHWC;
  auto storage_type = GetFastestStorageType(env.device().GetInfo());
  TensorDescriptor src_tensor_desc{data_type, storage_type, layout};
  ABSL_RETURN_IF_ERROR(src_tensor_desc.UpdateToSupportedStorageType(
      env.device().GetInfo(), src_shape));

  const auto dst_shape = CalculateOutputShape(src_shape, attr);
  TensorDescriptor dst_tensor_desc{data_type, storage_type, layout};
  ABSL_RETURN_IF_ERROR(dst_tensor_desc.UpdateToSupportedStorageType(
      env.device().GetInfo(), dst_shape));

  op_def.src_tensors.push_back(src_tensor_desc);
  op_def.dst_tensors.push_back(dst_tensor_desc);

  ABSL_RETURN_IF_ERROR(
      TestDepthwiseConvPerformance(attr, src_shape, op_def, precision, &env));

  return absl::OkStatus();
}

absl::Status QuantizationPerfTest(const BHWC& src_shape, DataType src_type,
                                  PackedType dst_type, bool calculate_sum) {
  Environment env;
  ABSL_RETURN_IF_ERROR(CreateEnvironment(&env));

  auto storage_type = GetFastestStorageType(env.device().GetInfo());
  Layout layout = src_shape.b == 1 ? Layout::kHWC : Layout::kBHWC;
  TensorDescriptor src_tensor_desc{src_type, storage_type, layout};
  TensorDescriptor dst_tensor_desc{ToSpatialTensorType(dst_type), storage_type,
                                   layout};
  TensorDescriptor& params_tensor_desc = src_tensor_desc;
  BHWC params_shape = src_shape;
  params_shape.c = calculate_sum ? 3 : 2;
  BHWC dst_shape = GetShapeForPackedType(src_shape, dst_type);

  OperationDef op_def;
  op_def.src_tensors.push_back(src_tensor_desc);
  op_def.dst_tensors.push_back(dst_tensor_desc);
  op_def.dst_tensors.push_back(params_tensor_desc);
  auto quant_op =
      CreateQuantization(op_def, dst_type, env.GetDevicePtr()->GetInfo(),
                         src_shape, calculate_sum);

  Tensor src, dst, params;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(src_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  descriptor_with_shape.SetBHWCShape(dst_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), descriptor_with_shape, &dst));
  descriptor_with_shape = op_def.dst_tensors[1];
  descriptor_with_shape.SetBHWCShape(params_shape);
  ABSL_RETURN_IF_ERROR(
      CreateTensor(env.context(), descriptor_with_shape, &params));

  std::cout << "Src size(HWC) - " << src_shape.h << "x" << src_shape.w << "x"
            << src_shape.c << std::endl;
  std::cout << "Dst size(HWC) - " << dst_shape.h << "x" << dst_shape.w << "x"
            << dst_shape.c << std::endl;

  ABSL_RETURN_IF_ERROR(quant_op->AssembleCode(env.GetDevicePtr()->GetInfo()));

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const double dst_gbytes =
      (dst.GetMemorySizeInBytes() + params.GetMemorySizeInBytes()) / kGByte;
  const double src_gbytes = src.GetMemorySizeInBytes() / kGByte;

  ClOperation cl_op;
  cl_op.Init(std::move(quant_op));
  ABSL_RETURN_IF_ERROR(
      cl_op.Compile(env.GetDevicePtr(), &env.context(), env.program_cache()));
  ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(0, &src));
  ABSL_RETURN_IF_ERROR(cl_op.SetDstTensor(0, &dst));
  ABSL_RETURN_IF_ERROR(cl_op.SetDstTensor(1, &params));
  ABSL_RETURN_IF_ERROR(cl_op.UpdateParams());
  ABSL_RETURN_IF_ERROR(cl_op.Tune(
      TuningType::kExhaustive, env.device().GetInfo(), env.profiling_queue()));

  for (int i = 0; i < 5; ++i) {
    ABSL_ASSIGN_OR_RETURN(auto duration,
                          cl_op.GetOpTime(env.device().GetInfo(), env.queue(),
                                          env.profiling_queue()));
    const auto time_ms = absl::ToDoubleMilliseconds(duration);
    const double fps = 1000.0 / time_ms;
    const double gbyte_read = fps * src_gbytes;
    const double gbyte_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(3) << "  Time - " << time_ms
              << std::setprecision(2) << "(ms), Bandwidth - "
              << gbyte_read + gbyte_write << "(GB/s), Read - " << gbyte_read
              << "(GB/s), Write - " << gbyte_write << "(GB/s)" << std::endl;
  }

  return absl::OkStatus();
}

}  // namespace cl
}  // namespace ml_drift
