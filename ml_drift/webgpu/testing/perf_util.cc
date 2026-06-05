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

#include "ml_drift/webgpu/testing/perf_util.h"

#include <cstdint>
#include <iomanip>
#include <ios>
#include <iostream>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/time/time.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernels/conv_constants.h"
#include "ml_drift/common/kernels/conv_generic.h"
#include "ml_drift/common/kernels/conv_wave_matrix.h"
#include "ml_drift/common/kernels/conv_wave_memory.h"
#include "ml_drift/common/kernels/fully_connected.h"
#include "ml_drift/common/kernels/quantize_and_dequantize.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/util.h"
#include "ml_drift/webgpu/compute_task.h"
#include "ml_drift/webgpu/environment.h"
#include "ml_drift/webgpu/execution_environment.h"
#include "ml_drift/webgpu/spatial_tensor.h"
#include "ml_drift/webgpu/webgpu_headers.h"

namespace ml_drift {
namespace webgpu {
namespace {
wgpu::BackendType GetBackendType() {
  wgpu::BackendType backend_type = wgpu::BackendType::Vulkan;
#ifdef __APPLE__
  backend_type = wgpu::BackendType::Metal;
#elif defined(_WIN32)
  backend_type = wgpu::BackendType::DX12;
#endif
  return backend_type;
}

}  // namespace
absl::Status TestConvolutionPerformance(const Convolution2DAttributes& attr,
                                        const BHWC& src_shape,
                                        const OperationDef& op_def,
                                        CalculationsPrecision precision,
                                        ExecutionEnvironment* env) {
  const GpuInfo& gpu_info = env->GetInfo();
  const auto dst_shape = CalculateOutputShape(src_shape, attr);

  std::unique_ptr<GPUOperation> conv;
  if (SupportsConvWaveMatrix(gpu_info, precision, attr)) {
    std::cout << "Using ConvWaveMatrix" << std::endl;
    conv = std::make_unique<ConvWaveMatrix>(
        CreateConvWaveMatrix(op_def, precision, dst_shape, attr, gpu_info));
  } else if (/* DISABLES CODE */ (true)) {
    std::cout << "Using ConvGeneric" << std::endl;
    conv = std::make_unique<ConvGeneric>(
        CreateConvGeneric(gpu_info, op_def, precision, attr, &dst_shape));
  } else {
    std::cout << "Using ConvConstants" << std::endl;
    conv = std::make_unique<ConvConstants>(
        CreateConvConstants(gpu_info, op_def, precision, attr));
  }

  //{
  //  FullyConnectedAttributes fc_attr;
  //  fc_attr.weights = attr.weights;
  //  fc_attr.bias = attr.bias;
  //  conv = std::make_unique<FullyConnected>(CreateFullyConnected(
  //      gpu_info, op_def, precision, fc_attr, &dst_shape));
  //}

  SpatialTensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(src_shape);
  RETURN_IF_ERROR(CreateTensor(env->device(), descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  descriptor_with_shape.SetBHWCShape(dst_shape);
  RETURN_IF_ERROR(CreateTensor(env->device(), descriptor_with_shape, &dst));

  const auto w_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);

  std::cout << "Src size(HWC) - " << src_shape.h << "x" << src_shape.w << "x"
            << src_shape.c << std::endl;
  std::cout << "Dst size(HWC) - " << dst_shape.h << "x" << dst_shape.w << "x"
            << dst_shape.c << std::endl;
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

  RETURN_IF_ERROR(conv->AssembleCode(gpu_info));

  const int float_size = precision == CalculationsPrecision::F32 ? 4 : 2;
  const int64_t flops_per_element = w_shape.i * w_shape.h * w_shape.w * 2;
  const int64_t dst_elements = dst.Width() * dst.Height() * dst.Channels();
  const int64_t flops_count = dst_elements * flops_per_element;
  const double gflops_count = flops_count * 1e-9;

  const double kGByte = 1024.0 * 1024.0 * 1024.0;
  const int64_t dst_elements_alignedx4 =
      dst.Width() * dst.Height() * dst.Slices() * 4;
  const int64_t src_elements_alignedx4 =
      src.Width() * src.Height() * src.Slices() * 4;
  const double dst_gbytes = dst_elements_alignedx4 * float_size / kGByte;
  const double src_gbytes = src_elements_alignedx4 * float_size / kGByte;
  const double weight_gbytes =
      w_shape.DimensionsProduct() * float_size / kGByte;
  const double bias_gbytes = w_shape.o * float_size / kGByte;

  ComputeTask webgpu_op;
  RETURN_IF_ERROR(webgpu_op.Init(*env, std::move(conv)));
  RETURN_IF_ERROR(webgpu_op.SetSrcTensor(0, &src));
  RETURN_IF_ERROR(webgpu_op.SetDstTensor(0, &dst));
  RETURN_IF_ERROR(webgpu_op.Update(env->device()));
  webgpu_op.UpdateGpuObjectBindings(env->device());
  RETURN_IF_ERROR(webgpu_op.Compile(*env));

  for (int i = 0; i < 5; ++i) {
    auto duration = webgpu_op.GetOperationTime(*env);
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

absl::Status ConvolutionPerfTest(CalculationsPrecision precision,
                                 const BHWC& src_shape, int dst_channels,
                                 const HW& kernel_size) {
  ExecutionEnvironment env(GetBackendType());
  RETURN_IF_ERROR(env.Initialize());
  const GpuInfo& gpu_info = env.GetInfo();
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(kernel_size.h / 2, kernel_size.w / 2);
  attr.padding.appended = HW(kernel_size.h / 2, kernel_size.w / 2);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto& attr_weights = attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape =
      OHWI(dst_channels, kernel_size.h, kernel_size.w, src_shape.c);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias.shape = Linear(dst_channels);
  attr.bias.data.resize(attr.bias.shape.DimensionsProduct());

  OperationDef op_def;
  auto data_type = DeduceDataTypeFromPrecision(precision);
  Layout layout = src_shape.b == 1 ? Layout::HWC : Layout::BHWC;
  auto storage_type = TensorStorageType::BUFFER;
  TensorDescriptor src_tensor_desc{data_type, storage_type, layout};
  RETURN_IF_ERROR(
      src_tensor_desc.UpdateToSupportedStorageType(gpu_info, src_shape));

  const auto dst_shape = CalculateOutputShape(src_shape, attr);
  TensorDescriptor dst_tensor_desc{data_type, storage_type, layout};
  RETURN_IF_ERROR(
      dst_tensor_desc.UpdateToSupportedStorageType(gpu_info, dst_shape));

  op_def.src_tensors.push_back(src_tensor_desc);
  op_def.dst_tensors.push_back(dst_tensor_desc);

  RETURN_IF_ERROR(
      TestConvolutionPerformance(attr, src_shape, op_def, precision, &env));

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
  ExecutionEnvironment env(GetBackendType());
  RETURN_IF_ERROR(env.Initialize());
  const GpuInfo& gpu_info = env.GetInfo();

  BHWC dst_shape;
  {
    Convolution2DAttributes attr;
    attr.padding.prepended = HW(0, 0);
    attr.padding.appended = HW(0, 0);
    attr.strides = HW(1, 1);
    attr.dilations = HW(1, 1);
    auto& attr_weights =
        attr.weights.emplace<Tensor<OHWI, DataType::FLOAT32>>();
    attr_weights.shape = OHWI(dst_channels, 1, 1, src_shape.c);
    attr.bias.shape = Linear(dst_channels);
    dst_shape = CalculateOutputShape(src_shape, attr);
  }

  const bool dequantize = true;
  const auto precision = CalculationsPrecision::F16;
  OperationDef op_def;
  auto dst_data_type =
      UseUint8MathForInt8Weights(gpu_info) ? DataType::UINT32 : DataType::INT32;
  Layout layout = src_shape.b == 1 ? Layout::HWC : Layout::BHWC;
  auto src_storage_type = GetFastestStorageType(gpu_info);
  auto dst_storage_type = src_storage_type;

  TensorDescriptor dst_tensor_desc{dst_data_type, dst_storage_type, layout};
  RETURN_IF_ERROR(
      dst_tensor_desc.UpdateToSupportedStorageType(gpu_info, dst_shape));

  op_def.dst_tensors.push_back(dst_tensor_desc);

  ml_drift::Tensor<OHWI, DataType::INT8> weights;
  weights.shape = OHWI(dst_channels, 1, 1, src_shape.c);
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(int8_t));
  ml_drift::Tensor<Linear, DataType::FLOAT32> weights_scale;
  weights_scale.shape = Linear(dst_channels);
  weights_scale.data.resize(weights_scale.shape.DimensionsProduct());
  ml_drift::Tensor<Linear, DataType::FLOAT32> weights_zero_point;
  weights_zero_point.shape = Linear(dst_channels);
  weights_zero_point.data.resize(weights_zero_point.shape.DimensionsProduct());

  PackedType quantized_type = PackedType::kUnknown;
  std::unique_ptr<GPUOperation> conv;
  if (SupportsConvWaveMatrixInt8(gpu_info, weights.shape)) {
    std::cout << "ConvWaveMatrix" << std::endl;
    quantized_type = GetConvWaveMatrixInt8SrcType();
    op_def.src_tensors.push_back(GetTensorDescriptor(
        quantized_type, src_storage_type, layout, src_shape, gpu_info));
    conv = std::make_unique<ConvWaveMatrix>(
        CreateConvWaveMatrixInt8(op_def, dst_shape, weights, gpu_info));
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
        TensorDescriptor{DataType::FLOAT32, src_storage_type, layout};
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
    RETURN_IF_ERROR(conv->AddOperation(gpu_info, &dequant_op));
    dst_tensor_desc = dequant_dst;
  }

  SpatialTensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(
      GetShapeForPackedType(src_shape, quantized_type));
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &src));
  descriptor_with_shape = dst_tensor_desc;
  descriptor_with_shape.SetBHWCShape(dst_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &dst));

  SpatialTensor src_params_tensor;
  SpatialTensor weights_scale_tensor;
  SpatialTensor weights_zero_point_tensor;
  SpatialTensor weights_sum_i_tensor;
  if (dequantize) {
    RETURN_IF_ERROR(
        CreateTensor(env.device(), src_params_td, &src_params_tensor));
    RETURN_IF_ERROR(
        CreateTensor(env.device(), weights_scale_td, &weights_scale_tensor));
    RETURN_IF_ERROR(CreateTensor(env.device(), weights_zero_point_td,
                                 &weights_zero_point_tensor));
    RETURN_IF_ERROR(
        CreateTensor(env.device(), weights_sum_i_td, &weights_sum_i_tensor));
  }

  const auto w_shape = weights.shape;

  std::cout << "Src size(HWC) - " << src_shape.h << "x" << src_shape.w << "x"
            << src_shape.c << std::endl;
  std::cout << "Dst size(HWC) - " << dst_shape.h << "x" << dst_shape.w << "x"
            << dst_shape.c << std::endl;

  RETURN_IF_ERROR(conv->AssembleCode(gpu_info));

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

  ComputeTask webgpu_op;
  RETURN_IF_ERROR(webgpu_op.Init(env, std::move(conv)));
  RETURN_IF_ERROR(webgpu_op.SetSrcTensor(0, &src));
  if (dequantize) {
    RETURN_IF_ERROR(webgpu_op.SetSrcTensor(1, &src_params_tensor));
    RETURN_IF_ERROR(webgpu_op.SetSrcTensor(2, &weights_sum_i_tensor));
    RETURN_IF_ERROR(webgpu_op.SetSrcTensor(3, &weights_scale_tensor));
    RETURN_IF_ERROR(webgpu_op.SetSrcTensor(4, &weights_zero_point_tensor));
  }
  RETURN_IF_ERROR(webgpu_op.SetDstTensor(0, &dst));
  RETURN_IF_ERROR(webgpu_op.Update(env.device()));
  webgpu_op.UpdateGpuObjectBindings(env.device());
  RETURN_IF_ERROR(webgpu_op.Compile(env));

  for (int i = 0; i < 5; ++i) {
    auto duration = webgpu_op.GetOperationTime(env);
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

absl::Status FullyConnectedPerfTest(CalculationsPrecision precision,
                                    DataType weights_type,
                                    const BHWC& src_shape, int dst_channels,
                                    OHWI scale_zp_shape) {
  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = OHWI(dst_channels, 1, 1, src_shape.c);
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));

  ml_drift::Tensor<OHWI, DataType::INT8> weights_i8;
  weights_i8.shape = OHWI(dst_channels, 1, 1, src_shape.c);
  weights_i8.data.resize(weights_i8.shape.DimensionsProduct() +
                         XNN_EXTRA_BYTES / sizeof(uint8_t));

  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_scale;
  weights_scale.shape = scale_zp_shape;
  weights_scale.data.resize(weights_scale.shape.DimensionsProduct(), 1.0f);
  ml_drift::Tensor<OHWI, DataType::FLOAT32> weights_zp;
  weights_zp.shape = scale_zp_shape;
  weights_zp.data.resize(weights_zp.shape.DimensionsProduct(), 0.0f);

  ExecutionEnvironment env(GetBackendType());
  RETURN_IF_ERROR(env.Initialize());
  const GpuInfo& gpu_info = env.GetInfo();

  OperationDef op_def;
  auto data_type = DeduceDataTypeFromPrecision(precision);
  Layout layout = src_shape.b == 1 ? Layout::HWC : Layout::BHWC;
  auto storage_type = GetFastestStorageType(gpu_info);
  op_def.src_tensors.push_back({data_type, storage_type, layout});
  op_def.dst_tensors.push_back({data_type, storage_type, layout});

  auto dst_shape = src_shape;
  dst_shape.c = dst_channels;

  SpatialTensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(src_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  descriptor_with_shape.SetBHWCShape(dst_shape);
  RETURN_IF_ERROR(CreateTensor(env.device(), descriptor_with_shape, &dst));

  const auto w_shape = weights.shape;

  std::cout << "Src size(HWC) - " << src_shape.h << "x" << src_shape.w << "x"
            << src_shape.c << std::endl;
  std::cout << "Dst size(HWC) - " << dst_shape.h << "x" << dst_shape.w << "x"
            << dst_shape.c << std::endl;

  double element_size = precision == CalculationsPrecision::F32 ? 4.0 : 2.0;
  double weight_element_size = element_size;
  if (weights_type == DataType::INT8) {
    weight_element_size = 1.0;
  } else if (weights_type == DataType::INT4) {
    weight_element_size = 0.5;
  } else if (weights_type == DataType::INT2) {
    weight_element_size = 0.25;
  }
  const int64_t flops_per_element = w_shape.i * w_shape.h * w_shape.w * 2;
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
      weights.shape.DimensionsProduct() * weight_element_size / kGByte;
  const double scale_gbytes =
      scale_zp_shape.DimensionsProduct() * element_size / kGByte;

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
    weights_gpu.push_back(
        GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc));
  } else if (weights_type == DataType::INT4) {
    weights_desc = GetFullyConnectedInt4WeightsDesc(gpu_info, weights_i8.shape);
    weights_gpu.push_back(
        GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc));
  } else if (weights_type == DataType::INT2) {
    weights_desc = GetFullyConnectedInt2WeightsDesc(gpu_info, weights_i8.shape);
    weights_gpu.push_back(
        GetTensorDescriptorForWeightsLayout(weights_i8, weights_desc));
  }

  std::vector<SpatialTensor> weights_tensors(weights_gpu.size());
  for (int i = 0; i < weights_gpu.size(); ++i) {
    RETURN_IF_ERROR(
        CreateTensor(env.device(), weights_gpu[i], &weights_tensors[i]));
  }

  TensorDescriptor scale_desc =
      ScaleOrZeroPointToFCTensorDesc(gpu_info, weights_scale, data_type);
  TensorDescriptor zp_desc =
      ScaleOrZeroPointToFCTensorDesc(gpu_info, weights_zp, data_type);

  const bool is_qunatized =
      weights_type != DataType::FLOAT16 && weights_type != DataType::FLOAT32;

  SpatialTensor scale_tensor;
  SpatialTensor zp_tensor;
  if (is_qunatized) {
    RETURN_IF_ERROR(CreateTensor(env.device(), scale_desc, &scale_tensor));
    RETURN_IF_ERROR(CreateTensor(env.device(), zp_desc, &zp_tensor));
  }

  ExternalWeights external_weights;
  external_weights.desc = weights_desc;
  external_weights.shape = weights.shape;
  if (is_qunatized) {
    external_weights.scale_zp_shape = scale_zp_shape;
    external_weights.scale = &scale_desc;
    external_weights.zero_point = &zp_desc;
  }

  ASSIGN_OR_RETURN(
      auto operation,
      CreateFullyConnectedExternalWeights(
          gpu_info, precision, op_def.src_tensors[0], op_def.dst_tensors[0],
          external_weights, /*bias=*/nullptr, &dst_shape));

  std::unique_ptr<GPUOperation> conv =
      std::make_unique<FullyConnected>(std::move(operation));
  RETURN_IF_ERROR(conv->AssembleCode(gpu_info));

  ComputeTask webgpu_op;
  RETURN_IF_ERROR(webgpu_op.Init(env, std::move(conv)));
  RETURN_IF_ERROR(webgpu_op.SetSrcTensor(0, &src));
  int index = 1;
  for (int i = 0; i < weights_tensors.size(); ++i) {
    RETURN_IF_ERROR(webgpu_op.SetSrcTensor(index++, &weights_tensors[i]));
  }
  if (is_qunatized) {
    RETURN_IF_ERROR(webgpu_op.SetSrcTensor(index++, &scale_tensor));
    RETURN_IF_ERROR(webgpu_op.SetSrcTensor(index++, &zp_tensor));
  }
  RETURN_IF_ERROR(webgpu_op.SetDstTensor(0, &dst));
  RETURN_IF_ERROR(webgpu_op.Update(env.device()));
  webgpu_op.UpdateGpuObjectBindings(env.device());
  RETURN_IF_ERROR(webgpu_op.Compile(env));

  for (int i = 0; i < 10; ++i) {
    auto duration = webgpu_op.GetOperationTime(env);
    double time_ms = absl::ToDoubleMilliseconds(duration);
    const double fps = 1000.0 / time_ms;
    const double gflops_real = fps * gflops_count;
    double gbytes_read = src_gbytes + weight_gbytes;
    if (weights_type != DataType::FLOAT16 &&
        weights_type != DataType::FLOAT32) {
      gbytes_read += scale_gbytes * 2.0;
    }
    double gbs_read = fps * gbytes_read;
    const double gbs_write = fps * dst_gbytes;
    std::cout << std::fixed << std::setprecision(4) << " Time - " << time_ms
              << "(ms), GFlops - " << gflops_real << ", Bandwidth - "
              << gbs_read + gbs_write << "(GB/s), Read - " << gbs_read
              << "(GB/s), Write - " << gbs_write << "(GB/s)" << std::endl;
  }

  return absl::OkStatus();
}

}  // namespace webgpu
}  // namespace ml_drift
