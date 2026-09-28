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

#include "ml_drift/gl/testing/perf_util.h"

#include <cstdint>
#include <iomanip>
#include <ios>
#include <iostream>
#include <memory>
#include <utility>
#include <variant>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/time/time.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/flops_util.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernels/conv_constants.h"
#include "ml_drift/common/kernels/conv_generic.h"
#include "ml_drift/common/kernels/conv_wave_memory.h"
#include "ml_drift/common/kernels/depthwise_conv_tiled.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/gl/egl_environment.h"
#include "ml_drift/gl/gl_operation.h"
#include "ml_drift/gl/gl_spatial_tensor.h"
#include "ml_drift/gl/portable_gl31.h"
#include "ml_drift/gl/request_gpu_info.h"

namespace ml_drift {
namespace gl {
absl::Status TestConvolutionPerformance(const Convolution2DAttributes& attr,
                                        const BHWC& src_shape,
                                        const OperationDef& op_def,
                                        CalculationsPrecision precision,
                                        const GpuInfo& gpu_info) {
  const auto dst_shape = CalculateOutputShape(src_shape, attr);

  std::unique_ptr<GPUOperation> conv;
  if (/* DISABLES CODE */ (false)) {
    std::cout << "Using ConvWaveMemory" << std::endl;
    conv = std::make_unique<ConvWaveMemory>(
        CreateConvWaveMemory(gpu_info, op_def, precision, attr, &dst_shape));
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

  GlSpatialTensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(src_shape);
  ABSL_RETURN_IF_ERROR(CreateTensor(descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  descriptor_with_shape.SetBHWCShape(dst_shape);
  ABSL_RETURN_IF_ERROR(CreateTensor(descriptor_with_shape, &dst));

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

  ABSL_RETURN_IF_ERROR(conv->AssembleCode(gpu_info));

  const int float_size = precision == CalculationsPrecision::kF32 ? 4 : 2;
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

  GlOperation gl_op;
  gl_op.Init(std::move(conv));
  ABSL_RETURN_IF_ERROR(gl_op.InitArgs(gpu_info));
  ABSL_RETURN_IF_ERROR(gl_op.SetSrcTensor(&src, 0));
  ABSL_RETURN_IF_ERROR(gl_op.SetDstTensor(&dst, 0));
  ABSL_RETURN_IF_ERROR(gl_op.Assemble(gpu_info));

  for (int i = 0; i < 5; ++i) {
    ABSL_ASSIGN_OR_RETURN(auto duration, gl_op.GetOperationTime());
    const double time_ms = absl::ToDoubleMilliseconds(duration);
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
  GLuint fb;
  ml_drift::GpuInfo gpu_info;
  std::unique_ptr<ml_drift::gl::EglEnvironment> egl_env;

  ABSL_RETURN_IF_ERROR(
      ml_drift::gl::EglEnvironment::NewEglEnvironment(&egl_env));

  glGenFramebuffers(1, &fb);
  glBindFramebuffer(GL_FRAMEBUFFER, fb);
  ABSL_RETURN_IF_ERROR(ml_drift::gl::RequestGpuInfo(&gpu_info));

  Convolution2DAttributes attr;
  attr.padding.prepended = HW(kernel_size.h / 2, kernel_size.w / 2);
  attr.padding.appended = HW(kernel_size.h / 2, kernel_size.w / 2);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto& attr_weights = attr.weights.emplace<Tensor<OHWI, DataType::kFloat32>>();
  attr_weights.shape =
      OHWI(dst_channels, kernel_size.h, kernel_size.w, src_shape.c);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct() +
                           XNN_EXTRA_BYTES / sizeof(float));
  attr.bias.shape = Linear(dst_channels);
  attr.bias.data.resize(attr.bias.shape.DimensionsProduct());

  OperationDef op_def;
  auto data_type = DeduceDataTypeFromPrecision(precision);
  Layout layout = src_shape.b == 1 ? Layout::kHWC : Layout::kBHWC;
  auto storage_type = TensorStorageType::kTexture2D;
  TensorDescriptor src_tensor_desc{data_type, storage_type, layout};
  ABSL_RETURN_IF_ERROR(
      src_tensor_desc.UpdateToSupportedStorageType(gpu_info, src_shape));

  const auto dst_shape = CalculateOutputShape(src_shape, attr);
  TensorDescriptor dst_tensor_desc{data_type, storage_type, layout};
  ABSL_RETURN_IF_ERROR(
      dst_tensor_desc.UpdateToSupportedStorageType(gpu_info, dst_shape));

  op_def.src_tensors.push_back(src_tensor_desc);
  op_def.dst_tensors.push_back(dst_tensor_desc);

  ABSL_RETURN_IF_ERROR(
      TestConvolutionPerformance(attr, src_shape, op_def, precision, gpu_info));

  return absl::OkStatus();
}

absl::Status TestDepthwiseConvPerformance(
    const DepthwiseConvolution2DAttributes& attr, const BHWC& src_shape,
    const OperationDef& op_def, CalculationsPrecision precision,
    const GpuInfo& gpu_info) {
  const auto dst_shape = CalculateOutputShape(src_shape, attr);

  std::unique_ptr<GPUOperation> conv =
      CreateDepthwiseConvTiled(gpu_info, op_def, precision, attr);

  GlSpatialTensor src, dst;
  TensorDescriptor descriptor_with_shape = op_def.src_tensors[0];
  descriptor_with_shape.SetBHWCShape(src_shape);
  ABSL_RETURN_IF_ERROR(CreateTensor(descriptor_with_shape, &src));
  descriptor_with_shape = op_def.dst_tensors[0];
  descriptor_with_shape.SetBHWCShape(dst_shape);
  ABSL_RETURN_IF_ERROR(CreateTensor(descriptor_with_shape, &dst));

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

  GlOperation gl_op;
  gl_op.Init(std::move(conv));
  ABSL_RETURN_IF_ERROR(gl_op.InitArgs(gpu_info));
  ABSL_RETURN_IF_ERROR(gl_op.SetSrcTensor(&src, 0));
  ABSL_RETURN_IF_ERROR(gl_op.SetDstTensor(&dst, 0));
  ABSL_RETURN_IF_ERROR(gl_op.Assemble(gpu_info));

  for (int i = 0; i < 5; ++i) {
    ABSL_ASSIGN_OR_RETURN(auto duration, gl_op.GetOperationTime());
    const double time_ms = absl::ToDoubleMilliseconds(duration);
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

absl::Status DepthwiseConvPerfTest(CalculationsPrecision precision,
                                   const BHWC& src_shape, const HW& kernel_size,
                                   const HW& strides, const HW& dilation) {
  GLuint fb;
  ml_drift::GpuInfo gpu_info;
  std::unique_ptr<ml_drift::gl::EglEnvironment> egl_env;

  ABSL_RETURN_IF_ERROR(
      ml_drift::gl::EglEnvironment::NewEglEnvironment(&egl_env));

  glGenFramebuffers(1, &fb);
  glBindFramebuffer(GL_FRAMEBUFFER, fb);
  ABSL_RETURN_IF_ERROR(ml_drift::gl::RequestGpuInfo(&gpu_info));

  DepthwiseConvolution2DAttributes attr;
  attr.padding.prepended = HW(kernel_size.h / 2, kernel_size.w / 2);
  attr.padding.appended = HW(kernel_size.h / 2, kernel_size.w / 2);
  attr.strides = strides;
  attr.dilations = dilation;
  auto& attr_weights = attr.weights.emplace<Tensor<OHWI, DataType::kFloat32>>();
  attr_weights.shape = OHWI(1, kernel_size.h, kernel_size.w, src_shape.c);
  attr_weights.data.resize(attr_weights.shape.DimensionsProduct());
  attr.bias.shape = Linear(src_shape.c);
  attr.bias.data.resize(attr.bias.shape.DimensionsProduct());

  OperationDef op_def;
  auto data_type = DeduceDataTypeFromPrecision(precision);
  Layout layout = src_shape.b == 1 ? Layout::kHWC : Layout::kBHWC;
  auto storage_type = TensorStorageType::kTexture2D;
  TensorDescriptor src_tensor_desc{data_type, storage_type, layout};
  ABSL_RETURN_IF_ERROR(
      src_tensor_desc.UpdateToSupportedStorageType(gpu_info, src_shape));

  const auto dst_shape = CalculateOutputShape(src_shape, attr);
  TensorDescriptor dst_tensor_desc{data_type, storage_type, layout};
  ABSL_RETURN_IF_ERROR(
      dst_tensor_desc.UpdateToSupportedStorageType(gpu_info, dst_shape));

  op_def.src_tensors.push_back(src_tensor_desc);
  op_def.dst_tensors.push_back(dst_tensor_desc);

  ABSL_RETURN_IF_ERROR(TestDepthwiseConvPerformance(attr, src_shape, op_def,
                                                    precision, gpu_info));

  return absl::OkStatus();
}
}  // namespace gl
}  // namespace ml_drift
