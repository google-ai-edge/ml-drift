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

#include "ml_drift/common/task/tensor_desc.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "ml_drift/common/access_type.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/util.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace {
std::string GetReadImageFromDataType(DataType data_type) {
  if (data_type == DataType::FLOAT32) {
    return "read_imagef";
  } else if (data_type == DataType::FLOAT16) {
    return "read_imageh";
  } else if (data_type == DataType::INT8 || data_type == DataType::INT16 ||
             data_type == DataType::INT32) {
    return "read_imagei";
  } else if (data_type == DataType::UINT8 || data_type == DataType::UINT16 ||
             data_type == DataType::UINT32 || data_type == DataType::BFLOAT16 ||
             data_type == DataType::BOOL) {
    return "read_imageui";
  } else {
    return "error";
  }
}

DataType ToClTextureType(DataType data_type) {
  switch (data_type) {
    case DataType::FLOAT32:
    case DataType::FLOAT16:
    case DataType::INT32:
    case DataType::UINT32:
      return data_type;
    case DataType::INT16:
    case DataType::INT8:
      return DataType::INT32;
    case DataType::BFLOAT16:
    case DataType::BOOL:
    case DataType::UINT16:
    case DataType::UINT8:
      return DataType::UINT32;
    default:
      return DataType::UNKNOWN;
  }
}

DataType ToWebGpuTextureType(DataType data_type) {
  switch (data_type) {
    case DataType::FLOAT32:
    case DataType::FLOAT16:
      return DataType::FLOAT32;
    case DataType::INT32:
    case DataType::INT16:
    case DataType::INT8:
      return DataType::INT32;
    case DataType::BFLOAT16:
    case DataType::UINT32:
    case DataType::UINT16:
    case DataType::UINT8:
    case DataType::BOOL:
      return DataType::UINT32;
    default:
      return DataType::UNKNOWN;
  }
}

DataType ToCudaTextureType(DataType data_type) {
  switch (data_type) {
    case DataType::FLOAT32:
    case DataType::FLOAT16:
      return DataType::FLOAT32;
    case DataType::INT32:
    case DataType::INT16:
    case DataType::INT8:
      return DataType::INT32;
    case DataType::UINT32:
    case DataType::UINT16:
    case DataType::UINT8:
    case DataType::BOOL:
      return DataType::UINT32;
    default:
      return DataType::UNKNOWN;
  }
}

std::string GetWriteImageFromDataType(DataType data_type) {
  if (data_type == DataType::FLOAT32) {
    return "write_imagef";
  } else if (data_type == DataType::FLOAT16) {
    return "write_imageh";
  } else if (data_type == DataType::INT8 || data_type == DataType::INT16 ||
             data_type == DataType::INT32) {
    return "write_imagei";
  } else if (data_type == DataType::UINT8 || data_type == DataType::UINT16 ||
             data_type == DataType::UINT32 || data_type == DataType::BFLOAT16 ||
             data_type == DataType::BOOL) {
    return "write_imageui";
  } else {
    return "error";
  }
}

std::string GetConversionForImage(const GpuInfo& gpu_info, DataType src_type,
                                  DataType dst_type) {
  DataType interm_type = src_type;
  if (gpu_info.IsApiOpenCl()) {
    if (src_type == DataType::FLOAT16 && dst_type == DataType::FLOAT32) {
      return "$0";
    }
    interm_type = ToClTextureType(src_type);
  } else if (gpu_info.IsApiMetal()) {
    interm_type = ToMetalTextureType(src_type);
  } else if (gpu_info.IsApiWebGpu()) {
    interm_type = ToWebGpuTextureType(src_type);
  } else if (gpu_info.IsApiCuda()) {
    interm_type = ToCudaTextureType(src_type);
  }
  if (src_type == DataType::BFLOAT16) {
    const std::string bfloat_val =
        interm_type == DataType::UINT16
            ? "$0"
            : "ucl::Convert<" + ToUclDataType(DataType::UINT16, 4) + ">($0)";
    if (gpu_info.IsApiMetal() &&
        gpu_info.metal_info.IsNativeBfloatSupported()) {
      return "ucl::Reinterpret<ushort4, bfloat4>(" + bfloat_val + ")";
    } else {
      return "ucl::ConvertFromBfloat<" + ToUclDataType(dst_type, 4) + ">(" +
             bfloat_val + ")";
    }
  }
  if (dst_type != interm_type) {
    return "ucl::Convert<" + ToUclDataType(dst_type, 4) + ">($0)";
  } else {
    return "$0";
  }
}

std::string GetConversion(const GpuInfo& gpu_info,
                          TensorStorageType storage_type, DataType src_type,
                          DataType dst_type) {
  if (src_type == DataType::BOOL) {
    // DataType::BOOL stored as DataType::UINT8
    src_type = DataType::UINT8;
  }
  if (storage_type == TensorStorageType::BUFFER) {
    if (src_type == DataType::BFLOAT16) {
      if (gpu_info.IsApiMetal() &&
          gpu_info.metal_info.IsNativeBfloatSupported()) {
        return "$0";
      } else {
        return "ucl::ConvertFromBfloat<" + ToUclDataType(dst_type, 4) + ">($0)";
      }
    }
    if (dst_type != src_type) {
      return "ucl::Convert<" + ToUclDataType(dst_type, 4) + ">($0)";
    } else {
      return "$0";
    }
  } else {
    return GetConversionForImage(gpu_info, src_type, dst_type);
  }
}

void MayBeAddConversion(const std::string& conversion, std::string* result) {
  *result = absl::Substitute(conversion, *result);
}

bool IsGlslBufferSupportedType(const GpuInfo& gpu_info, DataType data_type) {
  // Check extensions for shorter integer types. Assume both int16 or int8 are
  // supported if one of them is supported because some drivers don't advertise
  // both, e.g. Nvidia Vulkan drivers.
  if (data_type == DataType::INT16 || data_type == DataType::INT8 ||
      data_type == DataType::UINT16 || data_type == DataType::UINT8 ||
      data_type == DataType::BOOL) {
    if (gpu_info.IsApiOpenGl()) {
      return gpu_info.SupportsExtension("GL_EXT_shader_8bit_storage") ||
             gpu_info.SupportsExtension("GL_EXT_shader_16bit_storage") ||
             gpu_info.SupportsExtension(
                 "GL_EXT_shader_explicit_arithmetic_types_int8") ||
             gpu_info.SupportsExtension(
                 "GL_EXT_shader_explicit_arithmetic_types_int16");
    } else if (gpu_info.IsApiVulkan()) {
      return gpu_info.SupportsExtension("VK_KHR_8bit_storage") ||
             gpu_info.SupportsExtension("VK_KHR_16bit_storage") ||
             gpu_info.SupportsExtension("VK_KHR_shader_float16_int8");
    }
    return false;
  }
  return true;
}

TensorStorageType GetRecommendedStorageTypeForLinearTensor(
    const GpuInfo& gpu_info) {
  if (gpu_info.IsApiWebGpu()) {
    return TensorStorageType::BUFFER;
  }
  if (gpu_info.IsApple() &&
      gpu_info.apple_info.IsFamilyOrLower(AppleInfo::Family::kApple2)) {
    return TensorStorageType::TEXTURE_2D;
  }
  if (!gpu_info.SupportsImages() || gpu_info.IsMali() ||
      gpu_info.IsBroadcom() || gpu_info.IsApple() || gpu_info.IsAMD()) {
    return TensorStorageType::BUFFER;
  } else {
    return TensorStorageType::TEXTURE_2D;
  }
}

}  // namespace

std::string ToString(TensorStorageType type) {
  switch (type) {
    case TensorStorageType::UNKNOWN:
      return "TensorStorageType::UNKNOWN";
    case TensorStorageType::BUFFER:
      return "TensorStorageType::BUFFER";
    case TensorStorageType::TEXTURE_ARRAY:
      return "TensorStorageType::TEXTURE_ARRAY";
    case TensorStorageType::TEXTURE_2D:
      return "TensorStorageType::TEXTURE_2D";
    case TensorStorageType::TEXTURE_3D:
      return "TensorStorageType::TEXTURE_3D";
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return "TensorStorageType::SINGLE_TEXTURE_2D";
    case TensorStorageType::IMAGE_BUFFER:
      return "TensorStorageType::IMAGE_BUFFER";
  }
}

std::string ToStringWithShape(const TensorDescriptor& desc) {
  return ToString(desc.GetDataType()) + ", " + ToString(desc.GetStorageType()) +
         ", layout: " + ToString(desc.GetLayout()) +
         ", shape: " + ToString(desc.GetBHWCShape());
}

TensorDescriptor::TensorDescriptor(TensorDescriptor&& desc)
    : GPUObjectDescriptor(std::move(static_cast<GPUObjectDescriptor&>(desc))),
      data_type_(desc.data_type_),
      storage_type_(desc.storage_type_),
      layout_(desc.layout_),
      physical_layout_1d_(desc.physical_layout_1d_),
      physical_layout_2d_(desc.physical_layout_2d_),
      use_buffer_for_write_only_2d_texture_(
          desc.use_buffer_for_write_only_2d_texture_),
      use_buffer_for_write_only_image_buffer_(
          desc.use_buffer_for_write_only_image_buffer_),
      shape_(desc.shape_),
      data_(std::move(desc.data_)) {}
TensorDescriptor& TensorDescriptor::operator=(TensorDescriptor&& desc) {
  if (this != &desc) {
    std::swap(data_type_, desc.data_type_);
    std::swap(storage_type_, desc.storage_type_);
    std::swap(layout_, desc.layout_);
    std::swap(physical_layout_1d_, desc.physical_layout_1d_);
    std::swap(physical_layout_2d_, desc.physical_layout_2d_);
    std::swap(use_buffer_for_write_only_2d_texture_,
              desc.use_buffer_for_write_only_2d_texture_);
    std::swap(use_buffer_for_write_only_image_buffer_,
              desc.use_buffer_for_write_only_image_buffer_);
    std::swap(shape_, desc.shape_);
    data_ = std::move(desc.data_);
    GPUObjectDescriptor::operator=(std::move(desc));
  }
  return *this;
}

void TensorDescriptor::CopyWithoutData(TensorDescriptor* desc) const {
  desc->data_type_ = data_type_;
  desc->storage_type_ = storage_type_;
  desc->layout_ = layout_;
  desc->physical_layout_1d_ = physical_layout_1d_;
  desc->physical_layout_2d_ = physical_layout_2d_;
  desc->use_buffer_for_write_only_2d_texture_ =
      use_buffer_for_write_only_2d_texture_;
  desc->use_buffer_for_write_only_image_buffer_ =
      use_buffer_for_write_only_image_buffer_;
  desc->shape_ = shape_;
}

std::vector<uint64_t> TensorDescriptor::GetStorageDims(
    const BHWDC& shape) const {
  const int slices = DivideRoundUp(shape.c, 4);
  if (layout_ == Layout::LINEAR) {
    switch (storage_type_) {
      case TensorStorageType::BUFFER:
      case TensorStorageType::IMAGE_BUFFER:
        return {static_cast<uint64_t>(slices)};
      case TensorStorageType::TEXTURE_ARRAY:
      case TensorStorageType::TEXTURE_3D:
        return {static_cast<uint64_t>(slices), 1u, 1u};
      case TensorStorageType::TEXTURE_2D:
      case TensorStorageType::SINGLE_TEXTURE_2D:
        return {static_cast<uint64_t>(slices), 1u};
      case TensorStorageType::UNKNOWN:
        return {};
    }
  } else if (layout_ == Layout::HW) {
    switch (storage_type_) {
      case TensorStorageType::BUFFER:
      case TensorStorageType::IMAGE_BUFFER:
        return {static_cast<uint64_t>(shape.w * shape.h)};
      case TensorStorageType::TEXTURE_ARRAY:
      case TensorStorageType::TEXTURE_3D:
        return {static_cast<uint64_t>(shape.w), static_cast<uint64_t>(shape.h),
                1u};
      case TensorStorageType::TEXTURE_2D:
      case TensorStorageType::SINGLE_TEXTURE_2D:
        return {static_cast<uint64_t>(shape.w), static_cast<uint64_t>(shape.h)};
      case TensorStorageType::UNKNOWN:
        return {};
    }
  }
  // HWC/BHWC/HWDC/BHWDC
  switch (storage_type_) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return {static_cast<uint64_t>(shape.w * shape.b * shape.h * shape.d *
                                    slices)};
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return {static_cast<uint64_t>(shape.w * shape.b),
              static_cast<uint64_t>(shape.h),
              static_cast<uint64_t>(shape.d * slices)};
    case TensorStorageType::TEXTURE_2D:
      if (physical_layout_2d_ == PhysicalLayout2D::kXisWBDC4YisHC) {
        return {static_cast<uint64_t>(shape.w * shape.b * shape.d),
                static_cast<uint64_t>(shape.h * slices)};
      } else if (physical_layout_2d_ == PhysicalLayout2D::kXisCC4YisDHWB) {
        return {static_cast<uint64_t>(slices),
                static_cast<uint64_t>(shape.w * shape.h * shape.d * shape.b)};
      } else {
        return {0, 0};
      }
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return {static_cast<uint64_t>(shape.w * shape.b * shape.d),
              static_cast<uint64_t>(shape.h)};
    case TensorStorageType::UNKNOWN:
      return {};
  }
}

std::vector<uint64_t> TensorDescriptor::GetStorageDims() const {
  return GetStorageDims(shape_);
}

int3 TensorDescriptor::GetFullTensorRegion() const {
  std::vector<uint64_t> storage_dims = GetStorageDims();
  if (layout_ == Layout::LINEAR) {
    return int3(static_cast<int>(storage_dims[0]), 1, 1);
  } else if (layout_ == Layout::HW) {
    switch (storage_type_) {
      case TensorStorageType::BUFFER:
      case TensorStorageType::IMAGE_BUFFER:
        return int3(static_cast<int>(storage_dims[0]), 1, 1);
      case TensorStorageType::TEXTURE_2D:
      case TensorStorageType::SINGLE_TEXTURE_2D:
      case TensorStorageType::TEXTURE_ARRAY:
      case TensorStorageType::TEXTURE_3D:
        return int3(static_cast<int>(storage_dims[0]),
                    static_cast<int>(storage_dims[1]), 1);
      case TensorStorageType::UNKNOWN:
        return {-1, -1, -1};
    }
  }
  // HWC/BHWC/HWDC/BHWDC
  switch (storage_type_) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      // 1D resources
      return int3(static_cast<int>(storage_dims[0]), 1, 1);
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      // 2D resources
      return int3(static_cast<int>(storage_dims[0]),
                  static_cast<int>(storage_dims[1]), 1);
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      // 3D resources
      return int3(static_cast<int>(storage_dims[0]),
                  static_cast<int>(storage_dims[1]),
                  static_cast<int>(storage_dims[2]));
    case TensorStorageType::UNKNOWN:
      return {-1, -1, -1};
  }
}
uint64_t TensorDescriptor::GetMemorySizeInBytes() const {
  std::vector<uint64_t> storage_dims = GetStorageDims();
  uint64_t total_size = 1;
  for (int i = 0; i < storage_dims.size(); ++i) {
    total_size *= storage_dims[i];
  }
  const int element_size = GetElementSize() * SizeOf(data_type_);
  return total_size * element_size;
}

int TensorDescriptor::GetElementSize() const {
  if (storage_type_ == TensorStorageType::SINGLE_TEXTURE_2D) {
    return shape_.c;
  } else {
    return 4;
  }
}

GPUResources TensorDescriptor::GetGPUResources(const GpuInfo& gpu_info) const {
  GPUResources resources;
  resources.ints.push_back("slice_stride");
  if (HasAxis(Axis::WIDTH)) {
    resources.ints.push_back("width");
  }
  if (HasAxis(Axis::HEIGHT)) {
    resources.ints.push_back("height");
  }
  if (HasAxis(Axis::CHANNELS)) {
    resources.ints.push_back("slices");
    resources.ints.push_back("channels");
  }
  if (HasAxis(Axis::BATCH)) {
    resources.ints.push_back("batch");
  }
  if (HasAxis(Axis::DEPTH)) {
    resources.ints.push_back("depth");
  }
  if (storage_type_ == TensorStorageType::BUFFER) {
    GPUBufferDescriptor desc;
    desc.data_type = data_type_;
    desc.access_type = access_type_;
    desc.element_size = 4;
    resources.buffers.push_back({"buffer", desc});
  } else if (storage_type_ == TensorStorageType::SINGLE_TEXTURE_2D ||
             storage_type_ == TensorStorageType::TEXTURE_2D) {
    if (access_type_ == AccessType::WRITE &&
        use_buffer_for_write_only_2d_texture_) {
      resources.ints.push_back("aligned_texture_width");
      GPUBufferDescriptor desc;
      desc.data_type = data_type_;
      desc.access_type = access_type_;
      desc.element_size = 4;
      resources.buffers.push_back({"buffer", desc});
    } else {
      GPUImage2DDescriptor desc;
      desc.data_type = data_type_;
      desc.normalized = false;
      desc.access_type = access_type_;
      resources.images2d.push_back({"image2d", desc});
    }
  } else if (storage_type_ == TensorStorageType::TEXTURE_ARRAY) {
    GPUImage2DArrayDescriptor desc;
    desc.data_type = data_type_;
    desc.access_type = access_type_;
    resources.image2d_arrays.push_back({"image2d_array", desc});
  } else if (storage_type_ == TensorStorageType::TEXTURE_3D) {
    GPUImage3DDescriptor desc;
    desc.data_type = data_type_;
    desc.access_type = access_type_;
    resources.images3d.push_back({"image3d", desc});
  } else if (storage_type_ == TensorStorageType::IMAGE_BUFFER) {
    if (access_type_ == AccessType::WRITE &&
        use_buffer_for_write_only_image_buffer_) {
      GPUBufferDescriptor desc;
      desc.data_type = data_type_;
      desc.access_type = access_type_;
      desc.element_size = 4;
      resources.buffers.push_back({"buffer", desc});
    } else {
      GPUImageBufferDescriptor desc;
      desc.data_type = data_type_;
      desc.access_type = access_type_;
      resources.image_buffers.push_back({"image_buffer", desc});
    }
  }
  return resources;
}

void TensorDescriptor::GetGpuResources(
    const BHWDC& tensor_shape, GenericGPUResourcesWithValue* resources) const {
  // Reserve space for the maximum ints that can be added.
  resources->ints.reserve(7);
  if (IsLinear() && physical_layout_1d_ == PhysicalLayout1D::kDHWBCC4) {
    resources->AddInt("slice_stride", 1);
  } else {
    if (HasAxis(Axis::BATCH)) {
      resources->AddInt("slice_stride",
                        tensor_shape.w * tensor_shape.h * tensor_shape.b);
    } else {
      resources->AddInt("slice_stride", tensor_shape.w * tensor_shape.h);
    }
  }
  if (HasAxis(Axis::WIDTH)) {
    resources->AddInt("width", tensor_shape.w);
  }
  if (HasAxis(Axis::HEIGHT)) {
    resources->AddInt("height", tensor_shape.h);
  }
  if (HasAxis(Axis::CHANNELS)) {
    resources->AddInt("slices", DivideRoundUp(tensor_shape.c, 4));
    resources->AddInt("channels", tensor_shape.c);
  }
  if (HasAxis(Axis::BATCH)) {
    resources->AddInt("batch", tensor_shape.b);
  }
  if (HasAxis(Axis::DEPTH)) {
    resources->AddInt("depth", tensor_shape.d);
  }
}

absl::Status TensorDescriptor::PerformConstExpr(const GpuInfo& gpu_info,
                                                absl::string_view const_expr,
                                                std::string* result) const {
  DataType data_type = data_type_;
  if (!(gpu_info.IsApiMetal() &&
        gpu_info.metal_info.IsNativeBfloatSupported())) {
    data_type =
        data_type_ == DataType::BFLOAT16 ? DataType::FLOAT32 : data_type_;
  }
  if (const_expr == "type" || const_expr == "scalar_type") {
    const int vec_size = const_expr == "scalar_type" ? 1 : 4;
    *result = ToUclDataType(data_type, vec_size);
    return absl::OkStatus();
  } else if (const_expr == "zero_value" || const_expr == "scalar_zero_value") {
    const int vec_size = const_expr == "scalar_zero_value" ? 1 : 4;
    *result = "ucl::Init<" + ToUclDataType(data_type, vec_size) + ">(" +
              GetZeroValue(data_type) + ")";
    return absl::OkStatus();
  } else {
    return absl::UnimplementedError(
        absl::StrCat("Can not resolve constant expression - ", const_expr));
  }
}

absl::Status TensorDescriptor::PerformSelector(
    const GpuInfo& gpu_info, absl::string_view selector,
    const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
  if (selector == "Width") {
    *result = "width";
    return absl::OkStatus();
  } else if (selector == "Height") {
    *result = "height";
    return absl::OkStatus();
  } else if (selector == "Slices") {
    *result = "slices";
    return absl::OkStatus();
  } else if (selector == "SliceStride") {
    *result = "slice_stride";
    return absl::OkStatus();
  } else if (selector == "Channels") {
    *result = "channels";
    return absl::OkStatus();
  } else if (selector == "Batch") {
    if (HasAxis(Axis::BATCH)) {
      *result = "batch";
    } else {
      *result = "1";
    }
    return absl::OkStatus();
  } else if (selector == "Depth") {
    if (HasAxis(Axis::DEPTH)) {
      *result = "depth";
    } else {
      *result = "1";
    }
    return absl::OkStatus();
  } else if (selector == "SetBatchRef") {
    if (args.size() != 1) {
      return absl::InvalidArgumentError(
          "Unsupported arguments in SetBatchRef selector");
    }
    state_vars_["batch_id"] = args[0];
    *result = "";
    return absl::OkStatus();
  } else if (selector == "Read") {
    return PerformReadSelector(gpu_info, args, template_args, result);
  } else if (selector == "ReadNearest") {
    return PerformReadNearestSelector(gpu_info, args, result);
  } else if (selector == "ReadBilinear") {
    return PerformReadBilinearSelector(gpu_info, args, result);
  } else if (selector == "ReadPerChannel") {
    return PerformReadPerChannelSelector(gpu_info, args, template_args, result);
  } else if (selector == "Write") {
    return PerformWriteSelector(gpu_info, args, result);
  } else if (selector == "WriteLinear") {
    return PerformWriteLinearSelector(gpu_info, args, result);
  } else if (selector == "Write2D") {
    return PerformWrite2DSelector(gpu_info, args, result);
  } else if (selector == "GetAddress") {
    return PerformGetAddressSelector(args, result);
  } else if (selector == "GetHandle") {
    return PerformGetHandleSelector(args, result);
  } else {
    return absl::NotFoundError(absl::StrCat(
        "TensorDescriptor don't have selector with name - ", selector));
  }
}

absl::Status TensorDescriptor::PerformReadSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
  if (args.empty()) {
    return absl::InvalidArgumentError("Expected not empty arguments.");
  }
  DataType read_as_type = data_type_;
  RETURN_IF_ERROR(
      MaybeGetDataTypeFromTemplateArgs(template_args, &read_as_type));
  if (!(gpu_info.IsApiMetal() &&
        gpu_info.metal_info.IsNativeBfloatSupported())) {
    read_as_type =
        read_as_type == DataType::BFLOAT16 ? DataType::FLOAT32 : read_as_type;
  }
  if (layout_ == Layout::LINEAR) {
    if (args.size() != 1) {
      return absl::InvalidArgumentError(
          "Read selector for LINEAR tensor require single argument");
    }
    *result = Read(gpu_info, read_as_type, GetPhysicalCoordsLinear(args[0]));
    return absl::OkStatus();
  }
  if (layout_ == Layout::HW) {
    if (args.size() != 2) {
      return absl::InvalidArgumentError(
          "Read selector for HW tensor require two arguments");
    }
    *result =
        Read(gpu_info, read_as_type, GetPhysicalCoordsHW(args[0], args[1]));
    return absl::OkStatus();
  }
  if (args.size() == 1) {  // function overload for 1D linear types.
    if (storage_type_ == TensorStorageType::BUFFER ||
        storage_type_ == TensorStorageType::IMAGE_BUFFER) {
      *result = Read(gpu_info, read_as_type, {args[0]});
      return absl::OkStatus();
    } else {
      return absl::InvalidArgumentError(
          "Read selector with single argument can be used only with linear "
          "storage types(BUFFER or IMAGE_BUFFER)");
    }
  }
  std::string xc, yc, zc, sc, bc;
  RETURN_IF_ERROR(ParseCoordsFromArgs(args, 0, &xc, &yc, &zc, &sc, &bc));
  *result = Read(gpu_info, read_as_type, GetPhysicalCoords(xc, yc, zc, sc, bc));
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformReadNearestSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    std::string* result) const {
  // ReadNearest(result, fc_x, fc_y, {fc_z}, slice);
  if (!((args.size() == 5 && HasAxis(Axis::DEPTH)) || args.size() == 4)) {
    return absl::NotFoundError("Unrecognized ReadNearest selector");
  }
  std::vector<std::string> coord_args =
      std::vector<std::string>(args.begin() + 1, args.end());
  std::string c;
  c += "  {\n";
  c += "  int coord_x_TMP = ucl::Convert<int>(" + coord_args[0] + ");\n";
  c += "  coord_x_TMP = max(coord_x_TMP, 0);\n";
  c += "  coord_x_TMP = min(coord_x_TMP, width - 1);\n";
  coord_args[0] = "coord_x_TMP";
  c += "  int coord_y_TMP = ucl::Convert<int>(" + coord_args[1] + ");\n";
  c += "  coord_y_TMP = max(coord_y_TMP, 0);\n";
  c += "  coord_y_TMP = min(coord_y_TMP, height - 1);\n";
  coord_args[1] = "coord_y_TMP";
  if (HasAxis(Axis::DEPTH)) {
    c += "  int coord_z_TMP = ucl::Convert<int>(" + coord_args[2] + ");\n";
    c += "  coord_z_TMP = max(coord_z_TMP, 0);\n";
    c += "  coord_z_TMP = min(coord_z_TMP, depth - 1);\n";
    coord_args[2] = "coord_z_TMP";
  }
  std::string src_value;
  RETURN_IF_ERROR(PerformReadSelector(gpu_info, coord_args, {}, &src_value));
  c += "  " + args[0] + " = " + src_value + ";\n";
  c += "  }";
  *result = c;
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformReadBilinearSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    std::string* result) const {
  // ReadBilinear(result, fc_x, fc_y, {fc_z}, slice);
  if (!((args.size() == 5 && HasAxis(Axis::DEPTH)) || args.size() == 4)) {
    return absl::NotFoundError("Unrecognized ReadBilinear selector");
  }
  std::vector<std::string> coord_args =
      std::vector<std::string>(args.begin() + 1, args.end());
  std::string c;
  c += "  {\n";
  c += "  SType f_x_TMP = ucl::Convert<SType>(floor(" + coord_args[0] + "));\n";
  c += "  SType x_scale_TMP = ucl::Convert<SType>(" + coord_args[0] +
       ") - f_x_TMP;\n";
  c += "  int i_x_TMP = ucl::Convert<int>(f_x_TMP);\n";
  c += "  int start_x_TMP = max(i_x_TMP, 0);\n";
  c += "  int end_x_TMP = min(i_x_TMP + 1, width - 1);\n";
  c += "  SType f_y_TMP = ucl::Convert<SType>(floor(" + coord_args[1] + "));\n";
  c += "  SType y_scale_TMP = ucl::Convert<SType>(" + coord_args[1] +
       ") - f_y_TMP;\n";
  c += "  int i_y_TMP = ucl::Convert<int>(f_y_TMP);\n";
  c += "  int start_y_TMP = max(i_y_TMP, 0);\n";
  c += "  int end_y_TMP = min(i_y_TMP + 1, height - 1);\n";
  if (HasAxis(Axis::DEPTH)) {
    // 3d bilinear read, x, y, z
    c += "  SType f_z_TMP = ucl::Convert<SType>(floor(" + coord_args[2] +
         "));\n";
    c += "  SType z_scale_TMP = ucl::Convert<SType>(" + coord_args[2] +
         ") - f_z_TMP;\n";
    c += "  int i_z_TMP = ucl::Convert<int>(f_z_TMP);\n";
    c += "  int start_z_TMP = max(i_z_TMP, 0);\n";
    c += "  int end_z_TMP = min(i_z_TMP + 1, depth - 1);\n";
    int index = 0;
    for (const auto& src_z : {"start_z_TMP", "end_z_TMP"}) {
      for (const auto& src_y : {"start_y_TMP", "end_y_TMP"}) {
        for (const auto& src_x : {"start_x_TMP", "end_x_TMP"}) {
          coord_args[0] = src_x;
          coord_args[1] = src_y;
          coord_args[2] = src_z;
          std::string src_value;
          RETURN_IF_ERROR(
              PerformReadSelector(gpu_info, coord_args, {}, &src_value));
          c += "  Type src" + std::to_string(index) + "_TMP = " + src_value +
               ";\n";
          index++;
        }
      }
    }
    c += "  Type r01_TMP = src0_TMP + (src1_TMP - src0_TMP) * x_scale_TMP;\n";
    c += "  Type r23_TMP = src2_TMP + (src3_TMP - src2_TMP) * x_scale_TMP;\n";
    c += "  Type r45_TMP = src4_TMP + (src5_TMP - src4_TMP) * x_scale_TMP;\n";
    c += "  Type r67_TMP = src6_TMP + (src7_TMP - src6_TMP) * x_scale_TMP;\n";
    c += "  Type r0123_TMP = r01_TMP + (r23_TMP - r01_TMP) * y_scale_TMP;\n";
    c += "  Type r4567_TMP = r45_TMP + (r67_TMP - r45_TMP) * y_scale_TMP;\n";
    c += "  Type result_TMP = r0123_TMP + (r4567_TMP - r0123_TMP) * "
         "z_scale_TMP;\n";
  } else {
    // 2d bilinear read, x, y
    int index = 0;
    for (const auto& src_y : {"start_y_TMP", "end_y_TMP"}) {
      for (const auto& src_x : {"start_x_TMP", "end_x_TMP"}) {
        coord_args[0] = src_x;
        coord_args[1] = src_y;
        std::string src_value;
        RETURN_IF_ERROR(
            PerformReadSelector(gpu_info, coord_args, {}, &src_value));
        c += "  Type src" + std::to_string(index) + "_TMP = " + src_value +
             ";\n";
        index++;
      }
    }
    c += "  Type r01_TMP = src0_TMP + (src1_TMP - src0_TMP) * x_scale_TMP;\n";
    c += "  Type r23_TMP = src2_TMP + (src3_TMP - src2_TMP) * x_scale_TMP;\n";
    c += "  Type result_TMP = r01_TMP + (r23_TMP - r01_TMP) * y_scale_TMP;\n";
  }
  c += "  " + args[0] + " = result_TMP;\n";
  c += "  }";

  *result = absl::StrReplaceAll(c, {{"Type", ToUclDataType(data_type_, 4)},
                                    {"SType", ToUclDataType(data_type_, 1)}});
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformReadPerChannelSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
  std::vector<std::string> coord_args =
      std::vector<std::string>(args.begin() + 1, args.end());
  int channels_index = 0;
  if (layout_ == Layout::LINEAR) {
    if (coord_args.size() != 1) {
      return absl::NotFoundError(
          "Wrong number of coordinates in ReadPerChannel for Layout::LINEAR.");
    }
  } else {
    if (HasAxis(Axis::WIDTH)) {
      channels_index++;
    }
    if (HasAxis(Axis::HEIGHT)) {
      channels_index++;
    }
    if (HasAxis(Axis::DEPTH)) {
      channels_index++;
    }
    if (channels_index >= coord_args.size()) {
      return absl::NotFoundError(
          "Wrong number of coordinates in ReadPerChannel.");
    }
  }
  std::string c = "  {\n";
  c += "  int slice_coord_TMP = (" + coord_args[channels_index] + ") / 4;\n";
  c += "  int sub_ch_coord_TMP = (" + coord_args[channels_index] + ") % 4;\n";
  coord_args[channels_index] = "slice_coord_TMP";
  std::string src_value;
  RETURN_IF_ERROR(
      PerformReadSelector(gpu_info, coord_args, template_args, &src_value));
  DataType dst_type = data_type_;
  dst_type = dst_type == DataType::BFLOAT16 ? DataType::FLOAT32 : dst_type;
  RETURN_IF_ERROR(MaybeGetDataTypeFromTemplateArgs(template_args, &dst_type));
  if (gpu_info.IsApiOpenCl()) {
    c += "  " + ToUclDataType(dst_type, 4) + " src_TMP = " + src_value + ";\n";
    c +=
        "  " + args[0] + " = (" + ToCLDataType(dst_type, 1) +
        "[4]){src_TMP.x, src_TMP.y, src_TMP.z, src_TMP.w}[sub_ch_coord_TMP];\n";
  } else if (gpu_info.IsApiCuda()) {
    c += "  " + ToUclDataType(dst_type, 4) + " src_TMP = " + src_value + ";\n";
    c += "  " + ToUclDataType(dst_type, 1) +
         " src_TMP_arr[4] = {src_TMP.x, src_TMP.y, src_TMP.z, src_TMP.w};\n";
    c += "  " + args[0] + " = src_TMP_arr[sub_ch_coord_TMP];\n";
  } else {
    if (gpu_info.IsAdreno() && gpu_info.IsApiVulkan()) {
      c +=
          "  " + ToUclDataType(dst_type, 4) + " src_TMP = " + src_value + ";\n";
      c += "  " + args[0] + " = " +
           ToGlslShaderDataType(dst_type, 1, /*add_precision*/ false,
                                gpu_info.vulkan_info.SupportsExplicitFp16()) +
           "[4](src_TMP.x, src_TMP.y, src_TMP.z, "
           "src_TMP.w)[sub_ch_coord_TMP];\n";
    } else {
      c += "  " + args[0] + " = " + src_value + "[sub_ch_coord_TMP];\n";
    }
  }

  c += "  }";
  *result = c;
  return absl::OkStatus();
}

absl::Status TensorDescriptor::GetLinkingContextFromWriteSelector(
    const std::vector<std::string>& args, std::string* value_name,
    std::string* x_coord, std::string* y_coord, std::string* z_coord,
    std::string* s_coord, std::string* b_coord) const {
  if (args.size() < 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expected to get at least two arguments, but got ", args.size(),
        ". Actual arguments are: ", absl::StrJoin(args, ", "), "."));
  }
  std::string xc;
  std::string yc;
  std::string zc;
  std::string sc;
  std::string bc;
  RETURN_IF_ERROR(ParseCoordsFromArgs(args, 1, &xc, &yc, &zc, &sc, &bc));
  *value_name = args[0];
  *b_coord = absl::StrCat("(", bc, ")");
  *x_coord = absl::StrCat("(", xc, ")");
  *y_coord = absl::StrCat("(", yc, ")");
  *z_coord = absl::StrCat("(", zc, ")");
  *s_coord = absl::StrCat("(", sc, ")");
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformWriteSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    std::string* result) const {
  if (args.size() < 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expected to get at least two arguments, but got ", args.size(),
        ". Actual arguments are: ", absl::StrJoin(args, ", "), "."));
  }
  std::string xc;
  std::string yc;
  std::string zc;
  std::string sc;
  std::string bc;
  RETURN_IF_ERROR(ParseCoordsFromArgs(args, 1, &xc, &yc, &zc, &sc, &bc));
  *result = Write(gpu_info, args[0], GetPhysicalCoords(xc, yc, zc, sc, bc));
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformWriteLinearSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    std::string* result) const {
  if (storage_type_ != TensorStorageType::BUFFER &&
      storage_type_ != TensorStorageType::IMAGE_BUFFER &&
      storage_type_ != TensorStorageType::TEXTURE_2D) {
    return absl::InvalidArgumentError(
        "WriteLinear selector can be used only with linear "
        "storages(BUFFER/IMAGE_BUFFER)");
  }
  if (args.size() != 2) {
    return absl::NotFoundError("Unrecognized WriteLinear selector");
  }
  if (storage_type_ == TensorStorageType::TEXTURE_2D) {
    *result = Write(gpu_info, args[0], {args[1], "0"});
  } else {
    *result = Write(gpu_info, args[0], {args[1]});
  }
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformWrite2DSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    std::string* result) const {
  if (storage_type_ != TensorStorageType::TEXTURE_2D) {
    return absl::InvalidArgumentError(
        "Write2D selector can be used only with 2d "
        "storages(TEXTURE_2D)");
  }
  if (args.size() != 3) {
    return absl::NotFoundError("Unrecognized Write2D selector");
  }
  *result = Write(gpu_info, args[0], {args[1], args[2]});
  return absl::OkStatus();
}

std::string TensorDescriptor::Read(
    const GpuInfo& gpu_info, DataType read_as_type,
    const std::vector<std::string>& coords) const {
  const std::string conversion =
      GetConversion(gpu_info, storage_type_, data_type_, read_as_type);
  if (gpu_info.IsApiOpenCl() &&
      !(data_type_ == DataType::FLOAT16 && read_as_type == DataType::FLOAT32)) {
    read_as_type = data_type_;
  }
  switch (storage_type_) {
    case TensorStorageType::BUFFER: {
      std::string result;
      if (gpu_info.IsApiWebGpu()) {
        if (data_type_ == DataType::FLOAT16 &&
            !gpu_info.webgpu_info.supports_fp16) {
          result =
              absl::StrCat("Unpack4x16float(buffer.data[", coords[0], "])");
        } else if (data_type_ == DataType::UINT16 ||
                   data_type_ == DataType::INT16 ||
                   data_type_ == DataType::BFLOAT16) {
          if (data_type_ == DataType::UINT16 ||
              data_type_ == DataType::BFLOAT16) {
            result = "vec4<u32>(";
          } else {
            result = "vec4<i32>(";
          }
          result += "extractBits(buffer.data[" + coords[0] + "].x, 0, 16), ";
          result += "extractBits(buffer.data[" + coords[0] + "].x, 16, 16), ";
          result += "extractBits(buffer.data[" + coords[0] + "].y, 0, 16), ";
          result += "extractBits(buffer.data[" + coords[0] + "].y, 16, 16))";
        } else if (data_type_ == DataType::UINT8 ||
                   data_type_ == DataType::INT8 ||
                   data_type_ == DataType::BOOL) {
          if (data_type_ == DataType::INT8) {
            result = "vec4<i32>(";
          } else {
            result = "vec4<u32>(";
          }
          result += "extractBits(buffer.data[" + coords[0] + "], 0, 8), ";
          result += "extractBits(buffer.data[" + coords[0] + "], 8, 8), ";
          result += "extractBits(buffer.data[" + coords[0] + "], 16, 8), ";
          result += "extractBits(buffer.data[" + coords[0] + "], 24, 8))";
        } else {
          result = absl::StrCat("buffer.data[", coords[0], "]");
        }
      } else if (gpu_info.IsGlsl() && data_type_ == DataType::FLOAT16 &&
                 !gpu_info.IsGlslSupportsExplicitFp16()) {
        result =
            absl::StrCat("vec4(unpackHalf2x16(buffer[", coords[0],
                         "].x), unpackHalf2x16(buffer[", coords[0], "].y))");
      } else if (gpu_info.IsGlsl() && (data_type_ == DataType::UINT16 ||
                                       data_type_ == DataType::INT16 ||
                                       data_type_ == DataType::BFLOAT16)) {
        if (data_type_ == DataType::UINT16 ||
            data_type_ == DataType::BFLOAT16) {
          result = "uvec4(";
        } else {
          result = "ivec4(";
        }
        result += "bitfieldExtract(buffer[" + coords[0] + "].x, 0, 16), ";
        result += "bitfieldExtract(buffer[" + coords[0] + "].x, 16, 16), ";
        result += "bitfieldExtract(buffer[" + coords[0] + "].y, 0, 16), ";
        result += "bitfieldExtract(buffer[" + coords[0] + "].y, 16, 16))";
      } else if (gpu_info.IsGlsl() && (data_type_ == DataType::UINT8 ||
                                       data_type_ == DataType::INT8 ||
                                       data_type_ == DataType::BOOL)) {
        if (data_type_ == DataType::INT8) {
          result = "ivec4(";
        } else {
          result = "uvec4(";
        }
        result += "bitfieldExtract(buffer[" + coords[0] + "], 0, 8), ";
        result += "bitfieldExtract(buffer[" + coords[0] + "], 8, 8), ";
        result += "bitfieldExtract(buffer[" + coords[0] + "], 16, 8), ";
        result += "bitfieldExtract(buffer[" + coords[0] + "], 24, 8))";
      } else {
        result = absl::StrCat("buffer[", coords[0], "]");
      }
      MayBeAddConversion(conversion, &result);
      return result;
    }
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D: {
      std::string result;
      if (gpu_info.IsApiWebGpu()) {
        result = absl::Substitute("textureLoad(image2d, vec2<i32>($0, $1), 0)",
                                  coords[0], coords[1]);
      } else if (gpu_info.IsApiOpenCl()) {
        result = absl::Substitute("$0(image2d, smp_zero, (int2)($1, $2))",
                                  GetReadImageFromDataType(read_as_type),
                                  coords[0], coords[1]);
      } else if (gpu_info.IsApiMetal()) {
        result = absl::Substitute("image2d.read(ushort2($0, $1))", coords[0],
                                  coords[1]);
      } else if (gpu_info.IsGlsl()) {
        result = "texelFetch(image2d, ivec2(" + coords[0] + ", " + coords[1] +
                 "), 0)";
        if (data_type_ == DataType::FLOAT16 &&
            gpu_info.IsGlslSupportsExplicitFp16()) {
          result = "f16vec4(" + result + ")";
        }
      } else if (gpu_info.IsApiCuda()) {
        std::string type = ToUclDataType(ToCudaTextureType(data_type_), 4);
        result = absl::Substitute("tex2D<$2>(image2d, $0, $1)", coords[0],
                                  coords[1], type);
      }
      MayBeAddConversion(conversion, &result);
      return result;
    }
    case TensorStorageType::TEXTURE_3D: {
      std::string result;
      if (gpu_info.IsApiWebGpu()) {
        result =
            absl::Substitute("textureLoad(image3d, vec3<i32>($0, $1, $2), 0)",
                             coords[0], coords[1], coords[2]);
      } else if (gpu_info.IsApiOpenCl()) {
        result =
            absl::Substitute("$0(image3d, smp_zero, (int4)($1, $2, $3, 0))",
                             GetReadImageFromDataType(read_as_type), coords[0],
                             coords[1], coords[2]);
      } else if (gpu_info.IsApiMetal()) {
        result = absl::Substitute("image3d.read(ushort3($0, $1, $2))",
                                  coords[0], coords[1], coords[2]);
      } else if (gpu_info.IsGlsl()) {
        result = "texelFetch(image3d, ivec3(" + coords[0] + ", " + coords[1] +
                 ", " + coords[2] + "), 0)";
        if (data_type_ == DataType::FLOAT16 &&
            gpu_info.IsGlslSupportsExplicitFp16()) {
          result = "f16vec4(" + result + ")";
        }
      } else if (gpu_info.IsApiCuda()) {
        std::string type = ToUclDataType(ToCudaTextureType(data_type_), 4);
        result = absl::Substitute("tex3D<$3>(image3d, $0, $1, $2)", coords[0],
                                  coords[1], coords[2], type);
      }
      MayBeAddConversion(conversion, &result);
      return result;
    }
    case TensorStorageType::TEXTURE_ARRAY: {
      std::string result;
      if (gpu_info.IsApiWebGpu()) {
        result = absl::Substitute(
            "textureLoad(image2d_array, vec2<i32>($0, $1), $2, 0)", coords[0],
            coords[1], coords[2]);
      } else if (gpu_info.IsApiOpenCl()) {
        result = absl::Substitute(
            "$0(image2d_array, smp_zero, (int4)($1, $2, $3, 0))",
            GetReadImageFromDataType(read_as_type), coords[0], coords[1],
            coords[2]);
      } else if (gpu_info.IsApiMetal()) {
        result = absl::Substitute("image2d_array.read(ushort2($0, $1), $2)",
                                  coords[0], coords[1], coords[2]);
      } else if (gpu_info.IsGlsl()) {
        result = "texelFetch(image2d_array, ivec3(" + coords[0] + ", " +
                 coords[1] + ", " + coords[2] + "), 0)";
        if (data_type_ == DataType::FLOAT16 &&
            gpu_info.IsGlslSupportsExplicitFp16()) {
          result = "f16vec4(" + result + ")";
        }
      } else if (gpu_info.IsApiCuda()) {
        std::string type = ToUclDataType(ToCudaTextureType(data_type_), 4);
        result = absl::Substitute("tex2DLayered<$3>(image2d_array, $0, $1, $2)",
                                  coords[0], coords[1], coords[2], type);
      }
      MayBeAddConversion(conversion, &result);
      return result;
    }
    case TensorStorageType::IMAGE_BUFFER: {
      std::string result;
      if (gpu_info.IsApiOpenCl()) {
        result = absl::StrCat(GetReadImageFromDataType(read_as_type),
                              "(image_buffer, ", coords[0], ")");
      } else if (gpu_info.IsApiMetal()) {
        result = absl::Substitute("image_buffer.read(uint($0))", coords[0]);
      } else if (gpu_info.IsGlsl()) {
        result = "texelFetch(image_buffer, " + coords[0] + ")";
        if (data_type_ == DataType::FLOAT16 &&
            gpu_info.IsGlslSupportsExplicitFp16()) {
          result = "f16vec4(" + result + ")";
        }
      } else if (gpu_info.IsApiCuda()) {
        std::string type = ToUclDataType(ToCudaTextureType(data_type_), 4);
        result = absl::Substitute("tex1Dfetch<$1>(image_buffer, $0)", coords[0],
                                  type);
      }
      MayBeAddConversion(conversion, &result);
      return result;
    }
    case TensorStorageType::UNKNOWN:
      return "";
  }
}

std::string TensorDescriptor::Write(
    const GpuInfo& gpu_info, const std::string& var_name,
    const std::vector<std::string>& coords) const {
  bool is_texture_write = storage_type_ == TensorStorageType::IMAGE_BUFFER ||
                          storage_type_ == TensorStorageType::TEXTURE_2D ||
                          storage_type_ == TensorStorageType::TEXTURE_ARRAY ||
                          storage_type_ == TensorStorageType::TEXTURE_3D ||
                          storage_type_ == TensorStorageType::SINGLE_TEXTURE_2D;
  if (storage_type_ == TensorStorageType::IMAGE_BUFFER &&
      use_buffer_for_write_only_image_buffer_) {
    is_texture_write = false;
  }
  if (storage_type_ == TensorStorageType::TEXTURE_2D &&
      use_buffer_for_write_only_2d_texture_) {
    is_texture_write = false;
  }
  std::string write_expr = var_name;
  DataType write_required_type = data_type_;
  if (data_type_ == DataType::BOOL) {
    // DataType::BOOL stored as DataType::UINT8
    write_expr = "ucl::Convert<uchar4>(" + write_expr + ")";
    write_required_type = DataType::UINT8;
  } else if (data_type_ == DataType::BFLOAT16) {
    if (!(gpu_info.IsApiMetal() &&
          gpu_info.metal_info.IsNativeBfloatSupported())) {
      // DataType::BFLOAT16 stored as DataType::UINT16
      write_expr = "ucl::ConvertToBfloat<" +
                   ToUclDataType(DataType::FLOAT32, 4) + ">(" + write_expr +
                   ")";
      write_required_type = DataType::UINT16;
    }
  }
  if (is_texture_write) {
    if (gpu_info.IsApiOpenCl()) {
      write_required_type = ToClTextureType(write_required_type);
    } else if (gpu_info.IsApiMetal()) {
      write_required_type = ToMetalTextureType(write_required_type);
    } else if (gpu_info.IsApiWebGpu()) {
      write_required_type = ToWebGpuTextureType(write_required_type);
    }
  }
  if (data_type_ != write_required_type) {
    if (data_type_ == DataType::BFLOAT16 && gpu_info.IsApiMetal() &&
        gpu_info.metal_info.IsMslVersionEqualOrHigher(
            3, 1)) {  // Metal native bfloat16 support
      // Metal doesn't support bfloat texture reads/writes, so we need to
      // reinterpret instead of convert
      write_expr = "ucl::Reinterpret<bfloat4, ushort4>(" + write_expr + ")";
    } else {
      write_expr = "ucl::Convert<" + ToUclDataType(write_required_type, 4) +
                   ">(" + write_expr + ")";
    }
  }
  switch (storage_type_) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      if (gpu_info.IsApiWebGpu()) {
        if (data_type_ == DataType::FLOAT16 &&
            !gpu_info.webgpu_info.supports_fp16) {
          return absl::StrCat("buffer.data[", coords[0],
                              "] = Pack4x16float(vec4<f32>(", write_expr, "))");
        } else if (data_type_ == DataType::UINT16 ||
                   data_type_ == DataType::INT16 ||
                   data_type_ == DataType::BFLOAT16 ||
                   data_type_ == DataType::UINT8 ||
                   data_type_ == DataType::INT8 ||
                   data_type_ == DataType::BOOL) {
          return absl::StrCat("buffer.data[", coords[0],
                              "] = QuantizedBufferWrite(", write_expr, ")");
        } else {
          return absl::StrCat("buffer.data[", coords[0], "] = ", write_expr);
        }
      } else if (gpu_info.IsApiOpenCl()) {
        if (use_buffer_for_write_only_image_buffer_) {
          return absl::StrCat("buffer[", coords[0], "] = ", write_expr);
        } else {
          return absl::Substitute("$0(image_buffer, $1, $2)",
                                  GetWriteImageFromDataType(data_type_),
                                  coords[0], write_expr);
        }
      } else if (gpu_info.IsApiMetal()) {
        if (use_buffer_for_write_only_image_buffer_) {
          return absl::StrCat("buffer[", coords[0], "] = ", write_expr);
        } else {
          return absl::Substitute("image_buffer.write($0, uint($1))",
                                  write_expr, coords[0]);
        }
      } else if (gpu_info.IsGlsl()) {
        if (data_type_ == DataType::FLOAT16 &&
            !gpu_info.IsGlslSupportsExplicitFp16()) {
          return absl::StrCat("buffer[", coords[0], "] = uvec2(packHalf2x16(",
                              write_expr, ".xy), packHalf2x16(", write_expr,
                              ".zw))");
        } else if (data_type_ == DataType::UINT16 ||
                   data_type_ == DataType::INT16 ||
                   data_type_ == DataType::BFLOAT16 ||
                   data_type_ == DataType::UINT8 ||
                   data_type_ == DataType::INT8 ||
                   data_type_ == DataType::BOOL) {
          return absl::StrCat("buffer[", coords[0], "] = QuantizedBufferWrite(",
                              write_expr, ")");
        } else {
          return absl::StrCat("buffer[", coords[0], "] = ", write_expr);
        }
      } else {
        return absl::StrCat("buffer[", coords[0], "] = ", write_expr);
      }
    case TensorStorageType::SINGLE_TEXTURE_2D:
    case TensorStorageType::TEXTURE_2D:
      if (gpu_info.IsApiWebGpu()) {
        return absl::Substitute("textureStore(image2d, vec2<i32>($0, $1), $2)",
                                coords[0], coords[1], write_expr);
      } else if (gpu_info.IsApiOpenCl()) {
        if (use_buffer_for_write_only_2d_texture_) {
          return absl::Substitute(
              "buffer[($2) * aligned_texture_width + ($1)] = $0", write_expr,
              coords[0], coords[1]);
        } else {
          return absl::Substitute("$0(image2d, (int2)($1, $2), $3)",
                                  GetWriteImageFromDataType(data_type_),
                                  coords[0], coords[1], write_expr);
        }
      } else if (gpu_info.IsApiMetal()) {
        if (use_buffer_for_write_only_2d_texture_) {
          return absl::Substitute(
              "buffer[($2) * aligned_texture_width + ($1)] = $0", write_expr,
              coords[0], coords[1]);
        } else {
          return absl::Substitute("image2d.write($0, ushort2($1, $2))",
                                  write_expr, coords[0], coords[1]);
        }
      } else if (gpu_info.IsGlsl()) {
        return absl::Substitute("imageStore(image2d, ivec2($0, $1), $2)",
                                coords[0], coords[1], write_expr);
      } else if (gpu_info.IsApiCuda()) {
        if (data_type_ == DataType::FLOAT16) {
          write_expr = "half4_as_ushort4(" + write_expr + ")";
        }
        return absl::Substitute("surf2Dwrite($2, image2d, $0 * $3, $1)",
                                coords[0], coords[1], write_expr,
                                4 * SizeOf(data_type_));
      } else {
        return "";
      }
    case TensorStorageType::TEXTURE_3D:
      if (gpu_info.IsApiWebGpu()) {
        return absl::Substitute(
            "textureStore(image3d, vec3<i32>($0, $1, $2), $3)", coords[0],
            coords[1], coords[2], write_expr);
      } else if (gpu_info.IsApiOpenCl()) {
        return absl::Substitute("$0(image3d, (int4)($1, $2, $3, 0), $4)",
                                GetWriteImageFromDataType(data_type_),
                                coords[0], coords[1], coords[2], write_expr);
      } else if (gpu_info.IsApiMetal()) {
        return absl::Substitute("image3d.write($0, ushort3($1, $2, $3))",
                                write_expr, coords[0], coords[1], coords[2]);
      } else if (gpu_info.IsGlsl()) {
        return absl::Substitute("imageStore(image3d, ivec3($0, $1, $2), $3)",
                                coords[0], coords[1], coords[2], write_expr);
      } else if (gpu_info.IsApiCuda()) {
        if (data_type_ == DataType::FLOAT16) {
          write_expr = "half4_as_ushort4(" + write_expr + ")";
        }
        return absl::Substitute("surf3Dwrite($3, image3d, $0 * $4, $1, $2)",
                                coords[0], coords[1], coords[2], write_expr,
                                4 * SizeOf(data_type_));
      } else {
        return "";
      }
    case TensorStorageType::TEXTURE_ARRAY:
      if (gpu_info.IsApiWebGpu()) {
        return absl::Substitute(
            "textureStore(image2d_array, vec2<i32>($0, $1), $2, $3)", coords[0],
            coords[1], coords[2], write_expr);
      } else if (gpu_info.IsApiOpenCl()) {
        return absl::Substitute("$0(image2d_array, (int4)($1, $2, $3, 0), $4)",
                                GetWriteImageFromDataType(data_type_),
                                coords[0], coords[1], coords[2], write_expr);
      } else if (gpu_info.IsApiMetal()) {
        return absl::Substitute("image2d_array.write($0, ushort2($1, $2), $3)",
                                write_expr, coords[0], coords[1], coords[2]);
      } else if (gpu_info.IsGlsl()) {
        return absl::Substitute(
            "imageStore(image2d_array, ivec3($0, $1, $2), $3)", coords[0],
            coords[1], coords[2], write_expr);
      } else if (gpu_info.IsApiCuda()) {
        if (data_type_ == DataType::FLOAT16) {
          write_expr = "half4_as_ushort4(" + write_expr + ")";
        }
        return absl::Substitute(
            "surf2DLayeredwrite($3, image2d_array, $0 * $4, $1, $2)", coords[0],
            coords[1], coords[2], write_expr, 4 * SizeOf(data_type_));
      } else {
        return "";
      }
    case TensorStorageType::UNKNOWN:
      return "";
  }
}

absl::Status TensorDescriptor::PerformGetAddressSelector(
    const std::vector<std::string>& args, std::string* result) const {
  std::string xc, yc, zc, sc, bc;
  RETURN_IF_ERROR(ParseCoordsFromArgs(args, 0, &xc, &yc, &zc, &sc, &bc));

  *result = GetGlobalAddressNoDeclaration(xc, yc, zc, sc, bc);
  return absl::OkStatus();
}

absl::Status TensorDescriptor::PerformGetHandleSelector(
    const std::vector<std::string>& args, std::string* result) const {
  if (!args.empty()) {
    return absl::NotFoundError(
        absl::StrCat("GetHandle does not require arguments, but ", args.size(),
                     " was passed"));
  }
  switch (storage_type_) {
    case TensorStorageType::BUFFER:
      *result = "buffer";
      return absl::OkStatus();
    case TensorStorageType::IMAGE_BUFFER:
      if (access_type_ == AccessType::READ) {
        *result = "image_buffer";
      } else {
        *result = "buffer";
      }
      return absl::OkStatus();
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      *result = "image2d";
      return absl::OkStatus();
    case TensorStorageType::TEXTURE_ARRAY:
      *result = "image2d_array";
      return absl::OkStatus();
    case TensorStorageType::TEXTURE_3D:
      *result = "image3d";
      return absl::OkStatus();
    case TensorStorageType::UNKNOWN:
      return absl::UnavailableError("Unknown type");
  }
}

std::string TensorDescriptor::StorageTypeToAddressType() const {
  switch (storage_type_) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return "int";
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return "int2";
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return "int4";
    case TensorStorageType::UNKNOWN:
      return "";
  }
}

std::vector<std::string> TensorDescriptor::GetPhysicalCoordsLinear(
    const std::string& x) const {
  switch (storage_type_) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return {absl::Substitute("($0)", x)};
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return {absl::Substitute("($0)", x), "0"};
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return {absl::Substitute("($0)", x), "0", "0"};
    case TensorStorageType::UNKNOWN:
      return {""};
    default:
      return {""};
  }
}

std::vector<std::string> TensorDescriptor::GetPhysicalCoordsHW(
    const std::string& x, const std::string& y) const {
  switch (storage_type_) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return {absl::Substitute("(($1) * width + ($0))", x, y)};
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return {absl::Substitute("($0)", x), absl::Substitute("($0)", y)};
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return {absl::Substitute("($0)", x), absl::Substitute("($0)", y), "0"};
    case TensorStorageType::UNKNOWN:
      return {""};
    default:
      return {""};
  }
}

std::vector<std::string> TensorDescriptor::GetPhysicalCoordsWHS(
    const std::string& x, const std::string& y, const std::string& s) const {
  switch (storage_type_) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      if (physical_layout_1d_ == PhysicalLayout1D::kDCHWBC4) {
        return {absl::Substitute("((($2) * height + ($1)) * width + ($0))", x,
                                 y, s)};
      } else if (physical_layout_1d_ == PhysicalLayout1D::kDHWBCC4) {
        return {absl::Substitute("((($2) * width + ($1)) * slices + ($0))", s,
                                 x, y)};
      } else {
        return {};
      }
    case TensorStorageType::TEXTURE_2D:
      if (physical_layout_2d_ == PhysicalLayout2D::kXisWBDC4YisHC) {
        return {absl::Substitute("($0)", x),
                absl::Substitute("(($0) * slices + ($1))", y, s)};
      } else if (physical_layout_2d_ == PhysicalLayout2D::kXisCC4YisDHWB) {
        return {absl::Substitute("($0)", s),
                absl::Substitute("(($0) * width + ($1))", y, x)};
      } else {
        return {};
      }
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return {absl::Substitute("($0)", x), absl::Substitute("($0)", y)};
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return {absl::Substitute("($0)", x), absl::Substitute("($0)", y),
              absl::Substitute("($0)", s)};
    case TensorStorageType::UNKNOWN:
      return {""};
    default:
      return {""};
  }
}

std::vector<std::string> TensorDescriptor::GetPhysicalCoordsWHSB(
    const std::string& x, const std::string& y, const std::string& s,
    const std::string& b) const {
  switch (storage_type_) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      if (physical_layout_1d_ == PhysicalLayout1D::kDCHWBC4) {
        return {absl::Substitute(
            "(((($3) * height + $2) * width + ($1)) * batch + ($0))", b, x, y,
            s)};
      } else if (physical_layout_1d_ == PhysicalLayout1D::kDHWBCC4) {
        return {absl::Substitute(
            "(((($3) * width + $2) * batch + ($1)) * slices + ($0))", s, b, x,
            y)};
      } else {
        return {};
      }
    case TensorStorageType::TEXTURE_2D:
      if (physical_layout_2d_ == PhysicalLayout2D::kXisWBDC4YisHC) {
        return {absl::Substitute("(($0) * batch + ($1))", x, b),
                absl::Substitute("(($0) * slices + ($1))", y, s)};
      } else if (physical_layout_2d_ == PhysicalLayout2D::kXisCC4YisDHWB) {
        return {absl::Substitute("($0)", s),
                absl::Substitute("((($0) * width + ($1)) * batch + ($2))", y, x,
                                 b)};
      } else {
        return {};
      }
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return {absl::Substitute("(($0) * batch + ($1))", x, b),
              absl::Substitute("($0)", y)};
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return {absl::Substitute("(($0) * batch + ($1))", x, b),
              absl::Substitute("($0)", y), absl::Substitute("($0)", s)};
    case TensorStorageType::UNKNOWN:
      return {""};
    default:
      return {""};
  }
}

std::vector<std::string> TensorDescriptor::GetPhysicalCoordsWHDS(
    const std::string& x, const std::string& y, const std::string& z,
    const std::string& s) const {
  switch (storage_type_) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      if (physical_layout_1d_ == PhysicalLayout1D::kDCHWBC4) {
        return {absl::Substitute(
            "(((($3) * slices + ($2)) * height + ($1)) * width + ($0))", x, y,
            s, z)};
      } else if (physical_layout_1d_ == PhysicalLayout1D::kDHWBCC4) {
        return {absl::Substitute(
            "(((($3) * height + ($2)) * width + ($1)) * slices + ($0))", s, x,
            y, z)};
      } else {
        return {};
      }
    case TensorStorageType::TEXTURE_2D:
      if (physical_layout_2d_ == PhysicalLayout2D::kXisWBDC4YisHC) {
        return {absl::Substitute("(($0) * depth + ($1))", x, z),
                absl::Substitute("(($0) * slices + ($1))", y, s)};
      } else if (physical_layout_2d_ == PhysicalLayout2D::kXisCC4YisDHWB) {
        return {absl::Substitute("($0)", s),
                absl::Substitute("((($0) * height + ($1)) * width + ($2))", z,
                                 y, x)};
      } else {
        return {};
      }
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return {absl::Substitute("(($0) * depth + ($1))", x, z),
              absl::Substitute("($0)", y)};
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return {absl::Substitute("($0)", x), absl::Substitute("($0)", y),
              absl::Substitute("(($0) * slices + ($1))", z, s)};
    case TensorStorageType::UNKNOWN:
      return {""};
    default:
      return {""};
  }
}

std::vector<std::string> TensorDescriptor::GetPhysicalCoordsWHDSB(
    const std::string& x, const std::string& y, const std::string& z,
    const std::string& s, const std::string& b) const {
  switch (storage_type_) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      if (physical_layout_1d_ == PhysicalLayout1D::kDCHWBC4) {
        return {
            absl::Substitute("((((($4) * slices + ($3)) * height + $2) * width "
                             "+ ($1)) * batch + "
                             "($0))",
                             b, x, y, s, z)};
      } else if (physical_layout_1d_ == PhysicalLayout1D::kDHWBCC4) {
        return {
            absl::Substitute("((((($4) * height + ($3)) * width + $2) * batch "
                             "+ ($1)) * slices + "
                             "($0))",
                             s, b, x, y, z)};
      } else {
        return {};
      }
    case TensorStorageType::TEXTURE_2D:
      if (physical_layout_2d_ == PhysicalLayout2D::kXisWBDC4YisHC) {
        return {absl::Substitute("((($0)*batch + ($1))*depth + ($2))", x, b, z),
                absl::Substitute("(($0) * slices + ($1))", y, s)};
      } else if (physical_layout_2d_ == PhysicalLayout2D::kXisCC4YisDHWB) {
        return {absl::Substitute("($0)", s),
                absl::Substitute(
                    "(((($0) * height + ($1)) * width + ($2)) * batch + ($3))",
                    z, y, x, b)};
      } else {
        return {};
      }
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return {absl::Substitute("((($0)*batch + ($1))*depth + ($2))", x, b, z),
              absl::Substitute("($0)", y)};
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return {absl::Substitute("(($0) * batch + ($1))", x, b),
              absl::Substitute("($0)", y),
              absl::Substitute("(($0) * slices + ($1))", z, s)};
    case TensorStorageType::UNKNOWN:
      return {""};
    default:
      return {""};
  }
}

std::string TensorDescriptor::GetGlobalAddressNoDeclaration(
    const std::string& xc, const std::string& yc, const std::string& zc,
    const std::string& sc, const std::string& bc) const {
  auto coords = GetPhysicalCoords(xc, yc, zc, sc, bc);
  switch (storage_type_) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER: {
      return coords[0];
    }
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return absl::Substitute("(int2)($0, $1)", coords[0], coords[1]);
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return absl::Substitute("(int4)($0, $1, $2, 0)", coords[0], coords[1],
                              coords[2]);
    case TensorStorageType::UNKNOWN:
      return "error";
  }
}

std::vector<std::string> TensorDescriptor::GetPhysicalCoords(
    const std::string& xc, const std::string& yc, const std::string& zc,
    const std::string& sc, const std::string& bc) const {
  if (layout_ == Layout::HWC) {
    return GetPhysicalCoordsWHS(xc, yc, sc);
  } else if (layout_ == Layout::BHWC) {
    return GetPhysicalCoordsWHSB(xc, yc, sc, bc);
  } else if (layout_ == Layout::HWDC) {
    return GetPhysicalCoordsWHDS(xc, yc, zc, sc);
  } else if (layout_ == Layout::BHWDC) {
    return GetPhysicalCoordsWHDSB(xc, yc, zc, sc, bc);
  } else {
    return {""};
  }
}

absl::Status TensorDescriptor::MaybeGetDataTypeFromTemplateArgs(
    const std::vector<std::string>& template_args, DataType* result) const {
  for (const auto& template_arg : template_args) {
    const auto& read_type = template_arg;
    if (read_type == "half") {
      *result = DataType::FLOAT16;
      return absl::OkStatus();
    } else if (read_type == "float") {
      *result = DataType::FLOAT32;
      return absl::OkStatus();
    } else if (read_type == "int") {
      *result = DataType::INT32;
      return absl::OkStatus();
    } else if (read_type == "short") {
      *result = DataType::INT16;
      return absl::OkStatus();
    } else if (read_type == "char") {
      *result = DataType::INT8;
      return absl::OkStatus();
    } else if (read_type == "uint") {
      *result = DataType::UINT32;
      return absl::OkStatus();
    } else if (read_type == "ushort") {
      *result = DataType::UINT16;
      return absl::OkStatus();
    } else if (read_type == "uchar") {
      *result = DataType::UINT8;
      return absl::OkStatus();
    } else if (read_type == "bool") {
      *result = DataType::BOOL;
      return absl::OkStatus();
    }
  }
  return absl::OkStatus();
}

bool TensorDescriptor::HasAxis(Axis axis) const {
  if (axis == Axis::WIDTH || axis == Axis::HEIGHT || axis == Axis::CHANNELS) {
    return true;
  }
  if (axis == Axis::BATCH &&
      (layout_ == Layout::BHWC || layout_ == Layout::BHWDC)) {
    return true;
  }
  if (axis == Axis::DEPTH &&
      (layout_ == Layout::HWDC || layout_ == Layout::BHWDC)) {
    return true;
  }
  return false;
}

absl::Status TensorDescriptor::ParseCoordsFromArgs(
    absl::Span<const std::string> args, int offset, std::string* xc,
    std::string* yc, std::string* zc, std::string* sc, std::string* bc) const {
  if (HasAxis(Axis::WIDTH)) {
    if (offset >= args.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Unable to parse xc coord for Width axis. Layout is ",
          ToString(layout_), ", args are ", absl::StrJoin(args, ", "), "."));
    }
    *xc = args[offset++];
  }
  if (HasAxis(Axis::HEIGHT)) {
    if (offset >= args.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Unable to parse yc coord for Height axis. Layout is ",
          ToString(layout_), ", args are ", absl::StrJoin(args, ", "), "."));
    }
    *yc = args[offset++];
  }
  if (HasAxis(Axis::DEPTH)) {
    if (offset >= args.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Unable to parse zc coord for Depth axis. Layout is ",
          ToString(layout_), ", args are ", absl::StrJoin(args, ", "), "."));
    }
    *zc = args[offset++];
  }
  if (HasAxis(Axis::CHANNELS)) {
    if (offset >= args.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Unable to parse sc coord for Channels axis. Layout is ",
          ToString(layout_), ", args are ", absl::StrJoin(args, ", "), "."));
    }
    *sc = args[offset++];
  }
  if (HasAxis(Axis::BATCH)) {
    if (offset >= args.size()) {
      auto it = state_vars_.find("batch_id");
      if (it == state_vars_.end()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Unable to parse bc coord for BATCH axis. Layout is ",
            ToString(layout_), ", args are ", absl::StrJoin(args, ", "), "."));
      } else {
        *bc = it->second;
      }
    } else {
      *bc = args[offset++];
    }
  }
  return absl::OkStatus();
}

size_t TensorDescriptor::GetSizeInBytesForShape(const BHWDC& shape5d) const {
  size_t aligned_channels =
      storage_type_ == TensorStorageType::SINGLE_TEXTURE_2D
          ? shape5d.c
          : AlignByN(shape5d.c, 4);
  size_t elements_count =
      aligned_channels * shape5d.b * shape5d.w * shape5d.h * shape5d.d;
  return elements_count * SizeOf(data_type_);
}

int TensorDescriptor::GetLinearIndex(const BHWDC& shape5d, int b, int x, int y,
                                     int d, int s, int sub_c) const {
  const int slices = DivideRoundUp(shape5d.c, 4);
  if ((IsLinear() && physical_layout_1d_ == PhysicalLayout1D::kDHWBCC4) ||
      (Is2DStorage() &&
       physical_layout_2d_ == PhysicalLayout2D::kXisCC4YisDHWB)) {
    return ((((d * shape5d.h + y) * shape5d.w + x) * shape5d.b + b) * slices +
            s) *
               4 +
           sub_c;
  }
  switch (storage_type_) {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_3D:
      return ((((d * slices + s) * shape5d.h + y) * shape5d.w + x) * shape5d.b +
              b) *
                 4 +
             sub_c;  // DSHWBC4
    case TensorStorageType::TEXTURE_2D:
      return ((((y * slices + s) * shape5d.w + x) * shape5d.b + b) * shape5d.d +
              d) *
                 4 +
             sub_c;  // HSWBDC4
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return (((y * shape5d.w + x) * shape5d.b + b) * shape5d.d + d) *
                 shape5d.c +
             sub_c;  // HWBDC
    case TensorStorageType::UNKNOWN:
      return -1;
  }
}

void TensorDescriptor::UploadData(const Tensor<HWC, DataType::FLOAT32>& src) {
  shape_ = BHWDC(1, src.shape.h, src.shape.w, 1, src.shape.c);
  UploadData(src.Data());
}

bool TensorDescriptor::SupportsZeroClamp(const Axis& axis,
                                         const GpuInfo& gpu_info) const {
  switch (storage_type_) {
    case TensorStorageType::UNKNOWN:
      return false;
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      return false;
    case TensorStorageType::TEXTURE_ARRAY:
      return (axis == Axis::WIDTH || axis == Axis::HEIGHT) &&
             gpu_info.SupportsZeroClampForImages();
    case TensorStorageType::TEXTURE_2D:
      if (physical_layout_2d_ == PhysicalLayout2D::kXisWBDC4YisHC) {
        return (axis == Axis::WIDTH || axis == Axis::HEIGHT) &&
               gpu_info.SupportsZeroClampForImages();
      } else {
        return false;
      }
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return (axis == Axis::WIDTH || axis == Axis::HEIGHT) &&
             gpu_info.SupportsZeroClampForImages();
    case TensorStorageType::TEXTURE_3D:
      return (axis == Axis::WIDTH || axis == Axis::HEIGHT ||
              axis == Axis::DEPTH) &&
             gpu_info.SupportsZeroClampForImages();
  }
}

bool TensorDescriptor::CanReadOutOfBorder(const Axis& axis,
                                          const GpuInfo& gpu_info) const {
  switch (storage_type_) {
    case TensorStorageType::UNKNOWN:
      return false;
    case TensorStorageType::BUFFER:
      return false;
    case TensorStorageType::IMAGE_BUFFER:
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::TEXTURE_3D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
    case TensorStorageType::TEXTURE_ARRAY:
      return true;
  }
}

bool TensorDescriptor::IsLinear() const {
  return storage_type_ == TensorStorageType::BUFFER ||
         storage_type_ == TensorStorageType::IMAGE_BUFFER;
}

bool TensorDescriptor::Is2DStorage() const {
  return storage_type_ == TensorStorageType::TEXTURE_2D ||
         storage_type_ == TensorStorageType::SINGLE_TEXTURE_2D;
}

bool TensorDescriptor::ReturnsZeroForNegOneRead(const GpuInfo& gpu_info) const {
  return storage_type_ == TensorStorageType::IMAGE_BUFFER &&
         gpu_info.SupportsZeroClampForImageBuffer();
}

absl::Status TensorDescriptor::CanCreateTensorWithShape(
    const GpuInfo& gpu_info, const BHWDC& shape) const {
  const std::vector<uint64_t> storage_dims = GetStorageDims(shape);
  const int slices = DivideRoundUp(shape.c, 4);
  const uint64_t allocation_size = GetSizeInBytesForShape(shape);
  const std::string common_desc = "Shape - " + ToString(shape) +
                                  ", data type - " + ToString(data_type_) + ".";
  if (allocation_size > gpu_info.GetMaxMemoryAllocationSize()) {
    return absl::ResourceExhaustedError(absl::StrCat(
        "Requested allocation size - ", allocation_size,
        " bytes. Max allocation size for this GPU - ",
        gpu_info.GetMaxMemoryAllocationSize(), " bytes. ", common_desc));
  }
  switch (storage_type_) {
    case TensorStorageType::BUFFER: {
      if (allocation_size > gpu_info.GetMaxBufferSize()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Buffer with size - ", allocation_size,
            " bytes can not be created. Max buffer size for this GPU - ",
            gpu_info.GetMaxBufferSize(), " bytes. ", common_desc));
      } else if (gpu_info.IsGlsl() &&
                 !IsGlslBufferSupportedType(gpu_info, data_type_)) {
        return absl::UnavailableError(
            "Buffer with this type unsupported on current device.");
      } else {
        return absl::OkStatus();
      }
    }
    case TensorStorageType::IMAGE_BUFFER: {
      const uint64_t image_width = storage_dims[0];
      if (!gpu_info.SupportsImageBuffer()) {
        return absl::UnavailableError(
            "TensorStorageType::IMAGE_BUFFER unsupported on current device.");
      } else if (image_width > gpu_info.GetMaxImageBufferWidth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image buffer with width - ", image_width,
            " can not be created. Max image buffer width for this GPU - ",
            gpu_info.GetMaxImageBufferWidth(), ". ", common_desc));
      } else if (allocation_size > gpu_info.GetMaxBufferSize()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Buffer with size - ", allocation_size,
            " bytes can not be created. Max buffer size for this GPU - ",
            gpu_info.GetMaxBufferSize(), " bytes. ", common_desc));
      } else if (gpu_info.IsGlsl() &&
                 !IsGlslBufferSupportedType(gpu_info, data_type_)) {
        return absl::UnavailableError(
            "Buffer with this type unsupported on current device.");
      } else {
        return absl::OkStatus();
      }
    }
    case TensorStorageType::TEXTURE_3D: {
      if (gpu_info.IsApiOpenCl() &&
          gpu_info.opencl_info.cl_version < OpenClVersion::kCl1_2 &&
          slices == 1) {
        return absl::InternalError(
            "clCreateImage3D (that used in CL 1.0/1.1) can not create image "
            "with depth = 1 by specification.");
      }
      const int image_width = storage_dims[0];
      const int image_height = storage_dims[1];
      const int image_depth = storage_dims[2];
      if (image_width > gpu_info.GetMaxImage3DWidth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image3D with width - ", image_width,
            " can not be created. Max Image3D width for this GPU - ",
            gpu_info.GetMaxImage3DWidth(), ". ", common_desc));
      } else if (image_height > gpu_info.GetMaxImage3DHeight()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image3D with height - ", image_height,
            " can not be created. Max Image3D height for this GPU - ",
            gpu_info.GetMaxImage3DHeight(), ". ", common_desc));
      } else if (image_depth > gpu_info.GetMaxImage3DDepth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image3D with depth - ", image_depth,
            " can not be created. Max Image3D depth for this GPU - ",
            gpu_info.GetMaxImage3DDepth(), ". ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    case TensorStorageType::TEXTURE_ARRAY: {
      // Bug on some Adreno. b/131099086
      if (gpu_info.IsApiOpenCl() && slices == 1 && gpu_info.IsAdreno() &&
          !gpu_info.adreno_info.support_one_layer_texture_array) {
        return absl::InternalError(
            "Image2DArray with layer = 1 works incorrect on some Adreno in "
            "OpenCL. Can not be created.");
      }
      const int image_width = storage_dims[0];
      const int image_height = storage_dims[1];
      const int image_layers = storage_dims[2];
      if (image_width > gpu_info.GetMaxImage2DWidth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2DArray with width - ", image_width,
            " can not be created. Max Image2DArray width for this GPU - ",
            gpu_info.GetMaxImage2DWidth(), ". ", common_desc));
      } else if (image_height > gpu_info.GetMaxImage2DHeight()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2DArray with height - ", image_height,
            " can not be created. Max Image2DArray height for this GPU - ",
            gpu_info.GetMaxImage2DHeight(), ". ", common_desc));
      } else if (image_layers > gpu_info.GetMaxImage2DArrayLayers()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2DArray with layers - ", image_layers,
            " can not be created. Max Image2DArray layers for this GPU - ",
            gpu_info.GetMaxImage2DArrayLayers(), ". ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    case TensorStorageType::TEXTURE_2D: {
      const int image_width = storage_dims[0];
      const int image_height = storage_dims[1];
      if (image_width > gpu_info.GetMaxImage2DWidth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2D with width - ", image_width,
            " can not be created. Max Image2D width for this GPU - ",
            gpu_info.GetMaxImage2DWidth(), ". ", common_desc));
      } else if (image_height > gpu_info.GetMaxImage2DHeight()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2D with height - ", image_height,
            " can not be created. Max Image2D height for this GPU - ",
            gpu_info.GetMaxImage2DHeight(), ". ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    case TensorStorageType::SINGLE_TEXTURE_2D: {
      const int image_width = storage_dims[0];
      const int image_height = storage_dims[1];
      if (shape.c > 4) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2D with channels - ", shape.c, " can not be created."));
      } else if (!gpu_info.SupportsImage2D(data_type_, shape.c)) {
        return absl::ResourceExhaustedError(
            "Image2D doesn't support this pixel layout.");
      } else if (image_width > gpu_info.GetMaxImage2DWidth()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2D with width - ", image_width,
            " can not be created. Max Image2D width for this GPU - ",
            gpu_info.GetMaxImage2DWidth(), ". ", common_desc));
      } else if (image_height > gpu_info.GetMaxImage2DHeight()) {
        return absl::ResourceExhaustedError(absl::StrCat(
            "Image2D with height - ", image_height,
            " can not be created. Max Image2D height for this GPU - ",
            gpu_info.GetMaxImage2DHeight(), ". ", common_desc));
      } else {
        return absl::OkStatus();
      }
    }
    default:
      return absl::UnimplementedError(
          "Can not create resources for unknown storage type.");
  }
}

absl::Status TensorDescriptor::CanCreateTensorWithShape(
    const GpuInfo& gpu_info, const BHWC& shape) const {
  const BHWDC shape5d(shape.b, shape.h, shape.w, 1, shape.c);
  return CanCreateTensorWithShape(gpu_info, shape5d);
}

absl::Status TensorDescriptor::UpdateToSupportedStorageType(
    const GpuInfo& gpu_info, const BHWC& shape) {
  const BHWDC shape5d(shape.b, shape.h, shape.w, 1, shape.c);
  return UpdateToSupportedStorageType(gpu_info, shape5d);
}

absl::Status TensorDescriptor::UpdateToSupportedStorageType(
    const GpuInfo& gpu_info, const BHWDC& shape) {
  if (CanCreateTensorWithShape(gpu_info, shape).ok()) {
    return absl::OkStatus();
  }
  if (gpu_info.IsApiMetal()) {
    storage_type_ = TensorStorageType::BUFFER;
    return CanCreateTensorWithShape(gpu_info, shape);
  }

  storage_type_ = TensorStorageType::IMAGE_BUFFER;
  if (gpu_info.SupportsImageBuffer() &&
      CanCreateTensorWithShape(gpu_info, shape).ok()) {
    return absl::OkStatus();
  }
  if (gpu_info.IsGlsl()) {
    // trying to use texture based types in GLSL
    // we do not support in GLSL buffers some types (bool/uint8/etc)
    storage_type_ = TensorStorageType::TEXTURE_2D;
    if (CanCreateTensorWithShape(gpu_info, shape).ok()) {
      return absl::OkStatus();
    }
    storage_type_ = TensorStorageType::TEXTURE_ARRAY;
    if (CanCreateTensorWithShape(gpu_info, shape).ok()) {
      return absl::OkStatus();
    }
  }
  storage_type_ = TensorStorageType::BUFFER;
  return CanCreateTensorWithShape(gpu_info, shape);
}

TensorDescriptor CreateBhwcTensorDescriptor(DataType data_type,
                                            TensorStorageType storage_type,
                                            const BHWC& shape) {
  TensorDescriptor tensor_desc =
      TensorDescriptor(data_type, storage_type, Layout::BHWC);
  tensor_desc.SetBHWCShape(shape);
  return tensor_desc;
}

TensorDescriptor CreateHwcTensorDescriptor(DataType data_type,
                                           TensorStorageType storage_type,
                                           const HWC& shape) {
  TensorDescriptor tensor_desc =
      TensorDescriptor(data_type, storage_type, Layout::HWC);
  tensor_desc.SetBHWCShape(BHWC(1, shape.h, shape.w, shape.c));
  return tensor_desc;
}

TensorStorageType GetStorageTypeForLinearTensor(const GpuInfo& gpu_info,
                                                DataType data_type,
                                                const Linear& shape) {
  auto storage_type = GetRecommendedStorageTypeForLinearTensor(gpu_info);
  TensorDescriptor tensor_desc =
      TensorDescriptor(data_type, storage_type, Layout::LINEAR);
  const BHWDC full_shape = BHWDC(1, 1, 1, 1, shape.v);
  if (!tensor_desc.CanCreateTensorWithShape(gpu_info, full_shape).ok()) {
    if (gpu_info.IsAdreno() && gpu_info.IsApiOpenCl()) {
      storage_type = TensorStorageType::IMAGE_BUFFER;
      tensor_desc = TensorDescriptor(data_type, storage_type, Layout::LINEAR);
      if (!tensor_desc.CanCreateTensorWithShape(gpu_info, full_shape).ok()) {
        storage_type = TensorStorageType::BUFFER;
      }
    } else {
      storage_type = TensorStorageType::BUFFER;
    }
  }
  return storage_type;
}

TensorDescriptor CreateConstantLinearTensorDescriptor(
    DataType data_type, TensorStorageType storage_type,
    const Tensor<Linear, DataType::FLOAT32>& src) {
  TensorDescriptor tensor_desc =
      TensorDescriptor(data_type, storage_type, Layout::LINEAR);
  tensor_desc.SetBHWCShape(BHWC(1, 1, 1, src.shape.v));
  tensor_desc.UploadData(src.Data());
  return tensor_desc;
}

TensorDescriptor CreateConstantLinearTensorDescriptor(
    DataType data_type, TensorStorageType storage_type,
    const Tensor<Linear, DataType::FLOAT16>& src) {
  TensorDescriptor tensor_desc =
      TensorDescriptor(data_type, storage_type, Layout::LINEAR);
  tensor_desc.SetBHWCShape(BHWC(1, 1, 1, src.shape.v));
  tensor_desc.data_.resize(src.shape.v * sizeof(half));
  std::memcpy(&tensor_desc.data_[0], src.Data(), src.shape.v * sizeof(half));
  return tensor_desc;
}

TensorDescriptor CreateConstantLinearTensorDescriptor(
    const GpuInfo& gpu_info, DataType data_type,
    const Tensor<Linear, DataType::FLOAT32>& src) {
  TensorDescriptor tensor_desc = TensorDescriptor(
      data_type, GetStorageTypeForLinearTensor(gpu_info, data_type, src.shape),
      Layout::LINEAR);
  tensor_desc.SetBHWCShape(BHWC(1, 1, 1, src.shape.v));
  tensor_desc.UploadData(src.Data());
  return tensor_desc;
}

TensorDescriptor CreateConstantLinearTensorDescriptor(
    const GpuInfo& gpu_info, DataType data_type,
    const Tensor<Linear, DataType::FLOAT16>& src) {
  TensorDescriptor tensor_desc = TensorDescriptor(
      data_type, GetStorageTypeForLinearTensor(gpu_info, data_type, src.shape),
      Layout::LINEAR);
  tensor_desc.SetBHWCShape(BHWC(1, 1, 1, src.shape.v));
  tensor_desc.data_.resize(src.shape.v * sizeof(half));
  std::memcpy(&tensor_desc.data_[0], src.Data(), src.shape.v * sizeof(half));
  return tensor_desc;
}

TensorDescriptor CreateConstantLinearTensorDescriptor(
    const GpuInfo& gpu_info, const Tensor<Linear, DataType::INT32>& src) {
  TensorDescriptor tensor_desc = TensorDescriptor(
      DataType::INT32,
      GetStorageTypeForLinearTensor(gpu_info, DataType::INT32, src.shape),
      Layout::LINEAR);
  tensor_desc.SetBHWCShape(BHWC(1, 1, 1, src.shape.v));
  tensor_desc.UploadData(src.Data());
  return tensor_desc;
}

TensorDescriptor CreateConstantHWVec4TensorDescriptor(
    DataType data_type, TensorStorageType storage_type, int width, int height,
    const uint8_t* data) {
  TensorDescriptor tensor_desc =
      TensorDescriptor(data_type, storage_type, Layout::HW);
  tensor_desc.SetBHWCShape(BHWC(1, height, width, 4));
  int data_size = height * width * 4 * SizeOf(data_type);
  tensor_desc.data_.resize(data_size);
  memcpy(tensor_desc.data_.data(), data, data_size);
  return tensor_desc;
}

}  // namespace ml_drift
