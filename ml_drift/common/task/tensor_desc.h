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

#ifndef ML_DRIFT_COMMON_TASK_TENSOR_DESC_H_
#define ML_DRIFT_COMMON_TASK_TENSOR_DESC_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "flatbuffers/buffer.h"
#include "flatbuffers/flatbuffer_builder.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/serialization_base_generated.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

enum class TensorStorageType {
  UNKNOWN = 0,
  BUFFER = 1,
  IMAGE_BUFFER = 2,
  TEXTURE_2D = 3,
  TEXTURE_3D = 4,
  TEXTURE_ARRAY = 5,
  SINGLE_TEXTURE_2D = 6,
};

class TensorDescriptor : public GPUObjectDescriptor {
 public:
  bool IsTensorDescriptor() const override { return true; }
  enum class PhysicalLayout1D {
    kUnknown = 0,
    kDCHWBC4 = 1,
    kDHWBCC4 = 2,
  };
  enum class PhysicalLayout2D {
    kUnknown = 0,
    kXisWBDC4YisHC = 1,
    kXisCC4YisDHWB = 2,
  };
  TensorDescriptor() = default;
  TensorDescriptor(DataType data_type, TensorStorageType storage_type,
                   Layout layout)
      : data_type_(data_type), storage_type_(storage_type), layout_(layout) {}

  TensorDescriptor(DataType data_type, TensorStorageType storage_type,
                   Layout layout, PhysicalLayout1D physical_layout_1d)
      : data_type_(data_type),
        storage_type_(storage_type),
        layout_(layout),
        physical_layout_1d_(physical_layout_1d) {}

  TensorDescriptor(DataType data_type, TensorStorageType storage_type,
                   Layout layout, PhysicalLayout2D physical_layout_2d)
      : data_type_(data_type),
        storage_type_(storage_type),
        layout_(layout),
        physical_layout_2d_(physical_layout_2d) {}

  TensorDescriptor(const TensorDescriptor&) = default;
  TensorDescriptor& operator=(const TensorDescriptor&) = default;
  TensorDescriptor(TensorDescriptor&& desc);
  TensorDescriptor& operator=(TensorDescriptor&& desc);

  void CopyWithoutData(TensorDescriptor* desc) const;

  bool operator<(const TensorDescriptor& d) const {
    if (data_type_ != d.data_type_) {
      return data_type_ < d.data_type_;
    }
    if (storage_type_ != d.storage_type_) {
      return storage_type_ < d.storage_type_;
    }
    if (layout_ != d.layout_) {
      return layout_ < d.layout_;
    }
    if (physical_layout_1d_ != d.physical_layout_1d_) {
      return physical_layout_1d_ < d.physical_layout_1d_;
    }
    return physical_layout_2d_ < d.physical_layout_2d_;
  }

  bool operator!=(const TensorDescriptor& d) const { return !(*this == d); }

  void GetGpuResources(const BHWDC& tensor_shape,
                       GenericGPUResourcesWithValue* resources) const;

  absl::Status PerformConstExpr(const GpuInfo& gpu_info,
                                absl::string_view const_expr,
                                std::string* result) const override;

  absl::Status PerformSelector(const GpuInfo& gpu_info,
                               absl::string_view selector,
                               const std::vector<std::string>& args,
                               const std::vector<std::string>& template_args,
                               std::string* result) const override;

  GPUResources GetGPUResources(const GpuInfo& gpu_info) const override;

  uint64_t GetSizeInBytes() const override { return data_.size(); }
  size_t GetSizeInBytesForShape(const BHWDC& shape5d) const;

  bool HasAxis(Axis axis) const;

  absl::Status GetLinkingContextFromWriteSelector(
      const std::vector<std::string>& args, std::string* value_name,
      std::string* x_coord, std::string* y_coord, std::string* z_coord,
      std::string* s_coord, std::string* b_coord) const;

  template <DataType T>
  void UploadData(const Tensor<BHWDC, T>& src) {
    shape_ = src.shape;
    UploadData(src.data.data());
  }

  template <DataType T>
  void UploadData(const Tensor<BHWC, T>& src) {
    shape_ = BHWDC(src.shape.b, src.shape.h, src.shape.w, 1, src.shape.c);
    // Determine the correct data pointer based on whether the tensor owns its
    // data (`src.data`) or holds a span to external data (`src.spanned_data`).
    // We avoid using `src.Data()` directly here. `src.Data()` returns a pointer
    // to the underlying storage type (`ValueType`), which might differ from the
    // logical C++ type represented by `T`. For example, for `DataType::BOOL`,
    // `T` is `bool`, but `ValueType` is `uint8_t`.
    // Using `src.data.data()` or `src.spanned_data.data()` ensures that
    // `data_ptr` has the correct C++ pointer type (e.g., `const float*` for
    // `DataType::FLOAT32`, `const uint8_t*` for `DataType::BOOL`). This allows
    // the subsequent call to `UploadData(data_ptr)` to correctly deduce its
    // template argument `T_val` as the actual C++ type (e.g., `float`,
    // `uint8_t`), preventing potential type mismatches or incorrect
    // interpretations, especially when `ValueType` (like `uint8_t` for `bool`)
    // doesn't match the expected logical type `T` (`bool`).
    const auto* data_ptr =
        !src.data.empty() ? src.data.data() : src.spanned_data.data();
    UploadData(data_ptr);
  }

  void UploadData(const Tensor<HWC, DataType::FLOAT32>& src);

  template <typename T>
  void UploadData(const T* src) {
    data_.resize(GetSizeInBytesForShape(shape_));
    if (layout_ == Layout::LINEAR) {
      if (data_type_ == DataType::FLOAT16) {
        half* gpu_data = reinterpret_cast<half*>(data_.data());
        DataFromLinear(src, *this, gpu_data);
      } else {
        T* gpu_data = reinterpret_cast<T*>(data_.data());
        DataFromLinear(src, *this, gpu_data);
      }
    } else {  // HWC/BHWC/HWDC/BHWDC
      if (data_type_ == DataType::FLOAT16) {
        half* gpu_data = reinterpret_cast<half*>(data_.data());
        int gpu_data_byte_size = data_.size() / sizeof(half);
        DataFromBHWDC(src, shape_, *this,
                      absl::MakeSpan(gpu_data, gpu_data_byte_size));
      } else {
        T* gpu_data = reinterpret_cast<T*>(data_.data());
        int gpu_data_byte_size = data_.size() / sizeof(T);
        DataFromBHWDC(src, shape_, *this,
                      absl::MakeSpan(gpu_data, gpu_data_byte_size));
      }
    }
  }

  template <typename T>
  void UploadDataRaw(const absl::Span<const T> src) {
    data_.resize(GetMemorySizeInBytes());
    std::memcpy(data_.data(), src.data(), src.size() * sizeof(T));
  }

  template <DataType T>
  void DownloadData(Tensor<BHWDC, T>* dst) const {
    dst->shape = shape_;
    dst->data.resize(dst->shape.DimensionsProduct(), 0.0f);
    DownloadData(dst->data.data());
  }

  template <DataType T>
  void DownloadData(Tensor<BHWC, T>* dst) const {
    dst->shape = BHWC(shape_.b, shape_.h, shape_.w, shape_.c);
    dst->data.resize(dst->shape.DimensionsProduct(), 0.0f);
    DownloadData(dst->data.data());
  }

  template <typename T>
  void DownloadData(T* dst) const {
    if (data_type_ == DataType::FLOAT16) {
      const half* gpu_data = reinterpret_cast<const half*>(data_.data());
      DataToBHWDC(gpu_data, shape_, *this, dst);
    } else {
      const T* gpu_data = reinterpret_cast<const T*>(data_.data());
      DataToBHWDC(gpu_data, shape_, *this, dst);
    }
  }

  int GetLinearIndex(const BHWDC& shape5d, int b, int x, int y, int d, int s,
                     int sub_c) const;

  bool SupportsZeroClamp(const Axis& axis, const GpuInfo& gpu_info) const;
  bool CanReadOutOfBorder(const Axis& axis, const GpuInfo& gpu_info) const;
  bool IsLinear() const;
  bool Is2DStorage() const;
  bool IsCC4Layout() const {
    return (IsLinear() && physical_layout_1d_ == PhysicalLayout1D::kDHWBCC4) ||
           physical_layout_2d_ == PhysicalLayout2D::kXisCC4YisDHWB;
  }

  inline void SetDataType(DataType data_type) { data_type_ = data_type; }
  inline DataType GetDataType() const { return data_type_; }

  inline TensorStorageType GetStorageType() const { return storage_type_; }
  inline void SetStorageType(TensorStorageType storage_type) {
    storage_type_ = storage_type;
  }

  inline void SetLayout(Layout layout) { layout_ = layout; }
  inline Layout GetLayout() const { return layout_; }

  // applicable only for types that: IsLinear -> true.
  // In this case for address we have 1d component - addr (int)
  // If for addr == -1 this linear storage type returns zero value, this
  // function returns true, otherwise false
  bool ReturnsZeroForNegOneRead(const GpuInfo& gpu_info) const;

  absl::Status CanCreateTensorWithShape(const GpuInfo& gpu_info,
                                        const BHWDC& shape) const;

  absl::Status CanCreateTensorWithShape(const GpuInfo& gpu_info,
                                        const BHWC& shape) const;

  // Can update storage type if in the current storage type this tensor can not
  // be allocated with shape on specified device(gpu_info)
  // Usual scenario is to create new tensor_desc on base of another and may be
  // update storage type for new tensor_desc shape because it can be unsupported
  // with old storage type
  absl::Status UpdateToSupportedStorageType(const GpuInfo& gpu_info,
                                            const BHWC& shape);

  absl::Status UpdateToSupportedStorageType(const GpuInfo& gpu_info,
                                            const BHWDC& shape);

  std::vector<uint64_t> GetStorageDims(const BHWDC& shape) const;
  // shape must be initialized when using this function
  std::vector<uint64_t> GetStorageDims() const;
  // shape must be initialized when using this function
  int3 GetFullTensorRegion() const;
  // shape must be initialized when using this function
  uint64_t GetMemorySizeInBytes() const;
  // shape must be initialized when using this function
  int GetElementSize() const;

  void SetUseBufferForWriteOnlyTexture2d(bool value) {
    use_buffer_for_write_only_2d_texture_ = value;
  }
  bool GetUseBufferForWriteOnlyTexture2d() const {
    return use_buffer_for_write_only_2d_texture_;
  }

  void SetUseBufferForWriteOnlyImageBuffer(bool value) {
    use_buffer_for_write_only_image_buffer_ = value;
  }
  bool GetUseBufferForWriteOnlyImageBuffer() const {
    return use_buffer_for_write_only_image_buffer_;
  }

  void SetBHWCShape(const BHWC& new_shape) {
    shape_ = BHWDC(new_shape.b, new_shape.h, new_shape.w, 1, new_shape.c);
  }
  void SetBHWDCShape(const BHWDC& new_shape) { shape_ = new_shape; }
  BHWC GetBHWCShape() const {
    return BHWC(shape_.b, shape_.h, shape_.w, shape_.c);
  }
  const BHWDC& GetBHWDCShape() const { return shape_; }
  void SetData(std::vector<uint8_t>&& new_data) { data_ = std::move(new_data); }
  virtual absl::Span<const uint8_t> GetData() const {
    return absl::MakeSpan(data_);
  }

  bool operator==(const TensorDescriptor& d) const {
    return data_type_ == d.data_type_ && storage_type_ == d.storage_type_ &&
           layout_ == d.layout_ &&
           physical_layout_1d_ == d.physical_layout_1d_ &&
           physical_layout_2d_ == d.physical_layout_2d_;
  }

  // See go/absl-hash#making-hashable-types
  template <typename H>
  friend H AbslHashValue(H h, const TensorDescriptor& tensor_desc);

 private:
  friend flatbuffers::Offset<data::TensorDescriptor> Encode(
      const TensorDescriptor& desc, flatbuffers::FlatBufferBuilder* builder);
  friend void Decode(const data::TensorDescriptor* fb_desc,
                     TensorDescriptor* desc);

  friend TensorDescriptor CreateConstantLinearTensorDescriptor(
      DataType data_type, TensorStorageType storage_type,
      const Tensor<Linear, DataType::FLOAT32>& src);
  friend TensorDescriptor CreateConstantLinearTensorDescriptor(
      DataType data_type, TensorStorageType storage_type,
      const Tensor<Linear, DataType::FLOAT16>& src);
  friend TensorDescriptor CreateConstantLinearTensorDescriptor(
      const GpuInfo& gpu_info, DataType data_type,
      const Tensor<Linear, DataType::FLOAT32>& src);
  friend TensorDescriptor CreateConstantLinearTensorDescriptor(
      const GpuInfo& gpu_info, DataType data_type,
      const Tensor<Linear, DataType::FLOAT16>& src);
  friend TensorDescriptor CreateConstantLinearTensorDescriptor(
      const GpuInfo& gpu_info, const Tensor<Linear, DataType::INT32>& src);
  template <typename T>
  friend TensorDescriptor CreateRawConstantOHWITensorDescriptor(
      DataType data_type, const OHWI& shape, absl::Span<const T> weights_span);

  friend TensorDescriptor CreateConstantHWVec4TensorDescriptor(
      DataType data_type, TensorStorageType storage_type, int width, int height,
      const uint8_t* data);

  absl::Status PerformReadSelector(
      const GpuInfo& gpu_info, const std::vector<std::string>& args,
      const std::vector<std::string>& template_args, std::string* result) const;
  absl::Status PerformReadNearestSelector(const GpuInfo& gpu_info,
                                          const std::vector<std::string>& args,
                                          std::string* result) const;
  absl::Status PerformReadBilinearSelector(const GpuInfo& gpu_info,
                                           const std::vector<std::string>& args,
                                           std::string* result) const;
  absl::Status PerformReadPerChannelSelector(
      const GpuInfo& gpu_info, const std::vector<std::string>& args,
      const std::vector<std::string>& template_args, std::string* result) const;

  absl::Status PerformGetAddressSelector(const std::vector<std::string>& args,
                                         std::string* result) const;

  absl::Status PerformGetHandleSelector(const std::vector<std::string>& args,
                                        std::string* result) const;

  std::string StorageTypeToAddressType() const;

  absl::Status PerformWriteSelector(const GpuInfo& gpu_info,
                                    const std::vector<std::string>& args,
                                    std::string* result) const;

  absl::Status PerformWriteLinearSelector(const GpuInfo& gpu_info,
                                          const std::vector<std::string>& args,
                                          std::string* result) const;

  absl::Status PerformWrite2DSelector(const GpuInfo& gpu_info,
                                      const std::vector<std::string>& args,
                                      std::string* result) const;

  std::string Read(const GpuInfo& gpu_info, DataType read_as_type,
                   const std::vector<std::string>& coords) const;
  std::string Write(const GpuInfo& gpu_info, const std::string& var_name,
                    const std::vector<std::string>& coords) const;

  absl::Status MaybeGetDataTypeFromTemplateArgs(
      const std::vector<std::string>& template_args, DataType* result) const;

  std::string GetGlobalAddressNoDeclaration(const std::string& xc,
                                            const std::string& yc,
                                            const std::string& zc,
                                            const std::string& sc,
                                            const std::string& bc) const;

  std::vector<std::string> GetPhysicalCoordsWHS(const std::string& x,
                                                const std::string& y,
                                                const std::string& s) const;
  std::vector<std::string> GetPhysicalCoordsWHSB(const std::string& x,
                                                 const std::string& y,
                                                 const std::string& s,
                                                 const std::string& b) const;
  std::vector<std::string> GetPhysicalCoordsWHDS(const std::string& x,
                                                 const std::string& y,
                                                 const std::string& z,
                                                 const std::string& s) const;
  std::vector<std::string> GetPhysicalCoordsWHDSB(const std::string& x,
                                                  const std::string& y,
                                                  const std::string& z,
                                                  const std::string& s,
                                                  const std::string& b) const;
  std::vector<std::string> GetPhysicalCoords(const std::string& xc,
                                             const std::string& yc,
                                             const std::string& zc,
                                             const std::string& sc,
                                             const std::string& bc) const;
  std::vector<std::string> GetPhysicalCoordsLinear(const std::string& x) const;
  std::vector<std::string> GetPhysicalCoordsHW(const std::string& x,
                                               const std::string& y) const;

  // Initializes `xc`, `yc`, `zc`, `sc`, `bc` with the corresponding coordinates
  // for the appropriate axis based on the layout and offset.
  absl::Status ParseCoordsFromArgs(absl::Span<const std::string> args,
                                   int offset, std::string* xc, std::string* yc,
                                   std::string* zc, std::string* sc,
                                   std::string* bc) const;

  DataType data_type_ = DataType::UNKNOWN;
  TensorStorageType storage_type_ = TensorStorageType::UNKNOWN;

  // This field describes logical layout, actual(physical) GPU layout can be
  // totally different.
  Layout layout_ =
      Layout::UNKNOWN;  // Supported layouts is HWC, BHWC, HWDC, BHWDC
                        // HW and LINEAR (for constant objects only)

  // applicable only for BUFFER/IMAGE_BUFFER.
  PhysicalLayout1D physical_layout_1d_ = PhysicalLayout1D::kDCHWBC4;
  PhysicalLayout2D physical_layout_2d_ = PhysicalLayout2D::kXisWBDC4YisHC;

  // applicable only for TEXTURE_2D.
  // When Texture 2d created from buffer, we can use it as texture or as buffer.
  // This option allows to use texture 2d as buffer when we use it as dst
  // tensor(write only).
  // Currently supported only for Metal/OpenCL.
  // By default false.
  bool use_buffer_for_write_only_2d_texture_ = false;

  // applicable only for IMAGE_BUFFER.
  // We can use image buffer as image or as buffer.
  // This option allows to use image buffer as buffer when we use it as dst
  // tensor(write only).
  // Currently supported only for Metal/OpenCL.
  // By default true.
  bool use_buffer_for_write_only_image_buffer_ = true;

  // optional
  BHWDC shape_;
  std::vector<uint8_t> data_;
};

template <>
inline void TensorDescriptor::UploadDataRaw(const absl::Span<const float> src) {
  data_.resize(GetMemorySizeInBytes());
  if (data_type_ == DataType::FLOAT16) {
    // TODO: who/impjdi - Use xnn_run_unary_elementwise_nc instead.
    half* dst = reinterpret_cast<half*>(&data_[0]);
    for (int i = 0; i < shape_.DimensionsProduct(); ++i) {
      dst[i] = static_cast<half>(src.at(i));
    }
  } else {
    std::memcpy(data_.data(), src.data(), src.size() * sizeof(float));
  }
}

TensorDescriptor CreateBhwcTensorDescriptor(DataType data_type,
                                            TensorStorageType storage_type,
                                            const BHWC& shape);
TensorDescriptor CreateHwcTensorDescriptor(DataType data_type,
                                           TensorStorageType storage_type,
                                           const HWC& shape);

TensorStorageType GetStorageTypeForLinearTensor(const GpuInfo& gpu_info,
                                                DataType data_type,
                                                const Linear& shape);
TensorDescriptor CreateConstantLinearTensorDescriptor(
    DataType data_type, TensorStorageType storage_type,
    const Tensor<Linear, DataType::FLOAT32>& src);
TensorDescriptor CreateConstantLinearTensorDescriptor(
    DataType data_type, TensorStorageType storage_type,
    const Tensor<Linear, DataType::FLOAT16>& src);
TensorDescriptor CreateConstantLinearTensorDescriptor(
    const GpuInfo& gpu_info, DataType data_type,
    const Tensor<Linear, DataType::FLOAT32>& src);
TensorDescriptor CreateConstantLinearTensorDescriptor(
    const GpuInfo& gpu_info, DataType data_type,
    const Tensor<Linear, DataType::FLOAT16>& src);
TensorDescriptor CreateConstantLinearTensorDescriptor(
    const GpuInfo& gpu_info, const Tensor<Linear, DataType::INT32>& src);

TensorDescriptor CreateConstantHWVec4TensorDescriptor(
    DataType data_type, TensorStorageType storage_type, int width, int height,
    const uint8_t* data);

template <typename H>
H AbslHashValue(H h, const TensorDescriptor& tensor_desc) {
  return H::combine(std::move(h), tensor_desc.data_type_,
                    tensor_desc.storage_type_, tensor_desc.layout_,
                    tensor_desc.physical_layout_1d_,
                    tensor_desc.physical_layout_2d_);
}

template <typename FromType, typename ToType>
void DataFromLinear(const FromType* src, const TensorDescriptor& desc,
                    ToType* dst) {
  const int element_size = desc.GetElementSize();
  const Linear shape = Linear(desc.GetBHWCShape().c);
  const int slices = DivideRoundUp(shape.v, element_size);
  for (int s = 0; s < slices; ++s) {
    for (int c = 0; c < element_size; ++c) {
      FromType value;
      if (s * 4 + c < shape.v) {
        const int cpu_index = shape.LinearIndex({s * element_size + c});
        value = src[cpu_index];
      } else {
        value = 0;
      }
      int gpu_index = s * element_size + c;
      dst[gpu_index] = value;
    }
  }
}

template <typename FromType, typename ToType>
void DataFromBHWDC(const FromType* src, const BHWDC& shape,
                   const TensorDescriptor& desc, absl::Span<ToType> dst) {
  const int channels_alignment =
      desc.GetStorageType() == TensorStorageType::SINGLE_TEXTURE_2D ? shape.c
                                                                    : 4;
  const int slices = DivideRoundUp(shape.c, 4);
  for (int b = 0; b < shape.b; ++b) {
    for (int s = 0; s < slices; ++s) {
      for (int y = 0; y < shape.h; ++y) {
        for (int x = 0; x < shape.w; ++x) {
          for (int d = 0; d < shape.d; ++d) {
            for (int c = 0; c < channels_alignment; ++c) {
              FromType value;
              if (s * 4 + c < shape.c) {
                const int cpu_index =
                    shape.LinearIndex({b, y, x, d, s * 4 + c});
                value = src[cpu_index];
              } else {
                value = 0;
              }
              int gpu_index = desc.GetLinearIndex(shape, b, x, y, d, s, c);
              dst.at(gpu_index) = value;
            }
          }
        }
      }
    }
  }
}

template <typename FromType, typename ToType>
void DataToBHWDC(const FromType* src, const BHWDC& shape,
                 const TensorDescriptor& desc, ToType* dst) {
  const int channels_alignment =
      desc.GetStorageType() == TensorStorageType::SINGLE_TEXTURE_2D ? shape.c
                                                                    : 4;
  const int slices = DivideRoundUp(shape.c, 4);
  for (int b = 0; b < shape.b; ++b) {
    for (int s = 0; s < slices; ++s) {
      for (int y = 0; y < shape.h; ++y) {
        for (int x = 0; x < shape.w; ++x) {
          for (int d = 0; d < shape.d; ++d) {
            for (int c = 0; c < channels_alignment; ++c) {
              if (s * 4 + c >= shape.c) {
                continue;
              }
              int cpu_index = shape.LinearIndex({b, y, x, d, s * 4 + c});
              int gpu_index = desc.GetLinearIndex(shape, b, x, y, d, s, c);
              dst[cpu_index] = src[gpu_index];
            }
          }
        }
      }
    }
  }
}

constexpr std::array<TensorStorageType, 6> GetTensorStoragesTypes() {
  return {TensorStorageType::BUFFER,
          TensorStorageType::TEXTURE_ARRAY,
          TensorStorageType::TEXTURE_2D,
          TensorStorageType::TEXTURE_3D,
          TensorStorageType::SINGLE_TEXTURE_2D,
          TensorStorageType::IMAGE_BUFFER};
}

// SingleTexture2D not suitable for all tensor shapes.
constexpr std::array<TensorStorageType, 5>
GetTensorStoragesTypesWithoutSingleTexture2D() {
  return {TensorStorageType::BUFFER, TensorStorageType::TEXTURE_ARRAY,
          TensorStorageType::TEXTURE_2D, TensorStorageType::TEXTURE_3D,
          TensorStorageType::IMAGE_BUFFER};
}
std::string ToString(TensorStorageType type);
std::string ToStringWithShape(const TensorDescriptor& desc);

// For TensorDescriptor
inline TensorDescriptor* AsTensorDescriptor(GPUObjectDescriptor* desc) {
  return (desc && desc->IsTensorDescriptor())
             ? static_cast<TensorDescriptor*>(desc)
             : nullptr;
}

inline const TensorDescriptor* AsTensorDescriptor(
    const GPUObjectDescriptor* desc) {
  return (desc && desc->IsTensorDescriptor())
             ? static_cast<const TensorDescriptor*>(desc)
             : nullptr;
}

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_TASK_TENSOR_DESC_H_
