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

#ifndef ML_DRIFT_GL_GL_SPATIAL_TENSOR_H_
#define ML_DRIFT_GL_GL_SPATIAL_TENSOR_H_

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"
#include "ml_drift/gl/gl_texture_helper.h"
#include "ml_drift/gl/gpu_object.h"
#include "ml_drift/gl/portable_gl31.h"

namespace ml_drift {
namespace gl {

class GlSpatialTensor : public GPUObject, public ml_drift::GpuSpatialTensor {
 public:
  GlSpatialTensor()
      : memory_(-1), image_buffer_memory_(-1), memory_owner_(true) {}
  GlSpatialTensor(GLuint memory, bool memory_owner,
                  const ml_drift::TensorDescriptor& descriptor);
  GlSpatialTensor(GLuint memory, bool memory_owner, GLuint image_buffer_memory,
                  const ml_drift::TensorDescriptor& descriptor);

  // Move only
  GlSpatialTensor(GlSpatialTensor&& tensor);
  GlSpatialTensor& operator=(GlSpatialTensor&& tensor);
  GlSpatialTensor(const GlSpatialTensor&) = delete;
  GlSpatialTensor& operator=(const GlSpatialTensor&) = delete;

  ~GlSpatialTensor() override { Release(); }

  absl::Status GetGPUResources(const ml_drift::GPUObjectDescriptor* obj_ptr,
                               GPUResourcesWithValue* resources) const override;

  int Width() const override { return descriptor_.GetBHWDCShape().w; }
  int Height() const override { return descriptor_.GetBHWDCShape().h; }
  int Depth() const override { return descriptor_.GetBHWDCShape().d; }
  int Channels() const override { return descriptor_.GetBHWDCShape().c; }
  int Slices() const override {
    return ml_drift::DivideRoundUp(descriptor_.GetBHWDCShape().c, 4);
  }
  int Batch() const override { return descriptor_.GetBHWDCShape().b; }

  ml_drift::TensorDescriptor GetDescriptor() const override {
    return descriptor_;
  }
  ml_drift::DataType GetDataType() const { return descriptor_.GetDataType(); }
  ml_drift::TensorStorageType GetStorageType() const {
    return descriptor_.GetStorageType();
  }

  // for profiling and memory statistics
  uint64_t GetMemorySizeInBytes() const {
    return descriptor_.GetMemorySizeInBytes();
  }

  GLuint GetMemoryPtr() const;

  // This function returns buffer memory ptr for IMAGE_BUFFER instead of image
  // memory ptr.
  GLuint GetMemoryPtrForWriting() const;

  absl::Status CreateFromDescriptor(const ml_drift::TensorDescriptor& desc);
  absl::Status UploadDescriptorData(const ml_drift::TensorDescriptor& desc);
  absl::Status ToDescriptor(ml_drift::TensorDescriptor* desc) const;
  absl::Status WriteData(const uint8_t* ptr);

 private:
  absl::Status ReadData(uint8_t* ptr) const;

  void Release();

  GLuint memory_;
  GLuint image_buffer_memory_;  // for IMAGE_BUFFER/TEXTURE_2D/SINGLE_TEXTURE_2D
  bool memory_owner_;
  bool buffer_based_ = false;
  ml_drift::TensorDescriptor descriptor_;
};

absl::Status CreateTensor(const ml_drift::TensorDescriptor& descriptor,
                          GlSpatialTensor* result);

absl::Status CreateTensorShared(GLuint memory,
                                const ml_drift::TensorDescriptor& descriptor,
                                GlSpatialTensor* result);
}  // namespace gl
}  // namespace ml_drift

#endif  // ML_DRIFT_GL_GL_SPATIAL_TENSOR_H_
