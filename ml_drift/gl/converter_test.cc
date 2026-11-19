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

#include "ml_drift/gl/converter.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/status.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/gl/gl_buffer.h"
#include "ml_drift/gl/gl_spatial_tensor.h"
#include "ml_drift/gl/testing/gl_test.h"

namespace ml_drift {

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace gl {

TEST_F(OpenGlOperationTest, TensorToTensorConverterTest) {
  for (auto src_type : {DataType::kFloat32, DataType::kFloat16}) {
    for (auto src_storage : exec_env_.GetSupportedStorages(src_type)) {
      for (auto dst_type : {DataType::kFloat32, DataType::kFloat16}) {
        for (auto dst_storage : exec_env_.GetSupportedStorages(dst_type)) {
          TensorDescriptor src_desc(src_type, src_storage, Layout::kHWC);
          TensorDescriptor dst_desc(dst_type, dst_storage, Layout::kHWC);
          TensorToTensorConverter converter;
          ABSL_ASSERT_OK(converter.Init(exec_env_.GetGpuInfo(), src_desc, dst_desc));

          const BHWC shape(1, 18, 37, 17);
          TensorFloat32 src_tensor;
          src_tensor.shape = shape;
          src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
          for (int i = 0; i < src_tensor.data.size(); ++i) {
            src_tensor.data[i] = std::sin(i * 0.12345);
          }

          GlSpatialTensor src, dst;
          src_desc.UploadData(src_tensor);
          ABSL_ASSERT_OK(src.CreateFromDescriptor(src_desc));
          dst_desc.SetBHWCShape(shape);
          ABSL_ASSERT_OK(dst.CreateFromDescriptor(dst_desc));
          ABSL_ASSERT_OK(converter.Convert(&src, &dst));
          glFinish();
          ABSL_ASSERT_OK(dst.ToDescriptor(&dst_desc));

          TensorFloat32 dst_tensor;
          dst_desc.DownloadData(&dst_tensor);
          ASSERT_THAT(dst_tensor.data,
                      Pointwise(FloatNear(1e-3f), src_tensor.data));
        }
      }
    }
  }
}

TEST_F(OpenGlOperationTest, TensorToBHWCBufferConverterTest) {
  for (auto src_type : {DataType::kFloat32, DataType::kFloat16}) {
    for (auto src_storage : exec_env_.GetSupportedStorages(src_type)) {
      for (auto dst_type : {DataType::kFloat32}) {
        TensorDescriptor src_desc(src_type, src_storage, Layout::kHWC);

        BufferDescriptor dst_desc;
        dst_desc.element_type = dst_type;
        dst_desc.element_size = 1;
        dst_desc.memory_type = MemoryType::kGlobal;

        TensorToBHWCBufferConverter converter;
        ABSL_ASSERT_OK(converter.Init(exec_env_.GetGpuInfo(), src_desc, dst_desc));

        const BHWC shape(1, 18, 37, 17);
        TensorFloat32 src_tensor;
        src_tensor.shape = shape;
        src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
        for (int i = 0; i < src_tensor.data.size(); ++i) {
          src_tensor.data[i] = std::sin(i * 0.12345);
        }

        GlSpatialTensor src;
        src_desc.UploadData(src_tensor);
        ABSL_ASSERT_OK(src.CreateFromDescriptor(src_desc));

        GlBuffer dst;
        dst_desc.size = shape.DimensionsProduct() * SizeOf(dst_type);
        dst.CreateFromBufferDescriptor(dst_desc);
        ABSL_ASSERT_OK(converter.Convert(&src, &dst));
        glFinish();
        std::vector<float> dst_data;
        ABSL_ASSERT_OK(dst.ReadData(&dst_data));

        ASSERT_THAT(dst_data, Pointwise(FloatNear(1e-3f), src_tensor.data));
      }
    }
  }
}

TEST_F(OpenGlOperationTest, BHWCBufferToTensorConverterTest) {
  for (auto src_type : {DataType::kFloat32}) {
    for (auto dst_type : {DataType::kFloat32, DataType::kFloat16}) {
      for (auto dst_storage : exec_env_.GetSupportedStorages(src_type)) {
        TensorDescriptor dst_desc(dst_type, dst_storage, Layout::kHWC);

        BufferDescriptor src_desc;
        src_desc.element_type = src_type;
        src_desc.element_size = 1;
        src_desc.memory_type = MemoryType::kGlobal;

        BHWCBufferToTensorConverter converter;
        ABSL_ASSERT_OK(converter.Init(exec_env_.GetGpuInfo(), src_desc, dst_desc));

        const BHWC shape(1, 18, 37, 17);
        TensorFloat32 src_tensor;
        src_tensor.shape = shape;
        src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
        for (int i = 0; i < src_tensor.data.size(); ++i) {
          src_tensor.data[i] = std::sin(i * 0.12345);
        }

        GlBuffer src;
        src_desc.size = shape.DimensionsProduct() * SizeOf(src_type);
        src_desc.data.resize(src_desc.size);
        memcpy(src_desc.data.data(), src_tensor.data.data(), src_desc.size);
        src.CreateFromBufferDescriptor(src_desc);

        GlSpatialTensor dst;
        dst_desc.SetBHWCShape(shape);
        ABSL_ASSERT_OK(dst.CreateFromDescriptor(dst_desc));
        ABSL_ASSERT_OK(converter.Convert(&src, &dst));
        glFinish();
        ABSL_ASSERT_OK(dst.ToDescriptor(&dst_desc));
        TensorFloat32 dst_tensor;
        dst_desc.DownloadData(&dst_tensor);

        ASSERT_THAT(dst_tensor.data,
                    Pointwise(FloatNear(1e-3f), src_tensor.data));
      }
    }
  }
}

}  // namespace gl
}  // namespace ml_drift
