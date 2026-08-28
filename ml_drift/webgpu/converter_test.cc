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

#include "ml_drift/webgpu/converter.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"
#include "ml_drift/webgpu/buffer.h"
#include "ml_drift/webgpu/testing/webgpu_test.h"
#include "ml_drift/webgpu/spatial_tensor.h"
#include "ml_drift/webgpu/webgpu_api_util.h"
#include "ml_drift/webgpu/webgpu_headers.h"

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace ml_drift {
namespace webgpu {

TEST_F(WebGpuOperationTest, TensorToTensorConverterTest) {
  for (auto src_type : {DataType::FLOAT32, DataType::FLOAT16}) {
    for (auto src_storage : exec_env_.GetSupportedStorages(src_type)) {
      for (auto dst_type : {DataType::FLOAT32, DataType::FLOAT16}) {
        for (auto dst_storage : exec_env_.GetSupportedStorages(dst_type)) {
          TensorDescriptor src_desc(src_type, src_storage, Layout::HWC);
          TensorDescriptor dst_desc(dst_type, dst_storage, Layout::HWC);
          TensorToTensorConverter converter;
          ABSL_ASSERT_OK(converter.Init(exec_env_.GetEnv(), src_desc, dst_desc));

          const BHWC shape(1, 18, 37, 17);
          TensorFloat32 src_tensor;
          src_tensor.shape = shape;
          src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
          for (int i = 0; i < src_tensor.data.size(); ++i) {
            src_tensor.data[i] = std::sin(i * 0.12345);
          }

          SpatialTensor src, dst;
          src_desc.UploadData(src_tensor);
          ABSL_ASSERT_OK(
              src.CreateFromDescriptor(exec_env_.GetEnv().device(), src_desc));
          dst_desc.SetBHWCShape(shape);
          ABSL_ASSERT_OK(
              dst.CreateFromDescriptor(exec_env_.GetEnv().device(), dst_desc));

          wgpu::CommandEncoder encoder =
              exec_env_.GetEnv().device().CreateCommandEncoder();
          wgpu::ComputePassEncoder compute_encoder = encoder.BeginComputePass();
          ABSL_ASSERT_OK(converter.Convert(exec_env_.GetEnv().device(),
                                      compute_encoder, &src, &dst));
          compute_encoder.End();
          wgpu::CommandBuffer cb = encoder.Finish();
          exec_env_.GetEnv().queue().Submit(1, &cb);

          ABSL_ASSERT_OK(dst.ToDescriptor(exec_env_.GetEnv().device(), &dst_desc));

          TensorFloat32 dst_tensor;
          dst_desc.DownloadData(&dst_tensor);
          ASSERT_THAT(src_tensor.data,
                      Pointwise(FloatNear(0.001f), dst_tensor.data));
        }
      }
    }
  }
}

TEST_F(WebGpuOperationTest, FloatTensorToFloatBHWCBufferConverterTest) {
  for (auto src_type : {DataType::FLOAT32, DataType::FLOAT16}) {
    for (auto src_storage : exec_env_.GetSupportedStorages(src_type)) {
      for (auto dst_type : {DataType::FLOAT32, DataType::FLOAT16}) {
        TensorDescriptor src_desc(src_type, src_storage, Layout::BHWC);

        TensorToBHWCBufferConverter converter;
        ABSL_ASSERT_OK(converter.Init(exec_env_.GetEnv(), src_desc, dst_type));

        const BHWC shape(3, 18, 23, 17);
        TensorFloat32 src_tensor;
        src_tensor.shape = shape;
        src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
        for (int i = 0; i < src_tensor.data.size(); ++i) {
          src_tensor.data[i] = std::sin(i * 0.12345);
        }

        SpatialTensor src;
        src_desc.UploadData(src_tensor);
        ABSL_ASSERT_OK(
            src.CreateFromDescriptor(exec_env_.GetEnv().device(), src_desc));

        const size_t dst_buf_size =
            shape.DimensionsProduct() * SizeOf(dst_type);
        const size_t dst_buf_size_aligned =
            AlignByN(dst_buf_size, SizeOf(dst_type) * 4);
        Buffer dst = CreateBufferStorage(exec_env_.GetEnv().device(),
                                         dst_buf_size_aligned);

        wgpu::CommandEncoder encoder =
            exec_env_.GetEnv().device().CreateCommandEncoder();
        wgpu::ComputePassEncoder compute_encoder = encoder.BeginComputePass();
        ABSL_ASSERT_OK(converter.Convert(exec_env_.GetEnv().device(),
                                    compute_encoder, &src, &dst));
        compute_encoder.End();
        wgpu::CommandBuffer cb = encoder.Finish();
        exec_env_.GetEnv().queue().Submit(1, &cb);

        std::vector<float> dst_data(shape.DimensionsProduct());
        if (dst_type == DataType::FLOAT16) {
          std::vector<half> dst_data_f16(shape.DimensionsProduct());
          ABSL_ASSERT_OK(ReadDataFromBuffer(
              exec_env_.GetEnv().device(), exec_env_.GetEnv().queue(),
              dst.GetMemoryHandle(), dst_buf_size, dst_data_f16.data()));
          for (int i = 0; i < dst_data_f16.size(); ++i) {
            dst_data[i] = static_cast<float>(dst_data_f16[i]);
          }
        } else {
          ABSL_ASSERT_OK(ReadDataFromBuffer(
              exec_env_.GetEnv().device(), exec_env_.GetEnv().queue(),
              dst.GetMemoryHandle(), dst_buf_size, dst_data.data()));
        }

        ASSERT_THAT(src_tensor.data, Pointwise(FloatNear(0.001f), dst_data));
      }
    }
  }
}

TEST_F(WebGpuOperationTest, BoolTensorToBoolBHWCBufferConverterTest) {
  const auto src_type = DataType::BOOL;
  const auto dst_type = DataType::BOOL;
  for (auto src_storage : exec_env_.GetSupportedStorages(src_type)) {
    TensorDescriptor src_desc(src_type, src_storage, Layout::BHWC);

    TensorToBHWCBufferConverter converter;
    ABSL_ASSERT_OK(converter.Init(exec_env_.GetEnv(), src_desc, dst_type));

    const BHWC shape(3, 18, 23, 17);
    TensorBool src_tensor;
    src_tensor.shape = shape;
    src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
    for (int i = 0; i < src_tensor.data.size(); ++i) {
      src_tensor.data[i] = ((i % 5) % 3) % 2;
    }

    SpatialTensor src;
    src_desc.UploadData(src_tensor);
    ABSL_ASSERT_OK(src.CreateFromDescriptor(exec_env_.GetEnv().device(), src_desc));

    const size_t dst_buf_size = shape.DimensionsProduct() * SizeOf(dst_type);
    const size_t dst_buf_size_aligned =
        AlignByN(dst_buf_size, SizeOf(dst_type) * 4);
    Buffer dst =
        CreateBufferStorage(exec_env_.GetEnv().device(), dst_buf_size_aligned);

    wgpu::CommandEncoder encoder =
        exec_env_.GetEnv().device().CreateCommandEncoder();
    wgpu::ComputePassEncoder compute_encoder = encoder.BeginComputePass();
    ABSL_ASSERT_OK(converter.Convert(exec_env_.GetEnv().device(), compute_encoder,
                                &src, &dst));
    compute_encoder.End();
    wgpu::CommandBuffer cb = encoder.Finish();
    exec_env_.GetEnv().queue().Submit(1, &cb);

    std::vector<uint8_t> dst_data(shape.DimensionsProduct());
    ABSL_ASSERT_OK(ReadDataFromBuffer(
        exec_env_.GetEnv().device(), exec_env_.GetEnv().queue(),
        dst.GetMemoryHandle(), dst_buf_size, dst_data.data()));
    EXPECT_EQ(src_tensor.data, dst_data);
  }
}

TEST_F(WebGpuOperationTest, FloatBHWCBufferToFloatTensorConverterTest) {
  for (auto src_type : {DataType::FLOAT32, DataType::FLOAT16}) {
    for (auto dst_type : {DataType::FLOAT32, DataType::FLOAT16}) {
      for (auto dst_storage : exec_env_.GetSupportedStorages(src_type)) {
        TensorDescriptor dst_desc(dst_type, dst_storage, Layout::HWC);

        BHWCBufferToTensorConverter converter;
        ABSL_ASSERT_OK(converter.Init(exec_env_.GetEnv(), src_type, dst_desc));

        const BHWC shape(1, 18, 37, 17);
        std::vector<float> src_data(shape.DimensionsProduct());
        for (int i = 0; i < src_data.size(); ++i) {
          src_data[i] = std::sin(i * 0.12345);
        }

        const size_t src_buf_size = src_data.size() * SizeOf(src_type);
        const size_t src_buf_size_aligned =
            AlignByN(src_buf_size, SizeOf(src_type) * 4);
        Buffer src = CreateBufferStorage(exec_env_.GetEnv().device(),
                                         src_buf_size_aligned);
        if (src_type == DataType::FLOAT16) {
          std::vector<half> src_data_f16(shape.DimensionsProduct());
          for (int i = 0; i < src_data_f16.size(); ++i) {
            src_data_f16[i] = src_data[i];
          }
          WriteDataToBuffer(exec_env_.GetEnv().queue(), src.GetMemoryHandle(),
                            src_buf_size, src_data_f16.data());
        } else {
          WriteDataToBuffer(exec_env_.GetEnv().queue(), src.GetMemoryHandle(),
                            src_buf_size, src_data.data());
        }

        SpatialTensor dst;
        dst_desc.SetBHWCShape(shape);
        ABSL_ASSERT_OK(
            dst.CreateFromDescriptor(exec_env_.GetEnv().device(), dst_desc));

        wgpu::CommandEncoder encoder =
            exec_env_.GetEnv().device().CreateCommandEncoder();
        wgpu::ComputePassEncoder compute_encoder = encoder.BeginComputePass();
        ABSL_ASSERT_OK(converter.Convert(exec_env_.GetEnv().device(),
                                    compute_encoder, &src, &dst));
        compute_encoder.End();
        wgpu::CommandBuffer cb = encoder.Finish();
        exec_env_.GetEnv().queue().Submit(1, &cb);

        ABSL_ASSERT_OK(dst.ToDescriptor(exec_env_.GetEnv().device(), &dst_desc));
        TensorFloat32 dst_tensor;
        dst_desc.DownloadData(&dst_tensor);

        ASSERT_THAT(src_data, Pointwise(FloatNear(0.001f), dst_tensor.data));
      }
    }
  }
}

TEST_F(WebGpuOperationTest, Int32BHWCBufferToInt32TensorConverterTest) {
  const auto src_type = DataType::INT32;
  const auto dst_type = DataType::INT32;
  for (auto dst_storage : exec_env_.GetSupportedStorages(src_type)) {
    TensorDescriptor dst_desc(dst_type, dst_storage, Layout::HWC);

    BHWCBufferToTensorConverter converter;
    ABSL_ASSERT_OK(converter.Init(exec_env_.GetEnv(), src_type, dst_desc));

    const BHWC shape(1, 18, 37, 17);
    std::vector<int> src_data(shape.DimensionsProduct());
    for (int i = 0; i < src_data.size(); ++i) {
      src_data[i] = i - 1024;
    }

    const size_t src_buf_size = src_data.size() * SizeOf(src_type);
    const size_t src_buf_size_aligned =
        AlignByN(src_buf_size, SizeOf(src_type) * 4);
    Buffer src =
        CreateBufferStorage(exec_env_.GetEnv().device(), src_buf_size_aligned);
    WriteDataToBuffer(exec_env_.GetEnv().queue(), src.GetMemoryHandle(),
                      src_buf_size, src_data.data());

    SpatialTensor dst;
    dst_desc.SetBHWCShape(shape);
    ABSL_ASSERT_OK(dst.CreateFromDescriptor(exec_env_.GetEnv().device(), dst_desc));

    wgpu::CommandEncoder encoder =
        exec_env_.GetEnv().device().CreateCommandEncoder();
    wgpu::ComputePassEncoder compute_encoder = encoder.BeginComputePass();
    ABSL_ASSERT_OK(converter.Convert(exec_env_.GetEnv().device(), compute_encoder,
                                &src, &dst));
    compute_encoder.End();
    wgpu::CommandBuffer cb = encoder.Finish();
    exec_env_.GetEnv().queue().Submit(1, &cb);

    ABSL_ASSERT_OK(dst.ToDescriptor(exec_env_.GetEnv().device(), &dst_desc));
    TensorInt32 dst_tensor;
    dst_desc.DownloadData(&dst_tensor);

    EXPECT_EQ(src_data, dst_tensor.data);
  }
}

TEST_F(WebGpuOperationTest, Int16BHWCBufferToInt16TensorConverterTest) {
  const auto src_type = DataType::INT16;
  const auto dst_type = DataType::INT16;
  for (auto dst_storage : exec_env_.GetSupportedStorages(src_type)) {
    TensorDescriptor dst_desc(dst_type, dst_storage, Layout::HWC);

    BHWCBufferToTensorConverter converter;
    ABSL_ASSERT_OK(converter.Init(exec_env_.GetEnv(), src_type, dst_desc));

    const BHWC shape(1, 18, 37, 17);
    std::vector<int16_t> src_data(shape.DimensionsProduct());
    for (int i = 0; i < src_data.size(); ++i) {
      src_data[i] = i - 1024;
    }

    const size_t src_buf_size = src_data.size() * SizeOf(src_type);
    const size_t src_buf_size_aligned =
        AlignByN(src_buf_size, SizeOf(src_type) * 4);
    Buffer src =
        CreateBufferStorage(exec_env_.GetEnv().device(), src_buf_size_aligned);
    WriteDataToBuffer(exec_env_.GetEnv().queue(), src.GetMemoryHandle(),
                      src_buf_size, src_data.data());

    SpatialTensor dst;
    dst_desc.SetBHWCShape(shape);
    ABSL_ASSERT_OK(dst.CreateFromDescriptor(exec_env_.GetEnv().device(), dst_desc));

    wgpu::CommandEncoder encoder =
        exec_env_.GetEnv().device().CreateCommandEncoder();
    wgpu::ComputePassEncoder compute_encoder = encoder.BeginComputePass();
    ABSL_ASSERT_OK(converter.Convert(exec_env_.GetEnv().device(), compute_encoder,
                                &src, &dst));
    compute_encoder.End();
    wgpu::CommandBuffer cb = encoder.Finish();
    exec_env_.GetEnv().queue().Submit(1, &cb);

    ABSL_ASSERT_OK(dst.ToDescriptor(exec_env_.GetEnv().device(), &dst_desc));
    Tensor<BHWC, DataType::INT16> dst_tensor;
    dst_desc.DownloadData(&dst_tensor);

    EXPECT_EQ(src_data, dst_tensor.data);
  }
}

TEST_F(WebGpuOperationTest, BoolBHWCBufferToBoolTensorConverterTest) {
  const auto src_type = DataType::BOOL;
  const auto dst_type = DataType::BOOL;
  for (auto dst_storage : exec_env_.GetSupportedStorages(src_type)) {
    TensorDescriptor dst_desc(dst_type, dst_storage, Layout::HWC);

    BHWCBufferToTensorConverter converter;
    ABSL_ASSERT_OK(converter.Init(exec_env_.GetEnv(), src_type, dst_desc));

    const BHWC shape(1, 18, 37, 17);
    std::vector<uint8_t> src_data(shape.DimensionsProduct());
    for (int i = 0; i < src_data.size(); ++i) {
      src_data[i] = (i % 7) % 2;
    }

    const size_t src_buf_size = src_data.size() * SizeOf(src_type);
    const size_t src_buf_size_aligned =
        AlignByN(src_buf_size, SizeOf(src_type) * 4);
    Buffer src =
        CreateBufferStorage(exec_env_.GetEnv().device(), src_buf_size_aligned);
    WriteDataToBuffer(exec_env_.GetEnv().queue(), src.GetMemoryHandle(),
                      src_buf_size, src_data.data());

    SpatialTensor dst;
    dst_desc.SetBHWCShape(shape);
    ABSL_ASSERT_OK(dst.CreateFromDescriptor(exec_env_.GetEnv().device(), dst_desc));

    wgpu::CommandEncoder encoder =
        exec_env_.GetEnv().device().CreateCommandEncoder();
    wgpu::ComputePassEncoder compute_encoder = encoder.BeginComputePass();
    ABSL_ASSERT_OK(converter.Convert(exec_env_.GetEnv().device(), compute_encoder,
                                &src, &dst));
    compute_encoder.End();
    wgpu::CommandBuffer cb = encoder.Finish();
    exec_env_.GetEnv().queue().Submit(1, &cb);

    ABSL_ASSERT_OK(dst.ToDescriptor(exec_env_.GetEnv().device(), &dst_desc));
    Tensor<BHWC, DataType::BOOL> dst_tensor;
    dst_desc.DownloadData(&dst_tensor);

    EXPECT_EQ(src_data, dst_tensor.data);
  }
}

}  // namespace webgpu
}  // namespace ml_drift
