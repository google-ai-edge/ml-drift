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

#import <Metal/Metal.h>
#import <XCTest/XCTest.h>

#include "ml_drift/metal/buffer.h"
#include "ml_drift/metal/converter.h"
#include "ml_drift/metal/environment.h"
#include "ml_drift/metal/metal_spatial_tensor.h"
#include "ml_drift/metal/testing/test_util.h"

@interface ConverterTest : XCTestCase
@end

@implementation ConverterTest {
  ml_drift::metal::MetalExecutionEnvironment exec_env_;
}

namespace ml_drift {
namespace metal {

void TensorToBHWCBufferConverterTest(const TensorDescriptor& src_desc, const DataType& dst_type) {
  Environment env;

  TensorToBHWCBufferConverter converter;
  XCTAssertTrue(converter.Init(&env, src_desc, dst_type).ok());

  TensorDescriptor src_desc_copy = src_desc;
  BHWC shape(1, 18, 37, 17);
  if (src_desc.GetStorageType() == TensorStorageType::kSingleTexture2D) {
    shape.c = 1;
  }
  TensorFloat32 src_tensor;
  src_tensor.shape = shape;
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = ml_drift::half(std::sin(i * 0.12345));
  }

  MetalSpatialTensor src;
  src_desc_copy.UploadData(src_tensor);
  XCTAssertTrue(src.CreateFromDescriptor(src_desc_copy, env.device()).ok());

  Buffer dst;
  XCTAssertTrue(
      CreateBuffer(shape.DimensionsProduct() * SizeOf(dst_type), nullptr, env.device(), &dst).ok());

  id<MTLCommandQueue> command_queue = [env.device() newCommandQueue];
  id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  XCTAssertTrue(converter.Encode(encoder, &src, &dst).ok());
  [encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  std::vector<float> dst_data;
  if (dst_type == DataType::kFloat16) {
    std::vector<ml_drift::half> gpu_data_half;
    XCTAssertTrue(dst.ReadData(&gpu_data_half).ok());
    for (int i = 0; i < gpu_data_half.size(); ++i) {
      dst_data.push_back(static_cast<float>(gpu_data_half[i]));
    }
  } else {
    XCTAssertTrue(dst.ReadData(&dst_data).ok());
  }

  for (int i = 0; i < src_tensor.data.size(); ++i) {
    XCTAssertEqual(src_tensor.data[i], dst_data[i]);
  }
}

void BHWCBufferToTensorConverterTes(const DataType& src_type, const TensorDescriptor& dst_desc) {
  Environment env;

  BHWCBufferToTensorConverter converter;
  XCTAssertTrue(converter.Init(&env, src_type, dst_desc).ok());

  BHWC shape(1, 18, 37, 17);
  if (dst_desc.GetStorageType() == TensorStorageType::kSingleTexture2D) {
    shape.c = 2;
  }
  TensorFloat32 src_tensor;
  src_tensor.shape = shape;
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = ml_drift::half(std::sin(i * 0.12345));
  }

  Buffer src;
  if (src_type == DataType::kFloat16) {
    std::vector<ml_drift::half> gpu_data_half(src_tensor.data.size());
    for (int i = 0; i < src_tensor.data.size(); ++i) {
      gpu_data_half[i] = src_tensor.data[i];
    }
    XCTAssertTrue(CreateBuffer(shape.DimensionsProduct() * SizeOf(src_type), gpu_data_half.data(),
                               env.device(), &src)
                      .ok());
  } else {
    XCTAssertTrue(CreateBuffer(shape.DimensionsProduct() * SizeOf(src_type), src_tensor.data.data(),
                               env.device(), &src)
                      .ok());
  }

  TensorDescriptor dst_desc_copy = dst_desc;

  MetalSpatialTensor dst;
  dst_desc_copy.SetBHWCShape(shape);
  XCTAssertTrue(dst.CreateFromDescriptor(dst_desc_copy, env.device()).ok());

  id<MTLCommandQueue> command_queue = [env.device() newCommandQueue];
  id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  XCTAssertTrue(converter.Encode(encoder, &src, &dst).ok());
  [encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];

  XCTAssertTrue(dst.ToDescriptor(&dst_desc_copy, env.device()).ok());
  TensorFloat32 dst_tensor;
  dst_desc_copy.DownloadData(&dst_tensor);

  for (int i = 0; i < src_tensor.data.size(); ++i) {
    XCTAssertEqual(src_tensor.data[i], dst_tensor.data[i]);
  }
}

}  // namespace metal
}  // namespace ml_drift

- (void)testTensorToBHWCBufferConverter {
  for (auto src_type : {ml_drift::DataType::kFloat32, ml_drift::DataType::kFloat16}) {
    auto src_storages = exec_env_.GetSupportedStorages(src_type);
    src_storages.push_back(ml_drift::TensorStorageType::kSingleTexture2D);
    for (auto src_storage : src_storages) {
      for (auto dst_type : {ml_drift::DataType::kFloat32, ml_drift::DataType::kFloat16}) {
        ml_drift::TensorDescriptor src_desc(src_type, src_storage, ml_drift::Layout::kHWC);
        ml_drift::metal::TensorToBHWCBufferConverterTest(src_desc, dst_type);
      }
    }
  }
}
- (void)testBHWCBufferToTensorConverter {
  for (auto src_type : {ml_drift::DataType::kFloat32, ml_drift::DataType::kFloat16}) {
    for (auto dst_type : {ml_drift::DataType::kFloat32, ml_drift::DataType::kFloat16}) {
      auto dst_storages = exec_env_.GetSupportedStorages(dst_type);
      dst_storages.push_back(ml_drift::TensorStorageType::kSingleTexture2D);
      for (auto dst_storage : dst_storages) {
        ml_drift::TensorDescriptor dst_desc(dst_type, dst_storage, ml_drift::Layout::kHWC);
        ml_drift::metal::BHWCBufferToTensorConverterTes(src_type, dst_desc);
      }
    }
  }
}

@end
