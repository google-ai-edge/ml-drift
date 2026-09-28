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

#include "ml_drift/metal/metal_spatial_tensor.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"

#include <cmath>

#include "ml_drift/common/types.h"

#import <XCTest/XCTest.h>

#import <Metal/Metal.h>

@interface MetalSpatialTensorTest : XCTestCase
@end

@implementation MetalSpatialTensorTest
- (void)setUp {
  [super setUp];
}

using ml_drift::BHWC;
using ml_drift::BHWDC;
using ml_drift::DataType;
using ml_drift::Layout;
using ml_drift::TensorDescriptor;
using ml_drift::TensorStorageType;

namespace {

// Generates a reasonable distribution of numbers for testing, where i ranges
// from 0 to i_max.
double GenerateDouble(int i, int i_max, const ml_drift::TensorDescriptor& descriptor) {
  // val = [0, 1];
  const double val = static_cast<double>(i) / static_cast<double>(i_max);
  double transformed_val = sin(val * 2.0 * M_PI);
  auto data_type = descriptor.GetDataType();
  if (data_type == ml_drift::DataType::kUint16) {
    transformed_val = (transformed_val + 1) / 2;
    transformed_val *= std::numeric_limits<uint16_t>::max();
  }
  if (data_type == ml_drift::DataType::kUint32) {
    transformed_val = (transformed_val + 1) / 2;
    transformed_val *= std::numeric_limits<uint32_t>::max();
  }
  if (data_type == ml_drift::DataType::kInt16) {
    transformed_val *= std::numeric_limits<int16_t>::max();
  }
  if (data_type == ml_drift::DataType::kInt32) {
    transformed_val *= std::numeric_limits<int32_t>::max();
  }
  if (data_type == ml_drift::DataType::kFloat16) {
    transformed_val = ml_drift::half(transformed_val * 128.0);
  }
  if (data_type == ml_drift::DataType::kBool) {
    transformed_val = i % 2;
  }
  return transformed_val;
}

template <DataType T>
absl::Status TensorBHWCTest(const BHWC& shape, const TensorDescriptor& descriptor,
                            id<MTLDevice> device) {
  ml_drift::Tensor<BHWC, T> tensor_cpu;
  tensor_cpu.shape = shape;
  tensor_cpu.data.resize(shape.DimensionsProduct());
  for (int i = 0; i < tensor_cpu.data.size(); ++i) {
    tensor_cpu.data[i] = GenerateDouble(i, tensor_cpu.data.size() - 1, descriptor);
  }
  ml_drift::Tensor<BHWC, T> tensor_gpu;
  tensor_gpu.shape = shape;
  tensor_gpu.data.resize(shape.DimensionsProduct());
  for (int i = 0; i < tensor_gpu.data.size(); ++i) {
    tensor_gpu.data[i] = 0;
  }

  ml_drift::metal::MetalSpatialTensor tensor;
  ml_drift::TensorDescriptor descriptor_with_data = descriptor;
  descriptor_with_data.UploadData(tensor_cpu);
  ABSL_RETURN_IF_ERROR(tensor.CreateFromDescriptor(descriptor_with_data, device));
  ml_drift::TensorDescriptor output_descriptor;
  ABSL_RETURN_IF_ERROR(tensor.ToDescriptor(&output_descriptor, device));
  output_descriptor.DownloadData(&tensor_gpu);

  for (int i = 0; i < tensor_gpu.data.size(); ++i) {
    if (tensor_gpu.data[i] != tensor_cpu.data[i]) {
      return absl::InternalError("Wrong value at index - " + std::to_string(i) + ". GPU - " +
                                 std::to_string(tensor_gpu.data[i]) + ", CPU - " +
                                 std::to_string(tensor_cpu.data[i]));
    }
  }
  return absl::OkStatus();
}

template absl::Status TensorBHWCTest<DataType::kFloat32>(const BHWC& shape,
                                                         const TensorDescriptor& descriptor,
                                                         id<MTLDevice> device);
template absl::Status TensorBHWCTest<DataType::kInt32>(const BHWC& shape,
                                                       const TensorDescriptor& descriptor,
                                                       id<MTLDevice> device);

template absl::Status TensorBHWCTest<DataType::kInt16>(const BHWC& shape,
                                                       const TensorDescriptor& descriptor,
                                                       id<MTLDevice> device);

template absl::Status TensorBHWCTest<DataType::kInt8>(const BHWC& shape,
                                                      const TensorDescriptor& descriptor,
                                                      id<MTLDevice> device);
template absl::Status TensorBHWCTest<DataType::kUint32>(const BHWC& shape,
                                                        const TensorDescriptor& descriptor,
                                                        id<MTLDevice> device);

template absl::Status TensorBHWCTest<DataType::kUint16>(const BHWC& shape,
                                                        const TensorDescriptor& descriptor,
                                                        id<MTLDevice> device);

template absl::Status TensorBHWCTest<DataType::kUint8>(const BHWC& shape,
                                                       const TensorDescriptor& descriptor,
                                                       id<MTLDevice> device);

template absl::Status TensorBHWCTest<DataType::kBool>(const BHWC& shape,
                                                      const TensorDescriptor& descriptor,
                                                      id<MTLDevice> device);

template <DataType T>
absl::Status TensorBHWDCTest(const BHWDC& shape, const TensorDescriptor& descriptor,
                             id<MTLDevice> device) {
  ml_drift::Tensor<BHWDC, T> tensor_cpu;
  tensor_cpu.shape = shape;
  tensor_cpu.data.resize(shape.DimensionsProduct());
  for (int i = 0; i < tensor_cpu.data.size(); ++i) {
    tensor_cpu.data[i] = GenerateDouble(i, tensor_cpu.data.size() - 1, descriptor);
  }
  ml_drift::Tensor<BHWDC, T> tensor_gpu;
  tensor_gpu.shape = shape;
  tensor_gpu.data.resize(shape.DimensionsProduct());
  for (int i = 0; i < tensor_gpu.data.size(); ++i) {
    tensor_gpu.data[i] = 0;
  }

  ml_drift::metal::MetalSpatialTensor tensor;
  ml_drift::TensorDescriptor descriptor_with_data = descriptor;
  descriptor_with_data.UploadData(tensor_cpu);
  ABSL_RETURN_IF_ERROR(tensor.CreateFromDescriptor(descriptor_with_data, device));
  ml_drift::TensorDescriptor output_descriptor;
  ABSL_RETURN_IF_ERROR(tensor.ToDescriptor(&output_descriptor, device));
  output_descriptor.DownloadData(&tensor_gpu);

  for (int i = 0; i < tensor_gpu.data.size(); ++i) {
    if (tensor_gpu.data[i] != tensor_cpu.data[i]) {
      return absl::InternalError("Wrong value.");
    }
  }
  return absl::OkStatus();
}

template absl::Status TensorBHWDCTest<DataType::kFloat32>(const BHWDC& shape,
                                                          const TensorDescriptor& descriptor,
                                                          id<MTLDevice> device);
template absl::Status TensorBHWDCTest<DataType::kInt32>(const BHWDC& shape,
                                                        const TensorDescriptor& descriptor,
                                                        id<MTLDevice> device);

template absl::Status TensorBHWDCTest<DataType::kInt16>(const BHWDC& shape,
                                                        const TensorDescriptor& descriptor,
                                                        id<MTLDevice> device);

template absl::Status TensorBHWDCTest<DataType::kInt8>(const BHWDC& shape,
                                                       const TensorDescriptor& descriptor,
                                                       id<MTLDevice> device);
template absl::Status TensorBHWDCTest<DataType::kUint32>(const BHWDC& shape,
                                                         const TensorDescriptor& descriptor,
                                                         id<MTLDevice> device);

template absl::Status TensorBHWDCTest<DataType::kUint16>(const BHWDC& shape,
                                                         const TensorDescriptor& descriptor,
                                                         id<MTLDevice> device);

template absl::Status TensorBHWDCTest<DataType::kUint8>(const BHWDC& shape,
                                                        const TensorDescriptor& descriptor,
                                                        id<MTLDevice> device);

template absl::Status TensorBHWDCTest<DataType::kBool>(const BHWDC& shape,
                                                       const TensorDescriptor& descriptor,
                                                       id<MTLDevice> device);

template <DataType T>
absl::Status TensorTests(DataType data_type, TensorStorageType storage_type) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  ABSL_RETURN_IF_ERROR(
      TensorBHWCTest<T>(BHWC(1, 6, 7, 3), {data_type, storage_type, Layout::kHWC}, device));
  ABSL_RETURN_IF_ERROR(
      TensorBHWCTest<T>(BHWC(1, 1, 4, 12), {data_type, storage_type, Layout::kHWC}, device));
  ABSL_RETURN_IF_ERROR(
      TensorBHWCTest<T>(BHWC(1, 6, 1, 7), {data_type, storage_type, Layout::kHWC}, device));

  // Batch tests
  ABSL_RETURN_IF_ERROR(
      TensorBHWCTest<T>(BHWC(2, 6, 7, 3), {data_type, storage_type, Layout::kBHWC}, device));
  ABSL_RETURN_IF_ERROR(
      TensorBHWCTest<T>(BHWC(4, 1, 4, 12), {data_type, storage_type, Layout::kBHWC}, device));
  ABSL_RETURN_IF_ERROR(
      TensorBHWCTest<T>(BHWC(7, 6, 1, 7), {data_type, storage_type, Layout::kBHWC}, device));
  ABSL_RETURN_IF_ERROR(
      TensorBHWCTest<T>(BHWC(13, 7, 3, 3), {data_type, storage_type, Layout::kBHWC}, device));

  // 5D tests with batch = 1
  ABSL_RETURN_IF_ERROR(
      TensorBHWDCTest<T>(BHWDC(1, 6, 7, 4, 3), {data_type, storage_type, Layout::kHWDC}, device));
  ABSL_RETURN_IF_ERROR(
      TensorBHWDCTest<T>(BHWDC(1, 1, 4, 3, 12), {data_type, storage_type, Layout::kHWDC}, device));
  ABSL_RETURN_IF_ERROR(
      TensorBHWDCTest<T>(BHWDC(1, 6, 1, 7, 7), {data_type, storage_type, Layout::kHWDC}, device));

  // 5D tests
  ABSL_RETURN_IF_ERROR(
      TensorBHWDCTest<T>(BHWDC(2, 6, 7, 1, 3), {data_type, storage_type, Layout::kBHWDC}, device));
  ABSL_RETURN_IF_ERROR(
      TensorBHWDCTest<T>(BHWDC(4, 1, 4, 2, 12), {data_type, storage_type, Layout::kBHWDC}, device));
  ABSL_RETURN_IF_ERROR(
      TensorBHWDCTest<T>(BHWDC(7, 6, 1, 3, 7), {data_type, storage_type, Layout::kBHWDC}, device));
  ABSL_RETURN_IF_ERROR(
      TensorBHWDCTest<T>(BHWDC(13, 7, 3, 4, 3), {data_type, storage_type, Layout::kBHWDC}, device));
  return absl::OkStatus();
}

template absl::Status TensorTests<DataType::kFloat32>(DataType data_type,
                                                      TensorStorageType storage_type);
template absl::Status TensorTests<DataType::kInt32>(DataType data_type,
                                                    TensorStorageType storage_type);
template absl::Status TensorTests<DataType::kInt16>(DataType data_type,
                                                    TensorStorageType storage_type);
template absl::Status TensorTests<DataType::kInt8>(DataType data_type,
                                                   TensorStorageType storage_type);
template absl::Status TensorTests<DataType::kUint32>(DataType data_type,
                                                     TensorStorageType storage_type);
template absl::Status TensorTests<DataType::kUint16>(DataType data_type,
                                                     TensorStorageType storage_type);
template absl::Status TensorTests<DataType::kUint8>(DataType data_type,
                                                    TensorStorageType storage_type);
template absl::Status TensorTests<DataType::kBool>(DataType data_type,
                                                   TensorStorageType storage_type);

}  // namespace

- (void)testBufferF32 {
  auto status = TensorTests<DataType::kFloat32>(DataType::kFloat32, TensorStorageType::kBuffer);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testBufferF16 {
  auto status = TensorTests<DataType::kFloat32>(DataType::kFloat16, TensorStorageType::kBuffer);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testBufferInt32 {
  auto status = TensorTests<DataType::kInt32>(DataType::kInt32, TensorStorageType::kBuffer);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testBufferInt16 {
  auto status = TensorTests<DataType::kInt16>(DataType::kInt16, TensorStorageType::kBuffer);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testBufferInt8 {
  auto status = TensorTests<DataType::kInt8>(DataType::kInt8, TensorStorageType::kBuffer);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testBufferUint32 {
  auto status = TensorTests<DataType::kUint32>(DataType::kUint32, TensorStorageType::kBuffer);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testBufferUint16 {
  auto status = TensorTests<DataType::kUint16>(DataType::kUint16, TensorStorageType::kBuffer);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testBufferUint8 {
  auto status = TensorTests<DataType::kUint8>(DataType::kUint8, TensorStorageType::kBuffer);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testBufferBool {
  auto status = TensorTests<DataType::kBool>(DataType::kBool, TensorStorageType::kBuffer);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DF32 {
  auto status = TensorTests<DataType::kFloat32>(DataType::kFloat32, TensorStorageType::kTexture2D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DF16 {
  auto status = TensorTests<DataType::kFloat32>(DataType::kFloat16, TensorStorageType::kTexture2D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DInt32 {
  auto status = TensorTests<DataType::kInt32>(DataType::kInt32, TensorStorageType::kTexture2D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DInt16 {
  auto status = TensorTests<DataType::kInt16>(DataType::kInt16, TensorStorageType::kTexture2D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DInt8 {
  auto status = TensorTests<DataType::kInt8>(DataType::kInt8, TensorStorageType::kTexture2D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DUint32 {
  auto status = TensorTests<DataType::kUint32>(DataType::kUint32, TensorStorageType::kTexture2D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DUint16 {
  auto status = TensorTests<DataType::kUint16>(DataType::kUint16, TensorStorageType::kTexture2D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DUint8 {
  auto status = TensorTests<DataType::kUint8>(DataType::kUint8, TensorStorageType::kTexture2D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DBool {
  auto status = TensorTests<DataType::kBool>(DataType::kBool, TensorStorageType::kTexture2D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture3DF32 {
  auto status = TensorTests<DataType::kFloat32>(DataType::kFloat32, TensorStorageType::kTexture3D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture3DF16 {
  auto status = TensorTests<DataType::kFloat32>(DataType::kFloat16, TensorStorageType::kTexture3D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture3DInt32 {
  auto status = TensorTests<DataType::kInt32>(DataType::kInt32, TensorStorageType::kTexture3D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture3DInt16 {
  auto status = TensorTests<DataType::kInt16>(DataType::kInt16, TensorStorageType::kTexture3D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture3DInt8 {
  auto status = TensorTests<DataType::kInt8>(DataType::kInt8, TensorStorageType::kTexture3D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture3DUint32 {
  auto status = TensorTests<DataType::kUint32>(DataType::kUint32, TensorStorageType::kTexture3D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture3DUint16 {
  auto status = TensorTests<DataType::kUint16>(DataType::kUint16, TensorStorageType::kTexture3D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture3DUint8 {
  auto status = TensorTests<DataType::kUint8>(DataType::kUint8, TensorStorageType::kTexture3D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture3DBool {
  auto status = TensorTests<DataType::kBool>(DataType::kBool, TensorStorageType::kTexture3D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DArrayF32 {
  auto status =
      TensorTests<DataType::kFloat32>(DataType::kFloat32, TensorStorageType::kTextureArray);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DArrayF16 {
  auto status =
      TensorTests<DataType::kFloat32>(DataType::kFloat16, TensorStorageType::kTextureArray);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DArrayInt32 {
  auto status = TensorTests<DataType::kInt32>(DataType::kInt32, TensorStorageType::kTextureArray);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DArrayInt16 {
  auto status = TensorTests<DataType::kInt16>(DataType::kInt16, TensorStorageType::kTextureArray);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DArrayInt8 {
  auto status = TensorTests<DataType::kInt8>(DataType::kInt8, TensorStorageType::kTextureArray);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DArrayUint32 {
  auto status = TensorTests<DataType::kUint32>(DataType::kUint32, TensorStorageType::kTextureArray);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DArrayUint16 {
  auto status = TensorTests<DataType::kUint16>(DataType::kUint16, TensorStorageType::kTextureArray);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DArrayUint8 {
  auto status = TensorTests<DataType::kUint8>(DataType::kUint8, TensorStorageType::kTextureArray);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DArrayBool {
  auto status = TensorTests<DataType::kBool>(DataType::kBool, TensorStorageType::kTextureArray);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTextureBufferF32 {
  auto status =
      TensorTests<DataType::kFloat32>(DataType::kFloat32, TensorStorageType::kImageBuffer);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTextureBufferF16 {
  auto status =
      TensorTests<DataType::kFloat32>(DataType::kFloat16, TensorStorageType::kImageBuffer);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTextureBufferInt32 {
  auto status = TensorTests<DataType::kInt32>(DataType::kInt32, TensorStorageType::kImageBuffer);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTextureBufferInt16 {
  auto status = TensorTests<DataType::kInt16>(DataType::kInt16, TensorStorageType::kImageBuffer);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTextureBufferInt8 {
  auto status = TensorTests<DataType::kInt8>(DataType::kInt8, TensorStorageType::kImageBuffer);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTextureBufferUint32 {
  auto status = TensorTests<DataType::kUint32>(DataType::kUint32, TensorStorageType::kImageBuffer);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTextureBufferUint16 {
  auto status = TensorTests<DataType::kUint16>(DataType::kUint16, TensorStorageType::kImageBuffer);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTextureBufferUint8 {
  auto status = TensorTests<DataType::kUint8>(DataType::kUint8, TensorStorageType::kImageBuffer);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTextureBufferBool {
  auto status = TensorTests<DataType::kBool>(DataType::kBool, TensorStorageType::kImageBuffer);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

template <DataType T>
absl::Status SingleTextureTests(DataType data_type) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  ABSL_RETURN_IF_ERROR(TensorBHWCTest<T>(
      BHWC(1, 6, 14, 1), {data_type, TensorStorageType::kSingleTexture2D, Layout::kHWC}, device));
  ABSL_RETURN_IF_ERROR(TensorBHWCTest<T>(
      BHWC(1, 6, 14, 2), {data_type, TensorStorageType::kSingleTexture2D, Layout::kHWC}, device));

  // Batch tests
  ABSL_RETURN_IF_ERROR(TensorBHWCTest<T>(
      BHWC(7, 6, 14, 1), {data_type, TensorStorageType::kSingleTexture2D, Layout::kBHWC}, device));
  ABSL_RETURN_IF_ERROR(TensorBHWCTest<T>(
      BHWC(3, 6, 14, 2), {data_type, TensorStorageType::kSingleTexture2D, Layout::kBHWC}, device));

  // 5D tests with batch = 1
  ABSL_RETURN_IF_ERROR(
      TensorBHWDCTest<T>(BHWDC(1, 6, 14, 7, 1),
                         {data_type, TensorStorageType::kSingleTexture2D, Layout::kHWDC}, device));
  ABSL_RETURN_IF_ERROR(
      TensorBHWDCTest<T>(BHWDC(1, 6, 14, 4, 2),
                         {data_type, TensorStorageType::kSingleTexture2D, Layout::kHWDC}, device));

  // 5D tests
  ABSL_RETURN_IF_ERROR(
      TensorBHWDCTest<T>(BHWDC(7, 6, 14, 5, 1),
                         {data_type, TensorStorageType::kSingleTexture2D, Layout::kBHWDC}, device));
  ABSL_RETURN_IF_ERROR(
      TensorBHWDCTest<T>(BHWDC(3, 6, 14, 3, 2),
                         {data_type, TensorStorageType::kSingleTexture2D, Layout::kBHWDC}, device));
  return absl::OkStatus();
}

template absl::Status SingleTextureTests<DataType::kFloat32>(DataType data_type);
template absl::Status SingleTextureTests<DataType::kInt32>(DataType data_type);
template absl::Status SingleTextureTests<DataType::kInt16>(DataType data_type);
template absl::Status SingleTextureTests<DataType::kInt8>(DataType data_type);
template absl::Status SingleTextureTests<DataType::kUint32>(DataType data_type);
template absl::Status SingleTextureTests<DataType::kUint16>(DataType data_type);
template absl::Status SingleTextureTests<DataType::kUint8>(DataType data_type);

- (void)testSingleTextureFloat32 {
  auto status = SingleTextureTests<DataType::kFloat32>(DataType::kFloat32);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testSingleTextureFloat16 {
  auto status = SingleTextureTests<DataType::kFloat32>(DataType::kFloat16);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testSingleTextureInt32 {
  auto status = SingleTextureTests<DataType::kInt32>(DataType::kInt32);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testSingleTextureInt16 {
  auto status = SingleTextureTests<DataType::kInt16>(DataType::kInt16);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testSingleTextureInt8 {
  auto status = SingleTextureTests<DataType::kInt8>(DataType::kInt8);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testSingleTextureUint32 {
  auto status = SingleTextureTests<DataType::kUint32>(DataType::kUint32);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testSingleTextureUint16 {
  auto status = SingleTextureTests<DataType::kUint16>(DataType::kUint16);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testSingleTextureUint8 {
  auto status = SingleTextureTests<DataType::kUint8>(DataType::kUint8);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

@end
