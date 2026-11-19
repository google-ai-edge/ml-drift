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

#include "ml_drift/metal/buffer.h"

#import <Metal/Metal.h>
#import <XCTest/XCTest.h>

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/types.h"

@interface BufferTest : XCTestCase
@end

@implementation BufferTest
- (void)setUp {
  [super setUp];
}

using ml_drift::half;

- (void)testBufferF32 {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (device == nil) {
    return;
  }

  const std::vector<float> data = {1.0f, 2.0f, 3.0f, -4.0f, 5.1f};
  ml_drift::metal::Buffer buffer;
  XCTAssertTrue(ml_drift::metal::CreateBuffer(sizeof(float) * 5, nullptr, device, &buffer).ok());
  XCTAssertTrue(buffer.WriteData(absl::MakeConstSpan(data.data(), data.size())).ok());
  std::vector<float> gpu_data;
  XCTAssertTrue(buffer.ReadData<float>(&gpu_data).ok());

  XCTAssertEqual(gpu_data.size(), data.size());
  for (int i = 0; i < gpu_data.size(); ++i) {
    XCTAssertEqual(gpu_data[i], data[i]);
  }
}

- (void)testBufferF16 {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (device == nil) {
    return;
  }

  const std::vector<half> data = {half(1.0f), half(2.0f), half(3.0f), half(-4.0f), half(5.1f)};
  ml_drift::metal::Buffer buffer;
  XCTAssertTrue(
      ml_drift::metal::CreateBuffer(sizeof(ml_drift::half) * 5, nullptr, device, &buffer).ok());
  XCTAssertTrue(buffer.WriteData(absl::MakeConstSpan(data.data(), data.size())).ok());
  std::vector<half> gpu_data;
  XCTAssertTrue(buffer.ReadData<half>(&gpu_data).ok());

  XCTAssertEqual(gpu_data.size(), data.size());
  for (int i = 0; i < gpu_data.size(); ++i) {
    XCTAssertEqual(gpu_data[i], data[i]);
  }
}

- (void)testCreateBufferAllocationFailureReturnsResourceExhausted {
  ml_drift::metal::Buffer buffer;
  absl::Status empty_status = ml_drift::metal::CreateBuffer(/*size_in_bytes=*/64, /*data=*/nullptr,
                                                            /*device=*/nil, &buffer);
  XCTAssertEqual(empty_status.code(), absl::StatusCode::kResourceExhausted);

  const uint8_t payload[16] = {};
  absl::Status data_status =
      ml_drift::metal::CreateBuffer(sizeof(payload), payload, /*device=*/nil, &buffer);
  XCTAssertEqual(data_status.code(), absl::StatusCode::kResourceExhausted);

  // Zero-byte buffers do not treat a nil MTLBuffer handle as an OOM failure.
  XCTAssertTrue(ml_drift::metal::CreateBuffer(/*size_in_bytes=*/0, /*data=*/nullptr,
                                              /*device=*/nil, &buffer)
                    .ok());
}

- (void)testCreateFromBufferDescriptorFailureReturnsResourceExhausted {
  ml_drift::BufferDescriptor empty_desc;
  empty_desc.size = 128;
  ml_drift::metal::Buffer buffer;
  absl::Status empty_status = buffer.CreateFromBufferDescriptor(empty_desc, /*device=*/nil);
  XCTAssertEqual(empty_status.code(), absl::StatusCode::kResourceExhausted);
  XCTAssertEqual(buffer.GetMemorySizeInBytes(), 0ULL);

  ml_drift::BufferDescriptor initialized_desc;
  initialized_desc.size = 16;
  initialized_desc.data.resize(16, 0xAB);
  absl::Status init_status = buffer.CreateFromBufferDescriptor(initialized_desc, /*device=*/nil);
  XCTAssertEqual(init_status.code(), absl::StatusCode::kResourceExhausted);
  XCTAssertEqual(buffer.GetMemorySizeInBytes(), 0ULL);
}

@end
