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

#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/status.h"
#include "ml_drift/metal/inference_context.h"
#include "ml_drift/metal/memory_manager.h"
#include "ml_drift/metal/metal_spatial_tensor.h"

#import <Metal/Metal.h>
#import <XCTest/XCTest.h>

using namespace ml_drift;
using namespace ml_drift::metal;

@interface ResidencyTest : XCTestCase
@end

@implementation ResidencyTest

- (void)testResidencySet {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  XCTAssertNotNil(device, @"Metal device is required for this test");

  // Only run test if residency sets are supported on this OS version
  if (@available(macOS 15.0, iOS 18.0, *)) {
    NSError* error = nil;
    MTLResidencySetDescriptor* set_desc = [[MTLResidencySetDescriptor alloc] init];
    set_desc.label = @"Test Residency Set";
    id<MTLResidencySet> residency_set = [device newResidencySetWithDescriptor:set_desc
                                                                        error:&error];
    XCTAssertNotNil(residency_set, @"Failed to create residency set: %@", error);

    // 1. Setup GpuModel
    GpuModel model;

    // Constant tensor (weights)
    ValueId const_id = 1;
    TensorDescriptor const_desc;
    const_desc.SetDataType(DataType::FLOAT32);
    const_desc.SetStorageType(TensorStorageType::BUFFER);
    const_desc.SetLayout(Layout::HWC);
    const_desc.SetBHWCShape(BHWC(1, 1, 1, 4));
    model.const_tensors[const_id] = const_desc;

    // Intermediate tensor (activation)
    ValueId inter_id = 2;
    TensorDescriptor inter_desc;
    inter_desc.SetDataType(DataType::FLOAT32);
    inter_desc.SetStorageType(TensorStorageType::BUFFER);
    inter_desc.SetLayout(Layout::HWC);
    inter_desc.SetBHWCShape(BHWC(1, 1, 1, 4));
    model.tensors[inter_id] = inter_desc;
    model.input_ids_and_refs.push_back({inter_id, inter_id});  // Force allocation

    // External mutable tensor descriptor in GpuModel
    ValueId ext_mut_id = 3;
    TensorDescriptor ext_mut_desc;
    ext_mut_desc.SetDataType(DataType::FLOAT32);
    ext_mut_desc.SetStorageType(TensorStorageType::BUFFER);
    ext_mut_desc.SetLayout(Layout::HWC);
    ext_mut_desc.SetBHWCShape(BHWC(1, 1, 1, 4));
    model.tensors[ext_mut_id] = ext_mut_desc;

    // 2. Setup CreateGpuModelInfo with external tensors
    CreateGpuModelInfo create_info;
    create_info.precision = CalculationsPrecision::F32;
    create_info.storage_type = TensorStorageType::BUFFER;

    // External immutable tensor (weights passed from outside)
    ValueId ext_imm_id = 4;
    MetalSpatialTensor ext_imm_tensor;
    TensorDescriptor ext_imm_desc;
    ext_imm_desc.SetDataType(DataType::FLOAT32);
    ext_imm_desc.SetStorageType(TensorStorageType::BUFFER);
    ext_imm_desc.SetLayout(Layout::HWC);
    ext_imm_desc.SetBHWCShape(BHWC(1, 1, 1, 4));
    XCTAssertTrue(ext_imm_tensor.CreateFromDescriptor(ext_imm_desc, device).ok());
    create_info.external_immutable_tensors[ext_imm_id] = &ext_imm_tensor;

    // External mutable tensor declaration
    create_info.external_mutable_tensors[ext_mut_id] = ext_mut_desc;

    // 3. Initialize InferenceContext
    InferenceContext context;
    absl::Status status = context.InitFromGpuModel(create_info, &model, device, nullptr);
    XCTAssertTrue(status.ok(), @"InitFromGpuModel failed: %s",
                  std::string(status.message()).c_str());

    // Bind external mutable tensor
    MetalSpatialTensor ext_mut_tensor;
    XCTAssertTrue(ext_mut_tensor.CreateFromDescriptor(ext_mut_desc, device).ok());
    status = context.SetTensor(ext_mut_id, &ext_mut_tensor);
    XCTAssertTrue(status.ok(), @"SetTensor failed: %s", std::string(status.message()).c_str());

    // 4. Test AddConstantsToResidencySet
    context.AddConstantsToResidencySet(residency_set);
    [residency_set commit];

    // Verify constant tensor buffer is resident
    MetalSpatialTensor* const_tensor_ptr = context.GetTensor(const_id);
    XCTAssertTrue(const_tensor_ptr != nullptr);
    id<MTLBuffer> const_buffer = const_tensor_ptr->GetBufferHandle();
    XCTAssertNotNil(const_buffer);
    XCTAssertTrue([residency_set.allAllocations containsObject:const_buffer]);

    // 5. Test AddIntermediatesToResidencySet
    context.AddIntermediatesToResidencySet(residency_set);
    [residency_set commit];

    // Verify intermediate tensor buffer is resident
    MetalSpatialTensor* inter_tensor_ptr = context.GetTensor(inter_id);
    XCTAssertTrue(inter_tensor_ptr != nullptr);
    id<MTLBuffer> inter_buffer = inter_tensor_ptr->GetBufferHandle();
    XCTAssertNotNil(inter_buffer);
    XCTAssertTrue([residency_set.allAllocations containsObject:inter_buffer]);

    // 6. Test AddExternalImmutableToResidencySet
    context.AddExternalImmutableToResidencySet(residency_set);
    [residency_set commit];

    // Verify external immutable tensor buffer is resident
    id<MTLBuffer> ext_imm_buffer = ext_imm_tensor.GetBufferHandle();
    XCTAssertNotNil(ext_imm_buffer);
    XCTAssertTrue([residency_set.allAllocations containsObject:ext_imm_buffer]);

    // 7. Test AddExternalMutableToResidencySet
    context.AddExternalMutableToResidencySet(residency_set);
    [residency_set commit];

    // Verify external mutable tensor buffer is resident
    id<MTLBuffer> ext_mut_buffer = ext_mut_tensor.GetBufferHandle();
    XCTAssertNotNil(ext_mut_buffer);
    XCTAssertTrue([residency_set.allAllocations containsObject:ext_mut_buffer]);
  }
}

@end
