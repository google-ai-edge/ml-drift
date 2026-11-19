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

#ifndef ML_DRIFT_METAL_COMMON_H_
#define ML_DRIFT_METAL_COMMON_H_

#import <Metal/Metal.h>

#include <map>
#include <utility>

#include "absl/status/status.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"

namespace ml_drift {
namespace metal {

MetalLanguageVersion ToMetalLanguageVersion(MTLLanguageVersion version);
MTLLanguageVersion GetMaxSupportedMTLLanguageVersion(id<MTLDevice> device);

absl::Status CreateComputeProgram(
    id<MTLDevice> device, const std::string& code,
    const std::string& function_name,
    const std::map<std::string, std::string>& macros,
    id<MTLComputePipelineState>* program);

absl::Status CreateComputeProgramWithArgumentBuffer(
    id<MTLDevice> device, const std::string& code,
    const std::string& function_name,
    const std::map<std::string, std::string>& macros,
    id<MTLComputePipelineState>* program,
    id<MTLArgumentEncoder>* arguments_encoder);

// ICB - indirect command buffer
absl::Status CreateComputeProgramWithICBSupport(
    id<MTLDevice> device, const std::string& code,
    const std::string& function_name,
    const std::map<std::string, std::string>& macros,
    id<MTLComputePipelineState>* program,
    id<MTLArgumentEncoder>* arguments_encoder);

int PixelFormatToSizeInBytes(MTLPixelFormat pixel_format);
MTLPixelFormat DataTypeToPixelFormat(DataType type, int channels_count = 4,
                                     bool normalized = false);

void WriteDataToBuffer(id<MTLBuffer> buffer, size_t buffer_offset,
                       id<MTLCommandQueue> command_queue, const void* data,
                       size_t data_size, bool wait_for_completion);

void ReadDataFromBuffer(id<MTLBuffer> buffer, size_t buffer_offset,
                        id<MTLDevice> device, void* data, size_t data_size);

void WriteDataToTexture2D(id<MTLTexture> texture,
                          id<MTLCommandQueue> command_queue, const void* data,
                          bool wait_for_completion);
void ReadDataFromTexture2D(id<MTLTexture> texture, id<MTLDevice> device,
                           void* data);

void WriteDataToTexture3D(id<MTLTexture> texture,
                          id<MTLCommandQueue> command_queue, const void* data,
                          bool wait_for_completion);
void ReadDataFromTexture3D(id<MTLTexture> texture, id<MTLDevice> device,
                           void* data);

void WriteDataToTexture2DArray(id<MTLTexture> texture,
                               id<MTLCommandQueue> command_queue,
                               const void* data, bool wait_for_completion);
void ReadDataFromTexture2DArray(id<MTLTexture> texture, id<MTLDevice> device,
                                void* data);

// Create empty CB, add to queue and wait until it is completed.
void WaitUntilCompleted(id<MTLCommandQueue> command_queue);

}  // namespace metal
}  // namespace ml_drift

#endif  // ML_DRIFT_METAL_COMMON_H_
