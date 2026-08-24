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

#include "ml_drift/metal/common.h"
#include "absl/strings/match.h"

#import <Metal/Metal.h>

#include <Availability.h>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/gpu_info.h"

// Compile-time message: print define name and value.
#define VALUE_TO_STRING(x) #x
#define VALUE(x) VALUE_TO_STRING(x)
#define VAR_NAME_VALUE(var) #var "=" VALUE(var)

namespace ml_drift {
namespace metal {
namespace {
MTLPixelFormat DataTypeToRGBAPixelFormat(DataType type, bool normalized) {
  switch (type) {
    case DataType::FLOAT32:
      return MTLPixelFormatRGBA32Float;
    case DataType::FLOAT16:
      return MTLPixelFormatRGBA16Float;
    case DataType::INT8:
      return normalized ? MTLPixelFormatRGBA8Snorm : MTLPixelFormatRGBA8Sint;
    case DataType::UINT8:
      return normalized ? MTLPixelFormatRGBA8Unorm : MTLPixelFormatRGBA8Uint;
    case DataType::INT16:
      return normalized ? MTLPixelFormatRGBA16Snorm : MTLPixelFormatRGBA16Sint;
    case DataType::BFLOAT16:
    case DataType::UINT16:
      return normalized ? MTLPixelFormatRGBA16Unorm : MTLPixelFormatRGBA16Uint;
    case DataType::INT32:
      return MTLPixelFormatRGBA32Sint;
    case DataType::UINT32:
      return MTLPixelFormatRGBA32Uint;
    case DataType::BOOL:
      return MTLPixelFormatRGBA8Uint;
    default:
      return MTLPixelFormatInvalid;
  }
}

MTLPixelFormat DataTypeToRGPixelFormat(DataType type, bool normalized) {
  switch (type) {
    case DataType::FLOAT32:
      return MTLPixelFormatRG32Float;
    case DataType::FLOAT16:
      return MTLPixelFormatRG16Float;
    case DataType::INT8:
      return normalized ? MTLPixelFormatRG8Snorm : MTLPixelFormatRG8Sint;
    case DataType::UINT8:
      return normalized ? MTLPixelFormatRG8Unorm : MTLPixelFormatRG8Uint;
    case DataType::INT16:
      return normalized ? MTLPixelFormatRG16Snorm : MTLPixelFormatRG16Sint;
    case DataType::BFLOAT16:
    case DataType::UINT16:
      return normalized ? MTLPixelFormatRG16Unorm : MTLPixelFormatRG16Uint;
    case DataType::INT32:
      return MTLPixelFormatRG32Sint;
    case DataType::UINT32:
      return MTLPixelFormatRG32Uint;
    case DataType::BOOL:
      return MTLPixelFormatRG8Uint;
    default:
      return MTLPixelFormatInvalid;
  }
}

MTLPixelFormat DataTypeToRPixelFormat(DataType type, bool normalized) {
  switch (type) {
    case DataType::FLOAT32:
      return MTLPixelFormatR32Float;
    case DataType::FLOAT16:
      return MTLPixelFormatR16Float;
    case DataType::INT8:
      return normalized ? MTLPixelFormatR8Snorm : MTLPixelFormatR8Sint;
    case DataType::UINT8:
      return normalized ? MTLPixelFormatR8Unorm : MTLPixelFormatR8Uint;
    case DataType::INT16:
      return normalized ? MTLPixelFormatR16Snorm : MTLPixelFormatR16Sint;
    case DataType::BFLOAT16:
    case DataType::UINT16:
      return normalized ? MTLPixelFormatR16Unorm : MTLPixelFormatR16Uint;
    case DataType::INT32:
      return MTLPixelFormatR32Sint;
    case DataType::UINT32:
      return MTLPixelFormatR32Uint;
    case DataType::BOOL:
      return MTLPixelFormatR8Uint;
    default:
      return MTLPixelFormatInvalid;
  }
}

absl::Status CreateFunction(id<MTLDevice> device, const std::string& code,
                            const std::string& function_name,
                            const std::map<std::string, std::string>& macros,
                            id<MTLFunction>* function) {
  MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
  [options setLanguageVersion:GetMaxSupportedMTLLanguageVersion(device)];

  NSMutableDictionary<NSString*, NSString*>* macros_dict = [NSMutableDictionary dictionary];
  for (const auto& pair : macros) {
    std::string key = pair.first;
    std::string value = pair.second;
    if (absl::StrContains(key, ' ')) {
      key = "\"" + key + "\"";
    }
    if (absl::StrContains(value, ' ')) {
      value = "\"" + value + "\"";
    }
    [macros_dict setObject:[NSString stringWithCString:value.c_str()
                                              encoding:[NSString defaultCStringEncoding]]
                    forKey:[NSString stringWithCString:key.c_str()
                                              encoding:[NSString defaultCStringEncoding]]];
  }

  if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
    [options setMathMode:MTLMathModeFast];
    [options setMathFloatingPointFunctions:MTLMathFloatingPointFunctionsFast];
  } else {
    [options setFastMathEnabled:YES];
  }
  [options setPreprocessorMacros:macros_dict];
  NSError* error = nil;
  NSString* code_ns = [NSString stringWithCString:code.c_str()
                                         encoding:[NSString defaultCStringEncoding]];
  id<MTLLibrary> library = [device newLibraryWithSource:code_ns options:options error:&error];
  if (!library) {
    NSString* errorString =
        [NSString stringWithFormat:@"newLibraryWithSource: %@", [error localizedDescription]];
    return absl::InternalError([errorString UTF8String]);
  }

  NSString* function_name_ns = [NSString stringWithCString:function_name.c_str()
                                                  encoding:[NSString defaultCStringEncoding]];
  *function = [library newFunctionWithName:function_name_ns];
  if (!*function) {
    NSString* errorString =
        [NSString stringWithFormat:@"newFunctionWithName: %@", [error localizedDescription]];
    return absl::InternalError([errorString UTF8String]);
  }
  return absl::OkStatus();
}
}  // namespace

MetalLanguageVersion ToMetalLanguageVersion(MTLLanguageVersion version) {
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000 || __IPHONE_OS_VERSION_MAX_ALLOWED >= 260000 || \
    __TV_OS_VERSION_MAX_ALLOWED >= 260000
  if (@available(macOS 26.0, iOS 26.0, tvOS 26.0, *)) {
    if (version == MTLLanguageVersion4_0) {
      return MetalLanguageVersion::kMetal4_0;
    }
  }
#endif
  if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
    if (version == MTLLanguageVersion3_2) {
      return MetalLanguageVersion::kMetal3_2;
    }
  }
  if (@available(macOS 14.0, iOS 17.0, tvOS 17.0, *)) {
    if (version == MTLLanguageVersion3_1) {
      return MetalLanguageVersion::kMetal3_1;
    }
  }
  if (@available(macOS 13.0, iOS 16.0, tvOS 16.0, *)) {
    if (version == MTLLanguageVersion3_0) {
      return MetalLanguageVersion::kMetal3_0;
    }
  }
  if (@available(macOS 12.0, iOS 15.0, tvOS 15.0, *)) {
    if (version == MTLLanguageVersion2_4) {
      return MetalLanguageVersion::kMetal2_4;
    }
  }
  if (@available(macOS 11.0, iOS 14.0, tvOS 14.0, *)) {
    if (version == MTLLanguageVersion2_3) {
      return MetalLanguageVersion::kMetal2_3;
    }
  }
  if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
    if (version == MTLLanguageVersion2_2) {
      return MetalLanguageVersion::kMetal2_2;
    }
  }
  if (@available(macOS 10.14, iOS 12.0, tvOS 12.0, *)) {
    if (version == MTLLanguageVersion2_1) {
      return MetalLanguageVersion::kMetal2_1;
    }
  }
  if (@available(macOS 10.13, iOS 11.0, tvOS 11.0, *)) {
    if (version == MTLLanguageVersion2_0) {
      return MetalLanguageVersion::kMetal2_0;
    }
  }
  if (@available(macOS 10.12, iOS 10.0, tvOS 10.0, *)) {
    if (version == MTLLanguageVersion1_2) {
      return MetalLanguageVersion::kMetal1_2;
    }
  }
  return MetalLanguageVersion::kMetal1_1;
}

MTLLanguageVersion GetMaxSupportedMTLLanguageVersion(id<MTLDevice> device) {
  // MTLLanguageVersion1_0 deprecated
  MTLLanguageVersion max_supported_version = MTLLanguageVersion1_1;
  if (@available(macOS 12.0, iOS 15.0, tvOS 15.0, *)) {
    max_supported_version = MTLLanguageVersion2_4;
  } else if (@available(macOS 11.0, iOS 14.0, tvOS 14.0, *)) {
    max_supported_version = MTLLanguageVersion2_3;
  } else if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
    max_supported_version = MTLLanguageVersion2_2;
  } else if (@available(macOS 10.14, iOS 12.0, tvOS 12.0, *)) {
    max_supported_version = MTLLanguageVersion2_1;
  } else if (@available(macOS 10.13, iOS 11.0, tvOS 11.0, *)) {
    max_supported_version = MTLLanguageVersion2_0;
  } else if (@available(macOS 10.12, iOS 10.0, tvOS 10.0, *)) {
    max_supported_version = MTLLanguageVersion1_2;
  } else if (@available(macOS 10.11, iOS 9.0, tvOS 9.0, *)) {
    max_supported_version = MTLLanguageVersion1_1;
  } else {
    max_supported_version = MTLLanguageVersion1_1;
  }

  if (@available(macOS 13.0, iOS 16.0, tvOS 16.0, *)) {
    if ([device supportsFamily:MTLGPUFamilyMetal3]) {
      if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
        max_supported_version = MTLLanguageVersion3_2;
      } else if (@available(macOS 14.0, iOS 17.0, tvOS 17.0, *)) {
        max_supported_version = MTLLanguageVersion3_1;
      } else {
        max_supported_version = MTLLanguageVersion3_0;
      }
    }
  }

#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000 || __IPHONE_OS_VERSION_MAX_ALLOWED >= 260000 || \
    __TV_OS_VERSION_MAX_ALLOWED >= 260000
  if (@available(macOS 26.0, iOS 26.0, tvOS 26.0, *)) {
    // MTLGPUFamilyMetal4 is not available sometimes? Fails to build with this options:
    // --config=ios_sim_arm64 --config=apple_xcode_compile --compilation_mode=opt
    // --xcode_version=26.0.1
    // MTLGPUFamilyMetal4 API_AVAILABLE(macos(26.0), ios(26.0)) = 5002,
    MTLGPUFamily family_v4 = static_cast<MTLGPUFamily>(5002);
    if ([device supportsFamily:family_v4]) {
      max_supported_version = MTLLanguageVersion4_0;
    }
  }
#endif
  return max_supported_version;
}

absl::Status CreateComputeProgram(id<MTLDevice> device, const std::string& code,
                                  const std::string& function_name,
                                  const std::map<std::string, std::string>& macros,
                                  id<MTLComputePipelineState>* program) {
  id<MTLFunction> function;
  ABSL_RETURN_IF_ERROR(CreateFunction(device, code, function_name, macros, &function));

  NSError* error = nil;
  *program = [device newComputePipelineStateWithFunction:function error:&error];
  if (!*program) {
    NSString* errorString =
        [NSString stringWithFormat:@"newComputePipelineStateWithFunction error: %@",
                                   [error localizedDescription]];
    return absl::InternalError([errorString UTF8String]);
  }
  return absl::OkStatus();
}

absl::Status CreateComputeProgramWithArgumentBuffer(
    id<MTLDevice> device, const std::string& code, const std::string& function_name,
    const std::map<std::string, std::string>& macros, id<MTLComputePipelineState>* program,
    id<MTLArgumentEncoder>* arguments_encoder) {
  if (@available(macOS 10.13, iOS 11.0, tvOS 11.0, *)) {
    id<MTLFunction> function;
    ABSL_RETURN_IF_ERROR(CreateFunction(device, code, function_name, macros, &function));
    *arguments_encoder = [function newArgumentEncoderWithBufferIndex:0];
    if (!*arguments_encoder) {
      return absl::InternalError("Failed to get MTLArgumentEncoder.");
    }
    MTLComputePipelineDescriptor* pipeline_desc = [[MTLComputePipelineDescriptor alloc] init];
    pipeline_desc.computeFunction = function;
    NSError* error = nil;
    *program = [device newComputePipelineStateWithDescriptor:pipeline_desc
                                                     options:MTLPipelineOptionNone
                                                  reflection:nullptr
                                                       error:&error];
    if (!*program) {
      NSString* error_string =
          [NSString stringWithFormat:@"newComputePipelineStateWithDescriptor: %@",
                                     [error localizedDescription]];
      return absl::InternalError([error_string UTF8String]);
    }
    return absl::OkStatus();
  } else {
    return absl::InternalError("Metal argument buffers available since ios 11, tvos 11 or macos "
                               "10.13.");
  }
}

absl::Status CreateComputeProgramWithICBSupport(id<MTLDevice> device, const std::string& code,
                                                const std::string& function_name,
                                                const std::map<std::string, std::string>& macros,
                                                id<MTLComputePipelineState>* program,
                                                id<MTLArgumentEncoder>* arguments_encoder) {
  if (@available(macOS 11.00, iOS 13.0, tvOS 13.0, *)) {
    id<MTLFunction> function;
    ABSL_RETURN_IF_ERROR(CreateFunction(device, code, function_name, macros, &function));
    *arguments_encoder = [function newArgumentEncoderWithBufferIndex:0];
    if (!*arguments_encoder) {
      return absl::InternalError("Failed to get MTLArgumentEncoder.");
    }
    MTLComputePipelineDescriptor* pipeline_desc = [[MTLComputePipelineDescriptor alloc] init];
    pipeline_desc.computeFunction = function;
    pipeline_desc.supportIndirectCommandBuffers = TRUE;
    NSError* error = nil;
    *program = [device newComputePipelineStateWithDescriptor:pipeline_desc
                                                     options:MTLPipelineOptionNone
                                                  reflection:nullptr
                                                       error:&error];
    if (!*program) {
      NSString* error_string =
          [NSString stringWithFormat:@"newComputePipelineStateWithDescriptor: %@",
                                     [error localizedDescription]];
      return absl::InternalError([error_string UTF8String]);
    }
    return absl::OkStatus();
  } else {
    return absl::InternalError("Indirect compute command buffer available since ios 13, tvos 13 "
                               "or macos 11.00");
  }
}

int PixelFormatToSizeInBytes(MTLPixelFormat pixel_format) {
  if (pixel_format == MTLPixelFormatRGBA32Uint || pixel_format == MTLPixelFormatRGBA32Sint ||
      pixel_format == MTLPixelFormatRGBA32Float) {
    return 16;
  } else if (pixel_format == MTLPixelFormatRGBA16Unorm ||
             pixel_format == MTLPixelFormatRGBA16Snorm ||
             pixel_format == MTLPixelFormatRGBA16Uint || pixel_format == MTLPixelFormatRGBA16Sint ||
             pixel_format == MTLPixelFormatRGBA16Float) {
    return 8;
  } else if (pixel_format == MTLPixelFormatRGBA8Unorm || pixel_format == MTLPixelFormatRGBA8Snorm ||
             pixel_format == MTLPixelFormatRGBA8Uint || pixel_format == MTLPixelFormatRGBA8Sint) {
    return 4;
  } else if (pixel_format == MTLPixelFormatRG32Uint || pixel_format == MTLPixelFormatRG32Sint ||
             pixel_format == MTLPixelFormatRG32Float) {
    return 8;
  } else if (pixel_format == MTLPixelFormatRG16Unorm || pixel_format == MTLPixelFormatRG16Snorm ||
             pixel_format == MTLPixelFormatRG16Uint || pixel_format == MTLPixelFormatRG16Sint ||
             pixel_format == MTLPixelFormatRG16Float) {
    return 4;
  } else if (pixel_format == MTLPixelFormatRG8Unorm || pixel_format == MTLPixelFormatRG8Snorm ||
             pixel_format == MTLPixelFormatRG8Uint || pixel_format == MTLPixelFormatRG8Sint) {
    return 2;
  } else if (pixel_format == MTLPixelFormatR32Uint || pixel_format == MTLPixelFormatR32Sint ||
             pixel_format == MTLPixelFormatR32Float) {
    return 4;
  } else if (pixel_format == MTLPixelFormatR16Unorm || pixel_format == MTLPixelFormatR16Snorm ||
             pixel_format == MTLPixelFormatR16Uint || pixel_format == MTLPixelFormatR16Sint ||
             pixel_format == MTLPixelFormatR16Float) {
    return 2;
  } else if (pixel_format == MTLPixelFormatR8Unorm || pixel_format == MTLPixelFormatR8Snorm ||
             pixel_format == MTLPixelFormatR8Uint || pixel_format == MTLPixelFormatR8Sint) {
    return 1;
  }
  return -1;
}

MTLPixelFormat DataTypeToPixelFormat(DataType type, int channels_count, bool normalized) {
  if (channels_count == 4) {
    return DataTypeToRGBAPixelFormat(type, normalized);
  } else if (channels_count == 2) {
    return DataTypeToRGPixelFormat(type, normalized);
  } else if (channels_count == 1) {
    return DataTypeToRPixelFormat(type, normalized);
  } else {
    return MTLPixelFormatInvalid;
  }
}

void WriteDataToBuffer(id<MTLBuffer> buffer, size_t buffer_offset, id<MTLCommandQueue> command_queue,
                       const void* data, size_t data_size, bool wait_for_completion) {
  @autoreleasepool {
    // Command buffers created by the commandBuffer method retain data that is needed for execution.
    id<MTLBuffer> temp_buffer =
        [[command_queue device] newBufferWithBytes:data
                                            length:data_size
                                           options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLBlitCommandEncoder> blitCommandEncoder = [command_buffer blitCommandEncoder];
    [blitCommandEncoder copyFromBuffer:temp_buffer
                          sourceOffset:0
                              toBuffer:buffer
                     destinationOffset:buffer_offset
                                  size:data_size];
    [blitCommandEncoder endEncoding];
    [command_buffer commit];
    if (wait_for_completion) {
      [command_buffer waitUntilCompleted];
    }
  }
}

// Function to read data from a Metal buffer.
void ReadDataFromBuffer(id<MTLBuffer> buffer, size_t buffer_offset, id<MTLDevice> device,
                        void* data, size_t data_size) {
  @autoreleasepool {
    // Create a temporary buffer with shared storage mode to be accessible by the CPU.
    id<MTLBuffer> temp_buffer = [device newBufferWithLength:data_size
                                                    options:MTLResourceStorageModeShared];

    id<MTLCommandQueue> command_queue = [device newCommandQueue];
    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLBlitCommandEncoder> blitCommandEncoder = [command_buffer blitCommandEncoder];

    // Enqueue a command to copy data from the source buffer to the temporary buffer.
    [blitCommandEncoder copyFromBuffer:buffer
                          sourceOffset:buffer_offset
                              toBuffer:temp_buffer
                     destinationOffset:0
                                  size:data_size];
    [blitCommandEncoder endEncoding];

    // Commit the command buffer and wait for it to complete.
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    // Copy the data from the temporary buffer to the host memory (data pointer).
    // The .contents property is accessible here because temp_buffer uses
    // MTLResourceStorageModeShared.
    std::memcpy(data, [temp_buffer contents], data_size);
  }
}

void WriteDataToTexture2D(id<MTLTexture> texture, id<MTLCommandQueue> command_queue,
                          const void* data, bool wait_for_completion) {
  @autoreleasepool {
    const int pixel_size = PixelFormatToSizeInBytes(texture.pixelFormat);
    id<MTLBuffer> temp_buffer =
        [[command_queue device] newBufferWithBytes:data
                                            length:pixel_size * texture.width * texture.height
                                           options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLBlitCommandEncoder> blitCommandEncoder = [command_buffer blitCommandEncoder];
    [blitCommandEncoder copyFromBuffer:temp_buffer
                          sourceOffset:0
                     sourceBytesPerRow:pixel_size * texture.width
                   sourceBytesPerImage:pixel_size * texture.width * texture.height
                            sourceSize:MTLSizeMake(texture.width, texture.height, 1)
                             toTexture:texture
                      destinationSlice:0
                      destinationLevel:0
                     destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blitCommandEncoder endEncoding];
    [command_buffer commit];
    if (wait_for_completion) {
      [command_buffer waitUntilCompleted];
    }
  }
}

void ReadDataFromTexture2D(id<MTLTexture> texture, id<MTLDevice> device, void* data) {
  @autoreleasepool {
    const int pixel_size = PixelFormatToSizeInBytes(texture.pixelFormat);
    const int buffer_size = pixel_size * texture.width * texture.height;
    id<MTLBuffer> temp_buffer = [device newBufferWithLength:buffer_size
                                                    options:MTLResourceStorageModeShared];

    id<MTLCommandQueue> command_queue = [device newCommandQueue];
    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];

    id<MTLBlitCommandEncoder> blitCommandEncoder = [command_buffer blitCommandEncoder];
    [blitCommandEncoder copyFromTexture:texture
                            sourceSlice:0
                            sourceLevel:0
                           sourceOrigin:MTLOriginMake(0, 0, 0)
                             sourceSize:MTLSizeMake(texture.width, texture.height, 1)
                               toBuffer:temp_buffer
                      destinationOffset:0
                 destinationBytesPerRow:pixel_size * texture.width
               destinationBytesPerImage:pixel_size * texture.width * texture.height];
    [blitCommandEncoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    std::memcpy(data, [temp_buffer contents], buffer_size);
  }
}

void WriteDataToTexture3D(id<MTLTexture> texture, id<MTLCommandQueue> command_queue,
                          const void* data, bool wait_for_completion) {
  @autoreleasepool {
    const int pixel_size = PixelFormatToSizeInBytes(texture.pixelFormat);
    id<MTLBuffer> temp_buffer = [[command_queue device]
        newBufferWithBytes:data
                    length:pixel_size * texture.width * texture.height * texture.depth
                   options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLBlitCommandEncoder> blitCommandEncoder = [command_buffer blitCommandEncoder];
    [blitCommandEncoder copyFromBuffer:temp_buffer
                          sourceOffset:0
                     sourceBytesPerRow:pixel_size * texture.width
                   sourceBytesPerImage:pixel_size * texture.width * texture.height
                            sourceSize:MTLSizeMake(texture.width, texture.height, texture.depth)
                             toTexture:texture
                      destinationSlice:0
                      destinationLevel:0
                     destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blitCommandEncoder endEncoding];
    [command_buffer commit];
    if (wait_for_completion) {
      [command_buffer waitUntilCompleted];
    }
  }
}

void ReadDataFromTexture3D(id<MTLTexture> texture, id<MTLDevice> device, void* data) {
  @autoreleasepool {
    const int pixel_size = PixelFormatToSizeInBytes(texture.pixelFormat);
    const int buffer_size = pixel_size * texture.width * texture.height * texture.depth;
    id<MTLBuffer> temp_buffer = [device newBufferWithLength:buffer_size
                                                    options:MTLResourceStorageModeShared];

    id<MTLCommandQueue> command_queue = [device newCommandQueue];
    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];

    id<MTLBlitCommandEncoder> blitCommandEncoder = [command_buffer blitCommandEncoder];
    [blitCommandEncoder copyFromTexture:texture
                            sourceSlice:0
                            sourceLevel:0
                           sourceOrigin:MTLOriginMake(0, 0, 0)
                             sourceSize:MTLSizeMake(texture.width, texture.height, texture.depth)
                               toBuffer:temp_buffer
                      destinationOffset:0
                 destinationBytesPerRow:pixel_size * texture.width
               destinationBytesPerImage:pixel_size * texture.width * texture.height];
    [blitCommandEncoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    std::memcpy(data, [temp_buffer contents], buffer_size);
  }
}

void WriteDataToTexture2DArray(id<MTLTexture> texture, id<MTLCommandQueue> command_queue,
                               const void* data, bool wait_for_completion) {
  @autoreleasepool {
    const int pixel_size = PixelFormatToSizeInBytes(texture.pixelFormat);
    id<MTLBuffer> temp_buffer = [[command_queue device]
        newBufferWithBytes:data
                    length:pixel_size * texture.width * texture.height * texture.arrayLength
                   options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    for (int i = 0; i < texture.arrayLength; ++i) {
      id<MTLBlitCommandEncoder> blitCommandEncoder = [command_buffer blitCommandEncoder];
      [blitCommandEncoder copyFromBuffer:temp_buffer
                            sourceOffset:pixel_size * texture.width * texture.height * i
                       sourceBytesPerRow:pixel_size * texture.width
                     sourceBytesPerImage:pixel_size * texture.width * texture.height
                              sourceSize:MTLSizeMake(texture.width, texture.height, 1)
                               toTexture:texture
                        destinationSlice:i
                        destinationLevel:0
                       destinationOrigin:MTLOriginMake(0, 0, 0)];
      [blitCommandEncoder endEncoding];
    }
    [command_buffer commit];
    if (wait_for_completion) {
      [command_buffer waitUntilCompleted];
    }
  }
}

void ReadDataFromTexture2DArray(id<MTLTexture> texture, id<MTLDevice> device, void* data) {
  @autoreleasepool {
    const int pixel_size = PixelFormatToSizeInBytes(texture.pixelFormat);
    const int buffer_size = pixel_size * texture.width * texture.height * texture.arrayLength;
    id<MTLBuffer> temp_buffer = [device newBufferWithLength:buffer_size
                                                    options:MTLResourceStorageModeShared];

    id<MTLCommandQueue> command_queue = [device newCommandQueue];
    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];

    for (int i = 0; i < texture.arrayLength; ++i) {
      id<MTLBlitCommandEncoder> blitCommandEncoder = [command_buffer blitCommandEncoder];
      [blitCommandEncoder copyFromTexture:texture
                              sourceSlice:i
                              sourceLevel:0
                             sourceOrigin:MTLOriginMake(0, 0, 0)
                               sourceSize:MTLSizeMake(texture.width, texture.height, 1)
                                 toBuffer:temp_buffer
                        destinationOffset:pixel_size * texture.width * texture.height * i
                   destinationBytesPerRow:pixel_size * texture.width
                 destinationBytesPerImage:pixel_size * texture.width * texture.height];
      [blitCommandEncoder endEncoding];
    }

    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    std::memcpy(data, [temp_buffer contents], buffer_size);
  }
}

void WaitUntilCompleted(id<MTLCommandQueue> command_queue) {
  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
  }
}

}  // namespace metal
}  // namespace ml_drift
