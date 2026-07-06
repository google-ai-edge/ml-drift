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

#include "ml_drift/cl/util.h"

#include <cstddef>
#include <string>

#include "absl/debugging/leak_check.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/status.h"
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_gl.h>
#include <CL/cl_platform.h>

namespace ml_drift {
namespace cl {

std::string CLErrorCodeToString(cl_int error_code) {
  switch (error_code) {
    case CL_SUCCESS:
      return "Success";
    case CL_DEVICE_NOT_FOUND:
      return "Device not found";
    case CL_DEVICE_NOT_AVAILABLE:
      return "Device not available";
    case CL_COMPILER_NOT_AVAILABLE:
      return "Compiler not available";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "Memory object allocation failure";
    case CL_OUT_OF_RESOURCES:
      return "Out of resources";
    case CL_OUT_OF_HOST_MEMORY:
      return "Out of host memory";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return "Profiling information not available";
    case CL_MEM_COPY_OVERLAP:
      return "Memory copy overlap";
    case CL_IMAGE_FORMAT_MISMATCH:
      return "Image format mismatch";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return "Image format not supported";
    case CL_BUILD_PROGRAM_FAILURE:
      return "Build program failure";
    case CL_MAP_FAILURE:
      return "Mapping failure";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
      return "Misaligned sub-buffer offset";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
      return "Execution status error for events in wait list";
    case CL_COMPILE_PROGRAM_FAILURE:
      return "Compile program failure";
    case CL_LINKER_NOT_AVAILABLE:
      return "Linker not available";
    case CL_LINK_PROGRAM_FAILURE:
      return "Link program failure";
    case CL_DEVICE_PARTITION_FAILED:
      return "Device partition failed";
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
      return "Kernel argument information not available";

    case CL_INVALID_VALUE:
      return "Invalid value";
    case CL_INVALID_DEVICE_TYPE:
      return "Invalid device type";
    case CL_INVALID_PLATFORM:
      return "Invalid platform";
    case CL_INVALID_DEVICE:
      return "Invalid device";
    case CL_INVALID_CONTEXT:
      return "Invalid context";
    case CL_INVALID_QUEUE_PROPERTIES:
      return "Invalid queue properties";
    case CL_INVALID_COMMAND_QUEUE:
      return "Invalid command queue";
    case CL_INVALID_HOST_PTR:
      return "Invalid host pointer";
    case CL_INVALID_MEM_OBJECT:
      return "Invalid memory object";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return "Invalid image format descriptor";
    case CL_INVALID_IMAGE_SIZE:
      return "Invalid image size";
    case CL_INVALID_SAMPLER:
      return "Invalid sampler";
    case CL_INVALID_BINARY:
      return "Invalid binary";
    case CL_INVALID_BUILD_OPTIONS:
      return "Invalid build options";
    case CL_INVALID_PROGRAM:
      return "Invalid program";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return "Invalid program executable";
    case CL_INVALID_KERNEL_NAME:
      return "Invalid kernel name";
    case CL_INVALID_KERNEL_DEFINITION:
      return "Invalid kernel definition";
    case CL_INVALID_KERNEL:
      return "Invalid kernel";
    case CL_INVALID_ARG_INDEX:
      return "Invalid argument index";
    case CL_INVALID_ARG_VALUE:
      return "Invalid argument value";
    case CL_INVALID_ARG_SIZE:
      return "Invalid argument size";
    case CL_INVALID_KERNEL_ARGS:
      return "Invalid kernel arguments";
    case CL_INVALID_WORK_DIMENSION:
      return "Invalid work dimension";
    case CL_INVALID_WORK_GROUP_SIZE:
      return "Invalid work group size";
    case CL_INVALID_WORK_ITEM_SIZE:
      return "Invalid work item size";
    case CL_INVALID_GLOBAL_OFFSET:
      return "Invalid global offset";
    case CL_INVALID_EVENT_WAIT_LIST:
      return "Invalid event wait list";
    case CL_INVALID_EVENT:
      return "Invalid event";
    case CL_INVALID_OPERATION:
      return "Invalid operation";
    case CL_INVALID_GL_OBJECT:
      return "Invalid GL object";
    case CL_INVALID_BUFFER_SIZE:
      return "Invalid buffer size";
    case CL_INVALID_MIP_LEVEL:
      return "Invalid mip-level";
    case CL_INVALID_GLOBAL_WORK_SIZE:
      return "Invalid global work size";
    case CL_INVALID_PROPERTY:
      return "Invalid property";
    case CL_INVALID_IMAGE_DESCRIPTOR:
      return "Invalid image descriptor";
    case CL_INVALID_COMPILER_OPTIONS:
      return "Invalid compiler options";
    case CL_INVALID_LINKER_OPTIONS:
      return "Invalid linker options";
    case CL_INVALID_DEVICE_PARTITION_COUNT:
      return "Invalid device partition count";
    case CL_INVALID_PIPE_SIZE:
      return "Invalid pipe size";
    case CL_INVALID_DEVICE_QUEUE:
      return "Invalid device queue";
    case CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR:
      return "Invalid GL sharegroup reference KHR";
    case CL_INVALID_COMMAND_BUFFER_KHR:
      return "Invalid command buffer KHR";
    case CL_INVALID_SYNC_POINT_WAIT_LIST_KHR:
      return "Invalid sync point wait list KHR";
    case CL_INCOMPATIBLE_COMMAND_QUEUE_KHR:
      return "Incompatible command queue KHR";

    default:
      return absl::StrCat("Unknown OpenCL error code - ", error_code);
  }
}

int ChannelTypeToSizeInBytes(cl_channel_type type) {
  switch (type) {
    case CL_FLOAT:
      return 4;
    case CL_HALF_FLOAT:
      return 2;
    default:
      return 0;
  }
}

bool OpenCLSupported() { return LoadOpenCL().ok(); }

absl::Status CreateCLBuffer(cl_context context, size_t size_in_bytes,
                            bool read_only, void* data, cl_mem* result) {
  cl_mem_flags flags = read_only ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE;
  if (data) {
    flags |= CL_MEM_COPY_HOST_PTR;
  }
  cl_int error_code;
  {
    // TODO: b/279347631 - Remove after Nvidia driver is fixed.
    absl::LeakCheckDisabler disabler;
    *result = clCreateBuffer(context, flags, size_in_bytes, data, &error_code);
  }
  if (!*result) {
    return absl::UnknownError(
        absl::StrCat("Failed to allocate ", size_in_bytes,
                     " bytes of device memory (clCreateBuffer): ",
                     CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

absl::Status CreateCLSubBuffer(cl_context context, cl_mem parent,
                               size_t origin_in_bytes, size_t size_in_bytes,
                               bool read_only, cl_mem* result) {
  cl_mem_flags flags = read_only ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE;

  cl_buffer_region region{};
  region.origin = origin_in_bytes;
  region.size = size_in_bytes;

  cl_int error_code;
  if (!clCreateSubBuffer) {
    return absl::InternalError("clCreateSubBuffer is not supported.");
  }
  {
    // TODO: b/279347631 - Remove after Nvidia driver is fixed.
    absl::LeakCheckDisabler disabler;
    *result = clCreateSubBuffer(parent, flags, CL_BUFFER_CREATE_TYPE_REGION,
                                &region, &error_code);
  }

  if (!*result) {
    return absl::UnknownError(
        absl::StrCat("Failed to allocate device memory (clCreateSubBuffer): ",
                     CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

absl::Status CreateRGBAImage2D(cl_context context, int width, int height,
                               cl_channel_type channel_type, void* data,
                               cl_mem* result) {
  cl_image_desc desc;
  desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  desc.image_width = width;
  desc.image_height = height;
  desc.image_depth = 0;
  desc.image_row_pitch = 0;
  desc.image_slice_pitch = 0;
  desc.num_mip_levels = 0;
  desc.num_samples = 0;
  desc.buffer = nullptr;

  cl_image_format format;
  format.image_channel_order = CL_RGBA;
  format.image_channel_data_type = channel_type;

  cl_mem_flags flags = CL_MEM_READ_WRITE;
  if (data) {
    flags |= CL_MEM_COPY_HOST_PTR;
  }

  cl_int error_code;
  *result =
      CreateImage2DLegacy(context, flags, &format, &desc, data, &error_code);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to create 2D texture (clCreateImage): ",
                     CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

absl::StatusOr<cl_mem_flags> GetCLMemObjectFlags(cl_mem memobj) {
  cl_mem_flags flags;
  size_t param_value_size_ret;

  cl_int err = clGetMemObjectInfo(memobj, CL_MEM_FLAGS, sizeof(cl_mem_flags),
                                  &flags, &param_value_size_ret);

  if (err != CL_SUCCESS) {
    return absl::UnknownError(absl::StrCat("Failed to get CL mem object info: ",
                                           CLErrorCodeToString(err)));
  }
  return flags;
}

AdrenoInfo::OpenClCompilerVersion GetQualcommOpenClCompilerVersion(
    const std::string& cl_driver_version) {
  AdrenoInfo::OpenClCompilerVersion result;
  result.major = 0;
  result.minor = 0;
  result.patch = 0;
  // Searching this part: "Compiler E031.**.**.**" where * is digit
  const std::string start = "Compiler E031.";
  size_t position = cl_driver_version.find(start);
  if (position == std::string::npos) {
    return result;
  }
  const size_t main_part_length = 8;  // main part is **.**.**
  if (position + start.length() + main_part_length >
      cl_driver_version.length()) {
    return result;
  }

  const std::string main_part =
      cl_driver_version.substr(position + start.length(), main_part_length);
  if (!absl::ascii_isdigit(main_part[0]) ||
      !absl::ascii_isdigit(main_part[1]) || main_part[2] != '.' ||
      !absl::ascii_isdigit(main_part[3]) ||
      !absl::ascii_isdigit(main_part[4]) || main_part[5] != '.' ||
      !absl::ascii_isdigit(main_part[6]) ||
      !absl::ascii_isdigit(main_part[7])) {
    return result;
  }
  result.major = (main_part[0] - '0') * 10 + (main_part[1] - '0');
  result.minor = (main_part[3] - '0') * 10 + (main_part[4] - '0');
  result.patch = (main_part[6] - '0') * 10 + (main_part[7] - '0');
  return result;
}

}  // namespace cl
}  // namespace ml_drift
