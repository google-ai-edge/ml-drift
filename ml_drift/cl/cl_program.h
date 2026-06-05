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

#ifndef ML_DRIFT_CL_CL_PROGRAM_H_
#define ML_DRIFT_CL_CL_PROGRAM_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "ml_drift/cl/cl_context.h"
#include "ml_drift/cl/cl_device.h"
#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/compiler_options.h"

namespace ml_drift {
namespace cl {

std::string CompilerOptionsToString(
    const GpuInfo& gpu_info,
    const std::vector<CompilerOptions>& compiler_options);

class CLProgram {
 public:
  CLProgram() = default;
  CLProgram(cl_program program, cl_device_id device_id);

  // Move only
  CLProgram(CLProgram&& program);
  CLProgram& operator=(CLProgram&& program);
  CLProgram(const CLProgram&) = delete;
  CLProgram& operator=(const CLProgram&) = delete;

  ~CLProgram();

  cl_program program() const { return program_; }

  // Return the cl_device_id associated with the program object.
  // This can be the device associated with context on which the program object
  // has been created or can be device that was specified when a program object
  // was created using clCreateProgramWithBinary.
  cl_device_id GetDeviceId() const { return device_id_; }

  absl::Status GetBinary(std::vector<uint8_t>* result) const;

 private:
  void Release();

  cl_program program_ = nullptr;

  // reference
  cl_device_id device_id_ = nullptr;
};

absl::Status CreateCLProgram(const std::string& code,
                             const std::string& compiler_options,
                             const CLContext& context, const CLDevice& device,
                             CLProgram* result);

absl::Status CreateCLProgram(
    const std::string& code,
    const std::vector<CompilerOptions>& compiler_options,
    const CLContext& context, const CLDevice& device, CLProgram* result);

absl::Status CreateCLProgramFromBinary(const CLContext& context,
                                       const CLDevice& device,
                                       absl::Span<const uint8_t> binary,
                                       CLProgram* result);

}  // namespace cl
}  // namespace ml_drift

#endif  // ML_DRIFT_CL_CL_PROGRAM_H_
