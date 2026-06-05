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

#ifndef ML_DRIFT_CL_CL_ARGUMENTS_H_
#define ML_DRIFT_CL_CL_ARGUMENTS_H_

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "ml_drift/cl/cl_context.h"
#include "ml_drift/cl/gpu_object.h"
#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/types.h"

namespace ml_drift {
namespace cl {

class CLArguments : public ArgumentsBinder {
 public:
  CLArguments() = default;

  absl::Status Init(const GpuInfo& gpu_info,
                    CLContext* context, Arguments* args, std::string* code);
  absl::Status Init(const GpuInfo& gpu_info, Arguments* args,
                    CLContext* context);

  // Move only
  CLArguments(CLArguments&& args) = default;
  CLArguments& operator=(CLArguments&& args) = default;
  CLArguments(const CLArguments&) = delete;
  CLArguments& operator=(const CLArguments&) = delete;

  absl::Status SetInt(const std::string& name, int value) override;
  absl::Status SetUint(const std::string& name, unsigned int value) override;
  absl::Status SetFloat(const std::string& name, float value) override;
  absl::Status SetHalf(const std::string& name, half value) override;
  absl::Status SetObjectRef(const std::string& name, const GPUObject* object);

  absl::Status Bind(cl_kernel kernel, int offset = 0);

  // Compares Int, Float, Half names to values mapping with the mapping in
  // `other` and returns true if they are the same.
  bool HasEqualScalarArguments(const CLArguments& other) const;

  bool HasFloat16Objects() const;
  bool HasWriteOnly3dImages() const;

 private:
  absl::Status AllocateObjects(const Arguments& args, CLContext* context);
  absl::Status AddObjectArgs(const GpuInfo& gpu_info, const Arguments& args);

  void CopyArguments(const Arguments& args, bool use_f32_for_halfs);
  void RenameArgumentsInCode(std::string* code);
  std::string GetListOfArgs();

  void AddBuffer(const std::string& name, const GPUBufferDescriptor& desc);
  void AddImage2D(const std::string& name, const GPUImage2DDescriptor& desc);
  void AddImage2DArray(const std::string& name,
                       const GPUImage2DArrayDescriptor& desc);
  void AddImage3D(const std::string& name, const GPUImage3DDescriptor& desc);
  void AddImageBuffer(const std::string& name,
                      const GPUImageBufferDescriptor& desc);
  void AddCustomMemory(const std::string& name,
                       const GPUCustomMemoryDescriptor& desc);
  void AddGPUResources(const std::string& name, const GPUResources& resources);
  absl::Status SetObjectsResources(const Arguments& args);
  absl::Status SetGPUResources(const std::string& name,
                               const GPUResourcesWithValue& resources);

  absl::Status SetImage2D(const std::string& name, cl_mem memory);
  absl::Status SetBuffer(const std::string& name, cl_mem memory);
  absl::Status SetImage2DArray(const std::string& name, cl_mem memory);
  absl::Status SetImage3D(const std::string& name, cl_mem memory);
  absl::Status SetImageBuffer(const std::string& name, cl_mem memory);
  absl::Status SetCustomMemory(const std::string& name, cl_mem memory);

  static constexpr char kArgsPrefix[] = "args.";
  struct IntValue {
    int value;

    // many arguments generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;

    // offset to shared storage.
    uint32_t offset = -1;

    bool operator==(const IntValue& other) const {
      return value == other.value && offset == other.offset &&
             active == other.active;
    }
  };
  std::map<std::string, IntValue> int_values_;
  std::vector<int32_t> shared_int4s_data_;

  struct UintValue {
    unsigned int value;

    // many arguments generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;

    // offset to shared storage.
    uint32_t offset = -1;

    bool operator==(const UintValue& other) const {
      return value == other.value && offset == other.offset &&
             active == other.active;
    }
  };
  std::map<std::string, UintValue> uint_values_;
  std::vector<uint32_t> shared_uint4s_data_;

  struct FloatValue {
    float value;

    // many arguments generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;

    // offset to shared storage.
    uint32_t offset = -1;

    bool operator==(const FloatValue& other) const {
      return value == other.value && offset == other.offset &&
             active == other.active;
    }
  };
  std::map<std::string, FloatValue> float_values_;
  std::vector<float> shared_float4s_data_;

  struct HalfValue {
    half value;

    // many arguments generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;

    // some devices have issues with half parameters.
    bool store_as_f32 = false;

    // offset to shared uniform storage.
    uint32_t offset = -1;

    bool operator==(const HalfValue& other) const {
      return value == other.value && offset == other.offset &&
             active == other.active;
    }
  };
  std::map<std::string, HalfValue> half_values_;
  std::vector<half> shared_half4s_data_;

  struct CLBufferDescriptor {
    GPUBufferDescriptor desc;
    cl_mem memory;
  };
  struct CLImage2DDescriptor {
    GPUImage2DDescriptor desc;
    cl_mem memory;
  };
  struct CLImage2DArrayDescriptor {
    GPUImage2DArrayDescriptor desc;
    cl_mem memory;
  };
  struct CLImage3DDescriptor {
    GPUImage3DDescriptor desc;
    cl_mem memory;
  };
  struct CLImageBufferDescriptor {
    GPUImageBufferDescriptor desc;
    cl_mem memory;
  };
  struct CLCustomMemoryDescriptor {
    GPUCustomMemoryDescriptor desc;
    cl_mem memory;
  };

  std::map<std::string, CLBufferDescriptor> buffers_;
  std::map<std::string, CLImage2DDescriptor> images2d_;
  std::map<std::string, CLImage2DArrayDescriptor> image2d_arrays_;
  std::map<std::string, CLImage3DDescriptor> images3d_;
  std::map<std::string, CLImageBufferDescriptor> image_buffers_;
  std::map<std::string, CLCustomMemoryDescriptor> custom_memories_;

  std::map<std::string, GPUObjectDescriptorPtr, std::less<>> object_refs_;
  std::vector<GPUObjectPtr> objects_;
};

}  // namespace cl
}  // namespace ml_drift

#endif  // ML_DRIFT_CL_CL_ARGUMENTS_H_
