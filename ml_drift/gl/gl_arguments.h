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

#ifndef ML_DRIFT_GL_GL_ARGUMENTS_H_
#define ML_DRIFT_GL_GL_ARGUMENTS_H_

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/types.h"
#include "ml_drift/gl/gpu_object.h"
#include "ml_drift/gl/portable_gl31.h"

namespace ml_drift {
namespace gl {

class GlArguments : public ml_drift::ArgumentsBinder {
 public:
  GlArguments() = default;
  // Move only
  GlArguments(GlArguments&& args) = default;
  GlArguments& operator=(GlArguments&& args) = default;
  GlArguments(const GlArguments&) = delete;
  GlArguments& operator=(const GlArguments&) = delete;

  void AddFloat(const std::string& name, float value = 0.0f);
  void AddInt(const std::string& name, int value = 0);
  void AddUint(const std::string& name, uint value = 0);
  void AddBuffer(const std::string& name,
                 const ml_drift::GPUBufferDescriptor& desc);
  void AddTexture2D(const std::string& name,
                    const ml_drift::GPUImage2DDescriptor& desc);
  void AddTexture2DArray(const std::string& name,
                         const ml_drift::GPUImage2DArrayDescriptor& desc);
  void AddImage3D(const std::string& name,
                  const ml_drift::GPUImage3DDescriptor& desc);
  void AddImageBuffer(const std::string& name,
                      const ml_drift::GPUImageBufferDescriptor& desc);

  absl::Status SetInt(const std::string& name, int value) override;
  absl::Status SetUint(const std::string& name, uint value) override;
  absl::Status SetFloat(const std::string& name, float value) override;
  absl::Status SetHalf(const std::string& name, ml_drift::half value) override;
  absl::Status SetBuffer(const std::string& name, GLuint id);
  absl::Status SetTexture2D(const std::string& name, GLuint id);
  absl::Status SetTexture2DArray(const std::string& name, GLuint id);
  absl::Status SetImage3D(const std::string& name, GLuint id);
  absl::Status SetImageBuffer(const std::string& name, GLuint id);
  absl::Status SetObjectRef(const std::string& name,
                            const GPUObjectPtr& object);
  absl::Status SetObjectRef(const std::string& name, const GPUObject* object);

  void GetLocations(GLuint pipeline);

  void Bind();

  void RenameArgs(const std::string& postfix, std::string* code) const;

  absl::Status Init(const ml_drift::GpuInfo& gpu_info,
                    ml_drift::Arguments* args, std::string* code);

  absl::Status Init(const ml_drift::GpuInfo& gpu_info,
                    ml_drift::Arguments* args);

  void ReleaseCPURepresentation();

 private:
  void AddGPUResources(const std::string& name,
                       const ml_drift::GPUResources& resources);
  absl::Status SetGPUResources(const std::string& name,
                               const GPUResourcesWithValue& resources);
  void CopyArguments(const ml_drift::Arguments& args);

  absl::Status AddObjectArgs(const ml_drift::GpuInfo& gpu_info,
                             const ml_drift::Arguments& args);
  absl::Status AllocateObjects(const ml_drift::Arguments& args);
  absl::Status SetObjectsResources(const ml_drift::Arguments& args);

  std::string ToGLSLUniforms();

  void RenameArgumentsInCode(std::string* code) const;

  static constexpr char kArgsPrefix[] = "args.";

  struct IntValue {
    int value;

    // many uniforms generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;

    // offset to shared uniform storage.
    uint32_t offset = -1;
  };
  std::map<std::string, IntValue> int_values_;
  std::vector<int32_t> shared_int4s_data_;
  GLuint shared_int4s_loc_;

  struct UintValue {
    uint value;

    // many uniforms generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;

    // offset to shared uniform storage.
    uint32_t offset = -1;
  };
  std::map<std::string, UintValue> uint_values_;
  std::vector<uint32_t> shared_uint4s_data_;
  GLuint shared_uint4s_loc_;

  struct FloatValue {
    float value;

    // many uniforms generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;

    // offset to shared uniform storage.
    uint32_t offset = -1;
  };
  std::map<std::string, FloatValue> float_values_;
  std::vector<float> shared_float4s_data_;
  GLuint shared_float4s_loc_;

  struct GlBufferDescriptor {
    ml_drift::GPUBufferDescriptor desc;
    GLuint id;
  };
  struct GlImage2DDescriptor {
    ml_drift::GPUImage2DDescriptor desc;
    GLuint id;
  };
  struct GlImage2DArrayDescriptor {
    ml_drift::GPUImage2DArrayDescriptor desc;
    GLuint id;
  };
  struct GlImage3DDescriptor {
    ml_drift::GPUImage3DDescriptor desc;
    GLuint id;
  };
  struct GlImageBufferDescriptor {
    ml_drift::GPUImageBufferDescriptor desc;
    GLuint id;
  };

  std::map<std::string, GlBufferDescriptor> buffers_;
  std::map<std::string, GlImage2DDescriptor> textures2d_;
  std::map<std::string, GlImage2DArrayDescriptor> texture2d_arrays_;
  std::map<std::string, GlImage3DDescriptor> images3d_;
  std::map<std::string, GlImageBufferDescriptor> image_buffers_;
  std::map<std::string, int> ubo_slots_;
  std::map<std::string, int> ssbo_slots_;
  std::map<std::string, int> texture_slots_;
  std::map<std::string, int> image_slots_;

  std::map<std::string, ml_drift::GPUObjectDescriptorPtr, std::less<>>
      object_refs_;
  std::vector<GPUObjectPtr> objects_;
};

}  // namespace gl
}  // namespace ml_drift

#endif  // ML_DRIFT_GL_GL_ARGUMENTS_H_
