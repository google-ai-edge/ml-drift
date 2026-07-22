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

#ifndef ML_DRIFT_WEBGPU_ARGUMENTS_H_
#define ML_DRIFT_WEBGPU_ARGUMENTS_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ml_drift/common/status.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/types.h"
#include "ml_drift/webgpu/buffer.h"
#include "ml_drift/webgpu/gpu_object.h"
#include "ml_drift/webgpu/webgpu_headers.h"

namespace ml_drift {
namespace webgpu {

class WebGpuArguments : public ArgumentsBinder {
 public:
  WebGpuArguments() = default;

  // Initializes the arguments and updates the code. Also updates the pipeline
  // layout.
  absl::Status Init(const wgpu::Device& device, const GpuInfo& gpu_info,
                    Arguments* args, std::string* code,
                    UniformBufferCreator* uniform_buffer_creator = nullptr);

  // Initializes the arguments without generating the code. Also updates the
  // pipeline layout.
  absl::Status InitWithoutCodeGeneration(
      const wgpu::Device& device, const GpuInfo& gpu_info, Arguments* args,
      UniformBufferCreator* uniform_buffer_creator = nullptr);

  // Move only
  WebGpuArguments(WebGpuArguments&& args) = default;
  WebGpuArguments& operator=(WebGpuArguments&& args) = default;
  WebGpuArguments(const WebGpuArguments&) = delete;
  WebGpuArguments& operator=(const WebGpuArguments&) = delete;

  absl::Status SetInt(const std::string& name, int value) override;
  absl::Status SetUint(const std::string& name, unsigned int value) override;
  absl::Status SetFloat(const std::string& name, float value) override;
  absl::Status SetHalf(const std::string& name, half value) override;
  absl::Status SetObjectRef(const std::string& name, const GpuObject& object);

  void Bind(const wgpu::ComputePassEncoder& encoder) const;
  absl::Status UpdateScalars(const wgpu::Device& device);
  void UpdateBindingsOnGPU(const wgpu::Device& device);

  bool HasFloat16Buffers() const;

  wgpu::PipelineLayout GetPipelineLayout() const {
    return pipeline_layout_.layout_;
  }

  void CopyFrom(const WebGpuArguments& other);

 private:
  // creates structure with layout:
  // struct uniforms_buffer {
  //   int4 val_0_val_1_dummy_dummy;
  //   float4 val_2_dummy_dummy_dummy;
  // };
  std::string ScalarArgumentsToStructWithVec4Fields(const Arguments& args,
                                                    bool explicit_fp16,
                                                    std::string* code);

  std::string ToWgslArguments(bool supports_fp16);

  absl::Status AssembleLayout(const wgpu::Device& device);

  absl::Status AllocateObjects(const wgpu::Device& device,
                               const Arguments& args);
  absl::Status AddObjectArgs(const GpuInfo& gpu_info, const Arguments& args);

  void AddGPUResources(const std::string& name, const GPUResources& resources);

  void AddBuffer(const std::string& name, const GPUBufferDescriptor& desc);
  void AddImage2D(const std::string& name, const GPUImage2DDescriptor& desc);
  void AddImage2DArray(const std::string& name,
                       const GPUImage2DArrayDescriptor& desc);
  void AddImage3D(const std::string& name, const GPUImage3DDescriptor& desc);

  absl::Status SetBuffer(const std::string& name, const wgpu::Buffer& handle,
                         size_t size, size_t offset);
  absl::Status SetImage2D(const std::string& name, wgpu::TextureView handle);
  absl::Status SetImage2DArray(const std::string& name,
                               wgpu::TextureView handle);
  absl::Status SetImage3D(const std::string& name, wgpu::TextureView handle);

  absl::Status SetGPUResources(const std::string& name,
                               const GpuResourcesWithValue& resources);
  absl::Status SetObjectsResources(const Arguments& args);

  static constexpr char kArgsPrefix[] = "args.";
  struct IntValue {
    int value;

    // many arguments generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;

    // offset to shared storage.
    uint32_t bytes_offset = -1;
  };
  std::map<std::string, IntValue> int_values_;
  struct UintValue {
    unsigned int value;

    // many arguments generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;

    // offset to shared storage.
    uint32_t bytes_offset = -1;
  };
  std::map<std::string, UintValue> uint_values_;

  struct FloatValue {
    float value;

    // many arguments generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;

    // offset to shared storage.
    uint32_t bytes_offset = -1;
  };
  std::map<std::string, FloatValue> float_values_;

  struct WebGpuBufferDescriptor {
    GPUBufferDescriptor desc;
    wgpu::Buffer handle;
    size_t size;
    size_t offset = 0;
  };
  struct WebGpuImage2DDescriptor {
    GPUImage2DDescriptor desc;
    wgpu::TextureView handle;
  };
  struct WebGpuImage2DArrayDescriptor {
    GPUImage2DArrayDescriptor desc;
    wgpu::TextureView handle;
  };
  struct WebGpuImage3DDescriptor {
    GPUImage3DDescriptor desc;
    wgpu::TextureView handle;
  };

  std::map<std::string, WebGpuBufferDescriptor> buffers_;
  std::map<std::string, WebGpuImage2DDescriptor> images2d_;
  std::map<std::string, WebGpuImage2DArrayDescriptor> image2d_arrays_;
  std::map<std::string, WebGpuImage3DDescriptor> images3d_;

  // Subgraph ops may share the underlying GPU objects.
  std::map<std::string, std::shared_ptr<GPUObjectDescriptor>> object_refs_;
  std::vector<std::shared_ptr<GpuObject>> objects_;

  class PipelineLayout {
   public:
    wgpu::PipelineLayout layout_;
    wgpu::BindGroup bind_group_;
    wgpu::BindGroupLayout group_layout_;
    std::map<std::string, wgpu::BindGroupLayoutEntry> bindings_;

    int AddStorageBufferBinding(const std::string& name, bool read_only) {
      const uint32_t binding_index = GetNextBindingIndex();
      wgpu::BindGroupLayoutEntry layout;
      layout.binding = binding_index;
      layout.visibility = wgpu::ShaderStage::Compute;
      wgpu::BufferBindingType binding_type =
          read_only ? wgpu::BufferBindingType::ReadOnlyStorage
                    : wgpu::BufferBindingType::Storage;
      layout.buffer.type = binding_type;
      layout.buffer.hasDynamicOffset = false;
      layout.buffer.minBindingSize = 0;
      bindings_[name] = layout;
      return binding_index;
    }

    int AddUniformBufferBinding(const std::string& name) {
      const uint32_t binding_index = GetNextBindingIndex();
      wgpu::BindGroupLayoutEntry layout;
      layout.binding = binding_index;
      layout.visibility = wgpu::ShaderStage::Compute;
      layout.buffer.type = wgpu::BufferBindingType::Uniform;
      layout.buffer.hasDynamicOffset = false;
      layout.buffer.minBindingSize = 0;
      bindings_[name] = layout;
      return binding_index;
    }

    int AddTextureBinding(const std::string& name,
                          wgpu::TextureViewDimension view_dimension,
                          wgpu::TextureSampleType sampler_type) {
      const uint32_t binding_index = GetNextBindingIndex();
      wgpu::BindGroupLayoutEntry layout;
      layout.binding = binding_index;
      layout.visibility = wgpu::ShaderStage::Compute;
      layout.texture.sampleType = sampler_type;
      layout.texture.viewDimension = view_dimension;
      layout.texture.multisampled = false;
      bindings_[name] = layout;
      return binding_index;
    }

    int AddStorageTextureBinding(const std::string& name,
                                 wgpu::TextureFormat texture_format,
                                 wgpu::TextureViewDimension view_dimension) {
      const uint32_t binding_index = GetNextBindingIndex();
      wgpu::BindGroupLayoutEntry layout;
      layout.binding = binding_index;
      layout.visibility = wgpu::ShaderStage::Compute;
      layout.storageTexture.access = wgpu::StorageTextureAccess::WriteOnly,
      layout.storageTexture.format = texture_format,
      layout.storageTexture.viewDimension = view_dimension,
      bindings_[name] = layout;
      return binding_index;
    }

    absl::Status Assemble(const wgpu::Device& device);

    void UpdateBindingsOnGPU(
        const wgpu::Device& device,
        const std::map<std::string, WebGpuBufferDescriptor>& buffers,
        const std::map<std::string, WebGpuImage2DDescriptor>& images2d,
        const std::map<std::string, WebGpuImage2DArrayDescriptor>&
            image2d_arrays,
        const std::map<std::string, WebGpuImage3DDescriptor>& images3d,
        const Buffer& scalars);

   private:
    uint32_t GetNextBindingIndex() const { return bindings_.size(); }
  };

  std::vector<uint8_t> const_data_;
  // Keeps track of the last uploaded data to avoid uploading if nothing has
  // changed.
  std::vector<uint8_t> last_uploaded_const_data_;
  Buffer scalars_;
  PipelineLayout pipeline_layout_;
};

}  // namespace webgpu
}  // namespace ml_drift

#endif  // ML_DRIFT_WEBGPU_ARGUMENTS_H_
