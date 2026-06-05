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

#ifndef ML_DRIFT_COMMON_TASK_ARGUMENTS_H_
#define ML_DRIFT_COMMON_TASK_ARGUMENTS_H_

#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "flatbuffers/buffer.h"
#include "flatbuffers/flatbuffer_builder.h"
#include "ml_drift/common/access_type.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/serialization_base_generated.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

// Interface for binding arguments to a kernel.
class ArgumentsBinder {
 public:
  virtual absl::Status SetInt(const std::string& name, int value) = 0;
  virtual absl::Status SetUint(const std::string& name, unsigned int value) = 0;
  virtual absl::Status SetFloat(const std::string& name, float value) = 0;
  virtual absl::Status SetHalf(const std::string& name, half value) = 0;
  virtual ~ArgumentsBinder() = default;
};

// Manages arguments for a GPU kernel.
class Arguments : public ArgumentsBinder {
 public:
  Arguments() = default;
  ~Arguments() override = default;

  // Move only
  Arguments(Arguments&& args) = default;
  Arguments& operator=(Arguments&& args) = default;
  Arguments(const Arguments&) = delete;
  Arguments& operator=(const Arguments&) = delete;

  // Adds a float argument with a specific data type.
  void AddFloat(const std::string& name, float value, DataType type);
  // Adds a float argument.
  void AddFloat(const std::string& name, float value = 0.0f);
  // Adds a half argument.
  void AddHalf(const std::string& name, half value = half(0.0f));
  // Adds an int argument.
  void AddInt(const std::string& name, int value = 0);
  // Adds a uint argument.
  void AddUint(const std::string& name, unsigned int value = 0);
  // Sets an existing int argument.
  absl::Status SetInt(const std::string& name, int value) override;
  // Sets an existing uint argument.
  absl::Status SetUint(const std::string& name, unsigned int value) override;
  // Sets an existing float argument.
  absl::Status SetFloat(const std::string& name, float value) override;
  // Sets an existing half argument.
  absl::Status SetHalf(const std::string& name, half value) override;
  // Adds an object reference argument.
  void AddObjectRef(const std::string& name, AccessType access_type,
                    GPUObjectDescriptorPtr&& descriptor_ptr);
  // Adds an object argument.
  void AddObject(const std::string& name,
                 GPUObjectDescriptorPtr&& descriptor_ptr);

  // Renames arguments in the code with a postfix.
  void RenameArgs(const std::string& postfix, std::string* code) const;
  // Merges arguments from another Arguments object.
  absl::Status Merge(Arguments&& args, const std::string& postfix,
                     const std::vector<std::string>& exception_names = {});

  // Gets the descriptor for a given argument name.
  absl::Status GetDescriptor(absl::string_view name,
                             GPUObjectDescriptor** descriptor) const;

  // Returns the number of read textures.
  int GetReadTexturesCount(const GpuInfo& gpu_info) const;
  // Returns the number of write textures.
  int GetWriteTexturesCount(const GpuInfo& gpu_info) const;

  // Releases the CPU representation of the arguments.
  void ReleaseCPURepresentation();

  // Finds all active arguments in the code.
  void GetActiveArguments(const std::string& code);

  // Sets a state value for all objects.
  void SetStateValueForAllObjects(const std::string& key,
                                  const std::string& value);

  struct IntValue {
    int value;

    // many uniforms generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;
  };
  struct UintValue {
    unsigned int value;

    // many uniforms generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;
  };
  struct FloatValue {
    float value;

    // many uniforms generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;
  };
  struct HalfValue {
    half value;

    // many uniforms generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;
  };

  // Returns the int values.
  const std::map<std::string, IntValue, std::less<>>& GetIntValues() const {
    return int_values_;
  }
  // Returns the uint values.
  const std::map<std::string, UintValue, std::less<>>& GetUintValues() const {
    return uint_values_;
  }
  // Returns the float values.
  const std::map<std::string, FloatValue, std::less<>>& GetFloatValues() const {
    return float_values_;
  }
  // Returns the half values.
  const std::map<std::string, HalfValue, std::less<>>& GetHalfValues() const {
    return half_values_;
  }

  // Returns the object references.
  const std::map<std::string, GPUObjectDescriptorPtr, std::less<>>&
  GetObjectRefs() const {
    return object_refs_;
  }
  // Returns the objects.
  const std::map<std::string, GPUObjectDescriptorPtr, std::less<>>& GetObjects()
      const {
    return objects_;
  }
  // Moves the object references to the result map.
  void MoveObjectRefs(
      std::map<std::string, GPUObjectDescriptorPtr, std::less<>>* result) {
    *result = std::move(object_refs_);
  }

  // Compiles the arguments into the code.
  absl::Status Compile(const GpuInfo& gpu_info,
                       std::string* code);

  // Resolves object names in the code.
  void ResolveObjectNames(const std::string& object_name,
                          const std::vector<std::string>& member_names,
                          std::string* code) const;
  // Adds scalar arguments for the objects.
  absl::Status AddObjectsScalarArgs(const GpuInfo& gpu_info);
  // Resolves arguments in the code.
  void ResolveArgsPass(std::string* code) const;

 private:
  friend flatbuffers::Offset<data::Arguments> Encode(
      const Arguments& args, flatbuffers::FlatBufferBuilder* builder);
  friend absl::Status Decode(const data::Arguments* fb_args, Arguments* args);

  absl::Status ResolveKernelGlobalSpaceBuffers(const GpuInfo& gpu_info,
                                               std::string* code);

  static constexpr char kArgsPrefix[] = "args.";

  std::map<std::string, IntValue, std::less<>> int_values_;
  std::map<std::string, UintValue, std::less<>> uint_values_;
  std::map<std::string, FloatValue, std::less<>> float_values_;
  std::map<std::string, HalfValue, std::less<>> half_values_;

  std::map<std::string, GPUObjectDescriptorPtr, std::less<>> object_refs_;
  std::map<std::string, GPUObjectDescriptorPtr, std::less<>> objects_;
};

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_TASK_ARGUMENTS_H_
