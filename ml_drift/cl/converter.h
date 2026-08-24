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

#ifndef ML_DRIFT_CL_CONVERTER_H_
#define ML_DRIFT_CL_CONVERTER_H_

#include <memory>

#include "absl/status/status.h"
#include "ml_drift/cl/buffer.h"
#include "ml_drift/cl/cl_command_queue.h"
#include "ml_drift/cl/cl_context.h"
#include "ml_drift/cl/cl_device.h"
#include "ml_drift/cl/cl_event.h"
#include "ml_drift/cl/cl_operation.h"
#include "ml_drift/cl/environment.h"
#include "ml_drift/cl/program_cache.h"
#include "ml_drift/cl/tensor.h"
#include "ml_drift/common/api_cl_gl_vk.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/spi.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/tensor_desc.h"

namespace ml_drift {
namespace cl {

class OpenClConverterImpl : public TensorObjectConverter {
 public:
  virtual absl::Status Init(const TensorObjectDef& input_def,
                            const TensorObjectDef& output_def,
                            Environment* environment) = 0;

 protected:
  BHWC shape_;
  CLCommandQueue* queue_ = nullptr;
  const CLContext* context_ = nullptr;
};

// Implements conversion from OpenCL tensor to another OpenCL tensor.
class TensorToTensorConverter : public OpenClConverterImpl {
 public:
  static bool IsSupported(const ObjectDef& input, const ObjectDef& output);

  // Init can be used only with Convert
  absl::Status Init(const TensorObjectDef& input_def,
                    const TensorObjectDef& output_def,
                    Environment* environment) final;
  // Convert can be used only with Init
  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override;

  // InitExplicit can be used only with ConvertExplicit
  absl::Status InitExplicit(const CLDevice* device, CLContext* context,
                            ProgramCache* cache,
                            const TensorDescriptor& src_desc,
                            const TensorDescriptor& dst_desc);
  // ConvertExplicit can be used only with InitExplicit
  absl::Status ConvertExplicit(CLCommandQueue* queue, Tensor* src, Tensor* dst,
                               CLEvent* event = nullptr);

 private:
  TensorDescriptor src_tensor_descriptor_;
  TensorDescriptor dst_tensor_descriptor_;
  ClOperation op_;
};

// Implements conversion from OpenCL-specific tensor layout to BHWC OpenCL
// buffer.
class TensorToBHWCBufferConverter : public OpenClConverterImpl {
 public:
  static bool IsSupported(const ObjectDef& input, const ObjectDef& output);

  // Init can be used only with Convert
  absl::Status Init(const TensorObjectDef& input_def,
                    const TensorObjectDef& output_def,
                    Environment* environment) final;
  // Convert can be used only with Init
  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override;

  // InitExplicit can be used only with ConvertExplicit
  absl::Status InitExplicit(const CLDevice* device, CLContext* context,
                            ProgramCache* cache,
                            const TensorDescriptor& src_desc,
                            const BufferDescriptor& dst_desc);
  // ConvertExplicit can be used only with InitExplicit
  absl::Status ConvertExplicit(CLCommandQueue* queue, Tensor* src, Buffer* dst,
                               CLEvent* event = nullptr);

 private:
  TensorDescriptor src_tensor_descriptor_;
  BufferDescriptor dst_buffer_descriptor_;
  ClOperation op_;
};

// Implements conversion from BHWC OpenCL buffer to OpenCL-specific tensor
// layout.
class BHWCBufferToTensorConverter : public OpenClConverterImpl {
 public:
  static bool IsSupported(const ObjectDef& input, const ObjectDef& output);

  // Init can be used only with Convert
  absl::Status Init(const TensorObjectDef& input_def,
                    const TensorObjectDef& output_def,
                    Environment* environment) final;
  // Convert can be used only with Init
  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override;

  // InitExplicit can be used only with ConvertExplicit
  absl::Status InitExplicit(const CLDevice* device, CLContext* context,
                            ProgramCache* cache,
                            const BufferDescriptor& src_desc,
                            const TensorDescriptor& dst_desc);
  // ConvertExplicit can be used only with InitExplicit
  absl::Status ConvertExplicit(CLCommandQueue* queue, Buffer* src, Tensor* dst,
                               CLEvent* event = nullptr);

 private:
  BufferDescriptor src_buffer_descriptor_;
  TensorDescriptor dst_tensor_descriptor_;
  ClOperation op_;
};

// Supports conversions from BHWC to internal OpenCL tensor representation and
// back. Also supports F16/F32.
std::unique_ptr<TensorObjectConverterBuilder> NewConverterBuilder(
    Environment* environment);

}  // namespace cl
}  // namespace ml_drift

#endif  // ML_DRIFT_CL_CONVERTER_H_
