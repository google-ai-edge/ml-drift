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

#ifndef ML_DRIFT_COMMON_MODEL_HINTS_H_
#define ML_DRIFT_COMMON_MODEL_HINTS_H_

#include <cstdint>

namespace ml_drift {

struct ModelHints {
  using ModelHint = uint64_t;

  // Can improve compilation time, but inference can be slower.
  static constexpr ModelHint kReduceKernelsCount = 0x00000001;
  // Can improve tuning time, but inference can be slower.
  static constexpr ModelHint kFastTuning = 0x00000001 << 1;

  // Can improve performance and memory consumption, but slow down
  // initialization and create more unique kernels.
  static constexpr ModelHint kAllowSpecialKernels = 0x00000001 << 2;

  // By default we apply Winograd optimized kernels and it improves performance.
  // But it also can increase memory usage and decrease precision.
  // This hint can disable Winograd optimizations.
  static constexpr ModelHint kNoWinogradOptimizations = 0x00000001 << 3;

  // By default, the same weights can be duplicated among different nodes of
  // convolution. With this hint we will try to reuse common object, when it
  // possible.
  // Can decrease constant memory usage(if model has the same weights).
  static constexpr ModelHint kReuseConvWeights = 0x00000001 << 4;

  // By default, we use the highest performance GPU available. This hint allows
  // setting a preference for a low power GPU if possible.
  static constexpr ModelHint kLowPower = 0x00000001 << 5;

  // Tells the backend to use textures for weights if possible.
  static constexpr ModelHint kPreferTextureWeights = 0x00000001 << 6;

  // Tells the backend to enable directly mapping the host pointer to GPU
  // memory when possible.
  static constexpr ModelHint kEnableHostMappedPointer = 0x00000001 << 7;

  // Performs f32/f16 convolutions instead of any 8 bit convolutions.
  static constexpr ModelHint kDisallow8bitConvs = 0x00000001 << 8;

  // Allows to use 4 bit convolutions.
  // If weights is 4 bit, allowed to quantize float32/16 input to 4 bit and run
  // 4 bit conv(w4a4).
  static constexpr ModelHint kAllow4bitConvs = 0x00000001 << 9;

  void Add(ModelHint hint) { hints |= hint; }

  bool Check(ModelHint hint) const { return hints & hint; }

  uint64_t hints = 0;

  // If true, model will add additional op(s) for converting weights to the
  // winograd representation before every convolution. Allows to store in
  // permanent storage(constant tensors) only original weights that consume less
  // memory than winograd weights. Memory for intermediate tensors may increase.
  // But usually, with this option total memory(constant + runtime) should be
  // smaller. And runtime memory can be reused for different models, that is not
  // true for constant memory.
  // If false, convert weights to winograd representation at
  // model preparation stage and will store in permanent storage(constant
  // tensors) more data. Do not has overhead in execution for conversion before
  // every convolution, most optimal from performance point of view.
  bool winograd_runtime_weights_conversion = false;

  // Can be unstable on some drivers.
  bool allow_cl_khr_command_buffer = true;

  // If true, will use Metal argument buffers. This is only supported on Metal.
  bool use_metal_argument_buffers = false;
};

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_MODEL_HINTS_H_
