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

#ifndef ML_DRIFT_COMMON_TASK_COMPILER_OPTIONS_H_
#define ML_DRIFT_COMMON_TASK_COMPILER_OPTIONS_H_

#include <vector>

namespace ml_drift {

enum class CompilerOptions {
  kUnknown,
  // start
  kAdrenoFullSimd,
  kAdrenoMoreWaves,
  kCl20,
  kCl30,
  kClAdrenoFixBinary,
  kClAmdSaveTempsAndroid,
  kClDisableOptimizations,
  kClFastRelaxedMath,
  kClIntel256GRFPerThread,
  kClPixelDisableKernelBlobCache,
  kClPixelDisableRecompile,
  kClPixelEnableImmediateRecompile,
  kClRegisterAllocation64,
  kClUniformWorkGroupSize,
  kClVkDenormPreserve,
  kClVkNativeMath,
  kWaveSize128,
  kWaveSize16,
  kWaveSize32,
  kWaveSize64,
  kWaveSize8,
  // end
};

// Converts a wave size to a CompilerOptions enum value.
CompilerOptions WaveSizeToCompilerOption(int wave_size);
// Extracts the wave size from a list of compiler options.
// Returns 0 if no wave size option is found.
int GetWaveSizeFromCompilerOptions(
    const std::vector<CompilerOptions>& compiler_options);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_TASK_COMPILER_OPTIONS_H_
