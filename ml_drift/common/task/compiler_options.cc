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

#include "ml_drift/common/task/compiler_options.h"

#include <vector>

namespace ml_drift {

CompilerOptions WaveSizeToCompilerOption(int wave_size) {
  switch (wave_size) {
    case 8:
      return CompilerOptions::kWaveSize8;
    case 16:
      return CompilerOptions::kWaveSize16;
    case 32:
      return CompilerOptions::kWaveSize32;
    case 64:
      return CompilerOptions::kWaveSize64;
    case 128:
      return CompilerOptions::kWaveSize128;
    default:
      return CompilerOptions::kUnknown;
  }
}

int GetWaveSizeFromCompilerOptions(
    const std::vector<CompilerOptions>& compiler_options) {
  for (const auto& option : compiler_options) {
    if (option == CompilerOptions::kWaveSize8) {
      return 8;
    } else if (option == CompilerOptions::kWaveSize16) {
      return 16;
    } else if (option == CompilerOptions::kWaveSize32) {
      return 32;
    } else if (option == CompilerOptions::kWaveSize64) {
      return 64;
    } else if (option == CompilerOptions::kWaveSize128) {
      return 128;
    }
  }
  return 0;
}
}  // namespace ml_drift
