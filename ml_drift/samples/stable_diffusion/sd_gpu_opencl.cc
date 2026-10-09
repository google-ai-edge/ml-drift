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

#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/string_view.h"
#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/samples/stable_diffusion/diffuser.h"
#include "ml_drift/samples/stable_diffusion/util.h"

namespace ml_drift {
namespace cl {

// The OpenCL pipeline implementation has been factored out into `diffuser.h`
// and `diffuser.cc`, which allows the OpenCL diffuser library to be shared
// between this sample and the `libimagegenerator_gpu.so` C API library used by
// downstream integrations.
absl::Status RunStableDiffusion(absl::string_view pipeline_type) {
  ml_drift::cl::stable_diffusion::Diffuser::Config config;
  config.model_dir = std::string(pipeline_type);

  ABSL_ASSIGN_OR_RETURN(
      auto diffuser, ml_drift::cl::stable_diffusion::Diffuser::Create(config));

  unsigned int base_seed = 1;
  std::srand(base_seed);

  while (true) {
    std::string prompt = "a photo of an astronaut riding a horse on mars";
    int steps = 50;

    std::cout << "Enter phrase: ";
    if (!std::getline(std::cin, prompt)) break;
    if (prompt.empty()) continue;
    std::cout << "Enter steps: ";
    if (!(std::cin >> steps)) break;
    std::string dummy;
    std::getline(std::cin, dummy);

    int seed = std::rand() % 60000;

    ABSL_ASSIGN_OR_RETURN(
        auto result,
        diffuser->Diffuse(prompt, steps, seed, std::nullopt, {}, true));

    GenerateImage(result, "result.bmp");
    std::cout << "ready" << std::endl;
  }

  return absl::OkStatus();
}

}  // namespace cl
}  // namespace ml_drift

int main() {
  auto load_status = ml_drift::cl::LoadOpenCL();
  if (!load_status.ok()) {
    std::cout << load_status.message();
    return -1;
  }

  auto status = ml_drift::cl::RunStableDiffusion("sd_1_5/");
  if (!status.ok()) {
    std::cout << status.message() << std::endl;
    return -1;
  }
  return EXIT_SUCCESS;
}
