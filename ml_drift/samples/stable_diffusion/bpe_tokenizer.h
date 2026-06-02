// Copyright 2023 The ML Drift Authors.
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

#ifndef ML_DRIFT_SAMPLES_STABLE_DIFFUSION_BPE_TOKENIZER_H_
#define ML_DRIFT_SAMPLES_STABLE_DIFFUSION_BPE_TOKENIZER_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ml_drift {

class BPETokenizer {
 public:
  std::vector<std::string> GetPairs(const std::vector<std::string>& strings);

  void Init(const std::string& file_folder, int padding_token = 49407);

  std::vector<int> EncodeToken(const std::string& str);

  std::vector<int> Encode(const std::string& str);

 private:
  std::vector<std::string> bpe(const std::vector<uint32_t>& token);
  std::map<int, uint32_t> bytes_to_unicode_;
  std::map<std::string, int> ranks_;
  std::map<std::string, int> vocab_;
  std::string file_folder_;
  int padding_token_;
};

}  // namespace ml_drift

#endif  // ML_DRIFT_SAMPLES_STABLE_DIFFUSION_BPE_TOKENIZER_H_
