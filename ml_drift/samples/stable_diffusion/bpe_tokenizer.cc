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

#include "ml_drift/samples/stable_diffusion/bpe_tokenizer.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "re2/re2.h"

namespace ml_drift {
namespace {

constexpr const int kUtfMax = 4;

// A regexp used to split strings in the manner of OpenCLIP (and other
// tokenization). Some English contractions are handled as special cases
// ("I've" becomes "I" + "'ve"). Pure alphabetic sequences are taken
// together, while digits are split individually.
static constexpr LazyRE2 kOpenClipSplitter = {
    "("
    "'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+"
    ")"};

// Convert sequences of any whitespace to single space, and trim beginning and
// end.
std::string CleanString(const std::string& unprocessed_string) {
  // Space, tab, vertical tab, formfeed, linefeed, or carriage return.
  const std::vector<std::string> v =
      absl::StrSplit(absl::AsciiStrToLower(unprocessed_string),
                     absl::ByAnyChar(" \t\v\f\r\n"));
  std::string only_space_string = absl::StrJoin(v, " ");
  absl::RemoveExtraAsciiWhitespace(&only_space_string);
  return only_space_string;
}

// This function is functionally similar to utf/rune.c;l=252.
//
// It is modelled on icu/include/unicode/utf8.h;l=459, which at the time of
// coding is considered correct policy. But we need to avoid a large library
// dependency for such a small purpose.
inline int runetochar(const uint32_t rune, char rune_as_str[kUtfMax]) {
  int num_codepoints;

  if (rune <= 0x7f) {
    rune_as_str[0] = static_cast<uint8_t>(rune);
    num_codepoints = 1;
  } else if (rune <= 0x7ff) {
    rune_as_str[0] = static_cast<uint8_t>((rune >> 6) | 0xc0);
    rune_as_str[1] = static_cast<uint8_t>((rune & 0x3f) | 0x80);
    num_codepoints = 2;
  } else if ((rune <= 0xd7ff || (0xe000 <= rune && rune <= 0xffff))) {
    rune_as_str[0] = static_cast<uint8_t>((rune >> 12) | 0xe0);
    rune_as_str[1] = static_cast<uint8_t>(((rune >> 6) & 0x3f) | 0x80);
    rune_as_str[2] = static_cast<uint8_t>((rune & 0x3f) | 0x80);
    num_codepoints = 3;
  } else if (0xffff < rune && rune <= 0x10ffff) {
    rune_as_str[0] = static_cast<uint8_t>((rune >> 18) | 0xf0);
    rune_as_str[1] = static_cast<uint8_t>(((rune >> 12) & 0x3f) | 0x80);
    rune_as_str[2] = static_cast<uint8_t>(((rune >> 6) & 0x3f) | 0x80);
    rune_as_str[3] = static_cast<uint8_t>((rune & 0x3f) | 0x80);
    num_codepoints = 4;
  } else {
    num_codepoints = 0;
  }
  return num_codepoints;
}

std::vector<std::string> Tokenize(const std::string& unprocessed_string) {
  const std::string processed_string = CleanString(unprocessed_string);
  absl::string_view input_view(processed_string);
  std::vector<std::string> out;
  std::string chunk;
  while (RE2::FindAndConsume(&input_view, *kOpenClipSplitter, &chunk)) {
    out.push_back(chunk);
  }
  return out;
}

// Converts unicode-as-int-rune to unicode-as-std::string-utf-8.
//
// Requires that given rune is valid unicode.
inline std::string UnicodeToUtf8(uint32_t unicode_as_rune) {
  char unicode_as_char[kUtfMax];
  const int num_codepoints = runetochar(unicode_as_rune, unicode_as_char);
  return std::string(unicode_as_char, num_codepoints);
}

std::vector<std::string> SimpleSplit(const std::string& str, char delim) {
  std::vector<std::string> out;
  size_t start;
  size_t end = 0;
  while ((start = str.find_first_not_of(delim, end)) != std::string::npos) {
    end = str.find(delim, start);
    out.push_back(str.substr(start, end - start));
  }
  return out;
}
}  // namespace

std::vector<std::string> BPETokenizer::GetPairs(
    const std::vector<std::string>& strings) {
  std::vector<std::string> result;
  for (int i = 1; i < strings.size(); ++i) {
    result.push_back(strings[i - 1] + " " + strings[i]);
  }
  return result;
}

void BPETokenizer::Init(const std::string& file_folder, int padding_token) {
  padding_token_ = padding_token;
  // This is the number of vocab items imported from the vocab file.
  constexpr const int kImportVocabSize = 48895;
  std::vector<std::string> vocab_list;
  vocab_list.reserve(kImportVocabSize + 512);
  for (int i = 33; i <= 126; ++i) {
    const uint32_t character = i;
    bytes_to_unicode_[i] = character;
    vocab_list.push_back(UnicodeToUtf8(character));
  }
  for (int i = 161; i <= 172; ++i) {
    const uint32_t character = i;
    bytes_to_unicode_[i] = character;
    vocab_list.push_back(UnicodeToUtf8(character));
  }
  for (int i = 174; i <= 255; ++i) {
    const uint32_t character = i;
    bytes_to_unicode_[i] = character;
    vocab_list.push_back(UnicodeToUtf8(character));
  }
  uint32_t character = 256;
  for (int i = 0; i <= 255; ++i) {
    auto it = bytes_to_unicode_.find(i);
    if (it != bytes_to_unicode_.end()) {
      continue;
    }
    bytes_to_unicode_[i] = character;
    vocab_list.push_back(UnicodeToUtf8(character));
    ++character;
  }
  {
    const int vocab_size = vocab_list.size();
    for (int i = 0; i < vocab_size; ++i) {
      vocab_list.push_back(vocab_list[i] + "</w>");
    }
  }

  std::ifstream ifstr(file_folder + "/bpe_simple_vocab_16e6.txt");
  if (!ifstr) {
    ABSL_LOG(FATAL) << "Cannot open bpe_simple_vocab_16e6.txt in folder: "
                    << file_folder;
    return;
  }
  std::string line;
  std::getline(ifstr, line, '\n');

  for (int i = 1; i < kImportVocabSize; ++i) {
    std::getline(ifstr, line, '\n');
    ranks_[line] = i;
    auto tokens = SimpleSplit(line, ' ');
    std::string joined;
    for (const auto& token : tokens) {
      joined += token;
    }
    vocab_list.push_back(joined);
  }
  for (int i = 0; i < vocab_list.size(); ++i) {
    vocab_[vocab_list[i]] = i;
  }
}

std::vector<std::string> BPETokenizer::bpe(const std::vector<uint32_t>& token) {
  std::vector<std::string> word;
  word.reserve(token.size());
  for (int i = 0; i < token.size(); ++i) {
    word.push_back(UnicodeToUtf8(token[i]));
  }
  word[word.size() - 1] = word[word.size() - 1] + "</w>";

  std::vector<std::string> pairs = GetPairs(word);
  if (pairs.empty()) {
    // Note variation vs Python, that "</w>" has already been appended.
    return std::vector<std::string>({absl::StrJoin(word, "")});
  }
  int count = 0;
  while (true) {
    count += 1;
    if (count >= 8192) {
      std::cout << "encodeToken is trapped in a token factory for input"
                << std::endl;
    }
    int min_rank = 50000;
    std::string highest_ranked_bigram;
    for (int i = 0; i < pairs.size(); ++i) {
      const auto it = ranks_.find(pairs[i]);
      if (it != ranks_.end()) {
        if (it->second < min_rank) {
          min_rank = it->second;
          highest_ranked_bigram = pairs[i];
        }
      }
    }
    if (highest_ranked_bigram.empty()) {
      break;
    }
    const std::vector<std::string> fs = SimpleSplit(highest_ranked_bigram, ' ');
    const std::string& first = fs[0];
    const std::string& second = fs[1];
    std::vector<std::string> new_word;
    int i = 0;
    while (i < word.size()) {
      int j = -1;
      for (int k = i; k < word.size(); ++k) {
        if (word[k] == first) {
          j = k;
          break;
        }
      }
      if (j == -1) {
        new_word.insert(new_word.end(), word.begin() + i, word.end());
        break;
      } else {
        new_word.insert(new_word.end(), word.begin() + i, word.begin() + j);
        i = j;
      }
      if (word[i] == first && word[i + 1] == second) {
        new_word.insert(new_word.end(), first + second);
        i += 2;
      } else {
        new_word.insert(new_word.end(), word[i]);
        i += 1;
      }
    }
    word = new_word;
    if (word.size() == 1) {
      break;
    } else {
      pairs = GetPairs(word);
    }
  }
  return word;
}

std::vector<int> BPETokenizer::EncodeToken(const std::string& str) {
  // Apply bytes_to_unicode to each byte in input string.
  std::vector<uint32_t> token;
  token.reserve(str.size());
  for (int i = 0; i < str.size(); ++i) {
    token.push_back(bytes_to_unicode_[str[i]]);
  }

  // Apply BPE encoding.
  const std::vector<std::string> merged_word_tokens = bpe(token);

  // Finally convert vector of tokens to their vocab IDs.
  std::vector<int> result;
  result.reserve(merged_word_tokens.size());
  for (const auto& word_token : merged_word_tokens) {
    auto it = vocab_.find(word_token);
    if (it != vocab_.end()) {
      result.push_back(it->second);
    }
  }
  return result;
}

std::vector<int> BPETokenizer::Encode(const std::string& str) {
  // kMaxTokenVector is not a count of tokens because there are guard symbols at
  // the beginning and end.
  constexpr const int kMaxTokenVector = 77;
  std::vector<int> bpe;
  bpe.insert(bpe.end(), 49406);
  auto matches = Tokenize(str);
  for (const auto& match : matches) {
    const auto bpe_token = EncodeToken(match);
    bpe.insert(bpe.end(), bpe_token.begin(), bpe_token.end());
  }
  size_t size = bpe.size();
  bpe.resize(kMaxTokenVector);
  // Max tokens is kMaxTokenVector - 2, but we pushed the begin guard token.
  if (size < kMaxTokenVector) {
    bpe[size] = 49407;
  } else {
    std::cout << "Prompt of bpe tokens will be truncated." << std::endl;
    bpe[kMaxTokenVector - 1] = 49407;
  }
  for (int i = size + 1; i < bpe.size(); ++i) {
    bpe[i] = padding_token_;
  }
  return bpe;
}

}  // namespace ml_drift
