// Copyright 2024 The ML Drift Authors.
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

#include "ml_drift/webgpu/preprocessor.h"

#include <cctype>
#include <list>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/status.h"

namespace ml_drift {
namespace webgpu {
namespace {
inline bool IsWordFirstSymbol(char symbol) {
  const bool is_underscore = symbol == '_';
  return islower(symbol) || isupper(symbol) || is_underscore;
}

inline bool IsWordSymbol(char symbol) {
  return IsWordFirstSymbol(symbol) || isdigit(symbol);
}

class Lexer {
 public:
  enum class TokenType {
    kUnknown,
    kCommentLong,
    kCommentShort,
    kSpaces,
    kNewLine,
    kSpecial,
    kWord,
    kNumber,
    kDefine,
  };
  struct Token {
    TokenType type = TokenType::kUnknown;
    absl::string_view text;
  };
  absl::Status AddMacro(absl::string_view key, absl::string_view value) {
    RETURN_IF_ERROR(ParseToTokens(value, &macros_[key]));
    return absl::OkStatus();
  }
  absl::Status ParseToken(absl::string_view code, Token* result) {
    result->text = absl::string_view();
    int position = 0;
    int code_size = code.size();
    // Parse long comment /* ... */
    if (code_size >= 4 && code[0] == '/' && code[1] == '*') {
      result->type = TokenType::kCommentLong;
      position += 2;
      do {
        position++;
        if (position >= code_size) {
          return absl::InternalError("Can not find end of long comment.");
        }
        if (code[position] == '/' && code[position - 1] == '*') {
          break;
        }
      } while (true);
      result->text = code.substr(0, position + 1);
      return absl::OkStatus();
    }
    // Parse short comment //
    if (code_size >= 2 && code[0] == '/' && code[1] == '/') {
      result->type = TokenType::kCommentShort;
      while (position < code_size && code[position] != '\n') {
        position++;
      }
      result->text = code.substr(0, position);
      return absl::OkStatus();
    }
    // Parse #define
    if (code_size >= 7 && code[0] == '#' && code[1] == 'd' && code[2] == 'e' &&
        code[3] == 'f' && code[4] == 'i' && code[5] == 'n' && code[6] == 'e') {
      result->type = TokenType::kDefine;
      int prev_non_space_pos;
      do {
        while (position < code_size && code[position] != '\n') {
          position++;
        }
        prev_non_space_pos = position - 1;
        while (code[prev_non_space_pos] == ' ') {
          prev_non_space_pos--;
        }
        if (code[prev_non_space_pos] != '\\') {
          break;
        } else {
          position++;
        }
      } while (true);
      result->text = code.substr(0, position);
      return absl::OkStatus();
    }
    // Parse empty string from spaces
    if (code[0] == ' ') {
      result->type = TokenType::kSpaces;
      while (position < code_size && code[position] == ' ') {
        position++;
      }
      result->text = code.substr(0, position);
      return absl::OkStatus();
    }
    // Parse word
    if (IsWordFirstSymbol(code[0])) {
      result->type = TokenType::kWord;
      position += 1;
      while (position < code_size && IsWordSymbol(code[position])) {
        position++;
      }
      result->text = code.substr(0, position);
      return absl::OkStatus();
    }
    // Parse new line
    if (code[0] == '\n') {
      result->type = TokenType::kNewLine;
      result->text = code.substr(0, 1);
      return absl::OkStatus();
    }
    // Parse number
    if (isdigit(code[0])) {
      result->type = TokenType::kNumber;
      while (position < code_size &&
             (isdigit(code[position]) || code[position] == '.' ||
              code[position] == 'f')) {
        position++;
      }
      result->text = code.substr(0, position);
      return absl::OkStatus();
    }
    // Parse special operators
    if (code_size >= 3) {
      constexpr const char* kSpecialThree[] = {"<<=", ">>="};
      for (absl::string_view op : kSpecialThree) {
        if (code[0] == op[0] && code[1] == op[1] && code[2] == op[2]) {
          result->type = TokenType::kSpecial;
          result->text = code.substr(0, 3);
          return absl::OkStatus();
        }
      }
    }
    if (code_size >= 2) {
      constexpr const char* kSpecialTwo[] = {
          "+=", "-=", "/=", "*=", "%=", "&=", "^=", "|=",
          "==", "!=", ">=", "<=", "++", "--", ">>", "<<"};
      for (absl::string_view op : kSpecialTwo) {
        if (code[0] == op[0] && code[1] == op[1]) {
          result->type = TokenType::kSpecial;
          result->text = code.substr(0, 2);
          return absl::OkStatus();
        }
      }
    }
    result->type = TokenType::kSpecial;
    result->text = code.substr(0, 1);
    return absl::OkStatus();
  }
  absl::Status ParseToTokensInternal(std::string&& code,
                                     std::vector<Token>* tokens) {
    // Since tokens hold string_view, we need to make sure any string that gets
    // tokenized lives for the duration of the lexer.
    auto it = owned_strings_.insert(owned_strings_.end(), std::move(code));
    return ParseToTokens(*it, tokens);
  }
  absl::Status ParseToTokens(absl::string_view code,
                             std::vector<Token>* tokens) {
    while (!code.empty()) {
      Token token;
      RETURN_IF_ERROR(ParseToken(code, &token));
      code.remove_prefix(token.text.size());
      if (token.type == TokenType::kDefine) {
        // 7 is length of #define
        const absl::string_view macro_code =
            token.text.substr(7, token.text.size() - 7);
        std::vector<Token> macro_tokens;
        RETURN_IF_ERROR(ParseToTokens(macro_code, &macro_tokens));
        if (macro_tokens[0].type == TokenType::kSpaces) {
          macro_tokens.erase(macro_tokens.begin());
        }
        const absl::string_view define_name = macro_tokens[0].text;
        if (macro_tokens[1].type == TokenType::kSpaces) {
          macro_tokens.erase(macro_tokens.begin(), macro_tokens.begin() + 2);
          macros_[define_name] = std::move(macro_tokens);
        } else {
          RETURN_IF_ERROR(
              macros_with_arguments_[define_name].Init(macro_tokens));
        }
        continue;
      }
      auto it = macros_.find(token.text);
      if (it != macros_.end()) {
        for (auto& macro_token : it->second) {
          tokens->push_back(macro_token);
        }
      } else {
        auto it_with_arg = macros_with_arguments_.find(token.text);
        if (it_with_arg != macros_with_arguments_.end()) {
          Token arg_token;
          RETURN_IF_ERROR(ParseToken(code, &arg_token));
          code.remove_prefix(arg_token.text.size());
          if (arg_token.type == TokenType::kSpaces) {
            RETURN_IF_ERROR(ParseToken(code, &arg_token));
            code.remove_prefix(arg_token.text.size());
          }
          int opened = 1;
          int closed = 0;
          std::vector<std::vector<Token>> arguments(1);
          while (!code.empty()) {
            RETURN_IF_ERROR(ParseToken(code, &arg_token));
            code.remove_prefix(arg_token.text.size());
            if (arg_token.text == "(") {
              opened++;
            } else if (arg_token.text == ")") {
              closed++;
              if (opened == closed) {
                break;
              }
            }
            if (arg_token.text == ",") {
              arguments.push_back({});
            } else {
              auto it = macros_.find(arg_token.text);
              if (it != macros_.end()) {
                for (auto& macro_token : it->second) {
                  arguments.back().push_back(macro_token);
                }
              } else {
                arguments.back().push_back(arg_token);
              }
            }
          }
          auto new_tokens = it_with_arg->second.GetCode(arguments);
          for (const auto& new_token : new_tokens) {
            tokens->push_back(new_token);
          }
        } else {
          tokens->push_back(std::move(token));
        }
      }
    }
    return absl::OkStatus();
  }
  absl::Status Init(absl::string_view code_base) {
    return ParseToTokens(code_base, &tokens_);
  }
  int FindPrev(int index, TokenType type,
               const std::vector<absl::string_view>& texts) {
    while (index > 0) {
      index--;
      if (tokens_[index].type == type) {
        bool has_text = false;
        for (const auto& text : texts) {
          if (tokens_[index].text == text) {
            has_text = true;
            break;
          }
        }
        if (has_text) {
          if (tokens_[index + 1].type == TokenType::kSpaces) {
            index++;
          }
          if (tokens_[index + 1].type == TokenType::kNewLine) {
            index++;
          }
          return index;
        }
      }
    }
    return 0;
  }
  int FindNext(int index, TokenType type,
               const std::vector<absl::string_view>& texts) {
    while (index < tokens_.size() - 1) {
      index++;
      if (tokens_[index].type == type) {
        bool has_text = false;
        for (const auto& text : texts) {
          if (tokens_[index].text == text) {
            has_text = true;
            break;
          }
        }
        if (has_text) {
          return index;
        }
      }
    }
    return 0;
  }

  int FindEnclosingBracket(int open_bracket_index, char bracket_open) {
    const std::map<char, char> brackets = {
        {'(', ')'},
        {'{', '}'},
        {'[', ']'},
        {'<', '>'},
    };
    auto it = brackets.find(bracket_open);
    if (it == brackets.end()) {
      return -1;
    }
    char bracket_close = it->second;
    int index = open_bracket_index + 1;
    int opened = 1;
    int closed = 0;
    while (opened != closed && index < tokens_.size()) {
      if (tokens_[index].text[0] == bracket_open) {
        opened++;
      } else if (tokens_[index].text[0] == bracket_close) {
        closed++;
      }
      index++;
    }
    if (opened == closed) {
      return index - 1;
    } else {
      return -1;
    }
  }

  // "0.32423f" -> "0.32423"
  void RemoveFloatPostfix() {
    for (auto& token : tokens_) {
      if (token.type == TokenType::kNumber && token.text.back() == 'f') {
        token.text.remove_suffix(1);
      }
    }
  }

  // "cond ? b : c" -> "select(c, b, cond)"
  absl::Status ResolveTernaryOperator() {
    for (int i = 0; i < tokens_.size(); ++i) {
      if (tokens_[i].type == TokenType::kSpecial && tokens_[i].text == "?") {
        const int op_start =
            FindPrev(i, TokenType::kSpecial, {"=", "+=", "-=", "/=", "*="});
        const int op_middle = FindNext(i, TokenType::kSpecial, {":"});
        const int op_end = FindNext(i, TokenType::kSpecial, {";"});
        std::vector<Token> tokens_cond(tokens_.begin() + op_start + 1,
                                       tokens_.begin() + i);
        std::vector<Token> tokens_true(tokens_.begin() + i + 1,
                                       tokens_.begin() + op_middle);
        std::vector<Token> tokens_false(tokens_.begin() + op_middle + 1,
                                        tokens_.begin() + op_end);
        const std::string code_cond = GetCode(tokens_cond);
        const std::string code_true = GetCode(tokens_true);
        const std::string code_false = GetCode(tokens_false);
        tokens_.erase(tokens_.begin() + op_start + 1, tokens_.begin() + op_end);
        tokens_.insert(tokens_.begin() + op_start + 1,
                       {TokenType::kSpaces, " "});

        std::string insert_code = absl::StrCat("select(", code_false, ", ",
                                               code_true, ", ", code_cond, ")");
        std::vector<Token> insert_tokens;
        RETURN_IF_ERROR(
            ParseToTokensInternal(std::move(insert_code), &insert_tokens));
        tokens_.insert(tokens_.begin() + op_start + 2, insert_tokens.begin(),
                       insert_tokens.end());
        i = op_start;
      }
    }
    return absl::OkStatus();
  }

  int GetNextToken(int index) {
    for (int i = index + 1; i < tokens_.size(); ++i) {
      if (tokens_[i].type == TokenType::kWord ||
          tokens_[i].type == TokenType::kSpecial ||
          tokens_[i].type == TokenType::kNumber) {
        return i;
      }
    }
    return -1;
  }
  int GetPrevToken(int index) {
    for (int i = index - 1; i >= 0; --i) {
      if (tokens_[i].type == TokenType::kWord ||
          tokens_[i].type == TokenType::kSpecial ||
          tokens_[i].type == TokenType::kNumber) {
        return i;
      }
    }
    return -1;
  }

  // "if (x < y) return" -> "if (x < y) { return; }"
  absl::Status ResolveSingleStatementIf() {
    for (int i = 0; i < tokens_.size(); ++i) {
      if (tokens_[i].type == TokenType::kWord && tokens_[i].text == "if") {
        int open_bracket_index = GetNextToken(i);
        int close_bracket_index = FindEnclosingBracket(open_bracket_index, '(');
        int next_token_index = GetNextToken(close_bracket_index);
        if (tokens_[next_token_index].text != "{") {
          int end_statement_index = next_token_index + 1;
          for (int k = next_token_index + 1; k < tokens_.size(); ++k) {
            if (tokens_[k].type == TokenType::kSpecial &&
                tokens_[k].text == ";") {
              end_statement_index = k;
              break;
            }
          }
          tokens_.insert(tokens_.begin() + end_statement_index + 1,
                         {TokenType::kSpecial, "}"});
          tokens_.insert(tokens_.begin() + next_token_index,
                         {TokenType::kSpecial, "{"});
        }
      }
    }
    return absl::OkStatus();
  }

  // "do {instructions;} while (cond)" ->
  //  "loop { instructions; if (!cond) {break;}}"
  absl::Status ResolveDoWhileLoop() {
    for (int i = 0; i < tokens_.size(); ++i) {
      if (tokens_[i].type == TokenType::kWord && tokens_[i].text == "do") {
        int open_bracket_index = GetNextToken(i);
        int close_bracket_index = FindEnclosingBracket(open_bracket_index, '{');
        int while_token_index = GetNextToken(close_bracket_index);
        int loop_end_index = while_token_index + 1;
        for (int k = while_token_index + 1; k < tokens_.size(); ++k) {
          if (tokens_[k].type == TokenType::kSpecial &&
              tokens_[k].text == ";") {
            loop_end_index = k;
            break;
          }
        }
        tokens_[i].text = "loop";
        tokens_[while_token_index].text = "if";
        std::vector<Token> tokens_if(tokens_.begin() + while_token_index,
                                     tokens_.begin() + loop_end_index);
        tokens_if.push_back({TokenType::kSpecial, ")"});
        tokens_if.push_back({TokenType::kSpecial, "{"});
        tokens_if.push_back({TokenType::kWord, "break"});
        tokens_if.push_back({TokenType::kSpecial, ";"});
        tokens_if.push_back({TokenType::kSpecial, "}"});
        tokens_if.push_back({TokenType::kNewLine, "\n"});
        tokens_if.insert(tokens_if.begin() + 1, {TokenType::kSpecial, "("});
        tokens_if.insert(tokens_if.begin() + 2, {TokenType::kSpecial, "!"});
        tokens_.erase(tokens_.begin() + while_token_index,
                      tokens_.begin() + loop_end_index);
        tokens_.insert(tokens_.begin() + close_bracket_index, tokens_if.begin(),
                       tokens_if.end());
      }
    }
    return absl::OkStatus();
  }

  // for (a; b; c, d) { instructions;} ->
  // {a; loop {if(!b) {break}; instructions; continuing {c; d;}}}
  absl::Status ResolveForLoop() {
    for (int i = 0; i < tokens_.size(); ++i) {
      if (tokens_[i].type == TokenType::kWord && tokens_[i].text == "for") {
        int for_open_bracket_index = GetNextToken(i);
        int for_close_bracket_index =
            FindEnclosingBracket(for_open_bracket_index, '(');
        int body_open_bracket_index = GetNextToken(for_close_bracket_index);
        int body_close_bracket_index =
            FindEnclosingBracket(body_open_bracket_index, '{');
        std::vector<Token> part0;
        std::vector<Token> part1;
        std::vector<std::vector<Token>> part2(1);
        int pos = for_open_bracket_index + 1;
        for (; tokens_[pos].text != ";"; ++pos) {
          part0.push_back(tokens_[pos]);
        }
        pos++;
        for (; tokens_[pos].text != ";"; ++pos) {
          part1.push_back(tokens_[pos]);
        }
        pos++;
        for (; pos < for_close_bracket_index; ++pos) {
          if (tokens_[pos].text == ",") {
            part2.push_back({});
          } else {
            part2.back().push_back(tokens_[pos]);
          }
        }
        if (part2.size() == 1) {
          continue;
        }
        std::vector<Token> tokens_body(
            tokens_.begin() + body_open_bracket_index + 1,
            tokens_.begin() + body_close_bracket_index);
        std::string new_code;
        if (!part0.empty()) {
          absl::StrAppend(&new_code, "  {", GetCode(part0), ";\n");
        }
        absl::StrAppend(&new_code, "  loop {\n", "    if (!(", GetCode(part1),
                        ")) {break;}\n",
                        // body
                        "    ", GetCode(tokens_body), ";\n",
                        "    continuing {\n");
        for (const auto& instructions : part2) {
          absl::StrAppend(&new_code, "      ");
          for (const auto& instruction : instructions) {
            absl::StrAppend(&new_code, instruction.text);
          }
          absl::StrAppend(&new_code, ";\n");
        }
        absl::StrAppend(&new_code, "    }\n");
        if (!part0.empty()) {
          absl::StrAppend(&new_code, "  }\n");
        }
        tokens_.erase(tokens_.begin() + i,
                      tokens_.begin() + body_close_bracket_index);
        std::vector<Token> insert_tokens;
        RETURN_IF_ERROR(
            ParseToTokensInternal(std::move(new_code), &insert_tokens));
        tokens_.insert(tokens_.begin() + i, insert_tokens.begin(),
                       insert_tokens.end());
      }
    }
    return absl::OkStatus();
  }

  // "++a" -> "a += 1"
  absl::Status ResolvePreIncrement() {
    for (int i = 0; i < tokens_.size(); ++i) {
      if (tokens_[i].type == TokenType::kSpecial &&
          (tokens_[i].text == "++" || tokens_[i].text == "--") &&
          tokens_[i + 1].type == TokenType::kWord) {
        const absl::string_view op_text = tokens_[i].text == "++" ? "+" : "-";
        tokens_[i].text = "=";
        auto var_token = tokens_[i + 1];
        tokens_.insert(tokens_.begin() + i, var_token);
        tokens_.insert(tokens_.begin() + i + 3, {TokenType::kSpecial, op_text});
        tokens_.insert(tokens_.begin() + i + 4, {TokenType::kNumber, "1"});
      }
    }
    return absl::OkStatus();
  }

  // "int a" -> "var a : i32"
  absl::Status ConvertTypes(const WebGpuInfo& webgpu_info) {
    std::string fp_type = webgpu_info.supports_fp16 ? "f16" : "f32";
    std::string half2 = absl::StrCat("vec2<", fp_type, ">");
    std::string half3 = absl::StrCat("vec3<", fp_type, ">");
    std::string half4 = absl::StrCat("vec4<", fp_type, ">");
    std::string half16 = absl::StrCat("mat4x4<", fp_type, ">");
    const std::map<absl::string_view, absl::string_view> kTypeMap = {
        {"int", "i32"},
        {"int2", "vec2<i32>"},
        {"int3", "vec3<i32>"},
        {"int4", "vec4<i32>"},
        {"uint", "u32"},
        {"uint2", "vec2<u32>"},
        {"uint3", "vec3<u32>"},
        {"uint4", "vec4<u32>"},
        {"float", "f32"},
        {"float2", "vec2<f32>"},
        {"float3", "vec3<f32>"},
        {"float4", "vec4<f32>"},
        {"float16", "mat4x4<f32>"},
        {"half", fp_type},
        {"half2", half2},
        {"half3", half3},
        {"half4", half4},
        {"half16", half16},
        {"short", "i32"},
        {"short2", "vec2<i32>"},
        {"short3", "vec3<i32>"},
        {"short4", "vec4<i32>"},
        {"char", "i32"},
        {"char2", "vec2<i32>"},
        {"char3", "vec3<i32>"},
        {"char4", "vec4<i32>"},
        {"ushort", "u32"},
        {"ushort2", "vec2<u32>"},
        {"ushort3", "vec3<u32>"},
        {"ushort4", "vec4<u32>"},
        {"uchar", "u32"},
        {"uchar2", "vec2<u32>"},
        {"uchar3", "vec3<u32>"},
        {"uchar4", "vec4<u32>"},
        {"bool", "bool"},
        {"bool4", "vec4<bool>"},
    };
    for (int i = 0; i < tokens_.size() - 2; ++i) {
      if (tokens_[i].type == TokenType::kWord &&
          tokens_[i + 1].type == TokenType::kSpaces &&
          tokens_[i + 2].type == TokenType::kWord) {
        auto it = kTypeMap.find(tokens_[i].text);
        if (it != kTypeMap.end()) {
          std::vector<int> array_sizes;
          int next_token_index = GetNextToken(i);  // type_name
          int prev_token_index =
              GetPrevToken(i);  // maybe mem type local, constant, global
          const absl::string_view var_name = tokens_[next_token_index].text;
          next_token_index =
              GetNextToken(next_token_index);  // may be array bracket
          if (tokens_[next_token_index].text == ",") {
            // multiple declarations
            int index = next_token_index;
            while (tokens_[index].text != ";") {
              if (tokens_[index].text == ",") {
                tokens_[index].text = ";";
                tokens_.insert(tokens_.begin() + index + 1, tokens_[i]);
                index++;
              }
              index++;
            }
          }
          std::string type;
          while (tokens_[next_token_index].type == TokenType::kSpecial &&
                 tokens_[next_token_index].text == "[") {
            next_token_index = GetNextToken(next_token_index);  // size
            int array_size;
            if (!absl::SimpleAtoi(tokens_[next_token_index].text,
                                  &array_size)) {
              return absl::InternalError("Failed to parse array size.");
            }
            array_sizes.push_back(array_size);
            next_token_index = GetNextToken(next_token_index);  // close bracket
            next_token_index = GetNextToken(next_token_index);  // next token
            absl::StrAppend(&type, "array<");
          }
          absl::StrAppend(&type, it->second);
          for (int k = array_sizes.size() - 1; k >= 0; --k) {
            absl::StrAppend(&type, ", ", array_sizes[k], ">");
          }
          std::string space = "";  // can be <private>
          int start_index = i;
          const bool local_mem = tokens_[prev_token_index].text == "__local" ||
                                 tokens_[prev_token_index].text == "local";
          const bool constant_mem =
              tokens_[prev_token_index].text == "__constant" ||
              tokens_[prev_token_index].text == "constant";
          if (local_mem) {
            space = "<workgroup>";
            start_index = prev_token_index;
            next_token_index = GetNextToken(next_token_index);
          }
          std::string var_or_let = "var";
          if (constant_mem) {
            var_or_let = "let";
            start_index = prev_token_index;
            int open_bracket = GetNextToken(next_token_index);  // should be "{"
            int close_bracket = FindEnclosingBracket(open_bracket, '{');
            tokens_[open_bracket].text = "(";
            tokens_[close_bracket].text = ")";
            next_token_index = open_bracket;
          }
          tokens_.erase(tokens_.begin() + start_index,
                        tokens_.begin() + next_token_index);
          std::string new_code =
              absl::StrCat(var_or_let, space, " ", var_name, " : ", type);
          if (local_mem) {
            absl::StrAppend(&local_mem_declarations_, new_code, ";\n");
            i = start_index - 1;
            continue;
          }
          if (constant_mem) {
            absl::StrAppend(&new_code, " = ", type);
          }
          std::vector<Token> insert_tokens;
          RETURN_IF_ERROR(
              ParseToTokensInternal(std::move(new_code), &insert_tokens));
          tokens_.insert(tokens_.begin() + start_index, insert_tokens.begin(),
                         insert_tokens.end());
          i = start_index + insert_tokens.size() - 1;
        }
      }
    }
    return absl::OkStatus();
  }

  std::string GetCode() const { return GetCode(tokens_); }
  std::string GetCode(const std::vector<Token>& tokens) const {
    std::vector<absl::string_view> text;
    text.reserve(tokens.size());
    for (const auto& token : tokens) {
      text.push_back(token.text);
    }
    return absl::StrJoin(text, "");
  }

  absl::Status MoveLocalMemToGlobalSpace() {
    for (int i = 0; i < tokens_.size(); ++i) {
      if (tokens_[i].text == "MAIN_FUNCTION") {
        std::vector<Token> insert_tokens;
        RETURN_IF_ERROR(ParseToTokens(local_mem_declarations_, &insert_tokens));
        tokens_.insert(tokens_.begin() + i, insert_tokens.begin(),
                       insert_tokens.end());
        return absl::OkStatus();
      }
    }
    return absl::OkStatus();
  }

  absl::Status GetExtensionsInfo(ExtensionsInfo* extensions_info) {
    for (int i = 0; i < tokens_.size(); ++i) {
      if (tokens_[i].text == "enable" || tokens_[i].text == "requires") {
        const int end_index = FindNext(i, TokenType::kSpecial, {";"});
        if (end_index == 0) {
          return absl::InternalError("Failed to parse extension name.");
        }
        std::string extension_name;
        for (int k = i + 2; k < end_index; ++k) {
          extension_name += tokens_[k].text;
        }
        if (tokens_[i].text == "enable") {
          extensions_info->enable_extensions.push_back(extension_name);
        } else if (tokens_[i].text == "requires") {
          extensions_info->language_extensions.push_back(extension_name);
        }
        tokens_.erase(tokens_.begin() + i, tokens_.begin() + end_index + 1);
        i -= 1;
      }
    }
    return absl::OkStatus();
  }

  struct MacroWithArguments {
    std::vector<absl::string_view> argument_names;
    std::vector<Token> code_tokens;

    absl::Status Init(const std::vector<Token>& tokens) {
      int opened = 1;
      int closed = 0;
      int index = 2;
      while (opened != closed && index < tokens.size()) {
        if (tokens[index].type == TokenType::kWord) {
          argument_names.push_back(tokens[index].text);
        }
        if (tokens[index].text == "(") {
          opened++;
        } else if (tokens[index].text == ")") {
          closed++;
        }
        index++;
      }
      if (tokens[index].type == TokenType::kSpaces) {
        index++;
      }
      for (; index < tokens.size(); ++index) {
        if (tokens[index].text != "\\") {
          code_tokens.push_back(tokens[index]);
        }
      }
      return absl::OkStatus();
    }

    std::vector<Token> GetCode(
        const std::vector<std::vector<Token>>& arguments) {
      std::vector<Token> tokens;
      for (const auto& t : code_tokens) {
        int arg_index = -1;
        for (int i = 0; i < argument_names.size(); ++i) {
          if (argument_names[i] == t.text) {
            arg_index = i;
            break;
          }
        }
        if (arg_index != -1) {
          for (const auto& arg_token : arguments[arg_index]) {
            tokens.push_back(arg_token);
          }
        } else {
          tokens.push_back(t);
        }
      }
      return tokens;
    }
  };

 public:
  std::list<std::string> owned_strings_;
  std::string local_mem_declarations_;
  std::map<absl::string_view, std::vector<Token>> macros_;
  std::map<absl::string_view, MacroWithArguments> macros_with_arguments_;
  std::vector<Token> tokens_;
};

}  // namespace

absl::Status ConvertToWGSL(const WebGpuInfo& webgpu_info, std::string* code,
                           ExtensionsInfo* extensions_info) {
  Lexer lexer;
  RETURN_IF_ERROR(lexer.AddMacro("fabs", "abs"));
  RETURN_IF_ERROR(lexer.AddMacro("rsqrt", "inverseSqrt"));

  RETURN_IF_ERROR(lexer.Init(*code));
  RETURN_IF_ERROR(lexer.ConvertTypes(webgpu_info));
  lexer.RemoveFloatPostfix();
  RETURN_IF_ERROR(lexer.ResolveSingleStatementIf());
  RETURN_IF_ERROR(lexer.ResolveDoWhileLoop());
  RETURN_IF_ERROR(lexer.ResolveForLoop());
  RETURN_IF_ERROR(lexer.ResolveTernaryOperator());
  RETURN_IF_ERROR(lexer.ResolvePreIncrement());
  RETURN_IF_ERROR(lexer.MoveLocalMemToGlobalSpace());
  RETURN_IF_ERROR(lexer.GetExtensionsInfo(extensions_info));
  *code = lexer.GetCode();
  return absl::OkStatus();
}

}  // namespace webgpu
}  // namespace ml_drift
