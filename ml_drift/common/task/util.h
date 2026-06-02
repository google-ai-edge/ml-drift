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

#ifndef ML_DRIFT_COMMON_TASK_UTIL_H_
#define ML_DRIFT_COMMON_TASK_UTIL_H_

#include <stddef.h>

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

absl::Status DataTypeFromTemplateArg(absl::string_view type_str, DataType* type,
                                     int* vector_size);

std::string MemoryTypeToCLType(MemoryType type);

std::string MemoryTypeToMetalType(MemoryType type);

// Returns float4 mask for last plane(batch of 4 channels)
// assumes that plane size is 4;
// for example we have 7 channels, in our data structures we align it to 8
// but 8s-channel will be empty, then last plane (batch of 4 channels) will
// have this mask (1, 1, 1, 0).
float4 GetMaskForLastPlane(int channels);

// task_size as amount of FLT4 processed elements.
int GetRecommendedBlockSizeForConv(const GpuInfo& gpu_info,
                                   CalculationsPrecision precision,
                                   int task_size);

int3 GetWorkGroupsCount(const int3& grid_size, const int3& work_group_size);

std::string GetTypeDeclaration(const GpuInfo& gpu_info, DataType data_type,
                               int vec_size);

std::string GetZeroValue(DataType data_type);
std::string GetOneValue(DataType data_type);

absl::string_view GetNextWord(absl::string_view code, size_t first_position);

size_t FindEnclosingBracket(const std::string& text, size_t first_pos,
                            char bracket);

// Single pass function to parse arguments from a string.
// Arguments separated by comma.
// Arguments can have internal brackets ( [ { < with own arguments, but will be
// treated as single argument. Function accept string "text" and first position
// of open bracket to start parsing from.
// Important: '>>' and '<<' always treated as bitshift operators. It means that
// if you need nested '<' bracket use space between them. For example:
// func<vec4<float> >(...).
absl::Status ParseArguments(absl::string_view text, size_t open_bracket_pos,
                            size_t* close_bracket_pos,
                            std::vector<std::string>* arguments);

absl::Status PerformSystemFunction(
    const GpuInfo& gpu_info, absl::string_view function_name,
    const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_TASK_UTIL_H_
