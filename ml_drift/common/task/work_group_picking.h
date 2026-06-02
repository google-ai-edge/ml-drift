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

#ifndef ML_DRIFT_COMMON_TASK_WORK_GROUP_PICKING_H_
#define ML_DRIFT_COMMON_TASK_WORK_GROUP_PICKING_H_

#include <vector>

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

// multiplier can be power of two only
std::vector<int3> GetWorkGroupsXYMultipleOf(int multiplier,
                                            const GpuInfo& gpu_info,
                                            const KernelInfo& kernel_info,
                                            const int3& grid);

std::vector<int3> GetWorkGroupsXMultipleOf(int multiplier,
                                           const GpuInfo& gpu_info,
                                           const KernelInfo& kernel_info,
                                           const int3& grid);

int3 GetConvWorkGroupXisMultipleKYis1(int k, const int3& grid);
int3 GetConvWorkGroupXYisMultipleK(int k, const int3& grid,
                                   int max_wg_size = 512, int max_z_size = 4);
int3 GetSimpleWorkGroupXYisMultipleK(int k, const int3& grid);

std::vector<int3> GetPossibleWorkGroups(TuningType tuning_type,
                                        const GpuInfo& gpu_info,
                                        const KernelInfo& kernel_info,
                                        const int3& grid);

std::vector<int3> GetPossibleWorkGroupsConv(TuningType tuning_type,
                                            const GpuInfo& gpu_info,
                                            const KernelInfo& kernel_info,
                                            const int3& grid);

// returns first work group from wgs that has size not bigger than max_wg_size
// if no suitable groups among wgs, returns {1, 1, 1}
int3 GetFirstSuitableWorkGroup(const std::vector<int3>& wgs, int max_wg_size);

}  // namespace ml_drift

#endif  // ML_DRIFT_COMMON_TASK_WORK_GROUP_PICKING_H_
