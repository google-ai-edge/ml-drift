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

#include "ml_drift/common/task/work_group_picking.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <vector>

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

namespace {
void AddCornerCases(const int3& grid, int max_work_group_total_size,
                    const int3& max_work_group_sizes, bool aligned,
                    std::vector<int3>* work_groups) {
  for (int x = 1; x <= 4; ++x) {
    for (int y = 1; y <= 4; ++y) {
      for (int z = 1; z <= 4; ++z) {
        int wg_x = DivideRoundUp(grid.x, x);
        int wg_y = DivideRoundUp(grid.y, y);
        int wg_z = DivideRoundUp(grid.z, z);
        if (wg_x > max_work_group_sizes.x || wg_y > max_work_group_sizes.y ||
            wg_z > max_work_group_sizes.z ||
            wg_x * wg_y * wg_z > max_work_group_total_size) {
          continue;
        }
        if (aligned && grid.x % wg_x != 0) {
          continue;
        }
        if (aligned && grid.y % wg_y != 0) {
          continue;
        }
        if (aligned && grid.z % wg_z != 0) {
          continue;
        }
        work_groups->push_back({wg_x, wg_y, wg_z});
      }
    }
  }

  // this will add at least {1, 1, 1} always.
  for (int x = 1; x <= 4; ++x) {
    for (int y = 1; y <= 4; ++y) {
      for (int z = 1; z <= 4; ++z) {
        if (x > max_work_group_sizes.x || y > max_work_group_sizes.y ||
            z > max_work_group_sizes.z ||
            x * y * z > max_work_group_total_size) {
          continue;
        }
        if (aligned && grid.x % x != 0) {
          continue;
        }
        if (aligned && grid.y % y != 0) {
          continue;
        }
        if (aligned && grid.z % z != 0) {
          continue;
        }
        work_groups->push_back({x, y, z});
      }
    }
  }
}

std::set<int> GetSetOfDivisors(int number) {
  const int max_divisor = static_cast<int>(std::sqrt(number));
  std::set<int> divisors;
  for (int i = 1; i <= max_divisor; ++i) {
    const int d = number / i;
    if (i * d == number) {
      divisors.insert(i);
      if (d != i) {
        divisors.insert(d);
      }
    }
  }
  return divisors;
}

std::vector<int> GetDivisors(int number) {
  const auto set_divisors = GetSetOfDivisors(number);
  return std::vector<int>(set_divisors.begin(), set_divisors.end());
}

std::vector<int> GetDivisorsForRange(int number, int range) {
  const int last_number = number + range;
  const int max_divisor = static_cast<int>(std::sqrt(last_number));
  std::set<int> divisors;
  for (int i = 1; i <= max_divisor; ++i) {
    const int reminder = number % i;
    // iterate through numbers that divisible by i in our range;
    const int first_number = number + (i - reminder) % i;
    if (first_number <= last_number) {
      divisors.insert(i);
    }
    for (int j = first_number; j <= last_number; j += i) {
      const int d = j / i;
      if (d != i) {
        divisors.insert(d);
      }
    }
  }
  return std::vector<int>(divisors.begin(), divisors.end());
}

std::vector<int> GetPossibleSizes(int number, bool aligned) {
  if (aligned) {
    // we will use for potential sizes, sizes that cover grid precisely
    // work group size * k (k is integer) == grid_size
    return GetDivisors(number);
  } else {
    // when we chose work group size we can use work group size that
    //   work group size * k (k is integer) != grid_size (slightly bigger)
    // so in this heuristic we trying to find potential size, that satisfies
    //   to this : work group size * k (k is integer) <= grid_size + 5
    //   and this : work group size * k (k is integer) >= grid_size
    return GetDivisorsForRange(number, 5);
  }
}

std::vector<int3> GenerateWorkGroupSizes(const int3& grid,
                                         int min_work_group_total_size,
                                         int max_work_group_total_size,
                                         const int3& max_work_group_sizes,
                                         bool aligned) {
  std::vector<int3> work_groups;
  work_groups.reserve(64);

  std::vector<int> sizes_x = GetPossibleSizes(grid.x, aligned);
  std::vector<int> sizes_y = GetPossibleSizes(grid.y, aligned);
  std::vector<int> sizes_z = GetPossibleSizes(grid.z, aligned);
  if (grid.x == 1 && grid.y == 1) {
    sizes_z.push_back(32);
    sizes_z.push_back(64);
    sizes_z.push_back(128);
    sizes_z.push_back(256);
    sizes_z.push_back(512);
  }

  for (auto x : sizes_x) {
    if (x > max_work_group_sizes.x) continue;
    for (auto y : sizes_y) {
      if (y > max_work_group_sizes.y) continue;
      for (auto z : sizes_z) {
        if (z > max_work_group_sizes.z) continue;
        const int work_group_size = x * y * z;
        if (work_group_size < min_work_group_total_size ||
            work_group_size > max_work_group_total_size)
          continue;
        work_groups.push_back({x, y, z});
      }
    }
  }

  return work_groups;
}

std::vector<int3> GenerateWorkGroupSizesAlignedToGrid(
    const int3& grid, const int3& max_work_group_size,
    const int max_work_group_total_size) {
  auto work_groups = GenerateWorkGroupSizes(
      grid, /*min_work_group_total_size = */ 32, max_work_group_total_size,
      max_work_group_size, /*aligned=*/true);
  // If the grid parameter too small, method below cannot generate workgroups.
  if (work_groups.empty()) {
    AddCornerCases(grid, max_work_group_total_size, max_work_group_size,
                   /*aligned=*/true, &work_groups);
  }
  return work_groups;
}

std::vector<int2> Get2DWorkgroupsEqualToN(int n) {
  if (n == 128) {
    return {{128, 1}, {64, 2}, {32, 4}, {16, 8},
            {8, 16},  {4, 32}, {2, 64}, {1, 128}};
  } else if (n == 64) {
    return {{64, 1}, {32, 2}, {16, 4}, {8, 8}, {4, 16}, {2, 32}, {1, 64}};
  } else if (n == 32) {
    return {{32, 1}, {16, 2}, {8, 4}, {4, 8}, {2, 16}, {1, 32}};
  } else if (n == 16) {
    return {{16, 1}, {8, 2}, {4, 4}, {2, 8}, {1, 16}};
  } else if (n == 8) {
    return {{8, 1}, {4, 2}, {2, 4}, {1, 8}};
  } else if (n == 4) {
    return {{4, 1}, {2, 2}, {1, 4}};
  } else if (n == 2) {
    return {{2, 1}, {1, 2}};
  }
  return {{n, 1}, {1, n}};
}

std::vector<int> GetDefaultWGSizes(int grid_size) {
  const std::set<int> default_sizes = {1, 2, 3, 4, 6, 8, 16, 32, 64, 128, 256};
  auto possible_sizes = GetSetOfDivisors(grid_size);
  possible_sizes.insert(default_sizes.begin(), default_sizes.end());
  return std::vector<int>(possible_sizes.begin(), possible_sizes.end());
}

std::vector<int3> GetWorkGroupsAlignedToGrid(const GpuInfo& gpu_info,
                                             const KernelInfo& kernel_info,
                                             const int3& grid) {
  int3 max_wg_size;
  max_wg_size.x = gpu_info.GetMaxWorkGroupSizeForX();
  max_wg_size.y = gpu_info.GetMaxWorkGroupSizeForY();
  max_wg_size.z = gpu_info.GetMaxWorkGroupSizeForZ();
  return GenerateWorkGroupSizesAlignedToGrid(grid, max_wg_size,
                                             kernel_info.max_work_group_size);
}

int GetPenalty(int grid_size, int group_size) {
  const int reminder = grid_size % group_size;
  return reminder == 0 ? 0 : group_size - reminder;
}

int GetPenalty(int2 grid_size, int2 group_size) {
  const int p_x = GetPenalty(grid_size.x, group_size.x);
  const int p_y = GetPenalty(grid_size.y, group_size.y);
  return p_x * grid_size.y + p_y * grid_size.x + p_x * p_y;
}

int GetMaxSizeWithMinPenalty(int multiplier, int size, int max_size) {
  int best_size = multiplier;
  int min_penalty = GetPenalty(size, best_size);
  for (int i = 2; i * multiplier <= max_size; ++i) {
    if (GetPenalty(size, i * multiplier) == min_penalty) {
      best_size = i * multiplier;
    }
  }
  return best_size;
}

int2 GetMaxSizeWithMinPenalty(int multiplier, int2 size, int max_size) {
  std::vector<int2> base_groups = Get2DWorkgroupsEqualToN(/*n=*/multiplier);
  int min_penalty = std::numeric_limits<int>::max();
  for (const auto& group : base_groups) {
    min_penalty = std::min(GetPenalty(size, group), min_penalty);
  }
  for (const auto& group : base_groups) {
    const int max_y = std::max(1, max_size / group.y);
    for (int y = max_y; y > 0; y -= 1) {
      int new_group_y = y * group.y;
      const int max_x = std::max(1, max_size / group.x);
      for (int x = max_x; x > 0; x -= 1) {
        int new_group_x = x * group.x;
        if (new_group_x * new_group_y > max_size) {
          continue;
        }
        if (GetPenalty(size, int2(new_group_x, new_group_y)) == min_penalty) {
          return int2(new_group_x, new_group_y);
        }
      }
    }
  }
  return int2(0, 0);
}

int GetBiggestDividerWithPriority(int number, int max_divider) {
  if (number % 8 == 0 && 8 <= max_divider) {
    return 8;
  }
  if (number % 4 == 0 && 4 <= max_divider) {
    return 4;
  }
  if (number % 2 == 0 && 2 <= max_divider) {
    return 2;
  }
  for (int i = max_divider; i != 0; i--) {
    if (number % i == 0) {
      return i;
    }
  }
  return 1;
}

int GetBiggestDivider(int number, int max_divider) {
  for (int i = max_divider; i != 0; i--) {
    if (number % i == 0) {
      return i;
    }
  }
  return 1;
}

int GetOptimalSizeForApple(int grid_size) {
  if (grid_size % 8 == 0 || grid_size % 8 >= 4 || grid_size >= 16) {
    return 8;
  }
  if (grid_size % 4 == 0 || grid_size % 4 >= 2 || grid_size >= 8) {
    return 4;
  }
  if (grid_size % 2 == 0 || grid_size >= 4) {
    return 2;
  }
  return 1;
}

int3 GetWorkGroupSizeForApple(const int3& grid_size) {
  int x_size = GetOptimalSizeForApple(grid_size.x);
  int y_size = GetOptimalSizeForApple(grid_size.y);
  int z_size = std::max(1, 32 / (x_size * y_size));
  z_size = std::min(z_size, static_cast<int>(grid_size.z));
  return {x_size, y_size, z_size};
}

}  // namespace

int3 GetConvWorkGroupXisMultipleKYis1(int k, const int3& grid) {
  int grid_z = GetBiggestDividerWithPriority(grid.z, 4);
  if (grid.x <= k) {
    return int3(k, 1, grid_z);
  }
  int grid_x = GetMaxSizeWithMinPenalty(k, grid.x, 512 / grid_z);
  return {grid_x, 1, grid_z};
}

int3 GetConvWorkGroupXYisMultipleK(int k, const int3& grid, int max_wg_size,
                                   int max_z_size) {
  int grid_z = GetBiggestDividerWithPriority(grid.z, max_z_size);
  int wg_x = k;
  int wg_y = 1;
  while (wg_x > wg_y * 2 && wg_x % 2 == 0) {
    wg_x /= 2;
    wg_y *= 2;
  }
  if (grid.x <= wg_x && grid.y <= wg_y) {
    return int3(wg_x, wg_y, grid_z);
  }
  int2 grid_xy =
      GetMaxSizeWithMinPenalty(k, int2(grid.x, grid.y), max_wg_size / grid_z);
  return int3(grid_xy.x, grid_xy.y, grid_z);
}

int3 GetSimpleWorkGroupXYisMultipleK(int k, const int3& grid) {
  int wg_x = k;
  int wg_y = 1;
  while (wg_x > wg_y * 2 && wg_x % 2 == 0) {
    wg_x /= 2;
    wg_y *= 2;
  }
  return int3(wg_x, wg_y, 1);
}

int3 GetWorkGroup(const int3& grid, const GpuInfo& gpu_info,
                  int max_total_size) {
  int wg_z = GetBiggestDividerWithPriority(grid.z, 8);
  int wg_xy_size = max_total_size / wg_z;
  int wg_x = std::min(DivideRoundUp(grid.x, 2), wg_xy_size);
  if (grid.x == 1) {
    wg_x = 1;
  }
  wg_x = std::min(wg_x, gpu_info.GetMaxWorkGroupSizeForX());
  int wg_y = std::min(wg_xy_size / wg_x, grid.y);
  wg_y = std::min(wg_y, gpu_info.GetMaxWorkGroupSizeForY());
  if (wg_x * wg_y < max_total_size) {
    wg_z = max_total_size / (wg_x * wg_y);
  }
  wg_z = std::min(wg_z, gpu_info.GetMaxWorkGroupSizeForZ());
  return int3(wg_x, wg_y, wg_z);
}

int3 GetWorkGroupConv(const int3& grid, int max_size, int max_z_size) {
  int wg_z = GetBiggestDivider(grid.z, max_z_size);
  int wg_xy_size = std::min(256, max_size) / wg_z;
  int wg_x = std::min(grid.x, wg_xy_size);
  int wg_y = std::min(wg_xy_size / wg_x, grid.y);
  if (wg_y == grid.y && grid.y % 2 == 0) {
    wg_y = grid.y / 2;
  }
  return int3(wg_x, wg_y, wg_z);
}

std::vector<int3> GetWorkGroupsXYMultipleOf(int multiplier,
                                            const GpuInfo& gpu_info,
                                            const KernelInfo& kernel_info,
                                            const int3& grid) {
  std::vector<int3> work_groups;
  work_groups.reserve(32);
  const auto possible_z_sizes = GetDefaultWGSizes(grid.z);

  for (int x = 1; x <= kernel_info.max_work_group_size; x *= 2) {
    for (int y = 1; y <= kernel_info.max_work_group_size; y *= 2) {
      int work_group_size_xy = x * y;
      if (work_group_size_xy % multiplier != 0 ||
          work_group_size_xy > kernel_info.max_work_group_size) {
        continue;
      }
      for (auto z : possible_z_sizes) {
        if (work_group_size_xy * z > kernel_info.max_work_group_size) {
          continue;
        }
        if (x <= gpu_info.GetMaxWorkGroupSizeForX() &&
            y <= gpu_info.GetMaxWorkGroupSizeForY() &&
            z <= gpu_info.GetMaxWorkGroupSizeForZ()) {
          work_groups.push_back({x, y, z});
        }
      }
    }
  }
  return work_groups;
}

std::vector<int3> GetWorkGroupsXMultipleOf(int multiplier,
                                           const GpuInfo& gpu_info,
                                           const KernelInfo& kernel_info,
                                           const int3& grid) {
  auto possible_x_sizes = GetDefaultWGSizes(DivideRoundUp(grid.x, multiplier));
  auto possible_y_sizes = GetDefaultWGSizes(grid.y);
  auto possible_z_sizes = GetDefaultWGSizes(grid.z);
  for (auto& value : possible_x_sizes) {
    value *= multiplier;
  }
  auto remove_too_big_sizes = [](std::vector<int>& possible_sizes, int max_size,
                                 int grid_size) {
    for (int i = 1; i < possible_sizes.size(); ++i) {
      if (possible_sizes[i] > max_size || possible_sizes[i] >= grid_size * 2) {
        possible_sizes.erase(possible_sizes.begin() + i, possible_sizes.end());
        break;
      }
    }
  };
  remove_too_big_sizes(possible_x_sizes, gpu_info.GetMaxWorkGroupSizeForX(),
                       grid.x);
  remove_too_big_sizes(possible_y_sizes, gpu_info.GetMaxWorkGroupSizeForY(),
                       grid.y);
  remove_too_big_sizes(possible_z_sizes, gpu_info.GetMaxWorkGroupSizeForZ(),
                       grid.z);

  std::vector<int3> work_groups;
  work_groups.reserve(32);
  for (int x : possible_x_sizes) {
    for (int y : possible_y_sizes) {
      for (int z : possible_z_sizes) {
        if (x * y * z > kernel_info.max_work_group_size) {
          continue;
        }
        work_groups.push_back({x, y, z});
      }
    }
  }
  return work_groups;
}

std::vector<int3> GetPossibleWorkGroups(TuningType tuning_type,
                                        const GpuInfo& gpu_info,
                                        const KernelInfo& kernel_info,
                                        const int3& grid) {
  if (gpu_info.IsApple()) {
    return {GetWorkGroupSizeForApple(grid)};
  }
  switch (tuning_type) {
    case TuningType::kFast:
      return {GetWorkGroup(grid, gpu_info, kernel_info.max_work_group_size)};
    case TuningType::kExhaustive:
      return GetWorkGroupsAlignedToGrid(gpu_info, kernel_info, grid);
    default:
      return {{8, 4, 1}};
  }
}

std::vector<int3> GetPossibleWorkGroupsConv(TuningType tuning_type,
                                            const GpuInfo& gpu_info,
                                            const KernelInfo& kernel_info,
                                            const int3& grid) {
  if (gpu_info.IsApple()) {
    return {GetWorkGroupSizeForApple(grid)};
  }
  switch (tuning_type) {
    case TuningType::kFast: {
      int max_z_size = 16;
      if (gpu_info.IsAdreno()) {
        max_z_size = gpu_info.adreno_info.IsAdreno3xx() ? 16 : 64;
      }
      max_z_size = std::min(max_z_size, gpu_info.GetMaxWorkGroupSizeForZ());
      return {
          GetWorkGroupConv(grid, kernel_info.max_work_group_size, max_z_size)};
    }
    case TuningType::kExhaustive:
      return GetWorkGroupsAlignedToGrid(gpu_info, kernel_info, grid);
    default:
      return {{8, 4, 1}};
  }
}

int3 GetFirstSuitableWorkGroup(const std::vector<int3>& wgs, int max_wg_size) {
  for (const auto& wg : wgs) {
    const int wg_size = wg.x * wg.y * wg.z;
    if (wg_size <= max_wg_size) {
      return wg;
    }
  }
  return {1, 1, 1};
}

}  // namespace ml_drift
