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

#include "ml_drift/common/kernels/reduce.h"

#include <any>  // IWYU pragma: keep
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/substitute.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/util.h"
#include "ml_drift/common/task/work_group_picking.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

namespace {
int GetMaximumWGTotalSize(const GpuInfo& gpu_info) {
  // total_wg_size must be power of 2 and >= 4;
  int total_wg_size = 256;
  if (gpu_info.IsAdreno() && gpu_info.adreno_info.IsAdreno3xx()) {
    total_wg_size = 128;
  }
  if (gpu_info.IsMali()) {
    const MaliInfo& mali_info = gpu_info.mali_info;
    if (mali_info.IsMidgard()) {
      total_wg_size = 32;
    } else {
      total_wg_size = 64;
    }
  }
  return total_wg_size;
}

bool HasAxis(const std::vector<Axis>& axis, Axis a) {
  for (const auto& a2 : axis) {
    if (a2 == a) {
      return true;
    }
  }
  return false;
}

std::string MakeOp(Reduce::Type reduce_type, const std::string& accum,
                   const std::string& to_accum, const std::string& accum_ind,
                   const std::string& to_accum_ind, bool vec4 = true) {
  const std::string postfixes[4] = {"x", "y", "z", "w"};
  switch (reduce_type) {
    case Reduce::Type::kAll:
      if (vec4) {
        std::string vec4_code;
        for (int i = 0; i < 4; i++) {
          vec4_code +=
              absl::Substitute("$$0.$0 = $$0.$0 && $$1.$0;", postfixes[i]);
          if (i != 3) {
            vec4_code += "\n";
          }
        }
        return absl::Substitute(vec4_code, accum, to_accum);
      } else {
        return absl::Substitute("$0 = $0 && $1", accum, to_accum);
      }
    case Reduce::Type::kAny:
      if (vec4) {
        std::string vec4_code;
        for (int i = 0; i < 4; i++) {
          vec4_code +=
              absl::Substitute("$$0.$0 = $$0.$0 || $$1.$0;", postfixes[i]);
          if (i != 3) {
            vec4_code += "\n";
          }
        }
        return absl::Substitute(vec4_code, accum, to_accum);
      } else {
        return absl::Substitute("$0 = $0 || $1", accum, to_accum);
      }
    case Reduce::Type::kMean:
    case Reduce::Type::kMeanSquares:
    case Reduce::Type::kSum:
      return absl::Substitute("$0 += ($1);", accum, to_accum);
    case Reduce::Type::kProduct:
      return absl::Substitute("$0 *= ($1);", accum, to_accum);
    case Reduce::Type::kMaximum:
      return absl::Substitute("$0 = max($0, $1);", accum, to_accum);
    case Reduce::Type::kMinimum:
      return absl::Substitute("$0 = min($0, $1);", accum, to_accum);
    case Reduce::Type::kMaximumIndex:
      if (vec4) {
        std::string code;
        for (int i = 0; i < 4; i++) {
          code += absl::Substitute(
              "if ($$1.$0 > $$0.$0) { $$0.$0 = $$1.$0; $$2.$0 = $$3.$0;};",
              postfixes[i]);
          if (i != 3) {
            code += "\n";
          }
        }
        return absl::Substitute(code, accum, to_accum, accum_ind, to_accum_ind);
      } else {
        std::string code = R"( if ($1 >= $0) {
  if ($1 > $0) {
    $0 = $1; $2 = $3;
  }
  else {
    $2 = min($2, $3);
  }
})";
        return absl::Substitute(code, accum, to_accum, accum_ind, to_accum_ind);
      }
  }
}

// max_total_wg_size is pot
int3 GetMaximumPossibleWGSize(const std::vector<int>& ordered_sizes,
                              int max_total_wg_size) {
  int3 wg_size = int3(1, 1, 1);
  int wg_size_total = 1;
  for (int i = ordered_sizes.size() - 1; i >= 0; i--) {
    const int wg_index = ordered_sizes.size() - 1 - i;
    if (wg_index >= 3) {
      return wg_size;
    }
    while (ordered_sizes[i] >= wg_size[wg_index] * 2) {
      wg_size_total *= 2;
      if (wg_size_total > max_total_wg_size) {
        return wg_size;
      }
      wg_size[wg_index] *= 2;
    }
  }
  return wg_size;
}

std::map<Axis, int> GetSizesFromShape(const std::set<Axis>& axis,
                                      const BHWC& shape) {
  std::map<Axis, int> result;
  for (auto a : axis) {
    result[a] = shape.get(a);
  }
  return result;
}

std::map<Axis, int> GetSizesFromShape(const std::set<Axis>& axis,
                                      const BHWDC& shape) {
  std::map<Axis, int> result;
  for (auto a : axis) {
    result[a] = shape.get(a);
  }
  return result;
}

DataType GetAccumType(DataType src_type) {
  if (src_type == DataType::kFloat32 || src_type == DataType::kFloat16) {
    return DataType::kFloat32;
  } else if (src_type == DataType::kInt32 || src_type == DataType::kInt16 ||
             src_type == DataType::kInt8) {
    return DataType::kInt32;
  } else if (src_type == DataType::kUint32 || src_type == DataType::kUint16 ||
             src_type == DataType::kUint8) {
    return DataType::kUint32;
  } else {
    return src_type;
  }
}

bool IsAverageReduce(Reduce::Type reduce_type) {
  return reduce_type == Reduce::Type::kMean ||
         reduce_type == Reduce::Type::kMeanSquares;
}

}  // namespace

Reduce::Type GetReduceTypeFromOperationType(OperationType op_type) {
  if (op_type == OperationType::kMean) {
    return Reduce::Type::kMean;
  } else if (op_type == OperationType::kReduceAll) {
    return Reduce::Type::kAll;
  } else if (op_type == OperationType::kReduceAny) {
    return Reduce::Type::kAny;
  } else if (op_type == OperationType::kReduceSum) {
    return Reduce::Type::kSum;
  } else if (op_type == OperationType::kReduceProduct) {
    return Reduce::Type::kProduct;
  } else if (op_type == OperationType::kReduceMaximum) {
    return Reduce::Type::kMaximum;
  } else if (op_type == OperationType::kReduceMinimum) {
    return Reduce::Type::kMinimum;
  }
  return Reduce::Type::kMean;
}

Reduce::Reduce(const std::map<Axis, int>& axis_to_reduce, Type reduce_type,
               const OperationDef& definition, const GpuInfo& gpu_info,
               bool add_input) {
  std::vector<Axis> ordered_axis_to_reduce;
  std::vector<int> ordered_sizes;
  for (const auto& a : {Axis::kChannels, Axis::kDepth, Axis::kHeight,
                        Axis::kWidth, Axis::kBatch}) {
    auto it = axis_to_reduce.find(a);
    if (it != axis_to_reduce.end()) {
      ordered_axis_to_reduce.push_back(it->first);
      int reduction_size = it->second;
      if (a == Axis::kChannels) {
        reduction_size = DivideRoundUp(reduction_size, 4);
      }
      ordered_sizes.push_back(reduction_size);
    }
  }
  const int max_total_wg_size = GetMaximumWGTotalSize(gpu_info);
  int3 current_wg_size =
      GetMaximumPossibleWGSize(ordered_sizes, max_total_wg_size);
  int current_wg_size_total =
      current_wg_size.x * current_wg_size.y * current_wg_size.z;
  int threshold = max_total_wg_size / 4;
  if (gpu_info.IsApple()) {
    threshold = 16;
  }
  if (current_wg_size_total < threshold) {
    use_wg_reduction_ = false;
  } else {
    use_wg_reduction_ = true;
    work_group_size_ = current_wg_size;
  }
  code_ = GetReduceKernelCode(definition, gpu_info, work_group_size_,
                              ordered_axis_to_reduce, reduce_type, add_input);
}

Reduce::Reduce(Reduce&& operation)
    : GPUOperation(std::move(operation)),
      use_wg_reduction_(operation.use_wg_reduction_) {}

Reduce& Reduce::operator=(Reduce&& operation) {
  if (this != &operation) {
    use_wg_reduction_ = operation.use_wg_reduction_;
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

std::string Reduce::GetReduceKernelCode(const OperationDef& op_def,
                                        const GpuInfo& gpu_info,
                                        const int3& work_group_size,
                                        const std::vector<Axis>& axis_to_reduce,
                                        Type reduce_type, bool add_input) {
  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  if (add_input) {
    AddSrcTensor("src_tensor_1", op_def.src_tensors[1]);
  }
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  args_.AddFloat("inv_multiplier_1");
  args_.AddFloat("inv_multiplier_2");

  std::set<Axis> axis_to_leave;
  const std::vector<Axis> all_axis = {Axis::kWidth, Axis::kHeight, Axis::kDepth,
                                      Axis::kChannels, Axis::kBatch};
  for (const auto& a : all_axis) {
    if (op_def.dst_tensors[0].HasAxis(a)) {
      if (!HasAxis(axis_to_reduce, a)) {
        axis_to_leave.insert(a);
      }
    }
  }
  const bool channels_reduction = HasAxis(axis_to_reduce, Axis::kChannels);
  int wg_dims = 0;
  if (use_wg_reduction_) {
    if (work_group_size.y == 1 && work_group_size.z == 1) {
      wg_dims = 1;
    } else if (work_group_size.z == 1) {
      wg_dims = 2;
    } else {
      wg_dims = 3;
    }
  }

  auto get_global_id = [&](int i) {
    if (use_wg_reduction_) {
      return "ucl::GetGroupId<" + std::to_string(i) + ">()";
    } else {
      return "ucl::GetGlobalId<" + std::to_string(i) + ">()";
    }
  };

  auto accum_type = GetAccumType(op_def.src_tensors[0].GetDataType());
  const std::string accum_type_decl = ToUclDataType(accum_type, 4);
  std::string read_as_template;
  if (accum_type == DataType::kFloat32) {
    read_as_template = "<float>";
  } else if (accum_type == DataType::kInt32) {
    read_as_template = "<int>";
  } else if (accum_type == DataType::kUint32) {
    read_as_template = "<uint>";
  } else if (accum_type == DataType::kBool) {
    read_as_template = "<bool>";
  }

  std::string c;
  const std::string wg_x = std::to_string(work_group_size.x);
  const std::string wg_y = std::to_string(work_group_size.y);
  const std::string wg_z = std::to_string(work_group_size.z);
  const int wg_total_size =
      work_group_size.x * work_group_size.y * work_group_size.z;
  c += "MAIN_FUNCTION($0) {\n";
  if (use_wg_reduction_) {
    c += "  __local " + accum_type_decl + " accum[" +
         std::to_string(wg_total_size) + "];\n";
    if (reduce_type == Type::kMaximumIndex) {
      c += "  __local int4 accum_ind[" + std::to_string(wg_total_size) + "];\n";
    }
    if (wg_dims == 1) {
      c += "  int local_x = ucl::GetLocalId<0>();\n";
      c += "  int local_id = local_x;\n";
    } else if (wg_dims == 2) {
      c += "  int local_x = ucl::GetLocalId<0>();\n";
      c += "  int local_y = ucl::GetLocalId<1>();\n";
      c += "  int local_id = local_y * " + wg_x + " + local_x;\n";
    } else if (wg_dims == 3) {
      c += "  int local_x = ucl::GetLocalId<0>();\n";
      c += "  int local_y = ucl::GetLocalId<1>();\n";
      c += "  int local_z = ucl::GetLocalId<2>();\n";
      c += "  int local_id = (local_z * " + wg_y + " + local_y) * " + wg_x +
           " + local_x;\n";
    }
  }
  if (axis_to_leave.count(Axis::kWidth)) {
    if (axis_to_leave.count(Axis::kBatch)) {
      c += "  int linear_id = " + get_global_id(0) + ";\n";
      c += "  int DST_X = linear_id / args.dst_tensor.Batch();\n";
      c += "  int DST_B = linear_id % args.dst_tensor.Batch();\n";
    } else {
      c += "  int DST_X = " + get_global_id(0) + ";\n";
    }
  } else if (axis_to_leave.count(Axis::kBatch)) {
    c += "  int DST_B = " + get_global_id(0) + ";\n";
  }
  if (axis_to_leave.count(Axis::kHeight)) {
    if (axis_to_leave.count(Axis::kDepth)) {
      c += "  int linear_id = " + get_global_id(1) + ";\n";
      c += "  int DST_Y = linear_id % args.dst_tensor.Height();\n";
      c += "  int DST_Z = linear_id / args.dst_tensor.Height();\n";
    } else {
      c += "  int DST_Y = " + get_global_id(1) + ";\n";
    }
  } else if (axis_to_leave.count(Axis::kDepth)) {
    c += "  int DST_Z = " + get_global_id(1) + ";\n";
  }
  if (axis_to_leave.count(Axis::kChannels)) {
    c += "  int DST_S = " + get_global_id(2) + ";\n";
  }
  std::map<Axis, std::string> axis_to_selector = {
      {Axis::kBatch, "Batch()"},     {Axis::kWidth, "Width()"},
      {Axis::kHeight, "Height()"},   {Axis::kDepth, "Depth()"},
      {Axis::kChannels, "Slices()"},
  };
  std::map<Axis, std::string> axis_to_coord = {
      {Axis::kBatch, "B"}, {Axis::kWidth, "X"},    {Axis::kHeight, "Y"},
      {Axis::kDepth, "Z"}, {Axis::kChannels, "S"},
  };
  std::string dst_check;
  for (auto& axis : axis_to_leave) {
    if (!dst_check.empty()) {
      dst_check += " || ";
    }
    dst_check += "DST_" + axis_to_coord[axis] + " >= args.dst_tensor." +
                 axis_to_selector[axis];
  }
  if (!dst_check.empty()) {
    c += "  if (" + dst_check + ") return;\n";
  }
  std::map<Axis, std::string> src_coords;
  for (const auto& a : all_axis) {
    if (op_def.dst_tensors[0].HasAxis(a) && !HasAxis(axis_to_reduce, a)) {
      src_coords[a] = "DST_" + axis_to_coord[a];
    } else {
      src_coords[a] = "0";
    }
  }
  std::string src_coordinates;
  for (const auto& a : all_axis) {
    if (op_def.src_tensors[0].HasAxis(a)) {
      if (!src_coordinates.empty()) {
        src_coordinates += ", ";
      }
      src_coordinates += src_coords[a];
    }
  }
  const std::string vec4_one = "ucl::Init<" + ToUclDataType(accum_type, 4) +
                               ">(" + GetOneValue(accum_type) + ")";
  const std::string vec4_zero = "ucl::Init<" + ToUclDataType(accum_type, 4) +
                                ">(" + GetZeroValue(accum_type) + ")";
  if (reduce_type == Type::kMean || reduce_type == Type::kMeanSquares ||
      reduce_type == Type::kSum || reduce_type == Type::kAny) {
    c += "  " + accum_type_decl + " reducer = " + vec4_zero + ";\n";
  } else if (reduce_type == Type::kProduct || reduce_type == Type::kAll) {
    c += "  " + accum_type_decl + " reducer = " + vec4_one + ";\n";
  } else if (reduce_type == Type::kMaximum || reduce_type == Type::kMinimum ||
             reduce_type == Type::kMaximumIndex) {
    c += "  " + accum_type_decl + " reducer = args.src_tensor.Read" +
         read_as_template + "(" + src_coordinates + ");\n";
    if (channels_reduction) {
      c += "  reducer.y = reducer.x;\n";
      c += "  reducer.z = reducer.x;\n";
      c += "  reducer.w = reducer.x;\n";
    }
    if (reduce_type == Type::kMaximumIndex) {
      c += "  int4 reducer_ind = ucl::Init<int4>(0);\n";
    }
  }
  const std::vector<std::string> local_ids = {"local_x", "local_y", "local_z"};
  const std::vector<std::string> local_sizes = {wg_x, wg_y, wg_z};
  for (const auto& axis : axis_to_reduce) {
    if (axis == Axis::kChannels) {
      c += "  " + accum_type_decl + " mask;\n";
      const std::string one_or_zero_value =
          "ucl::Init<" + ToUclDataType(accum_type, 1) + ">(" +
          GetOneValue(accum_type) + ")" + " : " + "ucl::Init<" +
          ToUclDataType(accum_type, 1) + ">(" + GetZeroValue(accum_type) + ")";
      c += "  mask.x = (args.src_tensor.Slices() - 1) * 4 + 0 < "
           "args.src_tensor.Channels() ? " +
           one_or_zero_value + ";\n";
      c += "  mask.y = (args.src_tensor.Slices() - 1) * 4 + 1 < "
           "args.src_tensor.Channels() ? " +
           one_or_zero_value + ";\n";
      c += "  mask.z = (args.src_tensor.Slices() - 1) * 4 + 2 < "
           "args.src_tensor.Channels() ? " +
           one_or_zero_value + ";\n";
      c += "  mask.w = (args.src_tensor.Slices() - 1) * 4 + 3 < "
           "args.src_tensor.Channels() ? " +
           one_or_zero_value + ";\n";
    }
  }
  for (int i = 0; i < axis_to_reduce.size(); ++i) {
    const auto& axis = axis_to_reduce[i];
    const int index = axis_to_reduce.size() - 1 - i;
    const std::string first = index < wg_dims ? local_ids[index] : "0";
    const std::string step = index < wg_dims ? local_sizes[index] : "1";
    const std::string src_coord = "SRC_" + axis_to_coord[axis];
    src_coords[axis] = src_coord;
    c += "  for (int " + src_coord + " = " + first + "; " + src_coord +
         " < args.src_tensor." + axis_to_selector[axis] + "; " + src_coord +
         " += " + step + ") {\n";
    if (axis == Axis::kChannels) {
      c += "    bool last = SRC_S == args.src_tensor.Slices() - 1;\n";
      c += "    " + accum_type_decl + " mask_a = last ? mask : " + vec4_one +
           ";\n";
      if (reduce_type == Type::kProduct || reduce_type == Type::kMaximum ||
          reduce_type == Type::kMinimum) {
        c +=
            "    " + accum_type_decl + " mask_b = " + vec4_one + " - mask_a;\n";
      }
    }
  }
  src_coordinates = "";
  for (const auto& a : all_axis) {
    if (op_def.src_tensors[0].HasAxis(a)) {
      if (!src_coordinates.empty()) {
        src_coordinates += ", ";
      }
      src_coordinates += src_coords[a];
    }
  }
  c += "    " + accum_type_decl + " src_val = args.src_tensor.Read" +
       read_as_template + "(" + src_coordinates + ");\n";
  if (add_input) {
    c += "    src_val += args.src_tensor_1.Read" + read_as_template + "(" +
         src_coordinates + ");\n";
  }
  if (reduce_type == Type::kMaximumIndex) {
    if (channels_reduction) {
      c += "    int4 src_val_ind = ucl::Init<int4>(SRC_S * 4 + 0, SRC_S * 4 + "
           "1, SRC_S * 4 + 2, SRC_S * 4 + 3);\n";
    } else {
      c += "    int4 src_val_ind = ucl::Init<int4>(SRC_" +
           axis_to_coord[axis_to_reduce[0]] + ");\n";
    }
  }
  if (channels_reduction) {
    if (reduce_type == Type::kMean || reduce_type == Type::kSum) {
      c += "    src_val = src_val * mask_a;\n";
    } else if (reduce_type == Type::kAll) {
      c += "    src_val.x = mask_a.x ? src_val.x : ucl::Init<bool>(1);\n";
      c += "    src_val.y = mask_a.y ? src_val.y : ucl::Init<bool>(1);\n";
      c += "    src_val.z = mask_a.z ? src_val.z : ucl::Init<bool>(1);\n";
      c += "    src_val.w = mask_a.w ? src_val.w : ucl::Init<bool>(1);\n";
    } else if (reduce_type == Type::kAny) {
      c += "    src_val.x = mask_a.x ? src_val.x : ucl::Init<bool>(0);\n";
      c += "    src_val.y = mask_a.y ? src_val.y : ucl::Init<bool>(0);\n";
      c += "    src_val.z = mask_a.z ? src_val.z : ucl::Init<bool>(0);\n";
      c += "    src_val.w = mask_a.w ? src_val.w : ucl::Init<bool>(0);\n";
    } else if (reduce_type == Type::kMeanSquares) {
      c += "    src_val = src_val * src_val * mask_a;\n";
    } else if (reduce_type == Type::kProduct) {
      c += "    src_val = src_val * mask_a + mask_b;\n";
    } else if (reduce_type == Type::kMaximum || reduce_type == Type::kMinimum) {
      c += "    src_val = src_val * mask_a + mask_b * src_val.x;\n";
    }
  }
  c += "    " +
       MakeOp(reduce_type, "reducer", "src_val", "reducer_ind", "src_val_ind") +
       ";\n";
  for (int i = 0; i < axis_to_reduce.size(); ++i) {
    c += "  }\n";
  }
  if (IsAverageReduce(reduce_type)) {
    c += "  reducer *= args.inv_multiplier_1;\n";
  }
  if (use_wg_reduction_) {
    c += "  accum[local_id] = reducer;\n";
    if (reduce_type == Type::kMaximumIndex) {
      c += "  accum_ind[local_id] = reducer_ind;\n";
    }
    c += "  ucl::SyncThreads<WorkGroup, Local>();\n";
    const int total_size =
        work_group_size.x * work_group_size.y * work_group_size.z;
    int offset = 1;
    int remainder = total_size / 4;
    for (; remainder >= 8; remainder /= 4, offset *= 4) {
      c += "  if (local_id < " + std::to_string(remainder) + ") {\n";
      c += "    int t = local_id * " + std::to_string(offset * 4) + ";\n";
      c += "    " + accum_type_decl + " reduced = accum[t];\n";
      if (reduce_type == Type::kMaximumIndex) {
        c += "    int4 reduced_ind = accum_ind[t];\n";
      }
      c +=
          "    " +
          MakeOp(reduce_type, "reduced",
                 "accum[t + " + std::to_string(offset * 1) + "]", "reduced_ind",
                 "accum_ind[t + " + std::to_string(offset * 1) + "]") +
          ";\n";
      c +=
          "    " +
          MakeOp(reduce_type, "reduced",
                 "accum[t + " + std::to_string(offset * 2) + "]", "reduced_ind",
                 "accum_ind[t + " + std::to_string(offset * 2) + "]") +
          ";\n";
      c +=
          "    " +
          MakeOp(reduce_type, "reduced",
                 "accum[t + " + std::to_string(offset * 3) + "]", "reduced_ind",
                 "accum_ind[t + " + std::to_string(offset * 3) + "]") +
          ";\n";
      c += "    accum[t] = reduced;\n";
      if (reduce_type == Type::kMaximumIndex) {
        c += "    accum_ind[t] = reduced_ind;\n";
      }
      c += "  }\n";
      c += "  ucl::SyncThreads<WorkGroup, Local>();\n";
    }
    // Ensure only id 0 executes a write command.
    c += "  if (local_id != 0) { return; }\n";
    c += "  reducer = accum[0];\n";
    if (reduce_type == Type::kMaximumIndex) {
      c += "  reducer_ind = accum_ind[0];\n";
    }
    remainder *= 4;
    for (int i = 1; i < remainder; ++i) {
      c += "  " +
           MakeOp(reduce_type, "reducer",
                  "accum[" + std::to_string(offset * i) + "]", "reducer_ind",
                  "accum_ind[" + std::to_string(offset * i) + "]") +
           ";\n";
    }
    if (IsAverageReduce(reduce_type)) {
      c += "  reducer *= args.inv_multiplier_2;\n";
    }
  }
  if (channels_reduction) {
    c += "  " +
         MakeOp(reduce_type, "reducer.x", "reducer.y", "reducer_ind.x",
                "reducer_ind.y", false) +
         ";\n";
    c += "  " +
         MakeOp(reduce_type, "reducer.x", "reducer.z", "reducer_ind.x",
                "reducer_ind.z", false) +
         ";\n";
    c += "  " +
         MakeOp(reduce_type, "reducer.x", "reducer.w", "reducer_ind.x",
                "reducer_ind.w", false) +
         ";\n";
  }
  if (reduce_type == Type::kMaximumIndex) {
    c += "  args.dst_tensor::type result = "
         "ucl::Convert<args.dst_tensor::type>(reducer_ind);\n";
  } else {
    c += "  args.dst_tensor::type result = "
         "ucl::Convert<args.dst_tensor::type>(reducer);\n";
  }
  std::string dst_coordinates;
  for (const auto& a : all_axis) {
    if (op_def.dst_tensors[0].HasAxis(a)) {
      if (!dst_coordinates.empty()) {
        dst_coordinates += ", ";
      }
      if (axis_to_leave.count(a)) {
        dst_coordinates += "DST_" + axis_to_coord[a];
      } else {
        dst_coordinates += "0";
      }
    }
  }
  c += "  args.dst_tensor.Write(result, " + dst_coordinates + ");\n";
  c += "}\n";
  return c;
}

absl::Status Reduce::BindArguments(ArgumentsBinder* args) {
  const double total_src_elements = 1.0 * src_[0]->Batch() * src_[0]->Width() *
                                    src_[0]->Height() * src_[0]->Depth() *
                                    src_[0]->Channels();
  const double total_dst_elements = 1.0 * dst_[0]->Batch() * dst_[0]->Width() *
                                    dst_[0]->Height() * dst_[0]->Depth() *
                                    dst_[0]->Channels();
  const double reduction_size = total_src_elements / total_dst_elements;
  if (use_wg_reduction_) {
    const double size_0 =
        work_group_size_.x * work_group_size_.y * work_group_size_.z;
    const double size_1 = reduction_size / size_0;
    ABSL_RETURN_IF_ERROR(args->SetFloat("inv_multiplier_1", 1.0 / size_1));
    ABSL_RETURN_IF_ERROR(args->SetFloat("inv_multiplier_2", 1.0 / size_0));
  } else {
    ABSL_RETURN_IF_ERROR(
        args->SetFloat("inv_multiplier_1", 1.0 / reduction_size));
    ABSL_RETURN_IF_ERROR(args->SetFloat("inv_multiplier_2", 1.0));
  }
  return absl::OkStatus();
}

int3 Reduce::GetGridSize() const {
  int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  int grid_y = dst_[0]->Height() * dst_[0]->Depth();
  int grid_z = dst_[0]->Slices();
  if (use_wg_reduction_) {
    grid_x *= work_group_size_.x;
    grid_y *= work_group_size_.y;
    grid_z *= work_group_size_.z;
  }
  return int3(grid_x, grid_y, grid_z);
}

std::vector<int3> Reduce::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info) const {
  if (use_wg_reduction_) {
    return {work_group_size_};
  } else {
    return GetPossibleWorkGroups(tuning_type, gpu_info, kernel_info,
                                 grid_size_);
  }
}

Reduce CreateReduce(const std::set<Axis>& axis_to_reduce, const BHWC& src_shape,
                    OperationType op_type, const OperationDef& definition,
                    const GpuInfo& gpu_info) {
  return Reduce(GetSizesFromShape(axis_to_reduce, src_shape),
                GetReduceTypeFromOperationType(op_type), definition, gpu_info);
}

Reduce Create2InputReduce(const std::set<Axis>& axis_to_reduce,
                          const BHWC& src_shape, OperationType op_type,
                          const OperationDef& definition,
                          const GpuInfo& gpu_info) {
  return Reduce(GetSizesFromShape(axis_to_reduce, src_shape),
                GetReduceTypeFromOperationType(op_type), definition, gpu_info,
                /*add_input*/ true);
}

Reduce CreateReduce(const std::set<Axis>& axis_to_reduce, const BHWC& src_shape,
                    Reduce::Type reduce_type, const OperationDef& definition,
                    const GpuInfo& gpu_info) {
  return Reduce(GetSizesFromShape(axis_to_reduce, src_shape), reduce_type,
                definition, gpu_info);
}

Reduce CreateReduce(const std::set<Axis>& axis_to_reduce,
                    const BHWDC& src_shape, OperationType op_type,
                    const OperationDef& definition, const GpuInfo& gpu_info) {
  return Reduce(GetSizesFromShape(axis_to_reduce, src_shape),
                GetReduceTypeFromOperationType(op_type), definition, gpu_info);
}

Reduce CreateReduce(const std::set<Axis>& axis_to_reduce,
                    const BHWDC& src_shape, Reduce::Type reduce_type,
                    const OperationDef& definition, const GpuInfo& gpu_info) {
  return Reduce(GetSizesFromShape(axis_to_reduce, src_shape), reduce_type,
                definition, gpu_info);
}
}  // namespace ml_drift
