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

#include "ml_drift/common/kernels/conv_weights_converter.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/util.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/task/work_group_picking.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace {
bool Use2DGridXisIOgroupYisO(const GpuInfo& gpu_info, const OHWI& weights_shape,
                             const WeightsDescription& src_weights_desc,
                             const WeightsDescription& dst_weights_desc) {
  bool recommended_gpu =
      gpu_info.IsAdreno() &&
      (gpu_info.adreno_info.generation >= AdrenoInfo::Generation::kGen6);
  recommended_gpu = recommended_gpu || gpu_info.IsPowerVR();
  if (!recommended_gpu) {
    return false;
  }
  const bool supported_src_layout =
      src_weights_desc.layout == WeightsLayout::kOSpatialIOGroupI4O4 ||
      src_weights_desc.layout == WeightsLayout::kOSpatialIOGroupO4I4 ||
      src_weights_desc.layout == WeightsLayout::kOISpatialOGroupI4O4 ||
      src_weights_desc.layout == WeightsLayout::kOISpatialOGroupO4I4 ||
      src_weights_desc.layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4;
  const bool is_src_i_o_4x4 = supported_src_layout &&
                              src_weights_desc.GetOutputGroupSize() ==
                                  DivideRoundUp(weights_shape.o, 4) &&
                              weights_shape.w == 1 && weights_shape.h == 1;
  const bool supported_dst_layout =
      dst_weights_desc.layout == WeightsLayout::kOSpatialIOGroupI4O4 ||
      dst_weights_desc.layout == WeightsLayout::kOSpatialIOGroupO4I4 ||
      dst_weights_desc.layout == WeightsLayout::kOISpatialOGroupI4O4 ||
      dst_weights_desc.layout == WeightsLayout::kOISpatialOGroupO4I4;
  const bool is_dst_o_i_ogroup_4x4 =
      supported_dst_layout && dst_weights_desc.GetOutputGroupSize() <= 32 &&
      weights_shape.w == 1 && weights_shape.h == 1;
  return is_src_i_o_4x4 && is_dst_o_i_ogroup_4x4;
}

// For kCustomGroups layout
bool IsO2I2O2I2(const WeightsDescription& dst_weights_desc) {
  const std::pair<Axis, int> kI2 = {Axis::INPUT_CHANNELS, 2};
  const std::pair<Axis, int> kO2 = {Axis::OUTPUT_CHANNELS, 2};
  const auto& groups = dst_weights_desc.group_sizes;
  return groups.size() >= 4 && groups[0] == kI2 && groups[1] == kO2 &&
         groups[2] == kI2 && groups[3] == kO2;
}

// For kCustomGroups layout
bool IsO2I4O2(const WeightsDescription& dst_weights_desc) {
  const std::pair<Axis, int> kI4 = {Axis::INPUT_CHANNELS, 4};
  const std::pair<Axis, int> kO2 = {Axis::OUTPUT_CHANNELS, 2};
  const auto& groups = dst_weights_desc.group_sizes;
  return groups.size() >= 3 && groups[0] == kO2 && groups[1] == kI4 &&
         groups[2] == kO2;
}

// For kCustomGroups layout
bool IsI2O4I2(const WeightsDescription& dst_weights_desc) {
  const std::pair<Axis, int> kI2 = {Axis::INPUT_CHANNELS, 2};
  const std::pair<Axis, int> kO4 = {Axis::OUTPUT_CHANNELS, 4};
  const auto& groups = dst_weights_desc.group_sizes;
  return groups.size() >= 3 && groups[0] == kI2 && groups[1] == kO4 &&
         groups[2] == kI2;
}

// For kCustomGroups layout
int GetStartGroup(const WeightsDescription& dst_weights_desc) {
  if (IsO2I2O2I2(dst_weights_desc)) {
    return 4;
  } else if (IsO2I4O2(dst_weights_desc)) {
    return 3;
  } else if (IsI2O4I2(dst_weights_desc)) {
    return 3;
  } else {
    return 2;
  }
}

void AddCommonArgs(const WeightsDescription& dst_weights_desc,
                   const OHWI& weights_shape, Arguments& args) {
  args.AddInt("i_slices", DivideRoundUp(weights_shape.i, 4));
  args.AddInt("o_slices", DivideRoundUp(weights_shape.o, 4));
  args.AddInt("dst_o_group_size", dst_weights_desc.GetOutputGroupSize());
  if (dst_weights_desc.layout == WeightsLayout::kCustomGroups) {
    if (IsO2I2O2I2(dst_weights_desc) || IsO2I4O2(dst_weights_desc) ||
        IsI2O4I2(dst_weights_desc)) {
      args.AddInt("last_i_group_slices", 1);
      args.AddInt("last_o_group_slices", 1);
    } else {
      const bool is_i_first =
          dst_weights_desc.group_sizes[0].first == Axis::INPUT_CHANNELS;
      const auto& i_group = is_i_first ? dst_weights_desc.group_sizes[0]
                                       : dst_weights_desc.group_sizes[1];
      const auto& o_group = is_i_first ? dst_weights_desc.group_sizes[1]
                                       : dst_weights_desc.group_sizes[0];
      const int last_i_group_size =
          i_group.second <= 0 ? weights_shape.i : i_group.second;
      const int last_o_group_size =
          o_group.second <= 0 ? weights_shape.o : o_group.second;
      args.AddInt("last_i_group_slices", DivideRoundUp(last_i_group_size, 4));
      args.AddInt("last_o_group_slices", DivideRoundUp(last_o_group_size, 4));
    }
  }
}

// Whether the integer weights can be written with a vector operation.
// Currently, only custom buffer handles the case when a vector operation is
// not possible for int8 or uint8.
// TODO: b/475448239 - Check properly all supported cases, layouts and their
// parameters.
bool IsLastBlockHas4IAnd4OElements(const WeightsDescription& dst_weights_desc) {
  if (dst_weights_desc.layout == WeightsLayout::kCustomGroups &&
      dst_weights_desc.group_sizes[0].second > 4 &&
      (dst_weights_desc.type == DataType::INT8 ||
       dst_weights_desc.type == DataType::UINT8)) {
    return false;
  }
  return true;
}

int GetZeroPoint(DataType data_type) {
  if (data_type == DataType::UINT8) {
    return 128;
  } else if (data_type == DataType::UINT4) {
    return 8;
  } else if (data_type == DataType::UINT2) {
    return 2;
  }
  return 0;
}

std::string MultiplyMask(bool vector_mul) {
  std::string c;
  if (vector_mul) {
    c += "    w0 = w0 * mask;\n";
    c += "    w1 = w1 * mask;\n";
    c += "    w2 = w2 * mask;\n";
    c += "    w3 = w3 * mask;\n";
  } else {
    c += "    w0 = w0 * mask.x;\n";
    c += "    w1 = w1 * mask.y;\n";
    c += "    w2 = w2 * mask.z;\n";
    c += "    w3 = w3 * mask.w;\n";
  }
  return c;
}

std::string ApplyMask(const OHWI& weights_shape, bool is_i4o4) {
  std::string c;
  if (weights_shape.o % 4 != 0) {
    c += "  if (o_slice == args.o_slices - 1) {\n";
    c += "    Type mask = ucl::Init<Type>(args.omask_x, args.omask_y, "
         "args.omask_z, args.omask_w);\n";
    c += MultiplyMask(is_i4o4);
    c += "  }\n";
  }
  if (weights_shape.i % 4 != 0) {
    c += "  if (i_slice == args.i_slices - 1) {\n";
    c += "    Type mask = ucl::Init<Type>(args.imask_x, args.imask_y, "
         "args.imask_z, args.imask_w);\n";
    c += MultiplyMask(!is_i4o4);
    c += "  }\n";
  }
  return c;
}

std::string Dequantize(int zero_point, bool grouped_quantization,
                       bool batched_quantization, bool has_zero_point,
                       bool is_i4o4) {
  std::string c;
  c += "  if (o_slice < args.o_slices && i_slice < args.i_slices) {\n";
  std::string coords = "o_slice";
  if (batched_quantization) {
    coords = "o_slice, H, 0";
  }
  if (grouped_quantization) {
    c += "    int scale_zp_group = i_slice / args.scale_zp_group_size;\n";
    if (batched_quantization) {
      coords = "o_slice, H, scale_zp_group";
    } else {
      coords = "o_slice, 0, scale_zp_group";
    }
  }
  c += "    Type weight_scale = ucl::Convert<Type>(args.weights_scale.Read(" +
       coords + "));\n";
  if (has_zero_point) {
    c += "    Type weight_zp = "
         "ucl::Convert<Type>(args.weights_zero_point.Read(" +
         coords + "));\n";
  } else {
    c += "    Type weight_zp = ucl::Init<Type>(0.0f);\n";
  }
  const std::string zp_str =
      "ucl::Init<Type>(" + std::to_string(zero_point) + ".0f)";
  c += "  Type weight_bias = -weight_scale * (" + zp_str + " + weight_zp);\n";
  if (is_i4o4) {
    c += "  w0 = w0 * weight_scale + weight_bias;\n";
    c += "  w1 = w1 * weight_scale + weight_bias;\n";
    c += "  w2 = w2 * weight_scale + weight_bias;\n";
    c += "  w3 = w3 * weight_scale + weight_bias;\n";
  } else {
    c += "  w0 = w0 * weight_scale.x + weight_bias.x;\n";
    c += "  w1 = w1 * weight_scale.y + weight_bias.y;\n";
    c += "  w2 = w2 * weight_scale.z + weight_bias.z;\n";
    c += "  w3 = w3 * weight_scale.w + weight_bias.w;\n";
  }
  c += "  }  // i/o slice bounds\n";
  return c;
}

std::string GetLinearIndex() {
  std::string c;
  c += "  int group_id = ucl::GetGroupId<1>() * args.groups_per_x + "
       "ucl::GetGroupId<0>();\n";
  c += "  int linear_index = group_id * ucl::GetGroupSize<0>() + "
       "ucl::GetLocalId<0>();\n";
  return c;
}

int GetGroupsCount(const WeightsDescription& weights_desc,
                   const OHWI& weights_shape, int work_group_size) {
  if (weights_desc.IsLinearLayout()) {
    int threads_count = 1;
    if (weights_desc.layout == WeightsLayout::kISpatialOI4O4UnalignedIO) {
      threads_count = DivideRoundUp(weights_shape.DimensionsProduct(), 4);
    } else {
      const int elements_count =
          GetTotalElementsCountForLayout(weights_desc, weights_shape);
      threads_count = DivideRoundUp(elements_count, 16);
    }
    return DivideRoundUp(threads_count, work_group_size);
  }
  return 0;
}

int3 GetGrid(const WeightsDescription& dst_weights_desc,
             const OHWI& weights_shape, const int3& work_group_size,
             int groups_per_x) {
  if (dst_weights_desc.layout ==
          WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4 ||
      dst_weights_desc.layout ==
          WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4) {
    int2 tex_size = Get2dResourceSize(dst_weights_desc, weights_shape);
    return int3(tex_size.x, tex_size.y, 1);
  } else if (dst_weights_desc.layout ==
             WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    int2 tex_size = Get2dResourceSize(dst_weights_desc, weights_shape);
    if (dst_weights_desc.type == DataType::UINT8 ||
        dst_weights_desc.type == DataType::UINT4 ||
        dst_weights_desc.type == DataType::UINT2) {
      tex_size.x /= 4;  // storing 16 elements in one 2d texture element
    } else {
      // TODO: b/378522761 - Support other data types for
      // k2DYIsSpatialIOAndXIsOGroupI4O4 layout.
      ABSL_LOG(FATAL) << absl::StrCat(
          "Weights conversion to k2DYIsSpatialIOAndXIsOGroupI4O4 layout with "
          "data "
          "type ",
          ToString(dst_weights_desc.type), " is unsupported.");
    }
    return int3(tex_size.x, tex_size.y, 1);
  } else {
    const int groups_count =
        GetGroupsCount(dst_weights_desc, weights_shape, work_group_size.x);
    return int3(groups_per_x * work_group_size.x,
                DivideRoundUp(groups_count, groups_per_x), 1);
  }
}

// Find b that is the smallest power of 2 less than or equal to a.
uint64_t BitFloor(uint64_t a) {
  if (a == 0) return 0;

  uint64_t b = 1;
  // Find the highest bit
  uint64_t temp = a;
  while (temp >>= 1) {
    b <<= 1;
  }
  return b;
}

int GetGroupsPerX(const GpuInfo& gpu_info,
                  const WeightsDescription& weights_desc,
                  const OHWI& weights_shape, const int3& work_group_size) {
  const uint64_t groups_count =
      GetGroupsCount(weights_desc, weights_shape, work_group_size.x);
  uint64_t groups_per_x = groups_count;
  if (groups_per_x > gpu_info.GetMaxWorkGroupsCountForX()) {
    groups_per_x = BitFloor(gpu_info.GetMaxWorkGroupsCountForX());
  }
  return groups_per_x;
}
}  // namespace

ConverterToConvWeights::ConverterToConvWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const OHWI& weights_shape, const WeightsDescription& weights_desc,
    Layout input_layout, const TensorDescriptor* weights_scale,
    const TensorDescriptor* weights_zero_point)
    : weights_shape_(weights_shape),
      weights_desc_(weights_desc),
      input_layout_(input_layout) {
  args_.AddInt("wshape_o", weights_shape_.o);
  args_.AddInt("wshape_h", weights_shape_.h);
  args_.AddInt("wshape_w", weights_shape_.w);
  args_.AddInt("wshape_i", weights_shape_.i);
  args_.AddInt("wshape_spatial", weights_shape_.w * weights_shape_.h);
  work_group_size_ =
      int3(std::min(128, gpu_info.GetMaxWorkGroupSizeForX()), 1, 1);
  if (weights_desc_.IsLinearLayout()) {
    groups_per_x_ = GetGroupsPerX(gpu_info, weights_desc_, weights_shape_,
                                  work_group_size_);
    args_.AddInt("groups_per_x", groups_per_x_);
  }
  if (weights_desc.layout == WeightsLayout::kISpatialOI4O4UnalignedIO) {
    args_.AddInt("wshape_total", weights_shape_.DimensionsProduct());
    ABSL_CHECK_EQ(weights_scale, nullptr)
        << "Scale is not supported for unaligned IO.";
    ABSL_CHECK_EQ(weights_zero_point, nullptr)
        << "Zero point is not supported for unaligned IO.";
    code_ = GetCodeUnalignedIO(definition);
  } else {
    code_ = GetConverterToConvWeightsCode(gpu_info, definition, weights_scale,
                                          weights_zero_point);
  }
}

// requires this args:
//   args.dst_x_size, args.dst_y_size, args.dst_total_size
//   args.dst_o_group_size
//   args.wshape_spatial, args.wshape_w, args.wshape_h
//   args.i_slices, args.o_slices
// calculates this coords:
//   i_slice, o_slice, spatial_linear
// Return if the thread is not supposed to write values.
std::string GetWeightsCoords(const WeightsDescription& dst_weights_desc,
                             bool use_2d_grid_x_is_i_ogroup_y_is_o = false) {
  std::string c;
  if (use_2d_grid_x_is_i_ogroup_y_is_o) {
    c += "  if (ucl::GetGlobalId<2>() != 0) return;\n";
    c += "  int dst_i_ogroup = ucl::GetGlobalId<0>();\n";
    c += "  int dst_o = ucl::GetGlobalId<1>();\n";
    c += "  int linear_index = dst_o * args.i_slices * args.dst_o_group_size + "
         "dst_i_ogroup;\n";
    c += "  int dst_ogroup = dst_i_ogroup % args.dst_o_group_size;\n";
    c += "  int dst_i = dst_i_ogroup / args.dst_o_group_size;\n";
    c += "  if (dst_i >= args.i_slices) return;\n";
    c += "  if (dst_o * args.dst_o_group_size >= args.o_slices) return;\n";
    c += "  int i_slice = dst_i;\n";
    c += "  int o_slice = dst_o * args.dst_o_group_size + dst_ogroup;\n";
    c += "  int spatial_linear = 0;\n";
  } else if (dst_weights_desc.layout ==
                 WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4 ||
             dst_weights_desc.layout ==
                 WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4) {
    c += "  int dst_o_ogroup = ucl::GetGlobalId<0>();\n";
    c += "  if (dst_o_ogroup >= args.dst_x_size) return;\n";
    c += "  int dst_sp_i = ucl::GetGlobalId<1>();\n";
    c += "  if (dst_sp_i >= args.dst_y_size) return;\n";
    c += "  if (ucl::GetGlobalId<2>() != 0) return;\n";

    c += "  int dst_ogroup = dst_o_ogroup % args.dst_o_group_size;\n";
    c += "  int dst_o = dst_o_ogroup / args.dst_o_group_size;\n";
    c += "  int dst_i = dst_sp_i % args.i_slices;\n";
    c += "  int dst_sp = dst_sp_i / args.i_slices;\n";
    c += "  int i_slice = dst_i;\n";
    c += "  int o_slice = dst_o * args.dst_o_group_size + dst_ogroup;\n";
    c += "  int spatial_linear = dst_sp;\n";
  } else if (dst_weights_desc.layout ==
             WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    c += "  int dst_ogroup = ucl::GetGlobalId<0>();\n";
    c += "  if (dst_ogroup >= args.dst_x_size) return;\n";
    c += "  int dst_sp_i_o = ucl::GetGlobalId<1>();\n";
    c += "  if (dst_sp_i_o >= args.dst_y_size) return;\n";
    c += "  if (ucl::GetGlobalId<2>() != 0) return;\n";
    c += "  int o_groups = (args.o_slices + args.dst_o_group_size - 1) / "
         "args.dst_o_group_size;\n";
    c += "  int dst_o = dst_sp_i_o % o_groups;\n";
    c += "  int dst_sp_i = dst_sp_i_o / o_groups;\n";
    c += "  int dst_i = dst_sp_i % args.i_slices;\n";
    c += "  int dst_sp = dst_sp_i / args.i_slices;\n";

    c += "  int i_slice = dst_i;\n";
    c += "  int o_slice = dst_o * args.dst_o_group_size + dst_ogroup;\n";
    c += "  int spatial_linear = dst_sp;\n";

    // for writing into 2d texture
    c += "  int out_coord_2d_x = dst_ogroup;\n";
    c += "  int out_coord_2d_y = dst_sp_i_o;\n";
  } else {
    c += GetLinearIndex();
    c += "  if (linear_index >= args.dst_total_size) return;\n";
    c += "  if (ucl::GetGlobalId<2>() != 0) return;\n";

    if (dst_weights_desc.layout == WeightsLayout::kOSpatialIOGroupI4O4 ||
        dst_weights_desc.layout == WeightsLayout::kOSpatialIOGroupO4I4) {
      c += "  int dst_o_sp_i_ogroup = linear_index;\n";
      c += "  int dst_ogroup = dst_o_sp_i_ogroup % args.dst_o_group_size;\n";
      c += "  int dst_o_sp_i = dst_o_sp_i_ogroup / args.dst_o_group_size;\n";
      c += "  int dst_i = dst_o_sp_i % args.i_slices;\n";
      c += "  int dst_o_sp = dst_o_sp_i / args.i_slices;\n";
      c += "  int dst_sp = dst_o_sp % args.wshape_spatial;\n";
      c += "  int dst_o = dst_o_sp / args.wshape_spatial;\n";
      c += "  int i_slice = dst_i;\n";
      c += "  int o_slice = dst_o * args.dst_o_group_size + dst_ogroup;\n";
      c += "  int spatial_linear = dst_sp;\n";
    } else if (dst_weights_desc.layout == WeightsLayout::kOISpatialOGroupI4O4 ||
               dst_weights_desc.layout == WeightsLayout::kOISpatialOGroupO4I4) {
      c += "  int dst_o_i_sp_ogroup = linear_index;\n";
      c += "  int dst_ogroup = dst_o_i_sp_ogroup % args.dst_o_group_size;\n";
      c += "  int dst_o_i_sp = dst_o_i_sp_ogroup / args.dst_o_group_size;\n";
      c += "  int dst_sp = dst_o_i_sp % args.wshape_spatial;\n";
      c += "  int dst_o_i = dst_o_i_sp / args.wshape_spatial;\n";
      c += "  int dst_i = dst_o_i % args.i_slices;\n";
      c += "  int dst_o = dst_o_i / args.i_slices;\n";
      c += "  int i_slice = dst_i;\n";
      c += "  int o_slice = dst_o * args.dst_o_group_size + dst_ogroup;\n";
      c += "  int spatial_linear = dst_sp;\n";
    } else if (dst_weights_desc.layout == WeightsLayout::kOICustomSpatialI4O4 ||
               dst_weights_desc.layout == WeightsLayout::kOICustomSpatialO4I4) {
      c += "  int dst_o_i_csp = linear_index;\n";
      c += "  int dst_csp = dst_o_i_csp % args.wshape_spatial;\n";
      c += "  int dst_o_i = dst_o_i_csp / args.wshape_spatial;\n";
      c += "  int dst_i = dst_o_i % args.i_slices;\n";
      c += "  int dst_o = dst_o_i / args.i_slices;\n";
      c += "  int i_slice = dst_i;\n";
      c += "  int o_slice = dst_o;\n";
      c += "  int spatial_linear = dst_csp;\n";
    } else if (dst_weights_desc.layout == WeightsLayout::kCustomGroups) {
      c += "  int i_slice = 0;\n";
      c += "  int o_slice = 0;\n";
      c += "  int spatial_w = 0;\n";
      c += "  int spatial_h = 0;\n";
      const absl::flat_hash_map<Axis, std::string> coord_names = {
          {Axis::OUTPUT_CHANNELS, "o_slice"},
          {Axis::HEIGHT, "spatial_h"},
          {Axis::WIDTH, "spatial_w"},
          {Axis::INPUT_CHANNELS, "i_slice"},
      };
      const absl::flat_hash_map<Axis, std::string> shape_names = {
          {Axis::OUTPUT_CHANNELS, "args.wshape_o"},
          {Axis::HEIGHT, "args.wshape_h"},
          {Axis::WIDTH, "args.wshape_w"},
          {Axis::INPUT_CHANNELS, "args.wshape_i"},
      };
      c += "  int last_group_id = linear_index / (args.last_i_group_slices * "
           "args.last_o_group_slices);\n";
      c += "  int last_local_id = linear_index % (args.last_i_group_slices * "
           "args.last_o_group_slices);\n";
      c += "  int tmp_coord = last_group_id;\n";
      c += "  int tmp_divider = 1;\n";
      OHWI prev_sizes = OHWI(1, 1, 1, 1);
      const int start_group = GetStartGroup(dst_weights_desc);
      for (int group = start_group; group < dst_weights_desc.group_sizes.size();
           ++group) {
        const auto& [axis, group_size] = dst_weights_desc.group_sizes[group];
        const int prev_size = prev_sizes.get(axis);
        const std::string& src_coord_name = coord_names.at(axis);
        std::string group_size_str = std::to_string(group_size);
        if (group_size <= 0) {
          std::string size = std::to_string(prev_size);
          if (axis == Axis::OUTPUT_CHANNELS) {
            size += " * args.last_o_group_slices * 4";
          } else if (axis == Axis::INPUT_CHANNELS) {
            size += " * args.last_i_group_slices * 4";
          }
          group_size_str = "(" + shape_names.at(axis) + " + (" + size +
                           ") - 1) / (" + size + ")";
        }
        c += "  tmp_divider = " + group_size_str + ";\n";
        c += "  " + src_coord_name + " = (tmp_coord % tmp_divider) * " +
             std::to_string(prev_size) + " + " + src_coord_name + ";\n";
        c += "  tmp_coord = tmp_coord / tmp_divider;\n";
        prev_sizes.set(axis, group_size * prev_size);
      }
      if (dst_weights_desc.group_sizes[0].first == Axis::INPUT_CHANNELS) {
        c += "  i_slice = i_slice * args.last_i_group_slices + last_local_id % "
             "args.last_i_group_slices;\n";
        c += "  o_slice = o_slice * args.last_o_group_slices + last_local_id / "
             "args.last_i_group_slices;\n";
      } else {
        // dst_weights_desc.group_sizes[0].first == Axis::OUTPUT_CHANNELS
        c += "  o_slice = o_slice * args.last_o_group_slices + last_local_id % "
             "args.last_o_group_slices;\n";
        c += "  i_slice = i_slice * args.last_i_group_slices + last_local_id / "
             "args.last_o_group_slices;\n";
      }
      c += "  int spatial_linear = spatial_h * args.wshape_w + spatial_w;\n";
    }
  }
  if (!dst_weights_desc.spatial_remap.empty()) {
    c += "  int linear_remap = args.spatial_remap.Read(spatial_linear);\n";
    c += "  int W = linear_remap % args.wshape_w;\n";
    c += "  int H = linear_remap / args.wshape_w;\n";
  } else {
    c += "  int W = spatial_linear % args.wshape_w;\n";
    c += "  int H = spatial_linear / args.wshape_w;\n";
  }
  return c;
}

enum BlockLayout {
  kI4O4,
  kO4I4,
};

std::string Get8BitPacked() {
  return R"(
  uint4 packed_value;
  uint4 t = ucl::Convert<uint4>(ucl::Convert<int4>(r0));
  packed_value.x = ((t.w & 255u) << 24u) | ((t.z & 255u) << 16u) | ((t.y & 255u) << 8u) | (t.x & 255u);
  t = ucl::Convert<uint4>(ucl::Convert<int4>(r1));
  packed_value.y = ((t.w & 255u) << 24u) | ((t.z & 255u) << 16u) | ((t.y & 255u) << 8u) | (t.x & 255u);
  t = ucl::Convert<uint4>(ucl::Convert<int4>(r2));
  packed_value.z = ((t.w & 255u) << 24u) | ((t.z & 255u) << 16u) | ((t.y & 255u) << 8u) | (t.x & 255u);
  t = ucl::Convert<uint4>(ucl::Convert<int4>(r3));
  packed_value.w = ((t.w & 255u) << 24u) | ((t.z & 255u) << 16u) | ((t.y & 255u) << 8u) | (t.x & 255u);
)";
}

std::string Get4BitPacked() {
  return R"(
  uint4 t0 = ucl::Convert<uint4>(ucl::Convert<int4>(r0));
  uint4 t1 = ucl::Convert<uint4>(ucl::Convert<int4>(r1));
  uint4 t2 = ucl::Convert<uint4>(ucl::Convert<int4>(r2));
  uint4 t3 = ucl::Convert<uint4>(ucl::Convert<int4>(r3));
  uint p0 = ((t0.w & 15u) << 12u) | ((t0.z & 15u) << 8u) | ((t0.y & 15u) << 4u) | (t0.x & 15u);
  uint p1 = ((t1.w & 15u) << 12u) | ((t1.z & 15u) << 8u) | ((t1.y & 15u) << 4u) | (t1.x & 15u);
  uint p2 = ((t2.w & 15u) << 12u) | ((t2.z & 15u) << 8u) | ((t2.y & 15u) << 4u) | (t2.x & 15u);
  uint p3 = ((t3.w & 15u) << 12u) | ((t3.z & 15u) << 8u) | ((t3.y & 15u) << 4u) | (t3.x & 15u);
)";
}

std::string Get2BitPacked() {
  return R"(
  uint4 t0 = ucl::Convert<uint4>(ucl::Convert<int4>(r0));
  uint4 t1 = ucl::Convert<uint4>(ucl::Convert<int4>(r1));
  uint4 t2 = ucl::Convert<uint4>(ucl::Convert<int4>(r2));
  uint4 t3 = ucl::Convert<uint4>(ucl::Convert<int4>(r3));
  uint p0 = ((t0.w & 3u) << 6u) | ((t0.z & 3u) << 4u) | ((t0.y & 3u) << 2u) | (t0.x & 3u);
  uint p1 = ((t1.w & 3u) << 6u) | ((t1.z & 3u) << 4u) | ((t1.y & 3u) << 2u) | (t1.x & 3u);
  uint p2 = ((t2.w & 3u) << 6u) | ((t2.z & 3u) << 4u) | ((t2.y & 3u) << 2u) | (t2.x & 3u);
  uint p3 = ((t3.w & 3u) << 6u) | ((t3.z & 3u) << 4u) | ((t3.y & 3u) << 2u) | (t3.x & 3u);
  )";
}

std::string WriteResults(const WeightsDescription& dst_weights_desc,
                         const BlockLayout& src_layout,
                         const GpuInfo& gpu_info) {
  const bool dst_is_i2o4i2 =
      dst_weights_desc.layout == WeightsLayout::kCustomGroups &&
      IsI2O4I2(dst_weights_desc);
  const bool dst_is_o2i2o2i2 =
      dst_weights_desc.layout == WeightsLayout::kCustomGroups &&
      IsO2I2O2I2(dst_weights_desc);
  const bool dst_is_o2i4o2 =
      dst_weights_desc.layout == WeightsLayout::kCustomGroups &&
      IsO2I4O2(dst_weights_desc);
  std::string c;
  if (dst_is_i2o4i2) {
    if (src_layout == BlockLayout::kO4I4) {
      // 04I4 -> I2O4I2
      c += "  Type r0 = ucl::Init<Type>(w0.x, w0.y, w1.x, w1.y);\n";
      c += "  Type r1 = ucl::Init<Type>(w2.x, w2.y, w3.x, w3.y);\n";
      c += "  Type r2 = ucl::Init<Type>(w0.z, w0.w, w1.z, w1.w);\n";
      c += "  Type r3 = ucl::Init<Type>(w2.z, w2.w, w3.z, w3.w);\n";
    } else {
      // I4O4 -> I2O4I2
      c += "  Type r0 = ucl::Init<Type>(w0.x, w1.x, w0.y, w1.y);\n";
      c += "  Type r1 = ucl::Init<Type>(w0.z, w1.z, w0.w, w1.w);\n";
      c += "  Type r2 = ucl::Init<Type>(w2.x, w3.x, w2.y, w3.y);\n";
      c += "  Type r3 = ucl::Init<Type>(w2.z, w3.z, w2.w, w3.w);\n";
    }
  } else if (dst_is_o2i2o2i2) {
    if (src_layout == BlockLayout::kO4I4) {
      // 04I4 -> O2I2O2I2
      c += "  Type r0 = ucl::Init<Type>(w0.x, w0.y, w1.x, w1.y);\n";
      c += "  Type r1 = ucl::Init<Type>(w0.z, w0.w, w1.z, w1.w);\n";
      c += "  Type r2 = ucl::Init<Type>(w2.x, w2.y, w3.x, w3.y);\n";
      c += "  Type r3 = ucl::Init<Type>(w2.z, w2.w, w3.z, w3.w);\n";
    } else {
      // I4O4 -> O2I2O2I2
      c += "  Type r0 = ucl::Init<Type>(w0.x, w1.x, w0.y, w1.y);\n";
      c += "  Type r1 = ucl::Init<Type>(w2.x, w3.x, w2.y, w3.y);\n";
      c += "  Type r2 = ucl::Init<Type>(w0.z, w1.z, w0.w, w1.w);\n";
      c += "  Type r3 = ucl::Init<Type>(w2.z, w3.z, w2.w, w3.w);\n";
    }
  } else if (dst_is_o2i4o2) {
    if (src_layout == BlockLayout::kO4I4) {
      // 04I4 -> O2I4O2
      c += "  Type r0 = ucl::Init<Type>(w0.x, w1.x, w0.y, w1.y);\n";
      c += "  Type r1 = ucl::Init<Type>(w0.z, w1.z, w0.w, w1.w);\n";
      c += "  Type r2 = ucl::Init<Type>(w2.x, w3.x, w2.y, w3.y);\n";
      c += "  Type r3 = ucl::Init<Type>(w2.z, w3.z, w2.w, w3.w);\n";
    } else {
      // I4O4 -> O2I4O2
      c += "  Type r0 = ucl::Init<Type>(w0.x, w0.y, w1.x, w1.y);\n";
      c += "  Type r1 = ucl::Init<Type>(w2.x, w2.y, w3.x, w3.y);\n";
      c += "  Type r2 = ucl::Init<Type>(w0.z, w0.w, w1.z, w1.w);\n";
      c += "  Type r3 = ucl::Init<Type>(w2.z, w2.w, w3.z, w3.w);\n";
    }
  } else {
    const bool dst_is_o4 =
        dst_weights_desc.IsI4O4() ||
        (dst_weights_desc.layout == WeightsLayout::kCustomGroups &&
         dst_weights_desc.group_sizes[0].first == Axis::OUTPUT_CHANNELS);
    const bool need_transpose =
        (src_layout == BlockLayout::kI4O4 && !dst_is_o4) ||
        (src_layout == BlockLayout::kO4I4 && dst_is_o4);
    if (need_transpose) {
      if (gpu_info.IsApiOpenGl() && gpu_info.IsNvidia() &&
          !dst_weights_desc.IsLinearLayout()) {
        // some nvidia OpenGL related bug?
        c += "  Type r0, r1, r2, r3;\n";
        // dummy check, W is always >= 0;
        c += "  if (W == -1) {\n";
        c += "    r0 = w0;\n";
        c += "    r1 = w1;\n";
        c += "    r2 = w2;\n";
        c += "    r3 = w3;\n";
        c += "  } else {\n";
        c += "    r0.x = w0.x;\n";
        c += "    r0.y = w1.x;\n";
        c += "    r0.z = w2.x;\n";
        c += "    r0.w = w3.x;\n";
        c += "    r1.x = w0.y;\n";
        c += "    r1.y = w1.y;\n";
        c += "    r1.z = w2.y;\n";
        c += "    r1.w = w3.y;\n";
        c += "    r2.x = w0.z;\n";
        c += "    r2.y = w1.z;\n";
        c += "    r2.z = w2.z;\n";
        c += "    r2.w = w3.z;\n";
        c += "    r3.x = w0.w;\n";
        c += "    r3.y = w1.w;\n";
        c += "    r3.z = w2.w;\n";
        c += "    r3.w = w3.w;\n";
        c += "  }\n";
      } else {
        c += "  Type r0 = ucl::Init<Type>(w0.x, w1.x, w2.x, w3.x);\n";
        c += "  Type r1 = ucl::Init<Type>(w0.y, w1.y, w2.y, w3.y);\n";
        c += "  Type r2 = ucl::Init<Type>(w0.z, w1.z, w2.z, w3.z);\n";
        c += "  Type r3 = ucl::Init<Type>(w0.w, w1.w, w2.w, w3.w);\n";
      }
    } else {
      c += "  Type r0 = w0;\n";
      c += "  Type r1 = w1;\n";
      c += "  Type r2 = w2;\n";
      c += "  Type r3 = w3;\n";
    }
  }
  if (dst_weights_desc.layout ==
          WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4 ||
      dst_weights_desc.layout ==
          WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4) {
    c += "  args.dst_tensor0.Write2D(r0, dst_o_ogroup, dst_sp_i);\n";
    c += "  args.dst_tensor1.Write2D(r1, dst_o_ogroup, dst_sp_i);\n";
    c += "  args.dst_tensor2.Write2D(r2, dst_o_ogroup, dst_sp_i);\n";
    c += "  args.dst_tensor3.Write2D(r3, dst_o_ogroup, dst_sp_i);\n";
  } else if (dst_weights_desc.layout == WeightsLayout::kCustomGroups) {
    const bool is_i_first =
        dst_weights_desc.group_sizes[0].first == Axis::INPUT_CHANNELS;
    const std::string tile_size_x =
        is_i_first ? "args.last_i_group_slices" : "args.last_o_group_slices";
    c += "  int dst_offset = last_group_id * (args.last_i_group_slices * "
         "args.last_o_group_slices * 4) + last_local_id / " +
         tile_size_x + " * (" + tile_size_x + " * 4)" + " + last_local_id % " +
         tile_size_x + ";\n";
    if (dst_weights_desc.type == DataType::INT8 ||
        dst_weights_desc.type == DataType::UINT8) {
      c += Get8BitPacked();
      if (IsLastBlockHas4IAnd4OElements(dst_weights_desc)) {
        c += "  args.dst_buffer.Write(packed_value, dst_offset / 4);\n";
      } else {
        c += "  args.dst_buffer.Write(packed_value.x, dst_offset + 0);\n";
        c += "  args.dst_buffer.Write(packed_value.y, dst_offset + " +
             tile_size_x + " * 1);\n";
        c += "  args.dst_buffer.Write(packed_value.z, dst_offset + " +
             tile_size_x + " * 2);\n";
        c += "  args.dst_buffer.Write(packed_value.w, dst_offset + " +
             tile_size_x + " * 3);\n";
      }
    } else if (dst_weights_desc.type == DataType::INT4 ||
               dst_weights_desc.type == DataType::UINT4) {
      c += Get4BitPacked();
      c += "  uint2 value;\n";
      c += "  value.x = (p1 << 16u) | p0;\n";
      c += "  value.y = (p3 << 16u) | p2;\n";
      c += "  args.dst_buffer.Write(value, dst_offset / 4);\n";
    } else if (dst_weights_desc.type == DataType::INT2 ||
               dst_weights_desc.type == DataType::UINT2) {
      c += Get2BitPacked();
      c += "  uint value = (p3 << 24u) | (p2 << 16u) | (p1 << 8u) | p0;\n";
      c += "  args.dst_buffer.Write(value, dst_offset / 4);\n";
    } else {
      c += "  args.dst_tensor.WriteLinear(r0, dst_offset + 0);\n";
      c += "  args.dst_tensor.WriteLinear(r1, dst_offset + " + tile_size_x +
           " * 1);\n";
      c += "  args.dst_tensor.WriteLinear(r2, dst_offset + " + tile_size_x +
           " * 2);\n";
      c += "  args.dst_tensor.WriteLinear(r3, dst_offset + " + tile_size_x +
           " * 3);\n";
    }
  } else {
    if (dst_weights_desc.type == DataType::INT8 ||
        dst_weights_desc.type == DataType::UINT8) {
      c += Get8BitPacked();
      if (dst_weights_desc.layout ==
          WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
        c += "  args.dst_texture.Write2D(packed_value, out_coord_2d_x, "
             "out_coord_2d_y);\n";
      } else {
        c += "  args.dst_buffer.Write(packed_value, linear_index);\n";
      }
    } else if (dst_weights_desc.type == DataType::INT4 ||
               dst_weights_desc.type == DataType::UINT4) {
      c += Get4BitPacked();
      if (dst_weights_desc.layout ==
          WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
        // TODO: Support DST weights with k2DYIsIOAndXIsOGroupI4O4 layout and
        // INT4 type.
        if (dst_weights_desc.type == DataType::INT4) {
          ABSL_LOG(FATAL)
              << "Weights conversion to k2DYIsSpatialIOAndXIsOGroupI4O4 "
                 "layout with data type INT4 is unsupported.";
        }
        c += "  ushort4 value = ucl::Convert<ushort4>(ucl::Init<uint4>(p0, p1, "
             "p2, p3));\n";
        c += "  args.dst_texture.Write2D(value, out_coord_2d_x, "
             "out_coord_2d_y);\n";
      } else {
        c += "  uint2 value;\n";
        c += "  value.x = (p1 << 16u) | p0;\n";
        c += "  value.y = (p3 << 16u) | p2;\n";
        c += "  args.dst_buffer.Write(value, linear_index);\n";
      }
    } else if (dst_weights_desc.type == DataType::INT2 ||
               dst_weights_desc.type == DataType::UINT2) {
      c += Get2BitPacked();
      if (dst_weights_desc.layout ==
          WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
        c += "  uchar4 value = ucl::Convert<uchar4>(ucl::Init<uint4>(p0, p1, "
             "p2, p3));\n";
        c += "  args.dst_texture.Write2D(value, out_coord_2d_x, "
             "out_coord_2d_y);\n";
      } else {
        c += "  uint value = (p3 << 24u) | (p2 << 16u) | (p1 << 8u) | p0;\n";
        c += "  args.dst_buffer.Write(value, linear_index);\n";
      }
    } else {
      c += "  args.dst_tensor.WriteLinear(r0, linear_index * 4 + 0);\n";
      c += "  args.dst_tensor.WriteLinear(r1, linear_index * 4 + 1);\n";
      c += "  args.dst_tensor.WriteLinear(r2, linear_index * 4 + 2);\n";
      c += "  args.dst_tensor.WriteLinear(r3, linear_index * 4 + 3);\n";
    }
  }
  c += "}\n";
  return c;
}

std::string ParseLinearIndex() {
  return R"(
  {int ind = index_in;
  if (ind >= src_x4 * args.wshape_o * args.wshape_spatial * 4) {
    ind = ind - src_x4 * args.wshape_o * args.wshape_spatial * 4;
    spatial_linear = ind / (args.wshape_o * src_last_block_size);
    ind = ind % (args.wshape_o * src_last_block_size);
    if (ind >= dst_x4 * 4 * src_last_block_size) {
      ind = ind - dst_x4 * 4 * src_last_block_size;
      int i0 = ind % dst_last_block_size;
      int j0 = ind / dst_last_block_size;
      O = dst_x4 * 4 + i0;
      I = src_x4 * 4 + j0;
    } else {
      int block_size = src_last_block_size * 4;
      int dst_s = ind / block_size;
      int block_id = ind % block_size;
      int i0 = block_id % 4;
      int j0 = block_id / 4;
      O = dst_s * 4 + i0;
      I = src_x4 * 4 + j0;
    }
  } else {
    int src_s = ind / (args.wshape_o * args.wshape_spatial * 4);
    ind = ind % (args.wshape_o * args.wshape_spatial * 4);
    spatial_linear = ind / (args.wshape_o * 4);
    ind = ind % (args.wshape_o * 4);
    if (ind >= dst_x4 * 16) {
      ind = ind - dst_x4 * 16;
      int i0 = ind % dst_last_block_size;
      int j0 = ind / dst_last_block_size;
      O = dst_x4 * 4 + i0;
      I = src_s * 4 + j0;
    } else {
      int dst_s = ind / 16;
      int block_id = ind % 16;
      int i0 = block_id % 4;
      int j0 = block_id / 4;
      O = dst_s * 4 + i0;
      I = src_s * 4 + j0;
    }
  }}
)";
}

std::string ConverterToConvWeights::GetCodeUnalignedIO(
    const OperationDef& definition) {
  AddSrcTensor("src", definition.src_tensors[0]);
  AddDstTensor("dst", definition.dst_tensors[0]);
  args_.AddInt("task_size",
               DivideRoundUp(weights_shape_.DimensionsProduct(), 4));

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  c += "  if (ucl::GetGlobalId<2>() != 0) return;\n";
  c += GetLinearIndex();
  c += "  if (linear_index >= args.task_size) return;\n";
  c += "  int src_x4 = args.wshape_i / 4;\n";
  c += "  int dst_x4 = args.wshape_o / 4;\n";
  c += "  int src_last_block_size = args.wshape_i - src_x4 * 4;\n";
  c += "  int dst_last_block_size = args.wshape_o - dst_x4 * 4;\n";
  c += "  int spatial_linear = 0;\n";
  c += "  int I = 0;\n";
  c += "  int O = 0;\n";
  c += "  int index_in = 0;\n";
  c += "  SType temps[4];\n";
  c += "  temps[0] = ucl::Init<SType>(0);\n";
  c += "  temps[1] = ucl::Init<SType>(0);\n";
  c += "  temps[2] = ucl::Init<SType>(0);\n";
  c += "  temps[3] = ucl::Init<SType>(0);\n";
  c += "  for (int i = 0; i < 4; ++i) {\n";
  c += "    int index_in = linear_index * 4 + i;\n";
  c += "    if (linear_index < args.wshape_total) {\n";
  c += ParseLinearIndex();
  c += "      int W = spatial_linear % args.wshape_w;\n";
  c += "      int H = spatial_linear / args.wshape_w;\n";
  if (input_layout_ == Layout::OHWI) {
    if (definition.src_tensors[0].GetLayout() == Layout::LINEAR) {
      c += "      int linear_ohwi = ((O * args.wshape_h + H) * args.wshape_w + "
           "W) * args.wshape_i + I;\n";
      c += "      args.src.ReadPerChannel<SType>(temps[i], linear_ohwi);\n";
    } else {
      c += "      args.src.ReadPerChannel<SType>(temps[i], W, H, I, O);\n";
    }
  } else if (input_layout_ == Layout::HWIO) {
    c += "      args.src.ReadPerChannel<SType>(temps[i], I, W, O, H);\n";
  }
  c += "    }\n";
  c += "  }\n";
  c += "  Type result;\n";
  c += "  result.x = temps[0];\n";
  c += "  result.y = temps[1];\n";
  c += "  result.z = temps[2];\n";
  c += "  result.w = temps[3];\n";
  c += "  args.dst.WriteLinear(result, linear_index);\n";
  c += "}\n";

  const DataType data_type = definition.dst_tensors[0].GetDataType();
  absl::StrReplaceAll({{"SType", ToUclDataType(data_type, 1)},
                       {"Type", ToUclDataType(data_type, 4)}},
                      &c);
  return c;
}

std::string ConverterToConvWeights::GetConverterToConvWeightsCode(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const TensorDescriptor* weights_scale,
    const TensorDescriptor* weights_zero_point) {
  const DataType src_data_type = definition.src_tensors[0].GetDataType();
  if (src_data_type == DataType::INT4) {
    BufferDescriptor desc;
    desc.element_type = DataType::INT32;
    desc.element_size = 1;
    AddSrcBuffer("src", desc);
  } else if (src_data_type == DataType::INT2) {
    BufferDescriptor desc;
    desc.element_type = DataType::UINT32;
    desc.element_size = 1;
    AddSrcBuffer("src", desc);
  } else {
    AddSrcTensor("src", definition.src_tensors[0]);
  }
  const DataType dst_data_type = definition.dst_tensors[0].GetDataType();
  const bool src_quantized = SizeInBitsOf(src_data_type) <= 8;
  const bool dst_quantized = SizeInBitsOf(weights_desc_.type) <= 8;
  const auto type = dst_quantized ? DataType::FLOAT32 : dst_data_type;
  if (weights_shape_.i % 4 != 0) {
    const float4 i_mask = GetMaskForLastPlane(weights_shape_.i);
    args_.AddFloat("imask_x", i_mask.x, type);
    args_.AddFloat("imask_y", i_mask.y, type);
    args_.AddFloat("imask_z", i_mask.z, type);
    args_.AddFloat("imask_w", i_mask.w, type);
  }
  if (weights_shape_.o % 4 != 0) {
    const float4 o_mask = GetMaskForLastPlane(weights_shape_.o);
    args_.AddFloat("omask_x", o_mask.x, type);
    args_.AddFloat("omask_y", o_mask.y, type);
    args_.AddFloat("omask_z", o_mask.z, type);
    args_.AddFloat("omask_w", o_mask.w, type);
  }
  AddCommonArgs(weights_desc_, weights_shape_, args_);
  if (weights_desc_.layout ==
          WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4 ||
      weights_desc_.layout ==
          WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4) {
    AddDstTensor("dst_tensor0", definition.dst_tensors[0]);
    AddDstTensor("dst_tensor1", definition.dst_tensors[1]);
    AddDstTensor("dst_tensor2", definition.dst_tensors[2]);
    AddDstTensor("dst_tensor3", definition.dst_tensors[3]);
    uint2 tex_size = Get2dResourceSize(weights_desc_, weights_shape_);
    args_.AddInt("dst_x_size", tex_size.x);
    args_.AddInt("dst_y_size", tex_size.y);
  } else if (weights_desc_.layout ==
             WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    DataType texture_type;
    if (weights_desc_.type == DataType::UINT8) {
      texture_type = DataType::UINT32;
    } else if (weights_desc_.type == DataType::UINT4) {
      texture_type = DataType::UINT16;
    } else if (weights_desc_.type == DataType::UINT2) {
      texture_type = DataType::UINT8;
    } else {
      // TODO: b/378522761 - Support other data types for
      // k2DYIsSpatialIOAndXIsOGroupI4O4 layout.
      ABSL_LOG(FATAL) << absl::StrCat(
          "Weights conversion to k2DYIsSpatialIOAndXIsOGroupI4O4 layout with "
          "data "
          "type ",
          ToString(weights_desc_.type), " is unsupported.");
    }
    TensorDescriptor desc;
    uint2 tex_size = Get2dResourceSize(weights_desc_, weights_shape_);
    tex_size.x /= SizeInBitsOf(texture_type) / SizeInBitsOf(weights_desc_.type);
    desc = TensorDescriptor(texture_type, TensorStorageType::TEXTURE_2D,
                            Layout::HW);
    AddDstTensor("dst_texture", desc);
    args_.AddInt("dst_x_size", tex_size.x);
    args_.AddInt("dst_y_size", tex_size.y);
  } else {
    if (weights_desc_.type == DataType::UINT8) {
      BufferDescriptor desc;
      desc.element_type = DataType::UINT32;
      desc.element_size = IsLastBlockHas4IAnd4OElements(weights_desc_) ? 4 : 1;
      AddDstBuffer("dst_buffer", desc);
    } else if (weights_desc_.type == DataType::UINT4) {
      BufferDescriptor desc;
      desc.element_type = DataType::UINT32;
      desc.element_size = 2;
      AddDstBuffer("dst_buffer", desc);
    } else if (weights_desc_.type == DataType::UINT2) {
      BufferDescriptor desc;
      desc.element_type = DataType::UINT32;
      desc.element_size = 1;
      AddDstBuffer("dst_buffer", desc);
    } else {
      AddDstTensor("dst_tensor", definition.dst_tensors[0]);
    }
    int elements_count =
        GetTotalElementsCountForLayout(weights_desc_, weights_shape_);
    args_.AddInt("dst_total_size", DivideRoundUp(elements_count, 16));
  }

  if (!weights_desc_.spatial_remap.empty()) {
    std::vector<int32_t> remap(weights_desc_.spatial_remap.size());
    for (int i = 0; i < remap.size(); ++i) {
      remap[i] = weights_desc_.spatial_remap[i];
    }
    BufferDescriptor desc;
    desc.element_type = DataType::INT32;
    desc.element_size = 1;
    desc.memory_type = MemoryType::GLOBAL;
    desc.size = remap.size() * sizeof(int32_t);
    desc.data.resize(desc.size);
    std::memcpy(desc.data.data(), remap.data(), desc.size);
    args_.AddObject("spatial_remap",
                    std::make_unique<BufferDescriptor>(std::move(desc)));
  }

  bool grouped_quantization = false;
  bool batched_quantization = false;
  if (weights_scale && weights_scale->GetLayout() != Layout::LINEAR) {
    auto scale_shape = weights_scale->GetBHWCShape();
    const int scale_i_groups = DivideRoundUp(scale_shape.c, 4);
    const int batch_size = scale_shape.h;
    if (batch_size > 1) {
      batched_quantization = true;
    }
    if (scale_i_groups > 1) {
      grouped_quantization = true;
      args_.AddInt("scale_zp_group_size",
                   DivideRoundUp(weights_shape_.i / scale_i_groups, 4));
    }
  }
  if (weights_scale) {
    AddSrcTensor("weights_scale", *weights_scale);
  }
  const bool has_zero_point = weights_zero_point != nullptr;
  if (weights_zero_point) {
    AddSrcTensor("weights_zero_point", *weights_zero_point);
  }

  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  c += GetWeightsCoords(weights_desc_);
  // W and H is src coordinates, spatial_linear is dst coordinate
  c += "  Type w0 = ucl::Init<Type>(0);\n";
  c += "  Type w1 = ucl::Init<Type>(0);\n";
  c += "  Type w2 = ucl::Init<Type>(0);\n";
  c += "  Type w3 = ucl::Init<Type>(0);\n";
  c += "  if (o_slice < args.o_slices && i_slice < args.i_slices) {\n";
  if (input_layout_ == Layout::OHWI &&
      definition.src_tensors[0].GetLayout() == Layout::LINEAR) {
    if (weights_shape_.i % 4 == 0) {
      c += R"(
    int linear_ohwi = ((o_slice * 4 * args.wshape_h + H) * args.wshape_w + W) * args.i_slices + i_slice;
    int o_stride = args.wshape_h * args.wshape_w * args.i_slices;
)";
    }
    for (int o = 0; o < 4; ++o) {
      const std::string o_coord = absl::StrCat("o_slice * 4 + ", o);
      c += "  if (" + o_coord + " < args.wshape_o) {\n";
      if (weights_shape_.i % 4 == 0) {
        const std::string var_name = absl::StrCat("w", o);
        const std::string address =
            absl::StrCat("linear_ohwi + o_stride * ", o);
        if (src_data_type == DataType::INT4) {
          c += "      int i16_value = args.src.ReadAsI16(" + address + ");\n";
          c += "      " + var_name + " = ucl::I16ToVec4I4<SType>(i16_value);\n";
        } else if (src_data_type == DataType::INT2) {
          c += "      uint p = args.src.ReadAsU8(" + address + ");\n";
          c += "      " + var_name +
               " = ucl::Convert<Type>(ucl::U8ToVec4I2(p));\n";
        } else {
          c += "      " + var_name + " = args.src.Read<SType>(" + address +
               ");\n";
        }
      } else {
        c += "    int ohw_offset = ((" + o_coord +
             ") * args.wshape_h + H) * args.wshape_w + W;\n";
        c += "    int i_index;\n";
        for (int i = 0; i < 4; ++i) {
          const std::string coords[] = {".x", ".y", ".z", ".w"};
          const std::string var_name = absl::StrCat("w", o, coords[i]);
          c += "    i_index = 4 * i_slice + " + std::to_string(i) + ";\n";
          c += "    if (i_index < args.wshape_i) {\n";
          c += "      int ohwi_index = ohw_offset * args.wshape_i + i_index;\n";
          if (src_data_type == DataType::INT4) {
            c += "      int ival = args.src.Read(ohwi_index / 8);\n";
            c += "      uint sub_index = ucl::Convert<uint>(7 - ohwi_index % "
                 "8);\n";
            c += "      ival = (ival << (sub_index * 4u)) >> 28u;\n";
            c += "      " + var_name + " = ucl::Convert<SType>(ival);\n";
          } else if (src_data_type == DataType::INT2) {
            c += "      uint uval = args.src.Read(ohwi_index / 16);\n";
            c += "      int ival = ucl::Convert<int>(uval);\n";
            c += "      uint sub_index = ucl::Convert<uint>(15 - ohwi_index % "
                 "16);\n";
            c += "      ival = (ival << (sub_index * 2u)) >> 30u;\n";
            c += "      " + var_name + " = ucl::Convert<SType>(ival);\n";
          } else {
            c += "      args.src.ReadPerChannel<SType>(" + var_name +
                 ", ohwi_index);\n";
          }
          c += "    }\n";
        }
      }
      c += "  }\n";
    }
  } else if (input_layout_ == Layout::OHWI) {
    for (int o = 0; o < 4; ++o) {
      const std::string o_coord = absl::StrCat("o_slice * 4 + ", o);
      const std::string var_name = absl::StrCat("w", o);
      c += "  if (" + o_coord + " < args.wshape_o) {\n";
      c += "    " + var_name + " = args.src.Read<SType>(W, H, i_slice, " +
           o_coord + ");\n";
      c += "  }\n";
    }
  } else if (input_layout_ == Layout::HWIO) {
    for (int i = 0; i < 4; ++i) {
      const std::string i_coord = absl::StrCat("i_slice * 4 + ", i);
      const std::string var_name = absl::StrCat("w", i);
      c += "  if (" + i_coord + " < args.wshape_i) {\n";
      c += "    " + var_name + " = args.src.Read<SType>(" + i_coord +
           ", W, o_slice, H);\n";
      c += "  }\n";
    }
  }
  const int src_zero_point = GetZeroPoint(src_data_type);
  const std::string src_zp_str =
      "ucl::Init<Type>(" + std::to_string(src_zero_point) + ".0f)";
  if (src_zero_point != 0 && dst_quantized) {
    c += "  w0 -= " + src_zp_str + ";\n";
    c += "  w1 -= " + src_zp_str + ";\n";
    c += "  w2 -= " + src_zp_str + ";\n";
    c += "  w3 -= " + src_zp_str + ";\n";
  }
  c += "  }  // i/o slice bounds\n";

  const auto src_layout =
      input_layout_ == Layout::HWIO ? BlockLayout::kI4O4 : BlockLayout::kO4I4;

  if (src_quantized && !dst_quantized && weights_scale != nullptr) {
    c += Dequantize(src_zero_point, grouped_quantization, batched_quantization,
                    has_zero_point, src_layout == BlockLayout::kI4O4);
  }

  c += ApplyMask(weights_shape_, src_layout == BlockLayout::kI4O4);

  const int dst_zero_point = GetZeroPoint(weights_desc_.type);
  if (dst_zero_point != 0) {
    const std::string dst_zp_str =
        "ucl::Init<Type>(" + std::to_string(dst_zero_point) + ".0f)";
    c += "  w0 += " + dst_zp_str + ";\n";
    c += "  w1 += " + dst_zp_str + ";\n";
    c += "  w2 += " + dst_zp_str + ";\n";
    c += "  w3 += " + dst_zp_str + ";\n";
  }

  c += WriteResults(weights_desc_, src_layout, gpu_info);
  absl::StrReplaceAll(
      {{"SType", ToUclDataType(type, 1)}, {"Type", ToUclDataType(type, 4)}},
      &c);
  return c;
}

std::vector<int3> ConverterToConvWeights::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info) const {
  if (weights_desc_.IsLinearLayout()) {
    return {work_group_size_};
  } else {
    // In case of non-linear layout, we can use the default work group
    // picking strategy.
    return GetPossibleWorkGroups(tuning_type, gpu_info, kernel_info,
                                 grid_size_);
  }
}

int3 ConverterToConvWeights::GetGridSize() const {
  return GetGrid(weights_desc_, weights_shape_, work_group_size_,
                 groups_per_x_);
}

std::string ReadVec16AsVec4x4(const WeightsDescription& src_weights_desc,
                              const DataType& dst_type,
                              bool grouped_quantization,
                              bool batched_quantization, bool has_zero_point) {
  std::string c;
  c += "  w0 = ucl::Init<Type>(0);\n";
  c += "  w1 = ucl::Init<Type>(0);\n";
  c += "  w2 = ucl::Init<Type>(0);\n";
  c += "  w3 = ucl::Init<Type>(0);\n";
  c += "  if (o_slice < args.o_slices && i_slice < args.i_slices) {\n";
  c += "  int src_i = i_slice;\n";
  c += "  int src_o_group = o_slice % args.src_o_group_size;\n";
  c += "  int src_o = o_slice / args.src_o_group_size;\n";
  // kOSpatialIOGroupI4O4
  c += "  int src_linear = ((src_o * args.wshape_spatial + "
       "spatial_linear) * args.i_slices + src_i) * "
       "args.src_o_group_size + src_o_group;\n";
  std::string yc;
  if (src_weights_desc.layout ==
      WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    yc = "((spatial_linear * args.i_slices + src_i) * args.src_o_groups + "
         "src_o)";
  }
  if (src_weights_desc.type == DataType::UINT8) {
    if (src_weights_desc.layout ==
        WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
      c += "    uint4 w = args.src_texture.Read(src_o_group, " + yc + ");\n";
      c += "    ucl::U32x4ToU8x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
    } else {
      c += "    uint4 w = args.src_buffer.Read(src_linear);\n";
      c += "    ucl::U32x4ToU8x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
    }
  } else if (src_weights_desc.type == DataType::UINT4) {
    if (src_weights_desc.layout ==
        WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
      c += "    ushort4 w = args.src_texture.Read(src_o_group, " + yc + ");\n";
      c += "    ucl::U16x4ToU4x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
    } else {
      c += "    uint2 w = args.src_buffer.Read(src_linear);\n";
      c += "    ucl::U32x2ToU4x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
    }
  } else if (src_weights_desc.type == DataType::UINT2) {
    if (src_weights_desc.layout ==
        WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
      c += "    uchar4 w = args.src_texture.Read(src_o_group, " + yc + ");\n";
      c += "    ucl::U8x4ToU2x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
    } else {
      c += "    uint w = args.src_buffer.Read(src_linear);\n";
      c += "    ucl::U32x1ToU2x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
    }
  } else if (src_weights_desc.type == DataType::INT8) {
    if (src_weights_desc.layout ==
        WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
      c += "    int4 w = args.src_texture.Read(src_o_group, " + yc + ");\n";
      c += "    ucl::Int32x4ToInt8x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
    } else {
      c += "  int4 w = args.src_buffer.Read(src_linear);\n";
      c += "  ucl::Int32x4ToInt8x16AsVec4x4<SType>(w, w0, w1, w2, w3);\n";
    }
  } else {
    c += "    args.src_buffer.ReadVec16AsVec4x4(w0, w1, w2, w3, "
         "src_linear);\n";
  }

  const bool src_quantized = SizeInBitsOf(src_weights_desc.type) <= 8;
  const bool dst_quantized = SizeInBitsOf(dst_type) <= 8;

  const int src_zero_point = GetZeroPoint(src_weights_desc.type);
  const std::string src_zp_str =
      "ucl::Init<Type>(" + std::to_string(src_zero_point) + ".0f)";
  if (src_zero_point != 0 && dst_quantized) {
    c += "  w0 -= " + src_zp_str + ";\n";
    c += "  w1 -= " + src_zp_str + ";\n";
    c += "  w2 -= " + src_zp_str + ";\n";
    c += "  w3 -= " + src_zp_str + ";\n";
  }
  c += "  }  // i/o slice bounds\n";

  if (src_quantized && !dst_quantized) {
    c += Dequantize(src_zero_point, grouped_quantization, batched_quantization,
                    has_zero_point, src_weights_desc.IsI4O4());
  }

  const int dst_zero_point = GetZeroPoint(dst_type);
  if (dst_zero_point != 0) {
    const std::string dst_zp_str =
        "ucl::Init<Type>(" + std::to_string(dst_zero_point) + ".0f)";
    c += "  w0 += " + dst_zp_str + ";\n";
    c += "  w1 += " + dst_zp_str + ";\n";
    c += "  w2 += " + dst_zp_str + ";\n";
    c += "  w3 += " + dst_zp_str + ";\n";
  }
  return c;
}

namespace {
std::string AddBatchOffset(const std::string& stride) {
  return absl::Substitute(R"(
    a0 += spatial_linear * $0;
    a1 += spatial_linear * $0;
    a2 += spatial_linear * $0;
    a3 += spatial_linear * $0;
)",
                          stride);
}

std::string GetRingedOAddresses() {
  std::string c;
  // kOSpatialIOGroupO4I4 -> kBIOI4
  c += R"(    int o0 = (o_slice * 4 + ring_o_offset) % ring_size;
    int o1 = (o_slice * 4 + 1 + ring_o_offset) % ring_size;
    int o2 = (o_slice * 4 + 2 + ring_o_offset) % ring_size;
    int o3 = (o_slice * 4 + 3 + ring_o_offset) % ring_size;
    int a0 = i_slice * ring_size + o0;
    int a1 = i_slice * ring_size + o1;
    int a2 = i_slice * ring_size + o2;
    int a3 = i_slice * ring_size + o3;
)";
  c += AddBatchOffset("ring_size * args.i_slices");
  return c;
}

std::string GetRingedIAddresses() {
  std::string c;
  // kOSpatialIOGroupI4O4 -> kBIOI4O4
  c += R"(    int i0 = (i_slice * 4 + ring_i_offset) % ring_size;
    int i1 = (i_slice * 4 + 1 + ring_i_offset) % ring_size;
    int i2 = (i_slice * 4 + 2 + ring_i_offset) % ring_size;
    int i3 = (i_slice * 4 + 3 + ring_i_offset) % ring_size;
    int a0 = ((i0 / 4) * args.o_slices + o_slice) * 4 + i0 % 4;
    int a1 = ((i1 / 4) * args.o_slices + o_slice) * 4 + i1 % 4;
    int a2 = ((i2 / 4) * args.o_slices + o_slice) * 4 + i2 % 4;
    int a3 = ((i3 / 4) * args.o_slices + o_slice) * 4 + i3 % 4;
)";
  c += AddBatchOffset("ring_size * args.o_slices");
  return c;
}

std::string ReadRingedWeights(const ConvRuntimeCheckDesc& runtime_check,
                              const WeightsDescription& src_weights_desc,
                              const DataType& dst_type,
                              bool grouped_quantization,
                              bool batched_quantization, bool has_zero_point) {
  std::string c;
  c += "  w0 = ucl::Init<Type>(0);\n";
  c += "  w1 = ucl::Init<Type>(0);\n";
  c += "  w2 = ucl::Init<Type>(0);\n";
  c += "  w3 = ucl::Init<Type>(0);\n";
  c += "  if (o_slice < args.o_slices && i_slice < args.i_slices) {\n";
  if (runtime_check.ring_o_offset_index.has_value()) {
    c += "    int ring_o_offset = args.params.Read(" +
         std::to_string(runtime_check.ring_o_offset_index.value()) + ");\n";
    c += "    int ring_size = " +
         std::to_string(runtime_check.ring_size.value()) + ";\n";
    c += GetRingedOAddresses();
  } else if (runtime_check.ring_i_offset_index.has_value()) {
    c += "    int ring_i_offset = args.params.Read(" +
         std::to_string(runtime_check.ring_i_offset_index.value()) + ");\n";
    c += "    int ring_size = " +
         std::to_string(runtime_check.ring_size.value()) + ";\n";
    c += GetRingedIAddresses();
  }
  c += R"(
    w0 = args.src_buffer.Read(a0);
    w1 = args.src_buffer.Read(a1);
    w2 = args.src_buffer.Read(a2);
    w3 = args.src_buffer.Read(a3);
)";
  c += "  }  // o_slice/i_slice bounds\n";
  return c;
}
}  // namespace

std::string GetWeightsConverterCode(const GpuInfo& gpu_info, DataType dst_type,
                                    const WeightsDescription& src_weights_desc,
                                    const WeightsDescription& dst_weights_desc,
                                    bool grouped_quantization,
                                    bool batched_quantization,
                                    bool has_zero_point,
                                    bool use_2d_grid_x_is_i_ogroup_y_is_o,
                                    const ConvRuntimeCheckDesc& runtime_check) {
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  c += GetWeightsCoords(dst_weights_desc, use_2d_grid_x_is_i_ogroup_y_is_o);
  if (runtime_check.dst_end_ch_index.has_value()) {
    c += "  int dst_end_slice_runtime = " +
         runtime_check.GetRuntimeEndSlice(
             "args.params.Read(args.dst_end_ch_index)", "args.o_slices") +
         ";\n";
    c += "  if (o_slice >= dst_end_slice_runtime) return;\n";
  }
  if (runtime_check.src_end_ch_index.has_value()) {
    c += "  int src_end_slice_runtime = " +
         runtime_check.GetRuntimeEndSlice(
             "args.params.Read(args.src_end_ch_index)", "args.i_slices") +
         ";\n";
    c += "  if (i_slice >= src_end_slice_runtime) return;\n";
  }
  c += "  Type w0, w1, w2, w3;\n";
  if (runtime_check.ring_o_offset_index.has_value() ||
      runtime_check.ring_i_offset_index.has_value()) {
    c += ReadRingedWeights(runtime_check, src_weights_desc,
                           dst_weights_desc.type, grouped_quantization,
                           batched_quantization, has_zero_point);
  } else {
    c += ReadVec16AsVec4x4(src_weights_desc, dst_weights_desc.type,
                           grouped_quantization, batched_quantization,
                           has_zero_point);
  }

  const auto src_layout =
      src_weights_desc.IsI4O4() ? BlockLayout::kI4O4 : BlockLayout::kO4I4;
  c += WriteResults(dst_weights_desc, src_layout, gpu_info);
  const bool dst_quantized =
      SizeOf(dst_weights_desc.type) <= SizeOf(DataType::UINT8);
  const auto type = dst_quantized ? DataType::FLOAT32 : dst_type;
  absl::StrReplaceAll(
      {{"SType", ToUclDataType(type, 1)}, {"Type", ToUclDataType(type, 4)}},
      &c);
  return c;
}

// version for kISpatialOI4O4UnalignedIO
std::string GetWeightsConverterCodeUnalignedIO(
    const GpuInfo& gpu_info, const WeightsDescription& src_weights_desc,
    const WeightsDescription& dst_weights_desc, bool grouped_quantization,
    bool batched_quantization, bool has_zero_point) {
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  c += "  if (ucl::GetGlobalId<2>() != 0) return;\n";
  c += GetLinearIndex();
  c += "  if (linear_index >= args.task_size) return;\n";
  c += "  int src_x4 = args.wshape_i / 4;\n";
  c += "  int dst_x4 = args.wshape_o / 4;\n";
  c += "  int src_last_block_size = args.wshape_i - src_x4 * 4;\n";
  c += "  int dst_last_block_size = args.wshape_o - dst_x4 * 4;\n";
  c += "  SType temps[4];\n";
  c += "  temps[0] = ucl::Init<SType>(0);\n";
  c += "  temps[1] = ucl::Init<SType>(0);\n";
  c += "  temps[2] = ucl::Init<SType>(0);\n";
  c += "  temps[3] = ucl::Init<SType>(0);\n";
  c += "  for (int i = 0; i < 4; ++i) {\n";
  c += "    int index_in = linear_index * 4 + i;\n";
  c += "    if (linear_index < args.wshape_total) {\n";
  c += "      int spatial_linear = 0;\n";
  c += "      int I = 0;\n";
  c += "      int O = 0;\n";
  c += ParseLinearIndex();
  c += "      int W = spatial_linear % args.wshape_w;\n";
  c += "      int H = spatial_linear / args.wshape_w;\n";
  c += "      Type w0, w1, w2, w3;\n";
  c += "      int i_slice = I / 4;\n";
  c += "      int o_slice = O / 4;\n";
  c += "      int i_sub_ch = I % 4;\n";
  c += "      int o_sub_ch = O % 4;\n";
  c += ReadVec16AsVec4x4(src_weights_desc, dst_weights_desc.type,
                         grouped_quantization, batched_quantization,
                         has_zero_point);
  if (src_weights_desc.IsI4O4()) {
    c += "     int element_id = i_sub_ch;\n";
    c += "     int sub_ch_id = o_sub_ch;\n";
  } else {
    // O4I4
    c += "     int element_id = o_sub_ch;\n";
    c += "     int sub_ch_id = i_sub_ch;\n";
  }
  c += "     Type w_temp = w0;\n";
  c += "     if (element_id == 1) { w_temp = w1; }\n";
  c += "     if (element_id == 2) { w_temp = w2; }\n";
  c += "     if (element_id == 3) { w_temp = w3; }\n";
  c += "     temps[i] = w_temp.x;\n";
  c += "     if (sub_ch_id == 1) { temps[i] = w_temp.y; }\n";
  c += "     if (sub_ch_id == 2) { temps[i] = w_temp.z; }\n";
  c += "     if (sub_ch_id == 3) { temps[i] = w_temp.w; }\n";
  c += "    }\n";
  c += "  }\n";
  c += "  Type result;\n";
  c += "  result.x = temps[0];\n";
  c += "  result.y = temps[1];\n";
  c += "  result.z = temps[2];\n";
  c += "  result.w = temps[3];\n";
  c += "  args.dst_tensor.WriteLinear(result, linear_index);\n";
  c += "}\n";
  absl::StrReplaceAll({{"SType", ToUclDataType(dst_weights_desc.type, 1)},
                       {"Type", ToUclDataType(dst_weights_desc.type, 4)}},
                      &c);
  return c;
}

bool IsWeightsConversionSupported(const OHWI& weights_shape,
                                  const WeightsDescription& src_weights_desc,
                                  const WeightsDescription& dst_weights_desc) {
  if (weights_shape.w != 1 || weights_shape.h != 1) return false;
  if (weights_shape.i % 4 != 0 || weights_shape.o % 4 != 0) return false;
  if (!(dst_weights_desc.type == DataType::FLOAT32 ||
        dst_weights_desc.type == DataType::FLOAT16)) {
    return false;
  }
  if (src_weights_desc.IsCustomSpatial() ||
      dst_weights_desc.IsCustomSpatial()) {
    return false;
  }
  return true;
}

WeightsConverter::WeightsConverter(const GpuInfo& gpu_info,
                                   const OperationDef& definition,
                                   const OHWI& weights_shape,
                                   const WeightsDescription& src_weights_desc,
                                   const WeightsDescription& dst_weights_desc,
                                   const TensorDescriptor* weights_scale,
                                   const TensorDescriptor* weights_zero_point,
                                   const ConvRuntimeCheckDesc& runtime_check)
    : GPUOperation(),
      weights_shape_(weights_shape),
      dst_weights_desc_(dst_weights_desc),
      use_2d_grid_x_is_i_ogroup_y_is_o_(Use2DGridXisIOgroupYisO(
          gpu_info, weights_shape, src_weights_desc, dst_weights_desc)) {
  args_.AddInt("wshape_o", weights_shape_.o);
  args_.AddInt("wshape_h", weights_shape_.h);
  args_.AddInt("wshape_w", weights_shape_.w);
  args_.AddInt("wshape_i", weights_shape_.i);
  args_.AddInt("wshape_spatial", weights_shape_.w * weights_shape_.h);
  work_group_size_ =
      int3(std::min(128, gpu_info.GetMaxWorkGroupSizeForX()), 1, 1);
  if (dst_weights_desc_.IsLinearLayout()) {
    groups_per_x_ = GetGroupsPerX(gpu_info, dst_weights_desc_, weights_shape_,
                                  work_group_size_);
    args_.AddInt("groups_per_x", groups_per_x_);
  }
  const bool ringed_weights = runtime_check.ring_o_offset_index.has_value() ||
                              runtime_check.ring_i_offset_index.has_value();
  const int vec_size = ringed_weights ? 4 : 16;
  if (src_weights_desc.type == DataType::UINT8) {
    if (src_weights_desc.layout ==
        WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
      TensorDescriptor desc = TensorDescriptor(
          DataType::UINT32, TensorStorageType::TEXTURE_2D, Layout::HW);
      AddSrcTensor("src_texture", desc);
    } else {
      BufferDescriptor desc;
      desc.element_type = DataType::UINT32;
      desc.element_size = 4;
      AddSrcBuffer("src_buffer", desc);
    }
  } else if (src_weights_desc.type == DataType::UINT4) {
    if (src_weights_desc.layout ==
        WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
      TensorDescriptor desc = TensorDescriptor(
          DataType::UINT16, TensorStorageType::TEXTURE_2D, Layout::HW);
      AddSrcTensor("src_texture", desc);
    } else {
      BufferDescriptor desc;
      desc.element_type = DataType::UINT32;
      desc.element_size = 2;
      AddSrcBuffer("src_buffer", desc);
    }
  } else if (src_weights_desc.type == DataType::UINT2) {
    if (src_weights_desc.layout ==
        WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
      TensorDescriptor desc = TensorDescriptor(
          DataType::UINT8, TensorStorageType::TEXTURE_2D, Layout::HW);
      AddSrcTensor("src_texture", desc);
    } else {
      BufferDescriptor desc;
      desc.element_type = DataType::UINT32;
      desc.element_size = 1;
      AddSrcBuffer("src_buffer", desc);
    }
  } else if (src_weights_desc.type == DataType::INT8) {
    if (src_weights_desc.layout ==
        WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
      TensorDescriptor desc = TensorDescriptor(
          DataType::INT32, TensorStorageType::TEXTURE_2D, Layout::HW);
      AddSrcTensor("src_texture", desc);
    } else {
      BufferDescriptor desc;
      desc.element_type = DataType::INT32;
      desc.element_size = 4;
      AddSrcBuffer("src_buffer", desc);
    }
  } else {
    BufferDescriptor desc;
    desc.element_type = definition.src_tensors[0].GetDataType();
    desc.element_size = vec_size;
    AddSrcBuffer("src_buffer", desc);
  }
  if (dst_weights_desc.layout ==
          WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4 ||
      dst_weights_desc.layout ==
          WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4) {
    AddDstTensor("dst_tensor0", definition.dst_tensors[0]);
    AddDstTensor("dst_tensor1", definition.dst_tensors[1]);
    AddDstTensor("dst_tensor2", definition.dst_tensors[2]);
    AddDstTensor("dst_tensor3", definition.dst_tensors[3]);
    uint2 tex_size = Get2dResourceSize(dst_weights_desc, weights_shape);
    args_.AddInt("dst_x_size", tex_size.x);
    args_.AddInt("dst_y_size", tex_size.y);
  } else if (dst_weights_desc.layout ==
             WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4) {
    TensorDescriptor desc;
    uint2 tex_size = Get2dResourceSize(dst_weights_desc, weights_shape);
    if (dst_weights_desc.type == DataType::UINT8) {
      tex_size.x /= sizeof(uint32_t) / sizeof(uint8_t);
      desc = TensorDescriptor(DataType::UINT32, TensorStorageType::TEXTURE_2D,
                              Layout::HW);
    } else {
      // TODO: b/378522761 - Support other data types for
      // k2DYIsSpatialIOAndXIsOGroupI4O4 layout.
      ABSL_LOG(FATAL) << absl::StrCat(
          "Weights conversion to k2DYIsSpatialIOAndXIsOGroupI4O4 layout with "
          "data "
          "type ",
          ToString(dst_weights_desc_.type), " is unsupported.");
    }
    AddDstTensor("dst_texture", desc);
    args_.AddInt("dst_x_size", tex_size.x);
    args_.AddInt("dst_y_size", tex_size.y);
  } else {
    if (dst_weights_desc.type == DataType::INT8 ||
        dst_weights_desc.type == DataType::UINT8) {
      BufferDescriptor desc;
      desc.element_type = DataType::UINT32;
      desc.element_size =
          IsLastBlockHas4IAnd4OElements(dst_weights_desc) ? 4 : 1;
      AddDstBuffer("dst_buffer", desc);
    } else if (dst_weights_desc.type == DataType::INT4 ||
               dst_weights_desc.type == DataType::UINT4) {
      BufferDescriptor desc;
      desc.element_type = DataType::UINT32;
      desc.element_size = 2;
      AddDstBuffer("dst_buffer", desc);
    } else {
      AddDstTensor("dst_tensor", definition.dst_tensors[0]);
    }
    if (dst_weights_desc_.layout == WeightsLayout::kISpatialOI4O4UnalignedIO) {
      args_.AddInt("task_size",
                   DivideRoundUp(weights_shape.DimensionsProduct(), 4));
      args_.AddInt("wshape_total", weights_shape_.DimensionsProduct());
    } else {
      int elements_count =
          GetTotalElementsCountForLayout(dst_weights_desc, weights_shape);
      args_.AddInt("dst_total_size", DivideRoundUp(elements_count, 16));
    }
  }
  bool grouped_quantization = false;
  bool batched_quantization = false;
  if (weights_scale && weights_scale->GetLayout() != Layout::LINEAR) {
    auto scale_shape = weights_scale->GetBHWCShape();
    const int scale_i_groups = DivideRoundUp(scale_shape.c, 4);
    const int batch_size = scale_shape.h;
    if (batch_size > 1) {
      batched_quantization = true;
    }
    if (scale_i_groups > 1) {
      grouped_quantization = true;
      args_.AddInt("scale_zp_group_size",
                   DivideRoundUp(weights_shape_.i / scale_i_groups, 4));
    }
  }
  if (weights_scale) {
    AddSrcTensor("weights_scale", *weights_scale);
  }
  const bool has_zero_point = weights_zero_point != nullptr;
  if (weights_zero_point) {
    AddSrcTensor("weights_zero_point", *weights_zero_point);
  }

  bool has_runtime_check = false;
  if (runtime_check.src_end_ch_index.has_value()) {
    args_.AddInt("src_end_ch_index", *runtime_check.src_end_ch_index);
    has_runtime_check = true;
  }
  if (runtime_check.dst_end_ch_index.has_value()) {
    args_.AddInt("dst_end_ch_index", *runtime_check.dst_end_ch_index);
    has_runtime_check = true;
  }
  if (runtime_check.ring_o_offset_index.has_value()) {
    args_.AddInt("ring_o_offset_index", *runtime_check.ring_o_offset_index);
    has_runtime_check = true;
  }
  if (runtime_check.ring_i_offset_index.has_value()) {
    args_.AddInt("ring_i_offset_index", *runtime_check.ring_i_offset_index);
    has_runtime_check = true;
  }
  if (has_runtime_check) {
    BufferDescriptor desc;
    desc.element_type = DataType::INT32;
    desc.element_size = 1;
    AddSrcBuffer("params", desc);
  }

  AddCommonArgs(dst_weights_desc, weights_shape, args_);
  args_.AddInt("src_o_group_size", src_weights_desc.GetOutputGroupSize());
  const int o_slices = DivideRoundUp(weights_shape.o, 4);
  args_.AddInt("src_o_groups",
               DivideRoundUp(o_slices, src_weights_desc.GetOutputGroupSize()));
  if (dst_weights_desc.layout == WeightsLayout::kISpatialOI4O4UnalignedIO) {
    if (has_runtime_check) {
      ABSL_LOG(FATAL)
          << "Runtime check is unsupported for kISpatialOI4O4UnalignedIO "
             "layout.";
    }
    code_ = GetWeightsConverterCodeUnalignedIO(
        gpu_info, src_weights_desc, dst_weights_desc, grouped_quantization,
        batched_quantization, has_zero_point);
  } else {
    code_ = GetWeightsConverterCode(
        gpu_info, definition.dst_tensors[0].GetDataType(), src_weights_desc,
        dst_weights_desc, grouped_quantization, batched_quantization,
        has_zero_point, use_2d_grid_x_is_i_ogroup_y_is_o_, runtime_check);
  }
}

std::vector<int3> WeightsConverter::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info) const {
  if (use_2d_grid_x_is_i_ogroup_y_is_o_) {
    if (gpu_info.IsApiOpenCl() && gpu_info.IsAdreno()) {
      const std::vector<int3> work_groups_to_try = {
          int3(64, 4, 1),  int3(32, 8, 1),  int3(16, 16, 1), int3(8, 32, 1),
          int3(4, 64, 1),  int3(128, 4, 1), int3(64, 8, 1),  int3(32, 16, 1),
          int3(16, 32, 1), int3(8, 64, 1)};
      std::vector<int3> work_groups;
      for (const auto& work_group : work_groups_to_try) {
        if (IsValidWorkgroup(gpu_info, work_group)) {
          work_groups.push_back(work_group);
        }
      }
      return work_groups;
    } else {
      return {int3(8, 16, 1)};
    }
  } else {
    return {work_group_size_};
  }
}

int3 WeightsConverter::GetGridSize() const {
  if (use_2d_grid_x_is_i_ogroup_y_is_o_) {
    const int i_slices = DivideRoundUp(weights_shape_.i, 4);
    const int o_slices = DivideRoundUp(weights_shape_.o, 4);
    const int group_size = dst_weights_desc_.GetOutputGroupSize();
    const int grid_x = i_slices * group_size;
    const int grid_y = DivideRoundUp(o_slices, group_size);
    return int3(grid_x, grid_y, 1);
  } else {
    return GetGrid(dst_weights_desc_, weights_shape_, work_group_size_,
                   groups_per_x_);
  }
}

}  // namespace ml_drift
