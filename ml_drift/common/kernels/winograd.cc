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

#include "ml_drift/common/kernels/winograd.h"

#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/compiler_options.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/work_group_picking.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"
#include "ml_drift/common/winograd_util.h"

namespace ml_drift {
namespace {
void VectorToKernelBufferDesc(const std::vector<float>& data,
                              DataType data_type,
                              BufferDescriptor* buffer_desc) {
  buffer_desc->element_type = data_type;
  buffer_desc->element_size = 1;
  buffer_desc->memory_type = MemoryType::CONSTANT;
  buffer_desc->attributes.push_back("kernel_global_space");
  buffer_desc->size = SizeOf(data_type) * data.size();
  buffer_desc->data.resize(buffer_desc->size);
  if (data_type == DataType::FLOAT32) {
    memcpy(buffer_desc->data.data(), data.data(), buffer_desc->size);
  } else {
    half* hf_ptr = reinterpret_cast<half*>(buffer_desc->data.data());
    for (int i = 0; i < data.size(); ++i) {
      hf_ptr[i] = data[i];
    }
  }
}
std::string GetKernelWinograd4x4To36(const GpuInfo& gpu_info,
                                     const OperationDef& op_def) {
  std::string c;
  const auto src_desc = op_def.src_tensors[0];
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int X = (linear_id / args.dst_tensor.Batch()) * 4;\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int X = ucl::GetGlobalId<0>() * 4;\n";
  }
  c += R"(
  int Y = ucl::GetGlobalId<1>() * 4;
  int S = ucl::GetGlobalId<2>();

  if (X / 4 >= args.tiles_x || Y / 4 >= args.tiles_y) return;

  Type I[6][6];
  for (int y = 0; y < 6; ++y) {
    for (int x = 0; x < 6; ++x) {
      I[y][x] = ucl::Init<Type>(0.0f);
    }
  }
)";
  for (int y = 0; y < 6; ++y) {
    const std::string s_y = std::to_string(y);
    c += "  {\n";
    c += "    int coord_y = Y + " + s_y + " + args.padding_y;\n";
    if (!src_desc.SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
      c += "    bool in_y = coord_y >= 0 && coord_y < "
           "args.src_tensor.Height();\n";
      c += "    coord_y = clamp(coord_y, 0, args.src_tensor.Height() - 1);\n";
    }
    for (int x = 0; x < 6; ++x) {
      const std::string s_x = std::to_string(x);
      c += "    {\n";
      c += "      int coord_x = X + " + s_x + " + args.padding_x;\n";
      if (!src_desc.SupportsZeroClamp(Axis::WIDTH, gpu_info)) {
        c += "      bool in_x = coord_x >= 0 && coord_x < "
             "args.src_tensor.Width();\n";
        c += "      coord_x = clamp(coord_x, 0, args.src_tensor.Width()-1);\n";
      }
      std::string multiplier;
      if (!src_desc.SupportsZeroClamp(Axis::WIDTH, gpu_info) &&
          !src_desc.SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
        multiplier = " * ucl::Convert<SType>(in_y && in_x)";
      } else if (!src_desc.SupportsZeroClamp(Axis::WIDTH, gpu_info)) {
        multiplier = " * ucl::Convert<SType>(in_x)";
      } else if (!src_desc.SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
        multiplier = " * ucl::Convert<SType>(in_y)";
      }
      c += "      Type src = args.src_tensor.Read(coord_x, coord_y, S)" +
           multiplier + ";\n";
      c += "      I[0][" + s_x + "] += args.Bt.Read(" + std::to_string(y) +
           ") * src;\n";
      c += "      I[1][" + s_x + "] += args.Bt.Read(" + std::to_string(y + 6) +
           ") * src;\n";
      c += "      I[2][" + s_x + "] += args.Bt.Read(" + std::to_string(y + 12) +
           ") * src;\n";
      c += "      I[3][" + s_x + "] += args.Bt.Read(" + std::to_string(y + 18) +
           ") * src;\n";
      c += "      I[4][" + s_x + "] += args.Bt.Read(" + std::to_string(y + 24) +
           ") * src;\n";
      c += "      I[5][" + s_x + "] += args.Bt.Read(" + std::to_string(y + 30) +
           ") * src;\n";
      c += "    }\n";
    }
    c += "  }\n";
  }

  c += "  int dst_x = Y / 4 * args.tiles_x + X / 4;\n";
  c += "  for (int y = 0; y < 6; ++y) {\n";
  c += "    Type value;\n";
  for (int x = 0; x < 6; ++x) {
    c += "    value = ucl::Init<Type>(0.0f);\n";
    for (int k = 0; k < 6; ++k) {
      c += "    value += args.Bt.Read(" + std::to_string(x * 6 + k) +
           ") * I[y][" + std::to_string(k) + "];\n";
    }
    c += "    args.dst_tensor.Write(value, dst_x, y * 6 + " +
         std::to_string(x) + ", S);\n";
  }
  c += "  }\n";
  c += "}\n";
  const DataType type = op_def.src_tensors[0].GetDataType();
  absl::StrReplaceAll(
      {{"SType", ToUclDataType(type, 1)}, {"Type", ToUclDataType(type, 4)}},
      &c);
  return c;
}

std::string GetKernelWinograd36To4x4(const OperationDef& op_def) {
  std::string c;
  const auto src_desc = op_def.src_tensors[0];
  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int tile_id = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int tile_id = ucl::GetGlobalId<0>();\n";
  }
  c += R"(
  int Z = ucl::GetGlobalId<2>();
  int tiles_count_x = (args.dst_tensor.Width() + 3) / 4;
  int tile_x = (tile_id % tiles_count_x) * 4;
  int tile_y = (tile_id / tiles_count_x) * 4;
  if (tile_x >= args.dst_tensor.Width() || tile_y >= args.dst_tensor.Height()) return;

  Type I[4][6];
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 6; ++x) {
      I[y][x] = ucl::Init<Type>(0.0f);
    }
  }
  for (int y = 0; y < 6; ++y) {
    for (int x = 0; x < 6; ++x) {
      Type src = args.src_tensor.Read(tile_id, y * 6 + x, Z);
      I[0][x] += src * args.At.Read(y);
      I[1][x] += src * args.At.Read(y + 6);
      I[2][x] += src * args.At.Read(y + 12);
      I[3][x] += src * args.At.Read(y + 18);
    }
  }

  Type bias_val = args.biases.Read(Z);
)";

  c += "  for (int y = 0; y < 4; ++y) {\n";
  for (int x = 0; x < 4; ++x) {
    c += "    if (tile_x + " + std::to_string(x) +
         " < args.dst_tensor.Width() && tile_y + y < args.dst_tensor.Height()) "
         "{\n";
    c += "      Type value = bias_val;\n";
    for (int k = 0; k < 6; ++k) {
      c += "      value += args.At.Read(" + std::to_string(x * 6 + k) +
           ") * I[y][" + std::to_string(k) + "];\n";
    }
    c += "      args.dst_tensor.Write(value, tile_x + " + std::to_string(x) +
         ", tile_y + y, Z);\n";
    c += "    }\n";
  }
  c += "  }\n";
  c += "}\n";
  absl::StrReplaceAll(
      {{"Type", ToUclDataType(op_def.dst_tensors[0].GetDataType(), 4)}}, &c);
  return c;
}
}  // namespace

int3 Winograd4x4To36::GetGridSize() const {
  int new_width =
      src_[0]->Width() + padding_.prepended.w + padding_.appended.w - 2;
  int new_height =
      src_[0]->Height() + padding_.prepended.h + padding_.appended.h - 2;
  int tiles_x = DivideRoundUp(new_width, 4);
  int tiles_y = DivideRoundUp(new_height, 4);
  return int3(tiles_x * dst_[0]->Batch(), tiles_y, src_[0]->Slices());
}

absl::Status Winograd4x4To36::BindArguments(ArgumentsBinder* args) {
  int new_width =
      src_[0]->Width() + padding_.prepended.w + padding_.appended.w - 2;
  int new_height =
      src_[0]->Height() + padding_.prepended.h + padding_.appended.h - 2;
  int tiles_x = DivideRoundUp(new_width, 4);
  int tiles_y = DivideRoundUp(new_height, 4);
  RETURN_IF_ERROR(args->SetInt("tiles_x", tiles_x));
  RETURN_IF_ERROR(args->SetInt("tiles_y", tiles_y));
  return absl::OkStatus();
}

Winograd4x4To36 CreateWinograd4x4To36(const OperationDef& definition,
                                      const Padding2D& padding,
                                      const GpuInfo& gpu_info) {
  Winograd4x4To36 desc(definition, padding);
  desc.code_ = GetKernelWinograd4x4To36(gpu_info, definition);

  desc.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  desc.args_.AddInt("padding_x", -padding.prepended.w);
  desc.args_.AddInt("padding_y", -padding.prepended.h);
  desc.args_.AddInt("tiles_x");
  desc.args_.AddInt("tiles_y");

  BufferDescriptor buffer_desc;
  VectorToKernelBufferDesc(BtMatrixForWinograd3x3TileNxN(/*tile_size*/ 6),
                           definition.dst_tensors[0].GetDataType(),
                           &buffer_desc);
  desc.args_.AddObject(
      "Bt", std::make_unique<BufferDescriptor>(std::move(buffer_desc)));

  desc.work_group_size_ = int3(8, 4, 1);
  return desc;
}

Winograd3x3TiledXForward::Winograd3x3TiledXForward(
    const OperationDef& definition, const Padding2D& padding,
    const GpuInfo& gpu_info, int tile_size)
    : padding_(padding) {
  work_group_size_ = int3(32, 1, 1);
  tile_size_outer_ = tile_size;
  tile_size_inner_ = tile_size_outer_ - 2;
  use_spatial_caching_ = gpu_info.IsApple();
  if (use_spatial_caching_) {
    work_group_size_ = int3(8, tile_size_outer_, 1);
  }
  code_ = GetCode(definition, gpu_info);
  if (gpu_info.IsAdreno()) {
    compiler_options_.push_back(CompilerOptions::kAdrenoMoreWaves);
  }
  if (definition.src_tensors[0].GetDataType() == DataType::FLOAT16 &&
      gpu_info.IsPowerVR()) {
    compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
}

std::string GenerateSrcCaching(const GpuInfo& gpu_info,
                               const OperationDef& op_def, int wg_x_size,
                               int tile_size) {
  int cache_size_x = wg_x_size * tile_size;
  int cache_size_y = tile_size;
  int groups_x = DivideRoundUp(cache_size_x, wg_x_size);
  std::string c;
  c += "  __local Type spatial_cache[" + std::to_string(cache_size_y) + "][" +
       std::to_string(cache_size_x) + "];\n";
  c += "  {\n";
  c += "    int src_x;\n";
  c += "    int src_y = tile_y + args.padding_y + ucl::GetLocalId<1>();\n";
  std::string mx_multiplier, my_multiplier;
  if (!op_def.src_tensors[0].SupportsZeroClamp(Axis::WIDTH, gpu_info)) {
    c += "    bool inx;\n";
  }
  if (!op_def.src_tensors[0].SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
    c += "    bool iny = src_y >= 0 && src_y < args.src_tensor.Height();\n";
    c += "    src_y = clamp(src_y, 0, args.src_tensor.Height() - 1);\n";
    my_multiplier = " * ucl::Convert<SType>(iny)";
  }
  std::string ly = "ucl::GetLocalId<1>()";
  for (int gr_x = 0; gr_x < groups_x; ++gr_x) {
    c +=
        "    src_x = tile_x + args.padding_x + " + std::to_string(gr_x) + ";\n";
    if (!op_def.src_tensors[0].SupportsZeroClamp(Axis::WIDTH, gpu_info)) {
      c += "    inx = src_x >= 0 && src_x < args.src_tensor.Width();\n";
      c += "    src_x = clamp(src_x, 0, args.src_tensor.Width() - 1);\n";
      mx_multiplier = " * ucl::Convert<SType>(inx)";
    }
    c += "    spatial_cache[ucl::GetLocalId<1>()][ucl::GetLocalId<0>() + " +
         std::to_string(wg_x_size * gr_x) +
         "] = args.src_tensor.Read(src_x, src_y, DST_Z)" + mx_multiplier +
         my_multiplier + ";\n";
  }
  c += "  }\n";
  return c;
}

std::string Winograd3x3TiledXForward::GetCode(const OperationDef& op_def,
                                              const GpuInfo& gpu_info) {
  std::string c;
  const auto& src_desc = op_def.src_tensors[0];
  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  args_.AddInt("padding_x");
  args_.AddInt("padding_y");
  args_.AddInt("tiles_total");
  args_.AddInt("tiles_x");

  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int DST_X = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int DST_X = ucl::GetGlobalId<0>();\n";
  }
  c += "  int DST_Y = ucl::GetGlobalId<1>();\n";
  c += "  int DST_Z = ucl::GetGlobalId<2>();\n";
  c += "  int tile_x = (DST_X % args.tiles_x) * TILE_SIZE_INNER;\n";
  c += "  int tile_y = (DST_X / args.tiles_x) * TILE_SIZE_INNER;\n";
  if (use_spatial_caching_) {
    c += GenerateSrcCaching(gpu_info, op_def, work_group_size_.x,
                            tile_size_outer_);
    c += "  ucl::SyncThreads<WorkGroup, Local>();\n";
  }
  c += "  if (DST_X >= args.tiles_total || DST_Y >= TILE_SIZE_OUTER || DST_Z "
       ">= args.dst_tensor.Slices()) {\n";
  c += "    return; \n";
  c += "  }\n";
  c += "  Type I0";
  for (int y = 1; y < tile_size_outer_; ++y) {
    c += ", I" + std::to_string(y);
  }
  c += ";\n";
  c += "  SType bt_ar[TILE_SIZE_OUTER];\n";
  for (int k = 0; k < DivideRoundUp(tile_size_outer_, 4); ++k) {
    c += "  Type t" + std::to_string(k) +
         " = args.bt_non_uniform.Read(DST_Y * " +
         std::to_string(DivideRoundUp(tile_size_outer_, 4)) + " + " +
         std::to_string(k) + ");\n";
  }
  c += "  DST_Y *= TILE_SIZE_OUTER;\n";
  for (int y = 0; y < tile_size_outer_; ++y) {
    std::string postfixes[] = {"x", "y", "z", "w"};
    c += "  bt_ar[" + std::to_string(y) + "] = t" + std::to_string(y / 4) +
         "." + postfixes[y % 4] + ";\n";
  }
  auto write_result = [&]() {
    for (int y = 0; y < tile_size_outer_; ++y) {
      c += "  {\n";
      c += "    Type r0 = ucl::Init<Type>(0.0f);\n";
      for (int k = 0; k < tile_size_outer_; ++k) {
        c += "    r0 += args.Bt.Read(" +
             std::to_string(y * tile_size_outer_ + k) + ") * I" +
             std::to_string(k) + ";\n";
      }
      c += "    args.dst_tensor.Write(r0, DST_X, DST_Y, DST_Z);\n";
      c += "    DST_Y++;\n";
      c += "  }\n";
    }
  };
  if (use_spatial_caching_) {
    c += "  {\n";
    for (int x = 0; x < tile_size_outer_; ++x) {
      const std::string xs = std::to_string(x);
      c += "    I" + xs +
           " = bt_ar[0] * spatial_cache[0][ucl::GetLocalId<0>() + " +
           std::to_string(work_group_size_.x * x) + "];\n";
    }
    c += "  }\n";
    for (int y = 1; y < tile_size_outer_; ++y) {
      const std::string ys = std::to_string(y);
      c += "  {\n";
      for (int x = 0; x < tile_size_outer_; ++x) {
        const std::string xs = std::to_string(x);
        c += "    I" + xs + " += bt_ar[" + ys + "] * spatial_cache[" + ys +
             "][ucl::GetLocalId<0>() + " +
             std::to_string(work_group_size_.x * x) + "];\n";
      }
      c += "  }\n";
    }
    write_result();
    c += "}\n";
    const DataType type = op_def.src_tensors[0].GetDataType();
    return absl::StrReplaceAll(
        c, {{"TILE_SIZE_INNER", std::to_string(tile_size_inner_)},
            {"TILE_SIZE_OUTER", std::to_string(tile_size_outer_)},
            {"SType", ToUclDataType(type, 1)},
            {"Type", ToUclDataType(type, 4)}});
  }
  auto read_src = [&](const std::string& src, const std::string& xs) {
    std::string read_statement;
    read_statement = "args.src_tensor.Read(xc" + xs + ", yc, DST_Z)";
    std::string multiplier;
    if (!src_desc.SupportsZeroClamp(Axis::WIDTH, gpu_info)) {
      multiplier += " * m" + xs + "_x";
    }
    if (!src_desc.SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
      multiplier += " * ucl::Convert<SType>(iny)";
    }
    c += "    Type " + src + " = " + read_statement + multiplier + ";\n";
  };
  for (int x = 0; x < tile_size_outer_; ++x) {
    const std::string xs = std::to_string(x);
    c += "  int xc" + xs + " = tile_x + args.padding_x + " + xs + ";\n";
    if (!src_desc.SupportsZeroClamp(Axis::WIDTH, gpu_info)) {
      c += "  bool inx" + xs + " = (xc" + xs + " >= 0 && xc" + xs +
           " < args.src_tensor.Width());\n";
      c += "  SType m" + xs + "_x = ucl::Convert<SType>(inx" + xs + ");\n";
      c += "  xc" + xs + " = clamp(xc" + xs +
           ", 0, args.src_tensor.Width() - 1);\n";
    }
  }
  const bool manual_unroll =
      !(op_def.src_tensors[0].GetDataType() == DataType::FLOAT32 &&
        gpu_info.IsMali());
  if (manual_unroll) {
    c += "  {\n";
    c += "    int yc = tile_y + args.padding_y;\n";
    if (!src_desc.SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
      c += "    bool iny = (yc >= 0 && yc < args.src_tensor.Height());\n";
      c += "    yc = clamp(yc, 0, args.src_tensor.Height() - 1);\n";
      c += "    SType bt = bt_ar[0] * ucl::Convert<SType>(iny);\n";
    } else {
      c += "    SType bt = bt_ar[0];\n";
    }
    for (int x = 0; x < tile_size_outer_; ++x) {
      const std::string xs = std::to_string(x);
      const std::string src = "src" + xs;
      read_src(src, xs);
      c += "    I" + xs + " = bt * " + src + ";\n";
    }
    c += "  }\n";
    for (int y = 1; y < tile_size_outer_; ++y) {
      const std::string ys = std::to_string(y);
      c += "  {\n";
      c += "    int yc = tile_y + args.padding_y + (" + ys + ");\n";
      if (!src_desc.SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
        c += "    bool iny = (yc >= 0 && yc < args.src_tensor.Height());\n";
        c += "    yc = clamp(yc, 0, args.src_tensor.Height() - 1);\n";
        c += "    SType bt = bt_ar[" + ys + "] * ucl::Convert<SType>(iny);\n";
      } else {
        c += "    SType bt = bt_ar[" + ys + "];\n";
      }
      for (int x = 0; x < tile_size_outer_; ++x) {
        const std::string xs = std::to_string(x);
        const std::string src = "src" + xs;
        read_src(src, xs);
        c += "    I" + xs + " += bt * " + src + ";\n";
      }
      c += "  }\n";
    }
  } else {
    for (int y = 0; y < tile_size_outer_; ++y) {
      c += "  I" + std::to_string(y) + " = ucl::Init<Type>(0.0f);\n";
    }
    c += "  for (int y = 0; y < TILE_SIZE_OUTER; ++y) {\n";
    c += "    int yc = tile_y + args.padding_y + y;\n";
    if (!src_desc.SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
      c += "    bool iny = (yc >= 0 && yc < args.src_tensor.Height());\n";
      c += "    yc = clamp(yc, 0, args.src_tensor.Height() - 1);\n";
      c += "    SType bt = bt_ar[y] * ucl::Convert<SType>(iny);\n";
    } else {
      c += "    SType bt = bt_ar[y];\n";
    }
    for (int x = 0; x < tile_size_outer_; ++x) {
      const std::string xs = std::to_string(x);
      const std::string src = "src" + xs;
      read_src(src, xs);
      c += "    I" + xs + " += bt * " + src + ";\n";
    }
    c += "  }\n";
  }
  write_result();
  c += "}\n";
  const DataType type = op_def.src_tensors[0].GetDataType();
  return absl::StrReplaceAll(
      c, {{"TILE_SIZE_INNER", std::to_string(tile_size_inner_)},
          {"TILE_SIZE_OUTER", std::to_string(tile_size_outer_)},
          {"SType", ToUclDataType(type, 1)},
          {"Type", ToUclDataType(type, 4)}});
}

void Winograd3x3TiledXForward::UploadBt(const OperationDef& op_def) {
  Tensor<Linear, DataType::FLOAT32> bt_aligned;
  const int aligned_width = AlignByN(tile_size_outer_, 4);
  bt_aligned.shape = Linear(tile_size_outer_ * aligned_width);
  bt_aligned.data.resize(bt_aligned.shape.DimensionsProduct());
  auto bt_mat = BtMatrixForWinograd3x3TileNxN(tile_size_outer_);
  for (int y = 0; y < tile_size_outer_; ++y) {
    for (int x = 0; x < tile_size_outer_; ++x) {
      bt_aligned.data[y * aligned_width + x] = bt_mat[y * tile_size_outer_ + x];
    }
    for (int x = tile_size_outer_; x < aligned_width; ++x) {
      bt_aligned.data[y * aligned_width + x] = 0.0f;
    }
  }

  TensorDescriptor bt_tensor_desc = CreateConstantLinearTensorDescriptor(
      op_def.src_tensors[0].GetDataType(),
      op_def.src_tensors[0].GetStorageType(), bt_aligned);
  args_.AddObject("bt_non_uniform", std::make_unique<TensorDescriptor>(
                                        std::move(bt_tensor_desc)));

  BufferDescriptor buffer_desc;
  VectorToKernelBufferDesc(bt_mat, op_def.src_tensors[0].GetDataType(),
                           &buffer_desc);
  args_.AddObject("Bt",
                  std::make_unique<BufferDescriptor>(std::move(buffer_desc)));
}

int3 Winograd3x3TiledXForward::SelectBestWorkGroup(
    const KernelInfo& kernel_info) const {
  const std::vector<int3> wgs = {
      {8, tile_size_outer_, 4}, {8, tile_size_outer_, 2},
      {4, tile_size_outer_, 2}, {4, tile_size_outer_, 2},
      {2, tile_size_outer_, 2}, {2, tile_size_outer_, 1},
      {1, tile_size_outer_, 1}, {1, 1, 1}};
  return GetFirstSuitableWorkGroup(wgs, kernel_info.max_work_group_size);
}

absl::Status Winograd3x3TiledXForward::BindArguments(ArgumentsBinder* args) {
  const int tiles_x = DivideRoundUp(
      src_[0]->Width() + padding_.prepended.w + padding_.appended.w - 2,
      tile_size_inner_);
  const int tiles_y = DivideRoundUp(
      src_[0]->Height() + padding_.prepended.h + padding_.appended.h - 2,
      tile_size_inner_);
  const int tiles_total = tiles_x * tiles_y;
  RETURN_IF_ERROR(args->SetInt("padding_x", -padding_.prepended.w));
  RETURN_IF_ERROR(args->SetInt("padding_y", -padding_.prepended.h));
  RETURN_IF_ERROR(args->SetInt("tiles_total", tiles_total));
  RETURN_IF_ERROR(args->SetInt("tiles_x", tiles_x));
  return absl::OkStatus();
}

int3 Winograd3x3TiledXForward::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = tile_size_outer_;
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

std::vector<int3> Winograd3x3TiledXForward::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info) const {
  if (use_spatial_caching_) {
    return {work_group_size_};
  }
  if (gpu_info.IsIntel()) {
    return {int3(4, 6, 1)};
  }
  switch (tuning_type) {
    case TuningType::kExhaustive:
      return GetPossibleWorkGroups(tuning_type, gpu_info, kernel_info,
                                   grid_size_);
    case TuningType::kFast:
    default:
      return {SelectBestWorkGroup(kernel_info)};
  }
}

Winograd3x3TiledXForward CreateWinograd3x3TiledXForward(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const Padding2D& padding, int tile_size) {
  Winograd3x3TiledXForward result(definition, padding, gpu_info, tile_size);
  result.UploadBt(definition);
  return result;
}

int3 Winograd36To4x4::GetGridSize() const {
  return int3(src_[0]->Width() * dst_[0]->Batch(), 1, src_[0]->Slices());
}

Winograd36To4x4 CreateWinograd36To4x4(
    const OperationDef& definition,
    const Tensor<Linear, DataType::FLOAT32>& biases) {
  Winograd36To4x4 desc;
  desc.code_ = GetKernelWinograd36To4x4(definition);

  desc.AddSrcTensor("src_tensor", definition.src_tensors[0]);
  desc.AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
      definition.src_tensors[0].GetDataType(),
      definition.src_tensors[0].GetStorageType(), biases);
  desc.args_.AddObject("biases", std::make_unique<TensorDescriptor>(
                                     std::move(bias_tensor_desc)));

  BufferDescriptor buffer_desc;
  VectorToKernelBufferDesc(AtMatrixForWinograd3x3TileNxN(/*tile_size*/ 6),
                           definition.src_tensors[0].GetDataType(),
                           &buffer_desc);
  desc.args_.AddObject(
      "At", std::make_unique<BufferDescriptor>(std::move(buffer_desc)));

  desc.work_group_size_ = int3(32, 1, 1);
  return desc;
}

Winograd3x3TiledXBackward::Winograd3x3TiledXBackward(
    const OperationDef& definition, const GpuInfo& gpu_info, int tile_size) {
  work_group_size_ = int3(32, 1, 1);
  tile_size_outer_ = tile_size;
  tile_size_inner_ = tile_size_outer_ - 2;
  if (definition.src_tensors[0].GetDataType() == DataType::FLOAT16 &&
      gpu_info.IsPowerVR()) {
    compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
  code_ = GetCode(definition, gpu_info);
}

std::string Winograd3x3TiledXBackward::GetCode(const OperationDef& op_def,
                                               const GpuInfo& gpu_info) {
  std::string c;

  AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  args_.AddInt("tiles_x");

  c += "MAIN_FUNCTION($0) {\n";
  if (op_def.dst_tensors[0].HasAxis(Axis::BATCH)) {
    c += "  int linear_id = ucl::GetGlobalId<0>();\n";
    c += "  int tile_id = linear_id / args.dst_tensor.Batch();\n";
    c += "  int B = linear_id % args.dst_tensor.Batch();\n";
    c += "  args.src_tensor.SetBatchRef(B);\n";
    c += "  args.dst_tensor.SetBatchRef(B);\n";
  } else {
    c += "  int tile_id = ucl::GetGlobalId<0>();\n";
  }

  c += "  int DST_Y = ucl::GetGlobalId<1>();\n";
  c += "  int DST_Z = ucl::GetGlobalId<2>();\n";
  c += "  int tile_x = (tile_id % args.tiles_x) * TILE_SIZE_INNER;\n";
  c += "  int tile_y = (tile_id / args.tiles_x) * TILE_SIZE_INNER + DST_Y;\n";

  c += "  if (tile_x >= args.dst_tensor.Width() || tile_y >= "
       "args.dst_tensor.Height() || DST_Z >= args.dst_tensor.Slices()) {\n";
  c += "    return; \n";
  c += "  }\n";
  c += "  Type I0";
  for (int y = 1; y < tile_size_outer_; ++y) {
    c += ", I" + std::to_string(y);
  }
  c += ";\n";
  c += "  SType at_ar[TILE_SIZE_OUTER];\n";
  for (int k = 0; k < DivideRoundUp(tile_size_outer_, 4); ++k) {
    c += "  Type t0" + std::to_string(k) +
         " = args.at_non_uniform.Read(DST_Y * " +
         std::to_string(DivideRoundUp(tile_size_outer_, 4)) + " + " +
         std::to_string(k) + ");\n";
  }
  for (int y = 0; y < tile_size_outer_; ++y) {
    std::string postfixes[] = {"x", "y", "z", "w"};
    c += "  at_ar[" + std::to_string(y) + "] = t0" + std::to_string(y / 4) +
         "." + postfixes[y % 4] + ";\n";
  }
  const bool manual_unroll =
      !(op_def.src_tensors[0].GetDataType() == DataType::FLOAT32 &&
        gpu_info.IsMali());
  if (manual_unroll) {
    c += "  {\n";
    c += "    SType at = at_ar[0];\n";
    for (int x = 0; x < tile_size_outer_; ++x) {
      const std::string yc = std::to_string(x);
      const std::string src = "src" + std::to_string(x);
      c += "    Type " + src + " = args.src_tensor.Read(tile_id, " + yc +
           ", DST_Z);\n";
      c += "    I" + std::to_string(x) + " = at * " + src + ";\n";
    }
    c += "  }\n";
    for (int y = 1; y < tile_size_outer_; ++y) {
      c += "  {\n";
      c += "    SType at = at_ar[" + std::to_string(y) + "];\n";
      for (int x = 0; x < tile_size_outer_; ++x) {
        const std::string yc = std::to_string(y * tile_size_outer_ + x);
        const std::string src = "src" + std::to_string(x);
        c += "    Type " + src + " = args.src_tensor.Read(tile_id, " + yc +
             ", DST_Z);\n";
        c += "    I" + std::to_string(x) + " += at * " + src + ";\n";
      }
      c += "  }\n";
    }
  } else {
    for (int y = 0; y < tile_size_outer_; ++y) {
      c += "  I" + std::to_string(y) + " = ucl::Init<Type>(0.0f);\n";
    }
    c += "  for (int y = 0; y < TILE_SIZE_OUTER; ++y) {\n";
    c += "    SType at = at_ar[y];\n";
    for (int x = 0; x < tile_size_outer_; ++x) {
      const std::string src = "src" + std::to_string(x);
      c += "    Type " + src +
           " = args.src_tensor.Read(tile_id, y * TILE_SIZE_OUTER + " +
           std::to_string(x) + ", DST_Z);\n";
      c += "    I" + std::to_string(x) + " += at * " + src + ";\n";
    }
    c += "  }\n";
  }
  c += "  Type bias_val = args.biases.Read(DST_Z);\n";
  for (int y = 0; y < tile_size_inner_; ++y) {
    c += "  if (tile_x < args.dst_tensor.Width()) {\n";
    c += "    Type r0 = bias_val;\n";
    for (int k = 0; k < tile_size_outer_; ++k) {
      c += "    r0 += args.At.Read(" +
           std::to_string(y * tile_size_outer_ + k) + ") * I" +
           std::to_string(k) + ";\n";
    }
    c += "    args.dst_tensor.Write(r0, tile_x, tile_y, DST_Z);\n";
    c += "    tile_x++;\n";
    c += "  }\n";
  }
  c += "}\n";

  const DataType type = op_def.src_tensors[0].GetDataType();
  return absl::StrReplaceAll(
      c, {{"TILE_SIZE_INNER", std::to_string(tile_size_inner_)},
          {"TILE_SIZE_OUTER", std::to_string(tile_size_outer_)},
          {"SType", ToUclDataType(type, 1)},
          {"Type", ToUclDataType(type, 4)}});
}

void Winograd3x3TiledXBackward::UploadAt(const OperationDef& op_def) {
  Tensor<Linear, DataType::FLOAT32> at_aligned;
  const int aligned_width = AlignByN(tile_size_outer_, 4);
  at_aligned.shape = Linear(tile_size_inner_ * aligned_width);
  at_aligned.data.resize(tile_size_inner_ * aligned_width);
  auto at_mat = AtMatrixForWinograd3x3TileNxN(tile_size_outer_);
  for (int y = 0; y < tile_size_inner_; ++y) {
    for (int x = 0; x < tile_size_outer_; ++x) {
      at_aligned.data[y * aligned_width + x] = at_mat[y * tile_size_outer_ + x];
    }
    for (int x = tile_size_outer_; x < aligned_width; ++x) {
      at_aligned.data[y * aligned_width + x] = 0.0f;
    }
  }

  TensorDescriptor at_tensor_desc = CreateConstantLinearTensorDescriptor(
      op_def.src_tensors[0].GetDataType(),
      op_def.src_tensors[0].GetStorageType(), at_aligned);
  args_.AddObject("at_non_uniform", std::make_unique<TensorDescriptor>(
                                        std::move(at_tensor_desc)));

  BufferDescriptor buffer_desc;
  VectorToKernelBufferDesc(at_mat, op_def.src_tensors[0].GetDataType(),
                           &buffer_desc);
  args_.AddObject("At",
                  std::make_unique<BufferDescriptor>(std::move(buffer_desc)));
}

int3 Winograd3x3TiledXBackward::SelectBestWorkGroup(
    const KernelInfo& kernel_info) const {
  const std::vector<int3> wgs = {
      {32, tile_size_inner_, 1}, {16, tile_size_inner_, 2},
      {16, tile_size_inner_, 1}, {8, tile_size_inner_, 1},
      {4, tile_size_inner_, 1},  {2, tile_size_inner_, 1},
      {1, tile_size_inner_, 1},  {1, 1, 1}};
  return GetFirstSuitableWorkGroup(wgs, kernel_info.max_work_group_size);
}

absl::Status Winograd3x3TiledXBackward::BindArguments(ArgumentsBinder* args) {
  const int tiles_x = DivideRoundUp(dst_[0]->Width(), tile_size_inner_);
  RETURN_IF_ERROR(args->SetInt("tiles_x", tiles_x));
  return absl::OkStatus();
}

int3 Winograd3x3TiledXBackward::GetGridSize() const {
  const int tiles_x = DivideRoundUp(dst_[0]->Width(), tile_size_inner_);
  const int tiles_y = DivideRoundUp(dst_[0]->Height(), tile_size_inner_);
  const int grid_x = tiles_x * tiles_y * dst_[0]->Batch();
  const int grid_y = tile_size_inner_;
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

std::vector<int3> Winograd3x3TiledXBackward::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info) const {
  if (gpu_info.IsIntel()) {
    return {int3(8, 4, 1)};
  }
  switch (tuning_type) {
    case TuningType::kExhaustive:
      return GetPossibleWorkGroups(tuning_type, gpu_info, kernel_info,
                                   grid_size_);
    case TuningType::kFast:
    default:
      return {SelectBestWorkGroup(kernel_info)};
  }
}

Winograd3x3TiledXBackward CreateWinograd3x3TiledXBackward(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const Tensor<Linear, DataType::FLOAT32>& biases, int tile_size) {
  Winograd3x3TiledXBackward result(definition, gpu_info, tile_size);
  TensorDescriptor bias_tensor_desc = CreateConstantLinearTensorDescriptor(
      gpu_info, definition.src_tensors[0].GetDataType(), biases);
  result.args_.AddObject("biases", std::make_unique<TensorDescriptor>(
                                       std::move(bias_tensor_desc)));
  result.UploadAt(definition);
  return result;
}

std::string GetCodeWinogradWeightsTransform() {
  std::string c = R"(MAIN_FUNCTION($0) {
  int o_ch = ucl::GetGlobalId<0>();
  int i_slice = ucl::GetGlobalId<1>();
  int H = ucl::GetGlobalId<2>();
  if (o_ch >= args.dst.Batch() || i_slice >= args.dst.Slices() || H >= args.dst.Height()) return;

  float4 V0 = args.mat.Read(H);
  float4 S0, S1, S2;
  int ohwi_coord = o_ch * 9 * args.dst.Slices() + i_slice;
)";
  for (int y = 0; y < 3; ++y) {
    for (int x = 0; x < 3; ++x) {
      std::string src_val = "args.src.Read<float>(ohwi_coord + " +
                            std::to_string(y * 3 + x) + " * args.dst.Slices())";
      const std::string postfixes[] = {"x", "y", "z", "w"};
      src_val += " * V0." + postfixes[y];
      const std::string op = y == 0 ? "  = " : " += ";
      c += "  S" + std::to_string(x) + op + src_val + ";\n";
    }
  }
  c += R"(
  for (int x = 0; x < args.dst.Width(); ++x) {
    float4 V1 = args.mat.Read(x);
    float4 value = S0 * V1.x + S1 * V1.y + S2 * V1.z;
    args.dst::type out_value = ucl::Convert<args.dst::type>(value);
    args.dst.Write(out_value, x, H, i_slice, o_ch);
  }
}
)";
  return c;
}

Winograd3x3To36::Winograd3x3To36(const TensorDescriptor& src_desc,
                                 const TensorDescriptor& dst_desc) {
  code_ = GetCodeWinogradWeightsTransform();
  AddSrcTensor("src", src_desc);
  AddDstTensor("dst", dst_desc);
  AddTransformMatrix();
  work_group_size_ = int3(8, 4, 6);
}

void Winograd3x3To36::AddTransformMatrix() {
  auto mat = GetTransposedMatrixForWinograd3(6);
  Tensor<Linear, DataType::FLOAT32> mat_opt;
  mat_opt.shape = Linear(6 * 4);
  mat_opt.data.resize(mat_opt.shape.DimensionsProduct());
  for (int i = 0; i < 6; ++i) {
    mat_opt.data[i * 4 + 0] = mat[i];
    mat_opt.data[i * 4 + 1] = mat[i + 6];
    mat_opt.data[i * 4 + 2] = mat[i + 2 * 6];
    mat_opt.data[i * 4 + 3] = 0.0f;
  }

  TensorDescriptor mat_desc = CreateConstantLinearTensorDescriptor(
      DataType::FLOAT32, TensorStorageType::TEXTURE_2D, mat_opt);
  args_.AddObject("mat",
                  std::make_unique<TensorDescriptor>(std::move(mat_desc)));
}

}  // namespace ml_drift
