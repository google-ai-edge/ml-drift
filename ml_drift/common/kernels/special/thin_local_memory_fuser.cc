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

#include "ml_drift/common/kernels/special/thin_local_memory_fuser.h"

#include <any>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/flops_util.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/ir_model.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/kernels/resize.h"
#include "ml_drift/common/model.h"
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
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace {
std::string GetSecondPart(const GpuInfo& gpu_info, const OHWI& weights_shape,
                          const int2& block_size, DataType type,
                          bool need_write_check = true) {
  const int conv_i_slices = DivideRoundUp(weights_shape.i, 4);
  std::string c;
  for (int y = 0; y < block_size.y; ++y) {
    for (int x = 0; x < block_size.x; ++x) {
      c += "  Type dst_x" + std::to_string(x) + "_y" + std::to_string(y) +
           " = ucl::Init<Type>(0.0f);\n";
    }
  }
  for (int x = 0; x < block_size.x + 2; ++x) {
    c += "  Type src_x" + std::to_string(x) + ";\n";
  }
  const bool load_src_4half4_as_int8 =
      type == DataType::FLOAT16 && gpu_info.IsAdreno() &&
      gpu_info.IsApiOpenCl() && (block_size.x + 2) % 4 == 0;
  const bool load_src_2half4_as_int4 =
      type == DataType::FLOAT16 && gpu_info.IsAdreno() &&
      gpu_info.IsApiOpenCl() && (block_size.x + 2) % 2 == 0;
  if (load_src_4half4_as_int8) {
    c += "  __local int8* src_iptr;\n";
    c += "  int8 src_xi;\n";
  } else if (load_src_2half4_as_int4) {
    c += "  __local int4* src_iptr;\n";
    c += "  int4 src_xi;\n";
  }
  for (int src_s = 0; src_s < conv_i_slices; ++src_s) {
    const std::string local_mem =
        conv_i_slices == 1 ? "interm_tensor"
                           : "interm_tensor[" + std::to_string(src_s) + "]";
    for (int by = 0; by < block_size.y + 2; ++by) {
      if (load_src_4half4_as_int8) {
        c += "  src_iptr = (__local int8*)&(" + local_mem +
             "[second_local_y * BLOCK_SIZE_Y + " + std::to_string(by) +
             "][second_local_x * BLOCK_SIZE_X]);\n";
        for (int x = 0; x < (block_size.x + 2) / 4; ++x) {
          c += "  src_xi = src_iptr[" + std::to_string(x) + "];\n";
          c += "  src_x" + std::to_string(x * 4) + " = as_half4(src_xi.s01);\n";
          c += "  src_x" + std::to_string(x * 4 + 1) +
               " = as_half4(src_xi.s23);\n";
          c += "  src_x" + std::to_string(x * 4 + 2) +
               " = as_half4(src_xi.s45);\n";
          c += "  src_x" + std::to_string(x * 4 + 3) +
               " = as_half4(src_xi.s67);\n";
        }
      } else if (load_src_2half4_as_int4) {
        c += "  src_iptr = (__local int4*)&(" + local_mem +
             "[second_local_y * BLOCK_SIZE_Y + " + std::to_string(by) +
             "][second_local_x * BLOCK_SIZE_X]);\n";
        for (int x = 0; x < (block_size.x + 2) / 2; ++x) {
          c += "  src_xi = src_iptr[" + std::to_string(x) + "];\n";
          c += "  src_x" + std::to_string(x * 2) + " = as_half4(src_xi.xy);\n";
          c += "  src_x" + std::to_string(x * 2 + 1) +
               " = as_half4(src_xi.zw);\n";
        }
      } else {
        for (int bx = 0; bx < block_size.x + 2; ++bx) {
          const std::string src_idx = std::to_string(bx);
          c += "  src_x" + src_idx + " = " + local_mem +
               "[second_local_y * BLOCK_SIZE_Y + " + std::to_string(by) +
               "][second_local_x * BLOCK_SIZE_X + " + src_idx + "];\n";
        }
      }
      for (int bx = 0; bx < block_size.x; ++bx) {
        const std::string dst_idx = std::to_string(bx);
        for (int ky = 0; ky < 3; ++ky) {
          int dst_idy = by - ky;
          if (dst_idy < 0 || dst_idy >= block_size.y) {
            continue;
          }
          for (int kx = 0; kx < 3; ++kx) {
            if (weights_shape.o == 1) {
              c += "  dst_x" + dst_idx + "_y" + std::to_string(dst_idy) +
                   ".x += dot(src_x" + std::to_string(bx + kx) +
                   ", args.weights.Read(" + std::to_string(ky * 3 + kx) +
                   "));\n";
            } else {
              const int w_id = (ky * 3 + kx) * conv_i_slices + src_s;
              if (weights_shape.i == 3) {
                c += "  dst_x" + dst_idx + "_y" + std::to_string(dst_idy) +
                     " += src_x" + std::to_string(bx + kx) +
                     ".x * args.weights.Read(" + std::to_string(w_id * 3 + 0) +
                     ");\n";
                c += "  dst_x" + dst_idx + "_y" + std::to_string(dst_idy) +
                     " += src_x" + std::to_string(bx + kx) +
                     ".y * args.weights.Read(" + std::to_string(w_id * 3 + 1) +
                     ");\n";
                c += "  dst_x" + dst_idx + "_y" + std::to_string(dst_idy) +
                     " += src_x" + std::to_string(bx + kx) +
                     ".z * args.weights.Read(" + std::to_string(w_id * 3 + 2) +
                     ");\n";
              } else {
                c += "  dst_x" + dst_idx + "_y" + std::to_string(dst_idy) +
                     " += src_x" + std::to_string(bx + kx) +
                     ".x * args.weights.Read(" + std::to_string(w_id * 4 + 0) +
                     ");\n";
                c += "  dst_x" + dst_idx + "_y" + std::to_string(dst_idy) +
                     " += src_x" + std::to_string(bx + kx) +
                     ".y * args.weights.Read(" + std::to_string(w_id * 4 + 1) +
                     ");\n";
                c += "  dst_x" + dst_idx + "_y" + std::to_string(dst_idy) +
                     " += src_x" + std::to_string(bx + kx) +
                     ".z * args.weights.Read(" + std::to_string(w_id * 4 + 2) +
                     ");\n";
                c += "  dst_x" + dst_idx + "_y" + std::to_string(dst_idy) +
                     " += src_x" + std::to_string(bx + kx) +
                     ".w * args.weights.Read(" + std::to_string(w_id * 4 + 3) +
                     ");\n";
              }
            }
          }
        }
      }
    }
  }
  c += "  Type bias_val = args.bias.Read(0);\n";
  for (int by = 0; by < block_size.y; ++by) {
    for (int bx = 0; bx < block_size.x; ++bx) {
      const std::string block_idx = std::to_string(bx);
      const std::string block_idy = std::to_string(by);
      const std::string dst_name = "dst_x" + block_idx + "_y" + block_idy;
      c += "  " + dst_name + " += bias_val;\n";
      if (need_write_check) {
        c += "  if (DST_X + " + block_idx +
             " < args.dst_tensor.Width() && DST_Y + " + block_idy +
             " < args.dst_tensor.Height()) {\n";
      }
      c += "    args.dst_tensor.Write(" + dst_name + ", DST_X + " + block_idx +
           ", DST_Y + " + block_idy + ", 0);\n";
      if (need_write_check) {
        c += "  }\n";
      }
    }
  }
  c += "}\n";
  return c;
}

std::string GetCodeWGFirstTensorPrecise(const GpuInfo& gpu_info,
                                        const std::vector<Node*> nodes,
                                        const OHWI& weights_shape,
                                        const int3 wg_size, DataType type,
                                        const int2& block_size,
                                        bool need_write_check = true) {
  const int conv_i_slices = DivideRoundUp(weights_shape.i, 4);
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (conv_i_slices == 1) {
    c += "  __local Type interm_tensor[WG_SIZE_Y][WG_SIZE_X];\n";
    c += "  int local_s = 0;\n";
  } else {
    c += "  __local Type interm_tensor[WG_SIZE_Z][WG_SIZE_Y][WG_SIZE_X];\n";
    c += "  int local_s = ucl::GetLocalId<2>();\n";
  }
  c += R"(
  int local_x = ucl::GetLocalId<0>();
  int local_y = ucl::GetLocalId<1>();

  int interm_x = (WG_SIZE_X - 2) * ucl::GetGroupId<0>() + local_x - 1;
  int interm_y = (WG_SIZE_Y - 2) * ucl::GetGroupId<1>() + local_y - 1;
  Type value = ucl::Init<Type>(0.0f);
  if (interm_x >= 0 && interm_x < args.dst_tensor.Width() && interm_y >= 0 && interm_y < args.dst_tensor.Height()) {
)";
  for (auto* node : nodes) {
    const OperationType op_type = OperationTypeFromString(node->operation.type);
    if (op_type == OperationType::RESIZE) {
      auto resize_attr =
          std::any_cast<Resize2DAttributes>(node->operation.attributes);
      c += GetResize2dCode(resize_attr, "src_tensor", "interm_x", "interm_y",
                           "local_s", "value");
    } else if (op_type == OperationType::ADD) {
      c += "    value += args.add_tensor.Read(interm_x, interm_y, "
           "local_s);\n";
    } else if (op_type == OperationType::CONCAT) {
      c += "    value.x = args.src_tensor0.Read(interm_x, interm_y, 0).x;\n";
      c += "    value.y = args.src_tensor1.Read(interm_x, interm_y, 0).x;\n";
      c += "    value.z = args.src_tensor2.Read(interm_x, interm_y, 0).x;\n";
    }
  }
  c += "  }\n";
  if (conv_i_slices == 1) {
    c += "  interm_tensor[local_y][local_x] = value;\n";
  } else {
    c += "  interm_tensor[local_s][local_y][local_x] = value;\n";
  }
  c += "  ucl::SyncThreads<WorkGroup, Local>();\n";
  if (conv_i_slices != 1) {
    c += "  if (local_s != 0) { return; }\n";
  }
  if (block_size.x != 1 || block_size.y != 1) {
    // linear remapping:
    c += R"(
  int local_id = local_y * WG_SIZE_X + local_x;
  int second_local_id = local_id;
  if (second_local_id >= (WG_SIZE_X - 2) * (WG_SIZE_Y - 2) / BLOCK_SIZE_X / BLOCK_SIZE_Y) { return; }
  int second_local_x = second_local_id % ((WG_SIZE_X - 2) / BLOCK_SIZE_X);
  int second_local_y = second_local_id / ((WG_SIZE_X - 2) / BLOCK_SIZE_X);
)";
  } else {
    // 2d remapping:
    c += R"(
  // 2d remapping:
  int second_local_x = local_x;
  int second_local_y = local_y;
  if (second_local_x >= (WG_SIZE_X - 2) || second_local_y >= (WG_SIZE_Y - 2)) { return; }
)";
  }
  c += R"(
  int DST_X = (WG_SIZE_X - 2) * ucl::GetGroupId<0>() + second_local_x * BLOCK_SIZE_X;
  int DST_Y = (WG_SIZE_Y - 2) * ucl::GetGroupId<1>() + second_local_y * BLOCK_SIZE_Y;

  if (DST_X >= args.dst_tensor.Width() || DST_Y >= args.dst_tensor.Height()) { return; }
)";
  c += GetSecondPart(gpu_info, weights_shape, block_size, type,
                     need_write_check);
  return absl::StrReplaceAll(c,
                             {{"WG_SIZE_X", std::to_string(wg_size.x)},
                              {"WG_SIZE_Y", std::to_string(wg_size.y)},
                              {"WG_SIZE_Z", std::to_string(wg_size.z)},
                              {"Type", ToUclDataType(type, 4)},
                              {"BLOCK_SIZE_X", std::to_string(block_size.x)},
                              {"BLOCK_SIZE_Y", std::to_string(block_size.y)}});
}

std::string GetCodeWGSecondTensorPrecise(const GpuInfo& gpu_info,
                                         const std::vector<Node*> nodes,
                                         const OHWI& weights_shape,
                                         const int3 wg_size, DataType type,
                                         const int2& block_size,
                                         bool need_write_check = true) {
  const int conv_i_slices = DivideRoundUp(weights_shape.i, 4);
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (conv_i_slices == 1) {
    c += "  __local Type interm_tensor[LOCAL_MEM_SIZE_Y][LOCAL_MEM_SIZE_X];\n";
  } else {
    c += "  __local Type "
         "interm_tensor[LOCAL_MEM_SIZE_S][LOCAL_MEM_SIZE_Y][LOCAL_MEM_SIZE_X];"
         "\n";
  }
  c += R"(
  int local_x = ucl::GetLocalId<0>();
  int local_y = ucl::GetLocalId<1>();
  int local_id = local_y * WG_SIZE_X + local_x;
  for (int tile_id = 0; tile_id < args.tiles_count; ++tile_id) {
    int interm_linear_id = local_id + tile_id * WG_SIZE_Y * WG_SIZE_X;
    if (interm_linear_id < LOCAL_MEM_SIZE_Y * LOCAL_MEM_SIZE_X * LOCAL_MEM_SIZE_S) {
      int interm_local_x = interm_linear_id % LOCAL_MEM_SIZE_X;
)";
  if (conv_i_slices == 1) {
    c += "      int interm_local_y = interm_linear_id / LOCAL_MEM_SIZE_X;\n";
    c += "      int local_s = 0;\n";
  } else {
    c += "      interm_linear_id /= LOCAL_MEM_SIZE_X;\n";
    c += "      int interm_local_y = interm_linear_id % LOCAL_MEM_SIZE_Y;\n";
    c += "      int local_s = interm_linear_id / LOCAL_MEM_SIZE_Y;\n";
  }
  c += R"(
      int interm_x = WG_SIZE_X * BLOCK_SIZE_X * ucl::GetGroupId<0>() + interm_local_x - 1;
      int interm_y = WG_SIZE_Y * BLOCK_SIZE_Y * ucl::GetGroupId<1>() + interm_local_y - 1;
      Type value = ucl::Init<Type>(0.0f);
      if (interm_x >= 0 && interm_x < args.dst_tensor.Width() && interm_y >= 0 && interm_y < args.dst_tensor.Height()) {
)";
  for (auto* node : nodes) {
    const OperationType op_type = OperationTypeFromString(node->operation.type);
    if (op_type == OperationType::RESIZE) {
      auto resize_attr =
          std::any_cast<Resize2DAttributes>(node->operation.attributes);
      c += GetResize2dCode(resize_attr, "src_tensor", "interm_x", "interm_y",
                           "local_s", "value");
    } else if (op_type == OperationType::ADD) {
      c += "        value += args.add_tensor.Read(interm_x, interm_y, "
           "local_s);\n";
    } else if (op_type == OperationType::CONCAT) {
      c +=
          "        value.x = args.src_tensor0.Read(interm_x, interm_y, 0).x;\n";
      c +=
          "        value.y = args.src_tensor1.Read(interm_x, interm_y, 0).x;\n";
      c +=
          "        value.z = args.src_tensor2.Read(interm_x, interm_y, 0).x;\n";
    }
  }
  c += "  }\n";
  if (conv_i_slices == 1) {
    c += "      interm_tensor[interm_local_y][interm_local_x] = value;\n";
  } else {
    c += "      interm_tensor[local_s][interm_local_y][interm_local_x] = "
         "value;\n";
  }
  c += "    }\n";
  c += "  }\n";
  c += "  ucl::SyncThreads<WorkGroup, Local>();\n";
  c += R"(
  int DST_X = ucl::GetGlobalId<0>() * BLOCK_SIZE_X;
  int DST_Y = ucl::GetGlobalId<1>() * BLOCK_SIZE_Y;
  if (DST_X >= args.dst_tensor.Width() || DST_Y >= args.dst_tensor.Height()) { return; }
  int second_local_x = local_x;
  int second_local_y = local_y;
)";
  c += GetSecondPart(gpu_info, weights_shape, block_size, type,
                     need_write_check);
  const int local_mem_size_x = wg_size.x * block_size.x + 2;
  const int local_mem_size_y = wg_size.y * block_size.y + 2;
  return absl::StrReplaceAll(
      c, {{"LOCAL_MEM_SIZE_X", std::to_string(local_mem_size_x)},
          {"LOCAL_MEM_SIZE_Y", std::to_string(local_mem_size_y)},
          {"LOCAL_MEM_SIZE_S", std::to_string(conv_i_slices)},
          {"WG_SIZE_X", std::to_string(wg_size.x)},
          {"WG_SIZE_Y", std::to_string(wg_size.y)},
          {"WG_SIZE_Z", std::to_string(wg_size.z)},
          {"Type", ToUclDataType(type, 4)},
          {"BLOCK_SIZE_X", std::to_string(block_size.x)},
          {"BLOCK_SIZE_Y", std::to_string(block_size.y)}});
}

std::string GetCodeWGFirstTensorPrecise(const GpuInfo& gpu_info,
                                        const std::vector<const ir::IrOp*>& ops,
                                        const OHWI& weights_shape,
                                        const int3 wg_size, DataType type,
                                        const int2& block_size,
                                        bool need_write_check = true) {
  const int conv_i_slices = DivideRoundUp(weights_shape.i, 4);
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (conv_i_slices == 1) {
    c += "  __local Type interm_tensor[WG_SIZE_Y][WG_SIZE_X];\n";
    c += "  int local_s = 0;\n";
  } else {
    c += "  __local Type interm_tensor[WG_SIZE_Z][WG_SIZE_Y][WG_SIZE_X];\n";
    c += "  int local_s = ucl::GetLocalId<2>();\n";
  }
  c += R"(
  int local_x = ucl::GetLocalId<0>();
  int local_y = ucl::GetLocalId<1>();

  int interm_x = (WG_SIZE_X - 2) * ucl::GetGroupId<0>() + local_x - 1;
  int interm_y = (WG_SIZE_Y - 2) * ucl::GetGroupId<1>() + local_y - 1;
  Type value = ucl::Init<Type>(0.0f);
  if (interm_x >= 0 && interm_x < args.dst_tensor.Width() && interm_y >= 0 && interm_y < args.dst_tensor.Height()) {
)";
  for (const auto* op : ops) {
    const OperationType op_type = OperationTypeFromString(op->name);
    if (op_type == OperationType::RESIZE) {
      auto resize_attr = std::any_cast<Resize2DAttributes>(op->attr);
      c += GetResize2dCode(resize_attr, "src_tensor", "interm_x", "interm_y",
                           "local_s", "value");
    } else if (op_type == OperationType::ADD) {
      c += "    value += args.add_tensor.Read(interm_x, interm_y, "
           "local_s);\n";
    } else if (op_type == OperationType::CONCAT) {
      c += "    value.x = args.src_tensor0.Read(interm_x, interm_y, 0).x;\n";
      c += "    value.y = args.src_tensor1.Read(interm_x, interm_y, 0).x;\n";
      c += "    value.z = args.src_tensor2.Read(interm_x, interm_y, 0).x;\n";
    }
  }
  c += "  }\n";
  if (conv_i_slices == 1) {
    c += "  interm_tensor[local_y][local_x] = value;\n";
  } else {
    c += "  interm_tensor[local_s][local_y][local_x] = value;\n";
  }
  c += "  ucl::SyncThreads<WorkGroup, Local>();\n";
  if (conv_i_slices != 1) {
    c += "  if (local_s != 0) { return; }\n";
  }
  if (block_size.x != 1 || block_size.y != 1) {
    // linear remapping:
    c += R"(
  int local_id = local_y * WG_SIZE_X + local_x;
  int second_local_id = local_id;
  if (second_local_id >= (WG_SIZE_X - 2) * (WG_SIZE_Y - 2) / BLOCK_SIZE_X / BLOCK_SIZE_Y) { return; }
  int second_local_x = second_local_id % ((WG_SIZE_X - 2) / BLOCK_SIZE_X);
  int second_local_y = second_local_id / ((WG_SIZE_X - 2) / BLOCK_SIZE_X);
)";
  } else {
    // 2d remapping:
    c += R"(
  // 2d remapping:
  int second_local_x = local_x;
  int second_local_y = local_y;
  if (second_local_x >= (WG_SIZE_X - 2) || second_local_y >= (WG_SIZE_Y - 2)) { return; }
)";
  }
  c += R"(
  int DST_X = (WG_SIZE_X - 2) * ucl::GetGroupId<0>() + second_local_x * BLOCK_SIZE_X;
  int DST_Y = (WG_SIZE_Y - 2) * ucl::GetGroupId<1>() + second_local_y * BLOCK_SIZE_Y;

  if (DST_X >= args.dst_tensor.Width() || DST_Y >= args.dst_tensor.Height()) { return; }
)";
  c += GetSecondPart(gpu_info, weights_shape, block_size, type,
                     need_write_check);
  return absl::StrReplaceAll(c,
                             {{"WG_SIZE_X", std::to_string(wg_size.x)},
                              {"WG_SIZE_Y", std::to_string(wg_size.y)},
                              {"WG_SIZE_Z", std::to_string(wg_size.z)},
                              {"Type", ToUclDataType(type, 4)},
                              {"BLOCK_SIZE_X", std::to_string(block_size.x)},
                              {"BLOCK_SIZE_Y", std::to_string(block_size.y)}});
}

std::string GetCodeWGSecondTensorPrecise(
    const GpuInfo& gpu_info, const std::vector<const ir::IrOp*>& ops,
    const OHWI& weights_shape, const int3 wg_size, DataType type,
    const int2& block_size, bool need_write_check = true) {
  const int conv_i_slices = DivideRoundUp(weights_shape.i, 4);
  std::string c;
  c += "MAIN_FUNCTION($0) {\n";
  if (conv_i_slices == 1) {
    c += "  __local Type interm_tensor[LOCAL_MEM_SIZE_Y][LOCAL_MEM_SIZE_X];\n";
  } else {
    c += "  __local Type "
         "interm_tensor[LOCAL_MEM_SIZE_S][LOCAL_MEM_SIZE_Y][LOCAL_MEM_SIZE_X];"
         "\n";
  }
  c += R"(
  int local_x = ucl::GetLocalId<0>();
  int local_y = ucl::GetLocalId<1>();
  int local_id = local_y * WG_SIZE_X + local_x;
  for (int tile_id = 0; tile_id < args.tiles_count; ++tile_id) {
    int interm_linear_id = local_id + tile_id * WG_SIZE_Y * WG_SIZE_X;
    if (interm_linear_id < LOCAL_MEM_SIZE_Y * LOCAL_MEM_SIZE_X * LOCAL_MEM_SIZE_S) {
      int interm_local_x = interm_linear_id % LOCAL_MEM_SIZE_X;
)";
  if (conv_i_slices == 1) {
    c += "      int interm_local_y = interm_linear_id / LOCAL_MEM_SIZE_X;\n";
    c += "      int local_s = 0;\n";
  } else {
    c += "      interm_linear_id /= LOCAL_MEM_SIZE_X;\n";
    c += "      int interm_local_y = interm_linear_id % LOCAL_MEM_SIZE_Y;\n";
    c += "      int local_s = interm_linear_id / LOCAL_MEM_SIZE_Y;\n";
  }
  c += R"(
      int interm_x = WG_SIZE_X * BLOCK_SIZE_X * ucl::GetGroupId<0>() + interm_local_x - 1;
      int interm_y = WG_SIZE_Y * BLOCK_SIZE_Y * ucl::GetGroupId<1>() + interm_local_y - 1;
      Type value = ucl::Init<Type>(0.0f);
      if (interm_x >= 0 && interm_x < args.dst_tensor.Width() && interm_y >= 0 && interm_y < args.dst_tensor.Height()) {
)";
  for (const auto* op : ops) {
    const OperationType op_type = OperationTypeFromString(op->name);
    if (op_type == OperationType::RESIZE) {
      auto resize_attr = std::any_cast<Resize2DAttributes>(op->attr);
      c += GetResize2dCode(resize_attr, "src_tensor", "interm_x", "interm_y",
                           "local_s", "value");
    } else if (op_type == OperationType::ADD) {
      c += "        value += args.add_tensor.Read(interm_x, interm_y, "
           "local_s);\n";
    } else if (op_type == OperationType::CONCAT) {
      c +=
          "        value.x = args.src_tensor0.Read(interm_x, interm_y, 0).x;\n";
      c +=
          "        value.y = args.src_tensor1.Read(interm_x, interm_y, 0).x;\n";
      c +=
          "        value.z = args.src_tensor2.Read(interm_x, interm_y, 0).x;\n";
    }
  }
  c += "  }\n";
  if (conv_i_slices == 1) {
    c += "      interm_tensor[interm_local_y][interm_local_x] = value;\n";
  } else {
    c += "      interm_tensor[local_s][interm_local_y][interm_local_x] = "
         "value;\n";
  }
  c += "    }\n";
  c += "  }\n";
  c += "  ucl::SyncThreads<WorkGroup, Local>();\n";
  c += R"(
  int DST_X = ucl::GetGlobalId<0>() * BLOCK_SIZE_X;
  int DST_Y = ucl::GetGlobalId<1>() * BLOCK_SIZE_Y;
  if (DST_X >= args.dst_tensor.Width() || DST_Y >= args.dst_tensor.Height()) { return; }
  int second_local_x = local_x;
  int second_local_y = local_y;
)";
  c += GetSecondPart(gpu_info, weights_shape, block_size, type,
                     need_write_check);
  const int local_mem_size_x = wg_size.x * block_size.x + 2;
  const int local_mem_size_y = wg_size.y * block_size.y + 2;
  return absl::StrReplaceAll(
      c, {{"LOCAL_MEM_SIZE_X", std::to_string(local_mem_size_x)},
          {"LOCAL_MEM_SIZE_Y", std::to_string(local_mem_size_y)},
          {"LOCAL_MEM_SIZE_S", std::to_string(conv_i_slices)},
          {"WG_SIZE_X", std::to_string(wg_size.x)},
          {"WG_SIZE_Y", std::to_string(wg_size.y)},
          {"WG_SIZE_Z", std::to_string(wg_size.z)},
          {"Type", ToUclDataType(type, 4)},
          {"BLOCK_SIZE_X", std::to_string(block_size.x)},
          {"BLOCK_SIZE_Y", std::to_string(block_size.y)}});
}

void AddConstantsGpuBuffer(const GpuInfo& gpu_info, DataType data_type,
                           const std::vector<float>& weights, Arguments* args) {
  BufferDescriptor desc;
  desc.element_type = data_type;
  desc.element_size = 4;
  desc.memory_type =
      gpu_info.IsMali() || gpu_info.IsBroadcom() || gpu_info.IsAMD()
          ? MemoryType::GLOBAL
          : MemoryType::CONSTANT;
  desc.size = SizeOf(data_type) * weights.size();
  desc.data.resize(desc.size);

  if (data_type == DataType::FLOAT32) {
    memcpy(desc.data.data(), weights.data(), desc.size);
  } else {
    half* gpu_data_half = reinterpret_cast<half*>(desc.data.data());
    for (int i = 0; i < weights.size(); ++i) {
      gpu_data_half[i] = weights[i];
    }
  }
  args->AddObject("weights",
                  std::make_unique<BufferDescriptor>(std::move(desc)));
}

bool IsConcatConvLocalMemoryFuserRecommended(const GpuInfo& gpu_info) {
  return gpu_info.IsMali() && gpu_info.IsAdreno();
}

bool IsResizeAddConvLocalMemoryFuserRecommended(const GpuInfo& gpu_info) {
  const bool is_apple_recommended = gpu_info.IsApple();
  const bool is_adreno_recommended =
      gpu_info.IsAdreno() &&
      gpu_info.adreno_info.generation >= AdrenoInfo::Generation::kGen7;
  const bool is_amd_recommended = gpu_info.IsAMD();
  const bool is_mali_recommended =
      gpu_info.IsMali() &&
      gpu_info.mali_info.generation >= MaliInfo::Gen::kValhallV2;
  const bool is_powervr_recommended =
      gpu_info.IsPowerVR() && gpu_info.powervr_info.IsImgCxx();
  return is_apple_recommended || is_adreno_recommended || is_amd_recommended ||
         is_mali_recommended || is_powervr_recommended;
}

}  // namespace

class ThinLocalMemory : public GPUOperation {
 public:
  enum class TilingType {
    kWGFirstTensorPrecise = 0,
    kWGSecondTensorPrecise = 1,
  };
  ThinLocalMemory() = default;
  std::vector<int3> GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info) const override {
    return {work_group_size_};
  }
  int3 GetGridSize() const override {
    int workgroups_x = 1;
    int workgroups_y = 1;
    switch (tiling_type_) {
      case TilingType::kWGFirstTensorPrecise: {
        workgroups_x = DivideRoundUp(dst_[0]->Width(), work_group_size_.x - 2);
        workgroups_y = DivideRoundUp(dst_[0]->Height(), work_group_size_.y - 2);
        break;
      }
      case TilingType::kWGSecondTensorPrecise: {
        const int grid_x = DivideRoundUp(dst_[0]->Width(), block_size_.x);
        const int grid_y = DivideRoundUp(dst_[0]->Height(), block_size_.y);
        workgroups_x = DivideRoundUp(grid_x, work_group_size_.x);
        workgroups_y = DivideRoundUp(grid_y, work_group_size_.y);
        break;
      }
    }
    return int3(workgroups_x * work_group_size_.x,
                workgroups_y * work_group_size_.y, 1);
  }

  // Move only
  ThinLocalMemory(ThinLocalMemory&& kernel) = default;
  ThinLocalMemory& operator=(ThinLocalMemory&& kernel) = default;
  ThinLocalMemory(const ThinLocalMemory&) = delete;
  ThinLocalMemory& operator=(const ThinLocalMemory&) = delete;

  TilingType tiling_type_ = TilingType::kWGSecondTensorPrecise;
  int2 block_size_ = int2(1, 1);
};

absl::Status TryConcatConvLocalMemoryFuser(
    const GpuInfo& gpu_info, const GraphFloat32& graph, NodeId first_node_id,
    const std::set<NodeId>& consumed_nodes,
    std::set<NodeId>* new_consumed_nodes, GpuModelBuilder* model_builder) {
  // experimental disabled version
  if (!IsConcatConvLocalMemoryFuserRecommended(gpu_info)) {
    // performance reason
    return absl::NotFoundError("ConcatConv not recommended.");
  }

  auto* concat_node = graph.GetNode(first_node_id);
  if (concat_node == nullptr) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }
  if (OperationTypeFromString(concat_node->operation.type) !=
      OperationType::CONCAT) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }
  auto concat_inputs = graph.FindInputs(concat_node->id);
  if (concat_inputs.size() != 3 || concat_inputs[0]->tensor.shape.c != 1 ||
      concat_inputs[1]->tensor.shape.c != 1 ||
      concat_inputs[2]->tensor.shape.c != 1) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }
  ASSIGN_OR_RETURN(auto src0_handle,
                   model_builder->GetTensor(concat_inputs[0]->id));
  ASSIGN_OR_RETURN(auto src1_handle,
                   model_builder->GetTensor(concat_inputs[1]->id));
  ASSIGN_OR_RETURN(auto src2_handle,
                   model_builder->GetTensor(concat_inputs[2]->id));
  const auto& src0_td = src0_handle.tensor_desc;
  const auto& src1_td = src1_handle.tensor_desc;
  const auto& src2_td = src2_handle.tensor_desc;
  if (!src0_td.SupportsZeroClamp(Axis::WIDTH, gpu_info) ||
      !src0_td.SupportsZeroClamp(Axis::HEIGHT, gpu_info) ||
      !src1_td.SupportsZeroClamp(Axis::WIDTH, gpu_info) ||
      !src1_td.SupportsZeroClamp(Axis::HEIGHT, gpu_info) ||
      !src2_td.SupportsZeroClamp(Axis::WIDTH, gpu_info) ||
      !src2_td.SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }
  auto concat_output = graph.FindOutputs(concat_node->id)[0];
  auto concat_consumers = graph.FindConsumers(concat_output->id);
  if (concat_consumers.size() != 1) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }
  auto* conv_node = concat_consumers[0];
  if (conv_node == nullptr) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }
  if (OperationTypeFromString(conv_node->operation.type) !=
      OperationType::CONVOLUTION_2D) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }
  auto conv_inputs = graph.FindInputs(conv_node->id);
  auto conv_output = graph.FindOutputs(conv_node->id)[0];

  OperationDef op_def;
  op_def.src_tensors.push_back(src0_td);
  op_def.src_tensors.push_back(src1_td);
  op_def.src_tensors.push_back(src2_td);
  ASSIGN_OR_RETURN(auto dst_handle, model_builder->GetTensor(conv_output->id));
  op_def.dst_tensors.push_back(dst_handle.tensor_desc);

  auto concat_attr =
      std::any_cast<ConcatAttributes>(concat_node->operation.attributes);
  if (concat_attr.axis != Axis::CHANNELS) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }

  const auto& conv_attr =
      std::any_cast<Convolution2DAttributes&>(conv_node->operation.attributes);
  const auto& conv_weights = GetFloatWeights(conv_attr);

  if (conv_weights.shape != OHWI(4, 3, 3, 3) || conv_attr.strides != HW(1, 1) ||
      conv_attr.dilations != HW(1, 1) ||
      conv_attr.padding.prepended != HW(1, 1) ||
      conv_attr.padding.appended != HW(1, 1)) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }

  const DataType data_type = op_def.src_tensors[0].GetDataType();

  ThinLocalMemory operation;
  std::vector<float> weights_reordered(4 * 3 * 3 * 3);
  for (int i = 0; i < 3; ++i) {
    for (int ky = 0; ky < 3; ++ky) {
      for (int kx = 0; kx < 3; ++kx) {
        for (int o = 0; o < 4; ++o) {
          int w_index = conv_weights.shape.LinearIndex({o, ky, kx, i});
          float w_val = conv_weights.data[w_index];
          weights_reordered[((ky * 3 + kx) * 3 + i) * 4 + o] = w_val;
        }
      }
    }
  }
  AddConstantsGpuBuffer(gpu_info, data_type, weights_reordered,
                        &operation.args_);
  TensorDescriptor bias_tensor_desc =
      CreateConstantLinearTensorDescriptor(gpu_info, data_type, conv_attr.bias);
  operation.args_.AddObject(
      "bias", std::make_unique<TensorDescriptor>(std::move(bias_tensor_desc)));
  operation.AddSrcTensor("src_tensor0", op_def.src_tensors[0]);
  operation.AddSrcTensor("src_tensor1", op_def.src_tensors[1]);
  operation.AddSrcTensor("src_tensor2", op_def.src_tensors[2]);
  operation.AddDstTensor("dst_tensor", op_def.dst_tensors[0]);

  operation.tiling_type_ = ThinLocalMemory::TilingType::kWGSecondTensorPrecise;
  switch (operation.tiling_type_) {
    case ThinLocalMemory::TilingType::kWGFirstTensorPrecise: {
      // work_group_size_.x - 2 must be divisible by block_size.x
      // work_group_size_.y - 2 must be divisible by block_size.y
      operation.work_group_size_ = int3(16, 16, 1);
      operation.block_size_ = int2(2, 2);
      const bool x_div_by_block =
          conv_output->tensor.shape.w % operation.block_size_.x == 0;
      const bool y_div_by_block =
          conv_output->tensor.shape.h % operation.block_size_.y == 0;
      const bool need_write_check = !x_div_by_block || !y_div_by_block;
      operation.code_ = GetCodeWGFirstTensorPrecise(
          gpu_info, std::vector<Node*>{concat_node}, conv_weights.shape,
          operation.work_group_size_, data_type, operation.block_size_,
          need_write_check);
      break;
    }
    case ThinLocalMemory::TilingType::kWGSecondTensorPrecise: {
      operation.work_group_size_ = int3(16, 16, 1);
      operation.block_size_ = int2(2, 2);
      const int local_mem_size_x =
          operation.work_group_size_.x * operation.block_size_.x + 2;
      const int local_mem_size_y =
          operation.work_group_size_.y * operation.block_size_.y + 2;
      const int tiles_count = DivideRoundUp(
          local_mem_size_x * local_mem_size_y,
          operation.work_group_size_.x * operation.work_group_size_.y);
      operation.args_.AddFloat("tiles_count", tiles_count);
      const bool x_div_by_block =
          conv_output->tensor.shape.w % operation.block_size_.x == 0;
      const bool y_div_by_block =
          conv_output->tensor.shape.h % operation.block_size_.y == 0;
      const bool need_write_check = !x_div_by_block || !y_div_by_block;
      operation.code_ = GetCodeWGSecondTensorPrecise(
          gpu_info, std::vector<Node*>{concat_node}, conv_weights.shape,
          operation.work_group_size_, data_type, operation.block_size_,
          need_write_check);
      break;
    }
  }

  operation.flops_ =
      GetConvolutionFlops(conv_output->tensor.shape, conv_weights.shape);
  if (gpu_info.IsMali() || gpu_info.IsPowerVR()) {
    operation.compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
  const std::string op_name = absl::StrCat("convolution_2d (+ concat input) ",
                                           concat_node->id, " ", conv_node->id);
  model_builder->AddGpuOperation(
      std::vector<ValueId>(
          {concat_inputs[0]->id, concat_inputs[1]->id, concat_inputs[2]->id}),
      std::vector<ValueId>({conv_output->id}),
      std::make_unique<ThinLocalMemory>(std::move(operation)), op_name);

  new_consumed_nodes->insert(concat_node->id);
  new_consumed_nodes->insert(conv_node->id);
  return absl::OkStatus();
}

absl::Status TryResizeAddConvLocalMemoryFuser(
    const GpuInfo& gpu_info, const GraphFloat32& graph, NodeId first_node_id,
    const std::set<NodeId>& consumed_nodes,
    std::set<NodeId>* new_consumed_nodes, GpuModelBuilder* model_builder) {
  if (!IsResizeAddConvLocalMemoryFuserRecommended(gpu_info)) {
    // performance reason
    return absl::NotFoundError("ThinLocalMemoryFuser not recommended.");
  }
  auto* resize_node = graph.GetNode(first_node_id);
  if (resize_node == nullptr) {
    return absl::NotFoundError("ThinLocalMemoryFuser not suitable.");
  }
  if (OperationTypeFromString(resize_node->operation.type) !=
      OperationType::RESIZE) {
    return absl::NotFoundError("ThinLocalMemoryFuser not suitable.");
  }
  auto resize_input = graph.FindInputs(resize_node->id)[0];
  auto resize_output = graph.FindOutputs(resize_node->id)[0];
  auto resize_consumers = graph.FindConsumers(resize_output->id);
  if (resize_consumers.size() != 1) {
    return absl::NotFoundError("ThinLocalMemoryFuser not suitable.");
  }
  auto* add_node = resize_consumers[0];
  if (add_node == nullptr) {
    return absl::NotFoundError("ThinLocalMemoryFuser not suitable.");
  }
  if (OperationTypeFromString(add_node->operation.type) != OperationType::ADD) {
    return absl::NotFoundError("ThinLocalMemoryFuser not suitable.");
  }
  auto add_inputs = graph.FindInputs(add_node->id);
  auto add_second_input =
      add_inputs[0]->id == resize_output->id ? add_inputs[1] : add_inputs[0];
  auto add_output_id = graph.FindOutputs(add_node->id)[0]->id;
  auto add_consumers = graph.FindConsumers(add_output_id);
  if (add_consumers.size() != 1) {
    return absl::NotFoundError("ThinLocalMemoryFuser not suitable.");
  }
  auto* conv_node = add_consumers[0];
  if (conv_node == nullptr) {
    return absl::NotFoundError("ThinLocalMemoryFuser not suitable.");
  }
  if (OperationTypeFromString(conv_node->operation.type) !=
      OperationType::CONVOLUTION_2D) {
    return absl::NotFoundError("ThinLocalMemoryFuser not suitable.");
  }
  auto conv_output = graph.FindOutputs(conv_node->id)[0];

  ASSIGN_OR_RETURN(auto src0_handle,
                   model_builder->GetTensor(resize_input->id));
  ASSIGN_OR_RETURN(auto src1_handle,
                   model_builder->GetTensor(add_second_input->id));
  ASSIGN_OR_RETURN(auto dst_handle, model_builder->GetTensor(add_output_id));

  OperationDef op_def;
  op_def.src_tensors.push_back(src0_handle.tensor_desc);
  op_def.src_tensors.push_back(src1_handle.tensor_desc);
  op_def.dst_tensors.push_back(dst_handle.tensor_desc);

  auto resize_attr =
      std::any_cast<Resize2DAttributes>(resize_node->operation.attributes);
  auto conv_attr =
      std::any_cast<Convolution2DAttributes>(conv_node->operation.attributes);

  const auto& conv_weights = GetFloatWeights(conv_attr);
  bool supported_weights_shape = conv_weights.shape == OHWI(1, 3, 3, 4);
  if (conv_weights.shape == OHWI(4, 3, 3, 8)) {
    if (gpu_info.IsAdreno()) {
      supported_weights_shape = true;
    } else if (gpu_info.IsMali()) {
      const bool supported_mali =
          gpu_info.mali_info.generation >= MaliInfo::Gen::kValhallV4;
      const bool x_div_by_2 = conv_output->tensor.shape.w % 2 == 0;
      const bool y_div_by_2 = conv_output->tensor.shape.h % 2 == 0;
      if (supported_mali && x_div_by_2 && y_div_by_2) {
        supported_weights_shape = true;
      }
    } else if (gpu_info.IsApple()) {
      supported_weights_shape = true;
    } else if (gpu_info.IsPowerVR()) {
      supported_weights_shape = true;
    }
  }
  if (!supported_weights_shape || conv_attr.strides != HW(1, 1) ||
      conv_attr.dilations != HW(1, 1) ||
      conv_attr.padding.prepended != HW(1, 1) ||
      conv_attr.padding.appended != HW(1, 1)) {
    return absl::NotFoundError("ThinLocalMemoryFuser not suitable.");
  }

  const DataType data_type = op_def.src_tensors[0].GetDataType();
  ThinLocalMemory operation;
  if (conv_weights.shape == OHWI(4, 3, 3, 8)) {
    std::vector<float> weights_reordered(4 * 3 * 3 * 8);
    for (int i = 0; i < 8; ++i) {
      for (int ky = 0; ky < 3; ++ky) {
        for (int kx = 0; kx < 3; ++kx) {
          for (int o = 0; o < 4; ++o) {
            int w_index = conv_weights.shape.LinearIndex({o, ky, kx, i});
            float w_val = conv_weights.data[w_index];
            weights_reordered[((ky * 3 + kx) * 8 + i) * 4 + o] = w_val;
          }
        }
      }
    }
    AddConstantsGpuBuffer(gpu_info, data_type, weights_reordered,
                          &operation.args_);
  } else if (conv_weights.shape == OHWI(1, 3, 3, 4)) {
    AddConstantsGpuBuffer(gpu_info, data_type, conv_weights.data,
                          &operation.args_);
  }
  TensorDescriptor bias_tensor_desc =
      CreateConstantLinearTensorDescriptor(gpu_info, data_type, conv_attr.bias);
  operation.args_.AddObject(
      "bias", std::make_unique<TensorDescriptor>(std::move(bias_tensor_desc)));
  const float scale_factor_x = CalculateResizeScale(
      resize_input->tensor.shape.w, resize_output->tensor.shape.w, resize_attr);
  const float scale_factor_y = CalculateResizeScale(
      resize_input->tensor.shape.h, resize_output->tensor.shape.h, resize_attr);
  operation.args_.AddFloat("scale_factor_x", scale_factor_x);
  operation.args_.AddFloat("scale_factor_y", scale_factor_y);
  operation.tiling_type_ = ThinLocalMemory::TilingType::kWGFirstTensorPrecise;
  if (gpu_info.IsMali()) {
    operation.tiling_type_ =
        ThinLocalMemory::TilingType::kWGSecondTensorPrecise;
  }
  const int conv_i_slices = DivideRoundUp(conv_weights.shape.i, 4);
  if (conv_i_slices == 2 && gpu_info.IsApple()) {
    operation.tiling_type_ =
        ThinLocalMemory::TilingType::kWGSecondTensorPrecise;
  }
  operation.AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  operation.AddSrcTensor("add_tensor", op_def.src_tensors[1]);
  operation.AddDstTensor("dst_tensor", op_def.dst_tensors[0]);
  switch (operation.tiling_type_) {
    case ThinLocalMemory::TilingType::kWGFirstTensorPrecise: {
      // work_group_size_.x - 2 must be divisible by block_size.x
      // work_group_size_.y - 2 must be divisible by block_size.y
      operation.work_group_size_ = int3(32, 16, 1);
      operation.block_size_ = int2(1, 1);
      if (gpu_info.IsMali()) {
        if (conv_i_slices == 1) {
          operation.block_size_ = int2(2, 2);
          operation.work_group_size_ = int3(18, 18, 1);
        } else if (conv_i_slices == 2) {
          operation.block_size_ = int2(2, 1);
          operation.work_group_size_ = int3(18, 18, 2);
        }
      }
      if (gpu_info.IsAdreno()) {
        if (conv_i_slices == 1) {
          operation.block_size_ = int2(2, 2);
          operation.work_group_size_ = data_type == DataType::FLOAT16
                                           ? int3(32, 16, 1)
                                           : int3(16, 16, 1);
        } else if (conv_i_slices == 2) {
          operation.block_size_ = int2(2, 1);
          operation.work_group_size_ =
              data_type == DataType::FLOAT16 ? int3(18, 16, 2) : int3(18, 8, 2);
        }
      }
      if (gpu_info.IsPowerVR()) {
        if (conv_i_slices == 1) {
          operation.work_group_size_ = int3(18, 18, 1);
          operation.block_size_ = int2(2, 2);
        } else if (conv_i_slices == 2) {
          operation.work_group_size_ = int3(12, 10, 2);
          operation.block_size_ = int2(1, 1);
        }
      }
      const bool x_div_by_block =
          conv_output->tensor.shape.w % operation.block_size_.x == 0;
      const bool y_div_by_block =
          conv_output->tensor.shape.h % operation.block_size_.y == 0;
      const bool need_write_check = !x_div_by_block || !y_div_by_block;
      operation.code_ = GetCodeWGFirstTensorPrecise(
          gpu_info, std::vector<Node*>{resize_node, add_node},
          conv_weights.shape, operation.work_group_size_, data_type,
          operation.block_size_, need_write_check);
      break;
    }
    case ThinLocalMemory::TilingType::kWGSecondTensorPrecise: {
      operation.work_group_size_ = int3(32, 16, 1);
      operation.block_size_ = int2(1, 1);
      if (gpu_info.IsMali()) {
        if (gpu_info.mali_info.IsValhallGen2()) {
          operation.work_group_size_ = int3(16, 8, 1);
          operation.block_size_ = int2(2, 2);
        } else {
          if (conv_i_slices == 1) {
            operation.work_group_size_ = int3(16, 16, 1);
            operation.block_size_ = int2(2, 2);
          } else if (conv_i_slices == 2) {
            operation.block_size_ = int2(2, 2);
            operation.work_group_size_ = int3(16, 8, 1);
          }
        }
      }
      if (gpu_info.IsApple()) {
        operation.work_group_size_ = int3(16, 8, 1);
        operation.block_size_ = int2(2, 2);
      }
      const int local_mem_size_x =
          operation.work_group_size_.x * operation.block_size_.x + 2;
      const int local_mem_size_y =
          operation.work_group_size_.y * operation.block_size_.y + 2;
      const int tiles_count = DivideRoundUp(
          local_mem_size_x * local_mem_size_y * conv_i_slices,
          operation.work_group_size_.x * operation.work_group_size_.y);
      operation.args_.AddFloat("tiles_count", tiles_count);
      const bool x_div_by_block =
          conv_output->tensor.shape.w % operation.block_size_.x == 0;
      const bool y_div_by_block =
          conv_output->tensor.shape.h % operation.block_size_.y == 0;
      const bool need_write_check = !x_div_by_block || !y_div_by_block;
      operation.code_ = GetCodeWGSecondTensorPrecise(
          gpu_info, std::vector<Node*>{resize_node, add_node},
          conv_weights.shape, operation.work_group_size_, data_type,
          operation.block_size_, need_write_check);
      break;
    }
  }
  operation.flops_ =
      GetConvolutionFlops(conv_output->tensor.shape, conv_weights.shape);
  if (gpu_info.IsMali()) {
    operation.compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
  if (gpu_info.IsPowerVR()) {
    operation.compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }

  const std::string op_name =
      absl::StrCat("convolution_2d (+ resize input + add) ", resize_node->id,
                   " ", add_node->id, " ", conv_node->id);
  model_builder->AddGpuOperation(
      std::vector<ValueId>({resize_input->id, add_second_input->id}),
      std::vector<ValueId>({conv_output->id}),
      std::make_unique<ThinLocalMemory>(std::move(operation)), op_name);

  new_consumed_nodes->insert(resize_node->id);
  new_consumed_nodes->insert(add_node->id);
  new_consumed_nodes->insert(conv_node->id);
  return absl::OkStatus();
}

bool IsThinLocalMemoryFuserRecommended(const GpuInfo& gpu_info) {
  return IsConcatConvLocalMemoryFuserRecommended(gpu_info) ||
         IsResizeAddConvLocalMemoryFuserRecommended(gpu_info);
}

absl::Status TryThinLocalMemoryFuser(const GpuInfo& gpu_info,
                                     const GraphFloat32& graph,
                                     NodeId first_node_id,
                                     const std::set<NodeId>& consumed_nodes,
                                     std::set<NodeId>* new_consumed_nodes,
                                     GpuModelBuilder* model_builder) {
  if (TryConcatConvLocalMemoryFuser(gpu_info, graph, first_node_id,
                                    consumed_nodes, new_consumed_nodes,
                                    model_builder)
          .ok()) {
    return absl::OkStatus();
  } else if (TryResizeAddConvLocalMemoryFuser(gpu_info, graph, first_node_id,
                                              consumed_nodes,
                                              new_consumed_nodes, model_builder)
                 .ok()) {
    return absl::OkStatus();
  }
  return absl::NotFoundError("ThinLocalMemoryFuser not suitable.");
}

absl::Status TryConcatConvLocalMemoryFuser(
    const GpuInfo& gpu_info, const ir::IrModel& ir_model,
    ir::IrOpId first_op_id, const absl::flat_hash_set<ir::IrOpId>& consumed_ops,
    absl::flat_hash_set<ir::IrOpId>* new_consumed_ops,
    GpuModelBuilder* model_builder) {
  // experimental disabled version
  if (!IsConcatConvLocalMemoryFuserRecommended(gpu_info)) {
    // performance reason
    return absl::NotFoundError("ConcatConv not recommended.");
  }

  auto* concat_op = ir_model.op(first_op_id);
  if (concat_op == nullptr) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }
  if (OperationTypeFromString(concat_op->name) != OperationType::CONCAT) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }
  auto concat_inputs = concat_op->inputs;
  if (concat_inputs.size() != 3 ||
      ir_model.tensor(concat_inputs[0])->desc.GetBHWCShape().c != 1 ||
      ir_model.tensor(concat_inputs[1])->desc.GetBHWCShape().c != 1 ||
      ir_model.tensor(concat_inputs[2])->desc.GetBHWCShape().c != 1) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }
  ASSIGN_OR_RETURN(auto src0_handle,
                   model_builder->GetTensor(concat_inputs[0]));
  ASSIGN_OR_RETURN(auto src1_handle,
                   model_builder->GetTensor(concat_inputs[1]));
  ASSIGN_OR_RETURN(auto src2_handle,
                   model_builder->GetTensor(concat_inputs[2]));
  const auto& src0_td = src0_handle.tensor_desc;
  const auto& src1_td = src1_handle.tensor_desc;
  const auto& src2_td = src2_handle.tensor_desc;
  if (!src0_td.SupportsZeroClamp(Axis::WIDTH, gpu_info) ||
      !src0_td.SupportsZeroClamp(Axis::HEIGHT, gpu_info) ||
      !src1_td.SupportsZeroClamp(Axis::WIDTH, gpu_info) ||
      !src1_td.SupportsZeroClamp(Axis::HEIGHT, gpu_info) ||
      !src2_td.SupportsZeroClamp(Axis::WIDTH, gpu_info) ||
      !src2_td.SupportsZeroClamp(Axis::HEIGHT, gpu_info)) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }
  auto concat_output = concat_op->outputs[0];
  auto concat_consumers = ir_model.FindConsumers(concat_output);
  if (concat_consumers.size() != 1) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }
  auto* conv_op = concat_consumers[0];
  if (conv_op == nullptr) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }
  if (OperationTypeFromString(conv_op->name) != OperationType::CONVOLUTION_2D) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }
  auto conv_output = conv_op->outputs[0];

  OperationDef op_def;
  op_def.src_tensors.push_back(src0_td);
  op_def.src_tensors.push_back(src1_td);
  op_def.src_tensors.push_back(src2_td);
  ASSIGN_OR_RETURN(auto dst_handle, model_builder->GetTensor(conv_output));
  op_def.dst_tensors.push_back(dst_handle.tensor_desc);

  auto concat_attr = std::any_cast<ConcatAttributes>(concat_op->attr);
  if (concat_attr.axis != Axis::CHANNELS) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }

  const auto conv_attr = std::any_cast<Convolution2DAttributes>(conv_op->attr);
  const auto& conv_weights = GetFloatWeights(conv_attr);

  if (conv_weights.shape != OHWI(4, 3, 3, 3) || conv_attr.strides != HW(1, 1) ||
      conv_attr.dilations != HW(1, 1) ||
      conv_attr.padding.prepended != HW(1, 1) ||
      conv_attr.padding.appended != HW(1, 1)) {
    return absl::NotFoundError("ConcatConv not suitable.");
  }

  const DataType data_type = op_def.src_tensors[0].GetDataType();

  ThinLocalMemory operation;
  std::vector<float> weights_reordered(4 * 3 * 3 * 3);
  for (int i = 0; i < 3; ++i) {
    for (int ky = 0; ky < 3; ++ky) {
      for (int kx = 0; kx < 3; ++kx) {
        for (int o = 0; o < 4; ++o) {
          int w_index = conv_weights.shape.LinearIndex({o, ky, kx, i});
          float w_val = conv_weights.data[w_index];
          weights_reordered[((ky * 3 + kx) * 3 + i) * 4 + o] = w_val;
        }
      }
    }
  }
  AddConstantsGpuBuffer(gpu_info, data_type, weights_reordered,
                        &operation.args_);
  TensorDescriptor bias_tensor_desc =
      CreateConstantLinearTensorDescriptor(gpu_info, data_type, conv_attr.bias);
  operation.args_.AddObject(
      "bias", std::make_unique<TensorDescriptor>(std::move(bias_tensor_desc)));
  operation.AddSrcTensor("src_tensor0", op_def.src_tensors[0]);
  operation.AddSrcTensor("src_tensor1", op_def.src_tensors[1]);
  operation.AddSrcTensor("src_tensor2", op_def.src_tensors[2]);
  operation.AddDstTensor("dst_tensor", op_def.dst_tensors[0]);

  operation.tiling_type_ = ThinLocalMemory::TilingType::kWGSecondTensorPrecise;

  switch (operation.tiling_type_) {
    case ThinLocalMemory::TilingType::kWGFirstTensorPrecise: {
      operation.work_group_size_ = int3(16, 16, 1);
      operation.block_size_ = int2(2, 2);
      const bool x_div_by_block =
          ir_model.tensor(conv_output)->desc.GetBHWCShape().w %
              operation.block_size_.x ==
          0;
      const bool y_div_by_block =
          ir_model.tensor(conv_output)->desc.GetBHWCShape().h %
              operation.block_size_.y ==
          0;
      const bool need_write_check = !x_div_by_block || !y_div_by_block;
      operation.code_ = GetCodeWGFirstTensorPrecise(
          gpu_info, std::vector<const ir::IrOp*>{concat_op}, conv_weights.shape,
          operation.work_group_size_, data_type, operation.block_size_,
          need_write_check);
      break;
    }
    case ThinLocalMemory::TilingType::kWGSecondTensorPrecise: {
      operation.work_group_size_ = int3(16, 16, 1);
      operation.block_size_ = int2(2, 2);
      const int local_mem_size_x =
          operation.work_group_size_.x * operation.block_size_.x + 2;
      const int local_mem_size_y =
          operation.work_group_size_.y * operation.block_size_.y + 2;
      const int tiles_count = DivideRoundUp(
          local_mem_size_x * local_mem_size_y,
          operation.work_group_size_.x * operation.work_group_size_.y);
      operation.args_.AddFloat("tiles_count", tiles_count);
      const bool x_div_by_block =
          ir_model.tensor(conv_output)->desc.GetBHWCShape().w %
              operation.block_size_.x ==
          0;
      const bool y_div_by_block =
          ir_model.tensor(conv_output)->desc.GetBHWCShape().h %
              operation.block_size_.y ==
          0;
      const bool need_write_check = !x_div_by_block || !y_div_by_block;
      operation.code_ = GetCodeWGSecondTensorPrecise(
          gpu_info, std::vector<const ir::IrOp*>{concat_op}, conv_weights.shape,
          operation.work_group_size_, data_type, operation.block_size_,
          need_write_check);
      break;
    }
  }

  operation.flops_ = GetConvolutionFlops(
      ir_model.tensor(conv_output)->desc.GetBHWCShape(), conv_weights.shape);
  if (gpu_info.IsMali() || gpu_info.IsPowerVR()) {
    operation.compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
  const std::string op_name = absl::StrCat("convolution_2d (+ concat input) ",
                                           concat_op->id, " ", conv_op->id);
  model_builder->AddGpuOperation(
      std::vector<GpuModelBuilder::ValueId>(
          {static_cast<GpuModelBuilder::ValueId>(concat_inputs[0]),
           static_cast<GpuModelBuilder::ValueId>(concat_inputs[1]),
           static_cast<GpuModelBuilder::ValueId>(concat_inputs[2])}),
      std::vector<GpuModelBuilder::ValueId>(
          {static_cast<GpuModelBuilder::ValueId>(conv_output)}),
      std::make_unique<ThinLocalMemory>(std::move(operation)), op_name);

  new_consumed_ops->insert(concat_op->id);
  new_consumed_ops->insert(conv_op->id);
  return absl::OkStatus();
}

absl::Status TryResizeAddConvLocalMemoryFuser(
    const GpuInfo& gpu_info, const ir::IrModel& ir_model,
    ir::IrOpId first_op_id, const absl::flat_hash_set<ir::IrOpId>& consumed_ops,
    absl::flat_hash_set<ir::IrOpId>* new_consumed_ops,
    GpuModelBuilder* model_builder) {
  if (!IsResizeAddConvLocalMemoryFuserRecommended(gpu_info)) {
    return absl::NotFoundError("ThinLocalMemoryFuser not recommended.");
  }
  auto* resize_op = ir_model.op(first_op_id);
  if (resize_op == nullptr ||
      OperationTypeFromString(resize_op->name) != OperationType::RESIZE) {
    return absl::NotFoundError("ThinLocalMemoryFuser not suitable. 1");
  }
  auto resize_input = resize_op->inputs[0];
  auto resize_output = resize_op->outputs[0];
  auto resize_consumers = ir_model.FindConsumers(resize_output);
  if (resize_consumers.size() != 1) {
    return absl::NotFoundError("ThinLocalMemoryFuser not suitable. 2");
  }
  auto* add_op = resize_consumers[0];
  if (add_op == nullptr ||
      OperationTypeFromString(add_op->name) != OperationType::ADD) {
    return absl::NotFoundError("ThinLocalMemoryFuser not suitable. 3");
  }
  auto add_inputs = add_op->inputs;
  auto add_second_input =
      add_inputs[0] == resize_output ? add_inputs[1] : add_inputs[0];
  auto add_output_id = add_op->outputs[0];
  auto add_consumers = ir_model.FindConsumers(add_output_id);
  if (add_consumers.size() != 1) {
    return absl::NotFoundError("ThinLocalMemoryFuser not suitable. 4");
  }
  auto* conv_op = add_consumers[0];
  if (conv_op == nullptr ||
      OperationTypeFromString(conv_op->name) != OperationType::CONVOLUTION_2D) {
    return absl::NotFoundError("ThinLocalMemoryFuser not suitable. 5");
  }
  auto conv_output = conv_op->outputs[0];

  auto status1 = model_builder->GetTensor(resize_input);
  if (!status1.ok()) {
    return absl::NotFoundError("ThinLocalMemoryFuser not suitable. GetTensor1");
  }
  auto src0_handle = status1.value();

  auto status2 = model_builder->GetTensor(add_second_input);
  if (!status2.ok()) {
    return absl::NotFoundError("ThinLocalMemoryFuser not suitable. GetTensor2");
  }
  auto src1_handle = status2.value();

  auto status3 = model_builder->GetTensor(add_output_id);
  if (!status3.ok()) {
    return absl::NotFoundError("ThinLocalMemoryFuser not suitable. GetTensor3");
  }
  auto dst_handle = status3.value();

  OperationDef op_def;
  op_def.src_tensors.push_back(src0_handle.tensor_desc);
  op_def.src_tensors.push_back(src1_handle.tensor_desc);
  op_def.dst_tensors.push_back(dst_handle.tensor_desc);

  auto resize_attr = std::any_cast<Resize2DAttributes>(resize_op->attr);
  auto conv_attr = std::any_cast<Convolution2DAttributes>(conv_op->attr);

  const auto& conv_weights = GetFloatWeights(conv_attr);
  bool supported_weights_shape = conv_weights.shape == OHWI(1, 3, 3, 4);
  if (conv_weights.shape == OHWI(4, 3, 3, 8)) {
    if (gpu_info.IsAdreno()) {
      supported_weights_shape = true;
    } else if (gpu_info.IsMali()) {
      const bool supported_mali =
          gpu_info.mali_info.generation >= MaliInfo::Gen::kValhallV4;
      const bool x_div_by_2 =
          ir_model.tensor(conv_output)->desc.GetBHWCShape().w % 2 == 0;
      const bool y_div_by_2 =
          ir_model.tensor(conv_output)->desc.GetBHWCShape().h % 2 == 0;
      if (supported_mali && x_div_by_2 && y_div_by_2) {
        supported_weights_shape = true;
      }
    } else if (gpu_info.IsApple()) {
      supported_weights_shape = true;
    } else if (gpu_info.IsPowerVR()) {
      supported_weights_shape = true;
    }
  }
  if (!supported_weights_shape || conv_attr.strides != HW(1, 1) ||
      conv_attr.dilations != HW(1, 1) ||
      conv_attr.padding.prepended != HW(1, 1) ||
      conv_attr.padding.appended != HW(1, 1)) {
    return absl::NotFoundError("ThinLocalMemoryFuser not suitable.");
  }

  const DataType data_type = op_def.src_tensors[0].GetDataType();
  ThinLocalMemory operation;
  if (conv_weights.shape == OHWI(4, 3, 3, 8)) {
    std::vector<float> weights_reordered(4 * 3 * 3 * 8);
    for (int i = 0; i < 8; ++i) {
      for (int ky = 0; ky < 3; ++ky) {
        for (int kx = 0; kx < 3; ++kx) {
          for (int o = 0; o < 4; ++o) {
            int w_index = conv_weights.shape.LinearIndex({o, ky, kx, i});
            float w_val = conv_weights.data[w_index];
            weights_reordered[((ky * 3 + kx) * 8 + i) * 4 + o] = w_val;
          }
        }
      }
    }
    AddConstantsGpuBuffer(gpu_info, data_type, weights_reordered,
                          &operation.args_);
  } else if (conv_weights.shape == OHWI(1, 3, 3, 4)) {
    AddConstantsGpuBuffer(gpu_info, data_type, conv_weights.data,
                          &operation.args_);
  }
  TensorDescriptor bias_tensor_desc =
      CreateConstantLinearTensorDescriptor(gpu_info, data_type, conv_attr.bias);
  operation.args_.AddObject(
      "bias", std::make_unique<TensorDescriptor>(std::move(bias_tensor_desc)));
  const float scale_factor_x = CalculateResizeScale(
      ir_model.tensor(resize_input)->desc.GetBHWCShape().w,
      ir_model.tensor(resize_output)->desc.GetBHWCShape().w, resize_attr);
  const float scale_factor_y = CalculateResizeScale(
      ir_model.tensor(resize_input)->desc.GetBHWCShape().h,
      ir_model.tensor(resize_output)->desc.GetBHWCShape().h, resize_attr);
  operation.args_.AddFloat("scale_factor_x", scale_factor_x);
  operation.args_.AddFloat("scale_factor_y", scale_factor_y);
  operation.tiling_type_ = ThinLocalMemory::TilingType::kWGFirstTensorPrecise;
  if (gpu_info.IsMali()) {
    operation.tiling_type_ =
        ThinLocalMemory::TilingType::kWGSecondTensorPrecise;
  }
  const int conv_i_slices = DivideRoundUp(conv_weights.shape.i, 4);
  if (conv_i_slices == 2 && gpu_info.IsApple()) {
    operation.tiling_type_ =
        ThinLocalMemory::TilingType::kWGSecondTensorPrecise;
  }
  operation.AddSrcTensor("src_tensor", op_def.src_tensors[0]);
  operation.AddSrcTensor("add_tensor", op_def.src_tensors[1]);
  operation.AddDstTensor("dst_tensor", op_def.dst_tensors[0]);

  switch (operation.tiling_type_) {
    case ThinLocalMemory::TilingType::kWGFirstTensorPrecise: {
      operation.work_group_size_ = int3(32, 16, 1);
      operation.block_size_ = int2(1, 1);
      if (gpu_info.IsMali()) {
        if (conv_i_slices == 1) {
          operation.block_size_ = int2(2, 2);
          operation.work_group_size_ = int3(18, 18, 1);
        } else if (conv_i_slices == 2) {
          operation.block_size_ = int2(2, 1);
          operation.work_group_size_ = int3(18, 18, 2);
        }
      }
      if (gpu_info.IsAdreno()) {
        if (conv_i_slices == 1) {
          operation.block_size_ = int2(2, 2);
          operation.work_group_size_ = data_type == DataType::FLOAT16
                                           ? int3(32, 16, 1)
                                           : int3(16, 16, 1);
        } else if (conv_i_slices == 2) {
          operation.block_size_ = int2(2, 1);
          operation.work_group_size_ =
              data_type == DataType::FLOAT16 ? int3(18, 16, 2) : int3(18, 8, 2);
        }
      }
      if (gpu_info.IsPowerVR()) {
        if (conv_i_slices == 1) {
          operation.work_group_size_ = int3(18, 18, 1);
          operation.block_size_ = int2(2, 2);
        } else if (conv_i_slices == 2) {
          operation.work_group_size_ = int3(12, 10, 2);
          operation.block_size_ = int2(1, 1);
        }
      }
      const bool x_div_by_block =
          ir_model.tensor(conv_output)->desc.GetBHWCShape().w %
              operation.block_size_.x ==
          0;
      const bool y_div_by_block =
          ir_model.tensor(conv_output)->desc.GetBHWCShape().h %
              operation.block_size_.y ==
          0;
      const bool need_write_check = !x_div_by_block || !y_div_by_block;
      operation.code_ = GetCodeWGFirstTensorPrecise(
          gpu_info, std::vector<const ir::IrOp*>{resize_op, add_op},
          conv_weights.shape, operation.work_group_size_, data_type,
          operation.block_size_, need_write_check);
      break;
    }
    case ThinLocalMemory::TilingType::kWGSecondTensorPrecise: {
      operation.work_group_size_ = int3(32, 16, 1);
      operation.block_size_ = int2(1, 1);
      if (gpu_info.IsMali()) {
        if (gpu_info.mali_info.IsValhallGen2()) {
          operation.work_group_size_ = int3(16, 8, 1);
          operation.block_size_ = int2(2, 2);
        } else {
          if (conv_i_slices == 1) {
            operation.work_group_size_ = int3(16, 16, 1);
            operation.block_size_ = int2(2, 2);
          } else if (conv_i_slices == 2) {
            operation.block_size_ = int2(2, 2);
            operation.work_group_size_ = int3(16, 8, 1);
          }
        }
      }
      if (gpu_info.IsApple()) {
        operation.work_group_size_ = int3(16, 8, 1);
        operation.block_size_ = int2(2, 2);
      }
      const int local_mem_size_x =
          operation.work_group_size_.x * operation.block_size_.x + 2;
      const int local_mem_size_y =
          operation.work_group_size_.y * operation.block_size_.y + 2;
      const int tiles_count = DivideRoundUp(
          local_mem_size_x * local_mem_size_y * conv_i_slices,
          operation.work_group_size_.x * operation.work_group_size_.y);
      operation.args_.AddFloat("tiles_count", tiles_count);
      const bool x_div_by_block =
          ir_model.tensor(conv_output)->desc.GetBHWCShape().w %
              operation.block_size_.x ==
          0;
      const bool y_div_by_block =
          ir_model.tensor(conv_output)->desc.GetBHWCShape().h %
              operation.block_size_.y ==
          0;
      const bool need_write_check = !x_div_by_block || !y_div_by_block;
      operation.code_ = GetCodeWGSecondTensorPrecise(
          gpu_info, std::vector<const ir::IrOp*>{resize_op, add_op},
          conv_weights.shape, operation.work_group_size_, data_type,
          operation.block_size_, need_write_check);
      break;
    }
  }
  operation.flops_ = GetConvolutionFlops(
      ir_model.tensor(conv_output)->desc.GetBHWCShape(), conv_weights.shape);
  if (gpu_info.IsMali()) {
    operation.compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
  if (gpu_info.IsPowerVR()) {
    operation.compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }

  const std::string op_name =
      absl::StrCat("convolution_2d (+ resize input + add) ", resize_op->id, " ",
                   add_op->id, " ", conv_op->id);
  model_builder->AddGpuOperation(
      std::vector<GpuModelBuilder::ValueId>(
          {static_cast<GpuModelBuilder::ValueId>(resize_input),
           static_cast<GpuModelBuilder::ValueId>(add_second_input)}),
      std::vector<GpuModelBuilder::ValueId>(
          {static_cast<GpuModelBuilder::ValueId>(conv_output)}),
      std::make_unique<ThinLocalMemory>(std::move(operation)), op_name);

  new_consumed_ops->insert(resize_op->id);
  new_consumed_ops->insert(add_op->id);
  new_consumed_ops->insert(conv_op->id);
  return absl::OkStatus();
}

absl::Status TryThinLocalMemoryFuser(
    const GpuInfo& gpu_info, const ir::IrModel& ir_model,
    ir::IrOpId first_op_id, const absl::flat_hash_set<ir::IrOpId>& consumed_ops,
    absl::flat_hash_set<ir::IrOpId>* new_consumed_ops,
    GpuModelBuilder* model_builder) {
  if (TryConcatConvLocalMemoryFuser(gpu_info, ir_model, first_op_id,
                                    consumed_ops, new_consumed_ops,
                                    model_builder)
          .ok()) {
    return absl::OkStatus();
  } else if (TryResizeAddConvLocalMemoryFuser(gpu_info, ir_model, first_op_id,
                                              consumed_ops, new_consumed_ops,
                                              model_builder)
                 .ok()) {
    return absl::OkStatus();
  }
  return absl::NotFoundError("ThinLocalMemoryFuser not suitable.");
}

}  // namespace ml_drift
