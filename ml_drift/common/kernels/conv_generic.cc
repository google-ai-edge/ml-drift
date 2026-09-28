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

#include "ml_drift/common/kernels/conv_generic.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/kernels/fully_connected_util.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/compiler_options.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/util.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/task/work_group_picking.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {

namespace {
std::string GenerateUploadByThreads(
    const std::string& local_ptr_name, const std::string& name, bool use_ptrs,
    const std::string& global_offset_name, const std::string local_mem_type,
    const std::string& lid_name, int total_work_items, int elements_to_upload) {
  std::string c;
  std::string offset =
      global_offset_name.empty() ? "" : global_offset_name + " + ";
  const int groups = elements_to_upload / total_work_items;
  const int reminder = elements_to_upload % total_work_items;
  const std::string access_start = name + (use_ptrs ? "[" : ".Read(");
  const std::string access_end = use_ptrs ? "]" : ")";
  for (int i = 0; i < groups; ++i) {
    const std::string value = access_start + offset + lid_name + " + " +
                              std::to_string(total_work_items * i) + access_end;
    c += "    " + local_ptr_name + "[" + lid_name + " + " +
         std::to_string(total_work_items * i) + "] = ucl::Convert<" +
         local_mem_type + ">(" + value + ");\n";
  }
  if (reminder != 0) {
    const std::string value = access_start + offset + lid_name + " + " +
                              std::to_string(total_work_items * groups) +
                              access_end;
    c += "    if (" + lid_name + " < " + std::to_string(reminder) + ") {\n";
    c += "      " + local_ptr_name + "[" + lid_name + " + " +
         std::to_string(total_work_items * groups) + "] = ucl::Convert<" +
         local_mem_type + ">(" + value + ");\n";
    c += "    }\n";
  }
  return c;
}

std::string GenerateAsyncUpload(const std::string& local_ptr_name,
                                const std::string& global_ptr_name,
                                const std::string& global_offset_name,
                                int elements_to_upload) {
  std::string c;
  std::string offset =
      global_offset_name.empty() ? "" : " + " + global_offset_name;
  c += "    async_work_group_copy(" + local_ptr_name + ", " + global_ptr_name +
       offset + ", " + std::to_string(elements_to_upload) + ", 0);\n";
  return c;
}

std::string GenerateBlockCoords(const ConvGeneric::KernelParams& kernel_params,
                                const TensorDescriptor& src_desc) {
  bool spatial_uniform = false;
  if (kernel_params.fixed_work_group_size) {
    if (kernel_params.linear_all) {
      spatial_uniform = false;
    } else if (kernel_params.linear_spatial) {
      spatial_uniform = kernel_params.work_group_size.y == 1;
    } else {
      spatial_uniform = kernel_params.work_group_size.z == 1;
    }
  }
  std::string c;
  int3 launch_remap;
  launch_remap[kernel_params.work_group_launch_order.x] = 0;
  launch_remap[kernel_params.work_group_launch_order.y] = 1;
  launch_remap[kernel_params.work_group_launch_order.z] = 2;
  if (kernel_params.linear_all) {
    c += "  int linear_all = ucl::GetGlobalId<0>();\n";
    if (src_desc.HasAxis(Axis::kBatch)) {
      c += "  int B = linear_all % args.task_size_b;\n";
      c += "  linear_all = linear_all / args.task_size_b;\n";
    }
    if (kernel_params.linear_all_slices_first) {
      c += "  int DST_S = linear_all % args.task_size_s;\n";
      c += "  linear_all = linear_all / args.task_size_s;\n";
    }
    c += "  int DST_X = linear_all % args.task_size_x;\n";
    c += "  linear_all = linear_all / args.task_size_x;\n";
    c += "  int DST_Y = linear_all % args.task_size_y;\n";
    c += "  linear_all = linear_all / args.task_size_y;\n";
    if (src_desc.HasAxis(Axis::kDepth)) {
      c += "  int DST_Z = linear_all % args.task_size_z;\n";
      c += "  linear_all = linear_all / args.task_size_z;\n";
    }
    if (!kernel_params.linear_all_slices_first) {
      c += "  int DST_S = linear_all;\n";
    }
  } else if (kernel_params.linear_spatial) {
    if (kernel_params.work_group_launch_order[0] == 0) {
      c += "  int linear_spatial = ucl::GetGlobalId<0>();\n";
    } else {
      c += "  int linear_spatial = ucl::GetGroupId<" +
           std::to_string(launch_remap[0]) +
           ">() * ucl::GetGroupSize<0>() + ucl::GetLocalId<0>();\n";
    }
    if (src_desc.HasAxis(Axis::kBatch)) {
      c += "  int B = linear_spatial % args.task_size_b;\n";
      c += "  linear_spatial = linear_spatial / args.task_size_b;\n";
    }
    c += "  int DST_X = linear_spatial % args.task_size_x;\n";
    c += "  linear_spatial = linear_spatial / args.task_size_x;\n";
    c += "  int DST_Y = linear_spatial % args.task_size_y;\n";
    c += "  linear_spatial = linear_spatial / args.task_size_y;\n";
    if (src_desc.HasAxis(Axis::kDepth)) {
      c += "  int DST_Z = linear_spatial;\n";
    }
    if (kernel_params.work_group_launch_order[1] == 1) {
      if (spatial_uniform) {
        c += "  int DST_S = ucl::GetGroupId<1>();\n";
      } else {
        c += "  int DST_S = ucl::GetGlobalId<1>();\n";
      }
    } else {
      if (spatial_uniform) {
        c += "  int DST_S = ucl::GetGroupId<" +
             std::to_string(launch_remap[1]) + ">();\n";
      } else {
        c += "  int DST_S = ucl::GetGroupId<" +
             std::to_string(launch_remap[1]) +
             ">() * ucl::GetGroupSize<1>() + ucl::GetLocalId<1>();\n";
      }
    }
  } else {
    if (kernel_params.work_group_launch_order[0] == 0) {
      c += "  int DST_X = ucl::GetGlobalId<0>();\n";
    } else {
      c += "  int DST_X = ucl::GetGroupId<" + std::to_string(launch_remap[0]) +
           ">() * ucl::GetGroupSize<0>() + ucl::GetLocalId<0>();\n";
    }
    if (src_desc.HasAxis(Axis::kBatch)) {
      c += "  int B = DST_X % args.task_size_b;\n";
      c += "  DST_X = DST_X / args.task_size_b;\n";
    }
    std::string global_id_1;
    if (kernel_params.work_group_launch_order[1] == 1) {
      global_id_1 = "ucl::GetGlobalId<1>()";
    } else {
      global_id_1 = "ucl::GetGroupId<" + std::to_string(launch_remap[1]) +
                    ">() * ucl::GetGroupSize<1>() + ucl::GetLocalId<1>()";
    }
    if (src_desc.HasAxis(Axis::kDepth)) {
      c += "  int linear_id_1 = " + global_id_1 + ";\n";
      c += "  int DST_Y = linear_id_1 % args.task_size_y;\n";
      c += "  int DST_Z = linear_id_1 / args.task_size_y;\n";
    } else {
      c += "  int DST_Y = " + global_id_1 + ";\n";
    }
    if (kernel_params.work_group_launch_order[2] == 2) {
      if (spatial_uniform) {
        c += "  int DST_S = ucl::GetGroupId<2>();\n";
      } else {
        c += "  int DST_S = ucl::GetGlobalId<2>();\n";
      }
    } else {
      if (spatial_uniform) {
        c += "  int DST_S = ucl::GetGroupId<" +
             std::to_string(launch_remap[2]) + ">();\n";
      } else {
        c += "  int DST_S = ucl::GetGroupId<" +
             std::to_string(launch_remap[2]) +
             ">() * ucl::GetGroupSize<2>() + ucl::GetLocalId<2>();\n";
      }
    }
  }
  if (kernel_params.block_size.x != 1) {
    c += "  DST_X *= " + std::to_string(kernel_params.block_size.x) + ";\n";
  }
  if (kernel_params.block_size.y != 1) {
    c += "  DST_Y *= " + std::to_string(kernel_params.block_size.y) + ";\n";
  }
  if (src_desc.HasAxis(Axis::kDepth) && kernel_params.block_size.z != 1) {
    c += "  DST_Z *= " + std::to_string(kernel_params.block_size.z) + ";\n";
  }
  if (kernel_params.block_size.w != 1) {
    c += "  DST_S *= " + std::to_string(kernel_params.block_size.w) + ";\n";
  }

  return c;
}

std::string GetWorkGroupBaseDstX(const int3& work_group_launch_order) {
  int3 launch_remap;
  launch_remap[work_group_launch_order.x] = 0;
  launch_remap[work_group_launch_order.y] = 1;
  launch_remap[work_group_launch_order.z] = 2;
  return "ucl::GetGroupId<" + std::to_string(launch_remap[0]) +
         ">() * ucl::GetGroupSize<0>()";
}

std::string GetWorkGroupBaseDstY(const int3& work_group_launch_order) {
  int3 launch_remap;
  launch_remap[work_group_launch_order.x] = 0;
  launch_remap[work_group_launch_order.y] = 1;
  launch_remap[work_group_launch_order.z] = 2;
  return "ucl::GetGroupId<" + std::to_string(launch_remap[1]) +
         ">() * ucl::GetGroupSize<1>()";
}

bool SupportsImgMatMul(const GpuInfo& gpu_info,
                       const ConvGeneric::ConvParams& conv_params) {
  return !conv_params.Is8Bit() && gpu_info.IsApiOpenCl() &&
         gpu_info.IsPowerVR() && gpu_info.powervr_info.IsImgCxx();
}

class ConvCodeGenerator {
 public:
  ConvCodeGenerator(const GpuInfo& gpu_info,
                    const ConvGeneric::ConvParams& conv_params,
                    const ConvGeneric::KernelParams& kernel_params)
      : gpu_info_(gpu_info),
        conv_params_(conv_params),
        kernel_params_(kernel_params) {}

  std::string GenerateConv() {
    const bool use_intel_wave_matmul =
        kernel_params_.weights_upload_type ==
        ConvGeneric::WeightsUploadType::kIntelWave16MatMul;

    std::string c;
    if (kernel_params_.weights_upload_type ==
        ConvGeneric::WeightsUploadType::kWaveMemory) {
      c += "#pragma OPENCL EXTENSION ucl_wave_memory: enable\n";
      if (!kernel_params_.simd_sizes.empty()) {
        c += "__attribute__((ucl_wave_memory_sizes(" +
             std::to_string(kernel_params_.simd_sizes[0]);
        for (int i = 1; i < kernel_params_.simd_sizes.size(); ++i) {
          c += ", " + std::to_string(kernel_params_.simd_sizes[i]);
        }
        c += ")))\n";
      }
    }
    if (conv_params_.Is8Bit()) {
      if (gpu_info_.SupportsExtension("cl_qcom_dot_product8")) {
        c += "#pragma OPENCL EXTENSION cl_qcom_dot_product8 : enable\n";
      } else if (gpu_info_.SupportsExtension("cl_khr_integer_dot_product")) {
        c += "#pragma OPENCL EXTENSION "
             "cl_khr_integer_dot_product : enable\n";
      } else if (gpu_info_.SupportsExtension(
                     "cl_arm_integer_dot_product_accumulate_int8")) {
        c += "#pragma OPENCL EXTENSION "
             "cl_arm_integer_dot_product_accumulate_int8 : enable\n";
      } else if (gpu_info_.SupportsExtension(
                     "VK_KHR_shader_integer_dot_product")) {
        c += "#extension GL_EXT_integer_dot_product : require\n";
      }
      if (gpu_info_.IsApiWebGpu() && gpu_info_.SupportsAcceleratedDp4a()) {
        c += "requires packed_4x8_integer_dot_product;\n";
      }
    }
    if (use_intel_wave_matmul) {
      c += "#pragma OPENCL EXTENSION "
           "cl_intel_subgroup_matrix_multiply_accumulate : enable\n";
    }
    if (kernel_params_.fixed_work_group_size && gpu_info_.IsApiOpenCl()) {
      c += "__attribute__((reqd_work_group_size(" +
           std::to_string(kernel_params_.work_group_size.x) + ", " +
           std::to_string(kernel_params_.work_group_size.y) + ", " +
           std::to_string(kernel_params_.work_group_size.z) + ")))\n";
    }
    if (use_intel_wave_matmul && kernel_params_.simd_sizes.size() == 1 &&
        gpu_info_.IsApiOpenCl() &&
        gpu_info_.SupportsExtension("cl_intel_required_subgroup_size")) {
      c += "__attribute__((intel_reqd_sub_group_size(" +
           std::to_string(kernel_params_.simd_sizes[0]) + ")))\n";
    }
    c += "MAIN_FUNCTION($0) {\n";
    c += GenerateMainBody();
    c += "}\n";

    const DataType type = DeduceDataTypeFromPrecision(conv_params_.precision);
    absl::StrReplaceAll(
        {
            {"ScalarType", ToUclDataType(type, 1)},
            {"Type", ToUclDataType(type, 4)},
            {"AccType", ToUclDataType(GetAccumulatorType(), 4)},
            {"AccSType", ToUclDataType(GetAccumulatorType(), 1)},
        },
        &c);

    return c;
  }

 private:
  std::string GenerateMainBody() {
    const int4 block_size = kernel_params_.block_size;
    const auto& src_def = conv_params_.src_desc;
    const bool need_local_mem = NeedsLocalMemory();
    const bool is_wave_memory = NeedsWaveMemory();
    const int weights_vals_per_loop = GetWeightsValsPerLoop();
    const bool use_intel_wave_matmul =
        kernel_params_.weights_upload_type ==
        ConvGeneric::WeightsUploadType::kIntelWave16MatMul;
    const bool late_oob_check =
        need_local_mem || is_wave_memory || use_intel_wave_matmul;
    // weights_conversion currently only supported for Intel Wave MatMul
    const bool weights_conversion =
        conv_params_.weights_desc.layout != WeightsLayout::kUnknown;
    const bool quantized_weights =
        weights_conversion && SizeInBitsOf(conv_params_.weights_desc.type) <= 8;
    std::string c;

    std::string dst_oob_check;
    if (src_def.HasAxis(Axis::kDepth)) {
      if (kernel_params_.linear_all) {
        dst_oob_check = "DST_S >= args.dst_tensor.Slices()";
      } else if (kernel_params_.linear_spatial) {
        dst_oob_check =
            "DST_Z >= args.dst_tensor.Depth() || DST_S >= "
            "args.dst_tensor.Slices()";
      } else {
        dst_oob_check =
            "DST_X >= args.dst_tensor.Width() || DST_Z >= "
            "args.dst_tensor.Depth() || DST_S >= args.dst_tensor.Slices()";
      }
    } else {
      if (kernel_params_.linear_all) {
        dst_oob_check = "DST_S >= args.dst_tensor.Slices()";
      } else if (kernel_params_.linear_spatial) {
        dst_oob_check =
            "DST_Y >= args.dst_tensor.Height() || DST_S >= "
            "args.dst_tensor.Slices()";
      } else {
        dst_oob_check =
            "DST_X >= args.dst_tensor.Width() || DST_Y >= "
            "args.dst_tensor.Height() || DST_S >= args.dst_tensor.Slices()";
      }
    }

    c += GenerateBlockCoords(kernel_params_, src_def);
    if (src_def.HasAxis(Axis::kBatch)) {
      c += "  args.src_tensor.SetBatchRef(B);\n";
      c += "  args.dst_tensor.SetBatchRef(B);\n";
    }
    if (!kernel_params_.need_dst_loop) {
      c += "  DST_S = 0;\n";
    }
    c += "  if (DST_S >= args.dst_tensor.Slices()) return;\n";
    if (conv_params_.runtime_check.dst_end_ch_index.has_value()) {
      c += "  int dst_end_slice = " +
           conv_params_.runtime_check.GetRuntimeEndSlice(
               "args.params.Read(args.dst_end_ch_index)",
               "args.dst_tensor.Slices()") +
           ";\n";
      c += "  if (DST_S >= dst_end_slice) return;\n";
    }
    if (conv_params_.runtime_check.packed_groups.has_value()) {
      if (gpu_info_.IsApiWebGpu()) {
        c += "  int w_batch_id = " +
             GetWorkGroupBaseDstY(kernel_params_.work_group_launch_order) +
             ";\n";
      } else {
        c += "  int w_batch_id = DST_Y;\n";
      }
      c += "  DST_Y = 0;\n";
      c += "  int w_group_size = args.params.Read(args.packed_params_offset + "
           "w_batch_id);\n";
      c +=
          "  int w_group_offset = args.params.Read(args.packed_params_offset + "
          "w_batch_id + " +
          std::to_string(conv_params_.runtime_check.packed_groups->num_groups) +
          ");\n";
      if (kernel_params_.weights_upload_type ==
              ConvGeneric::WeightsUploadType::kLocalMemory ||
          kernel_params_.weights_upload_type ==
              ConvGeneric::WeightsUploadType::kLocalMemoryWGLoad) {
        c += "  int tile_first_w = " +
             GetWorkGroupBaseDstX(kernel_params_.work_group_launch_order) +
             " * " + std::to_string(kernel_params_.block_size.x) + ";\n";
      } else {
        int tile_size = 1;
        if (use_intel_wave_matmul) {
          tile_size = weights_conversion ? kernel_params_.work_group_size.x
                                         : kernel_params_.simd_sizes[0];
        } else if (kernel_params_.weights_upload_type ==
                   ConvGeneric::WeightsUploadType::kWaveMemory) {
          tile_size = kernel_params_.simd_sizes.back();
        }
        tile_size *= kernel_params_.block_size.x;
        const std::string tile_size_str = std::to_string(tile_size);
        c += "  int tile_first_w = (DST_X / " + tile_size_str + ") * " +
             tile_size_str + ";\n";
      }
      c += "  if (tile_first_w >= w_group_size) return;\n";
      c += "  DST_X = w_group_offset + DST_X;\n";
    } else {
      c += "  int w_batch_id = DST_Y;\n";
    }
    if (!late_oob_check) {
      c += "  if (" + dst_oob_check + ") {\n";
      c += "    return;\n";
      c += "  }\n";
    }
    std::string src_group_start_slice = "0";
    std::string src_group_end_slice = "args.src_tensor.Slices()";
    std::string src_group_slices = "args.src_tensor.Slices()";
    if (conv_params_.HasGroups()) {
      c += "      int conv_group_id = DST_S / args.dst_group_size;\n";
      c += "      int src_start_slice = conv_group_id * args.src_group_size;\n";
      c += "      int src_end_slice = src_start_slice + args.src_group_size;\n";
      src_group_start_slice = "src_start_slice";
      src_group_end_slice = "src_end_slice";
      src_group_slices = "args.src_group_size";
    }
    if (conv_params_.runtime_check.src_end_ch_index.has_value()) {
      c += "  int src_slices_dynamic = " +
           conv_params_.runtime_check.GetRuntimeEndSlice(
               "args.params.Read(args.src_end_ch_index)",
               "args.src_tensor.Slices()") +
           ";\n";
      src_group_end_slice = "src_slices_dynamic";
    }
    if (kernel_params_.weights_upload_type ==
        ConvGeneric::WeightsUploadType::kLocalMemory) {
      if (kernel_params_.linear_spatial) {
        c += "  int lid = ucl::GetLocalId<0>();\n";
      } else {
        c += "  int lid = ucl::GetLocalId<1>() * " +
             std::to_string(kernel_params_.work_group_size.x) +
             " + ucl::GetLocalId<0>();\n";
      }
    }
    if (use_intel_wave_matmul) {
      c += "  int simd_id = ucl::GetSubGroupLocalId();\n";
      if (weights_conversion) {
        ABSL_CHECK(kernel_params_.work_group_size.x ==
                   kernel_params_.block_size.w * 4);
        // WeightsLayout::kBIOI4O4
        const int cache_size = 16 * kernel_params_.work_group_size.x / 4;
        c += "  __local half4 w_cache[" + std::to_string(cache_size) + "];\n";
        c += "  int sub_i = ucl::GetLocalId<0>() / " +
             std::to_string(kernel_params_.block_size.w) + ";\n";
        c += "  int sub_o = ucl::GetLocalId<0>() % " +
             std::to_string(kernel_params_.block_size.w) + ";\n";
        c += "  int w_o_slice = min(DST_S + sub_o, args.dst_tensor.Slices() - "
             "1);\n";
        if (conv_params_.weights_desc.IsLinearLayout()) {
          const std::string batch_part =
              conv_params_.different_weights_for_height
                  ? "w_batch_id * args.src_tensor.Slices()"
                  : "0";
          c += "  int w_wg_offset = (" + batch_part +
               " + sub_i) * args.dst_tensor.Slices() + w_o_slice;\n";
          c += "  int w_wg_stride = args.dst_tensor.Slices() * 4;\n";
        }
        if (quantized_weights) {
          c += "  Type w_scale, w_bias;\n";
          if (conv_params_.scale_zp_shape.i != 1) {
            // grouped quantization
            c += "  int last_src_group_id = -1;\n";
          } else if (conv_params_.scale_zp_shape.o != 1) {
            // linear quantization
            c += fc::ReadScaleZeroPointLinear(
                "w_o_slice", conv_params_.scale_zp_shape,
                conv_params_.has_zero_point,
                fc::GetDataTypeForWeights(conv_params_.weights_desc.type));
          } else {
            // scalar quantization
            c += fc::ReadScaleZeroPointScalar(
                conv_params_.has_zero_point,
                fc::GetDataTypeForWeights(conv_params_.weights_desc.type));
          }
        }
      }
    }
    const bool use_vec8 = UseVec8Accumulator();
    const std::string acc_type =
        ToUclDataType(GetAccumulatorType(), use_vec8 ? 8 : 4);
    const std::string zero_value = GetZeroValue(GetAccumulatorType());
    for (int s = 0; s < block_size.w / (use_vec8 ? 2 : 1); ++s) {
      ForSpatial([&](int x, int y, int z) {
        c += "  " + acc_type + " r" + GenerateIdFull(x, y, z, s) +
             " = ucl::Init<" + acc_type + ">(" + zero_value + ");\n";
      });
    }
    if (!conv_params_.IsXKernelIs1()) {
      for (int x = 0; x < block_size.x; ++x) {
        const std::string xind = std::to_string(x);
        const std::string xc = "(DST_X + " + xind + ")";
        c += "  int xc" + xind + " = " + xc +
             " * args.stride_x + args.padding_x;\n";
      }
    } else {
      for (int x = 0; x < block_size.x; ++x) {
        const std::string xind = std::to_string(x);
        if (conv_params_.IsSrcPackedW4C4()) {
          c += "  int xc" + xind + " = DST_X / 4 + " + std::to_string(x / 4) +
               ";\n";
        } else {
          c += "  int xc" + xind + " = DST_X + " + xind + ";\n";
        }
        if (!src_def.CanReadOutOfBorder(Axis::kWidth, gpu_info_)) {
          c += "  xc" + xind + " = clamp(xc" + xind +
               ", 0, args.src_tensor.Width() - 1);\n";
        }
      }
    }
    if (!conv_params_.IsYKernelIs1()) {
      for (int y = 0; y < block_size.y; ++y) {
        const std::string yind = std::to_string(y);
        const std::string yc = "(DST_Y + " + yind + ")";
        c += "  int yc" + yind + " = " + yc +
             " * args.stride_y + args.padding_y;\n";
      }
    } else {
      for (int y = 0; y < block_size.y; ++y) {
        const std::string yind = std::to_string(y);
        c += "  int yc" + yind + " = DST_Y + " + yind + ";\n";
        if (!src_def.CanReadOutOfBorder(Axis::kHeight, gpu_info_)) {
          c += "  yc" + yind + " = clamp(yc" + yind +
               ", 0, args.src_tensor.Height() - 1);\n";
        }
      }
    }
    if (src_def.HasAxis(Axis::kDepth)) {
      if (!conv_params_.IsZKernelIs1()) {
        for (int z = 0; z < block_size.z; ++z) {
          const std::string zind = std::to_string(z);
          const std::string zc = "(DST_Z + " + zind + ")";
          c += "  int zc" + zind + " = " + zc +
               " * args.stride_z + args.padding_z;\n";
        }
      } else {
        for (int z = 0; z < block_size.z; ++z) {
          const std::string zind = std::to_string(z);
          c += "  int zc" + zind + " = DST_Z + " + zind + ";\n";
          if (!src_def.CanReadOutOfBorder(Axis::kDepth, gpu_info_)) {
            c += "  zc" + zind + " = clamp(zc" + zind +
                 ", 0, args.src_tensor.Depth() - 1);\n";
          }
        }
      }
    }
    const std::string weights_buffer_type = ToUclDataType(
        kernel_params_.weights_type, kernel_params_.weights_element_size);
    const std::string weights_cache_type = GetWeightsCacheType();
    const std::string weights_space =
        kernel_params_.weights_upload_type ==
                ConvGeneric::WeightsUploadType::kConstantMemory
            ? "__constant"
            : "__global";
    const std::string weights_global_ptr =
        weights_space + " " + weights_buffer_type + "*";
    if (need_local_mem) {
      c += "  __local " + weights_cache_type + " weights_cache[" +
           std::to_string(weights_vals_per_loop) + "];\n";
    } else if (is_wave_memory) {
      c += "  __wave " + weights_cache_type + " weights_cache[" +
           std::to_string(weights_vals_per_loop) + "];\n";
    } else if (kernel_params_.AreWeightsBuffer() &&
               gpu_info_.SupportsPointersInKernels()) {
      c += "  " + weights_global_ptr + " weights_cache;\n";
    } else if (!kernel_params_.AreWeightsBuffer() &&
               !conv_params_.IsTrivialKernelSize()) {
      c += "  int weights_offset = 0;\n";
    }
    std::string weights_offset;
    if (kernel_params_.AreWeightsBuffer() && !weights_conversion) {
      if (conv_params_.different_weights_for_height) {
        if (conv_params_.runtime_check.packed_groups.has_value()) {
          weights_offset =
              "(DST_S * " +
              std::to_string(
                  conv_params_.runtime_check.packed_groups->num_groups) +
              " + w_batch_id";
        } else {
          weights_offset = "(DST_S * args.src_tensor.Height() + DST_Y";
        }
        weights_offset += " * " + std::to_string(block_size.w) +
                          ") * 4 * args.src_tensor.Slices()";
      } else {
        std::string kernel_spatial_offset = "";
        if (!conv_params_.IsXKernelIs1()) {
          kernel_spatial_offset += " * args.kernel_size_x";
        }
        if (!conv_params_.IsYKernelIs1()) {
          kernel_spatial_offset += " * args.kernel_size_y";
        }
        if (src_def.HasAxis(Axis::kDepth) && !conv_params_.IsZKernelIs1()) {
          kernel_spatial_offset += " * args.kernel_size_z";
        }
        weights_offset =
            "DST_S * 4 * " + src_group_slices + kernel_spatial_offset;
      }
      if (conv_params_.IsSrcPackedC16() || conv_params_.IsSrcPackedC32()) {
        weights_offset += " * 4";
      }
      if (conv_params_.Is8Bit() && kernel_params_.weights_element_size == 4) {
        // 8-bit weights as uint32x4
        weights_offset += " / 4";
      }
      if (gpu_info_.SupportsPointersInKernels()) {
        c += "  " + weights_global_ptr +
             " weights_offset = args.weights.GetPtr() + " + weights_offset +
             ";\n";
      } else {
        c += "  int weights_offset = " + weights_offset + ";\n";
      }
    }
    if (!kernel_params_.slices_loop_first) {
      c += "  int s = " + src_group_start_slice + ";\n";
      if (kernel_params_.need_src_loop) {
        c += "  do {\n";
      }
    }
    if (src_def.HasAxis(Axis::kDepth) && !conv_params_.IsZKernelIs1()) {
      c += "  for (int kz = 0; kz < args.kernel_size_z; ++kz) {\n";
      for (int z = 0; z < block_size.z; ++z) {
        const std::string zck = "zck" + std::to_string(z);
        c += "  int zck" + std::to_string(z) + " = kz * args.dilation_z + zc" +
             std::to_string(z) + ";\n";
        if (!src_def.SupportsZeroClamp(Axis::kDepth, gpu_info_)) {
          c += "  bool in_z" + std::to_string(z) + " = " + zck + " >= 0 && " +
               zck + " < args.src_tensor.Depth();\n";
          if (!src_def.CanReadOutOfBorder(Axis::kDepth, gpu_info_)) {
            c += "  " + zck + " = clamp(" + zck +
                 ", 0, args.src_tensor.Depth() - 1);\n";
          }
        }
      }
    }
    const bool do_while_loop_for_y =
        kernel_params_.unroll_x_loop && !kernel_params_.slices_loop_first;
    if (!conv_params_.IsYKernelIs1()) {
      if (do_while_loop_for_y) {
        c += "  int ky = 0;\n";
        c += "  do {\n";
      } else {
        c += "  for (int ky = 0; ky < args.kernel_size_y; ++ky) {\n";
      }
      for (int y = 0; y < block_size.y; ++y) {
        const std::string yck = "yck" + std::to_string(y);
        c += "  int " + yck + " = ky * args.dilation_y + yc" +
             std::to_string(y) + ";\n";
        if (!src_def.SupportsZeroClamp(Axis::kHeight, gpu_info_)) {
          c += "  bool in_y" + std::to_string(y) + " = " + yck + " >= 0 && " +
               yck + " < args.src_tensor.Height();\n";
          if (!src_def.CanReadOutOfBorder(Axis::kHeight, gpu_info_)) {
            c += "  " + yck + " = clamp(" + yck +
                 ", 0, args.src_tensor.Height() - 1);\n";
          }
        }
      }
    }
    if (!kernel_params_.unroll_x_loop && !conv_params_.IsXKernelIs1()) {
      c += "  for (int kx = 0; kx < args.kernel_size_x; ++kx) {\n";
      for (int x = 0; x < block_size.x; ++x) {
        const std::string xck = "xck" + std::to_string(x);
        c += "  int xck" + std::to_string(x) + " = kx * args.dilation_x + xc" +
             std::to_string(x) + ";\n";
        if (!src_def.SupportsZeroClamp(Axis::kWidth, gpu_info_)) {
          c += "  bool in_x" + std::to_string(x) + " = " + xck + " >= 0 && " +
               xck + " < args.src_tensor.Width();\n";
          if (!src_def.CanReadOutOfBorder(Axis::kWidth, gpu_info_)) {
            c += "  " + xck + " = clamp(" + xck +
                 ", 0, args.src_tensor.Width() - 1);\n";
          }
        }
      }
    }
    const bool need_multiple_slice_strides =
        src_def.ReturnsZeroForNegOneRead(gpu_info_) &&
        !conv_params_.IsTrivialKernelSize();
    for (int z = 0; z < block_size.z; ++z) {
      const std::string zind = std::to_string(z);
      for (int y = 0; y < block_size.y; ++y) {
        const std::string yind = std::to_string(y);
        for (int x = 0; x < block_size.x; ++x) {
          const std::string xind = std::to_string(x);
          std::string xc =
              conv_params_.IsXKernelIs1() ? "xc" + xind : "xck" + xind;
          std::string yc =
              conv_params_.IsYKernelIs1() ? "yc" + yind : "yck" + yind;
          const std::string id = GenerateIdSpatial(xind, yind, zind);
          std::string coords = "" + xc + ", " + yc;
          if (src_def.HasAxis(Axis::kDepth)) {
            std::string zc =
                conv_params_.IsZKernelIs1() ? "zc" + zind : "zck" + zind;
            coords += ", " + zc;
          }
          if (src_def.IsLinear()) {
            c += "  int addr" + id + " = args.src_tensor.GetAddress(" + coords +
                 ", " + src_group_start_slice + ");\n";
            if (need_multiple_slice_strides) {
              const std::string check = GenerateCheck(xind, yind, zind);
              c += "  addr" + id + " = select(-1, addr" + id + ", (" + check +
                   "));\n";
              c += "  int ds" + id +
                   " = select(0, args.src_tensor.SliceStride(), (" + check +
                   "));\n";
            }
          }
        }
      }
    }
    if (src_def.IsLinear() && !need_multiple_slice_strides) {
      c += "  int ds = args.src_tensor.SliceStride();\n";
    }

    if (conv_params_.softmax_input_activation) {
      for (int y = 0; y < block_size.y; ++y) {
        const std::string yind = std::to_string(y);
        for (int x = 0; x < block_size.x; ++x) {
          const std::string xind = std::to_string(x);
          std::string id = GenerateIdSpatial(xind, yind, "0");
          std::string address;
          std::string xc =
              conv_params_.IsXKernelIs1() ? "xc" + xind : "xck" + xind;
          std::string yc =
              conv_params_.IsYKernelIs1() ? "yc" + yind : "yck" + yind;
          address = "" + xc + ", " + yc;
          address += ", 0";
          c += "  ScalarType2 exp_val" + id + " = args.src_exp.Read(" +
               address + ").xy;\n";
        }
      }
    }

    const bool has_dynamic_src_ch =
        conv_params_.runtime_check.src_end_ch_index.has_value();
    if (kernel_params_.AreWeightsBuffer() && has_dynamic_src_ch) {
      if (gpu_info_.SupportsPointersInKernels()) {
        c += "  weights_offset = args.weights.GetPtr() + ";
      } else {
        c += "  weights_offset = ";
      }
      std::string z_ind = "0";
      std::string kernel_size_z = "1";
      if (src_def.HasAxis(Axis::kDepth) && !conv_params_.IsZKernelIs1()) {
        z_ind = "kz";
        kernel_size_z = "args.kernel_size_z";
      }
      std::string y_ind = "0";
      std::string kernel_size_y = "1";
      if (!conv_params_.IsYKernelIs1()) {
        y_ind = "ky";
        kernel_size_y = "args.kernel_size_y";
      }
      std::string x_ind = "0";
      std::string kernel_size_x = "1";
      if (!conv_params_.IsXKernelIs1()) {
        x_ind = "kx";
        kernel_size_x = "args.kernel_size_x";
      }
      const std::string spatial_offset =
          "(" + z_ind + " * " + kernel_size_y + " * " + kernel_size_x + " + " +
          y_ind + " * " + kernel_size_x + " + " + x_ind + ")";
      c += weights_offset + " + " + spatial_offset + " * " + src_group_slices +
           " * " +
           std::to_string(weights_vals_per_loop /
                          kernel_params_.src_depth_loop_size) +
           ";\n";
    }
    if (kernel_params_.slices_loop_first) {
      c += "  int s = " + src_group_start_slice + ";\n";
    }

      if (kernel_params_.slices_loop_first) {
        if (kernel_params_.need_src_loop) {
          c += "  do {\n";
        }
      }
      if (use_intel_wave_matmul) {
        c += GenerateMainIntelWaveMatMulFp16Srcx4DstxN();
      } else {
        c += GenerateWeightsLoad(weights_vals_per_loop);
        if (!kernel_params_.unroll_x_loop) {
          c += GenerateMain();
        } else {
          for (int kx = 0; kx < conv_params_.kernel_size.x; ++kx) {
            c += "  {\n";
            for (int x = 0; x < block_size.x; ++x) {
              const std::string xck = "xck" + std::to_string(x);
              c += "    int xck" + std::to_string(x) + " = " +
                   std::to_string(kx) + " * args.dilation_x + xc" +
                   std::to_string(x) + ";\n";
              if (!src_def.SupportsZeroClamp(Axis::kWidth, gpu_info_)) {
                c += "    bool in_x" + std::to_string(x) + " = " + xck +
                     " >= 0 && " + xck + " < args.src_tensor.Width();\n";
                if (!src_def.CanReadOutOfBorder(Axis::kWidth, gpu_info_)) {
                  c += "    " + xck + " = clamp(" + xck +
                       ", 0, args.src_tensor.Width() - 1);\n";
                }
              }
            }
            c += GenerateMain(kx * kernel_params_.block_size.w * 4);
            c += "  }\n";
          }
        }
        if (kernel_params_.AreWeightsBuffer() && !weights_conversion) {
          c +=
              "    weights_offset += " + std::to_string(weights_vals_per_loop) +
              ";\n";
        }
      }
      if (kernel_params_.slices_loop_first) {
        if (kernel_params_.need_src_loop) {
          c += "  } while (s < " + src_group_end_slice + ");\n";
        }
      }
    if (!kernel_params_.unroll_x_loop && !conv_params_.IsXKernelIs1()) {
      c += "  }\n";
    }
    if (!conv_params_.IsYKernelIs1()) {
      if (do_while_loop_for_y) {
        c += "    ky += 1;\n";
        c += "  } while (ky < args.kernel_size_y);\n";
      } else {
        c += "  }\n";
      }
    }
    if (src_def.HasAxis(Axis::kDepth) && !conv_params_.IsZKernelIs1()) {
      c += "  }\n";
    }
    if (!kernel_params_.slices_loop_first) {
      if (kernel_params_.need_src_loop) {
        c += "    s += 1;\n";
        c += "  } while (s < " + src_group_end_slice + ");\n";
      }
    }
    if (kernel_params_.weights_upload_type ==
            ConvGeneric::WeightsUploadType::kWaveMemory &&
        kernel_params_.simd_sizes.size() > 1) {
      c += "__hint__((ucl_wave_memory_scope_end))\n";
    }
    if (late_oob_check) {
      c += "  if (" + dst_oob_check + ") {\n";
      c += "    return;\n";
      c += "  }\n";
    }

    auto generate_dst_check = [&](int x, int y, int z) {
      std::string check;
      const std::vector<Axis> axes{Axis::kWidth, Axis::kHeight, Axis::kDepth};
      const std::vector<std::string> names{"Width()", "Height()", "Depth()"};
      std::vector<std::string> coords(3);
      coords[0] = "DST_X + " + std::to_string(x);
      coords[1] = "DST_Y + " + std::to_string(y);
      coords[2] = "DST_Z + " + std::to_string(z);
      const std::vector<int> ids{x, y, z};
      for (int i = 0; i < axes.size(); ++i) {
        const auto& axis = axes[i];
        if (src_def.HasAxis(axis) && ids[i] != 0) {
          if (!check.empty()) {
            check += " && ";
          }
          check += coords[i] + " < args.dst_tensor." + names[i];
        }
      }
      return check;
    };

    for (int s = 0; s < block_size.w; ++s) {
      const std::string sind = std::to_string(s);
      c += "  if (DST_S + " + sind + " >= args.dst_tensor.Slices()) return;\n";
      c += "  {\n";
      if (conv_params_.has_bias) {
        c += "    Type bias_val = args.biases.Read(DST_S + " + sind + ");\n";
      } else if (!conv_params_.Is8Bit() && !conv_params_.Is4Bit()) {
        c += "    Type bias_val = ucl::Init<Type>(0.0f);\n";
        // fake if, without it we have worse performance
        c += "    if (DST_S < 0) {\n";
        c += "      bias_val = args.src_tensor.Read(0, 0, 0);\n";
        c += "    }\n";
      }
      for (int z = 0; z < block_size.z; ++z) {
        const std::string zind = std::to_string(z);
        for (int y = 0; y < block_size.y; ++y) {
          const std::string yind = std::to_string(y);
          for (int x = 0; x < block_size.x; ++x) {
            const std::string xind = std::to_string(x);
            const std::string check = generate_dst_check(x, y, z);
            std::string coords = "DST_X + " + xind + ", DST_Y + " + yind;
            if (src_def.HasAxis(Axis::kDepth)) {
              coords += ", DST_Z + " + zind;
            }
            coords += ", DST_S + " + sind;
            if (!check.empty()) {
              c += "  if (" + check + ") {\n";
            } else {
              c += "  {\n";
            }
            if (conv_params_.runtime_check.packed_groups.has_value()) {
              c += "  if (DST_X + " + xind +
                   " < w_group_offset + w_group_size) {\n";
            }
            if (UseVec8Accumulator()) {
              const std::string id =
                  GenerateIdFull(xind, yind, zind, std::to_string(s / 2));
              std::string postfixes[] = {".s0123", ".s4567"};
              c += "    args.dst_tensor::type res = "
                   "ucl::Convert<args.dst_tensor::type>(r" +
                   id + postfixes[s % 2] + ");\n";
            } else {
              const std::string id = GenerateIdFull(xind, yind, zind, sind);
              c += "    args.dst_tensor::type res = "
                   "ucl::Convert<args.dst_tensor::type>(r" +
                   id + ");\n";
            }
            if (conv_params_.has_bias ||
                (!conv_params_.Is8Bit() && !conv_params_.Is4Bit())) {
              c += "    res += bias_val;\n";
            }
            c += "    args.dst_tensor.Write(res, " + coords + ");\n";
            if (conv_params_.runtime_check.packed_groups.has_value()) {
              c += "  }\n";
            }
            c += "  }\n";
          }
        }
      }
      c += "  }\n";
    }
    return c;
  }

  std::string GenerateIdSpatial(const std::string& x, const std::string& y,
                                const std::string& z) {
    std::string id;
    if (conv_params_.src_desc.HasAxis(Axis::kWidth)) {
      id += "_w" + x;
    }
    if (conv_params_.src_desc.HasAxis(Axis::kHeight)) {
      id += "_h" + y;
    }
    if (conv_params_.src_desc.HasAxis(Axis::kDepth)) {
      id += "_d" + z;
    }
    return id;
  }

  std::string GenerateIdSpatial(int x, int y, int z) {
    std::string id;
    if (conv_params_.src_desc.HasAxis(Axis::kWidth)) {
      id += "_w" + std::to_string(x);
    }
    if (conv_params_.src_desc.HasAxis(Axis::kHeight)) {
      id += "_h" + std::to_string(y);
    }
    if (conv_params_.src_desc.HasAxis(Axis::kDepth)) {
      id += "_d" + std::to_string(z);
    }
    return id;
  }

  std::string GenerateIdFull(const std::string& x, const std::string& y,
                             const std::string& z, const std::string& s) {
    return GenerateIdSpatial(x, y, z) + "_s" + s;
  }

  std::string GenerateIdFull(int x, int y, int z, int s) {
    return GenerateIdSpatial(x, y, z) + "_s" + std::to_string(s);
  }

  std::string GenerateCheck(const std::string& x, const std::string& y,
                            const std::string& z) {
    std::string check;
    const std::vector<Axis> axes{Axis::kWidth, Axis::kHeight, Axis::kDepth};
    const std::vector<std::string> names{"in_x", "in_y", "in_z"};
    const std::vector<bool> is_1{conv_params_.IsXKernelIs1(),
                                 conv_params_.IsYKernelIs1(),
                                 conv_params_.IsZKernelIs1()};
    const std::vector<std::string> coords{x, y, z};
    for (int i = 0; i < axes.size(); ++i) {
      const auto& axis = axes[i];
      if (conv_params_.src_desc.HasAxis(axis) &&
          !conv_params_.src_desc.SupportsZeroClamp(axis, gpu_info_) &&
          !is_1[i]) {
        if (!check.empty()) {
          check += " && ";
        }
        check += names[i] + coords[i];
      }
    }
    return check;
  }

  DataType GetAccumulatorType() const {
    if (IsFloatType(conv_params_.weights_data_type)) {
      return conv_params_.precision == CalculationsPrecision::kF16
                 ? DataType::kFloat16
                 : DataType::kFloat32;
    } else {
      return IsSigned(conv_params_.weights_data_type) ? DataType::kInt32
                                                      : DataType::kUint32;
    }
  }

  DataType GetSummableType() const {
    DataType summable_data_type = conv_params_.weights_data_type;
    if (gpu_info_.IsPowerVR() && !gpu_info_.powervr_info.IsImgBxx() &&
        conv_params_.precision == CalculationsPrecision::kF32F16 &&
        kernel_params_.weights_upload_type ==
            ConvGeneric::WeightsUploadType::kLocalMemory) {
      summable_data_type = DataType::kFloat32;
    }
    return summable_data_type;
  }

  std::string GetWeightsCacheType() const {
    std::string weights_cache_type = ToUclDataType(GetSummableType(), 4);
    if (conv_params_.Is8Bit() || conv_params_.Is4Bit()) {
      weights_cache_type = ToUclDataType(kernel_params_.weights_type,
                                         kernel_params_.weights_element_size);
    }
    return weights_cache_type;
  }

  // amount of vec4/vec16 weights values per loop
  int GetWeightsValsPerLoop() const {
    int weights_vals_per_loop =
        kernel_params_.block_size.w * kernel_params_.src_depth_loop_size * 4;
    if (kernel_params_.unroll_x_loop) {
      weights_vals_per_loop *= conv_params_.kernel_size.x;
    }
    if (conv_params_.Is8Bit() && kernel_params_.weights_element_size == 4) {
      // 8-bit weights as uint32x4
      weights_vals_per_loop /= 4;
    }
    return weights_vals_per_loop;
  }

  bool NeedsLocalMemory() const {
    return kernel_params_.weights_upload_type ==
               ConvGeneric::WeightsUploadType::kLocalMemory ||
           kernel_params_.weights_upload_type ==
               ConvGeneric::WeightsUploadType::kLocalMemoryWGLoad;
  }

  bool NeedsWaveMemory() const {
    return kernel_params_.weights_upload_type ==
           ConvGeneric::WeightsUploadType::kWaveMemory;
  }

  // experimental, not fully supported
  bool UseVec8Accumulator() {
    return kernel_params_.weights_upload_type ==
               ConvGeneric::WeightsUploadType::kIntelWave16MatMul &&
           false;
  }

  std::string GenerateSrcDeclaration() {
    const DataType summable_data_type = GetSummableType();
    std::string c;
    for (int z = 0; z < kernel_params_.block_size.z; ++z) {
      const std::string zind = std::to_string(z);
      for (int y = 0; y < kernel_params_.block_size.y; ++y) {
        const std::string yind = std::to_string(y);
        for (int x = 0; x < kernel_params_.block_size.x; ++x) {
          const std::string xind = std::to_string(x);
          const std::string id = GenerateIdSpatial(xind, yind, zind);
          if (conv_params_.IsSrcPackedW4C4()) {
            if (x % 4 == 0) {
              const std::string src_packed_type =
                  conv_params_.src_desc.GetDataType() == DataType::kInt32
                      ? "int4"
                      : "uint4";
              c += "    " + src_packed_type + " src_packed" + id + ";\n";
            }
          } else if (conv_params_.IsSrcPackedC16() ||
                     conv_params_.IsSrcPackedC32()) {
            const std::string src_packed_type =
                conv_params_.src_desc.GetDataType() == DataType::kInt32
                    ? "int4"
                    : "uint4";
            c += "    " + src_packed_type + " src_packed" + id + ";\n";
          } else {
            c += "    " + ToUclDataType(summable_data_type, 4) + " src" + id +
                 ";\n";
          }
          if (conv_params_.Is8Bit()) {
            c += "    uint src" + id + "_uint;\n";
          }
        }
      }
    }
    return c;
  }

  std::string GenerateSrcReading() {
    const auto& src_def = conv_params_.src_desc;
    const DataType summable_data_type = GetSummableType();
    const bool conditional_read = gpu_info_.IsMali();

    std::string c;
    for (int z = 0; z < kernel_params_.block_size.z; ++z) {
      const std::string zind = std::to_string(z);
      for (int y = 0; y < kernel_params_.block_size.y; ++y) {
        const std::string yind = std::to_string(y);
        for (int x = 0; x < kernel_params_.block_size.x; ++x) {
          if (conv_params_.IsSrcPackedW4C4() && x % 4 != 0) {
            continue;
          }
          const std::string xind = std::to_string(x);
          std::string id = GenerateIdSpatial(xind, yind, zind);
          const std::string check = GenerateCheck(xind, yind, zind);
          std::string address;
          if (src_def.IsLinear()) {
            address = "addr" + id;
          } else {
            std::string xc =
                conv_params_.IsXKernelIs1() ? "xc" + xind : "xck" + xind;
            std::string yc =
                conv_params_.IsYKernelIs1() ? "yc" + yind : "yck" + yind;
            address = "" + xc + ", " + yc;
            if (src_def.HasAxis(Axis::kDepth)) {
              std::string zc =
                  conv_params_.IsZKernelIs1() ? "zc" + zind : "zck" + zind;
              address += ", " + zc;
            }
            address += ", s";
          }
          std::string read_as_type = ToUclDataType(summable_data_type);
          std::string src_val = "src" + id;
          if (conv_params_.IsSrcPackedW4C4()) {
            read_as_type =
                conv_params_.src_desc.GetDataType() == DataType::kInt32
                    ? "int"
                    : "uint";
            src_val = "src_packed" + id;
          }
          if (conv_params_.IsSrcPackedC16() || conv_params_.IsSrcPackedC32()) {
            read_as_type =
                conv_params_.src_desc.GetDataType() == DataType::kInt32
                    ? "int"
                    : "uint";
            src_val = "src_packed" + id;
          }
          if (src_def.ReturnsZeroForNegOneRead(gpu_info_)) {
            c += "    " + src_val + " = args.src_tensor.Read<" + read_as_type +
                 ">(" + address + ");\n";
            const std::string ds =
                conv_params_.IsTrivialKernelSize() ? "ds" : "ds" + id;
            c += "    " + address + " += " + ds + ";\n";
          } else {
            if (!check.empty()) {
              if (conditional_read) {
                c += "    " + src_val + " = " + check +
                     " ? args.src_tensor.Read<" + read_as_type + ">(" +
                     address + ") : ucl::Init<" +
                     ToUclDataType(summable_data_type, 4) + ">(0.0f);\n";
              } else {
                c += "    " + src_val + " = args.src_tensor.Read<" +
                     read_as_type + ">(" + address +
                     ") * ucl::Convert<ScalarType>(" + check + ");\n";
              }
            } else {
              c += "    " + src_val + " = args.src_tensor.Read<" +
                    read_as_type + ">(" + address + ");\n";
            }
            if (src_def.IsLinear()) {
              c += "    " + address + " += ds;\n";
            }
          }
          if (conv_params_.Is8Bit()) {
            if (conv_params_.IsSrcPackedW4C4()) {
              for (int sub_x = 0; sub_x < 4; ++sub_x) {
                const std::string src_name =
                    "src" +
                    GenerateIdSpatial(std::to_string(x + sub_x), yind, zind);
                const std::string src_packed_name =
                    "src_packed" + id + "." + postfixes_[sub_x];
                if (conv_params_.src_desc.GetDataType() == DataType::kInt32) {
                  c += "    " + src_name +
                       "_uint = ucl::Reinterpret<int, uint>(" +
                       src_packed_name + ");\n";
                } else {
                  c += "    " + src_name + "_uint = " + src_packed_name + ";\n";
                }
              }
            } else if (conv_params_.IsSrcPackedC16() ||
                       conv_params_.IsSrcPackedC32()) {
              // do nothing here, handled in the main function
            } else {
              c += "    src" + id +
                   "_uint = ucl::Reinterpret<char4, uint>(src" + id + ");\n";
            }
          }
        }
      }
    }
    if (conv_params_.softmax_input_activation) {
      c += GenerateSoftmaxActivation();
    }
    return c;
  }

  std::string GenerateSoftmaxActivation() {
    std::string c;
    for (int y = 0; y < kernel_params_.block_size.y; ++y) {
      const std::string yind = std::to_string(y);
      for (int x = 0; x < kernel_params_.block_size.x; ++x) {
        const std::string xind = std::to_string(x);
        std::string id = GenerateIdSpatial(xind, yind, "0");
        c += "  src" + id + " = ucl::Exp<Type>(src" + id + " - exp_val" + id +
             ".y) * exp_val" + id + ".x;\n";
      }
    }
    return c;
  }

  std::string GenerateWeightsLoad(int weights_vals) {
    const int wg_total_size = kernel_params_.work_group_size.x *
                              kernel_params_.work_group_size.y *
                              kernel_params_.work_group_size.z;
    const std::string scope = gpu_info_.IsApiMetal() && wg_total_size == 32 &&
                                      gpu_info_.IsWaveSizeEqualTo32()
                                  ? "SubGroup"
                                  : "WorkGroup";
    const std::string weights_cache_type = GetWeightsCacheType();
    std::string c;
    if (kernel_params_.weights_upload_type ==
        ConvGeneric::WeightsUploadType::kLocalMemoryWGLoad) {
      c += "    ucl::SyncThreads<WorkGroup, Local>();\n";
      c += GenerateAsyncUpload("weights_cache", "weights_offset",
                               /*global_offset_name*/ "", weights_vals);
    } else if (kernel_params_.weights_upload_type ==
               ConvGeneric::WeightsUploadType::kLocalMemory) {
      if (gpu_info_.IsApiMetal()) {
        c += "    ucl::SyncThreads<" + scope + ", None>();\n";
      } else {
        c += "    ucl::SyncThreads<" + scope + ", Local>();\n";
      }

      if (gpu_info_.SupportsPointersInKernels()) {
        c += GenerateUploadByThreads("weights_cache", "weights_offset",
                                     /*use_ptrs*/ true,
                                     /*global_offset_name*/ "",
                                     weights_cache_type, "lid", wg_total_size,
                                     weights_vals);
      } else {
        c += GenerateUploadByThreads("weights_cache", "args.weights",
                                     /*use_ptrs*/ false, "weights_offset",
                                     weights_cache_type, "lid", wg_total_size,
                                     weights_vals);
      }
    } else if (kernel_params_.weights_upload_type ==
               ConvGeneric::WeightsUploadType::kWaveMemory) {
      c += "    ucl::SyncThreads<WaveLoad>();\n";
      if (gpu_info_.SupportsPointersInKernels()) {
        c += "    ucl::WaveLoad(weights_cache, weights_offset, 0, " +
             std::to_string(weights_vals) + ");\n";
      } else {
        c += "    ucl::WaveLoad(weights_cache, args.weights.GetPtr(), "
             "weights_offset, " +
             std::to_string(weights_vals) + ");\n";
      }
      c += "    ucl::SyncThreads<WaveLoad>();\n";
    } else if (kernel_params_
                   .AreWeightsBuffer()) {  // kGlobalMemory/kConstantMemory
      if (gpu_info_.SupportsPointersInKernels()) {
        c += "    weights_cache = weights_offset;\n";
      }
    } else {  // TEXTURES_MEM
      for (int dst_s = 0; dst_s < kernel_params_.block_size.w; ++dst_s) {
        std::string f_y =
            conv_params_.IsTrivialKernelSize() ? "s" : "weights_offset";
        if (conv_params_.IsTrivialKernelSize() && conv_params_.HasGroups()) {
          f_y = "s - src_start_slice";
        }
        if (conv_params_.different_weights_for_height) {
          f_y = "w_batch_id * args.src_tensor.Slices() + s";
        }
        c += absl::Substitute(
            R"(      Type f$2 = args.weights0.Read(DST_S + $0, $1);
      Type f$3 = args.weights1.Read(DST_S + $0, $1);
      Type f$4 = args.weights2.Read(DST_S + $0, $1);
      Type f$5 = args.weights3.Read(DST_S + $0, $1);
  )",
            dst_s, f_y, dst_s * 4 + 0, dst_s * 4 + 1, dst_s * 4 + 2,
            dst_s * 4 + 3);
      }
      if (!conv_params_.IsTrivialKernelSize()) {
        c += "    weights_offset++;\n";
      }
    }
    return c;
  }

  std::string GenerateMain(int weights_offset = 0) {
    std::string c;
    if (IntermediateWeightsRead()) {
      c += "    uint4 w_i8x16;\n";
    }
    c += GenerateSrcDeclaration();
    c += GenerateSrcReading();
    if (kernel_params_.slices_loop_first) {
      c += "    s += 1;\n";
    }
    if ((kernel_params_.weights_upload_type ==
             ConvGeneric::WeightsUploadType::kLocalMemory ||
         kernel_params_.weights_upload_type ==
             ConvGeneric::WeightsUploadType::kLocalMemoryWGLoad) &&
        !kernel_params_.unroll_x_loop) {
      const int wg_total_size = kernel_params_.work_group_size.x *
                                kernel_params_.work_group_size.y *
                                kernel_params_.work_group_size.z;
      const std::string scope = gpu_info_.IsApiMetal() && wg_total_size == 32 &&
                                        gpu_info_.IsWaveSizeEqualTo32()
                                    ? "SubGroup"
                                    : "WorkGroup";
      c += "    ucl::SyncThreads<" + scope + ", Local>();\n";
    }
    if (conv_params_.IsSrcPackedC16()) {
      std::string maybe_cast =
          conv_params_.src_desc.GetDataType() == DataType::kInt32
              ? "ucl::Reinterpret<int, uint>("
              : "(";
      ForSpatial([&](int x, int y, int z) {
        const std::string id = GenerateIdSpatial(x, y, z);
        c += "    src" + id + "_uint = " + maybe_cast + "src_packed" + id +
             ".x);\n";
      });
      c += GenerateConvCore(0);
      ForSpatial([&](int x, int y, int z) {
        const std::string id = GenerateIdSpatial(x, y, z);
        c += "    src" + id + "_uint = " + maybe_cast + "src_packed" + id +
             ".y);\n";
      });
      c += GenerateConvCore(1 * kernel_params_.block_size.w * 4);
      ForSpatial([&](int x, int y, int z) {
        const std::string id = GenerateIdSpatial(x, y, z);
        c += "    src" + id + "_uint = " + maybe_cast + "src_packed" + id +
             ".z);\n";
      });
      c += GenerateConvCore(2 * kernel_params_.block_size.w * 4);
      ForSpatial([&](int x, int y, int z) {
        const std::string id = GenerateIdSpatial(x, y, z);
        c += "    src" + id + "_uint = " + maybe_cast + "src_packed" + id +
             ".w);\n";
      });
      c += GenerateConvCore(3 * kernel_params_.block_size.w * 4);
    } else {
      c += GenerateConvCore(weights_offset);
      for (int i = 1; i < kernel_params_.src_depth_loop_size; ++i) {
        c += GenerateSrcReading();
        c += GenerateConvCore(weights_offset +
                              i * kernel_params_.block_size.w * 4);
        c += "    s += 1;\n";
      }
    }
    return c;
  }

  template <typename Func>
  void ForSpatial(Func&& func) {
    for (int z = 0; z < kernel_params_.block_size.z; ++z) {
      for (int y = 0; y < kernel_params_.block_size.y; ++y) {
        for (int x = 0; x < kernel_params_.block_size.x; ++x) {
          func(x, y, z);
        }
      }
    }
  }

  std::string ReadWeights1D() const {
    std::string c;
    if (conv_params_.weights_desc.type == DataType::kUint8) {
      c += "    uint4 u8_i4o4 = args.weights.Read(w_wg_offset);\n";
      c += "    ucl::U32x4ToU8x16AsVec4x4<half>(u8_i4o4, w0, w1, w2, w3);\n";
    } else if (conv_params_.weights_desc.type == DataType::kUint4) {
      c += "    uint2 u4_i4o4 = args.weights.Read(w_wg_offset);\n";
      c += "    ucl::U32x2ToU4x16AsVec4x4<half>(u4_i4o4, w0, w1, w2, w3);\n";
    } else if (conv_params_.weights_desc.type == DataType::kUint2) {
      c += "    uint u2_i4o4 = args.weights.Read(w_wg_offset);\n";
      c += "    ucl::U32x1ToU2x16AsVec4x4<half>(u2_i4o4, w0, w1, w2, w3);\n";
    } else {
      c += "    args.weights.ReadVec16AsVec4x4(w0, w1, w2, w3, "
           "w_wg_offset);\n";
    }
    c += "    w_wg_offset += w_wg_stride;\n";
    return c;
  }

  std::string ReadWeights2D() const {
    std::string c;
    const std::string xc = "w_o_slice";
    const std::string batch_part = conv_params_.different_weights_for_height
                                       ? "w_batch_id * args.src_tensor.Slices()"
                                       : "0";
    const std::string yc = batch_part + " + s + sub_i";
    if (conv_params_.weights_desc.layout ==
        WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4) {
      c += "    w0 = args.weights0.Read<half>(" + xc + ", " + yc + ");\n";
      c += "    w1 = args.weights1.Read<half>(" + xc + ", " + yc + ");\n";
      c += "    w2 = args.weights2.Read<half>(" + xc + ", " + yc + ");\n";
      c += "    w3 = args.weights3.Read<half>(" + xc + ", " + yc + ");\n";
    } else if (conv_params_.weights_desc.type == DataType::kUint8) {
      c += "    uint4 u8_i4o4 = args.weights.Read(" + xc + ", " + yc + ");\n";
      c += "    ucl::U32x4ToU8x16AsVec4x4<half>(u8_i4o4, w0, w1, w2, w3);\n";
    } else if (conv_params_.weights_desc.type == DataType::kUint4) {
      c += "    ushort4 u4_i4o4 = args.weights.Read(" + xc + ", " + yc + ");\n";
      c += "    ucl::U16x4ToU4x16AsVec4x4<half>(u4_i4o4, w0, w1, w2, w3);\n";
    } else if (conv_params_.weights_desc.type == DataType::kUint2) {
      c += "    uchar4 u2_i4o4 = args.weights.Read(" + xc + ", " + yc + ");\n";
      c += "    ucl::U8x4ToU2x16AsVec4x4<half>(u2_i4o4, w0, w1, w2, w3);\n";
    }
    return c;
  }

  std::string GenerateMainIntelWaveMatMulFp16Srcx4DstxN() {
    std::string c;
    ForSpatial([&](int x, int y, int z) {
      c += "    int8 S" + GenerateIdSpatial(x, y, z) + ";\n";
    });
    c += GenerateSrcDeclaration();
    std::vector<std::string> postfixes = {".s01", ".s23", ".s45", ".s67"};
    std::string as_type = "as_int2";
    std::string src_prefix = "src";
    if (conv_params_.Is8Bit() || conv_params_.Is4Bit()) {
      postfixes = {".s0123", ".s4567"};
      as_type = "as_int4";
      src_prefix = "src_packed";
    }
    const bool weights_conversion =
        conv_params_.weights_desc.layout != WeightsLayout::kUnknown;
    const bool quantized_weights =
        weights_conversion && SizeInBitsOf(conv_params_.weights_desc.type) <= 8;
    if (weights_conversion) {
      if (conv_params_.scale_zp_shape.i != 1) {
        // grouped quantization
        c += "    int src_group_id = (s + sub_i) / " +
             std::to_string(conv_params_.src_group_slices) + ";\n";
        c += "    if (last_src_group_id != src_group_id) {\n";
        c += "      last_src_group_id = src_group_id;\n";
        c += fc::ReadScaleZeroPointBlock(
            "w_o_slice", conv_params_.scale_zp_shape,
            conv_params_.has_zero_point,
            fc::GetDataTypeForWeights(conv_params_.weights_desc.type));
        c += "    }\n";
      }
      c += "    half4 w0, w1, w2, w3;\n";
      if (conv_params_.weights_desc.IsLinearLayout()) {
        c += ReadWeights1D();
      } else {
        c += ReadWeights2D();
      }
      if (quantized_weights) {
        c += R"(
    w0 = w0 * w_scale + w_bias;
    w1 = w1 * w_scale + w_bias;
    w2 = w2 * w_scale + w_bias;
    w3 = w3 * w_scale + w_bias;
)";
      }
    }
    for (const auto& postfix : postfixes) {
      c += GenerateSrcReading();
      ForSpatial([&](int x, int y, int z) {
        const std::string id = GenerateIdSpatial(x, y, z);
        c += "    S" + id + postfix + " = " + as_type + "(" + src_prefix + id +
             ");\n";
      });
      c += "    s += 1;\n";
    }
    if (weights_conversion) {
      c += R"(
    ucl::SyncThreads<WorkGroup, Local>();
    w_cache[sub_o * 16 + sub_i * 4 + 0] = w0;
    w_cache[sub_o * 16 + sub_i * 4 + 1] = w1;
    w_cache[sub_o * 16 + sub_i * 4 + 2] = w2;
    w_cache[sub_o * 16 + sub_i * 4 + 3] = w3;
    ucl::SyncThreads<WorkGroup, Local>();
  )";
      for (int i = 0; i < kernel_params_.block_size.w; ++i) {
        const std::string w_name = "W" + std::to_string(i);
        c += "    short4 " + w_name + " = as_short4(w_cache[simd_id + " +
             std::to_string(i * 16) + "]);\n";
      }
    } else {
      for (int i = 0; i < kernel_params_.block_size.w; ++i) {
        const std::string w_name = "W" + std::to_string(i);
        c += "    short4 " + w_name +
             " = ((__global short4*)weights_offset)[simd_id + " +
             std::to_string(i * 16) + "];\n";
      }
    }
    if (!weights_conversion) {
      // for fp16 weights_offset is half4 ptr (8 bytes, float16t_x4)
      // for int8 weights_offset is uint ptr (4 bytes, int8t_x4)
      // for int4 weights_offset is uint ptr (4 bytes, int4t_x8)
      const int filter_size =
          conv_params_.Is8Bit() || conv_params_.Is4Bit() ? 4 : 8;
      const int filter_elements = 128 / filter_size;
      c += "    weights_offset += " +
           std::to_string(kernel_params_.block_size.w * filter_elements) +
           ";\n";
    }
    std::string function_name = "intel_sub_group_f16_f16_matrix_mad_k16";
    if (conv_params_.Is8Bit()) {
      function_name = "intel_sub_group_i8_i8_matrix_mad_k32";
    }
    if (conv_params_.Is4Bit()) {
      function_name = "intel_sub_group_i4_i4_matrix_mad_k64";
    }
    for (int i = 0; i < kernel_params_.block_size.w; ++i) {
      ForSpatial([&](int x, int y, int z) {
        const std::string r_name = "r" + GenerateIdFull(x, y, z, i);
        const std::string s_name = "S" + GenerateIdSpatial(x, y, z);
        const std::string w_name = "W" + std::to_string(i);
        c += "    " + r_name + " = " + function_name + "(" + w_name + ", " +
             s_name + ", " + r_name + ");\n";
      });
    }
    return c;
  }

  std::string GenerateConvCore(int shared_offset) {
    if (SupportsImgMatMul(gpu_info_, conv_params_)) {
      return GenerateConvImgMatMul(shared_offset);
    }
    if (conv_params_.Is8Bit()) {
      return GenerateConvInt8Core(shared_offset);
    }
    const bool use_weights_cache = NeedsLocalMemory() || NeedsWaveMemory();
    std::string c;
    const bool use_fma = gpu_info_.IsApiOpenCl() && gpu_info_.IsAMD();
    const std::string channels[] = {"x", "y", "z", "w"};
    const int4& block_size = kernel_params_.block_size;
    const DataType summable_data_type = GetSummableType();
    for (int s = 0; s < block_size.w; ++s) {
      const std::string sind = std::to_string(s);
      if (conv_params_.precision != CalculationsPrecision::kF32F16 ||
          summable_data_type == DataType::kFloat32) {
        for (int ch = 0; ch < 4; ++ch) {
          for (int z = 0; z < block_size.z; ++z) {
            const std::string zind = std::to_string(z);
            for (int y = 0; y < block_size.y; ++y) {
              const std::string yind = std::to_string(y);
              for (int x = 0; x < block_size.x; ++x) {
                const std::string xind = std::to_string(x);
                std::string R = "r" + GenerateIdFull(xind, yind, zind, sind);
                std::string S = "src" + GenerateIdSpatial(xind, yind, zind);
                const std::string weight_id =
                    std::to_string(s * 4 + ch + shared_offset);
                std::string w_val;
                if (kernel_params_.AreWeightsBuffer()) {
                  if (gpu_info_.SupportsPointersInKernels() ||
                      use_weights_cache) {
                    w_val = "weights_cache[" + weight_id + "]";
                  } else {
                    w_val =
                        "args.weights.Read(weights_offset + " + weight_id + ")";
                  }
                } else {
                  w_val = "f" + weight_id;
                }
                if (kernel_params_.weights_desc.IsI4O4()) {
                  if (use_fma) {
                    c += "    " + R + " = fma(" + w_val + ", " + S + "." +
                         channels[ch] + ", " + R + ");\n";
                  } else {
                    c += "    " + R + " += " + w_val + " * " + S + "." +
                         channels[ch] + ";\n";
                  }
                } else {
                  c += "    " + R + "." + channels[ch] + " += dot(" + w_val +
                       ", " + S + ");\n";
                }
              }
            }
          }
        }
      } else {  // F32_F16 precision
        for (int z = 0; z < block_size.z; ++z) {
          const std::string zind = std::to_string(z);
          for (int y = 0; y < block_size.y; ++y) {
            const std::string yind = std::to_string(y);
            for (int x = 0; x < block_size.x; ++x) {
              const std::string xind = std::to_string(x);
              std::string R = "r" + GenerateIdFull(xind, yind, zind, sind);
              std::string S = "src" + GenerateIdSpatial(xind, yind, zind);
              std::vector<std::string> F(4);
              for (int i = 0; i < 4; ++i) {
                std::string weight_id =
                    std::to_string(s * 4 + i + shared_offset);
                if (kernel_params_.AreWeightsBuffer()) {
                  if (gpu_info_.SupportsPointersInKernels() ||
                      use_weights_cache) {
                    F[i] = "weights_cache[" + weight_id + "]";
                  } else {
                    F[i] =
                        "args.weights.Read(weights_offset + " + weight_id + ")";
                  }
                } else {
                  F[i] = "f" + weight_id;
                }
              }
              if (kernel_params_.weights_desc.IsI4O4()) {
                c += "    " + R + " += ucl::Convert<AccType>(" + S + ".x * " +
                     F[0] + " + " + S + ".y * " + F[1] + " + " + S + ".z * " +
                     F[2] + " + " + S + ".w * " + F[3] + ");\n";
              } else {
                c += "    " + R + ".x += ucl::Convert<AccSType>(dot(" + S +
                     ", " + F[0] + "));\n";
                c += "    " + R + ".y += ucl::Convert<AccSType>(dot(" + S +
                     ", " + F[1] + "));\n";
                c += "    " + R + ".z += ucl::Convert<AccSType>(dot(" + S +
                     ", " + F[2] + "));\n";
                c += "    " + R + ".w += ucl::Convert<AccSType>(dot(" + S +
                     ", " + F[3] + "));\n";
              }
            }
          }
        }
      }
    }
    return c;
  }

  bool IntermediateWeightsRead() const {
    return gpu_info_.IsAMD() && conv_params_.Is8Bit() &&
           kernel_params_.weights_element_size == 4;
  }

  std::string GetWeightsValue(int offset) const {
    const bool use_weights_cache = NeedsLocalMemory() || NeedsWaveMemory();
    if (use_weights_cache) {
      return "weights_cache[" + std::to_string(offset) + "]";
    } else if (gpu_info_.SupportsPointersInKernels()) {
      return "weights_offset[" + std::to_string(offset) + "]";
    } else {
      return "args.weights.Read(weights_offset + " + std::to_string(offset) +
             ")";
    }
  }

  std::string GenerateConvInt8Core(int shared_offset) {
    std::string c;
    const int4& block_size = kernel_params_.block_size;
    for (int s = 0; s < block_size.w; ++s) {
      const std::string sind = std::to_string(s);
      if (IntermediateWeightsRead()) {
        c += "    w_i8x16 = " + GetWeightsValue(s + shared_offset / 4) + ";\n";
      }
      for (int z = 0; z < block_size.z; ++z) {
        const std::string zind = std::to_string(z);
        for (int y = 0; y < block_size.y; ++y) {
          const std::string yind = std::to_string(y);
          for (int x = 0; x < block_size.x; ++x) {
            const std::string xind = std::to_string(x);
            std::string R = "r" + GenerateIdFull(xind, yind, zind, sind);
            std::string S = "src" + GenerateIdSpatial(xind, yind, zind);
            std::vector<std::string> F(4);
            for (int ch = 0; ch < 4; ++ch) {
              const int weight_id = s * 4 + ch + shared_offset;
              const bool is_uint32x4 =
                  conv_params_.Is8Bit() &&
                  kernel_params_.weights_element_size ==
                      4;  // // 8-bit weights as uint32x4, otherwise uint32x1
              if (IntermediateWeightsRead()) {
                F[ch] = "w_i8x16." + postfixes_[ch];
              } else {
                F[ch] = GetWeightsValue(weight_id / (is_uint32x4 ? 4 : 1));
                if (is_uint32x4) {
                  F[ch] += "." + postfixes_[ch];
                }
              }
            }
            if (gpu_info_.SupportsExtension("cl_qcom_dot_product8")) {
              c += "  " + R + ".x = qcom_dot8_acc(" + F[0] + ", " + S +
                   "_uint, " + R + ".x);\n";
              c += "  " + R + ".y = qcom_dot8_acc(" + F[1] + ", " + S +
                   "_uint, " + R + ".y);\n";
              c += "  " + R + ".z = qcom_dot8_acc(" + F[2] + ", " + S +
                   "_uint, " + R + ".z);\n";
              c += "  " + R + ".w = qcom_dot8_acc(" + F[3] + ", " + S +
                   "_uint, " + R + ".w);\n";
            } else if (gpu_info_.SupportsExtension(
                           "cl_khr_integer_dot_product")) {
              if (conv_params_.weights_data_type == DataType::kInt8) {
                c += "    " + R + ".x += dot_4x8packed_ss_int(" + F[0] + ", " +
                     S + "_uint);\n";
                c += "    " + R + ".y += dot_4x8packed_ss_int(" + F[1] + ", " +
                     S + "_uint);\n";
                c += "    " + R + ".z += dot_4x8packed_ss_int(" + F[2] + ", " +
                     S + "_uint);\n";
                c += "    " + R + ".w += dot_4x8packed_ss_int(" + F[3] + ", " +
                     S + "_uint);\n";
              } else {
                c += "    " + R + ".x += dot_4x8packed_uu_uint(" + F[0] + ", " +
                     S + "_uint);\n";
                c += "    " + R + ".y += dot_4x8packed_uu_uint(" + F[1] + ", " +
                     S + "_uint);\n";
                c += "    " + R + ".z += dot_4x8packed_uu_uint(" + F[2] + ", " +
                     S + "_uint);\n";
                c += "    " + R + ".w += dot_4x8packed_uu_uint(" + F[3] + ", " +
                     S + "_uint);\n";
              }
            } else if (gpu_info_.SupportsExtension(
                           "cl_arm_integer_dot_product_accumulate_int8")) {
              c += "    " + R + ".x = arm_dot_acc(as_char4(" + F[0] +
                   "), as_char4(" + S + "_uint), " + R + ".x);\n";
              c += "    " + R + ".y = arm_dot_acc(as_char4(" + F[1] +
                   "), as_char4(" + S + "_uint), " + R + ".y);\n";
              c += "    " + R + ".z = arm_dot_acc(as_char4(" + F[2] +
                   "), as_char4(" + S + "_uint), " + R + ".z);\n";
              c += "    " + R + ".w = arm_dot_acc(as_char4(" + F[3] +
                   "), as_char4(" + S + "_uint), " + R + ".w);\n";
            } else if (gpu_info_.IsApiWebGpu() &&
                       gpu_info_.SupportsAcceleratedDp4a()) {
              if (conv_params_.weights_data_type == DataType::kInt8) {
                c += "    " + R + ".x += dot4I8Packed(" + F[0] + ", " + S +
                     "_uint);\n";
                c += "    " + R + ".y += dot4I8Packed(" + F[1] + ", " + S +
                     "_uint);\n";
                c += "    " + R + ".z += dot4I8Packed(" + F[2] + ", " + S +
                     "_uint);\n";
                c += "    " + R + ".w += dot4I8Packed(" + F[3] + ", " + S +
                     "_uint);\n";
              } else {
                c += "    " + R + ".x += dot4U8Packed(" + F[0] + ", " + S +
                     "_uint);\n";
                c += "    " + R + ".y += dot4U8Packed(" + F[1] + ", " + S +
                     "_uint);\n";
                c += "    " + R + ".z += dot4U8Packed(" + F[2] + ", " + S +
                     "_uint);\n";
                c += "    " + R + ".w += dot4U8Packed(" + F[3] + ", " + S +
                     "_uint);\n";
              }
            } else if (gpu_info_.SupportsExtension(
                           "VK_KHR_shader_integer_dot_product")) {
              c += "    " + R + ".x = dotPacked4x8AccSatEXT(int(" + F[0] +
                   "), int(" + S + "_uint), " + R + ".x);\n";
              c += "    " + R + ".y = dotPacked4x8AccSatEXT(int(" + F[1] +
                   "), int(" + S + "_uint), " + R + ".y);\n";
              c += "    " + R + ".z = dotPacked4x8AccSatEXT(int(" + F[2] +
                   "), int(" + S + "_uint), " + R + ".z);\n";
              c += "    " + R + ".w = dotPacked4x8AccSatEXT(int(" + F[3] +
                   "), int(" + S + "_uint), " + R + ".w);\n";
            }
          }
        }
      }
    }
    return c;
  }

  std::string GenerateConvImgMatMul(int shared_offset) {
    std::string c;
    for (int s = 0; s < kernel_params_.block_size.w; ++s) {
      const std::string sind = std::to_string(s);
      for (int z = 0; z < kernel_params_.block_size.z; ++z) {
        const std::string zind = std::to_string(z);
        for (int y = 0; y < kernel_params_.block_size.y; ++y) {
          const std::string yind = std::to_string(y);
          for (int x = 0; x < kernel_params_.block_size.x; ++x) {
            const std::string xind = std::to_string(x);
            std::string R = "r" + GenerateIdFull(xind, yind, zind, sind);
            std::string S = "src" + GenerateIdSpatial(xind, yind, zind);
            if (conv_params_.precision == CalculationsPrecision::kF32) {
              // img_matmul_float_acc_1x2_2x2 or img_dot_interleaved_acc?
              // layout is the same, but img_dot_interleaved_acc performs better
              c += "  " + R + ".xy = img_dot_interleaved_acc(" + S +
                   ".xy, weights_cache + " +
                   std::to_string(shared_offset + s * 4) + ", " + R + ".xy);\n";
              c += "  " + R + ".xy = img_dot_interleaved_acc(" + S +
                   ".zw, weights_cache + " +
                   std::to_string(shared_offset + s * 4 + 1) + ", " + R +
                   ".xy);\n";
              c += "  " + R + ".zw = img_dot_interleaved_acc(" + S +
                   ".xy, weights_cache + " +
                   std::to_string(shared_offset + s * 4 + 2) + ", " + R +
                   ".zw);\n";
              c += "  " + R + ".zw = img_dot_interleaved_acc(" + S +
                   ".zw, weights_cache + " +
                   std::to_string(shared_offset + s * 4 + 3) + ", " + R +
                   ".zw);\n";
            } else {
              const std::string func_name =
                  conv_params_.precision == CalculationsPrecision::kF16
                      ? "img_matmul_half2_acc_1x2_2x2h"
                      : "img_matmul_half2_acc_1x2_2x2f";
              c += "  " + R + ".xy = " + func_name + "(" + S +
                   ", (__local half8*)(weights_cache + " +
                   std::to_string(shared_offset + s * 4) + "), " + R +
                   ".xy);\n";
              c += "  " + R + ".zw = " + func_name + "(" + S +
                   ", (__local half8*)(weights_cache + " +
                   std::to_string(shared_offset + s * 4 + 2) + "), " + R +
                   ".zw);\n";
            }
          }
        }
      }
    }
    return c;
  }

  const std::vector<std::string> postfixes_ = {"x", "y", "z", "w"};

  const GpuInfo& gpu_info_;
  const ConvGeneric::ConvParams& conv_params_;
  const ConvGeneric::KernelParams& kernel_params_;
};

void AddRuntimeParams(GPUOperation& op,
                      const ConvRuntimeCheckDesc& runtime_check) {
  bool has_runtime_check = false;
  if (runtime_check.src_end_ch_index.has_value()) {
    op.args_.AddInt("src_end_ch_index", *runtime_check.src_end_ch_index);
    has_runtime_check = true;
  }
  if (runtime_check.dst_end_ch_index.has_value()) {
    op.args_.AddInt("dst_end_ch_index", *runtime_check.dst_end_ch_index);
    has_runtime_check = true;
  }
  if (runtime_check.packed_groups.has_value()) {
    op.args_.AddInt("packed_params_offset",
                    runtime_check.packed_groups->params_offset);
    has_runtime_check = true;
  }
  if (has_runtime_check) {
    BufferDescriptor desc;
    desc.element_type = DataType::kInt32;
    desc.element_size = 1;
    op.AddSrcBuffer("params", desc);
  }
}

BHWC GetOutputShape(const BHWC* dst_shape,
                    const ConvGeneric::ConvParams& conv_params) {
  if (dst_shape) {
    return *dst_shape;
  }
  const int dst_ch = conv_params.dst_slices * 4;
  int total_spatial = conv_params.IsTrivialKernelSize() ? 1024 : 256;
  if (dst_ch <= 64) {
    total_spatial *= 4;
  } else if (dst_ch <= 256) {
    total_spatial *= 2;
  }
  const int width = conv_params.different_weights_for_height ? 128 : 32;
  return BHWC(1, total_spatial / width, width, dst_ch);
}

int GetGroupsCount(const BHWC& dst_shape, const int3& wg_size,
                   const int4& block_size) {
  const int dst_slices = DivideRoundUp(dst_shape.c, 4);

  int grid_x = DivideRoundUp(dst_shape.w, block_size.x) * dst_shape.b;
  int grid_y = DivideRoundUp(dst_shape.h, block_size.y);
  int grid_z = DivideRoundUp(dst_slices, block_size.w);

  return DivideRoundUp(grid_x, wg_size.x) * DivideRoundUp(grid_y, wg_size.y) *
         DivideRoundUp(grid_z, wg_size.z);
}

int GetGroupsCountForLinearWH(const BHWC& dst_shape, const int3& wg_size,
                              const int4& block_size) {
  const int dst_slices = DivideRoundUp(dst_shape.c, 4);

  int grid_x = DivideRoundUp(dst_shape.w, block_size.x) * dst_shape.b;
  int grid_y = DivideRoundUp(dst_shape.h, block_size.y);
  int grid_z = DivideRoundUp(dst_slices, block_size.w);

  return DivideRoundUp(grid_x * grid_y, wg_size.x) *
         DivideRoundUp(grid_z, wg_size.y);
}

int GetGroupsCountForLinearWHS(const BHWC& dst_shape, const int3& wg_size,
                               const int4& block_size) {
  const int dst_slices = DivideRoundUp(dst_shape.c, 4);

  int grid_x = DivideRoundUp(dst_shape.w, block_size.x) * dst_shape.b;
  int grid_y = DivideRoundUp(dst_shape.h, block_size.y);
  int grid_z = DivideRoundUp(dst_slices, block_size.w);

  return DivideRoundUp(grid_x * grid_y * grid_z, wg_size.x);
}

int GetMaximumPossibleWavesCount(const AppleInfo& apple_info,
                                 const BHWC& dst_shape) {
  if (apple_info.IsLocalMemoryPreferredOverGlobal()) {
    return GetGroupsCountForLinearWH(dst_shape, {32, 1, 1}, int4(1, 1, 1, 1));
  } else {
    return GetGroupsCountForLinearWHS(dst_shape, {32, 1, 1}, int4(1, 1, 1, 1));
  }
}

int GetRecommendedBlockSize(const AppleInfo& apple_info,
                            const BHWC& dst_shape) {
  const int max_waves = GetMaximumPossibleWavesCount(apple_info, dst_shape);
  const int cu_count = apple_info.GetComputeUnitsCount();
  if (max_waves >= cu_count * 64) {
    return 8;
  } else if (max_waves >= cu_count * 32) {
    return 4;
  } else if (max_waves >= cu_count * 16) {
    return 2;
  } else {
    return 1;
  }
}

struct WorkGroupSizeOption {
  enum class ThreadMapping { kDefault, kLinearSpatial, kLinearAll };
  int3 work_group_size;
  int work_groups_count;
  ThreadMapping thread_mapping;
  float penalty = 1.0f;
};

WorkGroupSizeOption CreateWorkGroupSizeOption(
    const int3& work_group_size,
    WorkGroupSizeOption::ThreadMapping mapping_type, float penalty,
    const BHWC& dst_shape, const int4& block_size) {
  WorkGroupSizeOption wg;
  wg.work_group_size = work_group_size;
  wg.thread_mapping = mapping_type;
  wg.penalty = penalty;
  if (mapping_type == WorkGroupSizeOption::ThreadMapping::kDefault) {
    wg.work_groups_count =
        GetGroupsCount(dst_shape, work_group_size, block_size);
  } else if (mapping_type ==
             WorkGroupSizeOption::ThreadMapping::kLinearSpatial) {
    wg.work_groups_count =
        GetGroupsCountForLinearWH(dst_shape, work_group_size, block_size);
  } else if (mapping_type == WorkGroupSizeOption::ThreadMapping::kLinearAll) {
    wg.work_groups_count =
        GetGroupsCountForLinearWHS(dst_shape, work_group_size, block_size);
  }
  return wg;
}

ConvGeneric::KernelParams GetKernelParamsForA7A8(
    const AppleInfo& apple_info, const ConvGeneric::ConvParams& conv_params,
    const BHWC& dst_shape) {
  const int dst_slices = DivideRoundUp(dst_shape.c, 4);
  int blk_total_size = GetRecommendedBlockSize(apple_info, dst_shape);
  int3 block_size = int3(1, 1, 1);
  if (blk_total_size >= 4 && (dst_slices % 4 == 0 || dst_slices >= 16)) {
    block_size.z = 4;
    blk_total_size /= 4;
  } else if (blk_total_size >= 2 && (dst_slices % 2 == 0 || dst_slices >= 4)) {
    block_size.z = 2;
    blk_total_size /= 2;
  }
  if (blk_total_size >= 4) {
    block_size.x = 2;
    block_size.y = 2;
    blk_total_size /= 4;
  } else if (blk_total_size >= 2) {
    if (dst_shape.w % 2 != 0 && dst_shape.h % 2 == 0) {
      block_size.y = 2;
    } else {
      block_size.x = 2;
    }
    blk_total_size /= 2;
  }

  ConvGeneric::KernelParams kernel_params;
  kernel_params.weights_upload_type =
      ConvGeneric::WeightsUploadType::kLocalMemory;
  kernel_params.src_depth_loop_size = 1;
  kernel_params.block_size.x = block_size.x;
  kernel_params.block_size.y = block_size.y;
  kernel_params.block_size.z = 1;
  kernel_params.block_size.w = block_size.z;

  std::vector<WorkGroupSizeOption> options;
  options.push_back(CreateWorkGroupSizeOption(
      {8, 4, 1}, WorkGroupSizeOption::ThreadMapping::kDefault, 1.0f, dst_shape,
      kernel_params.block_size));
  if (!apple_info.IsFamilyApple1()) {
    options.push_back(CreateWorkGroupSizeOption(
        {4, 4, 1}, WorkGroupSizeOption::ThreadMapping::kDefault, 1.01f,
        dst_shape, kernel_params.block_size));
    options.push_back(CreateWorkGroupSizeOption(
        {4, 2, 1}, WorkGroupSizeOption::ThreadMapping::kDefault, 1.25f,
        dst_shape, kernel_params.block_size));
  }
  if (!conv_params.different_weights_for_height) {
    options.push_back(CreateWorkGroupSizeOption(
        {32, 1, 1}, WorkGroupSizeOption::ThreadMapping::kLinearSpatial, 1.0f,
        dst_shape, kernel_params.block_size));
    if (!apple_info.IsFamilyApple1()) {
      options.push_back(CreateWorkGroupSizeOption(
          {16, 1, 1}, WorkGroupSizeOption::ThreadMapping::kLinearSpatial, 1.01f,
          dst_shape, kernel_params.block_size));
      options.push_back(CreateWorkGroupSizeOption(
          {8, 1, 1}, WorkGroupSizeOption::ThreadMapping::kLinearSpatial, 1.25f,
          dst_shape, kernel_params.block_size));
      options.push_back(CreateWorkGroupSizeOption(
          {32, 1, 1}, WorkGroupSizeOption::ThreadMapping::kLinearAll,
          3.1 * 1.0f, dst_shape, kernel_params.block_size));
      options.push_back(CreateWorkGroupSizeOption(
          {16, 1, 1}, WorkGroupSizeOption::ThreadMapping::kLinearAll,
          3.1 * 1.01f, dst_shape, kernel_params.block_size));
      options.push_back(CreateWorkGroupSizeOption(
          {8, 1, 1}, WorkGroupSizeOption::ThreadMapping::kLinearAll,
          3.1 * 1.25f, dst_shape, kernel_params.block_size));
    }
  }

  float optimum = options[0].work_groups_count * options[0].penalty *
                  options[0].work_group_size.x * options[0].work_group_size.y *
                  options[0].work_group_size.z;
  int optimum_index = 0;
  for (int i = 1; i < options.size(); ++i) {
    float local_optimum = options[i].work_groups_count * options[i].penalty *
                          options[i].work_group_size.x *
                          options[i].work_group_size.y *
                          options[i].work_group_size.z;
    if (local_optimum < optimum) {
      optimum = local_optimum;
      optimum_index = i;
    }
  }

  WorkGroupSizeOption optimum_wg = options[optimum_index];
  if (optimum_wg.thread_mapping ==
      WorkGroupSizeOption::ThreadMapping::kLinearSpatial) {
    kernel_params.linear_spatial = true;
    kernel_params.linear_all = false;
    kernel_params.work_group_size = optimum_wg.work_group_size;
    kernel_params.work_group_launch_order = int3(1, 0, 2);
  } else if (optimum_wg.thread_mapping ==
             WorkGroupSizeOption::ThreadMapping::kLinearAll) {
    kernel_params.linear_spatial = false;
    kernel_params.linear_all = true;
    kernel_params.work_group_size = optimum_wg.work_group_size;
    kernel_params.work_group_launch_order = int3(0, 1, 2);
    kernel_params.weights_upload_type =
        ConvGeneric::WeightsUploadType::kGlobalMemory;
  } else {
    // default 3D workgroup
    kernel_params.linear_spatial = false;
    kernel_params.linear_all = false;
    kernel_params.work_group_size = optimum_wg.work_group_size;
    kernel_params.work_group_launch_order = int3(2, 0, 1);
  }
  int total_elements = kernel_params.block_size.x * kernel_params.block_size.y *
                       kernel_params.block_size.z * kernel_params.block_size.w;
  if (total_elements == 1) {
    if (conv_params.src_slices % 4 == 0) {
      kernel_params.src_depth_loop_size = 4;
    } else if (conv_params.src_slices % 2 == 0) {
      kernel_params.src_depth_loop_size = 2;
    }
  } else if (total_elements == 2) {
    if (conv_params.src_slices % 2 == 0) {
      kernel_params.src_depth_loop_size = 2;
    }
  }
  if (kernel_params.src_depth_loop_size == conv_params.src_slices) {
    kernel_params.need_src_loop = false;
  }
  return kernel_params;
}

ConvGeneric::KernelParams GetKernelParamsForA9AndHigher(
    const GpuInfo& gpu_info, const ConvGeneric::ConvParams& conv_params,
    const BHWC& dst_shape) {
  const AppleInfo& apple_info = gpu_info.apple_info;
  const int dst_slices = DivideRoundUp(dst_shape.c, 4);
  int blk_total_size = GetRecommendedBlockSize(apple_info, dst_shape);
  int3 block_size = int3(1, 1, 1);
  if (blk_total_size >= 2 && apple_info.IsBionic()) {
    if (dst_shape.h % 2 != 0 && dst_shape.w % 2 == 0) {
      block_size.x = 2;
    } else {
      block_size.y = 2;
    }
    blk_total_size /= 2;
  }
  if (blk_total_size >= 4 && (dst_slices % 4 == 0 || dst_slices >= 16)) {
    block_size.z = 4;
    blk_total_size /= 4;
  } else if (blk_total_size >= 2 && (dst_slices % 2 == 0 || dst_slices >= 4)) {
    block_size.z = 2;
    blk_total_size /= 2;
  }
  if (blk_total_size >= 4 && dst_slices == 3) {
    block_size.z = 3;
    blk_total_size /= 4;
  }

  ConvGeneric::KernelParams kernel_params;
  kernel_params.weights_upload_type =
      ConvGeneric::WeightsUploadType::kGlobalMemory;
  kernel_params.work_group_size = int3(8, 4, 1);
  if (dst_slices == 10) {
    block_size.x = 1;
    block_size.y = 1;
    block_size.z = 10;
    kernel_params.work_group_size = int3(64, 1, 1);
    kernel_params.weights_upload_type =
        ConvGeneric::WeightsUploadType::kLocalMemory;
    blk_total_size = 1;
  }
  kernel_params.src_depth_loop_size = 1;
  kernel_params.block_size.x = block_size.x;
  kernel_params.block_size.y = block_size.y;
  kernel_params.block_size.z = 1;
  kernel_params.block_size.w = block_size.z;
  kernel_params.linear_spatial = false;
  kernel_params.linear_all = false;
  kernel_params.work_group_launch_order = int3(2, 0, 1);
  int g1 = GetGroupsCount(dst_shape, kernel_params.work_group_size,
                          kernel_params.block_size);
  int g2 = GetGroupsCountForLinearWH(
      dst_shape,
      {kernel_params.work_group_size.x * kernel_params.work_group_size.y, 1,
       kernel_params.work_group_size.z},
      kernel_params.block_size);
  int g3 = GetGroupsCountForLinearWHS(
      dst_shape,
      {kernel_params.work_group_size.x * kernel_params.work_group_size.y *
           kernel_params.work_group_size.z,
       1, 1},
      kernel_params.block_size);
  if (g2 < g1) {
    kernel_params.linear_spatial = true;
    kernel_params.work_group_size =
        int3(kernel_params.work_group_size.x * kernel_params.work_group_size.y,
             1, kernel_params.work_group_size.z);
    kernel_params.work_group_launch_order = int3(0, 1, 2);
  }
  float precise_threshold = apple_info.IsBionic() ? 1.0f : 1.04f;
  float precise_ratio = static_cast<float>(g2) / static_cast<float>(g3);
  if (g3 > gpu_info.GetMaxWorkGroupsCountForX()) {
    // Disable linear spatial if we don't have enough workgroups in x-dimension.
    precise_ratio = 0.0f;
  }
  if (precise_ratio > precise_threshold) {
    kernel_params.linear_spatial = false;
    kernel_params.linear_all = true;
    kernel_params.work_group_size =
        int3(kernel_params.work_group_size.x * kernel_params.work_group_size.y *
                 kernel_params.work_group_size.z,
             1, 1);
    kernel_params.weights_upload_type =
        ConvGeneric::WeightsUploadType::kGlobalMemory;
  }
  int total_elements = kernel_params.block_size.x * kernel_params.block_size.y *
                       kernel_params.block_size.z * kernel_params.block_size.w;
  if (total_elements == 1) {
    if (conv_params.src_slices % 4 == 0) {
      kernel_params.src_depth_loop_size = 4;
    } else if (conv_params.src_slices % 2 == 0) {
      kernel_params.src_depth_loop_size = 2;
    }
  } else if (total_elements == 2) {
    if (conv_params.src_slices % 2 == 0) {
      kernel_params.src_depth_loop_size = 2;
    }
  }
  if (kernel_params.src_depth_loop_size == conv_params.src_slices) {
    kernel_params.need_src_loop = false;
  }
  return kernel_params;
}

ConvGeneric::KernelParams GetKernelParamsApple(
    const GpuInfo& gpu_info, const ConvGeneric::ConvParams& conv_params,
    const BHWC& dst_shape) {
  ConvGeneric::KernelParams kernel_params;
  if (gpu_info.apple_info.IsLocalMemoryPreferredOverGlobal()) {
    kernel_params =
        GetKernelParamsForA7A8(gpu_info.apple_info, conv_params, dst_shape);
  } else {
    kernel_params =
        GetKernelParamsForA9AndHigher(gpu_info, conv_params, dst_shape);
  }
  kernel_params.fixed_work_group_size = true;
  return kernel_params;
}

ConvGeneric::KernelParams GetKernelParamsIntel(
    const GpuInfo& gpu_info, const ConvGeneric::ConvParams& conv_params,
    const BHWC& dst_shape) {
  ConvGeneric::KernelParams kernel_params;
  kernel_params.linear_spatial = false;
  kernel_params.linear_all = false;
  kernel_params.block_size = int4(1, 1, 1, 1);

  const bool supported_wave_matmul_fp16 =
      !conv_params.Is8Bit() && !conv_params.Is4Bit() &&
      conv_params.precision == CalculationsPrecision::kF16 &&
      conv_params.src_slices % 4 == 0;
  const bool supported_wave_matmul_int8 = conv_params.Is8Bit() &&
                                          conv_params.IsSrcPackedC16() &&
                                          conv_params.src_slices % 8 == 0;
  const bool supported_wave_matmul_int4 = conv_params.Is4Bit() &&
                                          conv_params.IsSrcPackedC32() &&
                                          conv_params.src_slices % 16 == 0;
  if ((supported_wave_matmul_fp16 || supported_wave_matmul_int8 ||
       supported_wave_matmul_int4) &&
      conv_params.groups_count == 1 &&
      gpu_info.SupportsExtension(
          "cl_intel_subgroup_matrix_multiply_accumulate") &&
      gpu_info.SupportsExtension("cl_intel_required_subgroup_size") &&
      gpu_info.SupportsSubGroupWithSize(16) &&
      !gpu_info.SupportsSubGroupWithSize(8)) {
    kernel_params.work_group_launch_order = int3(2, 0, 1);
    if (!conv_params.different_weights_for_height) {
      kernel_params.linear_spatial = true;
      kernel_params.work_group_launch_order = int3(1, 0, 2);
    }
    kernel_params.simd_sizes = {16};
    kernel_params.fixed_work_group_size = false;
    kernel_params.work_group_size = int3(64, 1, 1);
    kernel_params.weights_upload_type =
        ConvGeneric::WeightsUploadType::kIntelWave16MatMul;
    const int task_size_spatial = dst_shape.w * dst_shape.b * dst_shape.h;
    const double task_size = 1.0 * task_size_spatial * conv_params.dst_slices;
    const double task_size_per_cu = task_size / gpu_info.GetComputeUnitsCount();
    int dst_slices_per_thread = 12;
    if (task_size_per_cu <= 512.0) {
      dst_slices_per_thread = 8;
      kernel_params.work_group_size = int3(32, 1, 1);
    }
    if (task_size_per_cu < 256.0) {
      dst_slices_per_thread = 4;
      kernel_params.work_group_size = int3(32, 1, 1);
    }
    if (task_size_per_cu < 64.0) {
      dst_slices_per_thread = 2;
      kernel_params.work_group_size = int3(16, 1, 1);
    }
    if (task_size_per_cu < 16.0) {
      dst_slices_per_thread = 1;
      kernel_params.work_group_size = int3(16, 1, 1);
    }
    kernel_params.block_size = int4(1, 1, 1, dst_slices_per_thread);
    if (conv_params.weights_desc.layout != WeightsLayout::kUnknown) {
      kernel_params.work_group_size = int3(64, 1, 1);
      kernel_params.fixed_work_group_size = true;
      if (conv_params.runtime_check.packed_groups.has_value()) {
        const int average_task_size = DivideRoundUp(
            dst_shape.w, conv_params.runtime_check.packed_groups->num_groups);
        if (average_task_size <= 32) {
          kernel_params.work_group_size = int3(32, 1, 1);
        }
      }
      kernel_params.block_size = int4(1, 1, 1, 1);
      kernel_params.block_size.w = kernel_params.work_group_size.x / 4;
    }
    kernel_params.src_depth_loop_size = 4;
    if (conv_params.Is8Bit()) {
      kernel_params.block_size = int4(1, 1, 1, 12);
      if (task_size_per_cu < 256.0) {
        kernel_params.block_size.w = 4;
      }
      if (task_size_per_cu < 64.0) {
        dst_slices_per_thread = 2;
      }
      kernel_params.src_depth_loop_size = 2;
      kernel_params.work_group_launch_order = int3(0, 1, 2);
      kernel_params.work_group_size =
          kernel_params.linear_spatial ? int3(16, 8, 1) : int3(16, 1, 8);
    }
    if (conv_params.Is4Bit()) {
      kernel_params.block_size = int4(1, 1, 1, 12);
      kernel_params.src_depth_loop_size = 2;
      kernel_params.work_group_launch_order = int3(0, 1, 2);
      kernel_params.work_group_size =
          kernel_params.linear_spatial ? int3(16, 8, 1) : int3(16, 1, 8);
    }
    return kernel_params;
  }

  if (conv_params.different_weights_for_height) {
    kernel_params.work_group_launch_order = int3(0, 1, 2);
    kernel_params.fixed_work_group_size = true;
  } else {
    kernel_params.linear_spatial = true;
    kernel_params.work_group_launch_order = int3(0, 1, 2);
    kernel_params.fixed_work_group_size = true;
  }
  kernel_params.block_size = int4(1, 1, 1, 4);
  kernel_params.src_depth_loop_size = 1;
  kernel_params.weights_upload_type =
      ConvGeneric::WeightsUploadType::kLocalMemory;
  kernel_params.work_group_size = int3(16, 1, 1);
  int max_dst_slices_per_thread = 4;
  {
    const int task_size_spatial = dst_shape.w * dst_shape.b * dst_shape.h;
    const double task_size = 1.0 * task_size_spatial * conv_params.dst_slices;
    const int cu_count =
        gpu_info.IsApiOpenCl() ? gpu_info.GetComputeUnitsCount() : 48;
    const double task_size_per_cu = task_size / cu_count;
    if ((gpu_info.IsApiOpenCl() || gpu_info.IsApiMetal() ||
         gpu_info.IsApiWebGpu()) &&
        conv_params.precision == CalculationsPrecision::kF16 &&
        conv_params.IsTrivialKernelSize() && task_size_per_cu >= 1024.0) {
      max_dst_slices_per_thread = 8;
    }
    if (task_size_per_cu < 128.0) {
      max_dst_slices_per_thread = 2;
    }
    if (task_size_per_cu < 32.0) {
      max_dst_slices_per_thread = 1;
    }
    const int task_uniform_size =
        dst_shape.w * dst_shape.b *
        (conv_params.different_weights_for_height ? 1 : dst_shape.h);
    if (kernel_params.work_group_size.x == 16) {
      const int aligned_task_size = AlignByN(task_uniform_size, 16);
      if (AlignByN(task_uniform_size, 64) == aligned_task_size) {
        kernel_params.work_group_size.x = 64;
      } else if (AlignByN(task_uniform_size, 32) == aligned_task_size) {
        kernel_params.work_group_size.x = 32;
      }
    }
    if (kernel_params.work_group_size.x == 32) {
      const int aligned_task_size = AlignByN(task_uniform_size, 32);
      if (AlignByN(task_uniform_size, 64) == aligned_task_size) {
        kernel_params.work_group_size.x = 64;
      }
    }
  }
  if ((conv_params.dst_slices % 8 == 0 || conv_params.dst_slices >= 64) &&
      max_dst_slices_per_thread >= 8) {
    kernel_params.block_size.w = 8;
  } else if ((conv_params.dst_slices % 4 == 0 || conv_params.dst_slices >= 8) &&
             max_dst_slices_per_thread >= 4) {
    kernel_params.block_size.w = 4;
  } else if ((conv_params.dst_slices % 2 == 0 || conv_params.dst_slices >= 4) &&
             max_dst_slices_per_thread >= 2) {
    kernel_params.block_size.w = 2;
  } else {
    kernel_params.block_size.w =
        std::min(max_dst_slices_per_thread, conv_params.dst_slices);
  }
  if (conv_params.src_slices % 2 == 0 && kernel_params.block_size.w <= 4) {
    kernel_params.src_depth_loop_size = 2;
  }
  if (conv_params.src_slices % 4 == 0 && kernel_params.block_size.w <= 2) {
    kernel_params.src_depth_loop_size = 4;
  }
  if (gpu_info.IsApiMetal() &&
      conv_params.precision != CalculationsPrecision::kF32F16 &&
      gpu_info.SupportsExtension("ucl_wave_memory")) {
    kernel_params.weights_upload_type =
        ConvGeneric::WeightsUploadType::kWaveMemory;
    kernel_params.simd_sizes = {8, 16};
  }
  if (gpu_info.IsApiOpenCl() && gpu_info.opencl_info.IsCLVK() &&
      conv_params.precision != CalculationsPrecision::kF32F16 &&
      gpu_info.SupportsExtension("ucl_wave_memory")) {
    // for CLVK it will work because of specific driver using subgroup
    // size 16
    kernel_params.weights_upload_type =
        ConvGeneric::WeightsUploadType::kWaveMemory;
    kernel_params.simd_sizes = {16};
  }
  if (gpu_info.IsApiOpenCl() && !gpu_info.opencl_info.IsCLVK() &&
      conv_params.precision != CalculationsPrecision::kF32F16 &&
      gpu_info.SupportsExtension("ucl_wave_memory")) {
    if (conv_params.precision == CalculationsPrecision::kF16 &&
        conv_params.IsTrivialKernelSize() && !conv_params.Is8Bit() &&
        gpu_info.SupportsSubGroupWithSize(32) &&
        kernel_params.work_group_size.x >= 32) {
      kernel_params.weights_upload_type =
          ConvGeneric::WeightsUploadType::kWaveMemory;
      kernel_params.simd_sizes = {32};
    } else if (gpu_info.SupportsSubGroupWithSize(16)) {
      kernel_params.weights_upload_type =
          ConvGeneric::WeightsUploadType::kWaveMemory;
      kernel_params.simd_sizes = {16};
    } else {
      // no support of subgroup size control
      // only smallest subgroup size (8) can be used safely, otherwise
      // correctness can not be guaranteed
      // kernel_params.weights_upload_type =
      //    WeightsUploadType::kWaveMemory;
      // kernel_params.simd_size = {8};
    }
  }
  if (gpu_info.IsApiWebGpu() && gpu_info.SupportsExtension("ucl_wave_memory") &&
      (conv_params.precision == CalculationsPrecision::kF32 ||
       (conv_params.precision == CalculationsPrecision::kF16 &&
        gpu_info.webgpu_info.supports_fp16))) {
    kernel_params.weights_upload_type =
        ConvGeneric::WeightsUploadType::kWaveMemory;
    // simd size is determined in the shader.
    kernel_params.simd_sizes.clear();
    for (unsigned int simd_size = gpu_info.webgpu_info.min_subgroup_size;
         simd_size <= gpu_info.webgpu_info.max_subgroup_size; simd_size *= 2) {
      kernel_params.simd_sizes.push_back(simd_size);
    }
    kernel_params.work_group_size.x =
        std::max(kernel_params.work_group_size.x,
                 static_cast<int>(gpu_info.webgpu_info.max_subgroup_size));
  }
  if (kernel_params.weights_upload_type ==
      ConvGeneric::WeightsUploadType::kLocalMemory) {
    // if kLocalMemory used, work-group total size recommended 64
    kernel_params.work_group_size = int3(64, 1, 1);
  }
  if (conv_params.IsSrcPackedC16()) {
    kernel_params.src_depth_loop_size = 4;
    kernel_params.block_size.x = 1;
    kernel_params.block_size.y = 1;
    kernel_params.block_size.z = 1;
    if (max_dst_slices_per_thread == 8) {
      kernel_params.block_size.w = 12;
    } else if (max_dst_slices_per_thread == 4) {
      kernel_params.block_size.w = 6;
    } else if (max_dst_slices_per_thread == 2) {
      kernel_params.block_size.w = 2;
    } else {
      kernel_params.block_size.w = 1;
    }
    kernel_params.work_group_size = int3(128, 2, 1);
    kernel_params.work_group_launch_order = int3(1, 0, 2);
  }
  return kernel_params;
}

ConvGeneric::KernelParams GetKernelParamsNvidia(
    const GpuInfo& gpu_info, const ConvGeneric::ConvParams& conv_params,
    const BHWC& dst_shape) {
  ConvGeneric::KernelParams kernel_params;
  kernel_params.linear_spatial = false;
  kernel_params.linear_all = false;
  kernel_params.block_size = int4(1, 1, 1, 1);

  if (conv_params.different_weights_for_height) {
    kernel_params.work_group_size = int3(32, 1, 1);
    kernel_params.work_group_launch_order = int3(2, 0, 1);
    kernel_params.fixed_work_group_size = true;
  } else {
    kernel_params.linear_spatial = true;
    kernel_params.work_group_size = int3(32, 1, 1);
    kernel_params.work_group_launch_order = int3(1, 0, 2);
    kernel_params.fixed_work_group_size = true;
  }
  kernel_params.block_size = int4(2, 1, 1, 4);
  kernel_params.src_depth_loop_size = 1;
  kernel_params.weights_upload_type =
      ConvGeneric::WeightsUploadType::kLocalMemory;
  if (conv_params.dst_slices % 4 == 0 || conv_params.dst_slices >= 8) {
    kernel_params.block_size.w = 4;
  } else if (conv_params.dst_slices % 2 == 0 || conv_params.dst_slices >= 4) {
    kernel_params.block_size.w = 2;
  } else {
    kernel_params.block_size.w = conv_params.dst_slices;
  }
  {
    int task_size =
        dst_shape.w * dst_shape.b * dst_shape.h * conv_params.dst_slices;
    float task_size_per_cu =
        static_cast<float>(task_size) / gpu_info.GetComputeUnitsCount();
    int block_size = kernel_params.block_size.x * kernel_params.block_size.y *
                     kernel_params.block_size.w;
    float threads_per_cu = task_size_per_cu / block_size;
    float warps_per_cu = threads_per_cu / 32 /*warp_size*/;
    if (warps_per_cu < 8.0f) {
      kernel_params.block_size.x = 1;
    }
    if (warps_per_cu < 4.0f && kernel_params.block_size.w >= 4) {
      kernel_params.block_size.w /= 2;
    }
    if (warps_per_cu < 2.0f && kernel_params.block_size.w >= 2) {
      kernel_params.block_size.w /= 2;
    }
  }
  if (conv_params.src_slices % 2 == 0) {
    kernel_params.src_depth_loop_size = 2;
  }
  if (conv_params.src_slices % 4 == 0 && kernel_params.block_size.w <= 2) {
    kernel_params.src_depth_loop_size = 4;
  }
  if (conv_params.Is8Bit()) {
    kernel_params.block_size = int4(4, 1, 1, 2);
    if (conv_params.src_slices % 4 == 0) {
      kernel_params.src_depth_loop_size = 4;
    } else if (conv_params.src_slices % 2 == 0) {
      kernel_params.src_depth_loop_size = 2;
    } else {
      kernel_params.src_depth_loop_size = 1;
    }
  }
  return kernel_params;
}

ConvGeneric::KernelParams GetKernelParamsPowerVRDXT(
    const GpuInfo& gpu_info, const ConvGeneric::ConvParams& conv_params,
    const BHWC& dst_shape) {
  ConvGeneric::KernelParams kernel_params;
  kernel_params.linear_spatial = !conv_params.different_weights_for_height;
  kernel_params.linear_all = false;
  kernel_params.block_size = int4(1, 1, 1, 1);
  kernel_params.src_depth_loop_size = 1;
  kernel_params.work_group_size = int3(128, 1, 1);
  kernel_params.fixed_work_group_size = true;
  kernel_params.weights_upload_type =
      gpu_info.SupportsExtension("ucl_wave_memory")
          ? ConvGeneric::WeightsUploadType::kWaveMemory
          : ConvGeneric::WeightsUploadType::kLocalMemory;
  kernel_params.simd_sizes = {128};

  bool slices_first = true;
  if (conv_params.Is8Bit()) {
    if (conv_params.IsSrcPackedW4C4()) {
      kernel_params.block_size = int4(4, 1, 1, 2);
      kernel_params.src_depth_loop_size =
          conv_params.src_slices % 4 == 0 ? 4 : 2;
      if (conv_params.src_slices % 2 != 0) {
        kernel_params.src_depth_loop_size = 1;
      }
    } else if (conv_params.IsSrcPackedC16()) {
      kernel_params.block_size = int4(1, 1, 1, 4);
      kernel_params.src_depth_loop_size = 4;
    } else {
      kernel_params.block_size = int4(2, 1, 1, 4);
      kernel_params.src_depth_loop_size = 1;
      kernel_params.weights_upload_type =
          ConvGeneric::WeightsUploadType::kGlobalMemory;
    }
  } else {
    int spatial_max_size = 1;
    int spatial_uniform = dst_shape.b * dst_shape.w;
    if (!conv_params.different_weights_for_height) {
      spatial_uniform *= dst_shape.h;
    }
    const int groups_x1 = DivideRoundUp(spatial_uniform, 128);
    const int groups_x2 = DivideRoundUp(spatial_uniform, 256);
    const int groups_x4 = DivideRoundUp(spatial_uniform, 512);
    if (groups_x2 * 2 * 0.75f <= groups_x1) {
      spatial_max_size = 2;
    }
    if (groups_x4 * 4 * 0.9f <= groups_x1 &&
        !conv_params.different_weights_for_height) {
      spatial_max_size = 4;
    }
    kernel_params.block_size = int4(1, 1, 1, 1);
    std::vector<int4> block_sizes;
    if (spatial_max_size >= 4) {
      block_sizes.push_back(int4(1, 4, 1, 2));
    }
    if (spatial_max_size >= 2) {
      block_sizes.push_back(int4(1, 2, 1, 2));
      block_sizes.push_back(int4(1, 2, 1, 1));
    }
    block_sizes.push_back(int4(1, 1, 1, 1));
    for (auto& block_size : block_sizes) {
      if (conv_params.different_weights_for_height) {
        block_size.x = block_size.y;
        block_size.y = 1;
      }
      if (dst_shape.h % block_size.y != 0) {
        if (block_size.y == 4 && dst_shape.h % 2 == 0) {
          block_size.x = 2;
          block_size.y = 2;
        } else {
          block_size.x = block_size.y;
          block_size.y = 1;
        }
      }
    }
    for (const auto& block_size : block_sizes) {
      if (block_size.w >= conv_params.dst_slices * 2) {
        continue;
      }
      const int wg_count =
          GetGroupsCountForLinearWH(dst_shape, {128, 1, 1}, block_size);
      const int wg_count_per_cu =
          DivideRoundUp(wg_count, gpu_info.GetComputeUnitsCount());
      if (wg_count_per_cu >= 8) {
        kernel_params.block_size = block_size;
        break;
      }
    }
    kernel_params.src_depth_loop_size = conv_params.src_slices % 2 == 0 ? 2 : 1;
    if (!conv_params.src_desc.IsLinear() &&
        !conv_params.src_desc.HasAxis(Axis::kDepth) &&
        (conv_params.kernel_size.x != 1 || conv_params.kernel_size.y != 1)) {
      kernel_params.src_depth_loop_size = 1;
      kernel_params.slices_loop_first = false;
      if (!conv_params.IsXKernelIs1()) {
        kernel_params.unroll_x_loop = true;
      }
      kernel_params.block_size = int4(1, 4, 1, 2);
      if (conv_params.stride.y != 1 || conv_params.stride.x != 1) {
        kernel_params.block_size = int4(1, 2, 1, 4);
      }
      if (conv_params.dst_slices == 1) {
        kernel_params.block_size = int4(1, 4, 1, 1);
      }
      if (conv_params.kernel_size.x >= 4) {
        kernel_params.block_size.y = 2;
      }
      if (conv_params.kernel_size.x >= 8) {
        kernel_params.block_size.y = 1;
      }
      if (spatial_max_size == 1) {
        kernel_params.block_size = int4(1, 1, 1, 4);
      }
    }
  }
  // TODO(sorokin): try to find more generic solution.
  // For winograd convs with sizes(Stable Diffusion):
  //   BHWC(2, 36, 64, 640) -> BHWC(2, 36, 64, 640);
  //   BHWC(2, 36, 16, 1280) -> BHWC(2, 36, 16, 1280);
  if (conv_params.different_weights_for_height) {
    if (dst_shape.b * dst_shape.w <= 128) {
      kernel_params.block_size = int4(8, 1, 1, 1);
      kernel_params.src_depth_loop_size = 1;
      kernel_params.weights_upload_type =
          ConvGeneric::WeightsUploadType::kGlobalMemory;
      kernel_params.fixed_work_group_size = true;
      kernel_params.work_group_size = int3(16, 1, 16);
    }
    if (dst_shape.b * dst_shape.w <= 32) {
      kernel_params.block_size = int4(8, 1, 1, 1);
      kernel_params.src_depth_loop_size = 1;
      kernel_params.weights_upload_type =
          ConvGeneric::WeightsUploadType::kTexturesX4;
      kernel_params.fixed_work_group_size = true;
      kernel_params.work_group_size = int3(4, 1, 32);
    }
  }

  if (slices_first) {
    kernel_params.work_group_launch_order =
        kernel_params.linear_spatial ? int3(1, 0, 2) : int3(2, 0, 1);
  } else {
    kernel_params.work_group_launch_order = int3(0, 1, 2);
  }
  return kernel_params;
}

ConvGeneric::KernelParams GetKernelParamsImgMatMul(
    const GpuInfo& gpu_info, const ConvGeneric::ConvParams& conv_params,
    const BHWC& dst_shape) {
  ConvGeneric::KernelParams kernel_params;
  kernel_params.linear_all = false;
  kernel_params.linear_spatial = !conv_params.different_weights_for_height;
  kernel_params.fixed_work_group_size = true;
  kernel_params.work_group_size = int3(128, 1, 1);
  kernel_params.weights_upload_type =
      ConvGeneric::WeightsUploadType::kLocalMemoryWGLoad;
  if (!conv_params.different_weights_for_height) {
    if (dst_shape.b * dst_shape.w % 128 == 0) {
      kernel_params.linear_spatial = false;
    } else if (dst_shape.b * dst_shape.w % 64 == 0 && dst_shape.h % 2 == 0) {
      kernel_params.linear_spatial = false;
      kernel_params.work_group_size = int3(64, 2, 1);
    } else if (dst_shape.b * dst_shape.w % 32 == 0 && dst_shape.h % 4 == 0) {
      kernel_params.linear_spatial = false;
      kernel_params.work_group_size = int3(32, 4, 1);
    } else if (dst_shape.b * dst_shape.w % 16 == 0 && dst_shape.h % 8 == 0) {
      kernel_params.linear_spatial = false;
      kernel_params.work_group_size = int3(16, 8, 1);
    }
  }
  kernel_params.work_group_launch_order =
      kernel_params.linear_spatial ? int3(1, 0, 2) : int3(2, 0, 1);
  kernel_params.block_size = int4(1, 1, 1, 2);
  kernel_params.src_depth_loop_size = 1;
  if (conv_params.src_slices % 2 == 0) {
    kernel_params.src_depth_loop_size = 2;
  }
  if (conv_params.dst_slices > 32) {
    kernel_params.block_size = int4(1, 1, 1, 4);
  }

  if (!conv_params.src_desc.IsLinear() &&
      !conv_params.src_desc.HasAxis(Axis::kDepth) &&
      (conv_params.kernel_size.x != 1 || conv_params.kernel_size.y != 1)) {
    kernel_params.src_depth_loop_size = 1;
    kernel_params.slices_loop_first = false;
    if (!conv_params.IsXKernelIs1()) {
      kernel_params.unroll_x_loop = true;
    }
  }
  return kernel_params;
}

ConvGeneric::KernelParams GetKernelParamsPowerVR(
    const GpuInfo& gpu_info, const ConvGeneric::ConvParams& conv_params,
    const BHWC& dst_shape) {
  ConvGeneric::KernelParams kernel_params;
  kernel_params.linear_spatial = false;
  kernel_params.linear_all = false;
  kernel_params.block_size = int4(1, 1, 1, 1);

  bool slices_first = !gpu_info.powervr_info.IsImgBxx();
  kernel_params.linear_spatial = !conv_params.different_weights_for_height;
  const int wg_size =
      gpu_info.powervr_info.gpu_version >= PowerVRGpu::kBXE ? 128 : 32;
  kernel_params.work_group_size = int3(wg_size, 1, 1);
  kernel_params.fixed_work_group_size = true;
  kernel_params.block_size = int4(1, 1, 1, 4);
  kernel_params.src_depth_loop_size = 1;
  if (!gpu_info.IsApiOpenCl() ||
      (gpu_info.IsApiOpenCl() && gpu_info.opencl_info.dedicated_local_memory)) {
    if (conv_params.precision == CalculationsPrecision::kF32F16 ||
        !gpu_info.IsApiOpenCl()) {
      kernel_params.weights_upload_type =
          ConvGeneric::WeightsUploadType::kLocalMemory;
    } else {
      kernel_params.weights_upload_type =
          ConvGeneric::WeightsUploadType::kLocalMemoryWGLoad;
    }
  } else {
    kernel_params.weights_upload_type =
        ConvGeneric::WeightsUploadType::kGlobalMemory;
  }
  if (conv_params.dst_slices % 8 == 0 || conv_params.dst_slices >= 32) {
    kernel_params.block_size.w = 8;
  } else if (conv_params.dst_slices % 4 == 0 || conv_params.dst_slices >= 8) {
    kernel_params.block_size.w = 4;
  } else if (conv_params.dst_slices % 2 == 0 || conv_params.dst_slices >= 4) {
    kernel_params.block_size.w = 2;
  } else {
    kernel_params.block_size.w = conv_params.dst_slices;
  }
  if (conv_params.precision == CalculationsPrecision::kF32 &&
      gpu_info.powervr_info.gpu_version >= PowerVRGpu::kBXE) {
    kernel_params.block_size.w = std::min(4, kernel_params.block_size.w);
  }
  if (conv_params.precision == CalculationsPrecision::kF16 ||
      (conv_params.precision == CalculationsPrecision::kF32F16 &&
       gpu_info.powervr_info.IsImgBxx())) {
    kernel_params.block_size.w = std::min(4, kernel_params.block_size.w);
    kernel_params.block_size.x = 2;
    const int block_total_size =
        kernel_params.block_size.x * kernel_params.block_size.y *
        kernel_params.block_size.z * kernel_params.block_size.w;
    const int max_src_depth_loop_size =
        (gpu_info.powervr_info.IsImgBxx() ? 8 : 16) / block_total_size;
    if (conv_params.src_slices % 2 == 0) {
      kernel_params.src_depth_loop_size = 2;
    }
    if (conv_params.src_slices % 4 == 0) {
      kernel_params.src_depth_loop_size = 4;
    }
    if (conv_params.src_slices <= 8) {
      kernel_params.src_depth_loop_size = conv_params.src_slices;
    }
    while (kernel_params.src_depth_loop_size > max_src_depth_loop_size) {
      kernel_params.src_depth_loop_size /= 2;
    }
    if (conv_params.src_slices % kernel_params.src_depth_loop_size != 0) {
      kernel_params.src_depth_loop_size = 1;
    }
  }
  if (gpu_info.powervr_info.IsImgCxx()) {
    kernel_params.work_group_size = int3(128, 1, 1);
    if (!conv_params.different_weights_for_height) {
      if (dst_shape.b * dst_shape.w % 128 == 0) {
        kernel_params.linear_spatial = false;
      } else if (dst_shape.b * dst_shape.w % 64 == 0 && dst_shape.h % 2 == 0) {
        kernel_params.linear_spatial = false;
        kernel_params.work_group_size = int3(64, 2, 1);
      } else if (dst_shape.b * dst_shape.w % 32 == 0 && dst_shape.h % 4 == 0) {
        kernel_params.linear_spatial = false;
        kernel_params.work_group_size = int3(32, 4, 1);
      } else if (dst_shape.b * dst_shape.w % 16 == 0 && dst_shape.h % 8 == 0) {
        kernel_params.linear_spatial = false;
        kernel_params.work_group_size = int3(16, 8, 1);
      }
    }
    kernel_params.weights_upload_type = gpu_info.IsApiOpenCl()
            ? ConvGeneric::WeightsUploadType::kLocalMemoryWGLoad
            : ConvGeneric::WeightsUploadType::kLocalMemory;
    kernel_params.block_size = int4(1, 1, 1, 4);
    kernel_params.src_depth_loop_size = 1;
    if (conv_params.src_slices % 2 == 0) {
      kernel_params.src_depth_loop_size = 2;
    }
    if (conv_params.src_slices % 4 == 0 &&
        conv_params.precision == CalculationsPrecision::kF16) {
      kernel_params.src_depth_loop_size = 4;
    }
    if (conv_params.IsSrcPackedC16()) {
      kernel_params.src_depth_loop_size = 4;
    }
    if (!conv_params.src_desc.IsLinear() &&
        !conv_params.src_desc.HasAxis(Axis::kDepth) &&
        (conv_params.kernel_size.x != 1 || conv_params.kernel_size.y != 1)) {
      kernel_params.src_depth_loop_size = 1;
      kernel_params.slices_loop_first = false;
      if (!conv_params.IsXKernelIs1()) {
        kernel_params.unroll_x_loop = true;
      }
    }
  }
  if (slices_first) {
    kernel_params.work_group_launch_order =
        kernel_params.linear_spatial ? int3(1, 0, 2) : int3(2, 0, 1);
  } else {
    kernel_params.work_group_launch_order = int3(0, 1, 2);
  }
  return kernel_params;
}

ConvGeneric::KernelParams GetKernelParamsAMD(
    const GpuInfo& gpu_info, const ConvGeneric::ConvParams& conv_params,
    const BHWC& dst_shape) {
  ConvGeneric::KernelParams kernel_params;
  kernel_params.linear_spatial = false;
  kernel_params.linear_all = false;
  kernel_params.block_size = int4(1, 1, 1, 1);

  kernel_params.work_group_size = int3(8, 4, 1);
  kernel_params.work_group_launch_order = int3(0, 1, 2);
  kernel_params.fixed_work_group_size = false;

  if (gpu_info.IsApiOpenCl()) {
    kernel_params.weights_upload_type =
        ConvGeneric::WeightsUploadType::kConstantMemory;
  } else {
    kernel_params.weights_upload_type =
        ConvGeneric::WeightsUploadType::kGlobalMemory;
  }
  if (conv_params.dst_slices % 4 == 0 || conv_params.dst_slices >= 8) {
    kernel_params.block_size = int4(2, 2, 1, 4);
  } else if (conv_params.dst_slices % 2 == 0 || conv_params.dst_slices >= 4) {
    kernel_params.block_size = int4(4, 2, 1, 2);
  } else {
    kernel_params.block_size = int4(4, 4, 1, 1);
  }
  auto reduce_block_size_wzyx = [](int4* block_size) {
    if (block_size->w % 2 == 0) {
      block_size->w /= 2;
    } else if (block_size->z % 2 == 0) {
      block_size->z /= 2;
    } else if (block_size->y % 2 == 0) {
      block_size->y /= 2;
    } else if (block_size->x % 2 == 0) {
      block_size->x /= 2;
    }
  };
  if (conv_params.precision != CalculationsPrecision::kF16) {
    reduce_block_size_wzyx(&kernel_params.block_size);
  }
  {
    int task_size =
        dst_shape.w * dst_shape.b * dst_shape.h * conv_params.dst_slices;
    float task_size_per_cu =
        static_cast<float>(task_size) / gpu_info.GetComputeUnitsCount();
    int block_size = kernel_params.block_size.x * kernel_params.block_size.y *
                     kernel_params.block_size.w;
    float threads_per_cu = task_size_per_cu / block_size;
    float warps_per_cu = threads_per_cu / 64;
    if (warps_per_cu < 4.0f) {
      reduce_block_size_wzyx(&kernel_params.block_size);
    }
    if (warps_per_cu < 2.0f) {
      reduce_block_size_wzyx(&kernel_params.block_size);
    }
    if (warps_per_cu < 1.0f) {
      reduce_block_size_wzyx(&kernel_params.block_size);
    }
    if (warps_per_cu < 0.5f) {
      reduce_block_size_wzyx(&kernel_params.block_size);
    }
  }
  int block_size = kernel_params.block_size.x * kernel_params.block_size.y *
                   kernel_params.block_size.w;
  kernel_params.src_depth_loop_size = 1;
  if (block_size <= 4 && conv_params.src_slices % 2 == 0) {
    kernel_params.src_depth_loop_size = 2;
  }
  if (block_size <= 2 && conv_params.src_slices % 4 == 0) {
    kernel_params.src_depth_loop_size = 4;
  }
  if (block_size <= 1 && conv_params.src_slices % 8 == 0) {
    kernel_params.src_depth_loop_size = 8;
  }
  if (conv_params.Is8Bit()) {
    kernel_params.fixed_work_group_size = true;
    kernel_params.work_group_size = int3(32, 1, 1);
    kernel_params.block_size = int4(2, 1, 1, 4);
    kernel_params.src_depth_loop_size = 1;
    kernel_params.work_group_launch_order = int3(2, 0, 1);
    if (conv_params.IsSrcPackedC16()) {
      kernel_params.src_depth_loop_size = 4;
    }
  }
  return kernel_params;
}

ConvGeneric::KernelParams GetKernelParamsMali(
    const GpuInfo& gpu_info, const ConvGeneric::ConvParams& conv_params,
    const BHWC& dst_shape) {
  ConvGeneric::KernelParams kernel_params;
  kernel_params.linear_spatial = false;
  kernel_params.linear_all = false;
  kernel_params.block_size = int4(1, 1, 1, 1);

  kernel_params.work_group_launch_order = int3(0, 1, 2);
  kernel_params.work_group_size = int3(4, 4, 1);
  kernel_params.fixed_work_group_size = false;
  kernel_params.weights_upload_type =
      ConvGeneric::WeightsUploadType::kGlobalMemory;
  MaliInfo mali_info = gpu_info.mali_info;
  int block_size = 2;
  {
    int task_size =
        dst_shape.w * dst_shape.b * dst_shape.h * conv_params.dst_slices;
    block_size = GetRecommendedBlockSizeForConv(gpu_info, conv_params.precision,
                                                task_size);
  }
  if (!conv_params.IsTrivialKernelSize() &&
      (gpu_info.mali_info.IsMidgard() || gpu_info.mali_info.IsBifrost())) {
    block_size = std::min(block_size, 4);
  }
  if (block_size == 8) {
    if (conv_params.dst_slices == 1 || conv_params.dst_slices == 3) {
      kernel_params.block_size = int4(2, 2, 1, 1);
    } else {
      kernel_params.block_size = int4(2, 2, 1, 2);
    }
  } else if (block_size == 4) {
    if (conv_params.dst_slices == 1 || conv_params.dst_slices == 3) {
      kernel_params.block_size = int4(2, 2, 1, 1);
    } else {
      kernel_params.block_size = int4(2, 1, 1, 1);
      if (conv_params.precision == CalculationsPrecision::kF32) {
        if (dst_shape.h == 1 || conv_params.different_weights_for_height) {
          kernel_params.block_size.w = 2;
        } else if (gpu_info.mali_info.generation >= MaliInfo::Gen::kValhallV1) {
          kernel_params.block_size.y = 2;
        } else {
          kernel_params.block_size.w = 2;
        }
      } else {
        kernel_params.block_size.w = 2;
      }
    }
  } else if (block_size == 2) {
    kernel_params.block_size = int4(2, 1, 1, 1);
  } else {
    kernel_params.block_size = int4(1, 1, 1, 1);
  }
  if (conv_params.dst_slices == 10 && block_size >= 4) {
    kernel_params.block_size = int4(2, 1, 1, 5);
  }
  if (block_size == 8 && mali_info.IsValhallGen4() &&
      conv_params.precision == CalculationsPrecision::kF16 &&
      (dst_shape.h == 1 || conv_params.different_weights_for_height)) {
    kernel_params.block_size = int4(2, 1, 1, 2);
  }
  if (conv_params.different_weights_for_height &&
      dst_shape.b * dst_shape.w <= 32 && block_size >= 8) {
    kernel_params.work_group_launch_order = int3(0, 2, 1);
    kernel_params.block_size = int4(2, 1, 1, 4);
    if (mali_info.IsValhallGen1() || mali_info.IsValhallGen2()) {
      kernel_params.weights_upload_type =
          ConvGeneric::WeightsUploadType::kLocalMemory;
      kernel_params.work_group_size = int3(16, 1, 1);
      kernel_params.fixed_work_group_size = true;
    }
  }
  if (conv_params.stride.x != 1 && conv_params.src_slices <= 4) {
    kernel_params.block_size.y =
        std::min(2, kernel_params.block_size.y * kernel_params.block_size.x);
    kernel_params.block_size.x = 1;
  }
  kernel_params.src_depth_loop_size = 1;
  if (conv_params.src_slices % 2 == 0 && block_size <= 2 &&
      !mali_info.IsMidgard()) {
    kernel_params.src_depth_loop_size = 2;
  }
  if (conv_params.src_slices % 2 == 0 && block_size <= 4 &&
      mali_info.IsValhallGen4() &&
      conv_params.precision == CalculationsPrecision::kF32) {
    kernel_params.src_depth_loop_size = 2;
  }
  if (conv_params.src_slices % 4 == 0 && block_size == 1 &&
      !mali_info.IsMidgard() &&
      conv_params.precision == CalculationsPrecision::kF16) {
    kernel_params.src_depth_loop_size = 4;
  }
  if (conv_params.Is8Bit()) {
    kernel_params.work_group_launch_order = int3(0, 1, 2);
    kernel_params.work_group_size = int3(4, 4, 1);
    kernel_params.fixed_work_group_size = false;
    kernel_params.weights_upload_type =
        ConvGeneric::WeightsUploadType::kGlobalMemory;
    if (conv_params.IsSrcPackedW4C4()) {
      kernel_params.block_size = int4(4, 1, 1, 2);
      kernel_params.src_depth_loop_size =
          conv_params.src_slices % 2 == 0 ? 2 : 1;
    } else {
      kernel_params.block_size = int4(2, 1, 1, 4);
      kernel_params.src_depth_loop_size = 1;
    }
  }
  return kernel_params;
}

ConvGeneric::KernelParams GetKernelParamsAdreno(
    const GpuInfo& gpu_info, const ConvGeneric::ConvParams& conv_params,
    const BHWC& dst_shape) {
  ConvGeneric::KernelParams kernel_params;
  kernel_params.linear_spatial = false;
  kernel_params.linear_all = false;
  kernel_params.block_size = int4(1, 1, 1, 1);

  {
    const int wave_size = gpu_info.adreno_info.GetWaveSize(
        conv_params.precision == CalculationsPrecision::kF16);
    const double task_size =
        1.0 * dst_shape.w * dst_shape.b * dst_shape.h * conv_params.dst_slices;
    const double waves =
        task_size / gpu_info.GetComputeUnitsCount() / wave_size;
    if (waves <= 6.0f) {
      kernel_params.block_size = int4(1, 1, 1, 1);
    } else if (waves <= 12.0f) {
      kernel_params.block_size = int4(2, 1, 1, 1);
    } else if (waves <= 24.0f) {
      kernel_params.block_size = int4(2, 1, 1, 2);
    } else {
      kernel_params.block_size = int4(2, 2, 1, 2);
    }
  }
  if (gpu_info.adreno_info.IsAdreno3xx()) {
    if (conv_params.precision == CalculationsPrecision::kF16) {
      kernel_params.block_size = int4(2, 2, 1, 2);
    } else if (conv_params.precision == CalculationsPrecision::kF32F16) {
      kernel_params.block_size = int4(2, 1, 1, 2);
    } else {  // F32
      kernel_params.block_size = int4(2, 2, 1, 1);
    }
  }
  if (conv_params.dst_slices == 10 && dst_shape.w >= 1024) {
    kernel_params.block_size = int4(2, 1, 1, 5);
  }
  kernel_params.work_group_size = int3(8, 2, 1);
  kernel_params.work_group_launch_order = int3(0, 1, 2);
  kernel_params.fixed_work_group_size = false;
  kernel_params.src_depth_loop_size = 1;
  kernel_params.weights_upload_type =
      ConvGeneric::WeightsUploadType::kTexturesX4;
  {  // Update storage if textures can not be allocated.
    WeightsDescription weights_desc{
        /*type=*/conv_params.weights_data_type,
        /*layout=*/WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4,
        /*output_group_size=*/kernel_params.block_size.w,
    };
    OHWDI weights_shape =
        OHWDI(conv_params.dst_slices * 4, conv_params.kernel_size.y,
              conv_params.kernel_size.x, conv_params.kernel_size.z,
              conv_params.src_slices * 4);
    if (conv_params.different_weights_for_height) {
      weights_shape.h = dst_shape.h;
    }
    const auto tex_size = Get2dResourceSize(weights_desc, weights_shape);
    if (tex_size.x > gpu_info.GetMaxImage2DWidth() ||
        tex_size.y > gpu_info.GetMaxImage2DHeight()) {
      kernel_params.weights_upload_type =
          ConvGeneric::WeightsUploadType::kGlobalMemory;
    }
  }
  if (conv_params.Is8Bit()) {
    kernel_params.weights_upload_type =
        ConvGeneric::WeightsUploadType::kGlobalMemory;
    if (conv_params.IsSrcPackedW4C4()) {
      kernel_params.block_size = int4(4, 1, 1, 2);
    }
  }

  return kernel_params;
}

ConvGeneric::KernelParams GetKernelParamsBroadcom(
    const GpuInfo& gpu_info, const ConvGeneric::ConvParams& conv_params,
    const BHWC& dst_shape) {
  ConvGeneric::KernelParams kernel_params;
  kernel_params.linear_spatial = false;
  kernel_params.linear_all = false;
  kernel_params.block_size = int4(1, 1, 1, 1);

  kernel_params.work_group_launch_order = int3(0, 1, 2);
  kernel_params.work_group_size = int3(4, 4, 1);
  kernel_params.fixed_work_group_size = false;
  kernel_params.weights_upload_type =
      ConvGeneric::WeightsUploadType::kGlobalMemory;
  int block_size = 2;

  if (!conv_params.IsTrivialKernelSize()) {
    block_size = std::min(block_size, 4);
  }
  if (block_size == 2) {
    kernel_params.block_size = int4(2, 1, 1, 1);
  } else {
    kernel_params.block_size = int4(1, 1, 1, 1);
  }
  if (conv_params.stride.x != 1 && conv_params.src_slices <= 4) {
    kernel_params.block_size.y =
        std::min(2, kernel_params.block_size.y * kernel_params.block_size.x);
    kernel_params.block_size.x = 1;
  }
  kernel_params.src_depth_loop_size = 1;
  if (conv_params.src_slices % 2 == 0 && block_size <= 2) {
    kernel_params.src_depth_loop_size = 2;
  }
  if (conv_params.src_slices % 2 == 0 && block_size <= 4 &&
      conv_params.precision == CalculationsPrecision::kF32) {
    kernel_params.src_depth_loop_size = 2;
  }
  if (conv_params.src_slices % 4 == 0 && block_size == 1 &&
      conv_params.precision == CalculationsPrecision::kF16) {
    kernel_params.src_depth_loop_size = 4;
  }

  return kernel_params;
}

WeightsDescription GetWeightsDescription(
    const GpuInfo& gpu_info, const ConvGeneric::KernelParams& kernel_params,
    const ConvGeneric::ConvParams& conv_params) {
  if (kernel_params.weights_upload_type ==
      ConvGeneric::WeightsUploadType::kIntelWave16MatMul) {
    WeightsDescription desc;
    desc.type = conv_params.weights_data_type;
    desc.layout = WeightsLayout::kCustomGroups;
    if (conv_params.Is8Bit()) {
      desc.group_sizes.push_back({Axis::kInputChannels, 2});
    } else if (conv_params.Is4Bit()) {
      desc.group_sizes.push_back({Axis::kInputChannels, 4});
    }
    desc.group_sizes.push_back({Axis::kOutputChannels, 4});
    if (conv_params.Is8Bit()) {
      // keep last block in i2o4i2 layout to work with gpu conversion kernel
      desc.group_sizes.push_back({Axis::kInputChannels, 2});
      desc.group_sizes.push_back({Axis::kInputChannels, 8});
    } else {
      desc.group_sizes.push_back({Axis::kInputChannels, 16});
    }
    desc.group_sizes.push_back(
        {Axis::kOutputChannels, kernel_params.block_size.w});
    desc.group_sizes.push_back({Axis::kInputChannels, 0});
    if (conv_params.kernel_size.x != 1) {
      desc.group_sizes.push_back({Axis::kWidth, 0});
    }
    if (conv_params.kernel_size.y != 1) {
      desc.group_sizes.push_back({Axis::kHeight, 0});
    }
    if (conv_params.different_weights_for_height) {
      desc.group_sizes.push_back({Axis::kWidth, 0});
      desc.group_sizes.push_back({Axis::kHeight, 0});
    }
    if (conv_params.kernel_size.z != 1) {
      desc.group_sizes.push_back({Axis::kDepth, 0});
    }
    desc.group_sizes.push_back({Axis::kOutputChannels, 0});
    return desc;
  }

  WeightsDescription desc;
  if (kernel_params.AreWeightsBuffer()) {
    if (gpu_info.IsDotPreferred() || conv_params.Is8Bit()) {
      desc.layout = WeightsLayout::kOSpatialIOGroupO4I4;
    } else {
      desc.layout = WeightsLayout::kOSpatialIOGroupI4O4;
    }
  } else {
    if (gpu_info.IsDotPreferred() || conv_params.Is8Bit()) {
      desc.layout = WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4;
    } else {
      desc.layout = WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4;
    }
  }
  if (!kernel_params.slices_loop_first) {
    desc.layout = WeightsLayout::kCustomGroups;
    desc.group_sizes = {{Axis::kOutputChannels, 4},
                        {Axis::kInputChannels, 4},
                        {Axis::kOutputChannels, kernel_params.block_size.w},
                        {Axis::kWidth, 0},
                        {Axis::kHeight, 0},
                        {Axis::kInputChannels, 0},
                        {Axis::kOutputChannels, 0}};
  }
  if (SupportsImgMatMul(gpu_info, conv_params)) {
    desc.layout = WeightsLayout::kCustomGroups;
    if (conv_params.precision == CalculationsPrecision::kF32) {
      desc.group_sizes = {{Axis::kOutputChannels, 2},
                          {Axis::kInputChannels, 4},
                          {Axis::kOutputChannels, 2}};
    } else {
      desc.group_sizes = {{Axis::kInputChannels, 2},
                          {Axis::kOutputChannels, 2},
                          {Axis::kInputChannels, 2},
                          {Axis::kOutputChannels, 2}};
    }
    desc.group_sizes.push_back(
        {Axis::kOutputChannels, kernel_params.block_size.w});
    if (kernel_params.slices_loop_first) {
      desc.group_sizes.push_back({Axis::kInputChannels, 0});
    }
    if (conv_params.kernel_size.x != 1) {
      desc.group_sizes.push_back({Axis::kWidth, 0});
    }
    if (conv_params.kernel_size.y != 1) {
      desc.group_sizes.push_back({Axis::kHeight, 0});
    }
    if (conv_params.different_weights_for_height) {
      desc.group_sizes.push_back({Axis::kWidth, 0});
      desc.group_sizes.push_back({Axis::kHeight, 0});
    }
    if (conv_params.kernel_size.z != 1) {
      desc.group_sizes.push_back({Axis::kDepth, 0});
    }
    if (!kernel_params.slices_loop_first) {
      desc.group_sizes.push_back({Axis::kInputChannels, 0});
    }
    desc.group_sizes.push_back({Axis::kOutputChannels, 0});
  }

  desc.type = conv_params.weights_data_type;
  if (desc.layout != WeightsLayout::kCustomGroups) {
    desc.output_group_size = kernel_params.block_size.w;
  }
  return desc;
}

ConvGeneric::KernelParams GetKernelParams(
    const GpuInfo& gpu_info, const ConvGeneric::ConvParams& conv_params,
    const BHWC& dst_shape) {
  ConvGeneric::KernelParams kernel_params;
  kernel_params.linear_spatial = false;
  kernel_params.linear_all = false;
  kernel_params.block_size = int4(1, 1, 1, 1);

  if (gpu_info.IsNvidia()) {
    kernel_params = GetKernelParamsNvidia(gpu_info, conv_params, dst_shape);
  } else if (gpu_info.IsPowerVR()) {
    if (SupportsImgMatMul(gpu_info, conv_params)) {
      kernel_params =
          GetKernelParamsImgMatMul(gpu_info, conv_params, dst_shape);
    } else if (gpu_info.powervr_info.gpu_version == PowerVRGpu::kDXT) {
      kernel_params =
          GetKernelParamsPowerVRDXT(gpu_info, conv_params, dst_shape);
    } else {
      kernel_params = GetKernelParamsPowerVR(gpu_info, conv_params, dst_shape);
    }
  } else if (gpu_info.IsAMD()) {
    kernel_params = GetKernelParamsAMD(gpu_info, conv_params, dst_shape);
  } else if (gpu_info.IsMali()) {
    kernel_params = GetKernelParamsMali(gpu_info, conv_params, dst_shape);
  } else if (gpu_info.IsAdreno()) {
    kernel_params = GetKernelParamsAdreno(gpu_info, conv_params, dst_shape);
  } else if (gpu_info.IsIntel()) {
    kernel_params = GetKernelParamsIntel(gpu_info, conv_params, dst_shape);
  } else if (gpu_info.IsApple()) {
    kernel_params = GetKernelParamsApple(gpu_info, conv_params, dst_shape);
  } else if (gpu_info.IsBroadcom()) {
    kernel_params = GetKernelParamsBroadcom(gpu_info, conv_params, dst_shape);
  } else {
    kernel_params.block_size = int4(1, 1, 1, 4);
    kernel_params.work_group_size = int3(8, 2, 1);
    kernel_params.work_group_launch_order = int3(0, 1, 2);
    kernel_params.fixed_work_group_size = false;
    kernel_params.src_depth_loop_size = 1;
    kernel_params.weights_upload_type =
        ConvGeneric::WeightsUploadType::kGlobalMemory;
    if (conv_params.dst_slices % 4 == 0 || conv_params.dst_slices >= 8) {
      kernel_params.block_size.w = 4;
    } else if (conv_params.dst_slices % 2 == 0 || conv_params.dst_slices >= 4) {
      kernel_params.block_size.w = 2;
    } else {
      kernel_params.block_size.w = conv_params.dst_slices;
    }
    if (conv_params.src_slices % 2 == 0) {
      kernel_params.src_depth_loop_size = 2;
    }
    if (conv_params.src_slices % 4 == 0 && kernel_params.block_size.w <= 2) {
      kernel_params.src_depth_loop_size = 4;
    }
  }
  if (gpu_info.vendor == GpuVendor::kUnknown && gpu_info.IsApiWebGpu()) {
    kernel_params.linear_spatial = false;
    kernel_params.linear_all = false;
    kernel_params.fixed_work_group_size = true;
    kernel_params.src_depth_loop_size = 1;
    kernel_params.work_group_launch_order = int3(2, 0, 1);
    kernel_params.work_group_size = int3(8, 4, 1);
    const bool thin_conv =
        conv_params.src_slices <= 16 && conv_params.dst_slices <= 32 &&
        (conv_params.src_slices * conv_params.dst_slices <= 128);
    if (thin_conv) {
      kernel_params.block_size = int4(1, 2, 1, 2);
      kernel_params.weights_upload_type =
          ConvGeneric::WeightsUploadType::kGlobalMemory;
    } else {
      kernel_params.block_size = int4(1, 1, 1, 2);
      kernel_params.weights_upload_type =
          ConvGeneric::WeightsUploadType::kTexturesX4;
    }
  }
  if (conv_params.HasGroups()) {
    const int dst_group_size =
        conv_params.dst_slices / conv_params.groups_count;
    if (dst_group_size % kernel_params.block_size.w != 0) {
      if (kernel_params.block_size.w == 4 && dst_group_size % 2 == 0) {
        kernel_params.block_size.w = 2;
      } else {
        kernel_params.block_size.w = 1;
      }
      if (gpu_info.IsApple()) {
        if (kernel_params.block_size.w == conv_params.dst_slices) {
          kernel_params.need_dst_loop = false;
        }
        const bool use_filters_constants = !kernel_params.need_dst_loop &&
                                           !kernel_params.need_src_loop &&
                                           conv_params.IsTrivialKernelSize();
        if (use_filters_constants) {
          kernel_params.weights_upload_type =
              ConvGeneric::WeightsUploadType::kConstantMemory;
        }
      }
    }
  }
  if (conv_params.runtime_check.src_end_ch_index.has_value()) {
    const int slices_alignment = conv_params.runtime_check.GetSlicesAlignment();
    if (slices_alignment % kernel_params.src_depth_loop_size != 0) {
      for (int i = slices_alignment; i > 0; i /= 2) {
        if (slices_alignment % i == 0 &&
            kernel_params.src_depth_loop_size % i == 0) {
          kernel_params.src_depth_loop_size = i;
          break;
        }
      }
    }
  }
  if (dst_shape.w == 1) {
    kernel_params.block_size.y *= kernel_params.block_size.x;
    kernel_params.block_size.x = 1;
  }
  if (dst_shape.h == 1) {
    kernel_params.block_size.x *= kernel_params.block_size.y;
    kernel_params.block_size.y = 1;
  }
  if (conv_params.Is8Bit() && conv_params.IsSrcPackedW4C4()) {
    if (kernel_params.block_size.x % 4 != 0) {
      kernel_params.block_size.x = 4;
    }
  }

  // TODO: b/319525628 - Fix kConstantMemory for WebGPU.
  if (gpu_info.IsApiWebGpu() &&
      kernel_params.weights_upload_type ==
          ConvGeneric::WeightsUploadType::kConstantMemory) {
    kernel_params.weights_upload_type =
        ConvGeneric::WeightsUploadType::kGlobalMemory;
  }

  if (conv_params.different_weights_for_height) {
    if (kernel_params.weights_upload_type ==
            ConvGeneric::WeightsUploadType::kLocalMemoryWGLoad ||
        kernel_params.weights_upload_type ==
            ConvGeneric::WeightsUploadType::kLocalMemory ||
        kernel_params.weights_upload_type ==
            ConvGeneric::WeightsUploadType::kWaveMemory) {
      kernel_params.work_group_size.x *= kernel_params.work_group_size.y;
      kernel_params.work_group_size.y = 1;
    }
    kernel_params.block_size.x *= kernel_params.block_size.y;
    kernel_params.block_size.y = 1;
  }

  kernel_params.weights_desc =
      GetWeightsDescription(gpu_info, kernel_params, conv_params);

  return kernel_params;
}

bool Bind8BitWeightsAsUint32x4(const GpuInfo& gpu_info,
                               const ConvGeneric::ConvParams& conv_params) {
  return conv_params.Is8Bit() &&
         (gpu_info.IsMali() || gpu_info.IsPowerVR() || gpu_info.IsAMD());
}

void AddWeightsBufferParams(const GpuInfo& gpu_info,
                            const ConvGeneric::ConvParams& conv_params,
                            ConvGeneric::KernelParams& kernel_params) {
  if (conv_params.Is8Bit()) {
    kernel_params.weights_type = DataType::kUint32;
    kernel_params.weights_element_size =
        Bind8BitWeightsAsUint32x4(gpu_info, conv_params) ? 4 : 1;
  } else if (conv_params.Is4Bit()) {
    kernel_params.weights_type = DataType::kUint32;
    kernel_params.weights_element_size = 1;
  } else {
    // float weights
    kernel_params.weights_type = conv_params.weights_data_type;
    kernel_params.weights_element_size = 4;
  }
}

ConvGeneric::KernelParams GetKernelParams(
    const GpuInfo& gpu_info, const ConvGeneric::ConvParams& conv_params,
    const BHWC* dst_shape) {
  auto kernel_params = GetKernelParams(gpu_info, conv_params,
                                       GetOutputShape(dst_shape, conv_params));
  AddWeightsBufferParams(gpu_info, conv_params, kernel_params);
  return kernel_params;
}

ConvGeneric::ConvParams GetConvParams(const OperationDef& definition,
                                      CalculationsPrecision precision,
                                      const Convolution2DAttributes& attr) {
  ConvGeneric::ConvParams conv_params;
  conv_params.stride = int4(attr.strides.w, attr.strides.h, 1, 1);
  conv_params.padding_prepended =
      int4(attr.padding.prepended.w, attr.padding.prepended.h, 0, 0);
  conv_params.padding_appended =
      int4(attr.padding.appended.w, attr.padding.appended.h, 0, 0);
  auto weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  conv_params.kernel_size = int4(weights_shape.w, weights_shape.h, 1, 1);
  conv_params.dilation = int4(attr.dilations.w, attr.dilations.h, 1, 1);
  conv_params.weights_data_type = DeduceDataTypeFromPrecision(precision);
  conv_params.src_desc = definition.src_tensors[0];
  conv_params.precision = precision;
  conv_params.groups_count = attr.groups;
  conv_params.src_slices = DivideRoundUp(weights_shape.i, 4);
  conv_params.dst_slices = DivideRoundUp(weights_shape.o, 4);
  return conv_params;
}
}  // namespace

void ConvGeneric::InitArgs(const OperationDef& definition) {
  if (!conv_params_.IsXKernelIs1()) {
    args_.AddInt("stride_x", conv_params_.stride.x);
    args_.AddInt("padding_x", -conv_params_.padding_prepended.x);
    args_.AddInt("kernel_size_x", conv_params_.kernel_size.x);
    args_.AddInt("dilation_x", conv_params_.dilation.x);
  }
  if (!conv_params_.IsYKernelIs1()) {
    args_.AddInt("stride_y", conv_params_.stride.y);
    args_.AddInt("padding_y", -conv_params_.padding_prepended.y);
    args_.AddInt("kernel_size_y", conv_params_.kernel_size.y);
    args_.AddInt("dilation_y", conv_params_.dilation.y);
  }
  if (definition.src_tensors[0].HasAxis(Axis::kDepth) &&
      !conv_params_.IsZKernelIs1()) {
    args_.AddInt("stride_z", conv_params_.stride.z);
    args_.AddInt("padding_z", -conv_params_.padding_prepended.z);
    args_.AddInt("kernel_size_z", conv_params_.kernel_size.z);
    args_.AddInt("dilation_z", conv_params_.dilation.z);
  }
  args_.AddInt("task_size_b");
  args_.AddInt("task_size_x");
  args_.AddInt("task_size_y");
  args_.AddInt("task_size_z");
  args_.AddInt("task_size_s");
  if (conv_params_.HasGroups()) {
    const int dst_group_size =
        conv_params_.dst_slices / conv_params_.groups_count;
    args_.AddInt("src_group_size", conv_params_.src_slices);
    args_.AddInt("dst_group_size", dst_group_size);
  }
}

void ConvGeneric::GenerateCode(const OperationDef& definition,
                               const GpuInfo& gpu_info) {
  work_group_size_ = kernel_params_.work_group_size;
  work_group_launch_order_ = kernel_params_.work_group_launch_order;
  if (kernel_params_.linear_all) {
    grid_dimension_ = 1;
  } else if (kernel_params_.linear_spatial) {
    grid_dimension_ = 2;
  }

  InitArgs(definition);
  AddSrcTensor("src_tensor", definition.src_tensors[0]);
  AddDstTensor("dst_tensor", definition.dst_tensors[0]);

  ConvCodeGenerator code_gen(gpu_info, conv_params_, kernel_params_);
  code_ = code_gen.GenerateConv();
  if (conv_params_.precision == CalculationsPrecision::kF16 &&
      gpu_info.IsPowerVR()) {
    compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
  if (SupportsImgMatMul(gpu_info, conv_params_)) {
    compiler_options_.push_back(CompilerOptions::kCl20);
  }
  if (gpu_info.IsMali()) {
    compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
    if (conv_params_.IsSrcPackedW4C4() ||
        !conv_params_.different_weights_for_height) {
      compiler_options_.push_back(CompilerOptions::kClRegisterAllocation64);
    }
  }
  if (conv_params_.precision == CalculationsPrecision::kF16 &&
      gpu_info.IsIntel()) {
    compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
  if (conv_params_.Is8Bit()) {
    if (gpu_info.SupportsExtension("cl_qcom_dot_product8")) {
      compiler_options_.push_back(CompilerOptions::kCl20);
    } else if (gpu_info.SupportsExtension("cl_khr_integer_dot_product")) {
      compiler_options_.push_back(CompilerOptions::kCl30);
    } else if (gpu_info.SupportsExtension(
                   "cl_arm_integer_dot_product_accumulate_int8")) {
      compiler_options_.push_back(CompilerOptions::kCl20);
    }
  }
  if (gpu_info.IsAdreno() && gpu_info.adreno_info.IsAdreno3xx() &&
      conv_params_.precision == CalculationsPrecision::kF16 &&
      conv_params_.IsTrivialKernelSize()) {
    compiler_options_.push_back(CompilerOptions::kAdrenoFullSimd);
  }
}

absl::Status ConvGeneric::BindArguments(ArgumentsBinder* args) {
  const int task_size_b = dst_[0]->Batch();
  const int task_size_x =
      DivideRoundUp(dst_[0]->Width(), kernel_params_.block_size.x);
  const int task_size_y =
      DivideRoundUp(dst_[0]->Height(), kernel_params_.block_size.y);
  const int task_size_z =
      DivideRoundUp(dst_[0]->Depth(), kernel_params_.block_size.z);
  const int task_size_s =
      DivideRoundUp(dst_[0]->Slices(), kernel_params_.block_size.w);
  ABSL_RETURN_IF_ERROR(args->SetInt("task_size_b", task_size_b));
  ABSL_RETURN_IF_ERROR(args->SetInt("task_size_x", task_size_x));
  ABSL_RETURN_IF_ERROR(args->SetInt("task_size_y", task_size_y));
  ABSL_RETURN_IF_ERROR(args->SetInt("task_size_z", task_size_z));
  ABSL_RETURN_IF_ERROR(args->SetInt("task_size_s", task_size_s));
  return absl::OkStatus();
}

int3 ConvGeneric::GetGridSize() const {
  const int task_size_b = dst_[0]->Batch();
  int task_size_x =
      DivideRoundUp(dst_[0]->Width(), kernel_params_.block_size.x);
  int task_size_y =
      DivideRoundUp(dst_[0]->Height(), kernel_params_.block_size.y);
  if (conv_params_.runtime_check.packed_groups.has_value()) {
    task_size_x = conv_params_.runtime_check.packed_groups->max_group_size;
    task_size_y = conv_params_.runtime_check.packed_groups->num_groups;
  }
  const int task_size_z =
      DivideRoundUp(dst_[0]->Depth(), kernel_params_.block_size.z);
  const int task_size_s =
      DivideRoundUp(dst_[0]->Slices(), kernel_params_.block_size.w);
  int3 wg;

  if (kernel_params_.linear_all) {
    return int3(
        task_size_x * task_size_b * task_size_y * task_size_z * task_size_s, 1,
        1);
  } else if (kernel_params_.linear_spatial) {
    return int3(task_size_x * task_size_b * task_size_y * task_size_z,
                task_size_s, 1);
  } else {
    return int3(task_size_x * task_size_b, task_size_y * task_size_z,
                task_size_s);
  }
}

std::vector<int3> ConvGeneric::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info) const {
  if (conv_params_.runtime_check.HasValues()) {
    tuning_type = TuningType::kFast;
  }
  if (kernel_params_.weights_upload_type ==
          WeightsUploadType::kLocalMemoryWGLoad ||
      kernel_params_.weights_upload_type == WeightsUploadType::kLocalMemory ||
      kernel_params_.weights_upload_type == WeightsUploadType::kWaveMemory ||
      kernel_params_.fixed_work_group_size) {
    return {work_group_size_};
  }
  if (kernel_params_.weights_upload_type ==
      WeightsUploadType::kIntelWave16MatMul) {
    if (tuning_type == TuningType::kExhaustive) {
      return GetWorkGroupsXMultipleOf(kernel_params_.simd_sizes[0], gpu_info,
                                      kernel_info, grid_size_);
    } else {
      return {work_group_size_};
    }
  }
  return GetPossibleWorkGroupsConv(tuning_type, gpu_info, kernel_info,
                                   grid_size_);
}

void ConvGeneric::AddRuntimeWeightsDef() {
  if (kernel_params_.AreWeightsBuffer()) {
    BufferDescriptor desc;
    desc.element_type = conv_params_.weights_data_type;
    desc.element_size = 4;
    desc.memory_type = GetBufferWeightsMemoryType();

    AddSrcBuffer("weights", desc);
  } else {
    TensorDescriptor desc{conv_params_.weights_data_type,
                          TensorStorageType::kTexture2D, Layout::kHW};
    for (int i = 0; i < 4; ++i) {
      const std::string name = "weights" + std::to_string(i);
      AddSrcTensor(name, desc);
    }
  }
}

ConvGeneric CreateConvGeneric(const GpuInfo& gpu_info,
                              const OperationDef& definition,
                              CalculationsPrecision precision,
                              const Convolution2DAttributes& attr,
                              const BHWC* dst_shape) {
  ConvGeneric result;
  result.conv_params_ = GetConvParams(definition, precision, attr);
  result.conv_params_.weights_desc.layout = WeightsLayout::kUnknown;
  result.kernel_params_ =
      GetKernelParams(gpu_info, result.conv_params_, dst_shape);
  result.conv_params_.has_bias = !attr.bias.data.empty();
  result.GenerateCode(definition, gpu_info);
  result.UploadWeights(GetFloatWeights(attr));
  result.UploadBias(gpu_info, attr.bias);
  return result;
}

ConvGeneric CreateConvGeneric(const GpuInfo& gpu_info,
                              const OperationDef& definition,
                              CalculationsPrecision precision,
                              const FullyConnectedAttributes& attr,
                              const BHWC* dst_shape) {
  ConvGeneric result;
  result.conv_params_.weights_data_type =
      DeduceDataTypeFromPrecision(precision);
  result.conv_params_.src_desc = definition.src_tensors[0];
  result.conv_params_.precision = precision;
  result.conv_params_.src_slices = DivideRoundUp(attr.weights.shape.i, 4);
  result.conv_params_.dst_slices = DivideRoundUp(attr.weights.shape.o, 4);
  result.conv_params_.weights_desc.layout = WeightsLayout::kUnknown;
  result.kernel_params_ =
      GetKernelParams(gpu_info, result.conv_params_, dst_shape);
  result.conv_params_.has_bias = !attr.bias.data.empty();
  result.GenerateCode(definition, gpu_info);
  result.UploadWeights(attr.weights);
  result.UploadBias(gpu_info, attr.bias);
  return result;
}

ConvGeneric CreateConvGenericExternalWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision, const Convolution2DAttributes& attr,
    const TensorDescriptor* bias, const BHWC* dst_shape,
    const TensorDescriptor* src_exp, bool different_weights_for_height,
    const ConvRuntimeCheckDesc& runtime_check) {
  ConvGeneric result;
  result.conv_params_ = GetConvParams(definition, precision, attr);
  result.conv_params_.weights_desc.layout = WeightsLayout::kUnknown;
  if (different_weights_for_height) {
    result.conv_params_.different_weights_for_height = true;
    result.conv_params_.kernel_size = int4(1, 1, 1, 1);
  }
  result.conv_params_.runtime_check = runtime_check;
  result.conv_params_.softmax_input_activation = src_exp != nullptr;
  result.conv_params_.has_bias = bias != nullptr;
  result.kernel_params_ =
      GetKernelParams(gpu_info, result.conv_params_, dst_shape);
  result.GenerateCode(definition, gpu_info);
  result.AddRuntimeWeightsDef();
  if (bias) {
    result.AddSrcTensor("biases", *bias);
  }
  if (src_exp) {
    result.AddSrcTensor("src_exp", *src_exp);
  }
  AddRuntimeParams(result, runtime_check);
  return result;
}

bool SupportsConvGeneric(const GpuInfo& gpu_info,
                         CalculationsPrecision precision,
                         const ExternalWeights& weights) {
  if (!gpu_info.IsApiOpenCl() || !gpu_info.IsIntel()) {
    return false;
  }
  if (precision != CalculationsPrecision::kF16) {
    return false;
  }
  const bool supported_intel_gpu =
      gpu_info.SupportsExtension(
          "cl_intel_subgroup_matrix_multiply_accumulate") &&
      gpu_info.SupportsExtension("cl_intel_required_subgroup_size") &&
      gpu_info.SupportsSubGroupWithSize(16) &&
      !gpu_info.SupportsSubGroupWithSize(8);
  if (!supported_intel_gpu) {
    return false;
  }
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  if (weights.desc.IsLinearLayout()) {
    const bool supported_type = weights.desc.type == DataType::kFloat32 ||
                                weights.desc.type == DataType::kFloat16 ||
                                weights.desc.type == DataType::kUint8 ||
                                weights.desc.type == DataType::kUint4 ||
                                weights.desc.type == DataType::kUint2;
    if (weights.desc.layout == WeightsLayout::kOSpatialIOGroupI4O4 &&
        weights.desc.output_group_size == dst_slices && supported_type) {
      return true;
    }
    return false;
  } else {
    if (weights.desc.layout ==
            WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4 &&
        (weights.desc.type == DataType::kFloat32 ||
         weights.desc.type == DataType::kFloat16) &&
        weights.desc.output_group_size == 1) {
      return true;
    }
    if (weights.desc.layout == WeightsLayout::k2DYIsSpatialIOAndXIsOGroupI4O4 &&
        (weights.desc.type == DataType::kUint8 ||
         weights.desc.type == DataType::kUint4 ||
         weights.desc.type == DataType::kUint2) &&
        weights.desc.output_group_size == dst_slices) {
      return true;
    }
    return false;
  }
}

ConvGeneric CreateConvGenericExternalWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    CalculationsPrecision precision, const ExternalWeights& weights,
    const TensorDescriptor* bias, const BHWC* dst_shape,
    const TensorDescriptor* src_exp, bool different_weights_for_height,
    const ConvRuntimeCheckDesc& runtime_check) {
  ConvGeneric::ConvParams conv_params;
  conv_params.weights_desc = weights.desc;
  conv_params.scale_zp_shape = weights.scale_zp_shape;
  conv_params.has_zero_point = weights.zero_point != nullptr;
  conv_params.src_group_slices =
      DivideRoundUp(weights.shape.i, 4) / weights.scale_zp_shape.i;
  conv_params.weights_data_type = DeduceDataTypeFromPrecision(precision);
  conv_params.src_desc = definition.src_tensors[0];
  conv_params.precision = precision;
  conv_params.src_slices = DivideRoundUp(weights.shape.i, 4);
  conv_params.dst_slices = DivideRoundUp(weights.shape.o, 4);
  conv_params.different_weights_for_height = different_weights_for_height;
  conv_params.runtime_check = runtime_check;
  conv_params.softmax_input_activation = src_exp != nullptr;
  conv_params.has_bias = bias != nullptr;

  ConvGeneric result;
  result.conv_params_ = conv_params;
  result.kernel_params_ =
      GetKernelParams(gpu_info, result.conv_params_, dst_shape);
  result.GenerateCode(definition, gpu_info);

  if (weights.desc.IsLinearLayout()) {
    BufferDescriptor buffer_desc;
    if (SizeInBitsOf(weights.desc.type) >= 16) {
      // float weights
      buffer_desc.element_type = weights.desc.type;
      buffer_desc.element_size = 16;
    } else {
      // quantized weights
      buffer_desc.element_type = DataType::kUint32;
      buffer_desc.element_size = SizeInBitsOf(weights.desc.type) / 2;
    }
    buffer_desc.memory_type = MemoryType::kGlobal;
    result.AddSrcBuffer("weights", buffer_desc);
  } else {
    if (SizeInBitsOf(weights.desc.type) >= 16) {
      // float weights
      // k2DX4I4YIsSpatialIAndXIsOOGroupO4
      TensorDescriptor desc{weights.desc.type, TensorStorageType::kTexture2D,
                            Layout::kHW};
      for (int i = 0; i < 4; ++i) {
        const std::string name = "weights" + std::to_string(i);
        result.AddSrcTensor(name, desc);
      }
    } else {
      // quantized weights
      DataType texture_type = DataType::kUint32;
      if (SizeInBitsOf(weights.desc.type) == 4) {
        texture_type = DataType::kUint16;
      } else if (SizeInBitsOf(weights.desc.type) == 2) {
        texture_type = DataType::kUint8;
      }
      TensorDescriptor desc = TensorDescriptor(
          texture_type, TensorStorageType::kTexture2D, Layout::kHW);
      result.AddSrcTensor("weights", desc);
    }
  }

  fc::AddWeightsScaleZeroPointArguments(weights, &result);

  if (bias) {
    result.AddSrcTensor("biases", *bias);
  }
  if (src_exp) {
    result.AddSrcTensor("src_exp", *src_exp);
  }
  AddRuntimeParams(result, runtime_check);
  return result;
}

ConvGeneric CreateConvGeneric3D(const GpuInfo& gpu_info,
                                const OperationDef& definition,
                                CalculationsPrecision precision,
                                const Convolution3DAttributes& attr,
                                const BHWDC* dst_shape) {
  ConvGeneric result;
  result.conv_params_.stride =
      int4(attr.strides.w, attr.strides.h, attr.strides.d, 1);
  result.conv_params_.padding_prepended =
      int4(attr.padding.prepended.w, attr.padding.prepended.h,
           attr.padding.prepended.d, 0);
  result.conv_params_.padding_appended =
      int4(attr.padding.appended.w, attr.padding.appended.h,
           attr.padding.appended.d, 0);
  result.conv_params_.kernel_size =
      int4(attr.weights.shape.w, attr.weights.shape.h, attr.weights.shape.d, 1);
  result.conv_params_.dilation =
      int4(attr.dilations.w, attr.dilations.h, attr.dilations.d, 1);
  result.conv_params_.weights_data_type =
      DeduceDataTypeFromPrecision(precision);
  result.conv_params_.src_desc = definition.src_tensors[0];
  result.conv_params_.precision = precision;
  result.conv_params_.src_slices = DivideRoundUp(attr.weights.shape.i, 4);
  result.conv_params_.dst_slices = DivideRoundUp(attr.weights.shape.o, 4);
  result.conv_params_.weights_desc.layout = WeightsLayout::kUnknown;
  if (dst_shape) {
    const BHWC shape = BHWC(dst_shape->b, dst_shape->h * dst_shape->d,
                            dst_shape->w, dst_shape->c);
    result.kernel_params_ =
        GetKernelParams(gpu_info, result.conv_params_, &shape);
  } else {
    result.kernel_params_ =
        GetKernelParams(gpu_info, result.conv_params_, nullptr);
  }
  result.conv_params_.has_bias = !attr.bias.data.empty();
  result.GenerateCode(definition, gpu_info);
  result.UploadWeights(attr.weights);
  result.UploadBias(gpu_info, attr.bias);
  return result;
}

bool SupportsConvGenericInt8(const GpuInfo& gpu_info) {
  if (gpu_info.IsApiOpenCl()) {
    if (gpu_info.adreno_info.adreno_gpu == AdrenoGpu::kAdreno650) {
      // Adreno650 has very bad performance in int8, worse than fp16
      return false;
    }
    return gpu_info.SupportsExtension("cl_qcom_dot_product8") ||
           gpu_info.SupportsExtension("cl_khr_integer_dot_product") ||
           gpu_info.SupportsExtension(
               "cl_arm_integer_dot_product_accumulate_int8");
  } else if (gpu_info.IsApiWebGpu()) {
    return gpu_info.SupportsAcceleratedDp4a();
  } else if (gpu_info.IsApiVulkan()) {
    return gpu_info.SupportsExtension("VK_KHR_shader_integer_dot_product");
  } else {
    return false;
  }
}

PackedType GetConvGenericInt8SrcType(const GpuInfo& gpu_info,
                                     const BHWC& src_shape) {
  if (gpu_info.SupportsExtension("cl_qcom_dot_product8")) {
    return PackedType::kUint8W4C4;
  } else if (gpu_info.IsApiOpenCl()) {
    const int src_slices = DivideRoundUp(src_shape.c, 4);
    if (gpu_info.IsMali() && src_shape.w >= 8) {
      return PackedType::kInt8W4C4;
    }
    if (gpu_info.IsPowerVR()) {
      if (gpu_info.powervr_info.IsImgCxx() && src_slices % 4 == 0) {
        return PackedType::kUint8C16;
      }
      if (src_shape.b * src_shape.h * src_shape.w <= 128 * 3 &&
          src_slices % 4 == 0) {
        return PackedType::kUint8C16;
      } else {
        return PackedType::kUint8W4C4;
      }
    }
    if (gpu_info.IsIntel() && src_slices % 4 == 0) {
      return PackedType::kInt8C16;
    }
    return PackedType::kInt8C4;
  } else if (gpu_info.IsApiWebGpu() && !gpu_info.IsIntel()) {
    return PackedType::kInt8W4C4;
  } else if (gpu_info.IsApiVulkan()) {
    return PackedType::kInt8C16;
  } else {
    return PackedType::kInt8C4;
  }
}

bool UseUint8MathForInt8Weights(const GpuInfo& gpu_info) {
  return gpu_info.IsApiOpenCl() && gpu_info.IsPowerVR();
}

void ConvGeneric::UploadWeightsI8AsU8(
    const Tensor<OHWI, DataType::kInt8>& weights) {
  const auto weights_desc = GetWeightsDescription();
  const int elements_count =
      GetTotalElementsCountForLayout(weights_desc, weights.shape);

  std::vector<uint8_t> weights_data(elements_count * SizeOf(weights_desc.type));
  RearrangeWeightsInt8AsUint8(weights, weights_desc,
                              absl::MakeSpan(weights_data), 128, 128u);

  if (kernel_params_.AreWeightsBuffer()) {
    BufferDescriptor desc;
    desc.element_type = kernel_params_.weights_type;
    desc.element_size = kernel_params_.weights_element_size;
    desc.memory_type = GetBufferWeightsMemoryType();
    desc.size = weights_data.size();
    desc.data = std::move(weights_data);
    args_.AddObject("weights",
                    std::make_unique<BufferDescriptor>(std::move(desc)));
  } else {
    uint2 tex_size = Get2dResourceSize(weights_desc, weights.shape);
    int sub_size = SizeOf(weights_desc.type) * 4 * tex_size.x * tex_size.y;
    for (int i = 0; i < 4; ++i) {
      TensorDescriptor desc = CreateConstantHWVec4TensorDescriptor(
          weights_desc.type, TensorStorageType::kTexture2D, tex_size.x,
          tex_size.y, weights_data.data() + sub_size * i);
      args_.AddObject("weights" + std::to_string(i),
                      std::make_unique<TensorDescriptor>(std::move(desc)));
    }
  }
}

void ConvGeneric::UploadWeightsI4(
    const Tensor<OHWI, DataType::kInt8>& weights_i4) {
  const auto weights_desc = GetWeightsDescription();
  const int elements_count =
      GetTotalElementsCountForLayout(weights_desc, weights_i4.shape);

  std::vector<uint8_t> weights_data(elements_count / 2);
  RearrangeWeightsInt4(weights_i4, weights_desc, absl::MakeSpan(weights_data));

  BufferDescriptor desc;
  desc.element_type = kernel_params_.weights_type;
  desc.element_size = kernel_params_.weights_element_size;
  desc.memory_type = MemoryType::kGlobal;
  desc.size = weights_data.size();
  desc.data = std::move(weights_data);
  args_.AddObject("weights",
                  std::make_unique<BufferDescriptor>(std::move(desc)));
}

ConvGeneric CreateConvGenericInt8(const GpuInfo& gpu_info,
                                  const OperationDef& definition,
                                  PackedType src_packed_type,
                                  const Tensor<OHWI, DataType::kInt8>& weights,
                                  const BHWC* dst_shape) {
  ConvGeneric result = ConvGeneric();
  if (UseUint8MathForInt8Weights(gpu_info)) {
    result.conv_params_.weights_data_type = DataType::kUint8;
  } else {
    result.conv_params_.weights_data_type = DataType::kInt8;
  }
  result.conv_params_.src_desc = definition.src_tensors[0];
  result.conv_params_.src_packed_type = src_packed_type;
  result.conv_params_.has_bias = false;
  result.conv_params_.src_slices = DivideRoundUp(weights.shape.i, 4);
  result.conv_params_.dst_slices = DivideRoundUp(weights.shape.o, 4);
  result.conv_params_.weights_desc.layout = WeightsLayout::kUnknown;
  result.kernel_params_ =
      GetKernelParams(gpu_info, result.conv_params_, dst_shape);
  result.GenerateCode(definition, gpu_info);
  if (UseUint8MathForInt8Weights(gpu_info)) {
    result.UploadWeightsI8AsU8(weights);
  } else {
    result.UploadWeights(weights);
  }
  return result;
}

ConvGeneric CreateConvGenericInt8ExternalWeights(const GpuInfo& gpu_info,
                                                 const OperationDef& definition,
                                                 PackedType src_packed_type,
                                                 const OHWI& weights_shape,
                                                 const BHWC* dst_shape) {
  ConvGeneric result = ConvGeneric();
  if (UseUint8MathForInt8Weights(gpu_info)) {
    result.conv_params_.weights_data_type = DataType::kUint8;
  } else {
    result.conv_params_.weights_data_type = DataType::kInt8;
  }
  result.conv_params_.src_desc = definition.src_tensors[0];
  result.conv_params_.src_packed_type = src_packed_type;
  result.conv_params_.has_bias = false;
  result.conv_params_.src_slices = DivideRoundUp(weights_shape.i, 4);
  result.conv_params_.dst_slices = DivideRoundUp(weights_shape.o, 4);
  result.conv_params_.weights_desc.layout = WeightsLayout::kUnknown;
  result.kernel_params_ =
      GetKernelParams(gpu_info, result.conv_params_, dst_shape);
  result.GenerateCode(definition, gpu_info);

  BufferDescriptor weights_desc;
  weights_desc.element_type = result.kernel_params_.weights_type;
  weights_desc.element_size = result.kernel_params_.weights_element_size;
  weights_desc.memory_type = result.GetBufferWeightsMemoryType();
  result.AddSrcBuffer("weights", weights_desc);

  return result;
}

bool SupportsConvGenericInt4(const GpuInfo& gpu_info, const BHWC& src_shape) {
  if (gpu_info.IsApiOpenCl()) {
    const int src_slices = DivideRoundUp(src_shape.c, 4);
    return gpu_info.SupportsExtension(
               "cl_intel_subgroup_matrix_multiply_accumulate") &&
           src_slices % 16 == 0;
  } else {
    return false;
  }
}

PackedType GetConvGenericInt4SrcType(const GpuInfo& gpu_info,
                                     const BHWC& src_shape) {
  return PackedType::kInt4C32;
}

ConvGeneric CreateConvGenericInt4(const GpuInfo& gpu_info,
                                  const OperationDef& definition,
                                  const Tensor<OHWI, DataType::kInt8>& weights,
                                  const BHWC* dst_shape) {
  ConvGeneric result = ConvGeneric();
  result.conv_params_.weights_data_type = DataType::kInt4;
  result.conv_params_.src_desc = definition.src_tensors[0];
  result.conv_params_.src_packed_type = PackedType::kInt4C32;
  result.conv_params_.has_bias = false;
  result.conv_params_.src_slices = DivideRoundUp(weights.shape.i, 4);
  result.conv_params_.dst_slices = DivideRoundUp(weights.shape.o, 4);
  result.conv_params_.weights_desc.layout = WeightsLayout::kUnknown;
  result.kernel_params_ =
      GetKernelParams(gpu_info, result.conv_params_, dst_shape);
  result.GenerateCode(definition, gpu_info);
  result.UploadWeightsI4(weights);
  return result;
}

ConvGeneric CreateConvGenericInt4ExternalWeights(const GpuInfo& gpu_info,
                                                 const OperationDef& definition,
                                                 const OHWI& weights_shape,
                                                 const BHWC* dst_shape) {
  ConvGeneric result = ConvGeneric();
  result.conv_params_.weights_data_type = DataType::kInt4;
  result.conv_params_.src_desc = definition.src_tensors[0];
  result.conv_params_.src_packed_type = PackedType::kInt4C32;
  result.conv_params_.has_bias = false;
  result.conv_params_.src_slices = DivideRoundUp(weights_shape.i, 4);
  result.conv_params_.dst_slices = DivideRoundUp(weights_shape.o, 4);
  result.conv_params_.weights_desc.layout = WeightsLayout::kUnknown;
  result.kernel_params_ =
      GetKernelParams(gpu_info, result.conv_params_, dst_shape);
  result.GenerateCode(definition, gpu_info);

  BufferDescriptor weights_desc;
  weights_desc.element_type = result.kernel_params_.weights_type;
  weights_desc.element_size = result.kernel_params_.weights_element_size;
  result.AddSrcBuffer("weights", weights_desc);
  return result;
}

}  // namespace ml_drift
