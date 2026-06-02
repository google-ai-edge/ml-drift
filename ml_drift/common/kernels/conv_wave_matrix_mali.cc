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

#include "ml_drift/common/kernels/conv_wave_matrix_mali.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernel_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/compiler_options.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/common/task/weights_conversion.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/task/work_group_picking.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace {
ConvWaveMatrixMali::KernelParams GetKernelParams(const MaliInfo& mali_info,
                                                 const OHWI& weights_shape) {
  const int src_slices = DivideRoundUp(weights_shape.i, 4);
  const int dst_slices = DivideRoundUp(weights_shape.o, 4);
  const int src_groups = DivideRoundUp(src_slices, 4);
  const int dst_groups = DivideRoundUp(dst_slices, 4);
  ConvWaveMatrixMali::KernelParams kernel_params;
  kernel_params.slices_first = false;
  kernel_params.dst_s4_block_size = 1;
  kernel_params.dst_w4_block_size = 1;
  if (dst_groups % 2 == 0) {
    kernel_params.dst_s4_block_size = 2;
  }
  kernel_params.src_slices_loop_unroll = 1;
  if (src_groups % 2 == 0 && kernel_params.dst_s4_block_size <= 2) {
    kernel_params.src_slices_loop_unroll = 2;
  }
  if (src_groups % 4 == 0 &&
      (kernel_params.dst_s4_block_size == 1 || mali_info.IsGen5())) {
    kernel_params.src_slices_loop_unroll = 4;
  }
  if (mali_info.IsMaliG1()) {
    kernel_params.src_slices_loop_unroll = 1;
  }
  return kernel_params;
}

std::string GetCode(const GpuInfo& gpu_info, const OperationDef& definition,
                    const ConvWaveMatrixMali::ConvParams& conv_params,
                    const ConvWaveMatrixMali::KernelParams& kernel_params) {
  auto GetGlobalID = [&](int id) {
    int3 work_group_launch_order =
        kernel_params.slices_first ? int3(1, 0, 2) : int3(0, 1, 2);
    int3 launch_remap;
    launch_remap[work_group_launch_order.x] = 0;
    launch_remap[work_group_launch_order.y] = 1;
    launch_remap[work_group_launch_order.z] = 2;
    std::string result;
    const std::string sid = std::to_string(id);
    if (work_group_launch_order[id] == id) {
      return "ucl::GetGlobalId<" + sid + ">()";
    } else {
      return "ucl::GetGroupId<" + std::to_string(launch_remap[id]) +
             ">() * ucl::GetGroupSize<" + sid + ">() + ucl::GetLocalId<" + sid +
             ">()";
    }
  };
  const auto src_desc = definition.src_tensors[0];
  // WG must be [M*16, N, K]
  // every wave handles w16xs4 block, every thread handles w4xs1 block
  // wave_local_id:
  //    0 |    1 |    2 |    3 |    4 |    5 |    6 |    7 | etc
  // w0s0 | w0s1 | w0s2 | w0s3 | w4s0 | w4s1 | w4s2 | w4s3 | etc
  // w1s0 | w1s1 | w1s2 | w1s3 | w5s0 | w5s1 | w5s2 | w5s3 | etc
  // w2s0 | w2s1 | w2s2 | w2s3 | w6s0 | w6s1 | w6s2 | w6s3 | etc
  // w3s0 | w3s1 | w3s2 | w3s3 | w7s0 | w7s1 | w7s2 | w7s3 | etc
  std::string c = "MAIN_FUNCTION($0) {\n";
  c += "  int wave_local_id = (" + GetGlobalID(0) + ") % 16;\n";
  c += "  int wave_id = (" + GetGlobalID(0) + ") / 16;\n";
  c += "  int wave_slice_offset = wave_local_id % 4;\n";
  c += "  int xb_per_wave = wave_local_id / 4;\n";
  c += "  int xb = wave_id * " +
       std::to_string(4 * kernel_params.dst_w4_block_size) +
       " + xb_per_wave;\n";
  c += "  int batch_id = xb % args.dst.Batch();\n";
  c += "  int x0_w0123 = xb / args.dst.Batch();\n";
  for (int b_x = 1; b_x < kernel_params.dst_w4_block_size; b_x++) {
    c += "  int x" + std::to_string(b_x) + "_w0123 = x0_w0123 + " +
         std::to_string(b_x * 4) + ";\n";
  }
  c += "  args.src.SetBatchRef(batch_id);\n";
  c += "  args.dst.SetBatchRef(batch_id);\n";
  c += "  int wave_xb_base = wave_id * " +
       std::to_string(16 * kernel_params.dst_w4_block_size) + ";\n";
  c += "  int y = " + GetGlobalID(2) + ";\n";
  c += "  int wave_dst_s_base = (" + GetGlobalID(1) + ") * " +
       std::to_string(kernel_params.dst_s4_block_size * 4) + ";\n";
  c += R"(
  int DST_S = wave_dst_s_base + wave_slice_offset;

  if (wave_xb_base >= args.src.Width() * 4 * args.dst.Batch() || y >= args.dst.Height() || wave_dst_s_base >= args.dst.Slices()) {
    return;
  }

  __global char16* w_base = args.weights.GetPtr() + wave_dst_s_base * args.src.Slices();
  int src_slice = wave_slice_offset;

  // _s03 means that thread in wave handles slice[thread_id % 4]
)";
  for (int s = 0; s < kernel_params.dst_s4_block_size; s++) {
    for (int b_x = 0; b_x < kernel_params.dst_w4_block_size; b_x++) {
      for (int x = 0; x < 4; x++) {
        const std::string rval = "r_w" + std::to_string(b_x * 4 + x) + "_s" +
                                 std::to_string(s * 4) +
                                 std::to_string(s * 4 + 3);
        c += "  int4 " + rval + " = ucl::Init<int4>(0);\n";
      }
    }
  }
  for (int b_x = 0; b_x < kernel_params.dst_w4_block_size; b_x++) {
    if (src_desc.CanReadOutOfBorder(Axis::WIDTH, gpu_info)) {
      c += "  int src_x" + std::to_string(b_x) + " = x" + std::to_string(b_x) +
           "_w0123;\n";
    } else {
      c += "  int src_x" + std::to_string(b_x) + " = min(x" +
           std::to_string(b_x) + "_w0123, args.src.Width() - 1);\n";
    }
  }
  c += "  do {\n";
  for (int src_block = 0; src_block < kernel_params.src_slices_loop_unroll;
       src_block++) {
    c += "  {\n";
    for (int b_x = 0; b_x < kernel_params.dst_w4_block_size; b_x++) {
      c += "    int4 src_x" + std::to_string(b_x) +
           "_w0123 = args.src.Read(src_x" + std::to_string(b_x) +
           ", y, src_slice);\n";
    }
    c += "    src_slice += 4;\n";
    c += "    char16 w0123;\n";
    for (int dst_s = 0; dst_s < kernel_params.dst_s4_block_size; dst_s++) {
      c += "    w0123 = w_base[wave_local_id];\n";
      c += "    w_base += 16;\n";
      for (int b_x = 0; b_x < kernel_params.dst_w4_block_size; b_x++) {
        for (int x = 0; x < 4; x++) {
          const std::string postfixes_r[] = {".x", ".y", ".z", ".w"};
          const std::string sval = "as_char4(src_x" + std::to_string(b_x) +
                                   "_w0123" + postfixes_r[x] + ")";
          for (int w = 0; w < 4; w++) {
            const std::string postfixes_w[] = {".s0123", ".s4567", ".s89ab",
                                               ".scdef"};
            const std::string wval = "w0123" + postfixes_w[w];
            const std::string rval = "r_w" + std::to_string(b_x * 4 + x) +
                                     "_s" + std::to_string(dst_s * 4) +
                                     std::to_string(dst_s * 4 + 3) +
                                     postfixes_r[w];
            c += "    " + rval + " = arm_matrix_multiply(" + sval + ", " +
                 wval + ", " + rval + ");\n";
          }
        }
      }
    }
    c += "  }\n";
  }
  c += R"(
  } while (src_slice < args.src.Slices());
)";
  for (int s = 0; s < kernel_params.dst_s4_block_size; s++) {
    const std::string s_coord = "DST_S + " + std::to_string(s * 4);
    c += "  if (" + s_coord + " >= args.dst.Slices()) {\n";
    c += "    return;\n";
    c += "  }\n";
    for (int b_x = 0; b_x < kernel_params.dst_w4_block_size; b_x++) {
      for (int x = 0; x < 4; x++) {
        const std::string x_coord =
            "x" + std::to_string(b_x) + "_w0123 * 4 + " + std::to_string(x);
        const std::string rval = "r_w" + std::to_string(b_x * 4 + x) + "_s" +
                                 std::to_string(s * 4) +
                                 std::to_string(s * 4 + 3);
        c += "  if (" + x_coord + " < args.dst.Width()) {\n";
        c += "    args.dst::type res = " + rval + ";\n";
        c += "    args.dst.Write(res, " + x_coord + ", y, " + s_coord + ");\n";
        c += "  }\n";
      }
    }
  }
  c += "}";
  return c;
}
}  // namespace

void ConvWaveMatrixMali::InitBase(const GpuInfo& gpu_info,
                                  const OperationDef& definition,
                                  const OHWI& weights_shape) {
  work_group_size_ = int3(16, 1, 1);
  conv_params_.weights_type = DataType::INT8;
  kernel_params_ = GetKernelParams(gpu_info.mali_info, weights_shape);
  code_ = GetCode(gpu_info, definition, conv_params_, kernel_params_);
  work_group_launch_order_ =
      kernel_params_.slices_first ? int3(1, 0, 2) : int3(0, 1, 2);
  compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);

  AddSrcTensor("src", definition.src_tensors[0]);
  AddDstTensor("dst", definition.dst_tensors[0]);
}

ConvWaveMatrixMali::ConvWaveMatrixMali(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i8) {
  InitBase(gpu_info, definition, weights_i8.shape);

  const WeightsDescription weights_desc = GetWeightsDescription();
  const int elements_count =
      GetTotalElementsCountForLayout(weights_desc, weights_i8.shape);

  BufferDescriptor buffer_desc;
  buffer_desc.element_type = DataType::INT8;
  buffer_desc.element_size = 16;
  buffer_desc.memory_type = MemoryType::GLOBAL;
  buffer_desc.size = elements_count * SizeOf(weights_desc.type);
  buffer_desc.data.resize(buffer_desc.size);
  RearrangeWeights(weights_i8, weights_desc, absl::MakeSpan(buffer_desc.data));
  args_.AddObject("weights",
                  std::make_unique<BufferDescriptor>(std::move(buffer_desc)));
}

ConvWaveMatrixMali::ConvWaveMatrixMali(const GpuInfo& gpu_info,
                                       const OperationDef& definition,
                                       const OHWI& weights_shape) {
  InitBase(gpu_info, definition, weights_shape);

  BufferDescriptor weights_desc;
  weights_desc.element_type = DataType::INT8;
  weights_desc.element_size = 16;
  AddSrcBuffer("weights", weights_desc);
}

int3 ConvWaveMatrixMali::GetGridSize() const {
  const int grid_x = DivideRoundUp(AlignByN(dst_[0]->Width(), 4),
                                   kernel_params_.dst_w4_block_size) *
                     dst_[0]->Batch();
  const int grid_y =
      DivideRoundUp(dst_[0]->Slices(), 4 * kernel_params_.dst_s4_block_size);
  const int grid_z = dst_[0]->Height();
  return int3(grid_x, grid_y, grid_z);
}

std::vector<int3> ConvWaveMatrixMali::GetPossibleKernelWorkGroups(
    TuningType tuning_type, const GpuInfo& gpu_info,
    const KernelInfo& kernel_info) const {
  switch (tuning_type) {
    case TuningType::kExhaustive:
      return GetWorkGroupsXMultipleOf(/*wave_size=*/16, gpu_info, kernel_info,
                                      grid_size_);
    case TuningType::kFast:
    default:
      return {{32, 2, 1}};
  }
}

bool SupportsConvWaveMatrixMaliInt8(const GpuInfo& gpu_info,
                                    const BHWC& src_shape) {
  const int src_slices = DivideRoundUp(src_shape.c, 4);
  return gpu_info.SupportsExtension("cl_arm_matrix_multiply") &&
         src_slices % 4 == 0;
}
}  // namespace ml_drift
