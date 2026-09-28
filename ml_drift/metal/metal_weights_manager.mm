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

#include "ml_drift/metal/metal_weights_manager.h"

#include <sys/types.h>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/metal/common.h"
#include "ml_drift/metal/compute_task.h"
#include "ml_drift/metal/environment.h"
#include "ml_drift/metal/metal_spatial_tensor.h"

namespace ml_drift {
namespace metal {
namespace {

absl::Status SubmitGpuOperations(
    std::vector<std::unique_ptr<MetalSpatialTensor>>&& src_tensors,
    std::vector<MetalSpatialTensor*>& dst_tensors,
    std::vector<std::unique_ptr<::ml_drift::GPUOperation>>&& operations, Environment& env,
    id<MTLCommandQueue> command_queue) {
  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLComputeCommandEncoder> compute_encoder = [command_buffer computeCommandEncoder];
    auto gpu_info = env.GetInfo();
    for (int i = 0; i < operations.size(); ++i) {
      auto operation = std::move(operations[i]);
      auto src = src_tensors[i].get();
      auto dst = dst_tensors[i];
      operation->SetSrc(src, 0);
      operation->SetDst(dst, 0);
      ABSL_RETURN_IF_ERROR(operation->AssembleCode(gpu_info));

      ComputeTask metal_op;
      metal_op.Init(std::move(operation));
      ABSL_RETURN_IF_ERROR(metal_op.Compile(&env));
      ABSL_RETURN_IF_ERROR(metal_op.SetSrcTensor(src, 0));
      ABSL_RETURN_IF_ERROR(metal_op.SetDstTensor(dst, 0));
      ABSL_RETURN_IF_ERROR(metal_op.UpdateParams());
      metal_op.Encode(compute_encoder);
    }
    [compute_encoder endEncoding];
    [command_buffer commit];
    return absl::OkStatus();
  }
}

absl::Status PrepareWeightsInBatchInternal(
    Environment& env, id<MTLCommandQueue> command_queue,
    std::vector<WeightsManager::WeightsPrepOperationInfo>& op_infos,
    absl::flat_hash_map<ValueId, std::unique_ptr<GpuSpatialTensor>>& main_model_id_to_tensor) {
  const size_t num_ops = op_infos.size();
  // Allocate memory for inputs & Outputs.
  std::vector<std::unique_ptr<MetalSpatialTensor>> src_tensors;
  src_tensors.reserve(num_ops);
  std::vector<MetalSpatialTensor*> dst_tensors;
  dst_tensors.reserve(num_ops);
  std::vector<std::unique_ptr<::ml_drift::GPUOperation>> gpu_operations;
  gpu_operations.reserve(num_ops);
  for (auto& op_info : op_infos) {
    auto src_tensor = std::make_unique<MetalSpatialTensor>();
    ABSL_RETURN_IF_ERROR(src_tensor->CreateFromDescriptor(op_info.src_desc, env.device()));
    // Upload raw weights from CPU to GPU.
    ABSL_RETURN_IF_ERROR(src_tensor->WriteData(command_queue, op_info.data_ptr,
                                               /*wait_for_completion=*/false));
    src_tensors.push_back(std::move(src_tensor));
    auto dst_tensor = std::make_unique<MetalSpatialTensor>();
    ABSL_RETURN_IF_ERROR(dst_tensor->CreateFromDescriptor(op_info.dst_descs[0], env.device()));
    dst_tensors.push_back(dst_tensor.get());
    main_model_id_to_tensor[op_info.main_model_weight_id] = std::move(dst_tensor);
    gpu_operations.push_back(std::move(op_info.gpu_operation));
  }
  // Submit the batch of GPU operations to GPU queue.
  ABSL_RETURN_IF_ERROR(SubmitGpuOperations(std::move(src_tensors), dst_tensors,
                                           std::move(gpu_operations), env, command_queue));
  WaitUntilCompleted(command_queue);
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::vector<std::vector<WeightsManager::WeightsPrepOperationInfo>>>
MetalWeightsManager::GetBatchesForWeightsPreparation(Environment& env,
                                                     const ScheduleStrategy schedule_strategy,
                                                     size_t total_shared_tensor_size) {
  // Convert weights conversion requests to GPU operations.
  std::vector<WeightsPrepOperationInfo> op_infos = ConvertWeightsPrepRequestsToOperations(
      env.GetInfo(), std::move(weights_conversion_requests_));
  weights_conversion_requests_.clear();

  // Schedule GPU operations to batches.
  std::vector<std::vector<WeightsPrepOperationInfo>> batches =
      BatchGpuOperations(std::move(op_infos), schedule_strategy, total_shared_tensor_size);
  return batches;
}

absl::StatusOr<absl::flat_hash_map<ValueId, std::unique_ptr<GpuSpatialTensor>>>
MetalWeightsManager::PrepareWeightsInBatch(
    Environment& env, std::vector<WeightsManager::WeightsPrepOperationInfo>& op_infos) {
  absl::flat_hash_map<ValueId, std::unique_ptr<GpuSpatialTensor>> main_model_id_to_tensor;

  @autoreleasepool {
    id<MTLCommandQueue> command_queue = [env.device() newCommandQueue];
    ABSL_RETURN_IF_ERROR(
        PrepareWeightsInBatchInternal(env, command_queue, op_infos, main_model_id_to_tensor));
  }
  return main_model_id_to_tensor;
}

absl::StatusOr<absl::flat_hash_map<ValueId, std::unique_ptr<GpuSpatialTensor>>>
MetalWeightsManager::PrepareWeightsInBatches(Environment& env,
                                             const ScheduleStrategy schedule_strategy,
                                             size_t total_shared_tensor_size) {
  ABSL_ASSIGN_OR_RETURN(auto batches, GetBatchesForWeightsPreparation(env, schedule_strategy,
                                                                      total_shared_tensor_size));

  absl::flat_hash_map<ValueId, std::unique_ptr<GpuSpatialTensor>> main_model_id_to_tensor;
  @autoreleasepool {
    id<MTLCommandQueue> command_queue = [env.device() newCommandQueue];
    for (auto& op_infos : batches) {
      ABSL_RETURN_IF_ERROR(
          PrepareWeightsInBatchInternal(env, command_queue, op_infos, main_model_id_to_tensor));
    }
  }
  return main_model_id_to_tensor;
}

}  // namespace metal
}  // namespace ml_drift
