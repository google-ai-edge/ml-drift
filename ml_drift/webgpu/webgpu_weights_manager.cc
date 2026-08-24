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

#include "ml_drift/webgpu/webgpu_weights_manager.h"

#include <sys/types.h>

#include <cstddef>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/common/task/tuning_type.h"
#include "ml_drift/webgpu/compute_task.h"
#include "ml_drift/webgpu/environment.h"
#include "ml_drift/webgpu/spatial_tensor.h"
#include "ml_drift/webgpu/webgpu_api_util.h"
#include "ml_drift/webgpu/webgpu_headers.h"

namespace ml_drift {
namespace webgpu {
namespace {
absl::Status SubmitGpuOperations(
    std::vector<std::unique_ptr<SpatialTensor>>&& src_tensors,
    std::vector<SpatialTensor*>& dst_tensors,
    std::vector<std::unique_ptr<::ml_drift::GPUOperation>>&& operations,
    const Environment& env) {
  wgpu::CommandEncoder encoder = env.device().CreateCommandEncoder();
  wgpu::ComputePassEncoder compute_encoder = encoder.BeginComputePass();
  for (int i = 0; i < operations.size(); ++i) {
    auto operation = std::move(operations[i]);
    auto src = src_tensors[i].get();
    auto dst = dst_tensors[i];
    auto gpu_info = env.GetInfo();
    ABSL_RETURN_IF_ERROR(operation->AssembleCode(gpu_info));
    operation->SetSrc(src, 0);
    operation->SetDst(dst, 0);

    ComputeTask webgpu_op;
    ABSL_RETURN_IF_ERROR(webgpu_op.Init(env, std::move(operation)));
    ABSL_RETURN_IF_ERROR(webgpu_op.SetSrcTensor(0, src));
    ABSL_RETURN_IF_ERROR(webgpu_op.SetDstTensor(0, dst));
    ABSL_RETURN_IF_ERROR(webgpu_op.Update(env.device()));
    webgpu_op.UpdateGpuObjectBindings(env.device());
    ABSL_RETURN_IF_ERROR(webgpu_op.TuneAndCompile(env, TuningType::kFast));
    ABSL_RETURN_IF_ERROR(webgpu_op.Encode(compute_encoder));
  }
  compute_encoder.End();
  wgpu::CommandBuffer cb = encoder.Finish();
  env.queue().Submit(1, &cb);
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<
    std::vector<std::vector<WeightsManager::WeightsPrepOperationInfo>>>
WebGpuWeightsManager::GetBatchesForWeightsPreparation(
    const Environment& env, const ScheduleStrategy schedule_strategy,
    size_t total_shared_tensor_size) {
  std::vector<WeightsManager::WeightsPrepOperationInfo> op_infos =
      ConvertWeightsPrepRequestsToOperations(
          env.GetInfo(), std::move(weights_conversion_requests_));
  weights_conversion_requests_.clear();

  return BatchGpuOperations(std::move(op_infos), schedule_strategy,
                            total_shared_tensor_size);
}

absl::StatusOr<absl::flat_hash_map<ValueId, std::unique_ptr<GpuSpatialTensor>>>
WebGpuWeightsManager::PrepareWeightsInBatch(
    const Environment& env,
    std::vector<WeightsManager::WeightsPrepOperationInfo>& op_infos) {
  absl::flat_hash_map<ValueId, std::unique_ptr<GpuSpatialTensor>>
      main_model_id_to_tensor;
  const size_t num_ops = op_infos.size();
  // Allocate memory for inputs & Outputs.
  std::vector<std::unique_ptr<SpatialTensor>> src_tensors;
  src_tensors.reserve(num_ops);
  std::vector<SpatialTensor*> dst_tensors;
  dst_tensors.reserve(num_ops);
  std::vector<std::unique_ptr<::ml_drift::GPUOperation>> gpu_operations;
  gpu_operations.reserve(num_ops);
  for (auto& op_info : op_infos) {
    auto src_tensor = std::make_unique<SpatialTensor>();
    ABSL_RETURN_IF_ERROR(
        src_tensor->CreateFromDescriptor(env.device(), op_info.src_desc));
    ABSL_RETURN_IF_ERROR(
        src_tensor->WriteDataViaStaging(env, op_info.data_ptr));
    src_tensors.push_back(std::move(src_tensor));

    auto dst_tensor = std::make_unique<SpatialTensor>();
    ABSL_RETURN_IF_ERROR(
        dst_tensor->CreateFromDescriptor(env.device(), op_info.dst_descs[0]));
    dst_tensors.push_back(dst_tensor.get());
    main_model_id_to_tensor[op_info.main_model_weight_id] =
        std::move(dst_tensor);

    gpu_operations.push_back(std::move(op_info.gpu_operation));
  }
  // Submit the batch of GPU operations to GPU queue.
  ABSL_RETURN_IF_ERROR(SubmitGpuOperations(std::move(src_tensors), dst_tensors,
                                           std::move(gpu_operations), env));
  ABSL_RETURN_IF_ERROR(
      WaitUntilCompleted(env.queue(), env.device(), absl::Seconds(10)));

  return main_model_id_to_tensor;
}

absl::StatusOr<absl::flat_hash_map<ValueId, std::unique_ptr<GpuSpatialTensor>>>
WebGpuWeightsManager::PrepareWeightsInBatches(
    const Environment& env, const ScheduleStrategy schedule_strategy,
    size_t total_shared_tensor_size) {
  ABSL_ASSIGN_OR_RETURN(
      auto batches, GetBatchesForWeightsPreparation(env, schedule_strategy,
                                                    total_shared_tensor_size));
  absl::flat_hash_map<ValueId, std::unique_ptr<GpuSpatialTensor>>
      main_model_id_to_tensor;
  for (auto& op_infos : batches) {
    ABSL_ASSIGN_OR_RETURN(auto tensor_map,
                          PrepareWeightsInBatch(env, op_infos));
    main_model_id_to_tensor.insert(std::make_move_iterator(tensor_map.begin()),
                                   std::make_move_iterator(tensor_map.end()));
  }
  return main_model_id_to_tensor;
}

}  // namespace webgpu
}  // namespace ml_drift
