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

#include "ml_drift/cl/cl_weights_manager.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ml_drift/cl/cl_operation.h"
#include "ml_drift/cl/environment.h"
#include "ml_drift/cl/tensor.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/gpu_tensor.h"

namespace ml_drift {
namespace cl {
namespace {

absl::Status SubmitGpuOperations(
    std::vector<std::unique_ptr<Tensor>>&& src_tensors,
    std::vector<Tensor*>& dst_tensors,
    std::vector<std::unique_ptr<::ml_drift::GPUOperation>>&& operations,
    Environment& env) {
  const auto& gpu_info = env.GetDevicePtr()->GetInfo();
  for (size_t i = 0; i < operations.size(); ++i) {
    auto operation = std::move(operations[i]);
    auto* src = src_tensors[i].get();
    auto* dst = dst_tensors[i];
    ABSL_RETURN_IF_ERROR(operation->AssembleCode(gpu_info));
    operation->SetSrc(src, 0);
    operation->SetDst(dst, 0);

    ClOperation cl_op;
    cl_op.Init(std::move(operation));
    ABSL_RETURN_IF_ERROR(
        cl_op.Compile(env.GetDevicePtr(), &env.context(), env.program_cache()));
    ABSL_RETURN_IF_ERROR(cl_op.SetSrcTensor(0, src));
    ABSL_RETURN_IF_ERROR(cl_op.SetDstTensor(0, dst));
    ABSL_RETURN_IF_ERROR(cl_op.UpdateParams());
    ABSL_RETURN_IF_ERROR(cl_op.AddToQueue(env.queue()));
  }
  ABSL_RETURN_IF_ERROR(env.queue()->WaitForCompletion());
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<
    std::vector<std::vector<WeightsManager::WeightsPrepOperationInfo>>>
ClWeightsManager::GetBatchesForWeightsPreparation(
    const Environment& env, const ScheduleStrategy schedule_strategy,
    size_t total_shared_tensor_size) {
  std::vector<WeightsPrepOperationInfo> op_infos =
      ConvertWeightsPrepRequestsToOperations(
          env.GetDevicePtr()->GetInfo(),
          std::move(weights_conversion_requests_));
  weights_conversion_requests_.clear();

  return BatchGpuOperations(std::move(op_infos), schedule_strategy,
                            total_shared_tensor_size);
}

absl::StatusOr<absl::flat_hash_map<ValueId, std::unique_ptr<GpuSpatialTensor>>>
ClWeightsManager::PrepareWeightsInBatch(
    Environment& env,
    std::vector<WeightsManager::WeightsPrepOperationInfo>& op_infos) {
  absl::flat_hash_map<ValueId, std::unique_ptr<GpuSpatialTensor>>
      main_model_id_to_tensor;
  const size_t num_ops = op_infos.size();
  std::vector<std::unique_ptr<Tensor>> src_tensors;
  src_tensors.reserve(num_ops);
  std::vector<Tensor*> dst_tensors;
  dst_tensors.reserve(num_ops);
  std::vector<std::unique_ptr<::ml_drift::GPUOperation>> gpu_operations;
  gpu_operations.reserve(num_ops);

  const auto& gpu_info = env.GetDevicePtr()->GetInfo();
  for (auto& op_info : op_infos) {
    auto src_tensor = std::make_unique<Tensor>();
    ABSL_RETURN_IF_ERROR(
        src_tensor->CreateFromDescriptor(op_info.src_desc, env.context()));
    if (gpu_info.IsMali()) {
      ABSL_RETURN_IF_ERROR(src_tensor->WriteData(op_info.data_ptr, env.queue(),
                                                 /*async=*/false));
    } else {
      ABSL_RETURN_IF_ERROR(src_tensor->WriteDataViaStaging(
          op_info.data_ptr, env.queue(), &env.context()));
    }
    src_tensors.push_back(std::move(src_tensor));

    auto dst_tensor = std::make_unique<Tensor>();
    ABSL_RETURN_IF_ERROR(
        dst_tensor->CreateFromDescriptor(op_info.dst_descs[0], env.context()));
    dst_tensors.push_back(dst_tensor.get());
    main_model_id_to_tensor[op_info.main_model_weight_id] =
        std::move(dst_tensor);

    gpu_operations.push_back(std::move(op_info.gpu_operation));
  }

  ABSL_RETURN_IF_ERROR(SubmitGpuOperations(std::move(src_tensors), dst_tensors,
                                           std::move(gpu_operations), env));

  return main_model_id_to_tensor;
}

absl::StatusOr<absl::flat_hash_map<ValueId, std::unique_ptr<GpuSpatialTensor>>>
ClWeightsManager::PrepareWeightsInBatches(
    Environment& env, const ScheduleStrategy schedule_strategy,
    size_t total_shared_tensor_size) {
  ABSL_ASSIGN_OR_RETURN(
      auto batches, GetBatchesForWeightsPreparation(env, schedule_strategy,
                                                    total_shared_tensor_size));
  absl::flat_hash_map<ValueId, std::unique_ptr<GpuSpatialTensor>>
      main_model_id_to_tensor;
  for (auto& batch : batches) {
    ABSL_ASSIGN_OR_RETURN(auto batch_tensors,
                          PrepareWeightsInBatch(env, batch));
    for (auto& [id, tensor] : batch_tensors) {
      main_model_id_to_tensor[id] = std::move(tensor);
    }
  }
  return main_model_id_to_tensor;
}

}  // namespace cl
}  // namespace ml_drift
