// Copyright 2023 The ML Drift Authors.
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

#ifndef ML_DRIFT_SAMPLES_STABLE_DIFFUSION_DIFFUSER_H_
#define ML_DRIFT_SAMPLES_STABLE_DIFFUSION_DIFFUSER_H_

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ml_drift/cl/environment.h"
#include "ml_drift/cl/inference_context.h"
#include "ml_drift/cl/tensor.h"
#include "ml_drift/common/gpu_model_builder.h"
#include "ml_drift/common/status.h"
#include "ml_drift/samples/stable_diffusion/bpe_tokenizer.h"

namespace ml_drift {
namespace cl {
namespace stable_diffusion {

// The runner of various diffusion models.
//
// Example usage:
//
// Diffuser::Config config;
// config.model_type = Diffuser::ModelType::kSd1;
// config.model_dir = "bins/";
// config.lora_dir = "lora/";
// config.seed = 0;
// config.image_height = 512;
// config.image_width = 512;
// ASSIGN_OR_RETURN(auto diffuser, Diffuser::Create(config));
// ASSIGN_OR_RETURN(auto result_image,
//                  diffuser->Diffuse("a cat and a dog", 20, 1337);
class Diffuser {
 public:
  enum class ModelType {
    // Stable diffusion v1 models, including SD 1.4 and 1.5.
    kSd1 = 0,
    // Google Latent Diffusion Model.
    kGldm = 1,
    // Distilled Google Latent Diffusion Model.
    kDistilledGldm = 2,
    // Stable diffusion v2 base models, including SD 2.0 base and 2.1 base.
    kSd2Base = 3,
    // Text-to-Image Generation On-device models.
    kTigo = 4,
    // One step Text-to-Image Generation On-device UFO models.
    kTigoUfo = 5,
  };

  struct Config {
    // Model type. Default to Stable Diffusion v1.
    ModelType model_type = ModelType::kSd1;
    // The model file folder that stores the weights files of text embedding,
    // UNet, and image decoding model.
    std::string model_dir = "";
    // Note: only one of lora_dir and lora_weights_layer_mapping should be
    // set.
    // The LoRA file folder.
    std::string lora_dir = "";
    // The LoRA layer name mapping to the weight buffer position in the file.
    std::map<std::string, const char*> lora_weights_layer_mapping;
    // The LoRA rank.
    int lora_rank = 4;
    // The random seed to control the randomness of the generated image.
    int seed = 0;
    // The target output image size. Must be a multiple of 8.
    int image_width = 512;
    int image_height = 512;
    // Whether to run UNet with the diffusion plugins.
    bool run_unet_with_plugins = false;
    // Only for Tigo v1.2 with masked image input tensor.
    bool run_unet_with_masked_image = false;
    EnvironmentOptions env_options;
  };

  static absl::StatusOr<std::unique_ptr<Diffuser>> Create(const Config& config);

  // The method that runs the end-to-end workflow.
  absl::StatusOr<TensorFloat32> Diffuse(
      const std::string& prompt, int total_steps, std::optional<uint> rand_seed,
      std::optional<float> plugins_strength,
      const std::vector<TensorFloat32>& plugin_tensors = {});

  // The init step to run text embedding and prepare the input tensors for the
  // first reverse diffusion iteration.
  absl::Status RunInitStep(
      const std::string& prompt, int total_steps, std::optional<uint> rand_seed,
      std::optional<float> plugins_strength,
      const std::vector<TensorFloat32>& plugin_tensors = {});
  // The reverse diffusion step.
  absl::Status RunIterationStep(int total_steps, int curr_iteration);
  // The image decoding step to generate output image. The output interleaved
  // RGB image is stored as a ml drift tensor.
  absl::StatusOr<TensorFloat32> RunDecodeStep();

 private:
  class OpHolder {
   public:
    struct ExecutionParams {
      std::vector<Tensor*> src;
      std::vector<Tensor*> dst;
      absl::flat_hash_map<std::string, float> float_params;
      absl::flat_hash_map<std::string, half> half_params;
      absl::flat_hash_map<std::string, int> int_params;
    };

    absl::Status Init(Environment* env, ml_drift::GPUOperation&& operation) {
      return Initialize(env, std::move(operation));
    }

    absl::Status InitElementwiseOneInput(OperationType op_type,
                                         const TensorDescriptor& src_desc,
                                         const TensorDescriptor& dst_desc,
                                         Environment* env);

    absl::Status Execute(CLCommandQueue* queue, const ExecutionParams& params);
    absl::Status Execute(CLCommandQueue* queue, Tensor* src, Tensor* dst);

   private:
    absl::Status Initialize(Environment* env, GPUOperation&& operation);
    ClOperation op_;
  };

  // Text embedding.
  class TextGuidance {
   public:
    absl::Status Init(const Config& runner_config, Environment* env);

    Tensor* GetGuidanceTensor() {
      return inference_context_.GetTensor(dst_.id);
    }

    Tensor* GetTextProjectionTensor() {
      return inference_context_.GetTensor(text_proj_.id);
    }

    absl::Status SetInput(Environment* env,
                          const ml_drift::Tensor<BHWC, DataType::INT32>& src);

    absl::Status Execute(CLCommandQueue* queue) {
      return inference_context_.AddToQueue(queue);
    }

    absl::Status GetOutput(Environment* env, TensorFloat32* dst,
                           TensorFloat32* text_proj = nullptr) {
      RETURN_IF_ERROR(
          inference_context_.GetOutputTensor(dst_.id, env->queue(), dst));
      if (text_proj) {
        RETURN_IF_ERROR(inference_context_.GetOutputTensor(
            text_proj_.id, env->queue(), text_proj));
      }
      return absl::OkStatus();
    }

   private:
    bool openclip_;
    GpuModelBuilder::TensorHandle src_, mask_, dst_, text_proj_;
    InferenceContext inference_context_;
  };

  // UNet for reverse diffusion process.
  class UNet {
   public:
    absl::Status Init(const Config& runner_config, int width, int height,
                      Environment* env);

    Tensor* GetLatentTensor() { return inference_context_.GetTensor(src_.id); }
    Tensor* GetTembTensor() { return inference_context_.GetTensor(temb_.id); }
    Tensor* GetMaskedImageTensor() {
      return inference_context_.GetTensor(masked_image_.id);
    }
    Tensor* GetGuidanceTensor() {
      return inference_context_.GetTensor(guidance_.id);
    }
    Tensor* GetEtaUncondTensor() {
      return inference_context_.GetTensor(eta0_.id);
    }
    Tensor* GetEtaCondTensor() {
      return inference_context_.GetTensor(eta1_.id);
    }
    Tensor* GetPluginsStrengthTensor() {
      return inference_context_.GetTensor(plugins_strength_.id);
    }
    Tensor* GetTextProjectionTensor() {
      return inference_context_.GetTensor(text_proj_.id);
    }

    std::vector<Tensor*> GetPluginTensors() {
      return {
          inference_context_.GetTensor(plugin_tensors_[0].id),
          inference_context_.GetTensor(plugin_tensors_[1].id),
          inference_context_.GetTensor(plugin_tensors_[2].id),
          inference_context_.GetTensor(plugin_tensors_[3].id),
          inference_context_.GetTensor(plugin_tensors_[4].id),
      };
    }
    absl::Status GenerateNoiseInput(CLCommandQueue* queue, uint32_t base_seed);
    absl::Status Execute(CLCommandQueue* queue, int step_index);
    absl::Status LogDebugTensor(CLCommandQueue* queue);

   private:
    GpuModelBuilder::TensorHandle src_, temb_, guidance_, text_proj_, eta0_,
        eta1_, dbg_, plugins_strength_, masked_image_;
    std::vector<GpuModelBuilder::TensorHandle> plugin_tensors_;
    InferenceContext inference_context_;
    OpHolder temb_generation_op_;
    OpHolder noise_op_;
  };

  // Image decoder.
  class Decoder {
   public:
    absl::Status Init(const Config& runner_config, int width, int height,
                      Environment* env);

    Tensor* GetInputTensor() { return inference_context_.GetTensor(src_.id); }

    absl::Status GetOutput(Environment* env, ml_drift::TensorFloat32* dst) {
      return inference_context_.GetOutputTensor(dst_.id, env->queue(), dst);
    }

    absl::Status Execute(CLCommandQueue* queue) {
      return inference_context_.AddToQueue(queue);
    }

   private:
    GpuModelBuilder::TensorHandle src_, dst_;
    InferenceContext inference_context_;
  };

  // The diffusion stepper to remove the predicted noise from the latent tensor
  // after each reverse diffusion step.
  class DiffusionStepper {
   public:
    absl::Status Init(const Config& runner_config, Environment* env);

    absl::Status StepCustomOp(CLCommandQueue* queue, Tensor* xIn,
                              Tensor* etaUncondIn, Tensor* etaCondIn,
                              Tensor* dst, int tIn, int tPrevIn,
                              float guidanceScaleIn);

   private:
    OpHolder custom_op_;
    OpHolder noise_op_;
    Tensor noise_;
    bool run_lcm_stepper_ = false;
    std::vector<half> alphas_;
    std::vector<half> alphas_prev_;

    GPUOperation CreateVPredictionStepOp(const TensorDescriptor& xIn,
                                         const TensorDescriptor& etaUncondIn,
                                         const TensorDescriptor& etaCondIn,
                                         const TensorDescriptor& dst);

    GPUOperation CreateLcmVPredictionStepOp(const TensorDescriptor& xIn,
                                            const TensorDescriptor& etaUncondIn,
                                            const TensorDescriptor& etaCondIn,
                                            const TensorDescriptor& noise,
                                            const TensorDescriptor& dst);
  };

  Config config_;
  std::unique_ptr<Environment> env_;
  std::unique_ptr<BPETokenizer> bpe_tokenizer_;
  std::unique_ptr<TextGuidance> text_guidance_graph_;
  std::unique_ptr<UNet> unet_;
  std::unique_ptr<Decoder> decoder_;
  std::unique_ptr<DiffusionStepper> diffusion_stepper_;
  std::unique_ptr<OpHolder> copier_;
  ml_drift::Tensor<BHWC, DataType::INT32> tokens_;
  Tensor latent_copy_;
  float plugins_strength_;

  Diffuser(const Diffuser::Config& config, std::unique_ptr<Environment> env,
           std::unique_ptr<TextGuidance> text_guidance_graph,
           std::unique_ptr<UNet> unet,
           std::unique_ptr<DiffusionStepper> diffusion_stepper,
           std::unique_ptr<Decoder> decoder, std::unique_ptr<OpHolder> copier,
           Tensor latent_copy);
};

}  // namespace stable_diffusion
}  // namespace cl
}  // namespace ml_drift

#endif  // ML_DRIFT_SAMPLES_STABLE_DIFFUSION_diffuser_H_
