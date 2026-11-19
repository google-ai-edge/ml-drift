# Copyright 2026 The ML Drift Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import os
import tempfile
import unittest

from safetensors.torch import save_file
from ml_drift.samples.llm import extract_weights_hf
import torch

_SKIP_SUBSTRINGS = ("vision_tower", "multi_modal_projector")


def _text_tensors() -> dict[str, torch.Tensor]:
  g = torch.Generator().manual_seed(0)
  p = "model.language_model."
  return {
      p + "embed_tokens.weight": torch.randn(128, 64, generator=g),
      p + "layers.0.self_attn.q_proj.weight": torch.randn(64, 64, generator=g),
      p + "layers.0.self_attn.q_norm.weight": torch.randn(64, generator=g),
      p + "layers.0.mlp.gate_proj.weight": torch.randn(128, 64, generator=g),
      p + "layers.0.input_layernorm.weight": torch.randn(64, generator=g),
      p + "norm.weight": torch.randn(64, generator=g),
  }


def _multimodal_tensors() -> dict[str, torch.Tensor]:
  g = torch.Generator().manual_seed(1)
  return {
      "model.vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight": (
          torch.randn(128, 64, generator=g)
      ),
      "model.multi_modal_projector.mm_input_projection_weight": torch.randn(
          64, 64, generator=g
      ),
  }


def _write_checkpoint(path: str, tensors: dict[str, torch.Tensor]) -> None:
  save_file(
      {k: v.to(torch.bfloat16).contiguous() for k, v in tensors.items()}, path
  )


def _digests(directory: str) -> dict[str, str]:
  result = {}
  for name in sorted(os.listdir(directory)):
    with open(os.path.join(directory, name), "rb") as f:
      result[name] = hashlib.sha256(f.read()).hexdigest()
  return result


class ExtractWeightsHfTest(unittest.TestCase):

  def _extract(self, tensors: dict[str, torch.Tensor], **kwargs) -> str:
    tmp = self.enterContext(tempfile.TemporaryDirectory())
    ckpt = os.path.join(tmp, "model.safetensors")
    out = os.path.join(tmp, "out")
    _write_checkpoint(ckpt, tensors)
    extract_weights_hf.extract_model_weights(
        model_path=ckpt, output_dir=out, **kwargs
    )
    return out

  def test_skip_tensors_are_not_written(self):
    for quant_kwargs in (
        {},  # No quantization.
        {
            "embedding_quant_bits": 8,
            "attention_quant_bits": 4,
            "feedforward_quant_bits": 4,
            "block_size": 32,
        },
    ):
      with self.subTest(quantized=bool(quant_kwargs)):
        out = self._extract(
            {**_text_tensors(), **_multimodal_tensors()}, **quant_kwargs
        )
        files = os.listdir(out)
        with open(os.path.join(out, "layer_info.txt")) as f:
          layer_info = f.read()

        for s in _SKIP_SUBSTRINGS:
          self.assertFalse([f for f in files if s in f], msg=s)
          self.assertNotIn(s, layer_info)
        for key in _text_tensors():
          self.assertTrue(any(f.startswith(key) for f in files), msg=key)

  def test_text_outputs_unaffected_by_skip_tensors(self):
    quant_kwargs = {
        "embedding_quant_bits": 8,
        "attention_quant_bits": 4,
        "feedforward_quant_bits": 4,
        "block_size": 32,
    }
    text_only = self._extract(_text_tensors(), **quant_kwargs)
    multimodal = self._extract(
        {**_text_tensors(), **_multimodal_tensors()}, **quant_kwargs
    )

    self.assertEqual(_digests(text_only), _digests(multimodal))


if __name__ == "__main__":
  unittest.main()
