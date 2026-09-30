# Copyright 2025 The ML Drift Authors.
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

"""Extracts weights from a HuggingFace model into separate binary files."""

import argparse
import os
from typing import Optional, Tuple

import numpy as np
from safetensors.torch import load_file as safetensors_load_file
import torch


def get_output_filename(key: str) -> str:
  """Determines the output filename based on the model's tensor key."""
  if (
      key.endswith(".weight")
      or key.endswith(".bias")
      or key.endswith(".scale")
      or key.endswith("_quantized_scale")
      or key.endswith("_quantized_zp")
  ):
    return key
  else:
    # For other parameters (e.g., norms without specific .weight/.bias suffixes,
    # or other custom tensors), append .bin for a clear file extension.
    return key + ".bin"


def get_layer_type(layer_name: str) -> str:
  """Gets the layer category for a given layer name."""
  ffn_layers = ["mlp"]
  attn_layers = ["self_attn"]
  emb_layers = ["embed_tokens", "embedder", "lm_head"]
  layer_norms = [
      "input_layernorm",
      "post_attention_layernorm",
      "final_layernorm",
      "model.norm.weight",
      "pre_feedforward_layernorm",
      "post_feedforward_layernorm",
  ]
  lora_layers = ["lora"]
  skip_layers = ["multi_modal_projector", "vision_tower"]

  if any(sub_name in layer_name for sub_name in skip_layers):
    return "SKIP"
  if any(sub_name in layer_name for sub_name in lora_layers):
    return "LORA"
  if any(sub_name in layer_name for sub_name in attn_layers):
    return "ATTENTION"
  if any(sub_name in layer_name for sub_name in ffn_layers):
    return "FEEDFORWARD"
  if any(sub_name in layer_name for sub_name in emb_layers):
    return "EMBEDDING"
  if any(sub_name in layer_name for sub_name in layer_norms):
    return "LAYER_NORM"
  return "NONE"


# Maximum number of elements to process per chunk during scale search (256 Ki
# float32 elements = 1 MiB buffer) to bound peak RAM for broadcast arrays.
MAX_CHUNK_ELEMENTS = 256 * 1024


def _optimize_scale_search(
    blocks: np.ndarray,
    bound: np.ndarray,
    scale_bound: float,
    min_q: int,
    max_q: int,
    alphas: np.ndarray,
    chunk_size: int = 65536,
    max_chunk_elements: int = MAX_CHUNK_ELEMENTS,
) -> Tuple[np.ndarray, np.ndarray]:
  """Performs grid-search over scale factors to minimize MSE per block/channel.

  Args:
    blocks: 2D array of shape (N, D) containing the values to quantize.
    bound: 2D array of shape (N, 1) containing max(abs(blocks)).
    scale_bound: Nominal max value (e.g., 7.0 for int4, 127.0 for int8).
    min_q: Minimum quantized integer (e.g., -8 for int4, -128 for int8).
    max_q: Maximum quantized integer (e.g., 7 for int4, 127 for int8).
    alphas: 1D array of candidate scaling factors (e.g., linspace(1.0, 0.5, 7)).
    chunk_size: Max number of blocks to process at once to bound peak memory.
    max_chunk_elements: Max total elements (chunk * D) to bound broadcast RAM.

  Returns:
    Tuple of (quantized_int8_array, scale_array), where quantized_int8_array has
    shape (N, D) and scale_array has shape (N, 1).
  """
  num_items = blocks.shape[0]
  out_scales = np.empty((num_items, 1), dtype=np.float32)
  out_q = np.empty_like(blocks, dtype=np.int8)
  effective_chunk = min(
      chunk_size, max(1, max_chunk_elements // max(1, blocks.shape[1]))
  )

  for start in range(0, num_items, effective_chunk):
    end = min(start + effective_chunk, num_items)
    c_blocks = blocks[start:end]
    c_bounds = bound[start:end]

    cand_scales = (c_bounds[:, None, :] * alphas[None, :, None]) / scale_bound
    cand_scales = np.where(cand_scales == 0.0, 1.0, cand_scales)

    cq = np.clip(np.round(c_blocks[:, None, :] / cand_scales), min_q, max_q)
    crecon = cq * cand_scales
    cmse = np.mean((c_blocks[:, None, :] - crecon) ** 2, axis=-1)
    best_idx = np.argmin(cmse, axis=1)

    chosen_scales = np.take_along_axis(
        cand_scales[:, :, 0], best_idx[:, None], axis=1
    )
    chosen_q = np.take_along_axis(cq, best_idx[:, None, None], axis=1)[
        :, 0, :
    ].astype(np.int8)

    out_scales[start:end] = chosen_scales
    out_q[start:end] = chosen_q

  return out_q, out_scales


def quantize_tensor_int8_sym(
    tensor_val: np.ndarray,
    quant_axis: int,
    optimize_scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
  """Performs symmetric 8-bit channelwise quantization along quant_axis."""
  if optimize_scale:
    transposed = np.moveaxis(tensor_val, quant_axis, -1)
    orig_trans_shape = transposed.shape
    channels = transposed.reshape(-1, orig_trans_shape[-1])
    bound = np.max(np.abs(channels), axis=-1, keepdims=True)
    alphas = np.linspace(1.0, 0.7, num=7, dtype=np.float32)
    qvar, scale = _optimize_scale_search(
        channels, bound, 127.0, -128, 127, alphas
    )
    qvar = qvar.reshape(orig_trans_shape)
    qvar = np.moveaxis(qvar, -1, quant_axis).astype(np.int8)
    scale = scale.reshape(orig_trans_shape[:-1]).astype(np.float32)
    return qvar, scale
  else:
    bound = np.max(np.abs(tensor_val), axis=quant_axis, keepdims=True)
    scale = bound / 127.0
    scale = np.where(scale == 0.0, 1.0, scale)

    qvar = np.round(tensor_val / scale)
    qvar = np.clip(qvar, -128, 127).astype(np.int8)
    scale = np.squeeze(scale, axis=quant_axis).astype(np.float32)
    return qvar, scale


def quantize_tensor_int4_sym(
    tensor_val: np.ndarray,
    quant_axis: int,
    block_size: int = 0,
    optimize_scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
  """Performs symmetric 4-bit channelwise or blockwise quantization along quant_axis."""
  if block_size > 0:
    shape = list(tensor_val.shape)
    if shape[quant_axis] % block_size != 0:
      raise ValueError(
          f"Dimension {shape[quant_axis]} along quant_axis {quant_axis} "
          f"is not divisible by block_size {block_size}."
      )
    sub_channels = shape[quant_axis] // block_size
    shape[quant_axis : quant_axis + 1] = [sub_channels, block_size]
    tensor_reshaped = tensor_val.reshape(shape)
    block_axis = quant_axis + 1

    if optimize_scale:
      transposed = np.moveaxis(tensor_reshaped, block_axis, -1)
      trans_shape = transposed.shape
      blocks = transposed.reshape(-1, block_size)
      bound = np.max(np.abs(blocks), axis=-1, keepdims=True)
      alphas = np.linspace(1.0, 0.5, num=7, dtype=np.float32)
      qvar_blocks, scale_blocks = _optimize_scale_search(
          blocks, bound, 7.0, -8, 7, alphas
      )
      qvar_trans = qvar_blocks.reshape(trans_shape)
      qvar_reshaped = np.moveaxis(qvar_trans, -1, block_axis)
      qvar = qvar_reshaped.reshape(tensor_val.shape).astype(np.int8)
      scale = scale_blocks.reshape(trans_shape[:-1]).astype(np.float32)
      return qvar, scale
    else:
      bound = np.max(np.abs(tensor_reshaped), axis=block_axis, keepdims=True)
      scale = bound / 7.0
      scale = np.where(scale == 0.0, 1.0, scale)
      qvar = np.round(tensor_reshaped / scale)
      qvar = np.clip(qvar, -8, 7).astype(np.int8)
      qvar = qvar.reshape(tensor_val.shape)
      scale = np.squeeze(scale, axis=block_axis).astype(np.float32)
      return qvar, scale
  else:
    if optimize_scale:
      transposed = np.moveaxis(tensor_val, quant_axis, -1)
      orig_trans_shape = transposed.shape
      channels = transposed.reshape(-1, orig_trans_shape[-1])
      bound = np.max(np.abs(channels), axis=-1, keepdims=True)
      alphas = np.linspace(1.0, 0.5, num=7, dtype=np.float32)
      qvar, scale = _optimize_scale_search(channels, bound, 7.0, -8, 7, alphas)
      qvar = qvar.reshape(orig_trans_shape)
      qvar = np.moveaxis(qvar, -1, quant_axis).astype(np.int8)
      scale = scale.reshape(orig_trans_shape[:-1]).astype(np.float32)
      return qvar, scale
    else:
      bound = np.max(np.abs(tensor_val), axis=quant_axis, keepdims=True)
      scale = bound / 7.0
      scale = np.where(scale == 0.0, 1.0, scale)

      qvar = np.round(tensor_val / scale)
      qvar = np.clip(qvar, -8, 7).astype(np.int8)
      scale = np.squeeze(scale, axis=quant_axis).astype(np.float32)
      return qvar, scale


def pack_int4(qvar: np.ndarray) -> np.ndarray:
  """Packs an array of int4 values (stored in int8) into pairs of 4-bit nibbles."""
  flat = qvar.ravel().astype(np.uint8)
  if flat.size % 2 != 0:
    raise ValueError(
        f"Cannot pack array with odd number of elements: {flat.size}"
    )
  low = flat[0::2] & 0x0F
  high = (flat[1::2] & 0x0F) << 4
  return low | high


def extract_model_weights(
    model_path: str,
    output_dir: str,
    embedding_quant_bits: Optional[int] = None,
    attention_quant_bits: Optional[int] = None,
    feedforward_quant_bits: Optional[int] = None,
    block_size: int = 0,
    optimize_scale: bool = True,
):
  """Extracts individual layer weights from a Hugging Face model.

  (.safetensors or .pt) or a directory of .safetensors files into a
  specified output directory, with optional int8 or int4 channelwise/blockwise
  quantization.

  Args:
      model_path (str): Path to the input model file (e.g., model.safetensors,
        model.pt) or directory containing .safetensors files.
      output_dir (str): Path to the directory where individual weight files will
        be stored.
      embedding_quant_bits (int, optional): Quantization bits for embedding
        layers.
      attention_quant_bits (int, optional): Quantization bits for attention
        layers.
      feedforward_quant_bits (int, optional): Quantization bits for feedforward
        layers.
      block_size (int, optional): Block size for 4-bit quantization (0, 32, 64,
        128). 0 means channelwise quantization.
      optimize_scale (bool, optional): Whether to search for optimal scale
        factors minimizing mean squared error (MSE). Recommended when using
        channelwise quantization.

  Raises:
      FileNotFoundError: If the provided `model_path` does not exist or if a
        directory contains no .safetensors files.
  """
  if os.path.isfile(model_path):
    model_files = [model_path]
  elif os.path.isdir(model_path):
    model_files = sorted(
        [
            os.path.join(model_path, f)
            for f in os.listdir(model_path)
            if f.endswith(".safetensors")
        ]
    )
    if not model_files:
      raise FileNotFoundError(
          f"No .safetensors files found in directory: {model_path}"
      )
  else:
    raise FileNotFoundError(f"Model path not found: {model_path}")

  os.makedirs(output_dir, exist_ok=True)
  print(f"Output directory '{output_dir}' ensured.")

  quantize_enabled = (
      embedding_quant_bits is not None
      or attention_quant_bits is not None
      or feedforward_quant_bits is not None
  )

  layer_info = []

  for cur_model_path in model_files:
    print(f"\nLoading model from: {cur_model_path}")
    state_dict = {}
    try:
      if cur_model_path.endswith(".safetensors"):
        state_dict = safetensors_load_file(cur_model_path, device="cpu")
      elif cur_model_path.endswith(".pt") or cur_model_path.endswith(".bin"):
        state_dict = torch.load(cur_model_path, map_location="cpu")
      else:
        raise ValueError(
            "Unsupported model file extension. Expected .safetensors or .pt."
        )
    except (ValueError, OSError, RuntimeError) as e:
      print(f"Error loading model from '{cur_model_path}': {e}")
      print("Please ensure it's a valid safetensors or PyTorch model file.")
      return

    if not isinstance(state_dict, dict):
      print(
          "Error: Loaded file does not contain a dictionary of tensors. "
          f"Found type: {type(state_dict)}"
      )
      return

    print(f"Found {len(state_dict)} tensors to process.")

    for i, (key, tensor) in enumerate(state_dict.items()):
      tensor_np = tensor.float().numpy()
      layer_type = get_layer_type(key)

      should_quantize = False
      quant_bits = None
      if (
          quantize_enabled
          and layer_type not in ("LAYER_NORM", "LORA", "SKIP", "NONE")
          and key.endswith(".weight")
      ):
        if layer_type == "FEEDFORWARD":
          quant_bits = feedforward_quant_bits
        elif layer_type == "ATTENTION":
          quant_bits = attention_quant_bits
          if "q_norm" in key or "k_norm" in key:
            quant_bits = None
        elif layer_type == "EMBEDDING":
          quant_bits = embedding_quant_bits

        if quant_bits in (4, 8):
          should_quantize = True

      tensor_val = tensor_np
      if should_quantize:
        quant_axis = 1 if tensor_val.ndim == 2 else 0
        current_block_size = block_size
        if block_size > 0 and tensor_val.shape[quant_axis] % block_size != 0:
          print(
              f"Warning: tensor '{key}' shape {tensor_val.shape} along"
              f" quant_axis {quant_axis} is not divisible by block_size"
              f" {block_size}. Falling back to channelwise (block_size=0) for"
              " this tensor."
          )
          current_block_size = 0

        if quant_bits == 8:
          qvar, scale = quantize_tensor_int8_sym(
              tensor_val, quant_axis, optimize_scale=optimize_scale
          )
          dtype_str = "int8"
          save_data = qvar
        elif quant_bits == 4:
          qvar, scale = quantize_tensor_int4_sym(
              tensor_val,
              quant_axis,
              block_size=current_block_size,
              optimize_scale=optimize_scale,
          )
          dtype_str = "int4"
          save_data = pack_int4(qvar)
        else:
          raise ValueError(f"Unsupported quant_bits: {quant_bits}")

        # Save quantized weight
        out_filename = get_output_filename(key)
        out_filepath = os.path.join(output_dir, out_filename)
        try:
          with open(out_filepath, "wb") as f:
            save_data.tofile(f)
          layer_info.append(
              f"mdl_vars.{key}.{dtype_str}.{'_'.join(map(str, qvar.shape))}"
          )
        except OSError as e:
          print(
              f"Error saving quantized tensor '{key}' to '{out_filepath}': {e}"
          )
          continue

        # Save quantized scale
        scale_key = key + "_quantized_scale"
        scale_filename = get_output_filename(scale_key)
        scale_filepath = os.path.join(output_dir, scale_filename)
        try:
          with open(scale_filepath, "wb") as f:
            scale.tofile(f)
          layer_info.append(
              f"mdl_vars.{scale_key}.{scale.dtype}.{'_'.join(map(str, scale.shape))}"
          )
        except OSError as e:
          print(f"Error saving scale for '{key}' to '{scale_filepath}': {e}")
          continue
      else:
        out_filename = get_output_filename(key)
        out_filepath = os.path.join(output_dir, out_filename)
        try:
          with open(out_filepath, "wb") as f:
            tensor_val.astype(np.float32).tofile(f)
          layer_info.append(
              f"mdl_vars.{key}.{tensor_val.dtype}.{'_'.join(map(str, tensor_val.shape))}"
          )
        except OSError as e:
          print(f"Error saving tensor '{key}' to '{out_filepath}': {e}")
          continue

      if (i + 1) % 50 == 0 or (i + 1) == len(state_dict):
        print(f"Processed {i + 1}/{len(state_dict)} tensors.")

  # Write layer_info.txt
  layer_info = list(set(layer_info))
  layer_info.sort()
  try:
    with open(os.path.join(output_dir, "layer_info.txt"), "w") as finfo:
      for line in layer_info:
        finfo.write(line + "\n\n")
  except OSError as e:
    print(f"Error writing layer_info.txt: {e}")

  if len(model_files) == 1:
    print(
        f"\nSuccessfully extracted all tensors from '{model_files[0]}' to"
        f" '{output_dir}'."
    )
  else:
    print(
        f"\nSuccessfully extracted all tensors from {len(model_files)} files in"
        f" '{model_path}' to '{output_dir}'."
    )


def main():
  parser = argparse.ArgumentParser(
      description=(
          "Extract individual layer weights from a Hugging Face safetensors"
          " model (or PyTorch .pt file, or directory of .safetensors files)"
          " into separate files, with optional int8 quantization."
      )
  )
  parser.add_argument(
      "--model_path",
      type=str,
      required=True,
      help=(
          "Path to the input model file (e.g., model.safetensors, model.pt) or"
          " a directory containing .safetensors files."
      ),
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      required=True,
      help=(
          "Path to the directory where individual weight files will be stored."
      ),
  )
  parser.add_argument(
      "--quantize",
      action="store_true",
      help=(
          "Enable 8-bit int8 or 4-bit int4 quantization for embedding,"
          " attention, and feedforward layers."
      ),
  )
  parser.add_argument(
      "--embedding_quant_bits",
      type=int,
      default=None,
      help="Target quantization bits for embedding layers (e.g., 8 or 4).",
  )
  parser.add_argument(
      "--attention_quant_bits",
      type=int,
      default=None,
      help="Target quantization bits for attention layers (e.g., 8 or 4).",
  )
  parser.add_argument(
      "--feedforward_quant_bits",
      type=int,
      default=None,
      help="Target quantization bits for feedforward layers (e.g., 8 or 4).",
  )
  parser.add_argument(
      "--block_size",
      type=int,
      default=0,
      help=(
          "The block size for any 4bit quant layers. 0 means no blockwise"
          " quant."
      ),
  )
  parser.add_argument(
      "--optimize_scale",
      action=argparse.BooleanOptionalAction,
      default=False,
      help=(
          "Optimize quantization scale factors by grid-searching bounds to"
          " minimize mean squared error (MSE). Recommended when using"
          " channelwise quantization."
      ),
  )
  args = parser.parse_args()

  embedding_bits = args.embedding_quant_bits
  attention_bits = args.attention_quant_bits
  feedforward_bits = args.feedforward_quant_bits

  # Default to int4 quantization for attention and feedforward layers,
  # and int8 quantization for embedding layers.
  if args.quantize:
    if embedding_bits is None:
      embedding_bits = 8
    if attention_bits is None:
      attention_bits = 4
    if feedforward_bits is None:
      feedforward_bits = 4

  if (
      embedding_bits not in (None, 4, 8)
      or attention_bits not in (None, 4, 8)
      or feedforward_bits not in (None, 4, 8)
  ):
    raise ValueError("Quantization bits must be either None, 4, or 8.")

  if args.block_size not in (0, 32, 64, 128):
    raise ValueError("block_size must be either 0, 32, 64, or 128.")

  extract_model_weights(
      model_path=args.model_path,
      output_dir=args.output_dir,
      embedding_quant_bits=embedding_bits,
      attention_quant_bits=attention_bits,
      feedforward_quant_bits=feedforward_bits,
      block_size=args.block_size,
      optimize_scale=args.optimize_scale,
  )


if __name__ == "__main__":
  main()
