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

import numpy as np
from safetensors.torch import load_file as safetensors_load_file
import torch


def get_output_filename(key: str) -> str:
  """Determines the output filename based on the model's tensor key."""
  # SD example expects all weights to have .bin appended.
  return key + ".bin"


def extract_model_weights(model_path: str, output_dir: str):
  """Extracts individual layer weights from a Hugging Face model.

  (.safetensors, .pt, or .bin) into a specified output directory.

  Args:
      model_path (str): Path to the input model file (e.g., model.safetensors,
        pytorch_model.bin).
      output_dir (str): Path to the directory where individual weight files will
        be stored.

  Raises:
      FileNotFoundError: If the provided `model_path` does not exist.
  """
  if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

  os.makedirs(output_dir, exist_ok=True)
  print(f"Output directory '{output_dir}' ensured.")

  print(f"Loading model from: {model_path}")
  state_dict = {}
  try:
    if model_path.endswith(".safetensors"):
      # Use safetensors_load_file for .safetensors files
      state_dict = safetensors_load_file(model_path, device="cpu")
    elif model_path.endswith(".pt") or model_path.endswith(".bin"):
      # Use torch.load for standard PyTorch checkpoint files
      state_dict = torch.load(model_path, map_location="cpu")
    else:
      raise ValueError(
          "Unsupported model file extension. Expected .safetensors, .pt, .bin."
      )

  except (ValueError, OSError, RuntimeError) as e:
    print(f"Error loading model from '{model_path}': {e}")
    print("Please ensure it's a valid safetensors or PyTorch model file.")
    return

  if not isinstance(state_dict, dict):
    print(
        "Error: Loaded file does not contain a dictionary of tensors. "
        f"Found type: {type(state_dict)}"
    )
    return

  print(f"Found {len(state_dict)} tensors to extract.")

  for i, (key, tensor) in enumerate(state_dict.items()):
    output_filename = get_output_filename(key)
    output_filepath = os.path.join(output_dir, output_filename)

    # convert to float numpy array
    tensor_np = tensor.float().numpy()
    # SD example requires all weights to be in float16.
    tensor_np = tensor_np.astype(np.float16)

    # Save the tensor to the output file
    try:
      with open(output_filepath, "wb") as f:
        tensor_np.tofile(f)

      # Print progress every 100 tensors or at the end
      if (i + 1) % 100 == 0 or (i + 1) == len(state_dict):
        print(
            f"Extracted {i + 1}/{len(state_dict)} tensors. "
            f"Last saved: {output_filepath}"
        )
    except OSError as e:
      print(f"Error saving tensor '{key}' to '{output_filepath}': {e}")
      # Continue processing other tensors even if one fails
      continue

  print(
      f"\nSuccessfully extracted all tensors from '{model_path}' to"
      f" '{output_dir}'."
  )


def main():
  parser = argparse.ArgumentParser(
      description=(
          "Extract individual layer weights from a Hugging Face safetensors "
          "model (or PyTorch .pt/.bin file) into separate files."
      )
  )
  parser.add_argument(
      "--model_path",
      type=str,
      required=True,
      help=(
          "Path to the input model file (e.g., model.safetensors,"
          " pytorch_model.bin, model.pt)."
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

  args = parser.parse_args()

  extract_model_weights(args.model_path, args.output_dir)


if __name__ == "__main__":
  main()
