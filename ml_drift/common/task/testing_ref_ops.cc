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

#include "ml_drift/common/task/testing_ref_ops.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <utility>
#include <variant>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/testing_util.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/util.h"
#include "ml_drift/common/winograd_util.h"

namespace ml_drift {

TensorInt32 TestingRuntimeChannels::GenerateTensorInt32() const {
  TensorInt32 params;
  params.shape = BHWC(1, 1, 1, kNumParams);
  params.data = std::vector<int>(params.shape.DimensionsProduct(), -1);
  if (src_end_ch.has_value()) {
    params.data[kSrcEndChIndex] = *src_end_ch;
  }
  if (dst_end_ch.has_value()) {
    params.data[kDstEndChIndex] = *dst_end_ch;
  }
  return params;
}

ConvRuntimeCheckDesc TestingRuntimeChannels::GenerateConvRuntimeCheckDesc()
    const {
  ConvRuntimeCheckDesc runtime_check;
  if (src_end_ch.has_value()) {
    runtime_check.src_end_ch_index = kSrcEndChIndex;
  }
  if (dst_end_ch.has_value()) {
    runtime_check.dst_end_ch_index = kDstEndChIndex;
  }
  return runtime_check;
}

TestingRuntimeChannels TestingRuntimeChannels::GenerateAlignedRuntimeChannels()
    const {
  TestingRuntimeChannels aligned_runtime_channels = *this;
  const int alignment = ConvRuntimeCheckDesc::kChannelsAlignment;
  if (src_end_ch.has_value()) {
    aligned_runtime_channels.src_end_ch = AlignByN(*src_end_ch, alignment);
  }
  if (dst_end_ch.has_value()) {
    aligned_runtime_channels.dst_end_ch = AlignByN(*dst_end_ch, alignment);
  }
  return aligned_runtime_channels;
}

void AddTableReference(const std::vector<TensorFloat32>& srcs,
                       TensorFloat32* dst) {
  for (int b = 0; b < dst->shape.b; ++b) {
    for (int c = 0; c < dst->shape.c; ++c) {
      for (int h = 0; h < dst->shape.h; ++h) {
        for (int w = 0; w < dst->shape.w; ++w) {
          const int index = dst->shape.LinearIndex({b, h, w, c});
          float value = 0.0f;
          for (size_t s = 0; s < srcs.size(); ++s) {
            value += srcs[s].data[index];
          }
          dst->data[index] = value;
        }
      }
    }
  }
}

TensorFloat32 AddTableReference(const std::vector<TensorFloat32>& inputs) {
  TensorFloat32 output = MakeZeroTensor(inputs[0].shape);
  AddTableReference(inputs, &output);
  return output;
}

void AddTableReference(const std::vector<Tensor5DFloat32>& srcs,
                       Tensor5DFloat32* dst) {
  for (int b = 0; b < dst->shape.b; ++b) {
    for (int c = 0; c < dst->shape.c; ++c) {
      for (int h = 0; h < dst->shape.h; ++h) {
        for (int w = 0; w < dst->shape.w; ++w) {
          for (int d = 0; d < dst->shape.d; ++d) {
            const int index = dst->shape.LinearIndex({b, h, w, d, c});
            float value = 0.0f;
            for (size_t s = 0; s < srcs.size(); ++s) {
              value += srcs[s].data[index];
            }
            dst->data[index] = value;
          }
        }
      }
    }
  }
}

Tensor5DFloat32 AddTableReference(const std::vector<Tensor5DFloat32>& inputs) {
  Tensor5DFloat32 output = MakeZeroTensor(inputs[0].shape);
  AddTableReference(inputs, &output);
  return output;
}

int PaddingReflect(int x, int size) {
  return size - 1 - abs(abs(x) - size + 1);
}

void PaddingReference(const PadAttributes& attr, const TensorFloat32& src,
                      TensorFloat32* dst) {
  for (int b = 0; b < dst->shape.b; ++b) {
    int src_b = b - attr.prepended.b;
    for (int y = 0; y < dst->shape.h; ++y) {
      int src_y = y - attr.prepended.h;
      for (int x = 0; x < dst->shape.w; ++x) {
        int src_x = x - attr.prepended.w;
        for (int z = 0; z < dst->shape.c; ++z) {
          int src_z = z - attr.prepended.c;
          float value;
          if (attr.type == PaddingContentType::ZEROS) {
            value = 0.0f;
            if (src_x >= 0 && src_x < src.shape.w && src_y >= 0 &&
                src_y < src.shape.h && src_z >= 0 && src_z < src.shape.c &&
                src_b >= 0 && src_b < src.shape.b) {
              const int src_index =
                  src.shape.LinearIndex({src_b, src_y, src_x, src_z});
              value = src.data[src_index];
            }
          } else if (attr.type == PaddingContentType::REFLECT) {
            src_x = PaddingReflect(src_x, src.shape.w);
            src_y = PaddingReflect(src_y, src.shape.h);
            src_z = PaddingReflect(src_z, src.shape.c);
            const int src_index =
                src.shape.LinearIndex({src_b, src_y, src_x, src_z});
            value = src.data[src_index];
          } else {
            value = -1.0f;
          }
          const int dst_index = dst->shape.LinearIndex({b, y, x, z});
          dst->data[dst_index] = value;
        }
      }
    }
  }
}

void PaddingReference(const Pad3DAttributes& attr, const Tensor5DFloat32& src,
                      Tensor5DFloat32* dst) {
  for (int b = 0; b < dst->shape.b; ++b) {
    int src_b = b - attr.prepended.b;
    for (int y = 0; y < dst->shape.h; ++y) {
      int src_y = y - attr.prepended.h;
      for (int x = 0; x < dst->shape.w; ++x) {
        int src_x = x - attr.prepended.w;
        for (int z = 0; z < dst->shape.d; ++z) {
          int src_z = z - attr.prepended.d;
          for (int c = 0; c < dst->shape.c; ++c) {
            int src_c = c - attr.prepended.c;
            float value;
            if (attr.type == PaddingContentType::ZEROS) {
              value = 0.0f;
              if (src_x >= 0 && src_x < src.shape.w && src_y >= 0 &&
                  src_y < src.shape.h && src_z >= 0 && src_z < src.shape.d &&
                  src_c >= 0 && src_c < src.shape.c && src_b >= 0 &&
                  src_b < src.shape.b) {
                const int src_index =
                    src.shape.LinearIndex({src_b, src_y, src_x, src_z, src_c});
                value = src.data[src_index];
              }
            } else if (attr.type == PaddingContentType::REFLECT) {
              src_x = PaddingReflect(src_x, src.shape.w);
              src_y = PaddingReflect(src_y, src.shape.h);
              src_z = PaddingReflect(src_z, src.shape.d);
              src_c = PaddingReflect(src_c, src.shape.c);
              const int src_index =
                  src.shape.LinearIndex({src_b, src_y, src_x, src_z, src_c});
              value = src.data[src_index];
            } else {
              value = -1.0f;
            }
            const int dst_index = dst->shape.LinearIndex({b, y, x, z, c});
            dst->data[dst_index] = value;
          }
        }
      }
    }
  }
}

TensorFloat32 PaddingReference(const PadAttributes& attr,
                               const TensorFloat32& input) {
  TensorFloat32 output =
      MakeZeroTensor(CalculateOutputShape(input.shape, attr));
  PaddingReference(attr, input, &output);
  return output;
}

Tensor5DFloat32 PaddingReference(const Pad3DAttributes& attr,
                                 const Tensor5DFloat32& input) {
  Tensor5DFloat32 output =
      MakeZeroTensor(CalculateOutputShape(input.shape, attr));
  PaddingReference(attr, input, &output);
  return output;
}

TensorFloat32 BatchedMatMulReference(const TensorFloat32& left,
                                     const TensorFloat32& right) {
  return BatchedMatMulReference(left, right, TestingRuntimeChannels());
}

TensorFloat32 BatchedMatMulReference(
    const TensorFloat32& left, const TensorFloat32& right,
    const TestingRuntimeChannels& runtime_channels) {
  TensorFloat32 output = MakeZeroTensor(
      BHWC(left.shape.b, left.shape.h, left.shape.w, right.shape.c));
  const int src_end_ch = runtime_channels.src_end_ch.has_value()
                             ? *runtime_channels.src_end_ch
                             : left.shape.c;
  const int dst_end_ch = runtime_channels.dst_end_ch.has_value()
                             ? *runtime_channels.dst_end_ch
                             : output.shape.c;
  for (int b = 0; b < output.shape.h; ++b) {
    for (int dst_x = 0; dst_x < output.shape.w; ++dst_x) {
      for (int dst_ch = 0; dst_ch < dst_end_ch; dst_ch++) {
        float sum = 0.0f;
        for (int src_ch = 0; src_ch < src_end_ch; src_ch++) {
          const int left_index = left.shape.LinearIndex({0, b, dst_x, src_ch});
          const int right_index =
              right.shape.LinearIndex({0, b, src_ch, dst_ch});
          sum += left.data[left_index] * right.data[right_index];
        }
        const int dst_index = output.shape.LinearIndex({0, b, dst_x, dst_ch});
        output.data[dst_index] = sum;
      }
    }
  }
  return output;
}

void ConcatZReference(const std::vector<TensorFloat32>& srcs,
                      TensorFloat32* dst) {
  for (int b = 0; b < dst->shape.b; ++b) {
    for (int h = 0; h < dst->shape.h; ++h) {
      for (int w = 0; w < dst->shape.w; ++w) {
        int dst_ch = 0;
        for (size_t i = 0; i < srcs.size(); ++i) {
          for (int c = 0; c < srcs[i].shape.c; ++c) {
            const int src_index = srcs[i].shape.LinearIndex({b, h, w, c});
            const int dst_index = dst->shape.LinearIndex({b, h, w, dst_ch});
            dst->data[dst_index] = srcs[i].data[src_index];
            dst_ch++;
          }
        }
      }
    }
  }
}

void ConcatZReference(const std::vector<Tensor5DFloat32>& srcs,
                      Tensor5DFloat32* dst) {
  for (int b = 0; b < dst->shape.b; ++b) {
    for (int h = 0; h < dst->shape.h; ++h) {
      for (int w = 0; w < dst->shape.w; ++w) {
        for (int d = 0; d < dst->shape.d; ++d) {
          int dst_ch = 0;
          for (size_t i = 0; i < srcs.size(); ++i) {
            for (int c = 0; c < srcs[i].shape.c; ++c) {
              const int src_index = srcs[i].shape.LinearIndex({b, h, w, d, c});
              const int dst_index =
                  dst->shape.LinearIndex({b, h, w, d, dst_ch});
              dst->data[dst_index] = srcs[i].data[src_index];
              dst_ch++;
            }
          }
        }
      }
    }
  }
}

void ConcatBatchReference(const std::vector<TensorFloat32>& srcs,
                          TensorFloat32* dst) {
  for (int w = 0; w < dst->shape.w; ++w) {
    for (int h = 0; h < dst->shape.h; ++h) {
      for (int c = 0; c < dst->shape.c; ++c) {
        int dst_b = 0;
        for (size_t i = 0; i < srcs.size(); ++i) {
          for (int b = 0; b < srcs[i].shape.b; ++b) {
            const int src_index = srcs[i].shape.LinearIndex({b, h, w, c});
            const int dst_index = dst->shape.LinearIndex({dst_b, h, w, c});
            dst->data[dst_index] = srcs[i].data[src_index];
            dst_b++;
          }
        }
      }
    }
  }
}

void ConcatBatchReference(const std::vector<Tensor5DFloat32>& srcs,
                          Tensor5DFloat32* dst) {
  for (int w = 0; w < dst->shape.w; ++w) {
    for (int h = 0; h < dst->shape.h; ++h) {
      for (int d = 0; d < dst->shape.d; ++d) {
        for (int c = 0; c < dst->shape.c; ++c) {
          int dst_b = 0;
          for (size_t i = 0; i < srcs.size(); ++i) {
            for (int b = 0; b < srcs[i].shape.b; ++b) {
              const int src_index = srcs[i].shape.LinearIndex({b, h, w, d, c});
              const int dst_index = dst->shape.LinearIndex({dst_b, h, w, d, c});
              dst->data[dst_index] = srcs[i].data[src_index];
              dst_b++;
            }
          }
        }
      }
    }
  }
}

void ConcatWidthReference(const std::vector<TensorFloat32>& srcs,
                          TensorFloat32* dst) {
  for (int b = 0; b < dst->shape.b; ++b) {
    for (int h = 0; h < dst->shape.h; ++h) {
      for (int c = 0; c < dst->shape.c; ++c) {
        int dst_w = 0;
        for (size_t i = 0; i < srcs.size(); ++i) {
          for (int w = 0; w < srcs[i].shape.w; ++w) {
            const int src_index = srcs[i].shape.LinearIndex({b, h, w, c});
            const int dst_index = dst->shape.LinearIndex({b, h, dst_w, c});
            dst->data[dst_index] = srcs[i].data[src_index];
            dst_w++;
          }
        }
      }
    }
  }
}

void ConcatWidthReference(const std::vector<Tensor5DFloat32>& srcs,
                          Tensor5DFloat32* dst) {
  for (int b = 0; b < dst->shape.b; ++b) {
    for (int h = 0; h < dst->shape.h; ++h) {
      for (int d = 0; d < dst->shape.d; ++d) {
        for (int c = 0; c < dst->shape.c; ++c) {
          int dst_w = 0;
          for (size_t i = 0; i < srcs.size(); ++i) {
            for (int w = 0; w < srcs[i].shape.w; ++w) {
              const int src_index = srcs[i].shape.LinearIndex({b, h, w, d, c});
              const int dst_index = dst->shape.LinearIndex({b, h, dst_w, d, c});
              dst->data[dst_index] = srcs[i].data[src_index];
              dst_w++;
            }
          }
        }
      }
    }
  }
}

void ConcatHeightReference(const std::vector<TensorFloat32>& srcs,
                           TensorFloat32* dst) {
  for (int b = 0; b < dst->shape.b; ++b) {
    for (int w = 0; w < dst->shape.w; ++w) {
      for (int c = 0; c < dst->shape.c; ++c) {
        int dst_h = 0;
        for (size_t i = 0; i < srcs.size(); ++i) {
          for (int h = 0; h < srcs[i].shape.h; ++h) {
            const int src_index = srcs[i].shape.LinearIndex({b, h, w, c});
            const int dst_index = dst->shape.LinearIndex({b, dst_h, w, c});
            dst->data[dst_index] = srcs[i].data[src_index];
            dst_h++;
          }
        }
      }
    }
  }
}

void ConcatHeightReference(const std::vector<Tensor5DFloat32>& srcs,
                           Tensor5DFloat32* dst) {
  for (int b = 0; b < dst->shape.b; ++b) {
    for (int w = 0; w < dst->shape.w; ++w) {
      for (int d = 0; d < dst->shape.d; ++d) {
        for (int c = 0; c < dst->shape.c; ++c) {
          int dst_h = 0;
          for (size_t i = 0; i < srcs.size(); ++i) {
            for (int h = 0; h < srcs[i].shape.h; ++h) {
              const int src_index = srcs[i].shape.LinearIndex({b, h, w, d, c});
              const int dst_index = dst->shape.LinearIndex({b, dst_h, w, d, c});
              dst->data[dst_index] = srcs[i].data[src_index];
              dst_h++;
            }
          }
        }
      }
    }
  }
}

void ConcatDepthReference(const std::vector<Tensor5DFloat32>& srcs,
                          Tensor5DFloat32* dst) {
  for (int b = 0; b < dst->shape.b; ++b) {
    for (int w = 0; w < dst->shape.w; ++w) {
      for (int h = 0; h < dst->shape.h; ++h) {
        for (int c = 0; c < dst->shape.c; ++c) {
          int dst_d = 0;
          for (size_t i = 0; i < srcs.size(); ++i) {
            for (int d = 0; d < srcs[i].shape.d; ++d) {
              const int src_index = srcs[i].shape.LinearIndex({b, h, w, d, c});
              const int dst_index = dst->shape.LinearIndex({b, h, w, dst_d, c});
              dst->data[dst_index] = srcs[i].data[src_index];
              dst_d++;
            }
          }
        }
      }
    }
  }
}

TensorFloat32 ConcatReference(const ConcatAttributes& attr,
                              const std::vector<TensorFloat32>& inputs) {
  std::vector<BHWC> shapes;
  shapes.reserve(inputs.size());
  for (auto& input : inputs) {
    shapes.push_back(input.shape);
  }
  BHWC output_shape;
  ABSL_QCHECK_OK(CalculateOutputShape(shapes, attr, &output_shape));
  TensorFloat32 output = MakeZeroTensor(output_shape);
  switch (attr.axis) {
    case Axis::CHANNELS:
      ConcatZReference(inputs, &output);
      break;
    case Axis::WIDTH:
      ConcatWidthReference(inputs, &output);
      break;
    case Axis::HEIGHT:
      ConcatHeightReference(inputs, &output);
      break;
    case Axis::BATCH:
      ConcatBatchReference(inputs, &output);
      break;
    default:
      ABSL_LOG(FATAL) << "Unsupported axis: " << ToString(attr.axis);
  }
  return output;
}

Tensor5DFloat32 ConcatReference(const ConcatAttributes& attr,
                                const std::vector<Tensor5DFloat32>& inputs) {
  std::vector<BHWDC> shapes;
  shapes.reserve(inputs.size());
  for (auto& input : inputs) {
    shapes.push_back(input.shape);
  }
  BHWDC output_shape;
  ABSL_QCHECK_OK(CalculateOutputShape(shapes, attr, &output_shape));
  Tensor5DFloat32 output = MakeZeroTensor(output_shape);
  switch (attr.axis) {
    case Axis::CHANNELS:
      ConcatZReference(inputs, &output);
      break;
    case Axis::WIDTH:
      ConcatWidthReference(inputs, &output);
      break;
    case Axis::HEIGHT:
      ConcatHeightReference(inputs, &output);
      break;
    case Axis::DEPTH:
      ConcatDepthReference(inputs, &output);
      break;
    case Axis::BATCH:
      ConcatBatchReference(inputs, &output);
      break;
    default:
      ABSL_LOG(FATAL) << "Unsupported axis: " << ToString(attr.axis);
  }
  return output;
}

void ConvolutionTransposedReference(const ConvolutionTransposedAttributes& attr,
                                    const TensorFloat32& src,
                                    TensorFloat32* dst) {
  for (int b = 0; b < dst->shape.b; ++b) {
    for (int out_ch = 0; out_ch < attr.weights.shape.o; out_ch++) {
      for (int in_y = 0; in_y < src.shape.h; ++in_y) {
        for (int in_x = 0; in_x < src.shape.w; ++in_x) {
          const int out_x_origin =
              (in_x * attr.stride.w) - attr.padding.prepended.w;
          const int out_y_origin =
              (in_y * attr.stride.h) - attr.padding.prepended.h;
          for (int ch = 0; ch < attr.weights.shape.i; ch++) {
            for (int fy = 0; fy < attr.weights.shape.h; ++fy) {
              for (int fx = 0; fx < attr.weights.shape.w; ++fx) {
                const int out_x = out_x_origin + fx;
                const int out_y = out_y_origin + fy;
                if (out_x < 0 || out_x >= dst->shape.w || out_y < 0 ||
                    out_y >= dst->shape.h)
                  continue;
                const int src_index =
                    src.shape.LinearIndex({b, in_y, in_x, ch});
                const int f_index =
                    attr.weights.shape.LinearIndex({out_ch, fy, fx, ch});
                const int dst_index =
                    dst->shape.LinearIndex({b, out_y, out_x, out_ch});
                dst->data[dst_index] +=
                    src.data[src_index] * attr.weights.data[f_index];
              }
            }
          }
        }
      }
    }
  }

  // Bias addition
  for (int b = 0; b < dst->shape.b; ++b) {
    for (int out_ch = 0; out_ch < attr.weights.shape.o; out_ch++) {
      for (int out_y = 0; out_y < dst->shape.h; ++out_y) {
        for (int out_x = 0; out_x < dst->shape.w; ++out_x) {
          float sum = attr.bias.data.empty() ? 0 : attr.bias.data[out_ch];
          const int dst_index =
              dst->shape.LinearIndex({b, out_y, out_x, out_ch});
          dst->data[dst_index] += sum;
        }
      }
    }
  }
}

void ConvolutionTransposedReference(
    const ConvolutionTransposed3DAttributes& attr, const Tensor5DFloat32& src,
    Tensor5DFloat32* dst) {
  for (int b = 0; b < dst->shape.b; ++b) {
    for (int out_ch = 0; out_ch < attr.weights.shape.o; out_ch++) {
      for (int in_y = 0; in_y < src.shape.h; ++in_y) {
        for (int in_x = 0; in_x < src.shape.w; ++in_x) {
          for (int in_z = 0; in_z < src.shape.d; ++in_z) {
            const int out_x_origin =
                (in_x * attr.stride.w) - attr.padding.prepended.w;
            const int out_y_origin =
                (in_y * attr.stride.h) - attr.padding.prepended.h;
            const int out_z_origin =
                (in_z * attr.stride.d) - attr.padding.prepended.d;
            for (int ch = 0; ch < attr.weights.shape.i; ch++) {
              for (int fy = 0; fy < attr.weights.shape.h; ++fy) {
                for (int fx = 0; fx < attr.weights.shape.w; ++fx) {
                  for (int fz = 0; fz < attr.weights.shape.d; ++fz) {
                    const int out_x = out_x_origin + fx;
                    const int out_y = out_y_origin + fy;
                    const int out_z = out_z_origin + fz;
                    if (out_x < 0 || out_x >= dst->shape.w || out_y < 0 ||
                        out_y >= dst->shape.h || out_z < 0 ||
                        out_z >= dst->shape.d)
                      continue;
                    const int src_index =
                        src.shape.LinearIndex({b, in_y, in_x, in_z, ch});
                    const int f_index = attr.weights.shape.LinearIndex(
                        {out_ch, fy, fx, fz, ch});
                    const int dst_index = dst->shape.LinearIndex(
                        {b, out_y, out_x, out_z, out_ch});
                    dst->data[dst_index] +=
                        src.data[src_index] * attr.weights.data[f_index];
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // Bias addition
  for (int b = 0; b < dst->shape.b; ++b) {
    for (int out_ch = 0; out_ch < attr.weights.shape.o; out_ch++) {
      for (int out_y = 0; out_y < dst->shape.h; ++out_y) {
        for (int out_x = 0; out_x < dst->shape.w; ++out_x) {
          for (int out_z = 0; out_z < dst->shape.d; ++out_z) {
            float sum = attr.bias.data.empty() ? 0 : attr.bias.data[out_ch];
            const int dst_index =
                dst->shape.LinearIndex({b, out_y, out_x, out_z, out_ch});
            dst->data[dst_index] += sum;
          }
        }
      }
    }
  }
}

TensorFloat32 ConvolutionTransposedReference(
    const ConvolutionTransposedAttributes& attr, const TensorFloat32& input) {
  TensorFloat32 output =
      MakeZeroTensor(CalculateOutputShape(input.shape, attr));
  ConvolutionTransposedReference(attr, input, &output);
  return output;
}

Tensor5DFloat32 ConvolutionTransposedReference(
    const ConvolutionTransposed3DAttributes& attr,
    const Tensor5DFloat32& input) {
  Tensor5DFloat32 output =
      MakeZeroTensor(CalculateOutputShape(input.shape, attr));
  ConvolutionTransposedReference(attr, input, &output);
  return output;
}

void ConvolutionReference(const Convolution2DAttributes& attr,
                          const TensorFloat32& src, TensorFloat32* dst,
                          const TestingRuntimeChannels& runtime_channels) {
  const int src_group_channels = src.shape.c / attr.groups;
  const int dst_group_channels = dst->shape.c / attr.groups;

  const int src_end_ch = runtime_channels.src_end_ch.has_value()
                             ? *runtime_channels.src_end_ch
                             : src.shape.c;
  const int dst_end_ch = runtime_channels.dst_end_ch.has_value()
                             ? *runtime_channels.dst_end_ch
                             : dst->shape.c;
  const auto& weights = GetFloatWeights(attr);
  for (int b = 0; b < dst->shape.b; ++b) {
    for (int out_ch = 0; out_ch < dst_end_ch; out_ch++) {
      const int group_id = out_ch / dst_group_channels;
      for (int out_y = 0; out_y < dst->shape.h; ++out_y) {
        for (int out_x = 0; out_x < dst->shape.w; ++out_x) {
          float sum = attr.bias.data.empty() ? 0 : attr.bias.data[out_ch];
          for (int ch = 0; ch < src_group_channels; ch++) {
            for (int fy = 0; fy < weights.shape.h; ++fy) {
              int py = out_y * attr.strides.h + fy * attr.dilations.h -
                       attr.padding.prepended.h;

              if (py < 0 || py >= src.shape.h) {
                continue;
              }

              for (int fx = 0; fx < weights.shape.w; ++fx) {
                int px = out_x * attr.strides.w + fx * attr.dilations.w -
                         attr.padding.prepended.w;

                if (px < 0 || px >= src.shape.w) {
                  continue;
                }
                const int src_ch = group_id * src_group_channels + ch;
                if (src_ch < 0 || src_ch >= src_end_ch) continue;
                const int src_index =
                    src.shape.LinearIndex({b, py, px, src_ch});
                const int f_index =
                    weights.shape.LinearIndex({out_ch, fy, fx, ch});
                sum += src.data[src_index] * weights.data[f_index];
              }
            }
          }
          const int dst_index =
              dst->shape.LinearIndex({b, out_y, out_x, out_ch});
          dst->data[dst_index] = sum;
        }
      }
    }
  }
}

TensorFloat32 ConvolutionReference(const Convolution2DAttributes& attr,
                                   const TensorFloat32& input) {
  return ConvolutionReference(attr, input, TestingRuntimeChannels());
}

TensorFloat32 ConvolutionReference(
    const Convolution2DAttributes& attr, const TensorFloat32& input,
    const TestingRuntimeChannels& runtime_channels) {
  TensorFloat32 output =
      MakeZeroTensor(CalculateOutputShape(input.shape, attr));
  ConvolutionReference(attr, input, &output, runtime_channels);
  return output;
}

void ConvolutionReference(const Convolution2DAttributes& attr,
                          const TensorFloat32& src,
                          const TensorFloat32& weights, TensorFloat32* dst) {
  for (int b = 0; b < dst->shape.b; ++b) {
    for (int out_ch = 0; out_ch < weights.shape.b; out_ch++) {
      for (int out_y = 0; out_y < dst->shape.h; ++out_y) {
        for (int out_x = 0; out_x < dst->shape.w; ++out_x) {
          float sum = attr.bias.data.empty() ? 0 : attr.bias.data[out_ch];
          for (int ch = 0; ch < weights.shape.c; ch++) {
            for (int fy = 0; fy < weights.shape.h; ++fy) {
              int py = out_y * attr.strides.h + fy * attr.dilations.h -
                       attr.padding.prepended.h;

              if (py < 0 || py >= src.shape.h) {
                continue;
              }

              for (int fx = 0; fx < weights.shape.w; ++fx) {
                int px = out_x * attr.strides.w + fx * attr.dilations.w -
                         attr.padding.prepended.w;

                if (px < 0 || px >= src.shape.w) {
                  continue;
                }
                const int src_index = src.shape.LinearIndex({b, py, px, ch});
                const int f_index =
                    weights.shape.LinearIndex({out_ch, fy, fx, ch});
                sum += src.data[src_index] * weights.data[f_index];
              }
            }
          }
          const int dst_index =
              dst->shape.LinearIndex({b, out_y, out_x, out_ch});
          dst->data[dst_index] = sum;
        }
      }
    }
  }
}

TensorFloat32 ConvolutionReference(const Convolution2DAttributes& attr,
                                   const TensorFloat32& input,
                                   const TensorFloat32& weights) {
  Convolution2DAttributes attr_copy = attr;
  auto& weights_shape_copy =
      std::visit([](auto& w) -> auto& { return w.shape; }, attr_copy.weights);
  weights_shape_copy.o = weights.shape.b;
  weights_shape_copy.h = weights.shape.h;
  weights_shape_copy.w = weights.shape.w;
  weights_shape_copy.i = weights.shape.c;
  TensorFloat32 output =
      MakeZeroTensor(CalculateOutputShape(input.shape, attr_copy));
  ConvolutionReference(attr, input, weights, &output);
  return output;
}

void ConvolutionReference(const Convolution3DAttributes& attr,
                          const Tensor5DFloat32& src, Tensor5DFloat32* dst) {
  const int src_group_channels = src.shape.c / attr.groups;
  const int dst_group_channels = dst->shape.c / attr.groups;
  for (int b = 0; b < dst->shape.b; ++b) {
    for (int out_ch = 0; out_ch < dst->shape.c; out_ch++) {
      const int group_id = out_ch / dst_group_channels;
      for (int out_y = 0; out_y < dst->shape.h; ++out_y) {
        for (int out_x = 0; out_x < dst->shape.w; ++out_x) {
          for (int out_z = 0; out_z < dst->shape.d; ++out_z) {
            float sum = attr.bias.data.empty() ? 0 : attr.bias.data[out_ch];
            for (int ch = 0; ch < src_group_channels; ch++) {
              for (int fy = 0; fy < attr.weights.shape.h; ++fy) {
                int py = out_y * attr.strides.h + fy * attr.dilations.h -
                         attr.padding.prepended.h;

                if (py < 0 || py >= src.shape.h) {
                  continue;
                }

                for (int fx = 0; fx < attr.weights.shape.w; ++fx) {
                  int px = out_x * attr.strides.w + fx * attr.dilations.w -
                           attr.padding.prepended.w;

                  if (px < 0 || px >= src.shape.w) {
                    continue;
                  }

                  for (int fz = 0; fz < attr.weights.shape.d; ++fz) {
                    int pz = out_z * attr.strides.d + fz * attr.dilations.d -
                             attr.padding.prepended.d;

                    if (pz < 0 || pz >= src.shape.d) {
                      continue;
                    }

                    const int src_ch = group_id * src_group_channels + ch;
                    const int src_index =
                        src.shape.LinearIndex({b, py, px, pz, src_ch});
                    const int f_index = attr.weights.shape.LinearIndex(
                        {out_ch, fy, fx, fz, ch});
                    sum += src.data[src_index] * attr.weights.data[f_index];
                  }
                }
              }
            }
            const int dst_index =
                dst->shape.LinearIndex({b, out_y, out_x, out_z, out_ch});
            dst->data[dst_index] = sum;
          }
        }
      }
    }
  }
}

Tensor5DFloat32 ConvolutionReference(const Convolution3DAttributes& attr,
                                     const Tensor5DFloat32& input) {
  Tensor5DFloat32 output =
      MakeZeroTensor(CalculateOutputShape(input.shape, attr));
  ConvolutionReference(attr, input, &output);
  return output;
}

void DepthWiseConvolutionReference(const DepthwiseConvolution2DAttributes& attr,
                                   const TensorFloat32& src,
                                   TensorFloat32* dst) {
  const auto& weights = GetFloatWeights(attr);
  const int groups = weights.shape.o;

  for (int b = 0; b < dst->shape.b; ++b) {
    for (int out_y = 0; out_y < dst->shape.h; ++out_y) {
      for (int out_x = 0; out_x < dst->shape.w; ++out_x) {
        for (int src_channel = 0; src_channel < weights.shape.i;
             ++src_channel) {
          for (int sub_layer = 0; sub_layer < groups; ++sub_layer) {
            const int dst_channel = src_channel * groups + sub_layer;
            float sum =
                attr.bias.data.empty() ? 0 : attr.bias.data[dst_channel];

            for (int fy = 0; fy < weights.shape.h; ++fy) {
              int py = out_y * attr.strides.h + fy * attr.dilations.h -
                       attr.padding.prepended.h;

              if (py < 0 || py >= src.shape.h) {
                continue;
              }

              for (int fx = 0; fx < weights.shape.w; ++fx) {
                int px = out_x * attr.strides.w + fx * attr.dilations.w -
                         attr.padding.prepended.w;

                if (px < 0 || px >= src.shape.w) {
                  continue;
                }

                const int src_index =
                    src.shape.LinearIndex({b, py, px, src_channel});
                const int f_index =
                    weights.shape.LinearIndex({sub_layer, fy, fx, src_channel});
                sum += src.data[src_index] * weights.data[f_index];
              }
            }
            const int dst_index =
                dst->shape.LinearIndex({b, out_y, out_x, dst_channel});
            dst->data[dst_index] = sum;
          }
        }
      }
    }
  }
}

void DepthWiseConvolutionReference(const DepthwiseConvolution3DAttributes& attr,
                                   const Tensor5DFloat32& src,
                                   Tensor5DFloat32* dst) {
  const int groups = attr.weights.shape.o;

  for (int b = 0; b < dst->shape.b; ++b) {
    for (int out_y = 0; out_y < dst->shape.h; ++out_y) {
      for (int out_x = 0; out_x < dst->shape.w; ++out_x) {
        for (int out_z = 0; out_z < dst->shape.d; ++out_z) {
          for (int src_channel = 0; src_channel < attr.weights.shape.i;
               ++src_channel) {
            for (int sub_layer = 0; sub_layer < groups; ++sub_layer) {
              const int dst_channel = src_channel * groups + sub_layer;
              float sum =
                  attr.bias.data.empty() ? 0 : attr.bias.data[dst_channel];

              for (int fy = 0; fy < attr.weights.shape.h; ++fy) {
                int py = out_y * attr.strides.h + fy * attr.dilations.h -
                         attr.padding.prepended.h;

                if (py < 0 || py >= src.shape.h) {
                  continue;
                }

                for (int fx = 0; fx < attr.weights.shape.w; ++fx) {
                  int px = out_x * attr.strides.w + fx * attr.dilations.w -
                           attr.padding.prepended.w;

                  if (px < 0 || px >= src.shape.w) {
                    continue;
                  }

                  for (int fz = 0; fz < attr.weights.shape.d; ++fz) {
                    int pz = out_z * attr.strides.d + fz * attr.dilations.d -
                             attr.padding.prepended.d;

                    if (pz < 0 || pz >= src.shape.d) {
                      continue;
                    }

                    const int src_index =
                        src.shape.LinearIndex({b, py, px, pz, src_channel});
                    const int f_index = attr.weights.shape.LinearIndex(
                        {sub_layer, fy, fx, fz, src_channel});
                    sum += src.data[src_index] * attr.weights.data[f_index];
                  }
                }
              }
              const int dst_index =
                  dst->shape.LinearIndex({b, out_y, out_x, out_z, dst_channel});
              dst->data[dst_index] = sum;
            }
          }
        }
      }
    }
  }
}

TensorFloat32 DepthWiseConvolutionReference(
    const DepthwiseConvolution2DAttributes& attr, const TensorFloat32& input) {
  TensorFloat32 output =
      MakeZeroTensor(CalculateOutputShape(input.shape, attr));
  DepthWiseConvolutionReference(attr, input, &output);
  return output;
}

Tensor5DFloat32 DepthWiseConvolutionReference(
    const DepthwiseConvolution3DAttributes& attr,
    const Tensor5DFloat32& input) {
  Tensor5DFloat32 output =
      MakeZeroTensor(CalculateOutputShape(input.shape, attr));
  DepthWiseConvolutionReference(attr, input, &output);
  return output;
}

void FullyConnectedReference(const FullyConnectedAttributes& attr,
                             const TensorFloat32& src, TensorFloat32* dst,
                             const TestingRuntimeChannels& runtime_channels) {
  const int src_end_ch = runtime_channels.src_end_ch.has_value()
                       ? *runtime_channels.src_end_ch
                       : attr.weights.shape.i;
  const int dst_end_ch = runtime_channels.dst_end_ch.has_value()
                       ? *runtime_channels.dst_end_ch
                       : attr.weights.shape.o;
  for (int b = 0; b < src.shape.b; ++b) {
    for (int y = 0; y < src.shape.h; ++y) {
      for (int x = 0; x < src.shape.w; ++x) {
        for (int d = 0; d < dst_end_ch; d++) {
          float sum = attr.bias.data.empty() ? 0 : attr.bias.data[d];
          for (int s = 0; s < src_end_ch; s++) {
            const int src_index = src.shape.LinearIndex({b, y, x, s});
            const int w_h = attr.weights.shape.h == 1 ? 0 : y;
            const int f_index = attr.weights.shape.LinearIndex({d, w_h, 0, s});
            sum += src.data[src_index] * attr.weights.data[f_index];
          }
          const int dst_index = dst->shape.LinearIndex({b, y, x, d});
          dst->data[dst_index] = sum;
        }
      }
    }
  }
}

TensorFloat32 FullyConnectedReference(const FullyConnectedAttributes& attr,
                                      const TensorFloat32& input) {
  return FullyConnectedReference(attr, input, TestingRuntimeChannels());
}

TensorFloat32 FullyConnectedReference(
    const FullyConnectedAttributes& attr, const TensorFloat32& input,
    const TestingRuntimeChannels& runtime_channels) {
  TensorFloat32 output =
      MakeZeroTensor(CalculateOutputShape(input.shape, attr));
  FullyConnectedReference(attr, input, &output, runtime_channels);
  return output;
}

TensorFloat32 FullyConnectedRefDifferentWeightsForHeight(
    Tensor<OHWI, DataType::FLOAT32> weights, const TensorFloat32& src) {
  return FullyConnectedRefDifferentWeightsForHeight(
      weights, src, TestingRuntimeChannels());
}

TensorFloat32 FullyConnectedRefDifferentWeightsForHeight(
    Tensor<OHWI, DataType::FLOAT32> weights, const TensorFloat32& src,
    const TestingRuntimeChannels& runtime_channels) {
  BHWC dst_shape(src.shape.b, src.shape.h, src.shape.w, weights.shape.o);
  TensorFloat32 dst = MakeZeroTensor(dst_shape);
  const int src_end_ch = runtime_channels.src_end_ch.has_value()
                       ? *runtime_channels.src_end_ch
                       : weights.shape.i;
  const int dst_end_ch = runtime_channels.dst_end_ch.has_value()
                       ? *runtime_channels.dst_end_ch
                       : weights.shape.o;
  for (int b = 0; b < src.shape.b; ++b) {
    for (int y = 0; y < src.shape.h; ++y) {
      for (int x = 0; x < src.shape.w; ++x) {
        for (int d = 0; d < dst_end_ch; d++) {
          float sum = 0.0f;
          for (int s = 0; s < src_end_ch; s++) {
            const int src_index = src.shape.LinearIndex({b, y, x, s});
            const int w_h = weights.shape.h == 1 ? 0 : y;
            const int f_index = weights.shape.LinearIndex({d, w_h, 0, s});
            sum += src.data[src_index] * weights.data[f_index];
          }
          const int dst_index = dst.shape.LinearIndex({b, y, x, d});
          dst.data[dst_index] = sum;
        }
      }
    }
  }
  return dst;
}

TensorFloat32 FullyConnectedWeightsBatchIdsReference(
    Tensor<OHWI, DataType::FLOAT32> weights, const TensorFloat32& src,
    const TensorInt32& ids) {
  BHWC dst_shape(src.shape.b, ids.shape.c, src.shape.w, weights.shape.o);
  TensorFloat32 dst = MakeZeroTensor(dst_shape);
  for (int b = 0; b < dst.shape.b; ++b) {
    for (int y = 0; y < dst.shape.h; ++y) {
      const int src_batch_index = src.shape.h == 1 ? 0 : y;
      for (int x = 0; x < dst.shape.w; ++x) {
        for (int d = 0; d < weights.shape.o; d++) {
          float sum = 0.0f;
          for (int s = 0; s < weights.shape.i; s++) {
            const int src_index =
                src.shape.LinearIndex({b, src_batch_index, x, s});
            const int f_index =
                weights.shape.LinearIndex({d, ids.data[y], 0, s});
            sum += src.data[src_index] * weights.data[f_index];
          }
          const int dst_index = dst.shape.LinearIndex({b, y, x, d});
          dst.data[dst_index] = sum;
        }
      }
    }
  }
  return dst;
}

TensorInt32 FullyConnectedReference(
    const ml_drift::Tensor<BHWC, DataType::INT8>& src_tensor_i8,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i8) {
  TensorInt32 dst_ref_tensor;
  dst_ref_tensor.shape = src_tensor_i8.shape;
  dst_ref_tensor.shape.c = weights_i8.shape.o;
  dst_ref_tensor.data.resize(dst_ref_tensor.shape.DimensionsProduct(), 0.0f);
  for (int b = 0; b < src_tensor_i8.shape.b; ++b) {
    for (int y = 0; y < src_tensor_i8.shape.h; ++y) {
      for (int x = 0; x < src_tensor_i8.shape.w; ++x) {
        for (int dst_ch = 0; dst_ch < weights_i8.shape.o; ++dst_ch) {
          int dst_val = 0;
          for (int src_ch = 0; src_ch < weights_i8.shape.i; ++src_ch) {
            int src_i32 =
                src_tensor_i8
                    .data[src_tensor_i8.shape.LinearIndex({b, y, x, src_ch})];
            const int weights_h = weights_i8.shape.h == 1 ? 0 : y;
            int weights_i32 = weights_i8.data[weights_i8.shape.LinearIndex(
                {dst_ch, weights_h, 0, src_ch})];
            dst_val += src_i32 * weights_i32;
          }
          dst_ref_tensor
              .data[dst_ref_tensor.shape.LinearIndex({b, y, x, dst_ch})] =
              dst_val;
        }
      }
    }
  }
  return dst_ref_tensor;
}

TensorInt32 FullyConnectedReference(
    const ml_drift::Tensor<BHWC, DataType::UINT8>& src_tensor_ui8,
    const ml_drift::Tensor<OHWI, DataType::INT8>& weights_i8,
    int src_sum_scale) {
  TensorInt32 dst_ref_tensor;
  dst_ref_tensor.shape = src_tensor_ui8.shape;
  dst_ref_tensor.shape.c = weights_i8.shape.o;
  dst_ref_tensor.data.resize(dst_ref_tensor.shape.DimensionsProduct(), 0.0f);
  for (int b = 0; b < src_tensor_ui8.shape.b; ++b) {
    for (int y = 0; y < src_tensor_ui8.shape.h; ++y) {
      for (int x = 0; x < src_tensor_ui8.shape.w; ++x) {
        for (int dst_ch = 0; dst_ch < weights_i8.shape.o; ++dst_ch) {
          int dst_val = 0;
          for (int src_ch = 0; src_ch < weights_i8.shape.i; ++src_ch) {
            int src_i32 =
                src_tensor_ui8
                    .data[src_tensor_ui8.shape.LinearIndex({b, y, x, src_ch})];
            int weights_i32 =
                weights_i8
                    .data[weights_i8.shape.LinearIndex({dst_ch, 0, 0, src_ch})];
            dst_val += src_i32 * weights_i32;
          }
          dst_ref_tensor
              .data[dst_ref_tensor.shape.LinearIndex({b, y, x, dst_ch})] =
              dst_val;
        }
      }
    }
  }
  if (src_sum_scale != 0) {
    for (int b = 0; b < src_tensor_ui8.shape.b; ++b) {
      for (int y = 0; y < src_tensor_ui8.shape.h; ++y) {
        for (int x = 0; x < src_tensor_ui8.shape.w; ++x) {
          int sum = 0;
          for (int ch = 0; ch < src_tensor_ui8.shape.c; ++ch) {
            sum += src_tensor_ui8
                       .data[src_tensor_ui8.shape.LinearIndex({b, y, x, ch})];
          }
          for (int dst_ch = 0; dst_ch < weights_i8.shape.o; ++dst_ch) {
            dst_ref_tensor
                .data[dst_ref_tensor.shape.LinearIndex({b, y, x, dst_ch})] +=
                src_sum_scale * sum;
          }
        }
      }
    }
  }
  return dst_ref_tensor;
}

void LSTMReference(const TensorFloat32& active_temp_in,
                   const TensorFloat32& prev_state_in,
                   TensorFloat32* new_state_out,
                   TensorFloat32* activation_out) {
  // http://google3/third_party/tensorflow/lite/kernels/internal/reference/reference_ops.h?l=1928
  const int out_depth = new_state_out->shape.c;
  for (int b = 0; b < new_state_out->shape.b; ++b) {
    for (int h = 0; h < new_state_out->shape.h; ++h) {
      for (int w = 0; w < new_state_out->shape.w; ++w) {
        for (int c = 0; c < new_state_out->shape.c; ++c) {
          int linear_index = prev_state_in.shape.LinearIndex({b, h, w, c});

          float prev_state = prev_state_in.data[linear_index];

          int gate0_index =
              active_temp_in.shape.LinearIndex({b, h, w, 0 * out_depth + c});
          int gate1_index =
              active_temp_in.shape.LinearIndex({b, h, w, 1 * out_depth + c});
          int gate2_index =
              active_temp_in.shape.LinearIndex({b, h, w, 2 * out_depth + c});
          int gate3_index =
              active_temp_in.shape.LinearIndex({b, h, w, 3 * out_depth + c});

          // input, new, forget, output
          float gate_0 = active_temp_in.data[gate0_index];
          float gate_1 = active_temp_in.data[gate1_index];
          float gate_2 = active_temp_in.data[gate2_index];
          float gate_3 = active_temp_in.data[gate3_index];

          float input_gate = 1.0f / (1.0f + std::exp(-gate_0));   // sig(x)
          float new_input = std::tanh(gate_1);                    // tanh(x)
          float forget_gate = 1.0f / (1.0f + std::exp(-gate_2));  // sig(x)
          float output_gate = 1.0f / (1.0f + std::exp(-gate_3));  // sig(x)

          float new_state = input_gate * new_input + forget_gate * prev_state;
          float activation = output_gate * std::tanh(new_state);

          new_state_out->data[linear_index] = new_state;
          activation_out->data[linear_index] = activation;
        }
      }
    }
  }
}

std::vector<TensorFloat32> LSTMReference(const TensorFloat32& active_temp,
                                         const TensorFloat32& prev_state) {
  TensorFloat32 new_state = MakeZeroTensor(prev_state.shape);
  TensorFloat32 activation = MakeZeroTensor(prev_state.shape);
  LSTMReference(active_temp, prev_state, &new_state, &activation);
  return std::vector<TensorFloat32>{new_state, activation};
}

void MaxUnpoolingReference(const MaxUnpooling2DAttributes& attr,
                           const TensorFloat32& src,
                           const TensorFloat32& src_indexes,
                           TensorFloat32* dst) {
  for (size_t i = 0; i < dst->data.size(); ++i) {
    dst->data[i] = 0.0f;
  }

  for (int b = 0; b < dst->shape.b; ++b) {
    for (int c = 0; c < dst->shape.c; ++c) {
      int y = -attr.padding.prepended.h;
      for (int i = 0; i < src.shape.h; i++, y += attr.strides.h) {
        int x = -attr.padding.prepended.w;
        for (int j = 0; j < src.shape.w; j++, x += attr.strides.w) {
          const int src_index = src.shape.LinearIndex({b, i, j, c});
          int ind = static_cast<int>(src_indexes.data[src_index]);
          int dy = ind / attr.kernel.w;
          int dx = ind % attr.kernel.w;
          const int dst_index = dst->shape.LinearIndex({b, y + dy, x + dx, c});
          dst->data[dst_index] = src.data[src_index];
        }
      }
    }
  }
}

void MaxUnpoolingReference(const MaxUnpooling3DAttributes& attr,
                           const Tensor5DFloat32& src,
                           const Tensor5DFloat32& src_indexes,
                           Tensor5DFloat32* dst) {
  for (size_t i = 0; i < dst->data.size(); ++i) {
    dst->data[i] = 0.0f;
  }

  for (int b = 0; b < dst->shape.b; ++b) {
    for (int c = 0; c < dst->shape.c; ++c) {
      int y = -attr.padding.prepended.h;
      for (int i = 0; i < src.shape.h; i++, y += attr.strides.h) {
        int x = -attr.padding.prepended.w;
        for (int j = 0; j < src.shape.w; j++, x += attr.strides.w) {
          int z = -attr.padding.prepended.d;
          for (int k = 0; k < src.shape.d; k++, z += attr.strides.d) {
            const int src_index = src.shape.LinearIndex({b, i, j, k, c});
            int ind = static_cast<int>(src_indexes.data[src_index]);
            int dd = ind % attr.kernel.d;
            ind /= attr.kernel.d;
            int dw = ind % attr.kernel.w;
            int dh = ind / attr.kernel.w;
            const int dst_index =
                dst->shape.LinearIndex({b, y + dh, x + dw, z + dd, c});
            dst->data[dst_index] = src.data[src_index];
          }
        }
      }
    }
  }
}

TensorFloat32 MaxUnpoolingReference(const MaxUnpooling2DAttributes& attr,
                                    const TensorFloat32& input,
                                    const TensorFloat32& src_indexes) {
  TensorFloat32 output =
      MakeZeroTensor(CalculateOutputShape(input.shape, attr));
  MaxUnpoolingReference(attr, input, src_indexes, &output);
  return output;
}

Tensor5DFloat32 MaxUnpoolingReference(const MaxUnpooling3DAttributes& attr,
                                      const Tensor5DFloat32& input,
                                      const Tensor5DFloat32& src_indexes) {
  Tensor5DFloat32 output =
      MakeZeroTensor(CalculateOutputShape(input.shape, attr));
  MaxUnpoolingReference(attr, input, src_indexes, &output);
  return output;
}

void AveragePoolingReference(const Pooling2DAttributes& attr,
                             const TensorFloat32& src, TensorFloat32* dst) {
  for (int b = 0; b < dst->shape.b; ++b) {
    for (int z = 0; z < dst->shape.c; ++z) {
      for (int y = 0; y < dst->shape.h; ++y) {
        for (int x = 0; x < dst->shape.w; ++x) {
          float summa = 0.0f;
          float window_size = 0.0f;
          for (int ky = 0; ky < attr.kernel.h; ++ky) {
            for (int kx = 0; kx < attr.kernel.w; ++kx) {
              const int src_x =
                  x * attr.strides.w - attr.padding.prepended.w + kx;
              const int src_y =
                  y * attr.strides.h - attr.padding.prepended.h + ky;
              if (src_x >= 0 && src_y >= 0 && src_x < src.shape.w &&
                  src_y < src.shape.h) {
                const int src_index =
                    src.shape.LinearIndex({b, src_y, src_x, z});
                summa += src.data[src_index];
                window_size += 1.0;
              }
            }
          }
          const int dst_index = dst->shape.LinearIndex({b, y, x, z});
          dst->data[dst_index] = summa / window_size;
        }
      }
    }
  }
}

void AveragePoolingReference(const Pooling3DAttributes& attr,
                             const Tensor5DFloat32& src, Tensor5DFloat32* dst) {
  for (int b = 0; b < dst->shape.b; ++b) {
    for (int z = 0; z < dst->shape.c; ++z) {
      for (int y = 0; y < dst->shape.h; ++y) {
        for (int x = 0; x < dst->shape.w; ++x) {
          for (int d = 0; d < dst->shape.d; ++d) {
            float summa = 0.0f;
            float window_size = 0.0f;
            for (int ky = 0; ky < attr.kernel.h; ++ky) {
              for (int kx = 0; kx < attr.kernel.w; ++kx) {
                for (int kd = 0; kd < attr.kernel.d; ++kd) {
                  const int src_x =
                      x * attr.strides.w - attr.padding.prepended.w + kx;
                  const int src_y =
                      y * attr.strides.h - attr.padding.prepended.h + ky;
                  const int src_d =
                      d * attr.strides.d - attr.padding.prepended.d + kd;
                  if (src_x >= 0 && src_y >= 0 && src_d >= 0 &&
                      src_x < src.shape.w && src_y < src.shape.h &&
                      src_d < src.shape.d) {
                    const int src_index =
                        src.shape.LinearIndex({b, src_y, src_x, src_d, z});
                    summa += src.data[src_index];
                    window_size += 1.0;
                  }
                }
              }
            }
            const int dst_index = dst->shape.LinearIndex({b, y, x, d, z});
            dst->data[dst_index] = summa / window_size;
          }
        }
      }
    }
  }
}

void MaxPoolingReference(const Pooling2DAttributes& attr,
                         const TensorFloat32& src, TensorFloat32* dst) {
  for (int b = 0; b < dst->shape.b; ++b) {
    for (int z = 0; z < dst->shape.c; ++z) {
      for (int y = 0; y < dst->shape.h; ++y) {
        for (int x = 0; x < dst->shape.w; ++x) {
          float max_val = -std::numeric_limits<float>::max();
          for (int ky = 0; ky < attr.kernel.h; ++ky) {
            for (int kx = 0; kx < attr.kernel.w; ++kx) {
              const int src_x =
                  x * attr.strides.w - attr.padding.prepended.w + kx;
              const int src_y =
                  y * attr.strides.h - attr.padding.prepended.h + ky;
              if (src_x >= 0 && src_y >= 0 && src_x < src.shape.w &&
                  src_y < src.shape.h) {
                const int src_index =
                    src.shape.LinearIndex({b, src_y, src_x, z});
                max_val = std::max(max_val, src.data[src_index]);
              }
            }
          }
          const int dst_index = dst->shape.LinearIndex({b, y, x, z});
          dst->data[dst_index] = max_val;
        }
      }
    }
  }
}

void MaxPoolingReference(const Pooling3DAttributes& attr,
                         const Tensor5DFloat32& src, Tensor5DFloat32* dst) {
  for (int b = 0; b < dst->shape.b; ++b) {
    for (int z = 0; z < dst->shape.c; ++z) {
      for (int y = 0; y < dst->shape.h; ++y) {
        for (int x = 0; x < dst->shape.w; ++x) {
          for (int d = 0; d < dst->shape.d; ++d) {
            float max_val = -std::numeric_limits<float>::max();
            for (int ky = 0; ky < attr.kernel.h; ++ky) {
              for (int kx = 0; kx < attr.kernel.w; ++kx) {
                for (int kd = 0; kd < attr.kernel.d; ++kd) {
                  const int src_x =
                      x * attr.strides.w - attr.padding.prepended.w + kx;
                  const int src_y =
                      y * attr.strides.h - attr.padding.prepended.h + ky;
                  const int src_d =
                      d * attr.strides.d - attr.padding.prepended.d + kd;
                  if (src_x >= 0 && src_y >= 0 && src_d >= 0 &&
                      src_x < src.shape.w && src_y < src.shape.h &&
                      src_d < src.shape.d) {
                    const int src_index =
                        src.shape.LinearIndex({b, src_y, src_x, src_d, z});
                    max_val = std::max(max_val, src.data[src_index]);
                  }
                }
              }
            }
            const int dst_index = dst->shape.LinearIndex({b, y, x, d, z});
            dst->data[dst_index] = max_val;
          }
        }
      }
    }
  }
}

std::vector<TensorFloat32> MaxPoolingIndicesReference(
    const Pooling2DAttributes& attr, const TensorFloat32& src) {
  TensorFloat32 output = MakeZeroTensor(CalculateOutputShape(src.shape, attr));
  TensorFloat32 indices = MakeZeroTensor(CalculateOutputShape(src.shape, attr));

  for (int b = 0; b < output.shape.b; ++b) {
    for (int z = 0; z < output.shape.c; ++z) {
      for (int y = 0; y < output.shape.h; ++y) {
        for (int x = 0; x < output.shape.w; ++x) {
          float max_val = -std::numeric_limits<float>::max();
          int max_ind = 0;
          for (int ky = 0; ky < attr.kernel.h; ++ky) {
            for (int kx = 0; kx < attr.kernel.w; ++kx) {
              const int src_x =
                  x * attr.strides.w - attr.padding.prepended.w + kx;
              const int src_y =
                  y * attr.strides.h - attr.padding.prepended.h + ky;
              if (src_x >= 0 && src_y >= 0 && src_x < src.shape.w &&
                  src_y < src.shape.h) {
                const int src_index =
                    src.shape.LinearIndex({b, src_y, src_x, z});
                if (src.data[src_index] > max_val) {
                  max_val = src.data[src_index];
                  max_ind = ky * attr.kernel.w + kx;
                }
              }
            }
          }
          const int dst_index = output.shape.LinearIndex({b, y, x, z});
          indices.data[dst_index] = static_cast<float>(max_ind);
          output.data[dst_index] = max_val;
        }
      }
    }
  }
  return {output, indices};
}

std::vector<Tensor5DFloat32> MaxPoolingIndicesReference(
    const Pooling3DAttributes& attr, const Tensor5DFloat32& src) {
  Tensor5DFloat32 output =
      MakeZeroTensor(CalculateOutputShape(src.shape, attr));
  Tensor5DFloat32 indices =
      MakeZeroTensor(CalculateOutputShape(src.shape, attr));

  for (int b = 0; b < output.shape.b; ++b) {
    for (int z = 0; z < output.shape.c; ++z) {
      for (int y = 0; y < output.shape.h; ++y) {
        for (int x = 0; x < output.shape.w; ++x) {
          for (int d = 0; d < output.shape.d; ++d) {
            float max_val = -std::numeric_limits<float>::max();
            int max_ind = 0;
            for (int ky = 0; ky < attr.kernel.h; ++ky) {
              for (int kx = 0; kx < attr.kernel.w; ++kx) {
                for (int kd = 0; kd < attr.kernel.d; ++kd) {
                  const int src_x =
                      x * attr.strides.w - attr.padding.prepended.w + kx;
                  const int src_y =
                      y * attr.strides.h - attr.padding.prepended.h + ky;
                  const int src_d =
                      d * attr.strides.d - attr.padding.prepended.d + kd;
                  if (src_x >= 0 && src_y >= 0 && src_d >= 0 &&
                      src_x < src.shape.w && src_y < src.shape.h &&
                      src_d < src.shape.d) {
                    const int src_index =
                        src.shape.LinearIndex({b, src_y, src_x, src_d, z});
                    if (src.data[src_index] > max_val) {
                      max_val = src.data[src_index];
                      max_ind = (ky * attr.kernel.w + kx) * attr.kernel.d + kd;
                    }
                  }
                }
              }
            }
            const int dst_index = output.shape.LinearIndex({b, y, x, d, z});
            indices.data[dst_index] = static_cast<float>(max_ind);
            output.data[dst_index] = max_val;
          }
        }
      }
    }
  }
  return {output, indices};
}

TensorFloat32 AveragePoolingReference(const Pooling2DAttributes& attr,
                                      const TensorFloat32& input) {
  TensorFloat32 output =
      MakeZeroTensor(CalculateOutputShape(input.shape, attr));
  AveragePoolingReference(attr, input, &output);
  return output;
}

Tensor5DFloat32 AveragePoolingReference(const Pooling3DAttributes& attr,
                                        const Tensor5DFloat32& input) {
  Tensor5DFloat32 output =
      MakeZeroTensor(CalculateOutputShape(input.shape, attr));
  AveragePoolingReference(attr, input, &output);
  return output;
}

std::vector<TensorFloat32> MaxPoolingReference(const Pooling2DAttributes& attr,
                                               const TensorFloat32& input) {
  if (!attr.output_indices) {
    TensorFloat32 output =
        MakeZeroTensor(CalculateOutputShape(input.shape, attr));
    MaxPoolingReference(attr, input, &output);
    return {output};
  }
  return MaxPoolingIndicesReference(attr, input);
}

std::vector<Tensor5DFloat32> MaxPoolingReference(
    const Pooling3DAttributes& attr, const Tensor5DFloat32& input) {
  if (!attr.output_indices) {
    Tensor5DFloat32 output =
        MakeZeroTensor(CalculateOutputShape(input.shape, attr));
    MaxPoolingReference(attr, input, &output);
    return {output};
  }
  return MaxPoolingIndicesReference(attr, input);
}

void ReduceReference(const std::set<Axis>& axis_to_reduce,
                     OperationType op_type, const TensorFloat32& src,
                     TensorFloat32* dst) {
  for (int b = 0; b < dst->shape.b; ++b) {
    for (int h = 0; h < dst->shape.h; ++h) {
      for (int w = 0; w < dst->shape.w; ++w) {
        for (int c = 0; c < dst->shape.c; ++c) {
          double reduced = 0.0;
          if (op_type == OperationType::REDUCE_PRODUCT) {
            reduced = 1.0;
          } else if (op_type == OperationType::REDUCE_MINIMUM) {
            reduced = DBL_MAX;
          } else if (op_type == OperationType::REDUCE_MAXIMUM) {
            reduced = -DBL_MAX;
          }
          int sb = axis_to_reduce.count(Axis::BATCH) ? 0 : b;
          int sb_size = axis_to_reduce.count(Axis::BATCH) ? src.shape.b : b + 1;
          int sh_size =
              axis_to_reduce.count(Axis::HEIGHT) ? src.shape.h : h + 1;
          int sw_size = axis_to_reduce.count(Axis::WIDTH) ? src.shape.w : w + 1;
          int sc_size =
              axis_to_reduce.count(Axis::CHANNELS) ? src.shape.c : c + 1;
          for (; sb < sb_size; ++sb) {
            int sh = axis_to_reduce.count(Axis::HEIGHT) ? 0 : h;
            for (; sh < sh_size; ++sh) {
              int sw = axis_to_reduce.count(Axis::WIDTH) ? 0 : w;
              for (; sw < sw_size; ++sw) {
                int sc = axis_to_reduce.count(Axis::CHANNELS) ? 0 : c;
                for (; sc < sc_size; ++sc) {
                  const int src_index = src.shape.LinearIndex({sb, sh, sw, sc});
                  double src_val = src.data[src_index];
                  if (op_type == OperationType::REDUCE_PRODUCT) {
                    reduced *= src_val;
                  } else if (op_type == OperationType::REDUCE_MINIMUM) {
                    reduced = std::min(src_val, reduced);
                  } else if (op_type == OperationType::REDUCE_MAXIMUM) {
                    reduced = std::max(src_val, reduced);
                  } else {
                    reduced += src_val;
                  }
                }
              }
            }
          }
          if (op_type == OperationType::MEAN) {
            double total_reduce = 1.0;
            if (axis_to_reduce.count(Axis::BATCH)) {
              total_reduce *= src.shape.b;
            }
            if (axis_to_reduce.count(Axis::HEIGHT)) {
              total_reduce *= src.shape.h;
            }
            if (axis_to_reduce.count(Axis::WIDTH)) {
              total_reduce *= src.shape.w;
            }
            if (axis_to_reduce.count(Axis::CHANNELS)) {
              total_reduce *= src.shape.c;
            }
            reduced /= total_reduce;
          }
          const int dst_index = dst->shape.LinearIndex({b, h, w, c});
          dst->data[dst_index] = reduced;
        }
      }
    }
  }
}

TensorFloat32 ReduceReference(const std::set<Axis>& axis_to_reduce,
                              OperationType op_type,
                              const TensorFloat32& input) {
  BHWC dst_shape;
  dst_shape.b = axis_to_reduce.count(Axis::BATCH) ? 1 : input.shape.b;
  dst_shape.h = axis_to_reduce.count(Axis::HEIGHT) ? 1 : input.shape.h;
  dst_shape.w = axis_to_reduce.count(Axis::WIDTH) ? 1 : input.shape.w;
  dst_shape.c = axis_to_reduce.count(Axis::CHANNELS) ? 1 : input.shape.c;
  TensorFloat32 output = MakeZeroTensor(dst_shape);
  ReduceReference(axis_to_reduce, op_type, input, &output);
  return output;
}

void ReduceReference(const std::set<Axis>& axis_to_reduce,
                     OperationType op_type, const Tensor5DFloat32& src,
                     Tensor5DFloat32* dst) {
  for (int b = 0; b < dst->shape.b; ++b) {
    for (int h = 0; h < dst->shape.h; ++h) {
      for (int w = 0; w < dst->shape.w; ++w) {
        for (int d = 0; d < dst->shape.d; ++d) {
          for (int c = 0; c < dst->shape.c; ++c) {
            double reduced = 0.0;
            if (op_type == OperationType::REDUCE_PRODUCT) {
              reduced = 1.0;
            } else if (op_type == OperationType::REDUCE_MINIMUM) {
              reduced = DBL_MAX;
            } else if (op_type == OperationType::REDUCE_MAXIMUM) {
              reduced = -DBL_MAX;
            }
            int sb = axis_to_reduce.count(Axis::BATCH) ? 0 : b;
            int sb_size =
                axis_to_reduce.count(Axis::BATCH) ? src.shape.b : b + 1;
            int sh_size =
                axis_to_reduce.count(Axis::HEIGHT) ? src.shape.h : h + 1;
            int sw_size =
                axis_to_reduce.count(Axis::WIDTH) ? src.shape.w : w + 1;
            int sd_size =
                axis_to_reduce.count(Axis::DEPTH) ? src.shape.d : d + 1;
            int sc_size =
                axis_to_reduce.count(Axis::CHANNELS) ? src.shape.c : c + 1;
            for (; sb < sb_size; ++sb) {
              int sh = axis_to_reduce.count(Axis::HEIGHT) ? 0 : h;
              for (; sh < sh_size; ++sh) {
                int sw = axis_to_reduce.count(Axis::WIDTH) ? 0 : w;
                for (; sw < sw_size; ++sw) {
                  int sd = axis_to_reduce.count(Axis::DEPTH) ? 0 : d;
                  for (; sd < sd_size; ++sd) {
                    int sc = axis_to_reduce.count(Axis::CHANNELS) ? 0 : c;
                    for (; sc < sc_size; ++sc) {
                      const int src_index =
                          src.shape.LinearIndex({sb, sh, sw, sd, sc});
                      double src_val = src.data[src_index];
                      if (op_type == OperationType::REDUCE_PRODUCT) {
                        reduced *= src_val;
                      } else if (op_type == OperationType::REDUCE_MINIMUM) {
                        reduced = std::min(src_val, reduced);
                      } else if (op_type == OperationType::REDUCE_MAXIMUM) {
                        reduced = std::max(src_val, reduced);
                      } else {
                        reduced += src_val;
                      }
                    }
                  }
                }
              }
            }
            if (op_type == OperationType::MEAN) {
              double total_reduce = 1.0;
              if (axis_to_reduce.count(Axis::BATCH)) {
                total_reduce *= src.shape.b;
              }
              if (axis_to_reduce.count(Axis::HEIGHT)) {
                total_reduce *= src.shape.h;
              }
              if (axis_to_reduce.count(Axis::WIDTH)) {
                total_reduce *= src.shape.w;
              }
              if (axis_to_reduce.count(Axis::DEPTH)) {
                total_reduce *= src.shape.d;
              }
              if (axis_to_reduce.count(Axis::CHANNELS)) {
                total_reduce *= src.shape.c;
              }
              reduced /= total_reduce;
            }
            const int dst_index = dst->shape.LinearIndex({b, h, w, d, c});
            dst->data[dst_index] = reduced;
          }
        }
      }
    }
  }
}

Tensor5DFloat32 ReduceReference(const std::set<Axis>& axis_to_reduce,
                                OperationType op_type,
                                const Tensor5DFloat32& input) {
  BHWDC dst_shape;
  dst_shape.b = axis_to_reduce.count(Axis::BATCH) ? 1 : input.shape.b;
  dst_shape.h = axis_to_reduce.count(Axis::HEIGHT) ? 1 : input.shape.h;
  dst_shape.w = axis_to_reduce.count(Axis::WIDTH) ? 1 : input.shape.w;
  dst_shape.d = axis_to_reduce.count(Axis::DEPTH) ? 1 : input.shape.d;
  dst_shape.c = axis_to_reduce.count(Axis::CHANNELS) ? 1 : input.shape.c;
  Tensor5DFloat32 output = MakeZeroTensor(dst_shape);
  ReduceReference(axis_to_reduce, op_type, input, &output);
  return output;
}

void ReshapeReference(const TensorFloat32& src, TensorFloat32* dst) {
  for (int dst_b = 0; dst_b < dst->shape.b; ++dst_b) {
    for (int dst_h = 0; dst_h < dst->shape.h; ++dst_h) {
      for (int dst_w = 0; dst_w < dst->shape.w; ++dst_w) {
        for (int dst_c = 0; dst_c < dst->shape.c; ++dst_c) {
          int dst_bhwc =
              ((dst_b * dst->shape.h + dst_h) * dst->shape.w + dst_w) *
                  dst->shape.c +
              dst_c;
          int src_c = dst_bhwc % src.shape.c;
          dst_bhwc = dst_bhwc / src.shape.c;
          int src_w = dst_bhwc % src.shape.w;
          dst_bhwc = dst_bhwc / src.shape.w;
          int src_h = dst_bhwc % src.shape.h;
          int src_b = dst_bhwc / src.shape.h;
          const int src_index =
              src.shape.LinearIndex({src_b, src_h, src_w, src_c});
          const float src_val = src.data[src_index];
          const int dst_index =
              dst->shape.LinearIndex({dst_b, dst_h, dst_w, dst_c});
          dst->data[dst_index] = src_val;
        }
      }
    }
  }
}

TensorFloat32 ReshapeReference(const ReshapeAttributes& attr,
                               const TensorFloat32& input) {
  TensorFloat32 output = MakeZeroTensor(attr.new_shape);
  ReshapeReference(input, &output);
  return output;
}

void ReshapeReference(const Tensor5DFloat32& src, Tensor5DFloat32* dst) {
  for (int dst_b = 0; dst_b < dst->shape.b; ++dst_b) {
    for (int dst_h = 0; dst_h < dst->shape.h; ++dst_h) {
      for (int dst_w = 0; dst_w < dst->shape.w; ++dst_w) {
        for (int dst_d = 0; dst_d < dst->shape.d; ++dst_d) {
          for (int dst_c = 0; dst_c < dst->shape.c; ++dst_c) {
            int dst_bhwdc =
                (((dst_b * dst->shape.h + dst_h) * dst->shape.w + dst_w) *
                     dst->shape.d +
                 dst_d) *
                    dst->shape.c +
                dst_c;
            int src_c = dst_bhwdc % src.shape.c;
            dst_bhwdc = dst_bhwdc / src.shape.c;
            int src_d = dst_bhwdc % src.shape.d;
            dst_bhwdc = dst_bhwdc / src.shape.d;
            int src_w = dst_bhwdc % src.shape.w;
            dst_bhwdc = dst_bhwdc / src.shape.w;
            int src_h = dst_bhwdc % src.shape.h;
            int src_b = dst_bhwdc / src.shape.h;
            const int src_index =
                src.shape.LinearIndex({src_b, src_h, src_w, src_d, src_c});
            const float src_val = src.data[src_index];
            const int dst_index =
                dst->shape.LinearIndex({dst_b, dst_h, dst_w, dst_d, dst_c});
            dst->data[dst_index] = src_val;
          }
        }
      }
    }
  }
}

Tensor5DFloat32 ReshapeReference(const Reshape3DAttributes& attr,
                                 const Tensor5DFloat32& input) {
  Tensor5DFloat32 output = MakeZeroTensor(attr.new_shape);
  ReshapeReference(input, &output);
  return output;
}

namespace {

float mix(float x, float y, float a) { return x + (y - x) * a; }

float BilinearInterpolation(float h, float w, int c, int b,
                            const TensorFloat32& input) {
  int ih = static_cast<int>(floor(h));
  int iw = static_cast<int>(floor(w));
  int st_h = std::max(std::min(ih, input.shape.h - 1), 0);
  int st_w = std::max(std::min(iw, input.shape.w - 1), 0);
  int st_z = std::min(ih + 1, input.shape.h - 1);
  int st_y = std::min(iw + 1, input.shape.w - 1);

  float t_h = h - floor(h);  // interpolating factors
  float t_w = w - floor(w);

  int src_index0 = input.shape.LinearIndex({b, st_h, st_w, c});
  int src_index1 = input.shape.LinearIndex({b, st_z, st_w, c});
  int src_index2 = input.shape.LinearIndex({b, st_h, st_y, c});
  int src_index3 = input.shape.LinearIndex({b, st_z, st_y, c});

  float tex11 = input.data[src_index0];
  float tex21 = input.data[src_index1];
  float tex12 = input.data[src_index2];
  float tex22 = input.data[src_index3];

  // bilinear interpolation
  return mix(mix(tex11, tex21, t_h), mix(tex12, tex22, t_h), t_w);
}

float BilinearInterpolation(float h, float w, float d, int c, int b,
                            const Tensor5DFloat32& input) {
  int ih = static_cast<int>(floor(h));
  int iw = static_cast<int>(floor(w));
  int id = static_cast<int>(floor(d));
  int st_h = std::max(std::min(ih, input.shape.h - 1), 0);
  int st_w = std::max(std::min(iw, input.shape.w - 1), 0);
  int st_d = std::max(std::min(id, input.shape.d - 1), 0);
  int st_y = std::min(ih + 1, input.shape.h - 1);
  int st_x = std::min(iw + 1, input.shape.w - 1);
  int st_z = std::min(id + 1, input.shape.d - 1);

  float t_h = h - floor(h);  // interpolating factors
  float t_w = w - floor(w);
  float t_d = d - floor(d);

  int src_index0 = input.shape.LinearIndex({b, st_h, st_w, st_d, c});
  int src_index1 = input.shape.LinearIndex({b, st_y, st_w, st_d, c});
  int src_index2 = input.shape.LinearIndex({b, st_h, st_x, st_d, c});
  int src_index3 = input.shape.LinearIndex({b, st_y, st_x, st_d, c});
  int src_index4 = input.shape.LinearIndex({b, st_h, st_w, st_z, c});
  int src_index5 = input.shape.LinearIndex({b, st_y, st_w, st_z, c});
  int src_index6 = input.shape.LinearIndex({b, st_h, st_x, st_z, c});
  int src_index7 = input.shape.LinearIndex({b, st_y, st_x, st_z, c});

  float tex111 = input.data[src_index0];
  float tex211 = input.data[src_index1];
  float tex121 = input.data[src_index2];
  float tex221 = input.data[src_index3];
  float tex112 = input.data[src_index4];
  float tex212 = input.data[src_index5];
  float tex122 = input.data[src_index6];
  float tex222 = input.data[src_index7];

  float t0 = mix(mix(tex111, tex211, t_h), mix(tex121, tex221, t_h), t_w);
  float t1 = mix(mix(tex112, tex212, t_h), mix(tex122, tex222, t_h), t_w);

  // bilinear interpolation
  return mix(t0, t1, t_d);
}

}  // namespace

TensorFloat32 ResizeReference(const Resize2DAttributes& attr,
                              const TensorFloat32& input) {
  TensorFloat32 output =
      MakeZeroTensor(CalculateOutputShape(input.shape, attr));

  float scale_h = CalculateResizeScale(input.shape.h, attr.new_shape.h, attr);
  float scale_w = CalculateResizeScale(input.shape.w, attr.new_shape.w, attr);
  for (int b = 0; b < output.shape.b; ++b) {
    for (int h = 0; h < output.shape.h; ++h) {
      for (int w = 0; w < output.shape.w; ++w) {
        for (int c = 0; c < output.shape.c; ++c) {
          const int index = output.shape.LinearIndex({b, h, w, c});
          if (attr.type == SamplingType::BILINEAR) {
            const float src_h = attr.half_pixel_centers
                                    ? (scale_h * (h + 0.5) - 0.5)
                                    : scale_h * h;
            const float src_w = attr.half_pixel_centers
                                    ? (scale_w * (w + 0.5) - 0.5)
                                    : scale_w * w;
            output.data[index] =
                BilinearInterpolation(src_h, src_w, c, b, input);
          } else {
            float fxc;
            float fyc;
            if (attr.half_pixel_centers) {
              fxc = (w + 0.5f) * scale_w;
              fyc = (h + 0.5f) * scale_h;
            } else {
              fxc = w * scale_w;
              fyc = h * scale_h;
            }
            if (attr.align_corners) {
              fxc = roundf(fxc);
              fyc = roundf(fyc);
            } else {
              fxc = floorf(fxc);
              fyc = floorf(fyc);
            }
            int src_w = static_cast<int>(fxc);
            int src_h = static_cast<int>(fyc);
            src_w = std::max(0, src_w);
            src_h = std::max(0, src_h);
            src_w = std::min(src_w, input.shape.w - 1);
            src_h = std::min(src_h, input.shape.h - 1);
            const int src_index = input.shape.LinearIndex({b, src_h, src_w, c});
            output.data[index] = input.data[src_index];
          }
        }
      }
    }
  }
  return output;
}

Tensor5DFloat32 ResizeReference(const Resize3DAttributes& attr,
                                const Tensor5DFloat32& input) {
  Tensor5DFloat32 output =
      MakeZeroTensor(CalculateOutputShape(input.shape, attr));

  float scale_h = CalculateResizeScale(input.shape.h, attr.new_shape.h, attr);
  float scale_w = CalculateResizeScale(input.shape.w, attr.new_shape.w, attr);
  float scale_d = CalculateResizeScale(input.shape.d, attr.new_shape.d, attr);
  for (int b = 0; b < output.shape.b; ++b) {
    for (int h = 0; h < output.shape.h; ++h) {
      for (int w = 0; w < output.shape.w; ++w) {
        for (int d = 0; d < output.shape.d; ++d) {
          for (int c = 0; c < output.shape.c; ++c) {
            const int index = output.shape.LinearIndex({b, h, w, d, c});
            if (attr.type == SamplingType::BILINEAR) {
              const float src_h = attr.half_pixel_centers
                                      ? (scale_h * (h + 0.5) - 0.5)
                                      : scale_h * h;
              const float src_w = attr.half_pixel_centers
                                      ? (scale_w * (w + 0.5) - 0.5)
                                      : scale_w * w;
              const float src_d = attr.half_pixel_centers
                                      ? (scale_d * (d + 0.5) - 0.5)
                                      : scale_d * d;
              output.data[index] =
                  BilinearInterpolation(src_h, src_w, src_d, c, b, input);
            } else {
              float fxc;
              float fyc;
              float fzc;
              if (attr.half_pixel_centers) {
                fxc = (w + 0.5f) * scale_w;
                fyc = (h + 0.5f) * scale_h;
                fzc = (d + 0.5f) * scale_d;
              } else {
                fxc = w * scale_w;
                fyc = h * scale_h;
                fzc = d * scale_d;
              }
              if (attr.align_corners) {
                fxc = roundf(fxc);
                fyc = roundf(fyc);
                fzc = roundf(fzc);
              } else {
                fxc = floorf(fxc);
                fyc = floorf(fyc);
                fzc = floorf(fzc);
              }
              int src_w = static_cast<int>(fxc);
              int src_h = static_cast<int>(fyc);
              int src_d = static_cast<int>(fzc);
              src_w = std::max(0, src_w);
              src_h = std::max(0, src_h);
              src_d = std::max(0, src_d);
              src_w = std::min(src_w, input.shape.w - 1);
              src_h = std::min(src_h, input.shape.h - 1);
              src_d = std::min(src_d, input.shape.d - 1);
              const int src_index =
                  input.shape.LinearIndex({b, src_h, src_w, src_d, c});
              output.data[index] = input.data[src_index];
            }
          }
        }
      }
    }
  }
  return output;
}

namespace {

void CalculateSoftmax(const std::vector<BHWC> points, const TensorFloat32& src,
                      TensorFloat32* dst) {
  float shift = std::numeric_limits<float>::lowest();
  for (const auto& pt : points) {
    shift = std::max(shift,
                     src.data[src.shape.LinearIndex({pt.b, pt.h, pt.w, pt.c})]);
  }
  float sum = 0;
  for (const auto& pt : points) {
    sum += std::exp(src.data[src.shape.LinearIndex({pt.b, pt.h, pt.w, pt.c})] -
                    shift);
  }
  for (const auto& pt : points) {
    const int index = src.shape.LinearIndex({pt.b, pt.h, pt.w, pt.c});
    dst->data[index] = std::exp(src.data[index] - shift) / sum;
  }
}

void CalculateSoftmax(const std::vector<BHWDC> points,
                      const Tensor5DFloat32& src, Tensor5DFloat32* dst) {
  float shift = std::numeric_limits<float>::lowest();
  for (const auto& pt : points) {
    shift = std::max(
        shift, src.data[src.shape.LinearIndex({pt.b, pt.h, pt.w, pt.d, pt.c})]);
  }
  float sum = 0;
  for (const auto& pt : points) {
    sum += std::exp(
        src.data[src.shape.LinearIndex({pt.b, pt.h, pt.w, pt.d, pt.c})] -
        shift);
  }
  for (const auto& pt : points) {
    const int index = src.shape.LinearIndex({pt.b, pt.h, pt.w, pt.d, pt.c});
    dst->data[index] = std::exp(src.data[index] - shift) / sum;
  }
}

}  // namespace

void SoftmaxReference(const SoftmaxAttributes& attr,
                      const ml_drift::TensorFloat32& src,
                      ml_drift::TensorFloat32* dst) {
  std::vector<ml_drift::BHWC> points;
  for (int b = 0; b < dst->shape.b; ++b) {
    switch (attr.axis) {
      case ml_drift::Axis::CHANNELS:
        for (int h = 0; h < dst->shape.h; ++h) {
          for (int w = 0; w < dst->shape.w; ++w) {
            points.clear();
            for (int c = 0; c < dst->shape.c; c++) {
              points.push_back(ml_drift::BHWC{b, h, w, c});
            }
            CalculateSoftmax(points, src, dst);
          }
        }
        break;
      case ml_drift::Axis::HEIGHT:
        for (int c = 0; c < dst->shape.c; c++) {
          for (int w = 0; w < dst->shape.w; ++w) {
            points.clear();
            for (int h = 0; h < dst->shape.h; ++h) {
              points.push_back(ml_drift::BHWC{b, h, w, c});
            }
            CalculateSoftmax(points, src, dst);
          }
        }
        break;
      case ml_drift::Axis::WIDTH:
        for (int h = 0; h < dst->shape.h; ++h) {
          for (int c = 0; c < dst->shape.c; c++) {
            points.clear();
            for (int w = 0; w < dst->shape.w; ++w) {
              points.push_back(ml_drift::BHWC{b, h, w, c});
            }
            CalculateSoftmax(points, src, dst);
          }
        }
        break;
      default:
        // tensor is hwc, therefore all cases are covered.
        break;
    }
  }
}

void SoftmaxReference(const ml_drift::SoftmaxAttributes& attr,
                      const ml_drift::Tensor5DFloat32& src,
                      ml_drift::Tensor5DFloat32* dst) {
  std::vector<ml_drift::BHWDC> points;
  for (int b = 0; b < dst->shape.b; ++b) {
    switch (attr.axis) {
      case ml_drift::Axis::CHANNELS:
        for (int h = 0; h < dst->shape.h; ++h) {
          for (int w = 0; w < dst->shape.w; ++w) {
            for (int d = 0; d < dst->shape.d; ++d) {
              points.clear();
              for (int c = 0; c < dst->shape.c; c++) {
                points.push_back(ml_drift::BHWDC{b, h, w, d, c});
              }
              CalculateSoftmax(points, src, dst);
            }
          }
        }
        break;
      case ml_drift::Axis::HEIGHT:
        for (int c = 0; c < dst->shape.c; c++) {
          for (int w = 0; w < dst->shape.w; ++w) {
            for (int d = 0; d < dst->shape.d; ++d) {
              points.clear();
              for (int h = 0; h < dst->shape.h; ++h) {
                points.push_back(ml_drift::BHWDC{b, h, w, d, c});
              }
              CalculateSoftmax(points, src, dst);
            }
          }
        }
        break;
      case ml_drift::Axis::WIDTH:
        for (int h = 0; h < dst->shape.h; ++h) {
          for (int c = 0; c < dst->shape.c; c++) {
            for (int d = 0; d < dst->shape.d; ++d) {
              points.clear();
              for (int w = 0; w < dst->shape.w; ++w) {
                points.push_back(ml_drift::BHWDC{b, h, w, d, c});
              }
              CalculateSoftmax(points, src, dst);
            }
          }
        }
        break;
      case ml_drift::Axis::DEPTH:
        for (int w = 0; w < dst->shape.w; ++w) {
          for (int h = 0; h < dst->shape.h; ++h) {
            for (int c = 0; c < dst->shape.c; c++) {
              points.clear();
              for (int d = 0; d < dst->shape.d; ++d) {
                points.push_back(ml_drift::BHWDC{b, h, w, d, c});
              }
              CalculateSoftmax(points, src, dst);
            }
          }
        }
        break;
      default:
        // tensor is hwc, therefore all cases are covered.
        break;
    }
  }
}

void SoftmaxReduceReference(const SoftmaxAttributes& attr,
                            const ml_drift::TensorFloat32& src,
                            ml_drift::TensorFloat32* dst) {
  std::vector<ml_drift::BHWC> points;
  for (int b = 0; b < dst->shape.b; ++b) {
    for (int h = 0; h < dst->shape.h; ++h) {
      for (int w = 0; w < dst->shape.w; ++w) {
        float max_value = src.data[src.shape.LinearIndex({b, h, w, 0})];
        for (int c = 0; c < src.shape.c; c++) {
          max_value = std::max(max_value,
                               src.data[src.shape.LinearIndex({b, h, w, c})]);
        }
        float sum = 0.0f;
        for (int c = 0; c < src.shape.c; c++) {
          sum += std::exp(src.data[src.shape.LinearIndex({b, h, w, c})] -
                          max_value);
        }
        dst->data[dst->shape.LinearIndex({b, h, w, 0})] = 1.0f / sum;
        dst->data[dst->shape.LinearIndex({b, h, w, 1})] = max_value;
      }
    }
  }
}

ml_drift::TensorFloat32 SoftmaxReference(
    const ml_drift::SoftmaxAttributes& attr,
    const ml_drift::TensorFloat32& input) {
  ml_drift::TensorFloat32 output = MakeZeroTensor(input.shape);
  SoftmaxReference(attr, input, &output);
  return output;
}

::ml_drift::Tensor5DFloat32 SoftmaxReference(
    const ::ml_drift::SoftmaxAttributes& attr,
    const ::ml_drift::Tensor5DFloat32& input) {
  ml_drift::Tensor5DFloat32 output = MakeZeroTensor(input.shape);
  SoftmaxReference(attr, input, &output);
  return output;
}

::ml_drift::TensorFloat32 SoftmaxReduceReference(
    const ::ml_drift::SoftmaxAttributes& attr,
    const ::ml_drift::TensorFloat32& input) {
  ml_drift::TensorFloat32 output =
      MakeZeroTensor(BHWC(input.shape.b, input.shape.h, input.shape.w, 2));
  SoftmaxReduceReference(attr, input, &output);
  return output;
}

namespace {

absl::StatusOr<int> GetNext(int starts, int strides, int ends, int p,
                            int shape_size) {
  int res = 0;
  if (strides > 0) {
    res = starts + p * strides;
    if (ends > 0) {
      if (res > ends) {
        return absl::InvalidArgumentError(
            "unsupported SliceAttributes, index > attr.ends");
      }
    } else {
      if (res > shape_size + ends) {
        return absl::InvalidArgumentError(
            "unsupported SliceAttributes, index > shape_size + attr.ends");
      }
    }
  } else if (strides < 0) {
    if (ends > 0) {
      res = ends + p * strides;
    } else {
      res = shape_size + ends + p * strides;
    }
    if (res < starts) {
      return absl::InvalidArgumentError(
          "unsupported SliceAttributes, index < attr.starts");
    }
  } else {
    return absl::InvalidArgumentError("wrong SliceAttributes, strides = 0");
  }
  return res;
}

}  // namespace

absl::Status SliceReference(const ml_drift::SliceAttributes& attr,
                            const ml_drift::TensorFloat32& src,
                            ml_drift::TensorFloat32* dst) {
  for (int b = 0; b < dst->shape.b; b++) {
    ABSL_ASSIGN_OR_RETURN(int next_b, GetNext(attr.starts.b, attr.strides.b,
                                              attr.ends.b, b, src.shape.b));
    const bool b_inside = next_b >= 0 && next_b < src.shape.b;
    for (int c = 0; c < dst->shape.c; c++) {
      ABSL_ASSIGN_OR_RETURN(int next_ch, GetNext(attr.starts.c, attr.strides.c,
                                                 attr.ends.c, c, src.shape.c));
      const bool c_inside = next_ch >= 0 && next_ch < src.shape.c;
      for (int h = 0; h < dst->shape.h; h++) {
        ABSL_ASSIGN_OR_RETURN(int next_h, GetNext(attr.starts.h, attr.strides.h,
                                                  attr.ends.h, h, src.shape.h));
        const bool h_inside = next_h >= 0 && next_h < src.shape.h;
        for (int w = 0; w < dst->shape.w; w++) {
          ABSL_ASSIGN_OR_RETURN(
              int next_w, GetNext(attr.starts.w, attr.strides.w, attr.ends.w, w,
                                  src.shape.w));
          const bool w_inside = next_w >= 0 && next_w < src.shape.w;
          const int src_index =
              src.shape.LinearIndex({next_b, next_h, next_w, next_ch});
          const int dst_index = dst->shape.LinearIndex({b, h, w, c});
          if (c_inside && h_inside && w_inside && b_inside) {
            dst->data[dst_index] = src.data[src_index];
          } else {
            dst->data[dst_index] = 0.0f;
          }
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<ml_drift::TensorFloat32> SliceReference(
    const ml_drift::SliceAttributes& attr,
    const ml_drift::TensorFloat32& input) {
  ml_drift::TensorFloat32 output =
      MakeZeroTensor(CalculateOutputShape(input.shape, attr));
  ABSL_RETURN_IF_ERROR(SliceReference(attr, input, &output));
  return output;
}

absl::Status SliceReference(const ::ml_drift::Slice3DAttributes& attr,
                            const ::ml_drift::Tensor5DFloat32& src,
                            ::ml_drift::Tensor5DFloat32* dst) {
  for (int b = 0; b < dst->shape.b; b++) {
    ABSL_ASSIGN_OR_RETURN(int next_b, GetNext(attr.starts.b, attr.strides.b,
                                              attr.ends.b, b, src.shape.b));
    const bool b_inside = next_b >= 0 && next_b < src.shape.b;
    for (int c = 0; c < dst->shape.c; c++) {
      ABSL_ASSIGN_OR_RETURN(int next_ch, GetNext(attr.starts.c, attr.strides.c,
                                                 attr.ends.c, c, src.shape.c));
      const bool c_inside = next_ch >= 0 && next_ch < src.shape.c;
      for (int h = 0; h < dst->shape.h; h++) {
        ABSL_ASSIGN_OR_RETURN(int next_h, GetNext(attr.starts.h, attr.strides.h,
                                                  attr.ends.h, h, src.shape.h));
        const bool h_inside = next_h >= 0 && next_h < src.shape.h;
        for (int w = 0; w < dst->shape.w; w++) {
          ABSL_ASSIGN_OR_RETURN(
              int next_w, GetNext(attr.starts.w, attr.strides.w, attr.ends.w, w,
                                  src.shape.w));
          const bool w_inside = next_w >= 0 && next_w < src.shape.w;
          for (int d = 0; d < dst->shape.d; d++) {
            ABSL_ASSIGN_OR_RETURN(
                int next_d, GetNext(attr.starts.d, attr.strides.d, attr.ends.d,
                                    d, src.shape.d));
            const bool d_inside = next_d >= 0 && next_d < src.shape.d;

            const int src_index = src.shape.LinearIndex(
                {next_b, next_h, next_w, next_d, next_ch});
            const int dst_index = dst->shape.LinearIndex({b, h, w, d, c});
            if (c_inside && h_inside && w_inside && d_inside && b_inside) {
              dst->data[dst_index] = src.data[src_index];
            } else {
              dst->data[dst_index] = 0.0f;
            }
          }
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<::ml_drift::Tensor5DFloat32> SliceReference(
    const ::ml_drift::Slice3DAttributes& attr,
    const ::ml_drift::Tensor5DFloat32& input) {
  ml_drift::Tensor5DFloat32 output =
      MakeZeroTensor(CalculateOutputShape(input.shape, attr));
  ABSL_RETURN_IF_ERROR(SliceReference(attr, input, &output));
  return output;
}

void TransposeReference(const ml_drift::TransposeAttributes& attr,
                        const ml_drift::TensorFloat32& src,
                        ml_drift::TensorFloat32* dst) {
  for (int src_b = 0; src_b < src.shape.b; ++src_b) {
    for (int src_h = 0; src_h < src.shape.h; ++src_h) {
      for (int src_w = 0; src_w < src.shape.w; ++src_w) {
        for (int src_c = 0; src_c < src.shape.c; ++src_c) {
          int arr[4] = {src_b, src_h, src_w, src_c};
          int dst_b = arr[attr.perm.b];
          int dst_h = arr[attr.perm.h];
          int dst_w = arr[attr.perm.w];
          int dst_c = arr[attr.perm.c];
          const int src_index =
              src.shape.LinearIndex({src_b, src_h, src_w, src_c});
          const int dst_index =
              dst->shape.LinearIndex({dst_b, dst_h, dst_w, dst_c});
          dst->data[dst_index] = src.data[src_index];
        }
      }
    }
  }
}

ml_drift::TensorFloat32 TransposeReference(
    const ml_drift::TransposeAttributes& attr,
    const ml_drift::TensorFloat32& src) {
  ml_drift::TensorFloat32 dst =
      MakeZeroTensor(CalculateOutputShape(src.shape, attr));
  TransposeReference(attr, src, &dst);
  return dst;
}

void TransposeReference(const ml_drift::Transpose3DAttributes& attr,
                        const ml_drift::Tensor5DFloat32& src,
                        ml_drift::Tensor5DFloat32* dst) {
  for (int src_b = 0; src_b < src.shape.b; ++src_b) {
    for (int src_h = 0; src_h < src.shape.h; ++src_h) {
      for (int src_w = 0; src_w < src.shape.w; ++src_w) {
        for (int src_d = 0; src_d < src.shape.d; ++src_d) {
          for (int src_c = 0; src_c < src.shape.c; ++src_c) {
            int arr[5] = {src_b, src_h, src_w, src_d, src_c};
            int dst_b = arr[attr.perm.b];
            int dst_h = arr[attr.perm.h];
            int dst_w = arr[attr.perm.w];
            int dst_d = arr[attr.perm.d];
            int dst_c = arr[attr.perm.c];
            const int src_index =
                src.shape.LinearIndex({src_b, src_h, src_w, src_d, src_c});
            const int dst_index =
                dst->shape.LinearIndex({dst_b, dst_h, dst_w, dst_d, dst_c});
            dst->data[dst_index] = src.data[src_index];
          }
        }
      }
    }
  }
}

ml_drift::Tensor5DFloat32 TransposeReference(
    const ml_drift::Transpose3DAttributes& attr,
    const ml_drift::Tensor5DFloat32& src) {
  ml_drift::Tensor5DFloat32 dst =
      MakeZeroTensor(CalculateOutputShape(src.shape, attr));
  TransposeReference(attr, src, &dst);
  return dst;
}

void PReLUReference(const ml_drift::PReLUAttributes& attr,
                    const ml_drift::TensorFloat32& src,
                    ml_drift::TensorFloat32* dst) {
  auto linear_alpha = std::get_if<
      ml_drift::Tensor<ml_drift::Linear, ml_drift::DataType::FLOAT32>>(
      &attr.alpha);
  auto full_alpha =
      std::get_if<ml_drift::Tensor<ml_drift::HWC, ml_drift::DataType::FLOAT32>>(
          &attr.alpha);
  if (linear_alpha) {
    ABSL_QCHECK_EQ(linear_alpha->shape.v, dst->shape.c);
  } else {
    ABSL_QCHECK(full_alpha) << "Alpha is missing in prelu";
    ABSL_QCHECK(full_alpha->shape ==
                ml_drift::HWC(dst->shape.h, dst->shape.w, dst->shape.c));
  }
  for (int b = 0; b < dst->shape.b; ++b) {
    for (int c = 0; c < dst->shape.c; ++c) {
      for (int h = 0; h < dst->shape.h; ++h) {
        for (int w = 0; w < dst->shape.w; ++w) {
          const int index = src.shape.LinearIndex({b, h, w, c});
          float value = src.data[index];

          float alpha;
          if (linear_alpha) {
            alpha = linear_alpha->data[c];
          } else {
            alpha = full_alpha->data[full_alpha->shape.LinearIndex({h, w, c})];
          }

          dst->data[index] =
              std::max(0.0f, value) + std::min(0.0f, value) * alpha;
        }
      }
    }
  }
}

ml_drift::TensorFloat32 PReLUReference(const ml_drift::PReLUAttributes& attr,
                                       const ml_drift::TensorFloat32& input) {
  ml_drift::TensorFloat32 output = MakeZeroTensor(input.shape);
  PReLUReference(attr, input, &output);
  return output;
}

void PReLUReference(const ::ml_drift::PReLUAttributes& attr,
                    const ::ml_drift::Tensor5DFloat32& src,
                    ::ml_drift::Tensor5DFloat32* dst) {
  auto linear_alpha = std::get_if<
      ml_drift::Tensor<ml_drift::Linear, ml_drift::DataType::FLOAT32>>(
      &attr.alpha);
  if (linear_alpha) {
    ABSL_QCHECK_EQ(linear_alpha->shape.v, dst->shape.c);
  }
  for (int b = 0; b < dst->shape.b; ++b) {
    for (int c = 0; c < dst->shape.c; ++c) {
      for (int h = 0; h < dst->shape.h; ++h) {
        for (int w = 0; w < dst->shape.w; ++w) {
          for (int d = 0; d < dst->shape.d; ++d) {
            const int index = src.shape.LinearIndex({b, h, w, d, c});
            float value = src.data[index];

            float alpha = 0.0f;
            if (linear_alpha) {
              alpha = linear_alpha->data[c];
            }

            dst->data[index] =
                std::max(0.0f, value) + std::min(0.0f, value) * alpha;
          }
        }
      }
    }
  }
}

::ml_drift::Tensor5DFloat32 PReLUReference(
    const ::ml_drift::PReLUAttributes& attr,
    const ::ml_drift::Tensor5DFloat32& input) {
  ml_drift::Tensor5DFloat32 output = MakeZeroTensor(input.shape);
  PReLUReference(attr, input, &output);
  return output;
}

TensorFloat32 ElementwiseReference(const TensorFloat32& src,
                                   OperationType op_type) {
  TensorFloat32 result;
  result.shape = src.shape;
  result.data.resize(result.shape.DimensionsProduct(), 0.0f);
  for (size_t i = 0; i < result.data.size(); ++i) {
    if (op_type == OperationType::SQUARE) {
      result.data[i] = src.data[i] * src.data[i];
    } else if (op_type == OperationType::SQRT) {
      result.data[i] = sqrt(src.data[i]);
    } else if (op_type == OperationType::RSQRT) {
      result.data[i] = 1.0f / sqrt(src.data[i]);
    }
  }
  return result;
}

TensorFloat32 ElementwiseReference(const TensorFloat32& src,
                                   OperationType op_type, float value) {
  TensorFloat32 result;
  result.shape = src.shape;
  result.data.resize(result.shape.DimensionsProduct(), 0.0f);
  for (size_t i = 0; i < result.data.size(); ++i) {
    if (op_type == OperationType::ADD) {
      result.data[i] = src.data[i] + value;
    } else if (op_type == OperationType::DIV) {
      result.data[i] = src.data[i] / value;
    } else if (op_type == OperationType::MUL) {
      result.data[i] = src.data[i] * value;
    }
  }
  return result;
}

TensorFloat32 ElementwiseReference(const TensorFloat32& src0,
                                   const TensorFloat32& src1,
                                   OperationType op_type) {
  TensorFloat32 result;
  result.shape = src0.shape;
  result.data.resize(result.shape.DimensionsProduct(), 0.0f);
  for (int src_b = 0; src_b < src0.shape.b; ++src_b) {
    for (int src_h = 0; src_h < src0.shape.h; ++src_h) {
      for (int src_w = 0; src_w < src0.shape.w; ++src_w) {
        for (int src_c = 0; src_c < src0.shape.c; ++src_c) {
          const int src0_index =
              src0.shape.LinearIndex({src_b, src_h, src_w, src_c});
          int src1_b = src_b % src1.shape.b;
          int src1_h = src_h % src1.shape.h;
          int src1_w = src_w % src1.shape.w;
          int src1_c = src_c % src1.shape.c;
          const int src1_index =
              src1.shape.LinearIndex({src1_b, src1_h, src1_w, src1_c});
          const float src0_value = src0.data[src0_index];
          const float src1_value = src1.data[src1_index];
          if (op_type == OperationType::ADD) {
            result.data[src0_index] = src0_value + src1_value;
          } else if (op_type == OperationType::SUB) {
            result.data[src0_index] = src0_value - src1_value;
          } else if (op_type == OperationType::MUL) {
            result.data[src0_index] = src0_value * src1_value;
          } else if (op_type == OperationType::DIV) {
            result.data[src0_index] = src0_value / src1_value;
          }
        }
      }
    }
  }
  return result;
}

TensorFloat32 RMSNormalizationReference(const TensorFloat32& src, float eps) {
  TensorFloat32 squares = ElementwiseReference(src, OperationType::SQUARE);
  TensorFloat32 means =
      ReduceReference({Axis::CHANNELS}, OperationType::MEAN, squares);
  TensorFloat32 var = ElementwiseReference(means, OperationType::ADD, eps);
  var = ElementwiseReference(var, OperationType::RSQRT);
  return ElementwiseReference(src, var, OperationType::MUL);
}

TensorFloat32 StatisticalTopKReference(const TensorFloat32& src,
                                       float stddev_multiplier) {
  TensorFloat32 mean =
      ReduceReference({Axis::CHANNELS}, OperationType::MEAN, src);
  TensorFloat32 sq_mean = ElementwiseReference(mean, OperationType::SQUARE);
  TensorFloat32 squares = ElementwiseReference(src, OperationType::SQUARE);
  TensorFloat32 mean_sq =
      ReduceReference({Axis::CHANNELS}, OperationType::MEAN, squares);
  TensorFloat32 var =
      ElementwiseReference(mean_sq, sq_mean, OperationType::SUB);
  TensorFloat32 stddev = ElementwiseReference(var, OperationType::SQRT);
  TensorFloat32 scaled_stddev =
      ElementwiseReference(stddev, OperationType::MUL, stddev_multiplier);
  TensorFloat32 cutoff =
      ElementwiseReference(mean, scaled_stddev, OperationType::ADD);
  TensorFloat32 subtract =
      ElementwiseReference(src, cutoff, OperationType::SUB);
  PReLUAttributes attr;
  attr.alpha = MakeZeroTensor(Linear(src.shape.c));
  return PReLUReference(attr, subtract);
}

TensorFloat32 Winograd3x3ForwardRef(const TensorFloat32& src_tensor,
                                    const Padding2D& padding, int tile_size) {
  Convolution2DAttributes attr;
  attr.padding = padding;
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  auto& attr_weights =
      attr.weights.emplace<ml_drift::Tensor<OHWI, DataType::FLOAT32>>();
  attr_weights.shape = OHWI(src_tensor.shape.c, 3, 3, src_tensor.shape.c);
  const auto conv_output_shape = CalculateOutputShape(src_tensor.shape, attr);

  const int tile_size_inner = tile_size - 2;
  const int tile_size_outer = tile_size;
  const int w_tiles = DivideRoundUp(conv_output_shape.w, tile_size_inner);
  const int h_tiles = DivideRoundUp(conv_output_shape.h, tile_size_inner);
  const BHWC src_transformed_shape{src_tensor.shape.b,
                                   tile_size_outer * tile_size_outer,
                                   w_tiles * h_tiles, src_tensor.shape.c};

  TensorFloat32 result;
  result.shape = BHWC(src_tensor.shape.b, tile_size_outer * tile_size_outer,
                      w_tiles * h_tiles, src_tensor.shape.c);
  result.data.resize(result.shape.DimensionsProduct(), 0.0f);

  auto b_t = BtMatrixForWinograd3x3TileNxN(tile_size_outer);

  for (int b = 0; b < src_tensor.shape.b; ++b) {
    for (int c = 0; c < src_tensor.shape.c; ++c) {
      for (int tile_y = 0; tile_y < h_tiles; ++tile_y) {
        for (int tile_x = 0; tile_x < w_tiles; ++tile_x) {
          // Bt * Src * B
          // 1: temp = Src * B
          std::vector<float> temp(tile_size_outer * tile_size_outer, 0.0f);
          for (int y = 0; y < tile_size_outer; ++y) {
            for (int x = 0; x < tile_size_outer; ++x) {
              float sum = 0.0f;
              for (int k = 0; k < tile_size_outer; ++k) {
                const int src_h =
                    tile_y * tile_size_inner + y - attr.padding.prepended.h;
                const int src_w =
                    tile_x * tile_size_inner + k - attr.padding.prepended.w;
                if (src_h < 0 || src_h >= src_tensor.shape.h) {
                  continue;
                }
                if (src_w < 0 || src_w >= src_tensor.shape.w) {
                  continue;
                }
                const int index =
                    src_tensor.shape.LinearIndex({b, src_h, src_w, c});
                sum += src_tensor.data[index] * b_t[x * tile_size_outer + k];
              }
              temp[y * tile_size_outer + x] = sum;
            }
          }
          // 2: ref = Bt * temp
          for (int y = 0; y < tile_size_outer; ++y) {
            for (int x = 0; x < tile_size_outer; ++x) {
              float sum = 0.0f;
              for (int k = 0; k < tile_size_outer; ++k) {
                sum += b_t[y * tile_size_outer + k] *
                       temp[k * tile_size_outer + x];
              }
              const int index = result.shape.LinearIndex(
                  {b, y * tile_size_outer + x, tile_y * w_tiles + tile_x, c});
              result.data[index] = sum;
            }
          }
        }
      }
    }
  }
  return result;
}

TensorFloat32 Winograd3x3BackwardRef(TensorFloat32 src_tensor,
                                     const BHWC& dst_shape, int tile_size) {
  const int tile_size_inner = tile_size - 2;
  const int tile_size_outer = tile_size;
  const int w_tiles = DivideRoundUp(dst_shape.w, tile_size_inner);
  const int h_tiles = DivideRoundUp(dst_shape.h, tile_size_inner);

  TensorFloat32 result;
  result.shape = dst_shape;
  result.data.resize(result.shape.DimensionsProduct(), 0.0f);

  auto a_t = AtMatrixForWinograd3x3TileNxN(tile_size_outer);

  for (int b = 0; b < src_tensor.shape.b; ++b) {
    for (int c = 0; c < src_tensor.shape.c; ++c) {
      for (int tile_y = 0; tile_y < h_tiles; ++tile_y) {
        for (int tile_x = 0; tile_x < w_tiles; ++tile_x) {
          // At * Src * A
          // 1: temp = Src * A
          std::vector<float> temp(tile_size_outer * tile_size_inner, 0.0f);
          for (int y = 0; y < tile_size_outer; ++y) {
            for (int x = 0; x < tile_size_inner; ++x) {
              float sum = 0.0f;
              for (int i = 0; i < tile_size_outer; ++i) {
                const int index = src_tensor.shape.LinearIndex(
                    {b, y * tile_size_outer + i, tile_y * w_tiles + tile_x, c});
                sum += src_tensor.data[index] * a_t[x * tile_size_outer + i];
              }
              temp[y * tile_size_inner + x] = sum;
            }
          }
          // 2: ref = At * temp
          for (int y = 0; y < tile_size_inner; ++y) {
            for (int x = 0; x < tile_size_inner; ++x) {
              float sum = 0.0f;
              for (int i = 0; i < tile_size_outer; ++i) {
                sum += a_t[y * tile_size_outer + i] *
                       temp[i * tile_size_inner + x];
              }
              if (tile_x * tile_size_inner + x < dst_shape.w &&
                  tile_y * tile_size_inner + y < dst_shape.h) {
                const int index =
                    result.shape.LinearIndex({b, tile_y * tile_size_inner + y,
                                              tile_x * tile_size_inner + x, c});
                result.data[index] = sum;
              }
            }
          }
        }
      }
    }
  }
  return result;
}

Tensor<OHWI, DataType::FLOAT32> MakeWeightsFromInt8(
    const Tensor<OHWI, DataType::INT8>& weights_i8, const float weights_scale,
    const float weights_zero_point) {
  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = weights_i8.shape;
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < weights.shape.i; ++i) {
    for (int o = 0; o < weights.shape.o; ++o) {
      weights.data[weights.shape.LinearIndex({o, 0, 0, i})] =
          (weights_i8.data[weights_i8.shape.LinearIndex({o, 0, 0, i})] -
           weights_zero_point) *
          weights_scale;
    }
  }
  return weights;
}

Tensor<OHWI, DataType::FLOAT32> MakeWeightsFromInt8(
    const Tensor<OHWI, DataType::INT8>& weights_i8,
    const Tensor<OHWI, DataType::FLOAT32>& weights_scale,
    const Tensor<OHWI, DataType::FLOAT32>& weights_zero_point) {
  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = weights_i8.shape;
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  int i_scale_group_size = weights.shape.i / weights_scale.shape.i;
  int o_scale_group_size = weights.shape.o / weights_scale.shape.o;
  for (int h = 0; h < weights.shape.h; ++h) {
    const int scale_h = weights_scale.shape.h == 1 ? 0 : h;
    const int zp_h = weights_zero_point.shape.h == 1 ? 0 : h;
    for (int i = 0; i < weights.shape.i; ++i) {
      const int i_scale_group = i / i_scale_group_size;
      for (int o = 0; o < weights.shape.o; ++o) {
        const int o_scale_group = o / o_scale_group_size;
        weights.data[weights.shape.LinearIndex({o, h, 0, i})] =
            (weights_i8.data[weights_i8.shape.LinearIndex({o, h, 0, i})] -
             weights_zero_point.data[weights_zero_point.shape.LinearIndex(
                 {o_scale_group, scale_h, 0, i_scale_group})]) *
            weights_scale.data[weights_scale.shape.LinearIndex(
                {o_scale_group, zp_h, 0, i_scale_group})];
      }
    }
  }
  return weights;
}

Tensor<OHWI, DataType::FLOAT32> MakeWeightsFromInt8(
    const Tensor<OHWI, DataType::INT8>& weights_i8,
    const Tensor<Linear, DataType::FLOAT32>& weights_scale,
    const Tensor<Linear, DataType::FLOAT32>* weights_zero_point) {
  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = weights_i8.shape;
  weights.data.resize(weights.shape.DimensionsProduct() +
                      XNN_EXTRA_BYTES / sizeof(float));
  for (int i = 0; i < weights.shape.i; ++i) {
    for (int o = 0; o < weights.shape.o; ++o) {
      float zero_point =
          weights_zero_point == nullptr ? 0.0f : weights_zero_point->data[o];
      weights.data[weights.shape.LinearIndex({o, 0, 0, i})] =
          weights_i8.data[weights_i8.shape.LinearIndex({o, 0, 0, i})] *
              weights_scale.data[o] +
          zero_point;
    }
  }
  return weights;
}

std::pair<TensorInt32, Tensor<Linear, DataType::INT32>> GroupsMapReference(
    const TensorInt32& group_ids, int num_groups) {
  TensorInt32 groups_map;
  groups_map.shape = BHWC(1, num_groups, group_ids.shape.w, 2);
  groups_map.data.resize(groups_map.shape.DimensionsProduct(), -1);
  Tensor<Linear, DataType::INT32> groups_sizes;
  groups_sizes.shape = Linear(num_groups);
  groups_sizes.data.resize(groups_sizes.shape.DimensionsProduct(), 0);
  for (int w = 0; w < group_ids.shape.w; ++w) {
    for (int c = 0; c < group_ids.shape.c; ++c) {
      int expert_id = group_ids.data[w * group_ids.shape.c + c];
      int index = expert_id * group_ids.shape.w + groups_sizes.data[expert_id];
      groups_map.data[index * 2 + 0] = c;
      groups_map.data[index * 2 + 1] = w;
      groups_sizes.data[expert_id]++;
    }
  }
  return std::make_pair(groups_map, groups_sizes);
}

std::pair<TensorInt32, Tensor<Linear, DataType::INT32>>
PackedGroupsMapReference(const TensorInt32& groups_map,
                         const Tensor<Linear, DataType::INT32>& groups_sizes) {
  int num_groups = groups_sizes.shape.v;
  Tensor<Linear, DataType::INT32> groups_offsets;
  groups_offsets.shape = Linear(num_groups);
  groups_offsets.data.resize(groups_offsets.shape.DimensionsProduct(), 0);
  for (int i = 1; i < num_groups; ++i) {
    groups_offsets.data[i] =
        groups_offsets.data[i - 1] + groups_sizes.data[i - 1];
  }
  int total_size =
      groups_offsets.data[num_groups - 1] + groups_sizes.data[num_groups - 1];

  TensorInt32 packed_groups_map;
  packed_groups_map.shape = BHWC(1, 1, total_size, 2);
  packed_groups_map.data.resize(packed_groups_map.shape.DimensionsProduct(),
                                -1);

  for (int g = 0; g < num_groups; ++g) {
    int group_offset = groups_offsets.data[g];
    int group_size = groups_sizes.data[g];
    for (int w = 0; w < group_size; ++w) {
      int src_index = g * groups_map.shape.w + w;
      int dst_index = group_offset + w;
      packed_groups_map.data[dst_index * 2 + 0] =
          groups_map.data[src_index * 2 + 0];
      packed_groups_map.data[dst_index * 2 + 1] =
          groups_map.data[src_index * 2 + 1];
    }
  }
  return std::make_pair(packed_groups_map, groups_offsets);
}

TensorFloat32 RemapToReference(const TensorFloat32& src,
                               const TensorInt32& packed_map) {
  TensorFloat32 dst;
  dst.shape = BHWC(1, 1, packed_map.shape.w, src.shape.c);
  dst.data.resize(dst.shape.DimensionsProduct(), -1.0f);
  for (int w = 0; w < dst.shape.w; ++w) {
    const int group_id = std::min(packed_map.data[w * 2 + 0], src.shape.h - 1);
    const int seq_id = packed_map.data[w * 2 + 1];
    for (int c = 0; c < dst.shape.c; ++c) {
      dst.data[dst.shape.LinearIndex({0, 0, w, c})] =
          src.data[src.shape.LinearIndex({0, group_id, seq_id, c})];
    }
  }
  return dst;
}

TensorFloat32 RemapFromReference(const TensorFloat32& src,
                                 const TensorInt32& packed_map,
                                 int num_groups_per_element) {
  TensorFloat32 dst;
  dst.shape = BHWC(1, num_groups_per_element,
                   packed_map.shape.w / num_groups_per_element, src.shape.c);
  dst.data.resize(dst.shape.DimensionsProduct(), -1.0f);
  for (int w = 0; w < src.shape.w; ++w) {
    const int group_id = packed_map.data[w * 2 + 0];
    const int seq_id = packed_map.data[w * 2 + 1];
    for (int c = 0; c < src.shape.c; ++c) {
      dst.data[dst.shape.LinearIndex({0, group_id, seq_id, c})] =
          src.data[src.shape.LinearIndex({0, 0, w, c})];
    }
  }
  return dst;
}

TensorFloat32 ConvolutionWithIds(
    const TensorFloat32& src_tensor,
    const ml_drift::Tensor<OHWI, DataType::FLOAT32>& weights,
    const TensorInt32& ids) {
  TensorFloat32 dst_ref;
  dst_ref.shape = BHWC(ids.shape.c, src_tensor.shape.h, src_tensor.shape.w,
                       weights.shape.o);
  dst_ref.data.resize(dst_ref.shape.DimensionsProduct(), 0.0f);
  for (int h = 0; h < dst_ref.shape.h; ++h) {
    for (int w = 0; w < dst_ref.shape.w; ++w) {
      for (int b = 0; b < dst_ref.shape.b; ++b) {
        int id = ids.data[ids.shape.LinearIndex({0, h, w, b})];
        for (int o = 0; o < dst_ref.shape.c; ++o) {
          float sum = 0.0f;
          for (int i = 0; i < src_tensor.shape.c; ++i) {
            int src_b = src_tensor.shape.b == 1 ? 0 : b;
            float src =
                src_tensor.data[src_tensor.shape.LinearIndex({src_b, h, w, i})];
            float weight =
                weights.data[weights.shape.LinearIndex({o, id, 0, i})];
            sum += src * weight;
          }
          dst_ref.data[dst_ref.shape.LinearIndex({b, h, w, o})] = sum;
        }
      }
    }
  }
  return dst_ref;
}

}  // namespace ml_drift
