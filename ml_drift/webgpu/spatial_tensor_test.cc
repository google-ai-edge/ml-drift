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

#include "ml_drift/webgpu/spatial_tensor.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/webgpu/environment.h"
#include "ml_drift/webgpu/webgpu_headers.h"

constexpr auto BOOL = ::ml_drift::DataType::BOOL;
constexpr auto FLOAT16 = ::ml_drift::DataType::FLOAT16;
constexpr auto FLOAT32 = ::ml_drift::DataType::FLOAT32;
constexpr auto INT16 = ::ml_drift::DataType::INT16;
constexpr auto INT32 = ::ml_drift::DataType::INT32;
constexpr auto INT8 = ::ml_drift::DataType::INT8;
constexpr auto UINT16 = ::ml_drift::DataType::UINT16;
constexpr auto UINT32 = ::ml_drift::DataType::UINT32;
constexpr auto UINT8 = ::ml_drift::DataType::UINT8;
constexpr auto SINGLE_TEXTURE_2D =
    ::ml_drift::TensorStorageType::SINGLE_TEXTURE_2D;
using ::testing::Eq;
using ::testing::FloatNear;
using ::testing::Pointwise;
using ::testing::ValuesIn;
using ::testing::TestWithParam;

namespace ml_drift {
namespace webgpu {
namespace {

class SpatialTensorTest : public TestWithParam<TensorStorageType> {
 public:
  SpatialTensorTest() : env_(wgpu::BackendType::Vulkan) {}

 protected:
  void SetUp() override {
    MLD_ASSERT_OK(env_.Initialize());
  }

  // Upload and immediately download; essentially an idempotent operation.
  template <typename TensorT>
  TensorT UploadAndDownload(const TensorT& input, TensorDescriptor desc) {
    // Update the descriptor with the input tensor's data.
    desc.UploadData(input);
    // Import the descriptor and materialize the CL tensor.
    SpatialTensor wgpu_tensor;
    ABSL_QCHECK_OK(wgpu_tensor.CreateFromDescriptor(env_.device(), desc));
    // Export the descriptor of the created CL tensor.
    TensorDescriptor wgpu_desc;
    ABSL_QCHECK_OK(wgpu_tensor.ToDescriptor(env_.device(), &wgpu_desc));
    // Download data from the descriptor to the output tensor.
    TensorT output;
    wgpu_desc.DownloadData(&output);
    return output;
  }

  Environment env_;
};

// Generates a reasonable distribution of numbers for testing, where i ranges
// from 0 to i_max.
double GenerateDouble(int i, int i_max, const TensorDescriptor& desc) {
  const double val = static_cast<double>(i) / i_max;    // [ 0.0, 1.0]
  double transformed_val = std::sin(val * 2.0 * M_PI);  // [-1.0, 1.0]
  const DataType dtype = desc.GetDataType();
  if (dtype == FLOAT16 || dtype == FLOAT32) {
    // pass through
  } else if (dtype == INT16) {
    transformed_val *= std::numeric_limits<int16_t>::max();
  } else if (dtype == INT32) {
    transformed_val *= std::numeric_limits<int32_t>::max();
  } else if (dtype == INT8) {
    transformed_val *= std::numeric_limits<int8_t>::max();
  } else if (dtype == UINT16) {
    transformed_val = (transformed_val + 1) / 2;
    transformed_val *= std::numeric_limits<uint16_t>::max();
  } else if (dtype == UINT32) {
    transformed_val = (transformed_val + 1) / 2;
    transformed_val *= std::numeric_limits<uint32_t>::max();
  } else if (dtype == UINT8) {
    transformed_val = (transformed_val + 1) / 2;
    transformed_val *= std::numeric_limits<uint8_t>::max();
  } else if (dtype == BOOL) {
    transformed_val = i % 2;
  } else {
    ABSL_LOG(FATAL) << "Unsupported data type: " << ToString(dtype);
  }
  return transformed_val;
}

template <typename TensorT, typename ShapeT>
TensorT Input(const ShapeT& shape, const TensorDescriptor& desc) {
  TensorT tensor;
  tensor.shape = shape;
  const int n = shape.DimensionsProduct();
  tensor.data.reserve(n);
  for (int i = 0; i < n; ++i) {
    tensor.data.push_back(GenerateDouble(i, n - 1, desc));
  }
  return tensor;
}

TEST_P(SpatialTensorTest, IdempotenceBool) {
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(BOOL, GetParam(), Layout::HWC);
    const BHWC shape(1, 6, 7, 3);
    const auto input = Input<ml_drift::Tensor<BHWC, BOOL>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(BOOL, GetParam(), Layout::HWC);
    const BHWC shape(1, 1, 4, 12);
    const auto input = Input<ml_drift::Tensor<BHWC, BOOL>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(BOOL, GetParam(), Layout::HWC);
    const BHWC shape(1, 6, 1, 7);
    const auto input = Input<ml_drift::Tensor<BHWC, BOOL>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(BOOL, GetParam(), Layout::BHWC);
    const BHWC shape(2, 6, 7, 3);
    const auto input = Input<ml_drift::Tensor<BHWC, BOOL>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(BOOL, GetParam(), Layout::BHWC);
    const BHWC shape(4, 1, 4, 12);
    const auto input = Input<ml_drift::Tensor<BHWC, BOOL>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(BOOL, GetParam(), Layout::BHWC);
    const BHWC shape(7, 6, 1, 7);
    const auto input = Input<ml_drift::Tensor<BHWC, BOOL>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(BOOL, GetParam(), Layout::BHWC);
    const BHWC shape(13, 7, 3, 3);
    const auto input = Input<ml_drift::Tensor<BHWC, BOOL>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(BOOL, GetParam(), Layout::HWDC);
    const BHWDC shape(1, 6, 7, 4, 3);
    const auto input = Input<ml_drift::Tensor<BHWDC, BOOL>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(BOOL, GetParam(), Layout::HWDC);
    const BHWDC shape(1, 1, 4, 3, 12);
    const auto input = Input<ml_drift::Tensor<BHWDC, BOOL>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(BOOL, GetParam(), Layout::HWDC);
    const BHWDC shape(1, 6, 1, 7, 7);
    const auto input = Input<ml_drift::Tensor<BHWDC, BOOL>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(BOOL, GetParam(), Layout::BHWDC);
    const BHWDC shape(2, 6, 7, 1, 3);
    const auto input = Input<ml_drift::Tensor<BHWDC, BOOL>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(BOOL, GetParam(), Layout::BHWDC);
    const BHWDC shape(4, 1, 4, 2, 12);
    const auto input = Input<ml_drift::Tensor<BHWDC, BOOL>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(BOOL, GetParam(), Layout::BHWDC);
    const BHWDC shape(7, 6, 1, 3, 7);
    const auto input = Input<ml_drift::Tensor<BHWDC, BOOL>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(BOOL, GetParam(), Layout::BHWDC);
    const BHWDC shape(13, 7, 3, 4, 3);
    const auto input = Input<ml_drift::Tensor<BHWDC, BOOL>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
}

TEST_P(SpatialTensorTest, IdempotenceFloat16) {
  // 2^-12 should be the maximum error for values in range [-1.0, 1.0].
  constexpr float kEps = 1.0f / (1 << 12);
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(FLOAT16, GetParam(), Layout::HWC);
    const BHWC shape(1, 6, 7, 3);
    const auto input = Input<ml_drift::Tensor<BHWC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT16, GetParam(), Layout::HWC);
    const BHWC shape(1, 1, 4, 12);
    const auto input = Input<ml_drift::Tensor<BHWC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT16, GetParam(), Layout::HWC);
    const BHWC shape(1, 6, 1, 7);
    const auto input = Input<ml_drift::Tensor<BHWC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(FLOAT16, GetParam(), Layout::BHWC);
    const BHWC shape(2, 6, 7, 3);
    const auto input = Input<ml_drift::Tensor<BHWC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT16, GetParam(), Layout::BHWC);
    const BHWC shape(4, 1, 4, 12);
    const auto input = Input<ml_drift::Tensor<BHWC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT16, GetParam(), Layout::BHWC);
    const BHWC shape(7, 6, 1, 7);
    const auto input = Input<ml_drift::Tensor<BHWC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT16, GetParam(), Layout::BHWC);
    const BHWC shape(13, 7, 3, 3);
    const auto input = Input<ml_drift::Tensor<BHWC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(FLOAT16, GetParam(), Layout::HWDC);
    const BHWDC shape(1, 6, 7, 4, 3);
    const auto input = Input<ml_drift::Tensor<BHWDC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT16, GetParam(), Layout::HWDC);
    const BHWDC shape(1, 1, 4, 3, 12);
    const auto input = Input<ml_drift::Tensor<BHWDC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT16, GetParam(), Layout::HWDC);
    const BHWDC shape(1, 6, 1, 7, 7);
    const auto input = Input<ml_drift::Tensor<BHWDC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(FLOAT16, GetParam(), Layout::BHWDC);
    const BHWDC shape(2, 6, 7, 1, 3);
    const auto input = Input<ml_drift::Tensor<BHWDC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT16, GetParam(), Layout::BHWDC);
    const BHWDC shape(4, 1, 4, 2, 12);
    const auto input = Input<ml_drift::Tensor<BHWDC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT16, GetParam(), Layout::BHWDC);
    const BHWDC shape(7, 6, 1, 3, 7);
    const auto input = Input<ml_drift::Tensor<BHWDC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT16, GetParam(), Layout::BHWDC);
    const BHWDC shape(13, 7, 3, 4, 3);
    const auto input = Input<ml_drift::Tensor<BHWDC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
}

TEST_P(SpatialTensorTest, IdempotenceFloat32) {
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(FLOAT32, GetParam(), Layout::HWC);
    const BHWC shape(1, 6, 7, 3);
    const auto input = Input<ml_drift::Tensor<BHWC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT32, GetParam(), Layout::HWC);
    const BHWC shape(1, 1, 4, 12);
    const auto input = Input<ml_drift::Tensor<BHWC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT32, GetParam(), Layout::HWC);
    const BHWC shape(1, 6, 1, 7);
    const auto input = Input<ml_drift::Tensor<BHWC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(FLOAT32, GetParam(), Layout::BHWC);
    const BHWC shape(2, 6, 7, 3);
    const auto input = Input<ml_drift::Tensor<BHWC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT32, GetParam(), Layout::BHWC);
    const BHWC shape(4, 1, 4, 12);
    const auto input = Input<ml_drift::Tensor<BHWC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT32, GetParam(), Layout::BHWC);
    const BHWC shape(7, 6, 1, 7);
    const auto input = Input<ml_drift::Tensor<BHWC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT32, GetParam(), Layout::BHWC);
    const BHWC shape(13, 7, 3, 3);
    const auto input = Input<ml_drift::Tensor<BHWC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(FLOAT32, GetParam(), Layout::HWDC);
    const BHWDC shape(1, 6, 7, 4, 3);
    const auto input = Input<ml_drift::Tensor<BHWDC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT32, GetParam(), Layout::HWDC);
    const BHWDC shape(1, 1, 4, 3, 12);
    const auto input = Input<ml_drift::Tensor<BHWDC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT32, GetParam(), Layout::HWDC);
    const BHWDC shape(1, 6, 1, 7, 7);
    const auto input = Input<ml_drift::Tensor<BHWDC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(FLOAT32, GetParam(), Layout::BHWDC);
    const BHWDC shape(2, 6, 7, 1, 3);
    const auto input = Input<ml_drift::Tensor<BHWDC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT32, GetParam(), Layout::BHWDC);
    const BHWDC shape(4, 1, 4, 2, 12);
    const auto input = Input<ml_drift::Tensor<BHWDC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT32, GetParam(), Layout::BHWDC);
    const BHWDC shape(7, 6, 1, 3, 7);
    const auto input = Input<ml_drift::Tensor<BHWDC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT32, GetParam(), Layout::BHWDC);
    const BHWDC shape(13, 7, 3, 4, 3);
    const auto input = Input<ml_drift::Tensor<BHWDC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
}

TEST_P(SpatialTensorTest, IdempotenceInt16) {
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(INT16, GetParam(), Layout::HWC);
    const BHWC shape(1, 6, 7, 3);
    const auto input = Input<ml_drift::Tensor<BHWC, INT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT16, GetParam(), Layout::HWC);
    const BHWC shape(1, 1, 4, 12);
    const auto input = Input<ml_drift::Tensor<BHWC, INT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT16, GetParam(), Layout::HWC);
    const BHWC shape(1, 6, 1, 7);
    const auto input = Input<ml_drift::Tensor<BHWC, INT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(INT16, GetParam(), Layout::BHWC);
    const BHWC shape(2, 6, 7, 3);
    const auto input = Input<ml_drift::Tensor<BHWC, INT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT16, GetParam(), Layout::BHWC);
    const BHWC shape(4, 1, 4, 12);
    const auto input = Input<ml_drift::Tensor<BHWC, INT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT16, GetParam(), Layout::BHWC);
    const BHWC shape(7, 6, 1, 7);
    const auto input = Input<ml_drift::Tensor<BHWC, INT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT16, GetParam(), Layout::BHWC);
    const BHWC shape(13, 7, 3, 3);
    const auto input = Input<ml_drift::Tensor<BHWC, INT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(INT16, GetParam(), Layout::HWDC);
    const BHWDC shape(1, 6, 7, 4, 3);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT16, GetParam(), Layout::HWDC);
    const BHWDC shape(1, 1, 4, 3, 12);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT16, GetParam(), Layout::HWDC);
    const BHWDC shape(1, 6, 1, 7, 7);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(INT16, GetParam(), Layout::BHWDC);
    const BHWDC shape(2, 6, 7, 1, 3);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT16, GetParam(), Layout::BHWDC);
    const BHWDC shape(4, 1, 4, 2, 12);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT16, GetParam(), Layout::BHWDC);
    const BHWDC shape(7, 6, 1, 3, 7);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT16, GetParam(), Layout::BHWDC);
    const BHWDC shape(13, 7, 3, 4, 3);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
}

TEST_P(SpatialTensorTest, IdempotenceInt32) {
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(INT32, GetParam(), Layout::HWC);
    const BHWC shape(1, 6, 7, 3);
    const auto input = Input<ml_drift::Tensor<BHWC, INT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT32, GetParam(), Layout::HWC);
    const BHWC shape(1, 1, 4, 12);
    const auto input = Input<ml_drift::Tensor<BHWC, INT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT32, GetParam(), Layout::HWC);
    const BHWC shape(1, 6, 1, 7);
    const auto input = Input<ml_drift::Tensor<BHWC, INT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(INT32, GetParam(), Layout::BHWC);
    const BHWC shape(2, 6, 7, 3);
    const auto input = Input<ml_drift::Tensor<BHWC, INT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT32, GetParam(), Layout::BHWC);
    const BHWC shape(4, 1, 4, 12);
    const auto input = Input<ml_drift::Tensor<BHWC, INT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT32, GetParam(), Layout::BHWC);
    const BHWC shape(7, 6, 1, 7);
    const auto input = Input<ml_drift::Tensor<BHWC, INT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT32, GetParam(), Layout::BHWC);
    const BHWC shape(13, 7, 3, 3);
    const auto input = Input<ml_drift::Tensor<BHWC, INT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(INT32, GetParam(), Layout::HWDC);
    const BHWDC shape(1, 6, 7, 4, 3);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT32, GetParam(), Layout::HWDC);
    const BHWDC shape(1, 1, 4, 3, 12);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT32, GetParam(), Layout::HWDC);
    const BHWDC shape(1, 6, 1, 7, 7);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(INT32, GetParam(), Layout::BHWDC);
    const BHWDC shape(2, 6, 7, 1, 3);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT32, GetParam(), Layout::BHWDC);
    const BHWDC shape(4, 1, 4, 2, 12);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT32, GetParam(), Layout::BHWDC);
    const BHWDC shape(7, 6, 1, 3, 7);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT32, GetParam(), Layout::BHWDC);
    const BHWDC shape(13, 7, 3, 4, 3);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
}

TEST_P(SpatialTensorTest, IdempotenceInt8) {
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(INT8, GetParam(), Layout::HWC);
    const BHWC shape(1, 6, 7, 3);
    const auto input = Input<ml_drift::Tensor<BHWC, INT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT8, GetParam(), Layout::HWC);
    const BHWC shape(1, 1, 4, 12);
    const auto input = Input<ml_drift::Tensor<BHWC, INT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT8, GetParam(), Layout::HWC);
    const BHWC shape(1, 6, 1, 7);
    const auto input = Input<ml_drift::Tensor<BHWC, INT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(INT8, GetParam(), Layout::BHWC);
    const BHWC shape(2, 6, 7, 3);
    const auto input = Input<ml_drift::Tensor<BHWC, INT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT8, GetParam(), Layout::BHWC);
    const BHWC shape(4, 1, 4, 12);
    const auto input = Input<ml_drift::Tensor<BHWC, INT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT8, GetParam(), Layout::BHWC);
    const BHWC shape(7, 6, 1, 7);
    const auto input = Input<ml_drift::Tensor<BHWC, INT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT8, GetParam(), Layout::BHWC);
    const BHWC shape(13, 7, 3, 3);
    const auto input = Input<ml_drift::Tensor<BHWC, INT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(INT8, GetParam(), Layout::HWDC);
    const BHWDC shape(1, 6, 7, 4, 3);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT8, GetParam(), Layout::HWDC);
    const BHWDC shape(1, 1, 4, 3, 12);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT8, GetParam(), Layout::HWDC);
    const BHWDC shape(1, 6, 1, 7, 7);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(INT8, GetParam(), Layout::BHWDC);
    const BHWDC shape(2, 6, 7, 1, 3);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT8, GetParam(), Layout::BHWDC);
    const BHWDC shape(4, 1, 4, 2, 12);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT8, GetParam(), Layout::BHWDC);
    const BHWDC shape(7, 6, 1, 3, 7);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT8, GetParam(), Layout::BHWDC);
    const BHWDC shape(13, 7, 3, 4, 3);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
}

TEST_P(SpatialTensorTest, IdempotenceUint16) {
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(UINT16, GetParam(), Layout::HWC);
    const BHWC shape(1, 6, 7, 3);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT16, GetParam(), Layout::HWC);
    const BHWC shape(1, 1, 4, 12);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT16, GetParam(), Layout::HWC);
    const BHWC shape(1, 6, 1, 7);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(UINT16, GetParam(), Layout::BHWC);
    const BHWC shape(2, 6, 7, 3);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT16, GetParam(), Layout::BHWC);
    const BHWC shape(4, 1, 4, 12);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT16, GetParam(), Layout::BHWC);
    const BHWC shape(7, 6, 1, 7);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT16, GetParam(), Layout::BHWC);
    const BHWC shape(13, 7, 3, 3);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(UINT16, GetParam(), Layout::HWDC);
    const BHWDC shape(1, 6, 7, 4, 3);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT16, GetParam(), Layout::HWDC);
    const BHWDC shape(1, 1, 4, 3, 12);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT16, GetParam(), Layout::HWDC);
    const BHWDC shape(1, 6, 1, 7, 7);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(UINT16, GetParam(), Layout::BHWDC);
    const BHWDC shape(2, 6, 7, 1, 3);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT16, GetParam(), Layout::BHWDC);
    const BHWDC shape(4, 1, 4, 2, 12);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT16, GetParam(), Layout::BHWDC);
    const BHWDC shape(7, 6, 1, 3, 7);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT16, GetParam(), Layout::BHWDC);
    const BHWDC shape(13, 7, 3, 4, 3);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
}

TEST_P(SpatialTensorTest, IdempotenceUint32) {
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(UINT32, GetParam(), Layout::HWC);
    const BHWC shape(1, 6, 7, 3);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT32, GetParam(), Layout::HWC);
    const BHWC shape(1, 1, 4, 12);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT32, GetParam(), Layout::HWC);
    const BHWC shape(1, 6, 1, 7);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(UINT32, GetParam(), Layout::BHWC);
    const BHWC shape(2, 6, 7, 3);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT32, GetParam(), Layout::BHWC);
    const BHWC shape(4, 1, 4, 12);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT32, GetParam(), Layout::BHWC);
    const BHWC shape(7, 6, 1, 7);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT32, GetParam(), Layout::BHWC);
    const BHWC shape(13, 7, 3, 3);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(UINT32, GetParam(), Layout::HWDC);
    const BHWDC shape(1, 6, 7, 4, 3);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT32, GetParam(), Layout::HWDC);
    const BHWDC shape(1, 1, 4, 3, 12);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT32, GetParam(), Layout::HWDC);
    const BHWDC shape(1, 6, 1, 7, 7);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(UINT32, GetParam(), Layout::BHWDC);
    const BHWDC shape(2, 6, 7, 1, 3);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT32, GetParam(), Layout::BHWDC);
    const BHWDC shape(4, 1, 4, 2, 12);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT32, GetParam(), Layout::BHWDC);
    const BHWDC shape(7, 6, 1, 3, 7);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT32, GetParam(), Layout::BHWDC);
    const BHWDC shape(13, 7, 3, 4, 3);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
}

TEST_P(SpatialTensorTest, IdempotenceUint8) {
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(UINT8, GetParam(), Layout::HWC);
    const BHWC shape(1, 6, 7, 3);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT8, GetParam(), Layout::HWC);
    const BHWC shape(1, 1, 4, 12);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT8, GetParam(), Layout::HWC);
    const BHWC shape(1, 6, 1, 7);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(UINT8, GetParam(), Layout::BHWC);
    const BHWC shape(2, 6, 7, 3);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT8, GetParam(), Layout::BHWC);
    const BHWC shape(4, 1, 4, 12);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT8, GetParam(), Layout::BHWC);
    const BHWC shape(7, 6, 1, 7);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT8, GetParam(), Layout::BHWC);
    const BHWC shape(13, 7, 3, 3);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(UINT8, GetParam(), Layout::HWDC);
    const BHWDC shape(1, 6, 7, 4, 3);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT8, GetParam(), Layout::HWDC);
    const BHWDC shape(1, 1, 4, 3, 12);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT8, GetParam(), Layout::HWDC);
    const BHWDC shape(1, 6, 1, 7, 7);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(UINT8, GetParam(), Layout::BHWDC);
    const BHWDC shape(2, 6, 7, 1, 3);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT8, GetParam(), Layout::BHWDC);
    const BHWDC shape(4, 1, 4, 2, 12);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT8, GetParam(), Layout::BHWDC);
    const BHWDC shape(7, 6, 1, 3, 7);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT8, GetParam(), Layout::BHWDC);
    const BHWDC shape(13, 7, 3, 4, 3);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
}

INSTANTIATE_TEST_SUITE_P(
    SpatialTensorTests, SpatialTensorTest,
    ValuesIn<TensorStorageType>({
        TensorStorageType::BUFFER,
        TensorStorageType::TEXTURE_2D,
        TensorStorageType::TEXTURE_3D,
        TensorStorageType::TEXTURE_ARRAY,
    }),
    [](const testing::TestParamInfo<SpatialTensorTest::ParamType>& info) {
      return ToString(info.param).substr(strlen("TensorStorageType::"));
    });

// SINGLE_TEXTURE_2D only works with channels < 3.
TEST_F(SpatialTensorTest, IdempotenceSingleTexture2dFloat16) {
  // 2^-12 should be the maximum error for values in range [-1.0, 1.0].
  constexpr float kEps = 1.0f / (1 << 12);
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(FLOAT16, SINGLE_TEXTURE_2D, Layout::HWC);
    const BHWC shape(1, 6, 3, 1);
    const auto input = Input<ml_drift::Tensor<BHWC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT16, SINGLE_TEXTURE_2D, Layout::HWC);
    const BHWC shape(1, 6, 3, 2);
    const auto input = Input<ml_drift::Tensor<BHWC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(FLOAT16, SINGLE_TEXTURE_2D, Layout::BHWC);
    const BHWC shape(7, 6, 3, 1);
    const auto input = Input<ml_drift::Tensor<BHWC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT16, SINGLE_TEXTURE_2D, Layout::BHWC);
    const BHWC shape(3, 6, 3, 2);
    const auto input = Input<ml_drift::Tensor<BHWC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(FLOAT16, SINGLE_TEXTURE_2D, Layout::HWDC);
    const BHWDC shape(1, 6, 14, 7, 1);
    const auto input = Input<ml_drift::Tensor<BHWDC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data,
                Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT16, SINGLE_TEXTURE_2D, Layout::HWDC);
    const BHWDC shape(1, 6, 14, 4, 2);
    const auto input = Input<ml_drift::Tensor<BHWDC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(FLOAT16, SINGLE_TEXTURE_2D, Layout::BHWDC);
    const BHWDC shape(7, 6, 14, 5, 1);
    const auto input = Input<ml_drift::Tensor<BHWDC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT16, SINGLE_TEXTURE_2D, Layout::BHWDC);
    const BHWDC shape(3, 6, 14, 3, 2);
    const auto input = Input<ml_drift::Tensor<BHWDC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
}

TEST_F(SpatialTensorTest, IdempotenceSingleTexture2dFloat32) {
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(FLOAT32, SINGLE_TEXTURE_2D, Layout::HWC);
    const BHWC shape(1, 6, 14, 1);
    const auto input = Input<ml_drift::Tensor<BHWC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT32, SINGLE_TEXTURE_2D, Layout::HWC);
    const BHWC shape(1, 6, 14, 2);
    const auto input = Input<ml_drift::Tensor<BHWC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(FLOAT32, SINGLE_TEXTURE_2D, Layout::BHWC);
    const BHWC shape(7, 6, 14, 1);
    const auto input = Input<ml_drift::Tensor<BHWC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT32, SINGLE_TEXTURE_2D, Layout::BHWC);
    const BHWC shape(3, 6, 14, 2);
    const auto input = Input<ml_drift::Tensor<BHWC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(FLOAT32, SINGLE_TEXTURE_2D, Layout::HWDC);
    const BHWDC shape(1, 6, 14, 7, 1);
    const auto input = Input<ml_drift::Tensor<BHWDC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT32, SINGLE_TEXTURE_2D, Layout::HWDC);
    const BHWDC shape(1, 6, 14, 4, 2);
    const auto input = Input<ml_drift::Tensor<BHWDC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(FLOAT32, SINGLE_TEXTURE_2D, Layout::BHWDC);
    const BHWDC shape(7, 6, 14, 5, 1);
    const auto input = Input<ml_drift::Tensor<BHWDC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(FLOAT32, SINGLE_TEXTURE_2D, Layout::BHWDC);
    const BHWDC shape(3, 6, 14, 3, 2);
    const auto input = Input<ml_drift::Tensor<BHWDC, FLOAT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
}

TEST_F(SpatialTensorTest, IdempotenceSingleTexture2dInt16) {
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(INT16, SINGLE_TEXTURE_2D, Layout::HWC);
    const BHWC shape(1, 6, 14, 1);
    const auto input = Input<ml_drift::Tensor<BHWC, INT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT16, SINGLE_TEXTURE_2D, Layout::HWC);
    const BHWC shape(1, 6, 14, 2);
    const auto input = Input<ml_drift::Tensor<BHWC, INT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(INT16, SINGLE_TEXTURE_2D, Layout::BHWC);
    const BHWC shape(7, 6, 14, 1);
    const auto input = Input<ml_drift::Tensor<BHWC, INT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT16, SINGLE_TEXTURE_2D, Layout::BHWC);
    const BHWC shape(3, 6, 14, 2);
    const auto input = Input<ml_drift::Tensor<BHWC, INT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(INT16, SINGLE_TEXTURE_2D, Layout::HWDC);
    const BHWDC shape(1, 6, 14, 7, 1);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT16, SINGLE_TEXTURE_2D, Layout::HWDC);
    const BHWDC shape(1, 6, 14, 4, 2);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(INT16, SINGLE_TEXTURE_2D, Layout::BHWDC);
    const BHWDC shape(7, 6, 14, 5, 1);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT16, SINGLE_TEXTURE_2D, Layout::BHWDC);
    const BHWDC shape(3, 6, 14, 3, 2);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
}

TEST_F(SpatialTensorTest, IdempotenceSingleTexture2dInt32) {
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(INT32, SINGLE_TEXTURE_2D, Layout::HWC);
    const BHWC shape(1, 6, 14, 1);
    const auto input = Input<ml_drift::Tensor<BHWC, INT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT32, SINGLE_TEXTURE_2D, Layout::HWC);
    const BHWC shape(1, 6, 14, 2);
    const auto input = Input<ml_drift::Tensor<BHWC, INT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(INT32, SINGLE_TEXTURE_2D, Layout::BHWC);
    const BHWC shape(7, 6, 14, 1);
    const auto input = Input<ml_drift::Tensor<BHWC, INT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT32, SINGLE_TEXTURE_2D, Layout::BHWC);
    const BHWC shape(3, 6, 14, 2);
    const auto input = Input<ml_drift::Tensor<BHWC, INT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(INT32, SINGLE_TEXTURE_2D, Layout::HWDC);
    const BHWDC shape(1, 6, 14, 7, 1);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT32, SINGLE_TEXTURE_2D, Layout::HWDC);
    const BHWDC shape(1, 6, 14, 4, 2);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(INT32, SINGLE_TEXTURE_2D, Layout::BHWDC);
    const BHWDC shape(7, 6, 14, 5, 1);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT32, SINGLE_TEXTURE_2D, Layout::BHWDC);
    const BHWDC shape(3, 6, 14, 3, 2);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
}

TEST_F(SpatialTensorTest, IdempotenceSingleTexture2dInt8) {
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(INT8, SINGLE_TEXTURE_2D, Layout::HWC);
    const BHWC shape(1, 6, 14, 1);
    const auto input = Input<ml_drift::Tensor<BHWC, INT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT8, SINGLE_TEXTURE_2D, Layout::HWC);
    const BHWC shape(1, 6, 14, 2);
    const auto input = Input<ml_drift::Tensor<BHWC, INT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(INT8, SINGLE_TEXTURE_2D, Layout::BHWC);
    const BHWC shape(7, 6, 14, 1);
    const auto input = Input<ml_drift::Tensor<BHWC, INT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT8, SINGLE_TEXTURE_2D, Layout::BHWC);
    const BHWC shape(3, 6, 14, 2);
    const auto input = Input<ml_drift::Tensor<BHWC, INT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(INT8, SINGLE_TEXTURE_2D, Layout::HWDC);
    const BHWDC shape(1, 6, 14, 7, 1);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT8, SINGLE_TEXTURE_2D, Layout::HWDC);
    const BHWDC shape(1, 6, 14, 4, 2);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(INT8, SINGLE_TEXTURE_2D, Layout::BHWDC);
    const BHWDC shape(7, 6, 14, 5, 1);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(INT8, SINGLE_TEXTURE_2D, Layout::BHWDC);
    const BHWDC shape(3, 6, 14, 3, 2);
    const auto input = Input<ml_drift::Tensor<BHWDC, INT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
}

TEST_F(SpatialTensorTest, IdempotenceSingleTexture2dUint16) {
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(UINT16, SINGLE_TEXTURE_2D, Layout::HWC);
    const BHWC shape(1, 6, 14, 1);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT16, SINGLE_TEXTURE_2D, Layout::HWC);
    const BHWC shape(1, 6, 14, 2);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(UINT16, SINGLE_TEXTURE_2D, Layout::BHWC);
    const BHWC shape(7, 6, 14, 1);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT16, SINGLE_TEXTURE_2D, Layout::BHWC);
    const BHWC shape(3, 6, 14, 2);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(UINT16, SINGLE_TEXTURE_2D, Layout::HWDC);
    const BHWDC shape(1, 6, 14, 7, 1);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT16, SINGLE_TEXTURE_2D, Layout::HWDC);
    const BHWDC shape(1, 6, 14, 4, 2);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(UINT16, SINGLE_TEXTURE_2D, Layout::BHWDC);
    const BHWDC shape(7, 6, 14, 5, 1);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT16, SINGLE_TEXTURE_2D, Layout::BHWDC);
    const BHWDC shape(3, 6, 14, 3, 2);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
}

TEST_F(SpatialTensorTest, IdempotenceSingleTexture2dUint32) {
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(UINT32, SINGLE_TEXTURE_2D, Layout::HWC);
    const BHWC shape(1, 6, 14, 1);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT32, SINGLE_TEXTURE_2D, Layout::HWC);
    const BHWC shape(1, 6, 14, 2);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(UINT32, SINGLE_TEXTURE_2D, Layout::BHWC);
    const BHWC shape(7, 6, 14, 1);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT32, SINGLE_TEXTURE_2D, Layout::BHWC);
    const BHWC shape(3, 6, 14, 2);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(UINT32, SINGLE_TEXTURE_2D, Layout::HWDC);
    const BHWDC shape(1, 6, 14, 7, 1);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT32, SINGLE_TEXTURE_2D, Layout::HWDC);
    const BHWDC shape(1, 6, 14, 4, 2);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(UINT32, SINGLE_TEXTURE_2D, Layout::BHWDC);
    const BHWDC shape(7, 6, 14, 5, 1);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT32, SINGLE_TEXTURE_2D, Layout::BHWDC);
    const BHWDC shape(3, 6, 14, 3, 2);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
}

TEST_F(SpatialTensorTest, IdempotenceSingleTexture2dUint8) {
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(UINT8, SINGLE_TEXTURE_2D, Layout::HWC);
    const BHWC shape(1, 6, 14, 1);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT8, SINGLE_TEXTURE_2D, Layout::HWC);
    const BHWC shape(1, 6, 14, 2);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(UINT8, SINGLE_TEXTURE_2D, Layout::BHWC);
    const BHWC shape(7, 6, 14, 1);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT8, SINGLE_TEXTURE_2D, Layout::BHWC);
    const BHWC shape(3, 6, 14, 2);
    const auto input = Input<ml_drift::Tensor<BHWC, UINT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(UINT8, SINGLE_TEXTURE_2D, Layout::HWDC);
    const BHWDC shape(1, 6, 14, 7, 1);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT8, SINGLE_TEXTURE_2D, Layout::HWDC);
    const BHWDC shape(1, 6, 14, 4, 2);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(UINT8, SINGLE_TEXTURE_2D, Layout::BHWDC);
    const BHWDC shape(7, 6, 14, 5, 1);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(UINT8, SINGLE_TEXTURE_2D, Layout::BHWDC);
    const BHWDC shape(3, 6, 14, 3, 2);
    const auto input = Input<ml_drift::Tensor<BHWDC, UINT8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
}

TEST_F(SpatialTensorTest, IdempotenceSingleTexture2dBool) {
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(BOOL, SINGLE_TEXTURE_2D, Layout::HWC);
    const BHWC shape(1, 6, 14, 1);
    const auto input = Input<ml_drift::Tensor<BHWC, BOOL>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(BOOL, SINGLE_TEXTURE_2D, Layout::HWC);
    const BHWC shape(1, 6, 14, 2);
    const auto input = Input<ml_drift::Tensor<BHWC, BOOL>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(BOOL, SINGLE_TEXTURE_2D, Layout::BHWC);
    const BHWC shape(7, 6, 14, 1);
    const auto input = Input<ml_drift::Tensor<BHWC, BOOL>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(BOOL, SINGLE_TEXTURE_2D, Layout::BHWC);
    const BHWC shape(3, 6, 14, 2);
    const auto input = Input<ml_drift::Tensor<BHWC, BOOL>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(BOOL, SINGLE_TEXTURE_2D, Layout::HWDC);
    const BHWDC shape(1, 6, 14, 7, 1);
    const auto input = Input<ml_drift::Tensor<BHWDC, BOOL>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(BOOL, SINGLE_TEXTURE_2D, Layout::HWDC);
    const BHWDC shape(1, 6, 14, 4, 2);
    const auto input = Input<ml_drift::Tensor<BHWDC, BOOL>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(BOOL, SINGLE_TEXTURE_2D, Layout::BHWDC);
    const BHWDC shape(7, 6, 14, 5, 1);
    const auto input = Input<ml_drift::Tensor<BHWDC, BOOL>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(BOOL, SINGLE_TEXTURE_2D, Layout::BHWDC);
    const BHWDC shape(3, 6, 14, 3, 2);
    const auto input = Input<ml_drift::Tensor<BHWDC, BOOL>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
}

}  // namespace
}  // namespace webgpu
}  // namespace ml_drift
