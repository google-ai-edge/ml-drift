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

#include "ml_drift/cl/tensor.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "ml_drift/cl/environment.h"
#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"

using ::testing::Eq;
using ::testing::FloatNear;
using ::testing::Pointwise;
using ::testing::ValuesIn;
using ::testing::TestWithParam;

namespace ml_drift {
namespace cl {
namespace {

class TensorTest : public TestWithParam<TensorStorageType> {
 protected:
  void SetUp() override {
    ABSL_ASSERT_OK(LoadOpenCL());
    ABSL_ASSERT_OK(CreateEnvironment(&env_));
  }

  // Upload and immediately download; essentially an idempotent operation.
  template <typename TensorT>
  TensorT UploadAndDownload(const TensorT& input, TensorDescriptor desc) {
    // Update the descriptor with the input tensor's data.
    desc.UploadData(input);
    // Import the descriptor and materialize the CL tensor.
    Tensor cl_tensor;
    ABSL_QCHECK_OK(cl_tensor.CreateFromDescriptor(desc, env_.context()));
    // Export the descriptor of the created CL tensor.
    TensorDescriptor cl_desc;
    ABSL_QCHECK_OK(cl_tensor.ToDescriptor(&cl_desc, env_.queue()));
    // Download data from the descriptor to the output tensor.
    TensorT output;
    cl_desc.DownloadData(&output);
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
  if (dtype == DataType::kFloat16 || dtype == DataType::kFloat32) {
    // pass through
  } else if (dtype == DataType::kInt16) {
    transformed_val *= std::numeric_limits<int16_t>::max();
  } else if (dtype == DataType::kInt32) {
    transformed_val *= std::numeric_limits<int32_t>::max();
  } else if (dtype == DataType::kInt8) {
    transformed_val *= std::numeric_limits<int8_t>::max();
  } else if (dtype == DataType::kUint16) {
    transformed_val = (transformed_val + 1) / 2;
    transformed_val *= std::numeric_limits<uint16_t>::max();
  } else if (dtype == DataType::kUint32) {
    transformed_val = (transformed_val + 1) / 2;
    transformed_val *= std::numeric_limits<uint32_t>::max();
  } else if (dtype == DataType::kUint8) {
    transformed_val = (transformed_val + 1) / 2;
    transformed_val *= std::numeric_limits<uint8_t>::max();
  } else if (dtype == DataType::kBool) {
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

TEST_P(TensorTest, IdempotenceBool) {
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(DataType::kBool, GetParam(), Layout::kHWC);
    const BHWC shape(1, 6, 7, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kBool>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kBool, GetParam(), Layout::kHWC);
    const BHWC shape(1, 1, 4, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kBool>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kBool, GetParam(), Layout::kHWC);
    const BHWC shape(1, 6, 1, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kBool>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(DataType::kBool, GetParam(), Layout::kBHWC);
    const BHWC shape(2, 6, 7, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kBool>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kBool, GetParam(), Layout::kBHWC);
    const BHWC shape(4, 1, 4, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kBool>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kBool, GetParam(), Layout::kBHWC);
    const BHWC shape(7, 6, 1, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kBool>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kBool, GetParam(), Layout::kBHWC);
    const BHWC shape(13, 7, 3, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kBool>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(DataType::kBool, GetParam(), Layout::kHWDC);
    const BHWDC shape(1, 6, 7, 4, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kBool>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kBool, GetParam(), Layout::kHWDC);
    const BHWDC shape(1, 1, 4, 3, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kBool>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kBool, GetParam(), Layout::kHWDC);
    const BHWDC shape(1, 6, 1, 7, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kBool>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(DataType::kBool, GetParam(), Layout::kBHWDC);
    const BHWDC shape(2, 6, 7, 1, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kBool>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kBool, GetParam(), Layout::kBHWDC);
    const BHWDC shape(4, 1, 4, 2, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kBool>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kBool, GetParam(), Layout::kBHWDC);
    const BHWDC shape(7, 6, 1, 3, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kBool>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kBool, GetParam(), Layout::kBHWDC);
    const BHWDC shape(13, 7, 3, 4, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kBool>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
}

TEST_P(TensorTest, IdempotenceFloat16) {
  // 2^-12 should be the maximum error for values in range [-1.0, 1.0].
  constexpr float kEps = 1.0f / (1 << 12);
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(DataType::kFloat16, GetParam(), Layout::kHWC);
    const BHWC shape(1, 6, 7, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kFloat16, GetParam(), Layout::kHWC);
    const BHWC shape(1, 1, 4, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kFloat16, GetParam(), Layout::kHWC);
    const BHWC shape(1, 6, 1, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(DataType::kFloat16, GetParam(), Layout::kBHWC);
    const BHWC shape(2, 6, 7, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kFloat16, GetParam(), Layout::kBHWC);
    const BHWC shape(4, 1, 4, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kFloat16, GetParam(), Layout::kBHWC);
    const BHWC shape(7, 6, 1, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kFloat16, GetParam(), Layout::kBHWC);
    const BHWC shape(13, 7, 3, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(DataType::kFloat16, GetParam(), Layout::kHWDC);
    const BHWDC shape(1, 6, 7, 4, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kFloat16, GetParam(), Layout::kHWDC);
    const BHWDC shape(1, 1, 4, 3, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kFloat16, GetParam(), Layout::kHWDC);
    const BHWDC shape(1, 6, 1, 7, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(DataType::kFloat16, GetParam(), Layout::kBHWDC);
    const BHWDC shape(2, 6, 7, 1, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kFloat16, GetParam(), Layout::kBHWDC);
    const BHWDC shape(4, 1, 4, 2, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kFloat16, GetParam(), Layout::kBHWDC);
    const BHWDC shape(7, 6, 1, 3, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kFloat16, GetParam(), Layout::kBHWDC);
    const BHWDC shape(13, 7, 3, 4, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
}

TEST_P(TensorTest, IdempotenceFloat32) {
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(DataType::kFloat32, GetParam(), Layout::kHWC);
    const BHWC shape(1, 6, 7, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kFloat32, GetParam(), Layout::kHWC);
    const BHWC shape(1, 1, 4, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kFloat32, GetParam(), Layout::kHWC);
    const BHWC shape(1, 6, 1, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(DataType::kFloat32, GetParam(), Layout::kBHWC);
    const BHWC shape(2, 6, 7, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kFloat32, GetParam(), Layout::kBHWC);
    const BHWC shape(4, 1, 4, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kFloat32, GetParam(), Layout::kBHWC);
    const BHWC shape(7, 6, 1, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kFloat32, GetParam(), Layout::kBHWC);
    const BHWC shape(13, 7, 3, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(DataType::kFloat32, GetParam(), Layout::kHWDC);
    const BHWDC shape(1, 6, 7, 4, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kFloat32, GetParam(), Layout::kHWDC);
    const BHWDC shape(1, 1, 4, 3, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kFloat32, GetParam(), Layout::kHWDC);
    const BHWDC shape(1, 6, 1, 7, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(DataType::kFloat32, GetParam(), Layout::kBHWDC);
    const BHWDC shape(2, 6, 7, 1, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kFloat32, GetParam(), Layout::kBHWDC);
    const BHWDC shape(4, 1, 4, 2, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kFloat32, GetParam(), Layout::kBHWDC);
    const BHWDC shape(7, 6, 1, 3, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kFloat32, GetParam(), Layout::kBHWDC);
    const BHWDC shape(13, 7, 3, 4, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
}

TEST_P(TensorTest, IdempotenceInt16) {
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(DataType::kInt16, GetParam(), Layout::kHWC);
    const BHWC shape(1, 6, 7, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kInt16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt16, GetParam(), Layout::kHWC);
    const BHWC shape(1, 1, 4, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kInt16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt16, GetParam(), Layout::kHWC);
    const BHWC shape(1, 6, 1, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kInt16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(DataType::kInt16, GetParam(), Layout::kBHWC);
    const BHWC shape(2, 6, 7, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kInt16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt16, GetParam(), Layout::kBHWC);
    const BHWC shape(4, 1, 4, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kInt16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt16, GetParam(), Layout::kBHWC);
    const BHWC shape(7, 6, 1, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kInt16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt16, GetParam(), Layout::kBHWC);
    const BHWC shape(13, 7, 3, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kInt16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(DataType::kInt16, GetParam(), Layout::kHWDC);
    const BHWDC shape(1, 6, 7, 4, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kInt16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt16, GetParam(), Layout::kHWDC);
    const BHWDC shape(1, 1, 4, 3, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kInt16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt16, GetParam(), Layout::kHWDC);
    const BHWDC shape(1, 6, 1, 7, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kInt16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(DataType::kInt16, GetParam(), Layout::kBHWDC);
    const BHWDC shape(2, 6, 7, 1, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kInt16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt16, GetParam(), Layout::kBHWDC);
    const BHWDC shape(4, 1, 4, 2, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kInt16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt16, GetParam(), Layout::kBHWDC);
    const BHWDC shape(7, 6, 1, 3, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kInt16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt16, GetParam(), Layout::kBHWDC);
    const BHWDC shape(13, 7, 3, 4, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kInt16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
}

TEST_P(TensorTest, IdempotenceInt32) {
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(DataType::kInt32, GetParam(), Layout::kHWC);
    const BHWC shape(1, 6, 7, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kInt32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt32, GetParam(), Layout::kHWC);
    const BHWC shape(1, 1, 4, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kInt32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt32, GetParam(), Layout::kHWC);
    const BHWC shape(1, 6, 1, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kInt32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(DataType::kInt32, GetParam(), Layout::kBHWC);
    const BHWC shape(2, 6, 7, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kInt32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt32, GetParam(), Layout::kBHWC);
    const BHWC shape(4, 1, 4, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kInt32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt32, GetParam(), Layout::kBHWC);
    const BHWC shape(7, 6, 1, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kInt32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt32, GetParam(), Layout::kBHWC);
    const BHWC shape(13, 7, 3, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kInt32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(DataType::kInt32, GetParam(), Layout::kHWDC);
    const BHWDC shape(1, 6, 7, 4, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kInt32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt32, GetParam(), Layout::kHWDC);
    const BHWDC shape(1, 1, 4, 3, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kInt32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt32, GetParam(), Layout::kHWDC);
    const BHWDC shape(1, 6, 1, 7, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kInt32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(DataType::kInt32, GetParam(), Layout::kBHWDC);
    const BHWDC shape(2, 6, 7, 1, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kInt32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt32, GetParam(), Layout::kBHWDC);
    const BHWDC shape(4, 1, 4, 2, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kInt32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt32, GetParam(), Layout::kBHWDC);
    const BHWDC shape(7, 6, 1, 3, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kInt32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt32, GetParam(), Layout::kBHWDC);
    const BHWDC shape(13, 7, 3, 4, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kInt32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
}

TEST_P(TensorTest, IdempotenceInt8) {
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(DataType::kInt8, GetParam(), Layout::kHWC);
    const BHWC shape(1, 6, 7, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kInt8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt8, GetParam(), Layout::kHWC);
    const BHWC shape(1, 1, 4, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kInt8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt8, GetParam(), Layout::kHWC);
    const BHWC shape(1, 6, 1, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kInt8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(DataType::kInt8, GetParam(), Layout::kBHWC);
    const BHWC shape(2, 6, 7, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kInt8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt8, GetParam(), Layout::kBHWC);
    const BHWC shape(4, 1, 4, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kInt8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt8, GetParam(), Layout::kBHWC);
    const BHWC shape(7, 6, 1, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kInt8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt8, GetParam(), Layout::kBHWC);
    const BHWC shape(13, 7, 3, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kInt8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(DataType::kInt8, GetParam(), Layout::kHWDC);
    const BHWDC shape(1, 6, 7, 4, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kInt8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt8, GetParam(), Layout::kHWDC);
    const BHWDC shape(1, 1, 4, 3, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kInt8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt8, GetParam(), Layout::kHWDC);
    const BHWDC shape(1, 6, 1, 7, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kInt8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(DataType::kInt8, GetParam(), Layout::kBHWDC);
    const BHWDC shape(2, 6, 7, 1, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kInt8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt8, GetParam(), Layout::kBHWDC);
    const BHWDC shape(4, 1, 4, 2, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kInt8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt8, GetParam(), Layout::kBHWDC);
    const BHWDC shape(7, 6, 1, 3, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kInt8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kInt8, GetParam(), Layout::kBHWDC);
    const BHWDC shape(13, 7, 3, 4, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kInt8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
}

TEST_P(TensorTest, IdempotenceUint16) {
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(DataType::kUint16, GetParam(), Layout::kHWC);
    const BHWC shape(1, 6, 7, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kUint16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint16, GetParam(), Layout::kHWC);
    const BHWC shape(1, 1, 4, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kUint16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint16, GetParam(), Layout::kHWC);
    const BHWC shape(1, 6, 1, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kUint16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(DataType::kUint16, GetParam(), Layout::kBHWC);
    const BHWC shape(2, 6, 7, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kUint16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint16, GetParam(), Layout::kBHWC);
    const BHWC shape(4, 1, 4, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kUint16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint16, GetParam(), Layout::kBHWC);
    const BHWC shape(7, 6, 1, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kUint16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint16, GetParam(), Layout::kBHWC);
    const BHWC shape(13, 7, 3, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kUint16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(DataType::kUint16, GetParam(), Layout::kHWDC);
    const BHWDC shape(1, 6, 7, 4, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kUint16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint16, GetParam(), Layout::kHWDC);
    const BHWDC shape(1, 1, 4, 3, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kUint16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint16, GetParam(), Layout::kHWDC);
    const BHWDC shape(1, 6, 1, 7, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kUint16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(DataType::kUint16, GetParam(), Layout::kBHWDC);
    const BHWDC shape(2, 6, 7, 1, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kUint16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint16, GetParam(), Layout::kBHWDC);
    const BHWDC shape(4, 1, 4, 2, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kUint16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint16, GetParam(), Layout::kBHWDC);
    const BHWDC shape(7, 6, 1, 3, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kUint16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint16, GetParam(), Layout::kBHWDC);
    const BHWDC shape(13, 7, 3, 4, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kUint16>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
}

TEST_P(TensorTest, IdempotenceUint32) {
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(DataType::kUint32, GetParam(), Layout::kHWC);
    const BHWC shape(1, 6, 7, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kUint32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint32, GetParam(), Layout::kHWC);
    const BHWC shape(1, 1, 4, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kUint32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint32, GetParam(), Layout::kHWC);
    const BHWC shape(1, 6, 1, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kUint32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(DataType::kUint32, GetParam(), Layout::kBHWC);
    const BHWC shape(2, 6, 7, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kUint32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint32, GetParam(), Layout::kBHWC);
    const BHWC shape(4, 1, 4, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kUint32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint32, GetParam(), Layout::kBHWC);
    const BHWC shape(7, 6, 1, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kUint32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint32, GetParam(), Layout::kBHWC);
    const BHWC shape(13, 7, 3, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kUint32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(DataType::kUint32, GetParam(), Layout::kHWDC);
    const BHWDC shape(1, 6, 7, 4, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kUint32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint32, GetParam(), Layout::kHWDC);
    const BHWDC shape(1, 1, 4, 3, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kUint32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint32, GetParam(), Layout::kHWDC);
    const BHWDC shape(1, 6, 1, 7, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kUint32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(DataType::kUint32, GetParam(), Layout::kBHWDC);
    const BHWDC shape(2, 6, 7, 1, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kUint32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint32, GetParam(), Layout::kBHWDC);
    const BHWDC shape(4, 1, 4, 2, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kUint32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint32, GetParam(), Layout::kBHWDC);
    const BHWDC shape(7, 6, 1, 3, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kUint32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint32, GetParam(), Layout::kBHWDC);
    const BHWDC shape(13, 7, 3, 4, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kUint32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
}

TEST_P(TensorTest, IdempotenceUint8) {
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(DataType::kUint8, GetParam(), Layout::kHWC);
    const BHWC shape(1, 6, 7, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kUint8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint8, GetParam(), Layout::kHWC);
    const BHWC shape(1, 1, 4, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kUint8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint8, GetParam(), Layout::kHWC);
    const BHWC shape(1, 6, 1, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kUint8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(DataType::kUint8, GetParam(), Layout::kBHWC);
    const BHWC shape(2, 6, 7, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kUint8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint8, GetParam(), Layout::kBHWC);
    const BHWC shape(4, 1, 4, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kUint8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint8, GetParam(), Layout::kBHWC);
    const BHWC shape(7, 6, 1, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kUint8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint8, GetParam(), Layout::kBHWC);
    const BHWC shape(13, 7, 3, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kUint8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(DataType::kUint8, GetParam(), Layout::kHWDC);
    const BHWDC shape(1, 6, 7, 4, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kUint8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint8, GetParam(), Layout::kHWDC);
    const BHWDC shape(1, 1, 4, 3, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kUint8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint8, GetParam(), Layout::kHWDC);
    const BHWDC shape(1, 6, 1, 7, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kUint8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(DataType::kUint8, GetParam(), Layout::kBHWDC);
    const BHWDC shape(2, 6, 7, 1, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kUint8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint8, GetParam(), Layout::kBHWDC);
    const BHWDC shape(4, 1, 4, 2, 12);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kUint8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint8, GetParam(), Layout::kBHWDC);
    const BHWDC shape(7, 6, 1, 3, 7);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kUint8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kUint8, GetParam(), Layout::kBHWDC);
    const BHWDC shape(13, 7, 3, 4, 3);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kUint8>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
}

INSTANTIATE_TEST_SUITE_P(
    TensorTests, TensorTest,
    ValuesIn<TensorStorageType>({
        TensorStorageType::kBuffer,
        TensorStorageType::kImageBuffer,
        TensorStorageType::kTexture2D,
        TensorStorageType::kTexture3D,
        TensorStorageType::kTextureArray,
    }),
    [](const testing::TestParamInfo<TensorTest::ParamType>& info) {
      return ToString(info.param).substr(strlen("TensorStorageType::"));
    });

// SINGLE_TEXTURE_2D only works with DataType::FLOAT16 & DataType::FLOAT32 and
// channels < 3.
TEST_F(TensorTest, IdempotenceSingleTexture2dFloat16) {
  // 2^-12 should be the maximum error for values in range [-1.0, 1.0].
  constexpr float kEps = 1.0f / (1 << 12);
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(
        DataType::kFloat16, TensorStorageType::kSingleTexture2D, Layout::kHWC);
    const BHWC shape(1, 6, 3, 1);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(
        DataType::kFloat16, TensorStorageType::kSingleTexture2D, Layout::kHWC);
    const BHWC shape(1, 6, 3, 2);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(
        DataType::kFloat16, TensorStorageType::kSingleTexture2D, Layout::kBHWC);
    const BHWC shape(7, 6, 3, 1);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(
        DataType::kFloat16, TensorStorageType::kSingleTexture2D, Layout::kBHWC);
    const BHWC shape(3, 6, 3, 2);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(
        DataType::kFloat16, TensorStorageType::kSingleTexture2D, Layout::kHWDC);
    const BHWDC shape(1, 6, 14, 7, 1);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data,
                Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(
        DataType::kFloat16, TensorStorageType::kSingleTexture2D, Layout::kHWDC);
    const BHWDC shape(1, 6, 14, 4, 2);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(DataType::kFloat16,
                                TensorStorageType::kSingleTexture2D,
                                Layout::kBHWDC);
    const BHWDC shape(7, 6, 14, 5, 1);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kFloat16,
                                TensorStorageType::kSingleTexture2D,
                                Layout::kBHWDC);
    const BHWDC shape(3, 6, 14, 3, 2);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(FloatNear(kEps), input.data));
  }
}

TEST_F(TensorTest, IdempotenceSingleTexture2dFloat32) {
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(
        DataType::kFloat32, TensorStorageType::kSingleTexture2D, Layout::kHWC);
    const BHWC shape(1, 6, 14, 1);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(
        DataType::kFloat32, TensorStorageType::kSingleTexture2D, Layout::kHWC);
    const BHWC shape(1, 6, 14, 2);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(
        DataType::kFloat32, TensorStorageType::kSingleTexture2D, Layout::kBHWC);
    const BHWC shape(7, 6, 14, 1);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(
        DataType::kFloat32, TensorStorageType::kSingleTexture2D, Layout::kBHWC);
    const BHWC shape(3, 6, 14, 2);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(
        DataType::kFloat32, TensorStorageType::kSingleTexture2D, Layout::kHWDC);
    const BHWDC shape(1, 6, 14, 7, 1);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(
        DataType::kFloat32, TensorStorageType::kSingleTexture2D, Layout::kHWDC);
    const BHWDC shape(1, 6, 14, 4, 2);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(DataType::kFloat32,
                                TensorStorageType::kSingleTexture2D,
                                Layout::kBHWDC);
    const BHWDC shape(7, 6, 14, 5, 1);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kFloat32,
                                TensorStorageType::kSingleTexture2D,
                                Layout::kBHWDC);
    const BHWDC shape(3, 6, 14, 3, 2);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
}

TEST_F(TensorTest, IdempotenceBufferFloat32) {
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(DataType::kFloat32, TensorStorageType::kBuffer,
                                Layout::kHWC,
                                TensorDescriptor::PhysicalLayout1D::kDHWBCC4);
    const BHWC shape(1, 6, 14, 1);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kFloat32, TensorStorageType::kBuffer,
                                Layout::kHWC,
                                TensorDescriptor::PhysicalLayout1D::kDHWBCC4);
    const BHWC shape(1, 6, 14, 2);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(DataType::kFloat32, TensorStorageType::kBuffer,
                                Layout::kBHWC,
                                TensorDescriptor::PhysicalLayout1D::kDHWBCC4);
    const BHWC shape(7, 6, 14, 1);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kFloat32, TensorStorageType::kBuffer,
                                Layout::kBHWC,
                                TensorDescriptor::PhysicalLayout1D::kDHWBCC4);
    const BHWC shape(3, 6, 14, 2);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(DataType::kFloat32, TensorStorageType::kBuffer,
                                Layout::kHWDC,
                                TensorDescriptor::PhysicalLayout1D::kDHWBCC4);
    const BHWDC shape(1, 6, 14, 7, 1);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kFloat32, TensorStorageType::kBuffer,
                                Layout::kHWDC,
                                TensorDescriptor::PhysicalLayout1D::kDHWBCC4);
    const BHWDC shape(1, 6, 14, 4, 2);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(DataType::kFloat32, TensorStorageType::kBuffer,
                                Layout::kBHWDC,
                                TensorDescriptor::PhysicalLayout1D::kDHWBCC4);
    const BHWDC shape(7, 6, 14, 5, 1);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(DataType::kFloat32, TensorStorageType::kBuffer,
                                Layout::kBHWDC,
                                TensorDescriptor::PhysicalLayout1D::kDHWBCC4);
    const BHWDC shape(3, 6, 14, 3, 2);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
}

TEST_F(TensorTest, IdempotenceTexture2dFloat32) {
  // 4D test with (1, h, w, c)
  {
    const TensorDescriptor desc(
        DataType::kFloat32, TensorStorageType::kTexture2D, Layout::kHWC,
        TensorDescriptor::PhysicalLayout2D::kXisCC4YisDHWB);
    const BHWC shape(1, 6, 14, 1);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(
        DataType::kFloat32, TensorStorageType::kTexture2D, Layout::kHWC,
        TensorDescriptor::PhysicalLayout2D::kXisCC4YisDHWB);
    const BHWC shape(1, 6, 14, 2);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 4D test with (b, h, w, c)
  {
    const TensorDescriptor desc(
        DataType::kFloat32, TensorStorageType::kTexture2D, Layout::kBHWC,
        TensorDescriptor::PhysicalLayout2D::kXisCC4YisDHWB);
    const BHWC shape(7, 6, 14, 1);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(
        DataType::kFloat32, TensorStorageType::kTexture2D, Layout::kBHWC,
        TensorDescriptor::PhysicalLayout2D::kXisCC4YisDHWB);
    const BHWC shape(3, 6, 14, 2);
    const auto input =
        Input<ml_drift::Tensor<BHWC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (1, h, w, d, c)
  {
    const TensorDescriptor desc(
        DataType::kFloat32, TensorStorageType::kTexture2D, Layout::kHWDC,
        TensorDescriptor::PhysicalLayout2D::kXisCC4YisDHWB);
    const BHWDC shape(1, 6, 14, 7, 1);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(
        DataType::kFloat32, TensorStorageType::kTexture2D, Layout::kHWDC,
        TensorDescriptor::PhysicalLayout2D::kXisCC4YisDHWB);
    const BHWDC shape(1, 6, 14, 4, 2);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  // 5D tests with (b, h, w, d, c)
  {
    const TensorDescriptor desc(
        DataType::kFloat32, TensorStorageType::kTexture2D, Layout::kBHWDC,
        TensorDescriptor::PhysicalLayout2D::kXisCC4YisDHWB);
    const BHWDC shape(7, 6, 14, 5, 1);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
  {
    const TensorDescriptor desc(
        DataType::kFloat32, TensorStorageType::kTexture2D, Layout::kBHWDC,
        TensorDescriptor::PhysicalLayout2D::kXisCC4YisDHWB);
    const BHWDC shape(3, 6, 14, 3, 2);
    const auto input =
        Input<ml_drift::Tensor<BHWDC, DataType::kFloat32>>(shape, desc);
    const auto output = UploadAndDownload(input, desc);
    EXPECT_THAT(output.data, Pointwise(Eq(), input.data));
  }
}

}  // namespace
}  // namespace cl
}  // namespace ml_drift
