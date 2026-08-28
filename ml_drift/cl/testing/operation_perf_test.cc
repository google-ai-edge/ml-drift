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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "ml_drift/cl/opencl_wrapper.h"
#include "ml_drift/cl/testing/perf_util.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"

namespace ml_drift {
namespace cl {

// List of convolutions from real networks:
// 1x3024x4032x4 * 4x3x3x4 -> 1x3024x4032x4
// 1x3024x4032x4 * 4x3x3x4 (stride 2x2) -> 1x1512x2016x4
// 1x1512x2016x8 * 4x3x3x8 -> 1x1512x2016x4

TEST(OpenCLOperationTest, Performance) {
  ABSL_ASSERT_OK(LoadOpenCL());
  ABSL_ASSERT_OK(ConvolutionPerfTest(CalculationsPrecision::F16,
                                BHWC(1, 32, 32, 1024), 1024, HW(1, 1)));
  // ABSL_ASSERT_OK(ConvolutionSf16Wi4BatchedPerfTest(BHWC(1, 8, 1024, 1024), 1024));
  // ABSL_ASSERT_OK(ConvMoEPerfTest(/*seq_size=*/1024, /*src_channels=*/512,
  //                            /*dst_channels=*/512,
  //                            /*num_experts=*/128, /*num_active_experts=*/8,
  //                            DataType::INT4));
  // ABSL_ASSERT_OK(ConvolutionWinogradPerfTest(BHWC(1, 36, 144, 960), 960));
  // ABSL_ASSERT_OK(FullyConnectedPerfTest(CalculationsPrecision::F16,
  //                                       DataType::INT4, BHWC(1, 1, 1, 4096),
  //                                       4096, OHWI(4096, 1, 1, 1)));
  // ABSL_ASSERT_OK(FullyConnectedOptimalWGSize(CalculationsPrecision::F16,
  //                                       DataType::INT4, BHWC(1, 1, 1, 4096),
  //                                       4096, OHWI(4096, 1, 1, 1)));
  // ABSL_ASSERT_OK(FullyConnectedWeightsBatchIdsPerfTest(
  //     CalculationsPrecision::F16, DataType::INT8, BHWC(1, 1, 1, 1024), 1024,
  //     128, 8, OHWI(1024, 128, 1, 1)));
  // ABSL_ASSERT_OK(FullyConnectedInt4Sparse2x4PerfTest(CalculationsPrecision::F16,
  //                                               BHWC(1, 1, 1, 4096), 4096));
  // ABSL_ASSERT_OK(ConvolutionInt8PerfTest(BHWC(1, 1, 2048, 2048), 2048));
  // ABSL_ASSERT_OK(ConvolutionInt8GroupedPerfTest(BHWC(1, 1, 2048, 2048), 2048,64));
  // ABSL_ASSERT_OK(ConvolutionInt4PerfTest(BHWC(1, 1, 4096, 4096), 4096));

  // Softmax cases from real networks
  // ABSL_ASSERT_OK(SoftmaxPerfTest(BHWC{1, 1, 1440, 256000}));
  // ABSL_ASSERT_OK(SoftmaxPerfTest(BHWC{1, 1, 4096, 4096}, /*reduce_only=*/false));

  // ABSL_ASSERT_OK(DepthwiseConvPerfTest(BHWC(1, 16, 16, 768), HW(5, 5)));

  // ABSL_ASSERT_OK(QuantizationPerfTest(BHWC(1, 1, 1024, 1536), DataType::FLOAT16,
  //                                PackedType::kUint8W4C4,
  //                                /*calculate_sum=*/true));
}

}  // namespace cl
}  // namespace ml_drift
