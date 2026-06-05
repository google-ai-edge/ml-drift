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

#include "ml_drift/common/task/tensor_desc.h"

#include <cstdint>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ml_drift/common/default/status_matchers.h"
#include "absl/hash/hash_testing.h"
#include "absl/types/span.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/status.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"

using ::testing::FloatNear;
using ::testing::HasSubstr;
using ::testing::Pointwise;
using ::ml_drift::StatusIs;

namespace ml_drift {

namespace {

TEST(TensorDescriptorTest, SupportsAbslHash) {
  TensorDescriptor tensor_default;
  TensorDescriptor tensor_a(DataType::INT8, TensorStorageType::BUFFER,
                            Layout::LINEAR);
  TensorDescriptor tensor_b(DataType::INT16, TensorStorageType::TEXTURE_2D,
                            Layout::SCALAR);

  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(
      {tensor_default, tensor_a, tensor_b}));
}

TEST(TensorDescriptorTest, UploadDataSuccessfully) {
  TensorDescriptor desc(DataType::UINT8, TensorStorageType::BUFFER,
                        Layout::BHWC);
  desc.SetBHWCShape(BHWC(1, 1, 1, 4));
  std::vector<uint8_t> in_data(4, 1);
  desc.UploadData(in_data.data());

  EXPECT_THAT(desc.GetData(), Pointwise(FloatNear(1e-6), {1, 1, 1, 1}));
}

TEST(TensorDescriptorTest, UploadDataWithSpanSuccessfully) {
  // Prepare source data.
  std::vector<uint8_t> source_data(4, 1);
  Tensor<BHWC, DataType::UINT8> src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 4);
  src_tensor.spanned_data = absl::MakeSpan(source_data);

  // Prepare the TensorDescriptor
  TensorDescriptor desc(DataType::UINT8, TensorStorageType::BUFFER,
                        Layout::BHWC);

  // Upload the data using the span.
  desc.UploadData(src_tensor);

  EXPECT_THAT(desc.GetData(), Pointwise(FloatNear(1e-6), {1, 1, 1, 1}));
}

TEST(TensorDescriptorTest, UploadRawDataToHalfSuccessfullyWithFloat) {
  TensorDescriptor desc(DataType::FLOAT16, TensorStorageType::BUFFER,
                        Layout::BHWC);
  desc.SetBHWCShape(BHWC(1, 1, 1, 2));
  std::vector<float> in_data(2, 0.1f);
  desc.UploadDataRaw(absl::MakeConstSpan(in_data));

  EXPECT_EQ(reinterpret_cast<const half*>(desc.GetData().data())[1],
            half(0.1f));
}

TEST(TensorDescriptorTest, UploadDataOutOfBounds) {
  TensorDescriptor desc(DataType::UINT8, TensorStorageType::BUFFER,
                        Layout::BHWC);
  desc.SetBHWCShape(BHWC(1, 1, 1, 4));
  std::vector<float> in_data(4, 1);
  // EXPECT_THROW(desc.UploadData(in_data.data()), std::out_of_range);
}

TEST(TensorDescriptorTest, PerformReadSelectorForHWBuffer) {
  TensorDescriptor desc(DataType::UINT8, TensorStorageType::BUFFER, Layout::HW);
  desc.SetBHWCShape(BHWC(1, 1, 1, 4));
  GpuInfo gpu_info;
  std::string result;
  MLD_ASSERT_OK(desc.PerformSelector(gpu_info, "Read", {"x", "y"},
                                 /*template_args=*/{}, &result));
  EXPECT_EQ(result, "buffer[((y) * width + (x))]");
}

TEST(TensorDescriptorTest, PerformReadSelectorForHWCBuffer) {
  TensorDescriptor desc(DataType::UINT8, TensorStorageType::BUFFER,
                        Layout::HWC);
  desc.SetBHWCShape(BHWC(1, 1, 1, 4));
  GpuInfo gpu_info;
  std::string result;
  MLD_ASSERT_OK(desc.PerformSelector(gpu_info, "Read", {"x", "y", "s"},
                                 /*template_args=*/{}, &result));
  EXPECT_EQ(result, "buffer[(((s) * height + (y)) * width + (x))]");
}

TEST(TensorDescriptorTest, PerformReadSelectorForBHWCBuffer) {
  TensorDescriptor desc(DataType::UINT8, TensorStorageType::BUFFER,
                        Layout::BHWC);
  desc.SetBHWCShape(BHWC(1, 1, 1, 4));
  GpuInfo gpu_info;
  std::string result;
  MLD_ASSERT_OK(desc.PerformSelector(gpu_info, "Read", {"x", "y", "s", "b"},
                                 /*template_args=*/{}, &result));
  EXPECT_EQ(result,
            "buffer[((((s) * height + y) * width + (x)) * batch + (b))]");
}

TEST(TensorDescriptorTest, PerformReadSelectorForHWDCBuffer) {
  TensorDescriptor desc(DataType::UINT8, TensorStorageType::BUFFER,
                        Layout::BHWC);
  desc.SetBHWCShape(BHWC(1, 1, 1, 4));
  GpuInfo gpu_info;
  std::string result;
  MLD_ASSERT_OK(desc.PerformSelector(gpu_info, "Read", {"x", "y", "z", "s"},
                                 /*template_args=*/{}, &result));
  EXPECT_EQ(result,
            "buffer[((((z) * height + y) * width + (x)) * batch + (s))]");
}

TEST(TensorDescriptorTest, PerformReadSelectorForBHWDCBuffer) {
  TensorDescriptor desc(DataType::UINT8, TensorStorageType::BUFFER,
                        Layout::BHWC);
  desc.SetBHWCShape(BHWC(1, 1, 1, 4));
  GpuInfo gpu_info;
  std::string result;
  MLD_ASSERT_OK(desc.PerformSelector(gpu_info, "Read", {"x", "y", "z", "s", "b"},
                                 /*template_args=*/{}, &result));
  EXPECT_EQ(result,
            "buffer[((((z) * height + y) * width + (x)) * batch + (s))]");
}

TEST(TensorDescriptorTest, PerformWriteLinearSelectorForBuffer) {
  TensorDescriptor desc(DataType::FLOAT32, TensorStorageType::BUFFER,
                        Layout::LINEAR);
  desc.SetBHWCShape(BHWC(1, 1, 1, 64));
  GpuInfo gpu_info;
  std::string result;
  MLD_ASSERT_OK(desc.PerformSelector(gpu_info, "WriteLinear",
                                 {/*var_name=*/"val", /*coord[0]=*/"id"},
                                 /*template_args=*/{}, &result));
  EXPECT_EQ(result, "buffer[id] = val");
}

TEST(TensorDescriptorTest, PerformWriteLinearSelectorForTexture2D) {
  TensorDescriptor desc(DataType::FLOAT32, TensorStorageType::TEXTURE_2D,
                        Layout::LINEAR);
  desc.SetBHWCShape(BHWC(1, 1, 1, 16));
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kOpenCl;
  std::string result;
  MLD_ASSERT_OK(desc.PerformSelector(
      gpu_info, "WriteLinear", {/*var_name=*/"value", /*coord[0]=*/"linear_id"},
      /*template_args=*/{}, &result));
  EXPECT_EQ(result, "write_imagef(image2d, (int2)(linear_id, 0), value)");
}

TEST(TensorDescriptorTest, ReadFailedDueToEmptyArguments) {
  TensorDescriptor desc(DataType::UINT8, TensorStorageType::BUFFER,
                        Layout::BHWC);
  desc.SetBHWCShape(BHWC(1, 1, 1, 4));
  GpuInfo gpu_info;
  std::string result;

  EXPECT_THAT(desc.PerformSelector(gpu_info, "Read", {},
                                   /*template_args=*/{}, &result),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected not empty arguments.")));
}

TEST(TensorDescriptorTest, GetAddressFailedDueToMissingBatchCoord) {
  TensorDescriptor desc(DataType::UINT8, TensorStorageType::BUFFER,
                        Layout::BHWC);
  desc.SetBHWCShape(BHWC(1, 1, 1, 4));
  GpuInfo gpu_info;
  std::string result;

  EXPECT_THAT(desc.PerformSelector(gpu_info, "GetAddress", {"x", "y", "z"},
                                   /*template_args=*/{}, &result),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unable to parse bc coord for BATCH axis.")));
}

TEST(TensorDescriptorTest, GetAddressFailedDueToMissingWidthCoord) {
  TensorDescriptor desc(DataType::UINT8, TensorStorageType::BUFFER,
                        Layout::HWC);
  desc.SetBHWCShape(BHWC(1, 1, 1, 4));
  GpuInfo gpu_info;
  std::string result;

  EXPECT_THAT(desc.PerformSelector(gpu_info, "GetAddress", {},
                                   /*template_args=*/{}, &result),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unable to parse xc coord for Width axis.")));
}

TEST(TensorDescriptorTest, GetAddressFailedDueToMissingHeightCoord) {
  TensorDescriptor desc(DataType::UINT8, TensorStorageType::BUFFER,
                        Layout::HWC);
  desc.SetBHWCShape(BHWC(1, 1, 1, 4));
  GpuInfo gpu_info;
  std::string result;

  EXPECT_THAT(desc.PerformSelector(gpu_info, "GetAddress", {"x"},
                                   /*template_args=*/{}, &result),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unable to parse yc coord for Height axis.")));
}

TEST(TensorDescriptorTest, GetAddressFailedDueToMissingChannelsCoord) {
  TensorDescriptor desc(DataType::UINT8, TensorStorageType::BUFFER,
                        Layout::BHWC);
  desc.SetBHWCShape(BHWC(1, 1, 1, 4));
  GpuInfo gpu_info;
  std::string result;

  EXPECT_THAT(
      desc.PerformSelector(gpu_info, "GetAddress", {"x", "y"},
                           /*template_args=*/{}, &result),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Unable to parse sc coord for Channels axis.")));
}

}  // namespace
}  // namespace ml_drift
