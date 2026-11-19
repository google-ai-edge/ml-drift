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

#include "ml_drift/common/transformations/make_padding.h"

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/status.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/model_transformer.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/tensor.h"

namespace ml_drift {
namespace {

TEST(MakePadding, Smoke) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  input->tensor.shape = BHWC(1, 2, 3, 5);

  auto concat_node = graph.NewNode();
  graph.AddConsumer(concat_node->id, input->id);
  concat_node->operation.type = ToString(OperationType::kConcat);
  ConcatAttributes attr;
  attr.axis = Axis::kHeight;
  concat_node->operation.attributes = attr;

  Value* output = nullptr;
  ABSL_ASSERT_OK(AddOutput(&graph, concat_node, &output));
  output->tensor.shape = BHWC(1, 7, 3, 5);

  auto const_node = graph.NewNode();
  const_node->operation.type = ToString(OperationType::kConstant);
  ConstTensorAttributes const_attr;
  TensorFloat32 const_data;
  const_data.shape = BHWC(1, 5, 3, 5);
  const_data.data = std::vector<float>(const_data.shape.DimensionsProduct(), 0);
  const_attr.tensor = std::move(const_data);
  const_node->operation.attributes = const_attr;

  Value* const_link = nullptr;
  ABSL_ASSERT_OK(ConnectTwoNodes(&graph, const_node, concat_node, &const_link));
  const_link->tensor.shape = BHWC(1, 5, 3, 5);

  ASSERT_EQ(2, graph.nodes().size());

  auto transformation = NewMakePaddingFromConcat();
  ModelTransformer transformer(&graph);
  transformer.Apply("make_padding", transformation.get());

  ASSERT_EQ(1, graph.nodes().size());
  ASSERT_EQ(2, graph.values().size());
  auto pad_node = graph.nodes()[0];
  ASSERT_EQ(ToString(OperationType::kPad), pad_node->operation.type);
  auto pad_attr = std::any_cast<PadAttributes>(pad_node->operation.attributes);
  EXPECT_EQ(BHWC(0, 0, 0, 0), pad_attr.prepended);
  EXPECT_EQ(BHWC(0, 5, 0, 0), pad_attr.appended);
}

}  // namespace
}  // namespace ml_drift
