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

#include "ml_drift/common/task/serialization_base.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "flatbuffers/buffer.h"
#include "flatbuffers/flatbuffer_builder.h"
#include "flatbuffers/string.h"
#include "ml_drift/common/access_type.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/arguments.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/serialization_base_generated.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/types.h"

namespace ml_drift {

namespace {
data::AccessType ToFB(AccessType type) {
  switch (type) {
    case AccessType::kRead:
      return data::AccessType::READ;
    case AccessType::kWrite:
      return data::AccessType::WRITE;
    case AccessType::kReadWrite:
      return data::AccessType::READ_WRITE;
    default:
      return data::AccessType::READ_WRITE;
  }
}

data::DataType ToFB(DataType type) {
  switch (type) {
    case DataType::kBool:
      return data::DataType::BOOL;
    case DataType::kFloat16:
      return data::DataType::FLOAT16;
    case DataType::kFloat32:
      return data::DataType::FLOAT32;
    case DataType::kFloat64:
      return data::DataType::FLOAT64;
    case DataType::kBfloat16:
      return data::DataType::BFLOAT16;
    case DataType::kUint8:
      return data::DataType::UINT8;
    case DataType::kInt8:
      return data::DataType::INT8;
    case DataType::kUint16:
      return data::DataType::UINT16;
    case DataType::kInt16:
      return data::DataType::INT16;
    case DataType::kUint32:
      return data::DataType::UINT32;
    case DataType::kInt32:
      return data::DataType::INT32;
    case DataType::kUint64:
      return data::DataType::UINT64;
    case DataType::kInt64:
      return data::DataType::INT64;
    case DataType::kInt4:
      return data::DataType::INT4;
    case DataType::kUint4:
      return data::DataType::UINT4;
    case DataType::kInt3:
      return data::DataType::INT3;
    case DataType::kUint3:
      return data::DataType::UINT3;
    case DataType::kInt2:
      return data::DataType::INT2;
    case DataType::kUint2:
      return data::DataType::UINT2;
    case DataType::kInt1:
      return data::DataType::INT1;
    case DataType::kUint1:
      return data::DataType::UINT1;
    case DataType::kUnknown:
      return data::DataType::UNKNOWN;
  }
}

data::MemoryType ToFB(MemoryType type) {
  switch (type) {
    case MemoryType::kConstant:
      return data::MemoryType::CONSTANT;
    case MemoryType::kGlobal:
      return data::MemoryType::GLOBAL;
    case MemoryType::kLocal:
      return data::MemoryType::LOCAL;
  }
}

data::TensorStorageType ToFB(TensorStorageType type) {
  switch (type) {
    case TensorStorageType::kBuffer:
      return data::TensorStorageType::BUFFER;
    case TensorStorageType::kImageBuffer:
      return data::TensorStorageType::IMAGE_BUFFER;
    case TensorStorageType::kTexture2D:
      return data::TensorStorageType::TEXTURE_2D;
    case TensorStorageType::kTextureArray:
      return data::TensorStorageType::TEXTURE_ARRAY;
    case TensorStorageType::kTexture3D:
      return data::TensorStorageType::TEXTURE_3D;
    case TensorStorageType::kSingleTexture2D:
      return data::TensorStorageType::SINGLE_TEXTURE_2D;
    case TensorStorageType::kUnknown:
      return data::TensorStorageType::UNKNOWN;
  }
}

data::Layout ToFB(Layout type) {
  switch (type) {
    case Layout::kHWC:
      return data::Layout::HWC;
    case Layout::kBHWC:
      return data::Layout::BHWC;
    case Layout::kHWDC:
      return data::Layout::HWDC;
    case Layout::kBHWDC:
      return data::Layout::BHWDC;
    case Layout::kLinear:
      return data::Layout::LINEAR;
    case Layout::kHW:
      return data::Layout::HW;
    default:
      return data::Layout::UNKNOWN;
  }
}

AccessType ToEnum(data::AccessType type) {
  switch (type) {
    case data::AccessType::READ:
      return AccessType::kRead;
    case data::AccessType::WRITE:
      return AccessType::kWrite;
    case data::AccessType::READ_WRITE:
      return AccessType::kReadWrite;
  }
}

MemoryType ToEnum(data::MemoryType type) {
  switch (type) {
    case data::MemoryType::CONSTANT:
      return MemoryType::kConstant;
    case data::MemoryType::GLOBAL:
      return MemoryType::kGlobal;
    case data::MemoryType::LOCAL:
      return MemoryType::kLocal;
  }
}

data::TensorToGrid ToFB(TensorToGrid type) {
  switch (type) {
    case TensorToGrid::kCustom:
      return data::TensorToGrid::CUSTOM;
    case TensorToGrid::kWBToX_HDToY_SToZ:
      return data::TensorToGrid::WB_TO_X_HD_TO_Y_S_TO_Z;
    case TensorToGrid::kWBToX_HDToY_ZIs1:
      return data::TensorToGrid::WB_TO_X_HD_TO_Y_Z_IS_1;
    case TensorToGrid::kWBToX_HToY_DToZ:
      return data::TensorToGrid::WB_TO_X_H_TO_Y_D_TO_Z;
    case TensorToGrid::kBToX_YIs1_ZIs1:
      return data::TensorToGrid::B_TO_X_Y_IS_1_Z_IS_1;
  }
}

TensorToGrid ToEnum(data::TensorToGrid type) {
  switch (type) {
    case data::TensorToGrid::CUSTOM:
      return TensorToGrid::kCustom;
    case data::TensorToGrid::WB_TO_X_HD_TO_Y_S_TO_Z:
      return TensorToGrid::kWBToX_HDToY_SToZ;
    case data::TensorToGrid::WB_TO_X_HD_TO_Y_Z_IS_1:
      return TensorToGrid::kWBToX_HDToY_ZIs1;
    case data::TensorToGrid::WB_TO_X_H_TO_Y_D_TO_Z:
      return TensorToGrid::kWBToX_HToY_DToZ;
    case data::TensorToGrid::B_TO_X_Y_IS_1_Z_IS_1:
      return TensorToGrid::kBToX_YIs1_ZIs1;
  }
}

data::PhysicalLayout1D ToFB(TensorDescriptor::PhysicalLayout1D type) {
  switch (type) {
    case TensorDescriptor::PhysicalLayout1D::kDCHWBC4:
      return data::PhysicalLayout1D::DCHWBC4;
    case TensorDescriptor::PhysicalLayout1D::kDHWBCC4:
      return data::PhysicalLayout1D::DHWBCC4;
    default:
      return data::PhysicalLayout1D::UNKNOWN;
  }
}

TensorDescriptor::PhysicalLayout1D ToEnum(data::PhysicalLayout1D type) {
  switch (type) {
    case data::PhysicalLayout1D::DCHWBC4:
      return TensorDescriptor::PhysicalLayout1D::kDCHWBC4;
    case data::PhysicalLayout1D::DHWBCC4:
      return TensorDescriptor::PhysicalLayout1D::kDHWBCC4;
    default:
      return TensorDescriptor::PhysicalLayout1D::kUnknown;
  }
}

}  // namespace

flatbuffers::Offset<data::Int2> Encode(
    const int2& v, flatbuffers::FlatBufferBuilder* builder) {
  data::Int2Builder int2_builder(*builder);
  int2_builder.add_x(v.x);
  int2_builder.add_y(v.y);
  return int2_builder.Finish();
}

flatbuffers::Offset<data::Int3> Encode(
    const int3& v, flatbuffers::FlatBufferBuilder* builder) {
  data::Int3Builder int3_builder(*builder);
  int3_builder.add_x(v.x);
  int3_builder.add_y(v.y);
  int3_builder.add_z(v.z);
  return int3_builder.Finish();
}

flatbuffers::Offset<data::GPUObjectDescriptor> Encode(
    const GPUObjectDescriptor& desc, flatbuffers::FlatBufferBuilder* builder) {
  std::vector<flatbuffers::Offset<data::StateVariable>> state_vars_fb;
  for (auto& v0 : desc.state_vars_) {
    auto key_fb = builder->CreateString(v0.first);
    auto value_fb = builder->CreateString(v0.second);
    data::StateVariableBuilder state_builder(*builder);
    state_builder.add_key(key_fb);
    state_builder.add_value(value_fb);
    state_vars_fb.push_back(state_builder.Finish());
  }
  auto state_vars_fb_vec = builder->CreateVector(state_vars_fb);
  data::GPUObjectDescriptorBuilder obj_builder(*builder);
  obj_builder.add_state_vars(state_vars_fb_vec);
  obj_builder.add_access_type(ToFB(desc.access_type_));
  return obj_builder.Finish();
}

absl::Status Decode(const data::GPUObjectDescriptor* fb_obj,
                    GPUObjectDescriptor* obj) {
  if (!fb_obj) {
    return absl::InvalidArgumentError("GPUObjectDescriptor is null.");
  }
  obj->access_type_ = ToEnum(fb_obj->access_type());
  if (fb_obj->state_vars()) {
    for (auto state_fb : *fb_obj->state_vars()) {
      if (!state_fb) {
        return absl::InvalidArgumentError("StateVariable is null.");
      }
      if (!state_fb->key() || !state_fb->value()) {
        return absl::InvalidArgumentError(
            "StateVariable key or value is null.");
      }
      std::string key(state_fb->key()->c_str(), state_fb->key()->size());
      std::string value(state_fb->value()->c_str(), state_fb->value()->size());
      obj->state_vars_[key] = value;
    }
  }
  return absl::OkStatus();
}

flatbuffers::Offset<data::BufferDescriptor> Encode(
    const BufferDescriptor& desc, flatbuffers::FlatBufferBuilder* builder) {
  auto obj_fb =
      Encode(*static_cast<const GPUObjectDescriptor*>(&desc), builder);

  std::vector<flatbuffers::Offset<flatbuffers::String>> attributes_fb;
  attributes_fb.reserve(desc.attributes.size());
  for (auto& attr : desc.attributes) {
    attributes_fb.push_back(builder->CreateString(attr));
  }
  auto attributes_fb_vec = builder->CreateVector(attributes_fb);
  auto data_fb = builder->CreateVector(desc.data);
  data::BufferDescriptorBuilder buf_builder(*builder);
  buf_builder.add_base_obj(obj_fb);
  buf_builder.add_element_type(ToFB(desc.element_type));
  buf_builder.add_element_size(desc.element_size);
  buf_builder.add_memory_type(ToFB(desc.memory_type));
  buf_builder.add_attributes(attributes_fb_vec);
  buf_builder.add_size(desc.size);
  buf_builder.add_data(data_fb);
  return buf_builder.Finish();
}

absl::Status Decode(const data::BufferDescriptor* fb_desc,
                    BufferDescriptor* desc) {
  if (!fb_desc) {
    return absl::InvalidArgumentError("BufferDescriptor is null.");
  }
  ABSL_RETURN_IF_ERROR(Decode(fb_desc->base_obj(), desc));
  desc->element_type = ToEnum(fb_desc->element_type());
  desc->element_size = fb_desc->element_size();
  desc->memory_type = ToEnum(fb_desc->memory_type());
  if (fb_desc->attributes()) {
    for (auto attr_fb : *fb_desc->attributes()) {
      if (!attr_fb) {
        return absl::InvalidArgumentError("Buffer attribute is null.");
      }
      std::string attr(attr_fb->c_str(), attr_fb->size());
      desc->attributes.push_back(attr);
    }
  }
  desc->size = fb_desc->size();
  if (fb_desc->data()) {
    desc->data =
        std::vector<uint8_t>(fb_desc->data()->data(),
                             fb_desc->data()->data() + fb_desc->data()->size());
  } else {
    desc->data.clear();
  }
  return absl::OkStatus();
}

flatbuffers::Offset<data::TensorDescriptor> Encode(
    const TensorDescriptor& desc, flatbuffers::FlatBufferBuilder* builder) {
  auto obj_fb =
      Encode(*static_cast<const GPUObjectDescriptor*>(&desc), builder);

  data::BHWDCBuilder shape_builder(*builder);
  shape_builder.add_b(desc.GetBHWDCShape().b);
  shape_builder.add_h(desc.GetBHWDCShape().h);
  shape_builder.add_w(desc.GetBHWDCShape().w);
  shape_builder.add_d(desc.GetBHWDCShape().d);
  shape_builder.add_c(desc.GetBHWDCShape().c);
  auto shape_fb = shape_builder.Finish();

  auto data_fb =
      builder->CreateVector(desc.GetData().data(), desc.GetData().size());
  data::TensorDescriptorBuilder tensor_builder(*builder);
  tensor_builder.add_base_obj(obj_fb);
  tensor_builder.add_data_type(ToFB(desc.data_type_));
  tensor_builder.add_storage_type(ToFB(desc.storage_type_));
  tensor_builder.add_layout(ToFB(desc.layout_));
  tensor_builder.add_physical_layout_1d(ToFB(desc.physical_layout_1d_));
  tensor_builder.add_shape(shape_fb);
  tensor_builder.add_data(data_fb);
  tensor_builder.add_use_buffer_for_write_only_2d_texture(
      desc.use_buffer_for_write_only_2d_texture_);
  tensor_builder.add_use_buffer_for_write_only_image_buffer(
      desc.use_buffer_for_write_only_image_buffer_);
  return tensor_builder.Finish();
}

absl::Status Decode(const data::TensorDescriptor* fb_desc,
                    TensorDescriptor* desc) {
  if (!fb_desc) {
    return absl::InvalidArgumentError("TensorDescriptor is null.");
  }
  ABSL_RETURN_IF_ERROR(Decode(fb_desc->base_obj(), desc));
  desc->data_type_ = ToEnum(fb_desc->data_type());
  desc->storage_type_ = ToEnum(fb_desc->storage_type());
  desc->layout_ = ToEnum(fb_desc->layout());
  desc->physical_layout_1d_ = ToEnum(fb_desc->physical_layout_1d());
  if (!fb_desc->shape()) {
    return absl::InvalidArgumentError("Tensor shape is null.");
  }
  desc->SetBHWDCShape(BHWDC(fb_desc->shape()->b(), fb_desc->shape()->h(),
                            fb_desc->shape()->w(), fb_desc->shape()->d(),
                            fb_desc->shape()->c()));
  if (fb_desc->data()) {
    desc->SetData(std::vector<uint8_t>(
        fb_desc->data()->data(),
        fb_desc->data()->data() + fb_desc->data()->size()));
  } else {
    desc->SetData({});
  }
  desc->use_buffer_for_write_only_2d_texture_ =
      fb_desc->use_buffer_for_write_only_2d_texture();
  desc->use_buffer_for_write_only_image_buffer_ =
      fb_desc->use_buffer_for_write_only_image_buffer();
  return absl::OkStatus();
}

absl::Status Decode(const data::Arguments* fb_args, Arguments* args) {
  args->int_values_.clear();
  for (auto int_values_fb : *fb_args->int_values()) {
    Arguments::IntValue value;
    value.value = int_values_fb->value();
    value.active = int_values_fb->active();
    std::string name(int_values_fb->name()->c_str(),
                     int_values_fb->name()->size());
    args->int_values_[name] = value;
  }

  args->float_values_.clear();
  for (auto float_values_fb : *fb_args->float_values()) {
    Arguments::FloatValue value;
    value.value = float_values_fb->value();
    value.active = float_values_fb->active();
    std::string name(float_values_fb->name()->c_str(),
                     float_values_fb->name()->size());
    args->float_values_[name] = value;
  }

  args->half_values_.clear();
  for (auto half_values_fb : *fb_args->half_values()) {
    Arguments::HalfValue value;
    value.value = half_values_fb->value();
    value.active = half_values_fb->active();
    std::string name(half_values_fb->name()->c_str(),
                     half_values_fb->name()->size());
    args->half_values_[name] = value;
  }

  for (auto buffer_pair_fb : *fb_args->buffer_objects()) {
    std::string key(buffer_pair_fb->key()->c_str(),
                    buffer_pair_fb->key()->size());
    BufferDescriptor desc;
    ABSL_RETURN_IF_ERROR(Decode(buffer_pair_fb->value(), &desc));
    args->AddObject(key, std::make_unique<BufferDescriptor>(std::move(desc)));
  }

  for (auto tensor_pair_fb : *fb_args->tensor_objects()) {
    std::string key(tensor_pair_fb->key()->c_str(),
                    tensor_pair_fb->key()->size());
    TensorDescriptor desc;
    ABSL_RETURN_IF_ERROR(Decode(tensor_pair_fb->value(), &desc));
    args->AddObject(key, std::make_unique<TensorDescriptor>(std::move(desc)));
  }

  for (auto buffer_pair_fb : *fb_args->buffer_refs()) {
    std::string key(buffer_pair_fb->key()->c_str(),
                    buffer_pair_fb->key()->size());
    BufferDescriptor desc;
    ABSL_RETURN_IF_ERROR(Decode(buffer_pair_fb->value(), &desc));
    auto access_type = desc.GetAccess();
    args->AddObjectRef(key, access_type,
                       std::make_unique<BufferDescriptor>(std::move(desc)));
  }

  for (auto tensor_pair_fb : *fb_args->tensor_refs()) {
    std::string key(tensor_pair_fb->key()->c_str(),
                    tensor_pair_fb->key()->size());
    TensorDescriptor desc;
    ABSL_RETURN_IF_ERROR(Decode(tensor_pair_fb->value(), &desc));
    auto access_type = desc.GetAccess();
    args->AddObjectRef(key, access_type,
                       std::make_unique<TensorDescriptor>(std::move(desc)));
  }
  return absl::OkStatus();
}

flatbuffers::Offset<data::Arguments> Encode(
    const Arguments& args, flatbuffers::FlatBufferBuilder* builder) {
  std::vector<flatbuffers::Offset<data::IntValue>> int_values_fb;
  for (auto& value : args.int_values_) {
    auto name_fb = builder->CreateString(value.first);
    data::IntValueBuilder value_builder(*builder);
    value_builder.add_name(name_fb);
    value_builder.add_value(value.second.value);
    value_builder.add_active(value.second.active);
    int_values_fb.push_back(value_builder.Finish());
  }

  std::vector<flatbuffers::Offset<data::FloatValue>> float_values_fb;
  for (auto& value : args.float_values_) {
    auto name_fb = builder->CreateString(value.first);
    data::FloatValueBuilder value_builder(*builder);
    value_builder.add_name(name_fb);
    value_builder.add_value(value.second.value);
    value_builder.add_active(value.second.active);
    float_values_fb.push_back(value_builder.Finish());
  }

  std::vector<flatbuffers::Offset<data::HalfValue>> half_values_fb;
  for (auto& value : args.half_values_) {
    auto name_fb = builder->CreateString(value.first);
    data::HalfValueBuilder value_builder(*builder);
    value_builder.add_name(name_fb);
    value_builder.add_value(value.second.value);
    value_builder.add_active(value.second.active);
    half_values_fb.push_back(value_builder.Finish());
  }

  std::vector<flatbuffers::Offset<data::BufferDescriptorMapValue>>
      buffer_objs_fb;
  for (auto& value : args.objects_) {
    const BufferDescriptor* buffer_desc =
        AsBufferDescriptor(value.second.get());
    if (!buffer_desc) continue;
    auto desc_fb = Encode(*buffer_desc, builder);
    auto key_fb = builder->CreateString(value.first);
    data::BufferDescriptorMapValueBuilder buf_map_builder(*builder);
    buf_map_builder.add_key(key_fb);
    buf_map_builder.add_value(desc_fb);
    buffer_objs_fb.push_back(buf_map_builder.Finish());
  }
  std::vector<flatbuffers::Offset<data::TensorDescriptorMapValue>>
      tensor_objs_fb;
  for (auto& value : args.objects_) {
    const TensorDescriptor* tensor_desc =
        AsTensorDescriptor(value.second.get());
    if (!tensor_desc) continue;
    auto desc_fb = Encode(*tensor_desc, builder);
    auto key_fb = builder->CreateString(value.first);
    data::TensorDescriptorMapValueBuilder ten_map_builder(*builder);
    ten_map_builder.add_key(key_fb);
    ten_map_builder.add_value(desc_fb);
    tensor_objs_fb.push_back(ten_map_builder.Finish());
  }

  std::vector<flatbuffers::Offset<data::BufferDescriptorMapValue>>
      buffer_refs_fb;
  for (auto& value : args.object_refs_) {
    const BufferDescriptor* buffer_desc =
        AsBufferDescriptor(value.second.get());
    if (!buffer_desc) continue;
    auto desc_fb = Encode(*buffer_desc, builder);
    auto key_fb = builder->CreateString(value.first);
    data::BufferDescriptorMapValueBuilder buf_map_builder(*builder);
    buf_map_builder.add_key(key_fb);
    buf_map_builder.add_value(desc_fb);
    buffer_refs_fb.push_back(buf_map_builder.Finish());
  }
  std::vector<flatbuffers::Offset<data::TensorDescriptorMapValue>>
      tensor_refs_fb;
  for (auto& value : args.object_refs_) {
    const TensorDescriptor* tensor_desc =
        AsTensorDescriptor(value.second.get());
    if (!tensor_desc) continue;
    auto desc_fb = Encode(*tensor_desc, builder);
    auto key_fb = builder->CreateString(value.first);
    data::TensorDescriptorMapValueBuilder ten_map_builder(*builder);
    ten_map_builder.add_key(key_fb);
    ten_map_builder.add_value(desc_fb);
    tensor_refs_fb.push_back(ten_map_builder.Finish());
  }

  auto int_values_fb_vec = builder->CreateVector(int_values_fb);
  auto float_values_fb_vec = builder->CreateVector(float_values_fb);
  auto half_values_fb_vec = builder->CreateVector(half_values_fb);
  auto buffer_objs_fb_vec = builder->CreateVector(buffer_objs_fb);
  auto tensor_objs_fb_vec = builder->CreateVector(tensor_objs_fb);
  auto buffer_refs_fb_vec = builder->CreateVector(buffer_refs_fb);
  auto tensor_refs_fb_vec = builder->CreateVector(tensor_refs_fb);
  data::ArgumentsBuilder arguments_builder(*builder);
  arguments_builder.add_int_values(int_values_fb_vec);
  arguments_builder.add_float_values(float_values_fb_vec);
  arguments_builder.add_half_values(half_values_fb_vec);
  arguments_builder.add_buffer_objects(buffer_objs_fb_vec);
  arguments_builder.add_tensor_objects(tensor_objs_fb_vec);
  arguments_builder.add_buffer_refs(buffer_refs_fb_vec);
  arguments_builder.add_tensor_refs(tensor_refs_fb_vec);
  return arguments_builder.Finish();
}

absl::Status Decode(const data::GPUOperation* fb_op, GPUOperation* op) {
  if (!fb_op) {
    return absl::InvalidArgumentError("GPUOperation is null.");
  }
  ABSL_RETURN_IF_ERROR(Decode(fb_op->arguments(), &op->args_));
  if (!fb_op->work_group_size()) {
    return absl::InvalidArgumentError("GPUOperation work_group_size is null.");
  }
  op->work_group_size_.x = fb_op->work_group_size()->x();
  op->work_group_size_.y = fb_op->work_group_size()->y();
  op->work_group_size_.z = fb_op->work_group_size()->z();
  op->tensor_to_grid_ = ToEnum(fb_op->tensor_to_grid());
  op->flops_ = fb_op->flops();
  op->grid_dimension_ = fb_op->grid_dimension();
  if (!fb_op->work_group_launch_order()) {
    return absl::InvalidArgumentError(
        "GPUOperation work_group_launch_order is null.");
  }
  op->work_group_launch_order_.x = fb_op->work_group_launch_order()->x();
  op->work_group_launch_order_.y = fb_op->work_group_launch_order()->y();
  op->work_group_launch_order_.z = fb_op->work_group_launch_order()->z();
  if (!fb_op->grid_size()) {
    return absl::InvalidArgumentError("GPUOperation grid_size is null.");
  }
  op->grid_size_.x = fb_op->grid_size()->x();
  op->grid_size_.y = fb_op->grid_size()->y();
  op->grid_size_.z = fb_op->grid_size()->z();
  if (fb_op->src_objects_names()) {
    for (auto name_fb : *fb_op->src_objects_names()) {
      if (!name_fb) {
        return absl::InvalidArgumentError(
            "GPUOperation src_objects_name is null.");
      }
      std::string name(name_fb->c_str(), name_fb->size());
      op->src_objects_names_.push_back(std::move(name));
    }
  }
  if (fb_op->dst_objects_names()) {
    for (auto name_fb : *fb_op->dst_objects_names()) {
      if (!name_fb) {
        return absl::InvalidArgumentError(
            "GPUOperation dst_objects_name is null.");
      }
      std::string name(name_fb->c_str(), name_fb->size());
      op->dst_objects_names_.push_back(std::move(name));
    }
  }
  if (!fb_op->work_groups_count()) {
    return absl::InvalidArgumentError(
        "GPUOperation work_groups_count is null.");
  }
  op->work_groups_count_.x = fb_op->work_groups_count()->x();
  op->work_groups_count_.y = fb_op->work_groups_count()->y();
  op->work_groups_count_.z = fb_op->work_groups_count()->z();
  op->CalculateConstArgsSize();
  return absl::OkStatus();
}

flatbuffers::Offset<data::GPUOperation> Encode(
    const GPUOperation& op, flatbuffers::FlatBufferBuilder* builder) {
  auto args_fb = Encode(op.args_, builder);
  auto work_group_size_fb = Encode(op.work_group_size_, builder);

  auto work_group_launch_order_fb =
      Encode(op.work_group_launch_order_, builder);
  auto grid_size_fb = Encode(op.grid_size_, builder);
  auto work_groups_count_fb = Encode(op.work_groups_count_, builder);

  std::vector<flatbuffers::Offset<flatbuffers::String>> src_names_fb;
  src_names_fb.reserve(op.src_objects_names_.size());
  for (auto& name : op.src_objects_names_) {
    src_names_fb.push_back(builder->CreateString(name));
  }
  auto src_names_fb_vec = builder->CreateVector(src_names_fb);

  std::vector<flatbuffers::Offset<flatbuffers::String>> dst_names_fb;
  dst_names_fb.reserve(op.dst_objects_names_.size());
  for (auto& name : op.dst_objects_names_) {
    dst_names_fb.push_back(builder->CreateString(name));
  }
  auto dst_names_fb_vec = builder->CreateVector(dst_names_fb);

  data::GPUOperationBuilder op_builder(*builder);
  op_builder.add_arguments(args_fb);
  op_builder.add_work_group_size(work_group_size_fb);
  op_builder.add_tensor_to_grid(ToFB(op.tensor_to_grid_));
  op_builder.add_flops(op.flops_);
  op_builder.add_grid_dimension(op.grid_dimension_);
  op_builder.add_work_group_launch_order(work_group_launch_order_fb);
  op_builder.add_grid_size(grid_size_fb);
  op_builder.add_src_objects_names(src_names_fb_vec);
  op_builder.add_dst_objects_names(dst_names_fb_vec);
  op_builder.add_work_groups_count(work_groups_count_fb);
  return op_builder.Finish();
}

DataType ToEnum(data::DataType type) {
  switch (type) {
    case data::DataType::BOOL:
      return DataType::kBool;
    case data::DataType::FLOAT16:
      return DataType::kFloat16;
    case data::DataType::FLOAT32:
      return DataType::kFloat32;
    case data::DataType::FLOAT64:
      return DataType::kFloat64;
    case data::DataType::BFLOAT16:
      return DataType::kBfloat16;
    case data::DataType::UINT8:
      return DataType::kUint8;
    case data::DataType::INT8:
      return DataType::kInt8;
    case data::DataType::UINT16:
      return DataType::kUint16;
    case data::DataType::INT16:
      return DataType::kInt16;
    case data::DataType::UINT32:
      return DataType::kUint32;
    case data::DataType::INT32:
      return DataType::kInt32;
    case data::DataType::UINT64:
      return DataType::kUint64;
    case data::DataType::INT64:
      return DataType::kInt64;
    case data::DataType::INT4:
      return DataType::kInt4;
    case data::DataType::UINT4:
      return DataType::kUint4;
    case data::DataType::INT3:
      return DataType::kInt3;
    case data::DataType::UINT3:
      return DataType::kUint3;
    case data::DataType::INT2:
      return DataType::kInt2;
    case data::DataType::UINT2:
      return DataType::kUint2;
    case data::DataType::INT1:
      return DataType::kInt1;
    case data::DataType::UINT1:
      return DataType::kUint1;
    case data::DataType::UNKNOWN:
      return DataType::kUnknown;
  }
}

TensorStorageType ToEnum(data::TensorStorageType type) {
  switch (type) {
    case data::TensorStorageType::BUFFER:
      return TensorStorageType::kBuffer;
    case data::TensorStorageType::IMAGE_BUFFER:
      return TensorStorageType::kImageBuffer;
    case data::TensorStorageType::TEXTURE_2D:
      return TensorStorageType::kTexture2D;
    case data::TensorStorageType::TEXTURE_ARRAY:
      return TensorStorageType::kTextureArray;
    case data::TensorStorageType::TEXTURE_3D:
      return TensorStorageType::kTexture3D;
    case data::TensorStorageType::SINGLE_TEXTURE_2D:
      return TensorStorageType::kSingleTexture2D;
    case data::TensorStorageType::UNKNOWN:
      return TensorStorageType::kUnknown;
  }
}

Layout ToEnum(data::Layout type) {
  switch (type) {
    case data::Layout::HWC:
      return Layout::kHWC;
    case data::Layout::BHWC:
      return Layout::kBHWC;
    case data::Layout::HWDC:
      return Layout::kHWDC;
    case data::Layout::BHWDC:
      return Layout::kBHWDC;
    case data::Layout::LINEAR:
      return Layout::kLinear;
    case data::Layout::HW:
      return Layout::kHW;
    default:
      return Layout::kUnknown;
  }
}

}  // namespace ml_drift
