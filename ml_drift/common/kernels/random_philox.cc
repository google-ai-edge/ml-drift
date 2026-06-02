// Copyright 2024 The ML Drift Authors.
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

#include "ml_drift/common/kernels/random_philox.h"

#include <string>
#include <utility>

#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/task/gpu_operation.h"

namespace ml_drift {
namespace {
constexpr char kCommonMain[] = R"(
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  int S = ucl::GetGlobalId<2>();
  if (X >= args.dst.Width() || Y >= args.dst.Height() || S >= args.dst.Slices()) return;

  uint4 counter;
  counter.x = ucl::Convert<uint>(args.seed);
  counter.y = 0;
  counter.z = ucl::Convert<uint>(args.seed2);
  counter.w = 0;
  uint2 key;
  key.y = 0x02461E29;
  key.x = 0x3EC8F720;

  uint4 mixed = philox_4x32_10(counter, key);
  key.x = mixed.x;
  key.y = mixed.y;
  counter.x = 0;
  counter.y = 0;
  counter.z = mixed.z;
  counter.w = mixed.w;

  uint offset = ucl::Convert<uint>((Y * args.dst.Width() + X) * args.dst.Slices() + S);
  counter = Skip(counter, offset);

  uint4 rand128bit = philox_4x32_10(counter, key);
  args.dst::type result = ucl::Convert<args.dst::type>(BoxMullerFloatX4(rand128bit));
  args.dst.Write(result, X, Y, S);
})";

std::string CreateWebGpuRandomNormalPhilox() {
  // Mirrors the Metal and OpenCL versions, except it contains a mulhi
  // polyfill adapted from a version of the same from ANGLE, and the syntax
  // and structure have been changed to match WGSL formatting.
  // TODO(tmullen): Unify these separate implementations as much as possible.
  return R"(
fn MultiplyHighLow(a: u32, b: u32) -> vec2u {
  var result: vec2u;
  result.x = a * b;
  let mA : u32 = (a & 0xffffu);
  let mB : u32 = (a >> 16);
  let mC : u32 = (b & 0xffffu);
  let mD : u32 = (b >> 16);
  let mAD : u32 = mA * mD + ((mA * mC) >> 16);
  let mBC : u32 = mB * mC;
  let carry : u32 = u32(mAD > (0xffffffffu - mBC));
  result.y = ((mAD + mBC) >> 16) + (carry << 16) + mB * mD;
  return result;
}

fn ComputeSingleRound(counter: vec4u, key: vec2u) -> vec4u {
  let kPhiloxM4x32A : u32 = 0xD2511F53;
  let kPhiloxM4x32B : u32 = 0xCD9E8D57;
  let v0 : vec2u = MultiplyHighLow(kPhiloxM4x32A, counter.x);
  let v1 : vec2u = MultiplyHighLow(kPhiloxM4x32B, counter.z);

  var result : vec4u;
  result.x = v1.y ^ counter.y ^ key.x;
  result.y = v1.x;
  result.z = v0.y ^ counter.w ^ key.y;
  result.w = v0.x;
  return result;
}

fn RaiseKey(key: vec2u) -> vec2u {
  let kPhiloxW32A : u32 = 0x9E3779B9;
  let kPhiloxW32B : u32 = 0xBB67AE85;
  var result : vec2u;
  result.x = key.x + kPhiloxW32A;
  result.y = key.y + kPhiloxW32B;
  return result;
}

fn philox_4x32_10(counter: vec4u, key: vec2u) -> vec4u {
  var tempCounter : vec4u = counter;
  var tempKey : vec2u = key;
  for (var i = 0 ; i < 10; i = i+1) {
    tempCounter = ComputeSingleRound(tempCounter, tempKey);
    tempKey = RaiseKey(tempKey);
  }
  return tempCounter;
}

fn Uint32ToFloat(x: u32) -> f32 {
  let man : u32 = x & 0x7fffffu;  // 23 bit mantissa
  let exp : u32 = 127u;
  let val : u32 = (exp << 23) | man;
  return bitcast<f32>(val) - 1.0f;
}

fn BoxMullerFloat(value: vec2u) -> vec2f {
  let epsilon : f32 = 1.0e-7f;
  var u1 : f32 = Uint32ToFloat(value.x);
  if (u1 < epsilon) {
    u1 = epsilon;
  }
  let v1 : f32 = 2.0f * 3.14159265359f * Uint32ToFloat(value.y);
  let u2 : f32 = sqrt(-2.0f * log(u1));

  var result : vec2f;
  result.x = sin(v1) * u2;
  result.y = cos(v1) * u2;
  return result;
}

fn BoxMullerFloatX4(value: vec4u) -> vec4f {
  var result : vec4f;
  let first : vec2f = BoxMullerFloat(value.xy);
  let second : vec2f = BoxMullerFloat(value.zw);
  result = vec4f(first.x, first.y, second.x, second.y);
  return result;
}

fn Skip(counter: vec4u, count: u32) -> vec4u {
  var tempCounter : vec4u = counter;
  tempCounter.x += count;
  if (tempCounter.x < count) {
    tempCounter.y += 1;
    if (tempCounter.y == 0) {
      tempCounter.z += 1;
      if (tempCounter.z == 0) {
        tempCounter.w += 1;
      }
    }
  }
  return tempCounter;
})";
}
}  // namespace

GPUOperation CreateRandomNormalPhilox(const GpuInfo& gpu_info,
                                      const OperationDef& definition) {
  GPUOperation op;
  op.AddDstTensor("dst", definition.dst_tensors[0]);
  op.args_.AddInt("seed", 0);
  op.args_.AddInt("seed2", 0);
  op.tensor_to_grid_ = TensorToGrid::kWBToX_HDToY_SToZ;

  // Use custom port if WebGPU
  if (gpu_info.IsApiWebGpu()) {
    op.code_ = CreateWebGpuRandomNormalPhilox() + kCommonMain;
    return op;
  }

  std::string c;
  if (gpu_info.IsApiMetal()) {
    c += R"(
uint2 MultiplyHighLow(uint a, uint b) {
  uint2 result;
  result.x = a * b;
  result.y = mulhi(a, b);
  return result;
})";
  } else if (gpu_info.IsApiOpenCl()) {
    c += R"(
uint2 MultiplyHighLow(uint a, uint b) {
  uint2 result;
  result.x = a * b;
  result.y = mul_hi(a, b);
  return result;
})";
  }
  c += R"(
uint4 ComputeSingleRound(uint4 counter, uint2 key) {
  uint kPhiloxM4x32A = 0xD2511F53;
  uint kPhiloxM4x32B = 0xCD9E8D57;
  uint2 v0 = MultiplyHighLow(kPhiloxM4x32A, counter.x);
  uint2 v1 = MultiplyHighLow(kPhiloxM4x32B, counter.z);

  uint4 result;
  result.x = v1.y ^ counter.y ^ key.x;
  result.y = v1.x;
  result.z = v0.y ^ counter.w ^ key.y;
  result.w = v0.x;
  return result;
}

uint2 RaiseKey(uint2 key) {
  uint kPhiloxW32A = 0x9E3779B9;
  uint kPhiloxW32B = 0xBB67AE85;
  uint2 result;
  result.x = key.x + kPhiloxW32A;
  result.y = key.y + kPhiloxW32B;
  return result;
}

uint4 philox_4x32_10(uint4 counter, uint2 key) {
  for (int i = 0 ; i < 10; ++i) {
    counter = ComputeSingleRound(counter, key);
    key = RaiseKey(key);
  }
  return counter;
}

float Uint32ToFloat(uint x) {
  // IEEE754 floats are formatted as follows (MSB first):
  //    sign(1) exponent(8) mantissa(23)
  // Conceptually construct the following:
  //    sign == 0
  //    exponent == 127  -- an excess 127 representation of a zero exponent
  //    mantissa == 23 random bits
  uint man = x & 0x7fffffu;  // 23 bit mantissa
  uint exp = 127u;
  uint val = (exp << 23) | man;

  // Assumes that endian-ness is same for float and uint32_t.
)";
  if (gpu_info.IsApiMetal()) {
    c += "  return as_type<float>(val) - 1.0f;\n";
  } else if (gpu_info.IsApiOpenCl()) {
    c += "  return as_float(val) - 1.0f;\n";
  }
  c += R"(
}

float2 BoxMullerFloat(uint2 value) {
  float epsilon = 1.0e-7f;
  float u1 = Uint32ToFloat(value.x);
  if (u1 < epsilon) {
    u1 = epsilon;
  }
  float v1 = 2.0f * 3.14159265359f * Uint32ToFloat(value.y);
  float u2 = sqrt(-2.0f * log(u1));

  float2 result;
  result.x = sin(v1) * u2;
  result.y = cos(v1) * u2;
  return result;
}

float4 BoxMullerFloatX4(uint4 value) {
  float4 result;
  result.xy = BoxMullerFloat(value.xy);
  result.zw = BoxMullerFloat(value.zw);
  return result;
}

uint4 Skip(uint4 counter, uint count) {
  counter.x += count;
  if (counter.x < count) {
    counter.y += 1;
    if (counter.y == 0) {
      counter.z += 1;
      if (counter.z == 0) {
        counter.w += 1;
      }
    }
  }
  return counter;
})";
  c += kCommonMain;
  op.code_ = std::move(c);
  return op;
}

}  // namespace ml_drift
