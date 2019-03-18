/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2019 by Contributors
 * \file mkldnn_quantized_sum.cc
 * \brief
 */

#if MXNET_USE_MKLDNN == 1
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"
#include "../quantization_utils.h"

namespace mxnet {
namespace op {

namespace quantized_sum_enum {
enum QuantizedSumOutputs { kOut, kMin, kMax };
enum QuantizedSumInputs { kDataA, kDataB, kAMin, kAMax, kBMin, kBMax};
}

struct RequantizeSumParam : public dmlc::Parameter<RequantizeSumParam> {
  dmlc::optional<float> min_calib_range;  // min float value calculated from calibration dataset
  dmlc::optional<float> max_calib_range;  // max float value calculated from calibration dataset
  DMLC_DECLARE_PARAMETER(RequantizeSumParam) {
    DMLC_DECLARE_FIELD(min_calib_range)
    .set_default(dmlc::optional<float>())
    .describe("The minimum scalar value in the form of float32 obtained "
              "through calibration. If present, it will be used to requantize the "
              "int8 output data.");
    DMLC_DECLARE_FIELD(max_calib_range)
    .set_default(dmlc::optional<float>())
    .describe("The maximum scalar value in the form of float32 obtained "
              "through calibration. If present, it will be used to requantize the "
              "int8 output data.");
  }
};

DMLC_REGISTER_PARAMETER(RequantizeSumParam);

static float GetScale(const NDArray& data, float min, float max) {
  auto data_range = (data.dtype() == mshadow::kInt8) ? kInt8Range : kUint8Range;
  return data_range / MaxAbs(min, max);
}

static void MKLDNNQuantizedSumForward(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                                         const std::vector<NDArray>& in_data,
                                         const std::vector<OpReqType>& req,
                                         const std::vector<NDArray>& out_data) {
  const RequantizeSumParam& params = nnvm::get<RequantizeSumParam>(attrs.parsed);
  // A, B, A_min, A_max, B_min, B_max
  CHECK_EQ(in_data.size(), static_cast<size_t>(6));
  // C, C_min, C_max
  CHECK_EQ(out_data.size(), 3U);
  // sum must fusion with requantize to get best accuracy
  //CHECK_EQ(params.max_calib_range.has_value(), true);
  //CHECK_EQ(params.min_calib_range.has_value(), true);
  
  // Collect data min/max
  float dataA_min = in_data[quantized_sum_enum::kAMin].data().dptr<float>()[0];
  float dataB_min = in_data[quantized_sum_enum::kBMin].data().dptr<float>()[0];
  float dataA_max = in_data[quantized_sum_enum::kAMax].data().dptr<float>()[0];
  float dataB_max = in_data[quantized_sum_enum::kBMax].data().dptr<float>()[0];

  auto dataA_mem  = in_data[quantized_sum_enum::kDataA].GetMKLDNNData();
  auto dataB_mem  = in_data[quantized_sum_enum::kDataB].GetMKLDNNData();
  bool dataA_int8 = (in_data[quantized_sum_enum::kDataA].dtype() == mshadow::kInt8) ? true : false;
  
  // rescaled_mem is for reorder mkldnn memory
  std::shared_ptr<mkldnn::memory> rescaled_mem;
  size_t output_data_range = kInt8Range;
  auto output_data_type = mkldnn::memory::s8;
  if(in_data[quantized_sum_enum::kDataA].dtype() != in_data[quantized_sum_enum::kDataB].dtype()) {
  auto u8_mem = (dataA_int8 == true) ? dataB_mem : dataA_mem;
  auto s8_mem = (dataA_int8 == true) ? dataA_mem : dataB_mem;
  auto s8_pd = s8_mem->get_primitive_desc();
  rescaled_mem = std::make_shared<mkldnn::memory>(s8_pd);
  //convert uint8 (0-255) to sint8 (0-125)
  std::vector<float> reorder_scale = {0.5};
  primitive_attr reorder_attr;
  reorder_attr.set_int_output_round_mode(round_mode::round_nearest);
  reorder_attr.set_output_scales(0, reorder_scale);
  const auto reorder_pd = mkldnn::reorder::primitive_desc(u8_mem->get_primitive_desc(), s8_pd, reorder_attr);
  MKLDNNStream::Get()->RegisterPrim(mkldnn::reorder(reorder_pd, *u8_mem, *rescaled_mem));
  
    if(dataA_int8 == true) {
    dataB_max *= 2;
      dataB_mem = rescaled_mem.get();
  } else {
    dataA_max *= 2;
      dataA_mem = rescaled_mem.get();
  }
  }else {
    if(dataA_int8 == false) {
      output_data_range = kUint8Range;
      output_data_type = mkldnn::memory::u8;
    }
  }

  std::vector<mkldnn::primitive::at> in_prims;
  std::vector<mkldnn::memory::primitive_desc> in_pds;
  in_prims.push_back(*dataA_mem);
  in_prims.push_back(*dataB_mem);
  in_pds.push_back(dataA_mem->get_primitive_desc());
  in_pds.push_back(dataB_mem->get_primitive_desc());
  
  float A_scale = GetScale(in_data[quantized_sum_enum::kDataA], dataA_min, dataA_max);
  float B_scale = GetScale(in_data[quantized_sum_enum::kDataB], dataB_min, dataB_max);
  
  std::vector<float> scales;
  float out_data_scale = 0;
  float output_min = 0;
  float output_max = 0;
  
  if(params.max_calib_range.has_value() && params.min_calib_range.has_value()) 
{
  output_min = params.min_calib_range.value();
  output_max = params.max_calib_range.value();
  out_data_scale = output_data_range/MaxAbs(output_min, output_max);
  scales.push_back(out_data_scale/A_scale);
  scales.push_back(out_data_scale/B_scale);
  } else {
    // reserved for no FNeedRequantize version
  scales.push_back(B_scale/(A_scale + B_scale));
  scales.push_back(1 - scales[0]);
    out_data_scale = (A_scale < B_scale) ? (1/scales[0]) : (1/scales[1]);
  float min_AB = Min(dataA_min, dataB_min);
  float max_AB = Max(dataA_max, dataB_max);
  output_min = Min(min_AB, 0.f)*out_data_scale;
  output_max = Max(max_AB, 0.f)*out_data_scale;;
  }
    
  size_t i_ndim = in_data[quantized_sum_enum::kDataA].shape().ndim();
  mkldnn::memory::dims i_dims = mkldnn::memory::dims(i_ndim);
  for (size_t i = 0; i < i_ndim; i++) {
    i_dims[i] = static_cast<int>(in_data[quantized_sum_enum::kDataA].shape()[i]);
  }

  mkldnn::memory::format i_fmt = static_cast<mkldnn::memory::format>(
                                   in_pds[quantized_sum_enum::kDataA].desc().data.format);
  auto output_desc = memory::desc(i_dims, output_data_type, i_fmt);
  mkldnn::sum::primitive_desc pdesc(output_desc, scales, in_pds);
  
  auto mem = CreateMKLDNNMem(out_data[quantized_sum_enum::kOut], pdesc.dst_primitive_desc(), req[0], &in_data[0]);
  MKLDNNStream *stream = MKLDNNStream::Get();
  stream->RegisterPrim(mkldnn::sum(pdesc, in_prims, *mem.second));
  CommitOutput(out_data[quantized_sum_enum::kOut], mem);
  stream->Submit();
  
  out_data[quantized_sum_enum::kMin].data().dptr<float>()[0] = output_min;
  out_data[quantized_sum_enum::kMax].data().dptr<float>()[0] = output_max;
}

inline static bool SumStorageType(const nnvm::NodeAttrs& attrs, const int dev_mask,
                                     DispatchMode* dispatch_mode, std::vector<int>* in_attrs,
                                     std::vector<int>* out_attrs) {
  // A, B, A_min, A_max, B_min, B_max                                     
  CHECK_EQ(in_attrs->size(), 6U);
  // C, C_min, C_max                                     
  CHECK_EQ(out_attrs->size(), 3U);

  return MKLDNNStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
}

NNVM_REGISTER_OP(_contrib_quantized_sum)
.set_attr<FInferStorageType>("FInferStorageType", SumStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", MKLDNNQuantizedSumForward)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<bool>("TIsMKLDNN", true)
.set_attr_parser(ParamParser<RequantizeSumParam>)
.add_arguments(RequantizeSumParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
