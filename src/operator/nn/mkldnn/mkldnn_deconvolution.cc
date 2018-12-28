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
 * \file mkldnn_deconvolution.cc
 * \brief
 * \author Da Zheng, Rong Zhang (rong.a.zhang@intel.com)
*/

#if MXNET_USE_MKLDNN == 1

#include "../deconvolution-inl.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {

bool SupportMKLDNNDeconv(const DeconvolutionParam& params, const NDArray &input) {
  if (params.kernel.ndim() != 2)
    return false;
  return input.dtype() == mshadow::kFloat32 && input.shape().ndim() == 4;
}

static inline mkldnn::memory::desc GetBiasDesc(mkldnn::memory::desc md) {
  mkldnn::memory::dims dims(1);
  // This is deconvolution on 4D data. The second dimension is the channel.
  dims[0] = md.data.dims[1];
  return mkldnn::memory::desc(dims,
      static_cast<mkldnn::memory::data_type>(md.data.data_type),
      mkldnn::memory::format::any);
}

static mkldnn::deconvolution_forward::primitive_desc GetDeconvFwdImpl(
    const DeconvolutionParam& param, const bool is_train, const NDArray &data, const NDArray &weights,
    bool has_bias, const NDArray &output) {
  auto prop = is_train ? mkldnn::prop_kind::forward_training : mkldnn::prop_kind::forward_scoring;
  auto data_md = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weights, param.num_group);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Get()->get_engine();
  CHECK_GE(param.stride.ndim(), 2U);
  CHECK_GE(param.pad.ndim(), 2U);
  CHECK_GE(param.dilate.ndim(), 2U);
  mkldnn::memory::dims strides{0, 0};
  strides[0] = param.stride[0];
  strides[1] = param.stride[1];
  mkldnn::memory::dims padding{0, 0};
  padding[0] = param.pad[0];
  padding[1] = param.pad[1];
  mkldnn::memory::dims dilate{0, 0};
  dilate[0] = param.dilate[0] - 1;
  dilate[1] = param.dilate[1] - 1;
  mkldnn::deconvolution_forward::desc desc(prop, mkldnn::algorithm::deconvolution_direct,
      out_md, weight_md, data_md, strides, dilate, padding, padding,
      mkldnn::padding_kind::zero);
  auto deconv_pd = mkldnn::deconvolution_forward::primitive_desc(desc, engine);
  // MKL-DNN introduced padded formats since 0.15 which require more memory
  // for computation compared with the actual tensor size. Currently, MKL-DNN
  // operators are still reusing those memory from memory planning and the
  // memory size may smaller than what MKL-DNN kernels require. So here we need
  // select suboptimal kernel for computation according to tensor sizes.
  while (deconv_pd.dst_primitive_desc().get_size() != GetMemDescSize(data_md) ||
         deconv_pd.src_primitive_desc().get_size() != GetMemDescSize(out_md) ||
         deconv_pd.weights_primitive_desc().get_size() != GetMemDescSize(weight_md)) {
    CHECK(deconv_pd.next_impl()) << "No implementation";
  }
  return deconv_pd;
}

static mkldnn::deconvolution_backward_data::primitive_desc GetDeconvBwdDataImpl(
    const DeconvolutionParam &param, const NDArray &data,
    const NDArray &weights, bool has_bias, const NDArray &output,
    const mkldnn::deconvolution_forward::primitive_desc &fwd_pd) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weights, param.num_group);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Get()->get_engine();
  CHECK_GE(param.stride.ndim(), 2U);
  CHECK_GE(param.pad.ndim(), 2U);
  CHECK_GE(param.dilate.ndim(), 2U);
  mkldnn::memory::dims strides{0, 0};
  strides[0] = param.stride[0];
  strides[1] = param.stride[1];
  mkldnn::memory::dims padding{0, 0};
  padding[0] = param.pad[0];
  padding[1] = param.pad[1];
  mkldnn::memory::dims dilate{0, 0};
  dilate[0] = param.dilate[0] - 1;
  dilate[1] = param.dilate[1] - 1;
  mkldnn::deconvolution_backward_data::desc desc(mkldnn::algorithm::deconvolution_direct,
      data_md, weight_md, out_md, strides, dilate, padding, padding,
      mkldnn::padding_kind::zero);
  auto deconv_pd = mkldnn::deconvolution_backward_data::primitive_desc(desc, engine, fwd_pd);
  while (deconv_pd.diff_dst_primitive_desc().get_size() != GetArraySize(output) ||
          deconv_pd.diff_src_primitive_desc().get_size() != GetArraySize(data) ||
          deconv_pd.weights_primitive_desc().get_size() != GetArraySize(weights)) {
    CHECK(deconv_pd.next_impl()) << "No implementation";
  }
  return deconv_pd;
}

static mkldnn::deconvolution_backward_weights::primitive_desc
GetDeconvBwdWeightsImpl(
    const DeconvolutionParam &param, const NDArray &data,
    const NDArray &weights, bool has_bias, const NDArray &output,
    const mkldnn::deconvolution_forward::primitive_desc &fwd_pd) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weights, param.num_group);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Get()->get_engine();
  CHECK_GE(param.stride.ndim(), 2U);
  CHECK_GE(param.pad.ndim(), 2U);
  CHECK_GE(param.dilate.ndim(), 2U);
  mkldnn::memory::dims strides{0, 0};
  strides[0] = param.stride[0];
  strides[1] = param.stride[1];
  mkldnn::memory::dims padding{0, 0};
  padding[0] = param.pad[0];
  padding[1] = param.pad[1];
  mkldnn::memory::dims dilate{0, 0};
  dilate[0] = param.dilate[0] - 1;
  dilate[1] = param.dilate[1] - 1;

  // MKL-DNN introduced padded formats since 0.15 which require more memory
  // for computation compared with the actual tensor size. Currently, MKL-DNN
  // operators are still reusing those memory from memory planning and the
  // memory size may smaller than what MKL-DNN kernels require. So here we need
  // select suboptimal kernel for computation according to tensor sizes.
  if (!has_bias) {
    mkldnn::deconvolution_backward_weights::desc desc(mkldnn::algorithm::deconvolution_direct,
        out_md, weight_md, data_md, strides, dilate, padding, padding, mkldnn::padding_kind::zero);
    auto deconv_pd = mkldnn::deconvolution_backward_weights::primitive_desc(desc, engine, fwd_pd);
    while (deconv_pd.diff_dst_primitive_desc().get_size() != GetMemDescSize(data_md) ||
           deconv_pd.src_primitive_desc().get_size() != GetMemDescSize(out_md) ||
           deconv_pd.diff_weights_primitive_desc().get_size() != GetMemDescSize(weight_md)) {
      CHECK(deconv_pd.next_impl()) << "No implementation";
    }
    return deconv_pd;
  } else {
    auto bias_md = GetBiasDesc(data_md);
    mkldnn::deconvolution_backward_weights::desc desc(mkldnn::algorithm::deconvolution_direct,
        out_md, weight_md, bias_md, data_md, strides, dilate, padding, padding,
        mkldnn::padding_kind::zero);
    auto deconv_pd = mkldnn::deconvolution_backward_weights::primitive_desc(desc, engine, fwd_pd);
    while (deconv_pd.diff_dst_primitive_desc().get_size() != GetMemDescSize(data_md) ||
           deconv_pd.src_primitive_desc().get_size() != GetMemDescSize(out_md) ||
           deconv_pd.diff_weights_primitive_desc().get_size() != GetMemDescSize(weight_md)) {
      CHECK(deconv_pd.next_impl()) << "No implementation";
    }
    return deconv_pd;
  }
}

class MKLDNNDeconvForward {
  std::shared_ptr<mkldnn::deconvolution_forward> fwd;
  std::shared_ptr<mkldnn::memory> data;
  std::shared_ptr<mkldnn::memory> weight;
  std::shared_ptr<mkldnn::memory> bias;
  std::shared_ptr<mkldnn::memory> out;
  OutDataOp data_op;

 public:
  MKLDNNDeconvForward(const DeconvolutionParam& param,
                      bool is_train,
                      const NDArray &data,
                      const NDArray &weights,
                      bool has_bias,
                      const NDArray &output);
  void SetDataHandle(const DeconvolutionParam& param,
                     const OpContext &ctx,
                     const NDArray &in_data,
                     const NDArray &weight,
                     const std::vector<OpReqType> &req,
                     const std::vector<NDArray> &out_data);

  void Execute(const std::vector<NDArray> &out_data);

 private:
  mkldnn::deconvolution_forward::primitive_desc fwd_pd;
};  // class MKLDNNDeconvForward

MKLDNNDeconvForward::MKLDNNDeconvForward(const DeconvolutionParam& param,
                                bool is_train,
                                const NDArray &data,
                                const NDArray &weights,
                                bool has_bias,
                                const NDArray &output)
                                :fwd_pd(GetDeconvFwdImpl(param, is_train, data, weights, has_bias, output)) {
  this->data = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
          fwd_pd.dst_primitive_desc()));
  this->weight = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
          fwd_pd.weights_primitive_desc()));
  this->out = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
          fwd_pd.src_primitive_desc()));
  this->fwd = std::shared_ptr<mkldnn::deconvolution_forward>(
    new mkldnn::deconvolution_forward(fwd_pd,
                                            mkldnn::primitive::at(*this->data),
                                            mkldnn::primitive::at(*this->weight),
                                            *this->out));
}

void MKLDNNDeconvForward::SetDataHandle(const DeconvolutionParam& param,
                                        const OpContext &ctx,
                                        const NDArray &in_data,
                                        const NDArray &weight,
                                        const std::vector<OpReqType> &req,
                                        const std::vector<NDArray> &out_data) {
  auto data_mem = in_data.GetMKLDNNDataReorder(
      fwd_pd.dst_primitive_desc());
  const mkldnn::memory *weight_mem;
  if (ctx.is_train) {
    // TODO(zhengda) kvstore doesn't handle MKLDNN correctly. Let's reorder it
    // to the default format for now.
    if (weight.IsMKLDNNData())
      // This asks the engine to reorder data after the weight array is used.
      const_cast<NDArray&>(weight).Reorder2DefaultAsync();
    weight_mem = GetWeights(weight, fwd_pd.weights_primitive_desc(), param.num_group);
  } else {
    // For inference, we want to reorder the weight array so we don't need to
    // reorder data every time.
    if (weight.IsDefaultData()) {
      weight_mem = GetWeights(weight, fwd_pd.weights_primitive_desc(), param.num_group);
      // We also need to modify the layout on the original weight array. The
      // data conversion happens after the weight array is used.
      const_cast<NDArray&>(weight).MKLDNNDataReorderAsync(fwd_pd.weights_primitive_desc());
    } else {
      weight_mem = weight.GetMKLDNNData();
      CHECK(weight_mem->get_primitive_desc() == fwd_pd.weights_primitive_desc());
    }
  }
  auto out_mem = CreateMKLDNNMem(out_data[deconv::kOut],
      fwd_pd.src_primitive_desc(), req[deconv::kOut]);
  auto output = out_mem.second;
  this->data->set_data_handle(data_mem->get_data_handle());
  this->weight->set_data_handle(weight_mem->get_data_handle());
  this->out->set_data_handle(output->get_data_handle());
  this->data_op = out_mem.first;
}

void MKLDNNDeconvForward::Execute(const std::vector<NDArray> &out_data) {
  MKLDNNStream::Get()->RegisterPrim(*fwd);
  CommitOutput(out_data[deconv::kOut], mkldnn_output_t(this->data_op, this->out.get()));
  MKLDNNStream::Get()->Submit();
}

static void MKLDNNDeconvFwdBiasPostProcess(const DeconvolutionParam& param,
                                           const OpContext &ctx,
                                           const NDArray &bias,
                                           const std::vector<NDArray> &out_data) {
  // add bias, broadcast bias to dim 1: channel
  if (!param.no_bias) {
    // MKLDNN only supports float right now.
    typedef float DType;
    Stream<cpu> *s = ctx.get_stream<cpu>();
    Tensor<cpu, 1, DType> b = bias.data().get<cpu, 1, DType>(s);
    // If the output data is stored in a special MKLDNN format, data()
    // automatically converts its format to the default format.
    // Unfortunately, MKLDNN doesn't support broadcast.
    Tensor<cpu, 4, DType> out_cpu = out_data[deconv::kOut].data().get<cpu, 4, DType>(s);
    out_cpu += mshadow::expr::broadcast<1>(b, out_cpu.shape_);
  }
}

static inline MKLDNNDeconvForward &GetDeconvFwd(
    const nnvm::NodeAttrs& attrs, const NDArray &data,
    const NDArray &weights, const NDArray *bias,
    const NDArray &output, const bool is_train) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local
        std::unordered_map<DeconvSignature, MKLDNNDeconvForward, OpHash> fwds;
#else
  static MX_THREAD_LOCAL
        std::unordered_map<DeconvSignature, MKLDNNDeconvForward, OpHash> fwds;
#endif
  const DeconvolutionParam& param = nnvm::get<DeconvolutionParam>(attrs.parsed);
  DeconvSignature key(param);
  // Here we can sign the conv op with NDArray because conv primitive will
  // decide the right layout for the, so we only need to get the shape and the
  // data type of the arrays.
  key.AddSign(data);
  key.AddSign(weights);
  key.AddSign(is_train);
  key.AddSign(output);
  if (bias)
    key.AddSign(*bias);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    bool has_bias = (bias != nullptr);
    MKLDNNDeconvForward fwd(param, is_train, data, weights, has_bias, output);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

void MKLDNNDeconvolutionForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                                const std::vector<NDArray> &in_data,
                                const std::vector<OpReqType> &req,
                                const std::vector<NDArray> &out_data) {
  TmpMemMgr::Get()->Init(ctx.requested[deconv::kTempSpace]);
  const DeconvolutionParam& param = nnvm::get<DeconvolutionParam>(attrs.parsed);

  auto data = in_data[deconv::kData];
  if (data.IsView() && data.IsMKLDNNData())
    data = data.Reorder2Default();

  auto weight = in_data[deconv::kWeight];
  if (weight.IsView() && weight.IsMKLDNNData())
    weight = weight.Reorder2Default();

  const NDArray* bias = param.no_bias ? nullptr : &in_data[deconv::kBias];

  MKLDNNDeconvForward &deconvFwd = GetDeconvFwd(
      attrs, data, weight, bias, out_data[deconv::kOut], ctx.is_train);

  deconvFwd.SetDataHandle(param, ctx, data, weight, req, out_data);

  deconvFwd.Execute(out_data);

  MKLDNNDeconvFwdBiasPostProcess(param, ctx, *bias, out_data);
}

class MKLDNNDeconvBackward {
  std::shared_ptr<mkldnn::deconvolution_backward_data> bwd_data;
  std::shared_ptr<mkldnn::deconvolution_backward_weights> bwd_weight;
  // conv::kData
  std::shared_ptr<mkldnn::memory> out_grad;
  std::shared_ptr<mkldnn::memory> in_grad;
  std::shared_ptr<mkldnn::memory> weight;
  // conv::kWeight
  std::shared_ptr<mkldnn::memory> data;
  std::shared_ptr<mkldnn::memory> output;
  std::shared_ptr<mkldnn::memory> in_grad_weight;

 public:
  const mkldnn::deconvolution_backward_data::primitive_desc bwdData_pd;
  const mkldnn::deconvolution_backward_weights::primitive_desc bwdWeights_pd;

  MKLDNNDeconvBackward(const DeconvolutionParam &param, const NDArray &data,
                           const NDArray &weights, const NDArray &output,
                           const mkldnn::deconvolution_forward::primitive_desc &fwd_pd)
      : bwdData_pd(GetDeconvBwdDataImpl(param, data, weights, false, output, fwd_pd)),
        bwdWeights_pd(GetDeconvBwdWeightsImpl(param, data, weights, false, output,
                                   fwd_pd)) {
  }

  void SetDataNewMem(const mkldnn::memory &out_grad, const mkldnn::memory &weight,
                 const mkldnn::memory &in_grad) {
    if (this->out_grad == nullptr)
      this->out_grad = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
        bwdData_pd.diff_dst_primitive_desc(), out_grad.get_data_handle()));
    else
      this->out_grad->set_data_handle(out_grad.get_data_handle());
    if (this->in_grad == nullptr)
      this->in_grad = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
        bwdData_pd.diff_src_primitive_desc(), in_grad.get_data_handle()));
    else
      this->in_grad->set_data_handle(in_grad.get_data_handle());
    if (this->weight == nullptr)
      this->weight = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
         bwdData_pd.weights_primitive_desc(), weight.get_data_handle()));
    else
      this->weight->set_data_handle(weight.get_data_handle());
    if (this->bwd_data == nullptr)
      this->bwd_data = std::shared_ptr<mkldnn::deconvolution_backward_data>(
        new mkldnn::deconvolution_backward_data(
          this->bwdData_pd, mkldnn::primitive::at(*this->out_grad),
          mkldnn::primitive::at(*this->weight), *this->in_grad));
  }

  void SetWeightsNewMem(const mkldnn::memory &data,
                 const mkldnn::memory &out_grad,
                 const mkldnn::memory &in_grad_weight) {
    if (this->data == nullptr)
      this->data = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
          bwdWeights_pd.src_primitive_desc(), data.get_data_handle()));
    else
      this->data->set_data_handle(data.get_data_handle());
    if (this->output == nullptr)
      this->output = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
          bwdWeights_pd.diff_dst_primitive_desc(), out_grad.get_data_handle()));
    else
      this->output->set_data_handle(out_grad.get_data_handle());
    if (this->in_grad_weight == nullptr)
      this->in_grad_weight = std::shared_ptr<mkldnn::memory>(
          new mkldnn::memory(bwdWeights_pd.diff_weights_primitive_desc(),
                             in_grad_weight.get_data_handle()));
    else
      this->in_grad_weight->set_data_handle(in_grad_weight.get_data_handle());

    if (this->bwd_weight == nullptr)
      this->bwd_weight = std::shared_ptr<mkldnn::deconvolution_backward_weights>(
          new mkldnn::deconvolution_backward_weights(
              this->bwdWeights_pd, mkldnn::primitive::at(*this->data),
              mkldnn::primitive::at(*this->output), *this->in_grad_weight));
  }


  const mkldnn::deconvolution_backward_data &GetBwdData() const { return *bwd_data; }

  const mkldnn::deconvolution_backward_weights &GetBwdWeights() const { return *bwd_weight; }

};

typedef ParamOpSign<DeconvolutionParam> MKLDNNDeconvSignature;

static inline MKLDNNDeconvBackward &GetDeconvBwd(
    const DeconvolutionParam &param, const NDArray &data,
    const NDArray &weights, const NDArray &output,
    const mkldnn::deconvolution_forward::primitive_desc &fwd_pd) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNDeconvSignature,
                                         MKLDNNDeconvBackward, OpHash>
      bwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNDeconvSignature,
                                            MKLDNNDeconvBackward, OpHash>
      bwds;
#endif
  MKLDNNDeconvSignature key(param);
  // Here we can sign the conv op with NDArray because conv primitive will
  // decide the right layout for the, so we only need to get the shape and the
  // data type of the arrays.
  key.AddSign(data);
  key.AddSign(weights);
  key.AddSign(output);

  auto it = bwds.find(key);
  if (it == bwds.end()) {
    MKLDNNDeconvBackward bwd(param, data, weights, output, fwd_pd);
    it = AddToCache(&bwds, key, bwd);
  }
  return it->second;
}

void MKLDNNDeconvolutionBackward(const nnvm::NodeAttrs &attrs,
                                 const OpContext &ctx,
                                 const std::vector<NDArray> &inputs,
                                 const std::vector<OpReqType> &req,
                                 const std::vector<NDArray> &outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[deconv::kTempSpace]);
  const std::vector<NDArray> &in_grad = outputs;
  const DeconvolutionParam &param = nnvm::get<DeconvolutionParam>(attrs.parsed);

  auto data = inputs[deconv::kData + 1];
  if (data.IsView() && data.IsMKLDNNData())
    data = data.Reorder2Default();

  auto weight = inputs[deconv::kWeight + 1];
  if (weight.IsView() && weight.IsMKLDNNData())
    weight = weight.Reorder2Default();

  auto out_grad = inputs[deconv::kOut];
  if (out_grad.IsView() && out_grad.IsMKLDNNData())
    out_grad = out_grad.Reorder2Default();

  CHECK_NE(req[deconv::kWeight], kWriteInplace)
      << "cannot write weight inplace";
  
  mkldnn::deconvolution_forward::primitive_desc fwd_pd = GetDeconvFwdImpl(
    param, ctx.is_train, data, weight, !param.no_bias, out_grad);
  
  MKLDNNDeconvBackward &bwd_data =
      GetDeconvBwd(param, data, weight, inputs[deconv::kOut], fwd_pd);
  auto out_grad_mem = inputs[deconv::kOut].GetMKLDNNDataReorder(
      bwd_data.bwdData_pd.diff_src_primitive_desc());
  if (req[deconv::kData]) {
    auto weight_mem =
        GetWeights(weight, bwd_data.bwdData_pd.weights_primitive_desc(), param.num_group);
    auto in_grad_mem =
        CreateMKLDNNMem(in_grad[deconv::kData],
                        bwd_data.bwdData_pd.diff_dst_primitive_desc(), req[deconv::kData]);
    bwd_data.SetDataNewMem(*out_grad_mem, *weight_mem, *in_grad_mem.second);
    MKLDNNStream::Get()->RegisterPrim(bwd_data.GetBwdData());
    CommitOutput(in_grad[deconv::kData], in_grad_mem);
  }
  if (req[deconv::kWeight]) {
    MKLDNNDeconvBackward &bwd_weights = GetDeconvBwd(
        param, data, weight,
        inputs[deconv::kOut], fwd_pd);
    if (bwd_data.bwdData_pd.diff_dst_primitive_desc() != bwd_weights.bwdWeights_pd.diff_dst_primitive_desc())
      out_grad_mem = inputs[deconv::kOut].GetMKLDNNDataReorder(
          bwd_weights.bwdWeights_pd.diff_dst_primitive_desc());
    auto data_mem = data.GetMKLDNNDataReorder(
        bwd_weights.bwdWeights_pd.src_primitive_desc());
    auto in_grad_weight = CreateMKLDNNWeightGrad(
        in_grad[deconv::kWeight], bwd_weights.bwdWeights_pd.diff_weights_primitive_desc(),
        req[deconv::kWeight]);
    bwd_weights.SetWeightsNewMem(*data_mem, *out_grad_mem, *in_grad_weight.second);
    MKLDNNStream::Get()->RegisterPrim(bwd_weights.GetBwdWeights());
    CommitOutput(in_grad[deconv::kWeight], in_grad_weight);
  }
  MKLDNNStream::Get()->Submit();
  if (!param.no_bias) {
    typedef float DType;
    Stream<cpu> *s = ctx.get_stream<cpu>();
    Tensor<cpu, 1, DType> gbias =
        in_grad[deconv::kBias].data().get<cpu, 1, DType>(s);
    // If there is bias, the out grad has already been converted to the default
    // format, so this shouldn't cause any performance issues.
    Tensor<cpu, 4, DType> grad =
        inputs[deconv::kOut].data().get<cpu, 4, DType>(s);
    Assign(gbias, req[deconv::kBias],
           mshadow::expr::sumall_except_dim<1>(grad));
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
