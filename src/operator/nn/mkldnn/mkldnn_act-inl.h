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
 * \file mkldnn_act.cc
 * \brief
 * \author Da Zheng
*/

#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_ACT_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_ACT_INL_H_

#if MXNET_USE_MKLDNN == 1

#include <mkldnn.hpp>
#include "../activation-inl.h"

namespace mxnet {
namespace op {

static inline mkldnn::algorithm GetMKLDNNActAlgo(const ActivationParam& param) {
  switch (param.act_type) {
    case activation::kReLU:
      return mkldnn::algorithm::eltwise_relu;
    case activation::kSigmoid:
      return mkldnn::algorithm::eltwise_logistic;
    case activation::kTanh:
      return mkldnn::algorithm::eltwise_tanh;
    case activation::kSoftReLU:
      return mkldnn::algorithm::eltwise_soft_relu;
    default:
      LOG(FATAL) << "unknown activation type";
      return mkldnn::algorithm::eltwise_relu;
  }
}
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_BATCH_NORM_INL_H_
