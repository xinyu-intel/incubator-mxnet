# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import mxnet as mx
import numpy as np
from mxnet.test_utils import assert_almost_equal, rand_ndarray, rand_shape_nd, same, set_default_context


def test_quantize_float32_to_int8():
    shape = rand_shape_nd(4)
    data = rand_ndarray(shape, 'default', dtype='float32')
    min_range = mx.nd.min(data)
    max_range = mx.nd.max(data)
    qdata, min_val, max_val = mx.nd.contrib.quantize(data, min_range, max_range, out_type='int8')
    data_np = data.asnumpy()
    min_range = min_range.asscalar()
    max_range = max_range.asscalar()
    real_range = np.maximum(np.abs(min_range), np.abs(max_range))
    quantized_range = 127.0
    scale = quantized_range / real_range
    assert qdata.dtype == np.int8
    assert min_val.dtype == np.float32
    assert max_val.dtype == np.float32
    assert same(min_val.asscalar(), -real_range)
    assert same(max_val.asscalar(), real_range)
    qdata_np = (np.sign(data_np) * np.minimum(np.abs(data_np) * scale + 0.5, quantized_range)).astype(np.int8)
    assert same(qdata.asnumpy(), qdata_np)


def test_dequantize_int8_to_float32():
    shape = rand_shape_nd(4)
    qdata_np = np.random.uniform(low=-127, high=127, size=shape).astype(dtype=np.int8)
    qdata = mx.nd.array(qdata_np, dtype=np.int8)
    real_range = 402.3347
    min_range = mx.nd.array([-real_range], dtype=np.float32)
    max_range = mx.nd.array([real_range], dtype=np.float32)
    data = mx.nd.contrib.dequantize(qdata, min_range, max_range, out_type='float32')
    quantized_range = 127.0
    scale = real_range / quantized_range
    assert data.dtype == np.float32
    data_np = qdata_np * scale
    assert_almost_equal(data.asnumpy(), data_np)


def test_requantize_int32_to_int8():
    def quantized_int32_to_float(qdata, min_range, max_range):
        assert qdata.dtype == 'int32'
        quantized_range = np.iinfo('int32').max
        real_range = np.maximum(np.abs(min_range), np.abs(max_range))
        scale = float(real_range) / float(quantized_range)
        return qdata.astype('float32') * scale

    def float_to_quantized_int8(data, min_range, max_range):
        assert data.dtype == 'float32'
        real_range = np.maximum(np.abs(min_range), np.abs(max_range))
        quantized_range = np.iinfo('int8').max
        scale = float(quantized_range) / float(real_range)
        return (np.sign(data) * np.minimum(np.abs(data) * scale + 0.5, quantized_range)).astype('int8')

    def requantize(qdata, min_data, max_data, real_range):
        data = quantized_int32_to_float(qdata, min_data, max_data)
        output = float_to_quantized_int8(data, -real_range, real_range)
        return output, -real_range, real_range

    def requantize_baseline(qdata, min_data, max_data, min_calib_range=None, max_calib_range=None):
        if min_calib_range is not None and max_calib_range is not None:
            real_range = np.maximum(np.abs(min_calib_range), np.abs(max_calib_range))
            return requantize(qdata, min_data, max_data, real_range)
        else:
            min_range = quantized_int32_to_float(np.min(qdata), min_data, max_data)
            max_range = quantized_int32_to_float(np.max(qdata), min_data, max_data)
            return requantize(qdata, min_data, max_data, np.maximum(np.abs(min_range), np.abs(max_range)))

    def check_requantize(shape, min_calib_range=None, max_calib_range=None):
        qdata = mx.nd.random.uniform(low=-1000.0, high=1000.0, shape=shape).astype('int32')
        min_range = mx.nd.array([-1010.0])
        max_range = mx.nd.array([1020.0])
        if min_calib_range is None or max_calib_range is None:
            qdata_int8, min_output, max_output = mx.nd.contrib.requantize(qdata, min_range, max_range)
        else:
            qdata_int8, min_output, max_output = mx.nd.contrib.requantize(qdata, min_range, max_range,
                                                                          min_calib_range, max_calib_range)

        qdata_int8_np, min_output_np, max_output_np = requantize_baseline(qdata.asnumpy(), min_range.asscalar(),
                                                                          max_range.asscalar(),
                                                                          min_calib_range=min_calib_range,
                                                                          max_calib_range=max_calib_range)
        assert_almost_equal(qdata_int8.asnumpy(), qdata_int8_np)
        assert_almost_equal(min_output.asnumpy(), np.array([min_output_np]))
        assert_almost_equal(max_output.asnumpy(), np.array([max_output_np]))

    check_requantize((3, 4, 10, 10))
    check_requantize((32, 3, 23, 23))
    check_requantize((3, 4, 10, 10), min_calib_range=-1050.0, max_calib_range=1040.0)
    check_requantize((32, 3, 23, 23), min_calib_range=-134.349, max_calib_range=523.43)


def test_quantized_conv():
    if mx.current_context().device_type != 'cpu':
        return

    def check_quantized_conv(data_shape, kernel, num_filter, pad, stride, no_bias):
        with mx.Context('cpu', 0):
            # run fp32 conv
            data = mx.sym.Variable(name='data', shape=data_shape, dtype='float32')
            conv2d = mx.sym.Convolution(data=data, kernel=kernel, num_filter=num_filter, pad=pad, stride=stride,
                                        no_bias=no_bias, cudnn_off=False, name='conv2d')
            arg_shapes, _, _ = conv2d.infer_shape(data=data_shape)
            arg_names = conv2d.list_arguments()
            conv_exe_fp32 = conv2d.simple_bind(ctx=mx.current_context(), grad_req='null')
            conv_exe_fp32.arg_dict[arg_names[0]][:] = mx.nd.random.uniform(low=0, high=127.0,
                                                                           shape=data_shape).astype('int32')
            conv_exe_fp32.arg_dict[arg_names[1]][:] = mx.nd.random.uniform(low=-127.0, high=127.0,
                                                                           shape=arg_shapes[1]).astype('int32')
            if not no_bias:
                conv_exe_fp32.arg_dict[arg_names[2]][:] = mx.nd.random.uniform(low=-127.0, high=127.0,
                                                                               shape=arg_shapes[2]).astype('int32')
            output = conv_exe_fp32.forward()[0]

            # run quantized conv
            qdata = mx.sym.Variable(name='qdata', shape=data_shape, dtype='uint8')
            qweight = mx.sym.Variable(name='qweight', dtype='int8')
            min_data = mx.sym.Variable(name='min_data')
            max_data = mx.sym.Variable(name='max_data')
            min_weight = mx.sym.Variable(name='min_weight')
            max_weight = mx.sym.Variable(name='max_weight')
            quantized_conv2d = mx.sym.contrib.quantized_conv(data=qdata, weight=qweight, min_data=min_data,
                                                             max_data=max_data, min_weight=min_weight,
                                                             max_weight=max_weight, kernel=kernel,
                                                             num_filter=num_filter, pad=pad, stride=stride,
                                                             no_bias=no_bias)
            qarg_names = quantized_conv2d.list_arguments()
            type_dict = None
            if not no_bias:
                type_dict = {qarg_names[2]: 'int8'}
            conv_exe_int8 = quantized_conv2d.simple_bind(ctx=mx.current_context(), type_dict=type_dict, grad_req='null')
            conv_exe_int8.arg_dict[qarg_names[0]][:] = conv_exe_fp32.arg_dict[arg_names[0]].astype('uint8')
            conv_exe_int8.arg_dict[qarg_names[1]][:] = conv_exe_fp32.arg_dict[arg_names[1]].astype('int8')
            quantized_range = 127.0
            if no_bias:
                conv_exe_int8.arg_dict[qarg_names[2]][:] = -quantized_range
                conv_exe_int8.arg_dict[qarg_names[3]][:] = quantized_range
                conv_exe_int8.arg_dict[qarg_names[4]][:] = -quantized_range
                conv_exe_int8.arg_dict[qarg_names[5]][:] = quantized_range
            else:
                conv_exe_int8.arg_dict[qarg_names[2]][:] = conv_exe_fp32.arg_dict[arg_names[2]].astype('int8')
                conv_exe_int8.arg_dict[qarg_names[3]][:] = -quantized_range
                conv_exe_int8.arg_dict[qarg_names[4]][:] = quantized_range
                conv_exe_int8.arg_dict[qarg_names[5]][:] = -quantized_range
                conv_exe_int8.arg_dict[qarg_names[6]][:] = quantized_range
                conv_exe_int8.arg_dict[qarg_names[7]][:] = -quantized_range
                conv_exe_int8.arg_dict[qarg_names[8]][:] = quantized_range
            qoutput, min_range, max_range = conv_exe_int8.forward()

            if no_bias:
                assert_almost_equal(output.asnumpy(), qoutput.asnumpy())
            else:
                # with adding bias, accuracy loss should not be greater than one
                diff = mx.nd.abs(output - qoutput.astype(output.dtype))
                cond = mx.nd.lesser(2, diff).sum().asscalar()
                assert cond == 0

    check_quantized_conv((3, 4, 28, 28), (3, 3), 128, (1, 1), (1, 1), True)
    check_quantized_conv((3, 4, 28, 28), (3, 3), 128, (1, 1), (1, 1), False)

def test_quantized_pooling():
    if mx.current_context().device_type != 'cpu':
        return

    def check_quantized_pooling(data_shape, kernel, pool_type, pad, stride, global_pool):
        with mx.Context('cpu', 0):
            data = mx.sym.Variable(name='data', shape=data_shape, dtype='float32')
            pooling_fp32 = mx.sym.Pooling(data=data, kernel=kernel, pad=pad, stride=stride,
                                          pool_type=pool_type, global_pool=global_pool, cudnn_off=False)
            arg_shapes, _, _ = pooling_fp32.infer_shape(data=data_shape)
            arg_names = pooling_fp32.list_arguments()
            pooling_fp32_exe = pooling_fp32.simple_bind(ctx=mx.current_context(), grad_req='null')
            pooling_fp32_exe.arg_dict[arg_names[0]][:] = mx.nd.random.uniform(low=0, high=127.0,
                                                                              shape=data_shape).astype('int32')
            output = pooling_fp32_exe.forward()[0]

            qdata = mx.sym.Variable(name='qdata', shape=data_shape, dtype='uint8')
            min_data = mx.sym.Variable(name='min_data')
            max_data = mx.sym.Variable(name='max_data')
            quantized_pooling = mx.sym.contrib.quantized_pooling(data=qdata, min_data=min_data,
                                                                 max_data=max_data, kernel=kernel,
                                                                 pad=pad, stride=stride, pool_type=pool_type,
                                                                 global_pool=global_pool)
            pooling_int8_exe = quantized_pooling.simple_bind(ctx=mx.current_context(), grad_req='null')
            qarg_names = quantized_pooling.list_arguments()
            pooling_int8_exe.arg_dict[qarg_names[0]][:] = pooling_fp32_exe.arg_dict[arg_names[0]].astype('uint8')
            quantized_range = 127.0
            pooling_int8_exe.arg_dict[qarg_names[1]][:] = -quantized_range
            pooling_int8_exe.arg_dict[qarg_names[2]][:] = quantized_range
            qoutput, min_range, max_range = pooling_int8_exe.forward()

            if pool_type == 'max':
                assert_almost_equal(output.asnumpy(), qoutput.asnumpy())
            elif pool_type == 'avg':  # for avg pooling, fp32 and int8 may be different due to rounding errors
                diff = mx.nd.abs(output - qoutput.astype(output.dtype))
                cond = mx.nd.lesser(2, diff).sum().asscalar()
                assert cond == 0

    check_quantized_pooling((3, 4, 56, 56), (3, 3), 'max', (0, 0), (2, 2), False)
    check_quantized_pooling((3, 4, 56, 56), (3, 3), 'max', (0, 0), (2, 2), True)
    check_quantized_pooling((3, 512, 7, 7), (7, 7), 'avg', (0, 0), (1, 1), False)
    check_quantized_pooling((3, 512, 7, 7), (7, 7), 'avg', (0, 0), (1, 1), True)


def test_quantized_flatten():
    def check_quantized_flatten(shape):
        qdata = mx.nd.random.uniform(low=0, high=127, shape=shape).astype('uint8')
        min_data = mx.nd.array([-1023.343], dtype='float32')
        max_data = mx.nd.array([2343.324275], dtype='float32')
        qoutput, min_output, max_output = mx.nd.contrib.quantized_flatten(qdata, min_data, max_data)
        assert qoutput.ndim == 2
        assert qoutput.shape[0] == qdata.shape[0]
        assert qoutput.shape[1] == np.prod(qdata.shape[1:])
        assert same(qdata.asnumpy().flatten(), qoutput.asnumpy().flatten())
        assert same(min_data.asnumpy(), min_output.asnumpy())
        assert same(max_data.asnumpy(), max_output.asnumpy())

    check_quantized_flatten((10,))
    check_quantized_flatten((10, 15))
    check_quantized_flatten((10, 15, 18))
    check_quantized_flatten((3, 4, 23, 23))


def test_quantize_params():
    data = mx.sym.Variable('data')
    conv = mx.sym.Convolution(data, kernel=(1, 1), num_filter=2048, name='conv')
    sym = mx.sym.BatchNorm(data=conv, eps=2e-05, fix_gamma=False, momentum=0.9, use_global_stats=False, name='bn')
    offline_params = [name for name in sym.list_arguments()
                      if not name.startswith('data') and not name.endswith('label')]
    params = {}
    for name in offline_params:
        params[name] = mx.nd.uniform(shape=(2, 2))
    qsym = mx.contrib.quantization._quantize_symbol(sym, offline_params=offline_params, context=mx.current_context())
    qparams = mx.contrib.quantization._quantize_params(qsym, params)
    param_names = params.keys()
    qparam_names = qparams.keys()
    for name in qparam_names:
        if name.startswith('bn'):
            assert name in param_names
        elif name.startswith('conv'):
            assert name not in param_names
            assert name.find('quantize') != -1


def test_quantize_sym_with_calib():
    data = mx.sym.Variable('data')
    conv = mx.sym.Convolution(data, kernel=(1, 1), num_filter=2048, name='conv')
    bn = mx.sym.BatchNorm(data=conv, eps=2e-05, fix_gamma=False, momentum=0.9, use_global_stats=False, name='bn')
    act = mx.sym.Activation(data=bn, act_type='relu', name='relu')
    pool = mx.sym.Pooling(act, kernel=(7, 7), pool_type='avg', name='pool')
    fc = mx.sym.FullyConnected(pool, num_hidden=1000, flatten=True, name='fc')
    sym = mx.sym.SoftmaxOutput(fc, grad_scale=1, ignore_label=-1, multi_output=False,
                               out_grad=False, preserve_shape=False, use_ignore=False, name='softmax')
    offline_params = [name for name in sym.list_arguments()
                      if not name.startswith('data') and not name.endswith('label')]
    qsym = mx.contrib.quantization._quantize_symbol(sym, offline_params=offline_params, context=mx.current_context())
    requantize_op_names = ['requantize_conv']
    th_dict = {'conv_output': (np.random.uniform(low=100.0, high=200.0), np.random.uniform(low=100.0, high=200.0))}
    op_name_to_th_name = {'requantize_conv': 'conv_output'}
    cqsym = mx.contrib.quantization._calibrate_quantized_sym(qsym, th_dict)
    attr_dict = cqsym.attr_dict()
    for name in requantize_op_names:
        assert name in attr_dict
        lhs = float(attr_dict[name]['min_calib_range'])
        rhs = th_dict[op_name_to_th_name[name]][0]
        assert_almost_equal(np.array([lhs]), np.array([rhs]))
        lhs = float(attr_dict[name]['max_calib_range'])
        rhs = th_dict[op_name_to_th_name[name]][1]
        assert_almost_equal(np.array([lhs]), np.array([rhs]), rtol=1e-3, atol=1e-4)


def test_get_optimal_thresholds():
    """Given an ndarray with elements following a uniform distribution,
    the optimal threshold for quantizing the ndarray should be either
    abs(min(nd)) or abs(max(nd))."""
    def get_threshold(nd):
        min_nd = mx.nd.min(nd)
        max_nd = mx.nd.max(nd)
        return mx.nd.maximum(mx.nd.abs(min_nd), mx.nd.abs(max_nd)).asnumpy()

    nd_dict = {'layer1': mx.nd.uniform(low=-10.532, high=11.3432, shape=(8, 3, 23, 23))}
    expected_threshold = get_threshold(nd_dict['layer1'])
    th_dict = mx.contrib.quantization._get_optimal_thresholds(nd_dict)
    assert 'layer1' in th_dict
    assert_almost_equal(np.array([th_dict['layer1'][1]]), expected_threshold)


if __name__ == "__main__":
    import nose
    nose.runmodule()
