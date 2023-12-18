from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def CuDNN_Batch_Normalization(in_arr, bn_scale_arr, bn_bias_arr, out_arr, save_mean, save_var, running_mean, running_var, momentum=0.99, eps=0.001, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(bn_scale_arr, _nd.NDArray)
    assert isinstance(bn_bias_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.CuDNN_DLGpuBatch_Normalization(in_arr.handle, bn_scale_arr.handle, bn_bias_arr.handle,
                                        out_arr.handle, ctypes.c_float(momentum), ctypes.c_float(eps), save_mean.handle, save_var.handle, running_mean.handle, running_var.handle, stream.handle if stream else None)


def CuDNN_Batch_Normalization_gradient(in_gradient_y, in_arr_x, in_bn_scale, out_gradient_x, out_gradient_bn_scale, out_gradient_bn_bias, running_mean, running_var, eps=0.001, stream=None):
    assert isinstance(in_gradient_y, _nd.NDArray)
    assert isinstance(in_arr_x, _nd.NDArray)
    assert isinstance(in_bn_scale, _nd.NDArray)
    assert isinstance(out_gradient_x, _nd.NDArray)
    assert isinstance(out_gradient_bn_scale, _nd.NDArray)
    assert isinstance(out_gradient_bn_bias, _nd.NDArray)
    _LIB.CuDNN_DLGpuBatch_Normalization_gradient(in_gradient_y.handle, in_arr_x.handle, in_bn_scale.handle,
                                                 out_gradient_x.handle, out_gradient_bn_scale.handle, out_gradient_bn_bias.handle, ctypes.c_float(eps), running_mean.handle, running_var.handle, stream.handle if stream else None)


def CuDNN_Batch_Normalization_inference(in_arr, bn_scale_arr, bn_bias_arr, out_arr, save_mean, save_var, eps=0.001, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(bn_scale_arr, _nd.NDArray)
    assert isinstance(bn_bias_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.CuDNN_DLGpuBatch_Normalization_inference(in_arr.handle, bn_scale_arr.handle, bn_bias_arr.handle,
                                                  out_arr.handle, ctypes.c_float(eps), save_mean.handle, save_var.handle, stream.handle if stream else None)
