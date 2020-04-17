#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 17:00:44 2019

@author: weidongming
"""
from .utils import Gaussian_Smooth, Voxel_wise_Gaussian_Smooth
from keras.layers import Layer
from keras import backend as K
import tensorflow as tf

class GaussianSmoother(Layer):
    """
    """

    def __init__(self, interp_method='linear', indexing='ij', **kwargs):
        """
        Parameters: 
            interp_method: 'linear' or 'nearest'
            indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian)
                'xy' indexing will have the first two entries of the flow 
                (along last axis) flipped compared to 'ij' indexing
        """
        self.ndims = None
        self.inshape = None

        assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing

        super(self.__class__, self).__init__(**kwargs)


    def build(self, input_shape):
        """
        input_shape should be a list for two inputs:
        input1: deformation field.
        input2: gaussian fileter size
            if affine:
                should be a N+1 x N+1 matrix
                *or* a N*N+1 tensor (which will be reshape to N x (N+1) and an identity row added)
            if not affine:
                should be a *vol_shape x N
        """

        if len(input_shape) > 5:
            print(len(input_shape))
            raise Exception('Spatial Transformer must be called on a list of length 2.'
                            'First argument is the image, second is the transform.')
        

        # set up number of dimensions
        self.ndims = len(input_shape) - 2
        self.inshape = input_shape
#        vol_shape = input_shape[0][1:-1]
#        kernen_size_shape = input_shape[1][1:]

        
        # confirm built
        self.built = True


    def call(self, inputs):
        """
        Parameters
            inputs: list with two entries
        """

        # check shapes
        assert len(inputs) == 2, "inputs has to be len 2, found: %d" % len(inputs)
        field = inputs[0]
        field._keras_shape = inputs[0]._keras_shape
        log = tf.exp(inputs[1])
        log._keras_shape = inputs[1]._keras_shape
#        kernel_size = inputs[1]

        # necessary for multi_gpu models...
        field = K.reshape(field, [-1, *self.inshape[0][1:]])
        log = K.reshape(log, [-1, *self.inshape[1][1:]])
#        kernel_size = tf.Variable(3.) #K.reshape(kernel_size, [-1, *self.inshape[1][1:]])

        # map transform across batch
        '''
        return tf.map_fn(self._gaussian_smooth, [field], dtype=tf.float32)
        '''
        return tf.concat([
            tf.map_fn(self._gaussian_smooth, [field[...,0:1], log], dtype=tf.float32),
            tf.map_fn(self._gaussian_smooth, [field[...,1:2], log], dtype=tf.float32),
            tf.map_fn(self._gaussian_smooth, [field[...,2:],  log], dtype=tf.float32)], 
            axis=-1)[0,...]

    def compute_output_shape(self, input_shape):
        return input_shape
  
    def _gaussian_smooth(self, inputs):
        inputs[0] = tf.expand_dims(inputs[0], axis=0)
        #return  Gaussian_Smooth(inputs[0], size=3, std=)
        return Voxel_wise_Gaussian_Smooth(inputs[0], size=3, log=inputs[1])
