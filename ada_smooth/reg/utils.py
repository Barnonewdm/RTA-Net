#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:27:17 2019

@author: weidongming
"""
import tensorflow as tf
import keras.backend as K
#https://stackoverflow.com/questions/52012657/how-to-make-a-2d-gaussian-filter-in-tensorflow/52012658    
    
def gaussian_kernel(size: int,
                    mean: float,
                    std: float,
                   ):
    """Makes 3D gaussian Kernel for convolution."""
    #if std<=0:
     #   std = float(size)/4
    d = tf.distributions.Normal(mean, std)

    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))

    gauss_kernel = tf.einsum('i,j,k->ijk',
                                  vals,
                                  vals,
                                  vals)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)


def Gaussian_Smooth(image, size=1, mean=0., std=1.):
    # Make Gaussian Kernel with desired specs.
    gauss_kernel = gaussian_kernel(size=size, mean=mean, std=std)
    
    # Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature.
    gauss_kernel = gauss_kernel[ :, :, :, tf.newaxis, tf.newaxis]
    
    # Convolve.
    smoothed_image = tf.nn.conv3d(image, gauss_kernel, strides=[1, 1, 1, 1, 1], padding="SAME")
    
    return smoothed_image

#https://github.com/NVlabs/pacnet/blob/master/pac.py
def Voxel_wise_Gaussian_Smooth(image, size=7, mean=0., std=3., log=None, strides=[1, 1, 1, 1, 1]):
    #define the gaussian filter
    gaussa_kernel = gaussian_kernel(size=size, mean=mean, std=std)
    gaussa_kernel = gaussa_kernel[:,:,:,tf.newaxis, tf.newaxis]
    if log!=None:
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
        image_padded = tf.pad(image, paddings=paddings, mode="CONSTANT")
        # generate the masks and stds
        [mask_1, mask_2, mask_3, std_1, std_2, std_3] = Generate_Mask(log)
        
        gauss_kernel_1 = gaussian_kernel(size=size, mean=mean, std=std_1)
        gauss_kernel_1 = gauss_kernel_1[:,:,:, tf.newaxis, tf.newaxis]
        gauss_kernel_2 = gaussian_kernel(size=size, mean=mean, std=std_2)
        gauss_kernel_2 = gauss_kernel_2[:,:,:, tf.newaxis, tf.newaxis]
        gauss_kernel_3 = gaussian_kernel(size=size, mean=mean, std=std_3)
        gauss_kernel_3 = gauss_kernel_3[:,:,:, tf.newaxis, tf.newaxis]
        #smooth DF
        smoothed_image_1 = tf.nn.conv3d(image, gauss_kernel_1, strides=strides, padding="SAME")
        smoothed_image_2 = tf.nn.conv3d(image, gauss_kernel_2, strides=strides, padding="SAME")
        smoothed_image_3 = tf.nn.conv3d(image, gauss_kernel_3, strides=strides, padding="SAME")
        #concatenate to a single DF
        smoothed_image = tf.add(tf.multiply(smoothed_image_1, mask_1), tf.multiply(smoothed_image_2, mask_2))
        smoothed_image = tf.add(smoothed_image, tf.multiply(smoothed_image_3, mask_3))
    if log == None:    
        smoothed_image = tf.nn.conv3d(image, gaussa_kernel, strides=strides, padding="SAME")
    
    return smoothed_image

def Generate_Mask(log):
    log = tf.exp(log/2.0)
    # choose the maximum of uncertainty vector
    log_norm = tf.reduce_max(log, axis=-1)
    # For simplification, thress thresholds are generated.
    std_1 = (2.*tf.reduce_min(log_norm) + tf.reduce_max(log_norm))/3.
    std_2 = (tf.reduce_min(log_norm) + 2.*tf.reduce_max(log_norm))/3.
    std_3 = (tf.reduce_max(log_norm))
    # generate the masks
    mask_1 = K.cast(K.less_equal(log_norm, std_1), K.floatx())
    mask_2 = K.cast(K.greater(log_norm, std_1), K.floatx())
    mask_i = K.cast(K.less_equal(log_norm, std_2), K.floatx())
    mask_2 = tf.multiply(mask_2, mask_i)
    mask_3 = K.cast(K.greater(log_norm, std_2), K.floatx())
    # adapt axis 
    mask_1 = mask_1[ tf.newaxis, :,:,:, tf.newaxis]
    mask_2 = mask_2[ tf.newaxis, :,:,:, tf.newaxis]
    mask_3 = mask_3[ tf.newaxis, :,:,:, tf.newaxis]
    
    return [mask_1, mask_2, mask_3, std_1, std_2, std_3]


