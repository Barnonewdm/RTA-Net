#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 13:51:26 2019
Example: python DNN_Flow_Warped.py gpu moving.nii field.mha
@author: weidongming
"""

import tensorflow as tf
import SimpleITK as sitk
import networks
import numpy as np
import os
import sys
sys.path.append('../ext/medipy-lib')
from keras.backend.tensorflow_backend import set_session

MOV_NAME = sys.argv[2]
mov = sitk.ReadImage(MOV_NAME)
mov_data = sitk.GetArrayFromImage(mov)
mov_data = np.reshape(mov_data,(1,) + mov_data.shape + (1,))
X_seg = mov_data
n_batches = 1

def test(gpu_id, compute_type='GPU', vol_size = (256,256,256), flow=None):
    gpu = '/gpu:' + str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))
    flow = sitk.GetArrayFromImage(sitk.ReadImage(flow))
    flow = np.reshape(flow, (1,) + np.shape(flow))
    flow_tensor = tf.convert_to_tensor(flow)
    print(flow_tensor.shape)
    # load weights of model
    with tf.device(gpu):
        nn_trf_model = networks.nn_trf(vol_size, indexing='ij')
        warped_seg = nn_trf_model.predict([X_seg, flow])[0,...,0]
        print(warped_seg.shape)
        
    result_seg = sitk.GetImageFromArray(warped_seg)
    result_seg.CopyInformation(mov)
    sitk.WriteImage(result_seg,'../data/results/warped_seg.nii.gz')
        
        
if __name__=="__main__":
    test(sys.argv[1], flow = sys.argv[3])
