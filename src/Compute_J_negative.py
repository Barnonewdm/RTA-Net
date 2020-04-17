#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:57:48 2019

@author: weidongming
"""
import tensorflow as tf
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras.backend as K

def compute_neg_Jacobian(field_file, mask=None):
    field = sitk.ReadImage(field_file)
    J_f = sitk.DisplacementFieldJacobianDeterminant(field)
    J_f_data = sitk.GetArrayFromImage(J_f)
    ROI_volume = np.size(J_f_data)
    if mask is not None and mask != '':
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask))
        mask = np.greater(mask, 0)
        J_f_data = mask*J_f_data

        ROI_volume = mask
        for i in range(len(np.shape(mask))):
            ROI_volume = sum(ROI_volume)


    hist = plt.hist(J_f_data.flatten(), bins=300)
    if hist[1].min()<0:
        k = np.where(hist[1]<0)
        num = np.sum(hist[0][:k[0][-1]])
        return num, 100 * num/ROI_volume
    else: 
        return 0

if __name__ == '__main__':
    import sys
    print('[SimpleITK]:')
    print(compute_neg_Jacobian(sys.argv[1]))
    if sys.argv[2] is not None and sys.argv[2] != '':
        print('[SimpleITK masked]:')
        print(compute_neg_Jacobian(sys.argv[1], sys.argv[2]))

