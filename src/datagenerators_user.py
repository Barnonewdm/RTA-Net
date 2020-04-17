'''
# to do list:
1. sparse loss

'''

import os
import numpy as np
import SimpleITK as sitk

def data_generator_vertices(seg_names, vol_size):
    L=0
    while True:
        # randomly read moving image
        idx_mov = np.random.randint(len(seg_names))
        mov = sitk.ReadImage(seg_names[idx_mov])
        mov_data = sitk.GetArrayFromImage(mov)
        mov_data = np.float32(np.reshape(mov_data, (1,) + mov_data.shape + (1,)))
        
        idx_fix = np.random.randint(len(seg_names))
        # randomly read fixed image, which is different with movin
        #while idx_fix == idx_mov:
        #    idx_fix = np.random.randint(len(seg_names))
        fix = sitk.ReadImage(seg_names[idx_fix])
        fix_data = sitk.GetArrayFromImage(fix)
        fix_data = np.float32(np.reshape(fix_data, (1,) + fix_data.shape + (1,)))
        
        # zero_flow
        zero_v = np.zeros((1, int(vol_size[0]/2), int(vol_size[1]/2), int(vol_size[2]/2), 3))
        zero_flow = np.zeros((1, vol_size[0], vol_size[1], vol_size[2], 3))
        
            
        yield [mov_data/250., fix_data/250.], [fix_data/250., zero_v, zero_flow]


        
        
