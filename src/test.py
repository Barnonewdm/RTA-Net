"""
Test models
"""

# py imports
import os
import sys
import glob

# third party
import tensorflow as tf
import scipy.io as sio
import numpy as np
import keras
from keras.backend.tensorflow_backend import set_session
from scipy.interpolate import interpn

# project
sys.path.append('../ext/medipy-lib')
sys.path.append('../ext/neuron')
sys.path.append('../ext/pynd-lib')
sys.path.append('../ext/pytools-lib')
import neuron.utils as nrn_utils
import neuron.layers as nrn_layers
import medipy
import networks
from networks import trf_resize
from networks import Log_Exp
# import util
from medipy.metrics import dice
import SimpleITK as sitk
# read fix, moving images, and moving label
n_batches = 1
FIX_NAME = sys.argv[4]    
fix = sitk.ReadImage(FIX_NAME)
fix_data = sitk.GetArrayFromImage(fix)
fix_data = np.float32(fix_data)
fix_data = fix_data/np.float(np.max(fix_data))
fix_data = np.reshape(fix_data, (1,) + fix_data.shape + (1,))
MOV_NAME = sys.argv[5]
mov = sitk.ReadImage(MOV_NAME)
mov_data = sitk.GetArrayFromImage(mov)
mov_data = np.float32(mov_data)/np.float(np.max(mov_data))
mov_data = np.reshape(mov_data,(1,) + mov_data.shape + (1,))
atlas_vol = fix_data
atlas_seg = fix_data
LABEL_NAME = sys.argv[6]
mov_label = sitk.GetArrayFromImage(sitk.ReadImage(LABEL_NAME))
mov_label = np.reshape(mov_label, (1,) + mov_label.shape + (1,))
X_vol = mov_data
X_seg = mov_data
X_label = mov_label

def test(gpu_id, model_dir, iter_num, 
         compute_type = 'GPU',  # GPU or CPU
         vol_size=(256,256,256),
         nf_enc=[16,32,32,32],
         nf_dec=[32,32,32,32,16,3],
         save_file=None):

    # GPU handling
    gpu = '/gpu:' + str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # load weights of model
    with tf.device(gpu):
        net = networks.Reg_Net(vol_size, nf_enc, nf_dec, use_miccai_int=False, indexing='ij', stage='test')  
        net.load_weights(os.path.join(model_dir, str(iter_num) + '.h5'))

        # compose diffeomorphic flow output model
        diff_net = keras.models.Model(net.inputs, net.get_layer('diffflow').output)

        # NN transfer model
        nn_trf_model = networks.nn_trf(vol_size, indexing='ij')
        
        # compose log model
        log_net = keras.models.Model(net.inputs, net.get_layer('log_sigma').output)
        log = log_net.output
        log = Log_Exp()(log)
        log_rescale = nrn_layers.VecInt(method='ss', name='flow-int', int_steps=7)(log)
        log_upsample = trf_resize(log_rescale, 1/2, name='diffflow')
        log_flow_net = keras.models.Model(net.inputs, log_upsample) 
    # if CPU, prepare grid
    if compute_type == 'CPU':
        grid, xx, yy, zz = util.volshape2grid_3d(vol_size, nargout=4)
    
    for k in range(n_batches):

        # predict transform
        with tf.device(gpu):
            pred = diff_net.predict([X_vol, atlas_vol])
            log = log_flow_net.predict([X_vol, atlas_vol])
        # Warp segments with flow
        if compute_type == 'CPU':
            flow = pred[0, :, :, :, :]
            warp_seg = util.warp_seg(X_seg, flow, grid=grid, xx=xx, yy=yy, zz=zz)

        else:  # GPU
            warp_label = nn_trf_model.predict([X_label, pred])[0,...,0]
            warp_seg = nn_trf_model.predict([X_seg, pred])[0,...,0]
            flow = pred[0, ...]
            log = log[0, ...]

        result_seg = sitk.GetImageFromArray(warp_seg*250)
        result_seg.CopyInformation(mov)
        sitk.WriteImage(result_seg,'../data/baseline_without_surface/warped_seg.nii.gz')
        result_label = sitk.GetImageFromArray(warp_label)
        result_label.CopyInformation(mov)
        sitk.WriteImage(result_label, '../data/baseline_without_surface/warped_label.nii.gz')
        #compute deformation field
        result_field = flow
        result_field = sitk.GetImageFromArray(result_field)
        result_field.CopyInformation(mov)
        sitk.WriteImage(result_field,'../data/baseline_without_surface/field.mha')

        #compute the log
        log = sitk.GetImageFromArray(log)
        sitk.WriteImage(log, '../data/baseline_without_surface/log.mha')

if __name__ == "__main__":
    """
    python test.py gpu_id model_dir iter_num fix.nii.gz moving.nii.gz moving_label.nii.gz
    """
    test(sys.argv[1], sys.argv[2], sys.argv[3])
