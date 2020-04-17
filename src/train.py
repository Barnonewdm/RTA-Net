# python imports
import os
import glob
import sys
import random
from argparse import ArgumentParser
import datetime

# third-party imports
import tensorflow as tf
import numpy as np
import keras
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model 
from keras.models import Model
from keras.layers import Input

# project imports
import datagenerators_user as datagenerators
import networks
import losses
import losses_user

sys.path.append('../ext/neuron')
import neuron.callbacks as nrn_gen
# Training dataset directory
base_data_dir = '../../data/MAPS_Dataset_Affined/'
train_set = 'Train_Set'
train_vol_names = glob.glob(base_data_dir + train_set +'/*.img')
train_vol_names = sorted(train_vol_names)

# It is better to assign
vol_size = (256, 256, 256)

def train(model_dir,
          gpu_id,
          lr,
          nb_epochs,
          prior_lambda,
          image_sigma,
          steps_per_epoch,
          batch_size,
          load_model_file,
          bidir,
          initial_epoch=0):
    """
    model training function
    :param data_dir: folder with npz files for each subject.
    :param model_dir: model folder to save to
    :param gpu_id: integer specifying the gpu to use
    :param lr: learning rate
    :param nb_epochs: number of training iterations
    :param prior_lambda: the prior_lambda, the scalar in front of the smoothing laplacian, in MICCAI paper
    :param image_sigma: the image sigma in MICCAI paper
    :param steps_per_epoch: frequency with which to save models
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param load_model_file: optional h5 model file to initialize with
    :param bidir: logical whether to use bidirectional cost function
    """
    

    # Diffeomorphic network architecture used in VoxelMorph MICCAI 2018 paper
    nf_enc = [16,32,32,32]
    nf_dec = [32,32,32,32,16,3]

    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # gpu handling
    gpu = '/gpu:%d' % 0 # gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # prepare the model
    with tf.device(gpu):
        model = networks.Reg_Net(vol_size, nf_enc, nf_dec, bidir=bidir)
        flow = model.get_layer('flow-int').output
        [src, tgt] = model.inputs
        [y, flow_parameters] = model.outputs
        model = Model(inputs=[src, tgt], outputs=[y, flow_parameters, flow])
        # load initial weights
        if load_model_file is not None and load_model_file != "":
            model.load_weights(load_model_file)

        # save first iteration
        model.save(os.path.join(model_dir, '%02d.h5' % initial_epoch))

        # compile
        # note: best to supply vol_shape here than to let tf figure it out.
        flow_vol_shape = model.outputs[-2].shape[1:-1]
        loss_class = losses.Miccai2018(image_sigma, prior_lambda, flow_vol_shape=flow_vol_shape)
        if bidir:
            model_losses = [loss_class.recon_loss, loss_class.recon_loss, loss_class.kl_loss]
            loss_weights = [0.5, 0.5, 1]
        else:
            model_losses = [loss_class.recon_loss, loss_class.kl_loss, 
                            losses_user.Sparse_Loss(
                                                    tissue=src, loss_weights=1, 
                                                    ).loss]
            loss_weights = [1, 1, 1]
        
    
    # data generator
    nb_gpus = len(gpu_id.split(','))
    assert np.mod(batch_size, nb_gpus) == 0, \
        'batch_size should be a multiple of the nr. of gpus. ' + \
        'Got batch_size %d, %d gpus' % (batch_size, nb_gpus)

    data_gen = datagenerators.data_generator_vertices(train_vol_names, vol_size)

    # prepare callbacks
    save_file_name = os.path.join(model_dir, '{epoch:02d}.h5')
    log_dir = "./logs/"+ model_dir[9:] + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=0)
    # fit generator
    with tf.device(gpu):

        # multi-gpu support
        if nb_gpus > 1:
            save_callback = nrn_gen.ModelCheckpointParallel(save_file_name)
            mg_model = multi_gpu_model(model, gpus=nb_gpus)
        
        # single gpu
        else:
            save_callback = ModelCheckpoint(save_file_name)
            mg_model = model

        mg_model.compile(optimizer=Adam(lr=lr), loss=model_losses, loss_weights=loss_weights)
        mg_model.fit_generator(data_gen, 
                               initial_epoch=initial_epoch,
                               epochs=nb_epochs,
                               callbacks=[save_callback, tensorboard_callback],
                               steps_per_epoch=steps_per_epoch,
                               verbose=1)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default='../models/',
                        help="models folder")
    parser.add_argument("--gpu", type=str, default=0,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=1e-4, help="learning rate")
    parser.add_argument("--epochs", type=int,
                        dest="nb_epochs", default=1500,
                        help="number of iterations")
    parser.add_argument("--prior_lambda", type=float,
                        dest="prior_lambda", default=20,
                        help="prior_lambda regularization parameter")
    parser.add_argument("--image_sigma", type=float,
                        dest="image_sigma", default=0.02,
                        help="image noise parameter")
    parser.add_argument("--steps_per_epoch", type=int,
                        dest="steps_per_epoch", default=100,
                        help="frequency of model saves")
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=1,
                        help="batch_size")
    parser.add_argument("--load_model_file", type=str,
                        dest="load_model_file", default='',
                        help="optional h5 model file to initialize with")
    parser.add_argument("--bidir", type=int,
                        dest="bidir", default=0,
                        help="whether to use bidirectional cost function")
    parser.add_argument("--initial_epoch", type=int,
                        dest="initial_epoch", default=0,
                        help="first epoch")

    args = parser.parse_args()
    train(**vars(args))
