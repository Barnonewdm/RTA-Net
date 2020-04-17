"""
losses for Jac
"""


# Third party inports
import tensorflow as tf
import keras.backend as K
import numpy as np
#import point_cloud_utils as pcu
#import sys
#sys.path.append('../ext/neuron')
#import neuron.utils as nrn_utils
# For surface displacement smoothness and tissue Jacobian Determinant
class Sparse_Loss():
    def __init__(self, penalty='l2', lapacian_matrix=0, vertices_1=1, vertices_fix=2, vertices_2=2, vertices_3=3, vertices_4=4,
                 tissue=0, mask_value_1=150, mask_value_2=250, loss_weights=1.0, mask_value_3=0., weights_for_each_vertice = None,
                 weights_dir='./dice_matrix.npy'):
        self.loss_weights = 1.0
        self.penalty = penalty
        self.L = lapacian_matrix
        self.vertices_1 = vertices_1
# =============================================================================
#         self.vertices_2 = vertices_2
#         self.vertices_3 = vertices_3
#         self.vertices_4 = vertices_4
# =============================================================================
        self.vertices_fix = vertices_fix
        self.tissue = tissue
        self.mask_value_1 = mask_value_1
        self.mask_value_2 = mask_value_2
        self.mask_value_3 = mask_value_3
        self.weights_for_displacment_mse = None
        self.weights_dir = weights_dir
        self.weights_for_each_vertice = weights_for_each_vertice
    
    #compute gradient of f_x f_y f_z
    def _diffs_for_jacobian(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y_new = K.permute_dimensions(y, r)
            dfi = y_new[:-2, ...] - y_new[2:, ...] #- tf.scalar_mul(2., y_new[1:-1,...])
#            tem = dfi[...,:1]
#            dfi = tf.scatter_add(dfi, indices=[...,i], 2.)
#            Error = tf.constant(2., shape=tem.shape)
#            Zero = tf.constant(0., shape=tem.shape)
            Error = tf.constant(2., shape=[vol_shape[0]-2, 1, vol_shape[0], vol_shape[0], 1])
            Zero = tf.constant(0., shape=[vol_shape[0]-2, 1, vol_shape[0], vol_shape[0], 1])
            if i == 0:
                Error = tf.concat([Error, Zero, Zero], axis=-1)
            if i == 1:
                Error = tf.concat([Zero, Error, Zero], axis=-1)
            if i == 2:
                Error = tf.concat([Zero, Zero, Error], axis=-1)
            dfi = tf.add(dfi, Error)
            dfi = tf.multiply(dfi, 0.5)
            
            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)
        
        return df
    
    def _jacobian(self, _, y_pred):
        flow = y_pred
        vol_shape = flow.get_shape().as_list()[1:-1]
        df = [d for d in self._diffs_for_jacobian(flow)]
        df_x = df[0]
        df_y = df[1]
        df_z = df[2]
        
        df_f_x = df_x[0,0:vol_shape[0]-2,1:vol_shape[1]-1,1:vol_shape[2]-1,0]
        df_g_x = df_x[0,0:vol_shape[0]-2,1:vol_shape[1]-1,1:vol_shape[2]-1,1]
        df_h_x = df_x[0,0:vol_shape[0]-2,1:vol_shape[1]-1,1:vol_shape[2]-1,2]
        
        df_f_y = df_y[0,1:vol_shape[0]-1,0:vol_shape[1]-2,1:vol_shape[2]-1,0]
        df_g_y = df_y[0,1:vol_shape[0]-1,0:vol_shape[1]-2,1:vol_shape[2]-1,1]
        df_h_y = df_y[0,1:vol_shape[0]-1,0:vol_shape[1]-2,1:vol_shape[2]-1,2]
        
        df_f_z = df_z[0,1:vol_shape[0]-1,1:vol_shape[1]-1,0:vol_shape[2]-2,0]
        df_g_z = df_z[0,1:vol_shape[0]-1,1:vol_shape[1]-1,0:vol_shape[2]-2,1]
        df_h_z = df_z[0,1:vol_shape[0]-1,1:vol_shape[1]-1,0:vol_shape[2]-2,2]

        tem_1 = tf.multiply(df_f_x, df_g_y)
        tem_1 = tf.multiply(tem_1, df_h_z)
        tem_2 = tf.multiply(df_f_y, df_g_z)
        tem_2 = tf.multiply(tem_2, df_h_x)
        tem_3 = tf.multiply(df_f_z, df_g_x)
        tem_3 = tf.multiply(tem_3, df_h_y)
        
        tem_4 = tf.multiply(df_f_x, df_g_z)
        tem_4 = tf.multiply(tem_4, df_h_y)
        tem_5 = tf.multiply(df_f_y, df_g_x)
        tem_5 = tf.multiply(tem_5, df_h_z)
        tem_6 = tf.multiply(df_f_z, df_g_y)
        tem_6 = tf.multiply(tem_6, df_h_x)
        
        J_D = tf.add(tem_1, tem_2)
        J_D = tf.add(J_D, tem_3)
        J_D = tf.subtract(J_D, tem_4)
        J_D = tf.subtract(J_D, tem_5)
        J_D = tf.subtract(J_D, tem_6)
        return J_D
    
    def Masked_Jacobian(self, y_truth, y_pred):
        tissue = tf.concat([self.tissue, self.tissue, self.tissue], axis=-1)
        J_D = self._jacobian(y_pred, y_pred)
        mask = K.cast(K.not_equal(tissue, self.mask_value_3), K.floatx())
        # resize mask
        mask = mask[:, ::2, ::2, ::2, 0]
        mask = mask[:, 1:-1, 1:-1, 1:-1]
        masked_J_D = tf.multiply(J_D, mask)
        mask = K.cast(K.equal(tissue, self.mask_value_3), K.floatx())
        mask = mask[:, ::2, ::2, ::2, 0]
        mask = mask[:, 1:-1, 1:-1, 1:-1]
        masked_J_D_0 = tf.multiply(J_D, mask)
#        num_of_neg = np.count_nonzero(sum(J<0))
        #mask_of_neg = K.cast(tf.less_equal(J_D, -0.001), K.floatx())
#        num_of_neg = index_of_neg.shape[0]
#        num_of_neg = num_of_neg.value
        distance_min = tf.exp(tf.subtract(tf.norm(tf.subtract(tf.reduce_min(masked_J_D), 1.)), 1.)) - 1.
        distance_mean =  tf.exp(10*(K.mean(K.square( tf.reduce_sum(masked_J_D_0) / tf.cast( tf.count_nonzero(mask), tf.float32) - 1.)))) - 1.
        #distance_neg = tf.exp(tf.reduce_max(tf.subtract(tf.abs(masked_J_D), masked_J_D))) - 1.
        #distance = tf.add(distance_min, distance_neg)
        #distance = tf.exp(tf.reduce_mean(tf.subtract(tf.abs(masked_J_D),(masked_J_D)))) - 1.0
        #distance = distance_min
        l = [tf.reduce_sum(distance_mean + distance_min)]
        
        return l
    
    def loss(self, y_truth, y_pred):
        # number of negative jacobian determinant of tissue (GM, WM)
        loss_tissue_jacobian = [tf.reduce_mean(tf.multiply(self.Masked_Jacobian(y_truth, y_pred), 1))]
        
        # final loss
        loss = loss_tissue_jacobian# + loss_surface_distance#+ loss_mse_vertices_displacement + loss_surface_distance #+ loss_surface_displacement_gradient
        
        return self.loss_weights * tf.add_n(loss) / len(loss)
        
