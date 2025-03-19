""" Tensorflow """
import tensorflow as tf

""" Import numpy """
import numpy as np

""" keras backend """
from tensorflow.keras import backend as K

""" Import wsbmr """
import wsbmr

# Snippet to parse loss 
def parse_loss(loss):
    l = None
    if loss is not None:
        if hasattr(tf.keras.losses, loss):
            l = getattr(tf.keras.losses, loss)
            wsbmr.utils.log._custom('BMK',f'Loss {loss} found in tf.keras.losses with definition {l}.')
        else:
            if loss.lower() in LOSSES:
                l = LOSSES[loss.lower()]
                wsbmr.utils.log._custom('BMK',f'Loss {loss} found in custom_losses with definition {l}.')
            else:
                wsbmr.utils.log._warn(f'Loss {loss} not found. Keeping as string.')
                l = loss
    return l


""" Custom telescopeMSE """
# Extracted directly from: https://github.com/oliviaweng/fastml-science/blob/main/sensor-data-compression/telescope.py
class telescopeMSE2(tf.keras.losses.Loss):
    def __init__(self, name="custom_loss", **kwargs):
        super().__init__(name=name)

        # Build remaps
        self.build_remaps_for_loss()


    """ We need this for the telescopic loss """
    def build_remaps_for_loss(self):

        # combine neighbor cells in 2x2 grids, record weights
        # multilpy weights by 0.25 for now to account for effective increase in cells from 12 (sum weights now 48 not 12)
        SCmask_48_36 = np.array([
            [ 0,  1,  4,  5, 0.25*1.5], # 2x2 supercells that perfectly tile the sensor
            [ 2,  3,  6,  7, 0.25*1.+1./12], #4 TC indices for 1 supercell (+) weight
            [ 8,  9, 12, 13, 0.25*2.25], 
            [10, 11, 14, 15, 0.25*1.5], 
            [16, 17, 20, 21, 0.25*1.5], 
            [18, 19, 22, 23, 0.25*1.+1./12], 
            [24, 25, 28, 29, 0.25*2.25], 
            [26, 27, 30, 31, 0.25*1.5], 
            [32, 33, 36, 37, 0.25*1.5], 
            [34, 35, 38, 39, 0.25*1.+1./12], 
            [40, 41, 44, 45, 0.25*2.25], 
            [42, 43, 46, 47, 0.25*1.5], 
            [ 4,  5,  8,  9, 0.25*1.5], # shift right by one TC (2/2x2)
            [ 6,  7, 10, 11, 0.25*1.],
            [20, 21, 24, 25, 0.25*1.5],
            [22, 23, 26, 27, 0.25*1.],
            [36, 37, 40, 41, 0.25*1.5],
            [38, 39, 42, 43, 0.25*1.],
            [ 1,  2,  5,  6, 0.25*1.], # shift down by one TC (2/2x2)
            [ 9, 10, 13, 14, 0.25*1.5],
            [17, 18, 21, 22, 0.25*1.],
            [25, 26, 29, 30, 0.25*1.5],
            [33, 34, 37, 38, 0.25*1.],
            [41, 42, 45, 46, 0.25*1.5],
            [ 5,  6,  9, 10, 0.25*1.], # shift down and right by one TC (1/2x2)
            [21, 22, 25, 26, 0.25*1.],
            [37, 38, 41, 42, 0.25*1.],
            [ 0,  1, 27, 31, 0.25*1.5], # inter-2x2 overlaps
            [ 1,  2, 23, 27, 0.25*1.],
            [ 2,  3, 19, 23, 0.25*1.+1./6],
            [ 3,  7, 34, 35, 0.25*1.+1./6],
            [ 7, 11, 33, 34, 0.25*1.],
            [11, 15, 32, 33, 0.25*1.5],
            [16, 17, 47, 43, 0.25*1.5],
            [17, 18, 43, 39, 0.25*1.],
            [18, 19, 39, 35, 0.25*1.+1./6],
        ])

        """ Remap for 8x8 to 4x4 """
        Remap_48_36 = np.zeros((48,36))
        for isc,sc in enumerate(SCmask_48_36): 
            for tc in sc[:4]:
                Remap_48_36[int(tc),isc]=1
        self.tf_Remap_48_36 = tf.constant(Remap_48_36,dtype=tf.float32)

        """ Remap """
        Weights_48_36 = SCmask_48_36[:,4]
        self.tf_Weights_48_36 = tf.constant(Weights_48_36,dtype=tf.float32)

        """ Remap """
        Remap_12_3 = np.zeros((12,3))
        for i in range(12): 
            Remap_12_3[i,int(i/4)]=1
        self.tf_Remap_12_3 = tf.constant(Remap_12_3,dtype=tf.float32)

        # keep simplified 12 x 3 mapping for now
        SCmask_48_12 = np.array([
            [ 0,  1,  4,  5],
            [ 2,  3,  6,  7],
            [ 8,  9, 12, 13],
            [10, 11, 14, 15],
            [16, 17, 20, 21],
            [18, 19, 22, 23],
            [24, 25, 28, 29],
            [26, 27, 30, 31],
            [32, 33, 36, 37],
            [34, 35, 38, 39],
            [40, 41, 44, 45],
            [42, 43, 46, 47],
        ])
        Remap_48_12 = np.zeros((48,12))
        for isc,sc in enumerate(SCmask_48_12): 
            for tc in sc:
                Remap_48_12[int(tc),isc]=1
        self.tf_Remap_48_12 = tf.constant(Remap_48_12,dtype=tf.float32)

        Remap_12_3 = np.zeros((12,3))
        for i in range(12): Remap_12_3[i,int(i/4)]=1
        self.tf_Remap_12_3 = tf.constant(Remap_12_3,dtype=tf.float32)


    def call(self, y_true, y_pred):
        # Cast to float32
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        # TC-level MSE
        y_pred_rs = K.reshape(y_pred, (-1,48))
        y_true_rs = K.reshape(y_true, (-1,48))
        # lossTC1 = K.mean(K.square(y_true_rs - y_pred_rs), axis=(-1))
        lossTC1 = K.mean(K.square(y_true_rs - y_pred_rs) * K.maximum(y_pred_rs, y_true_rs), axis=(-1))

        # map TCs to 2x2 supercells and compute MSE
        y_pred_36 = tf.matmul(y_pred_rs, self.tf_Remap_48_36)
        y_true_36 = tf.matmul(y_true_rs, self.tf_Remap_48_36)
        # lossTC2 = K.mean(K.square(y_true_12 - y_pred_12), axis=(-1))
        lossTC2 = K.mean(K.square(y_true_36 - y_pred_36) * K.maximum(y_pred_36, y_true_36) * self.tf_Weights_48_36, axis=(-1))
    
        # map 2x2 supercells to 4x4 supercells and compute MSE
        y_pred_12 = tf.matmul(y_pred_rs, self.tf_Remap_48_12)
        y_true_12 = tf.matmul(y_true_rs, self.tf_Remap_48_12)
        y_pred_3 = tf.matmul(y_pred_12, self.tf_Remap_12_3)
        y_true_3 = tf.matmul(y_true_12, self.tf_Remap_12_3)
        # lossTC3 = K.mean(K.square(y_true_3 - y_pred_3), axis=(-1))
        lossTC3 = K.mean(K.square(y_true_3 - y_pred_3) * K.maximum(y_pred_3, y_true_3), axis=(-1))

        # sum MSEs
        #return lossTC1 + lossTC2 + lossTC3
        return 4*lossTC1 + 2*lossTC2 + lossTC3

LOSSES = {'telescopemse2': telescopeMSE2(),
          'mean_squared_error': tf.keras.losses.MeanSquaredError(),
          'mean_absolute_error': tf.keras.losses.MeanAbsoluteError()}