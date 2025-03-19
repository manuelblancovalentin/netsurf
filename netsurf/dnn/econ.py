# Numpy 
import numpy as np

""" Tensorflow """
import tensorflow as tf
from keras import backend as K # To get intermediate activations between layers

# Local losses and layers
from . import losses
from .models import QModel

import wsbmr

""" Custom layer implementations s"""
from .layers import QQDense, QQConv1D, QQConv2D, QQDepthwiseConv2D, QQSeparableConv2D, QQConv2DTranspose, \
                    QQGlobalAveragePooling2D, QQMaxPooling2D, QQAveragePooling2D, \
                    QQActivation, QQSoftmax, \
                    QQBatchNormalization, \
                    QQFlatten, QQDropout, QQReshape, \
                    QQLSTM, QQUpSampling2D, QQApplyAlpha


class EconParams:
    def __init__(self):
        self.arrange = {'arrange': np.array([28,29,30,31,0,4,8,12,
                                    24,25,26,27,1,5,9,13,
                                    20,21,22,23,2,6,10,14,
                                    16,17,18,19,3,7,11,15,
                                    47,43,39,35,35,34,33,32,
                                    46,42,38,34,39,38,37,36,
                                    45,41,37,33,43,42,41,40,
                                    44,40,36,32,47,46,45,44]),
        'arrMask': np.array([1,1,1,1,1,1,1,1,
                            1,1,1,1,1,1,1,1,
                            1,1,1,1,1,1,1,1,
                            1,1,1,1,1,1,1,1,
                            1,1,1,1,0,0,0,0,
                            1,1,1,1,0,0,0,0,
                            1,1,1,1,0,0,0,0,
                            1,1,1,1,0,0,0,0,]),
        'calQMask': np.array([1,1,1,1,1,1,1,1,
                            1,1,1,1,1,1,1,1,
                            1,1,1,1,1,1,1,1,
                            1,1,1,1,1,1,1,1,
                            1,1,1,1,0,0,0,0,
                            1,1,1,1,0,0,0,0,
                            1,1,1,1,0,0,0,0,
                            1,1,1,1,0,0,0,0,])
        }

        self.params = {
            'CNN_layer_nodes':[8],
            'CNN_kernel_size':[3],
            'CNN_pool': [False],
            'CNN_padding': ['same'],
            'CNN_strides':[(2,2)],
            'Dense_layer_nodes': [], #does not include encoded layer
            'array': self.arrange['arrange'],
            'arrMask': self.arrange['arrMask'],
            'calQMask': self.arrange['calQMask'],
            'nBitsEncode': {'total':  9, 'integer': 1,'keep_negative':0},
            'channels_first': False,
            'encoded_dim': 16,
            'maskConvOutput': [],
            'n_copy': 0,      # no. of copy for hi occ datasets
            'activation'       : 'relu',
            'shape': (8,8,1)
        }

        self.eval_dict={
                # compare to other algorithms
                'algnames'    :['ae','stc','thr_lo','thr_hi','bc'],
                # 'metrics'     :{'EMD':emd},
                "occ_nbins"   :12,
                "occ_range"   :(0,24),
            "occ_bins"    : [0,2,5,10,15],
            "chg_nbins"   :20,
                "chg_range"   :(0,200),
                "chglog_nbins":20,
                "chglog_range":(0,2.5),
                "chg_bins"    :[0,2,5,10,50],
                "occTitle"    :r"occupancy [1 MIP$_{\mathrm{T}}$ TCs]"       ,
                "logMaxTitle" :r"log10(Max TC charge/MIP$_{\mathrm{T}}$)",
            "logTotTitle" :r"log10(Sum of TC charges/MIP$_{\mathrm{T}}$)",
            }
    @property
    def shape(self):
        return self.params['shape']
    @property
    def algnames(self):
        return self.eval_dict['algnames']
    @property
    def occ_nbins(self):
        return self.eval_dict['occ_nbins']
    @property
    def occ_range(self):
        return self.eval_dict['occ_range']
    @property
    def occ_bins(self):
        return self.eval_dict['occ_bins']
    @property
    def chg_nbins(self):
        return self.eval_dict['chg_nbins']
    @property
    def chg_range(self):
        return self.eval_dict['chg_range']
    @property
    def chglog_nbins(self):
        return self.eval_dict['chglog_nbins']
    @property
    def chglog_range(self):
        return self.eval_dict['chglog_range']
    @property
    def chg_bins(self):
        return self.eval_dict['chg_bins']
    @property
    def occTitle(self):
        return self.eval_dict['occTitle']
    @property
    def logMaxTitle(self):
        return self.eval_dict['logMaxTitle']
    @property
    def logTotTitle(self):
        return self.eval_dict['logTotTitle']
    @property
    def CNN_layer_nodes(self):
        return self.params['CNN_layer_nodes']
    @property
    def CNN_kernel_size(self):
        return self.params['CNN_kernel_size']
    @property
    def CNN_pool(self):
        return self.params['CNN_pool']
    @property
    def CNN_padding(self):
        return self.params['CNN_padding']
    @property
    def CNN_strides(self):
        return self.params['CNN_strides']
    @property
    def Dense_layer_nodes(self):
        return self.params['Dense_layer_nodes']
    @property
    def array(self):
        return self.params['array']
    @property
    def arrMask(self):
        return self.arrange['arrMask']
    @property
    def calQMask(self):
        return self.arrange['calQMask']
    @property
    def nBitsEncode(self):
        return self.params['nBitsEncode']
    @property
    def channels_first(self):
        return self.params['channels_first']
    @property
    def encoded_dim(self):
        return self.params['encoded_dim']
    @property
    def maskConvOutput(self):
        return self.params['maskConvOutput']
    @property
    def n_copy(self):
        return self.params['n_copy']
    @property
    def activation(self):
        return self.params['activation']
    
    

# Extracted directly from: https://github.com/oliviaweng/fastml-science/blob/main/sensor-data-compression/denseCNN.py#L31
# User for ECONTAE
class MaskLayer(tf.keras.layers.Layer):
    def __init__(self,nFilter,arrMask):
        super(MaskLayer, self).__init__()
        self.nFilter = tf.constant(nFilter)
        self.arrayMask = np.array([arrMask])
        self.mask = tf.reshape(tf.stack(
                        tf.repeat(self.arrayMask,repeats=[nFilter],axis=0),axis=1),
                        shape=[-1])      
    def call(self, inputs):
        return tf.reshape(tf.boolean_mask(inputs,self.mask,axis=1),
                          shape=(tf.shape(inputs)[0],48*self.nFilter))
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'nFilter': self.nFilter.numpy(),
            'arrMask': self.arrayMask.tolist(),
        })
        return config


""" AE for ECONT (telescopic) """
# We are basically implementing model: 8x8_c8_S2_tele from here: https://github.com/oliviaweng/fastml-science/blob/main/sensor-data-compression/networks.py#L134
class ECONTAE(QModel):
    def __init__(self, quantizer, loss = 'telescopeMSE2', 
                 in_shape = (8, 8, 1), metrics=['mse', 'r2score', 'pearsoncorrelation'], 
                 type = 'unsupervised', out_shape = (8,8,1), **kwargs):

        # Set loss and optimizer
        loss = loss if loss != 'telescopeMSE2' else losses.telescopeMSE2()

        """ Specific to ECOND AE """
        name = '8x8_c8_S2_tele'
        self.label = '8x8_c[8]_S2(tele)'
        self.arr_key = '8x8'
        self.isDense2D = False
        self.isQK = False
        self.ws = ''
        #self.loss = loss

        # Init params
        self.params = EconParams()

        # Call super 
        super().__init__(quantizer, in_shape, in_shape, name = name, loss = loss, metrics = metrics, type = type, **kwargs)

        # Finally build model
        self.build_model(quantizer, in_shape, in_shape, loss = loss, metrics = metrics, type = type, **kwargs)

        # Make sure we set original metrics here
        self._metrics = metrics

    def build_model(self, quantizer: 'QuantizationScheme', in_shape, out_shape, **kwargs):
        super().build_model(*self.econae(quantizer, in_shape, **kwargs), **kwargs)

    def evaluate(self, x, y, **kwargs):
        # Make sure x and y have a num_samples that's a multiple of 48/8/8 
        batch_size = 3
        nsamples = (x.shape[0]//3)*3
        x = x[:nsamples]
        y = y[:nsamples]
        wsbmr.utils.log._info(f'Restricting samples to {nsamples} for evaluation, so we have a multiple of 48 when reshaping to 8x8. Using batch size of {batch_size}.')
        return super().evaluate(x, y, batch_size = batch_size, **kwargs)

    """ Build model method """    
    def econae(self, quantizer: 'QuantizationScheme', in_shape, 
                    use_bias = True, use_constraint = True, **kwargs):

        """ Get params first """
        encoded_dim = self.params.encoded_dim
        CNN_layer_nodes   = self.params.CNN_layer_nodes
        CNN_kernel_size   = self.params.CNN_kernel_size
        CNN_padding       = self.params.CNN_padding
        CNN_strides       = self.params.CNN_strides
        CNN_pool          = self.params.CNN_pool
        Dense_layer_nodes = self.params.Dense_layer_nodes #does not include encoded layer
        channels_first    = self.params.channels_first

        maskConvOutput = self.params.maskConvOutput
        activation = self.params.activation

        """ Build encoder """
        ip = tf.keras.layers.Input(shape = in_shape)  # adapt this if using `channels_first` image data format
        x = ip

        # Init activations
        activations = []

        for i, n_nodes in enumerate(CNN_layer_nodes):
            x = QQConv2D(quantizer, n_nodes, CNN_kernel_size[i], 
                            strides=CNN_strides[i], 
                            padding=CNN_padding[i],
                            data_format='channels_first' if channels_first else 'channels_last',
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            use_bias = use_bias,
                            use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]

            # QQApplyAlpha
            x = QQApplyAlpha(quantizer)(x)
            # Append to activations
            activations += [x]
            
            x = QQActivation(quantizer, activation, use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]
            
            if CNN_pool[i]:
                x = QQMaxPooling2D(quantizer, (2, 2), padding='same',
                        data_format='channels_first' if channels_first else 'channels_last',
                        use_bias = use_bias, use_constraint=use_constraint)(x)
                # Append to activations
                activations += [x]
                
                # Add quantization here 
                x = QQActivation(quantizer, activation, use_constraint=use_constraint)(x)
                # Append to activations
                activations += [x]

        shape = K.int_shape(x)

        x = QQFlatten(quantizer, use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        if len(maskConvOutput)>0:
            if np.count_nonzero(maskConvOutput)!=48:
                raise ValueError("Trying to mask conv output with an array mask that does not contain exactly 48 calQ location. maskConvOutput = ",self.pams['maskConvOutput'])
            
            x = MaskLayer( nFilter = CNN_layer_nodes[-1] , arrMask = maskConvOutput )(x)
            # Append to activations
            activations += [x]

            # QQApplyAlpha
            x = QQApplyAlpha(quantizer)(x)
            # Append to activations
            activations += [x]

        #encoder dense nodes
        for n_nodes in Dense_layer_nodes:
            x = QQDense(quantizer, n_nodes, 
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            kernel_initializer='he_normal',
                            bias_initializer='zeros',
                            use_constraint=use_constraint,
                            use_bias = use_bias)(x)
            # Append to activations
            activations += [x]

            # QQApplyAlpha
            x = QQApplyAlpha(quantizer)(x)
            # Append to activations
            activations += [x]

            x = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]

        encodedLayer = QQDense(quantizer, encoded_dim, name='encoded_vector', 
            kernel_quantizer = 'logit',
            bias_quantizer = 'logit',
            kernel_initializer='he_normal',
            bias_initializer='zeros',
            use_constraint=use_constraint,
            use_bias = use_bias)(x)
        # Append to activations
        activations += [encodedLayer]

        # QQApplyAlpha
        x = QQApplyAlpha(quantizer)(x)
        # Append to activations
        activations += [x]

        encodedLayer = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(encodedLayer)
        # Append to activations
        activations += [encodedLayer]
        
        #encoded_inputs = tf.keras.layers.Input(shape=(encoded_dim,), name='decoder_input')
        #x = encoded_inputs
        x = encodedLayer

        #decoder dense nodes
        for n_nodes in Dense_layer_nodes:
            x = QQDense(quantizer, n_nodes, 
                                    kernel_quantizer = 'logit',
                                    bias_quantizer = 'logit',
                                    kernel_initializer='he_normal',
                                    bias_initializer='zeros',
                                    use_constraint=use_constraint,
                                    use_bias = use_bias)(x)
             # Append to activations
            activations += [x]

            # QQApplyAlpha
            x = QQApplyAlpha(quantizer)(x)
            # Append to activations
            activations += [x]

            x = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]

        x = QQDense(quantizer, shape[1] * shape[2] * shape[3], 
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            kernel_initializer='he_normal',
                            bias_initializer='zeros',
                            use_constraint=use_constraint,
                            use_bias = True)(x)
        # Append to activations
        activations += [x]

        # QQApplyAlpha
        x = QQApplyAlpha(quantizer)(x)
        # Append to activations
        activations += [x]
        
        x = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        x = QQReshape(quantizer, (shape[1], shape[2], shape[3]), use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        for i,n_nodes in enumerate(CNN_layer_nodes):

            if CNN_pool[i]:
                x = QQUpSampling2D(quantizer, (2, 2),
                        data_format = 'channels_first' if channels_first else 'channels_last',
                        kernel_quantizer = 'logit',
                        bias_quantizer = 'logit',
                        kernel_initializer='he_normal',
                        bias_initializer='zeros',
                        use_constraint=use_constraint,
                        use_bias = use_bias)(x)
                # Append to activations
                activations += [x]
            
            x = QQConv2DTranspose(quantizer, n_nodes, 
                                    CNN_kernel_size[i], 
                                    strides = CNN_strides[i],
                                    padding = CNN_padding[i], 
                                    data_format = 'channels_first' if channels_first else 'channels_last',
                                    kernel_quantizer = 'logit',
                                    bias_quantizer = 'logit',
                                    kernel_initializer='he_normal',
                                    bias_initializer='zeros',
                                    use_constraint=use_constraint,
                                    use_bias = use_bias)(x)
            
            # Append to activations
            activations += [x]

            # QQApplyAlpha
            x = QQApplyAlpha(quantizer)(x)
            # Append to activations
            activations += [x]
            
            x = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]


        #shape[0] will be # of channel
        x = QQConv2DTranspose(quantizer, filters = in_shape[0] if channels_first else in_shape[2], 
                                kernel_size = CNN_kernel_size[0], 
                                padding = 'same',
                                data_format = 'channels_first' if channels_first else 'channels_last',
                                kernel_quantizer = 'logit',
                                bias_quantizer = 'logit',
                                kernel_initializer='he_normal',
                                bias_initializer='zeros',
                                use_constraint=use_constraint,
                                use_bias = use_bias)(x)
        # Append to activations
        activations += [x]

        out = QQActivation(quantizer, 'smooth_sigmoid', name='decoder_output', use_constraint=use_constraint)(x)
        # Append to activations
        activations += [out]

        return ip, out, activations

    # Create model name based on architecture 
    def create_model_name_by_architecture(self):
        name = f'{self.count_params()}_'
        name += f'AE_{self.in_shape[0]}x{self.in_shape[1]}_c{self.params.encoded_dim}'
        return name


# Register model
ECONMODELS = {'ECONTAE': ECONTAE}
