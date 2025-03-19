
""" Import tensorflow """
import tensorflow as tf

""" Basic class """
from .models import QModel

""" Custom layer implementations """
from .layers import QQDense, QQConv1D, QQConv2D, QQDepthwiseConv2D, QQSeparableConv2D, QQConv2DTranspose, \
                    QQGlobalAveragePooling2D, QQMaxPooling2D, QQAveragePooling2D, \
                    QQActivation, QQSoftmax, \
                    QQBatchNormalization, \
                    QQFlatten, QQDropout, QQReshape, \
                    QQLSTM, QQUpSampling2D, QQApplyAlpha

# As seen in here: http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf
class LeNet5(QModel):
    def __init__(self, quantizer, in_shape = (32, 32, 3), out_shape = (10,), 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'], type = 'classification', **kwargs):
        """."""
        super().__init__(quantizer, in_shape, out_shape, loss = loss,
                         metrics = metrics, type = type, **kwargs)
        
        # Now build model 
        self.build_model(quantizer, in_shape, out_shape, 
                         loss = loss, metrics = metrics, type = type, **kwargs)
        
        # Make sure we set original metrics here
        self._metrics = metrics

    def build_model(self, quantizer: 'QuantizationScheme', in_shape, out_shape, **kwargs):
        super().build_model(*self.lenet5(quantizer, in_shape, out_shape, **kwargs), **kwargs)

    """ Build the model """
    def lenet5(self, quantizer, in_shape, out_shape, use_bias = True,  
               use_constraint = True, **kwargs):

        """ Build input layer """
        x = ip = tf.keras.layers.Input(shape = in_shape)

        # Init activations
        activations = []

        """ Stack convolutional layers"""
        x = QQConv2D(quantizer, 6, (5, 5), strides = (1,1),
                            padding = 'same',
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
        
        x = QQActivation(quantizer, 'tanh', use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        # TODO: This is failing right now, so let's just use normal layer
        # x = QQAveragePooling2D(quantizer, pool_size=(2, 2), strides = (1,1), padding='valid',
        #                              average_quantizer='logit', use_constraint=use_constraint)(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides = (1,1), padding='valid')(x)

        # Append to activations
        activations += [x]

        x = QQConv2D(quantizer, 16, (5, 5), strides = (1,1),
                            padding = 'valid',
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

        x = QQActivation(quantizer, 'tanh', use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides = (1,1), padding='valid')(x)
        # x = QQAveragePooling2D(quantizer, pool_size=(2, 2), strides = (2,2), padding='valid',
        #                              average_quantizer='logit',
        #                              use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        x = QQConv2D(quantizer, 120, (5, 5), strides = (1,1),
                            padding = 'valid',
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
        

        x = QQActivation(quantizer, 'tanh', use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        """ Flatten """
        x = QQFlatten(quantizer, use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        """ Stack dense layers """
        x = QQDense(quantizer, 84,
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
        
        x = QQActivation(quantizer, 'tanh', use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        """ Last dense """
        x = QQDense(quantizer, out_shape[0],
                        kernel_quantizer = 'logit',
                        bias_quantizer = 'logit',
                        use_bias = use_bias,
                        use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        """ Softmax """
        out = QQSoftmax(quantizer, name="softmax", use_constraint=use_constraint)(x)
        # Append to activations
        activations += [out]

        return ip, out, activations
    
    # Create model name based on architecture 
    def create_model_name_by_architecture(self):
        return 'lenet5'



""" Build mobilenet v1 """
# Implementtion extracted from: https://github.com/SiliconLabs/platform_ml_models/blob/master/eembc/Person_detection/mobilenet_v1_eembc.py
class MobileNetV1(QModel):
    def __init__(self, quantizer, in_shape = (96, 96, 3), out_shape = (2,),
                    optimizer='adam', loss='categorical_crossentropy', 
                    metrics=['accuracy'], type = 'classification', **kwargs):
        """."""
        super().__init__(quantizer, in_shape, out_shape, optimizer = optimizer, loss =loss, 
                         metrics = metrics, type = type, **kwargs)
        
        # Now build 
        self.build_model(quantizer, in_shape = in_shape, out_shape = out_shape, 
                         loss = loss, metrics = metrics, type = type, **kwargs)
        
        # Make sure we set original metrics here
        self._metrics = metrics

    def build_model(self, quantizer: 'QuantizationScheme', in_shape, out_shape, **kwargs):
        super().build_model(*self.mobilenet_v1(quantizer, in_shape, out_shape, **kwargs), **kwargs)

    """ Build model method """
    def mobilenet_v1(self, quantizer: 'QuantizationScheme', in_shape, out_shape, 
                     use_bias = True, use_constraint = True, **kwargs):
        
        num_filters = 8 # normally 32, but running with alpha=.25 per EEMBC requirement

        # Init activations 
        activations = []

        # Input
        ip = tf.keras.layers.Input(shape=in_shape)
        x = ip # Keras model uses ZeroPadding2D()

        params = [  (3,3,2,1,2), 
                    (1,3,1,2,2), 
                    (1,3,1,1,1), 
                    (1,3,1,2,2), 
                    (1,3,1,1,1), 
                    (1,3,1,2,2), 
                    (1,3,1,1,1), 
                    (1,3,1,1,1), 
                    (1,3,1,1,1),
                    (1,3,1,1,1), 
                    (1,3,1,1,1), 
                    (1,3,1,2,2), 
                    (1,3,1,1,1)]

        for ks1, ks2, st1, st2, multiplier in params:
            # 1st layer, pure conv
            # Keras 2.2 model has padding='valid' and disables bias
            x = QQConv2D(quantizer, num_filters,
                        kernel_size=ks1,
                        strides=st1,
                        use_bias = use_bias,
                        padding='same',
                        kernel_initializer='he_normal',
                        bias_initializer = 'zeros',
                        kernel_quantizer = 'logit',
                        bias_quantizer = 'logit',
                        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                        use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]

            # QQApplyAlpha
            x = QQApplyAlpha(quantizer)(x)
            # Append to activations
            activations += [x]

            # Batchnorm
            x = QQBatchNormalization(quantizer, beta_quantizer = 'po2', 
                                            gamma_quantizer = 'relu_po2', 
                                            mean_quantizer = 'po2', 
                                            variance_quantizer = 'relu_po2_quadratic',
                                            beta_initializer = 'zeros', 
                                            gamma_initializer = 'ones', 
                                            moving_mean_initializer = 'zeros', 
                                            moving_variance_initializer = 'ones', 
                                            use_constraint= use_constraint)(x)
            # Append to activations
            activations += [x]
            
            x = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(x) # Keras uses ReLU6 instead of pure ReLU
            # Append to activations
            activations += [x]

            # 2nd layer, depthwise separable conv
            # Filter size is always doubled before the pointwise conv
            # Keras uses ZeroPadding2D() and padding='valid'
            x = QQDepthwiseConv2D(quantizer, kernel_size=ks2,
                        strides=st2,
                        padding='same',
                        use_bias = use_bias,
                        depthwise_quantizer = 'logit',
                        bias_quantizer = 'logit',
                        kernel_initializer = 'he_normal',
                        bias_initializer = 'zeros',
                        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                        use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]

            # QQApplyAlpha
            x = QQApplyAlpha(quantizer)(x)
            # Append to activations
            activations += [x]
            
            # Batchnorm
            x = QQBatchNormalization(quantizer, beta_quantizer = 'po2', 
                                            gamma_quantizer = 'relu_po2', 
                                            mean_quantizer = 'po2', 
                                            variance_quantizer = 'relu_po2_quadratic',
                                            beta_initializer = 'zeros', 
                                            gamma_initializer = 'ones', 
                                            moving_mean_initializer = 'zeros', 
                                            moving_variance_initializer = 'ones', 
                                            use_constraint= use_constraint)(x)
            # Append to activations
            activations += [x]
            
            x = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]

            # Change filters
            num_filters = multiplier*num_filters

        # Last conv group
        x = QQConv2D(quantizer, num_filters,
                    kernel_size=1,
                    strides=1,
                    padding='same',
                    kernel_initializer='he_normal',
                    bias_initializer = 'zeros',
                    kernel_quantizer = 'logit',
                    bias_quantizer = 'logit',
                    use_bias = use_bias,
                    kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                    use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        # QQApplyAlpha
        x = QQApplyAlpha(quantizer)(x)
        # Append to activations
        activations += [x]

        # Batchnorm
        x = QQBatchNormalization(quantizer, beta_quantizer = 'po2', 
                                            gamma_quantizer = 'relu_po2', 
                                            mean_quantizer = 'po2', 
                                            variance_quantizer = 'relu_po2_quadratic',
                                            beta_initializer = 'zeros', 
                                            gamma_initializer = 'ones', 
                                            moving_mean_initializer = 'zeros', 
                                            moving_variance_initializer = 'ones', 
                                            use_constraint= use_constraint)(x)
        # Append to activations
        activations += [x]
        
        x = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        # Average pooling, max polling may be used also
        # Keras employs Globalkeras.layers.AveragePooling2D 
        # TODO: This layer is erroring so let's use normal layer
        #x = QQAveragePooling2D(quantizer, pool_size=x.shape[1:3], use_constraint=use_constraint)(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=x.shape[1:3])(x)

        # Append to activations
        activations += [x]

        #x = MaxPooling2D(pool_size=x.shape[1:3])(x)

        # Keras inserts Dropout() and a pointwise keras.layers.Conv2D() here
        # We are staying with the paper base structure

        # keras.layers.Flatten, FC layer and classify
        x = QQFlatten(quantizer, use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        x = QQDense(quantizer, out_shape[0], 
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            kernel_initializer = 'he_normal',
                            bias_initializer = 'zeros',
                            use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]
        
        out = QQSoftmax(quantizer, use_constraint=use_constraint)(x)
        # Append to activations
        activations += [out]

        return ip, out, activations
    
    """ Override NN method because otherwise this will be too long """
    def create_model_name_by_architecture(self):
        return 'mobilenetv1'


""" DSConv model """
# Taken from: https://github.com/SiliconLabs/platform_ml_models/blob/master/eembc/KWS10_ARM_DSConv/dsconv_arm_eembc.py
class DSConv(QModel):
    def __init__(self, quantizer, in_shape = (50,10,1), out_shape = (12,),
                    optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'], 
                    type = 'classification', **kwargs):
        """."""
        super().__init__(quantizer, in_shape , out_shape, optimizer = optimizer, 
                         loss = loss, metrics = metrics, type = type, **kwargs)
        
        # Now build model
        self.build_model(quantizer, in_shape, out_shape,
                         loss = loss, metrics = metrics, type = type, **kwargs)
        
        # Make sure we set original metrics here
        self._metrics = metrics

    def build_model(self, quantizer: 'QuantizationScheme', input_shape, output_shape, **kwargs):
        super().build_model(*self.dsconv(quantizer, input_shape, output_shape, **kwargs), **kwargs)

    """ Build model method """
    def dsconv(self, quantizer: 'QuantizationScheme', in_shape, out_shape, use_bias = True, 
               use_constraint = True, **kwargs):
        
        # Init activations
        activations = []

        # Parameters
        filters = 64

        # Input layer
        ip = tf.keras.layers.Input(shape=in_shape)

        # Input pure conv2d
        x = QQConv2D(quantizer, filters, (10,4), strides=(2,2), 
                            padding='same', 
                            use_bias = use_bias, 
                            kernel_initializer='he_normal',
                            bias_initializer = 'zeros',
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                            use_constraint=use_constraint)(ip)
        # Append to activations
        activations += [x]

        # QQApplyAlpha
        x = QQApplyAlpha(quantizer)(x)
        # Append to activations
        activations += [x]
        
        x = QQBatchNormalization(quantizer, beta_quantizer = 'po2', 
                                            gamma_quantizer = 'relu_po2', 
                                            mean_quantizer = 'po2', 
                                            variance_quantizer = 'relu_po2_quadratic',
                                            beta_initializer = 'zeros', 
                                            gamma_initializer = 'ones', 
                                            moving_mean_initializer = 'zeros', 
                                            moving_variance_initializer = 'ones', 
                                            use_constraint= use_constraint)(x)
        # Append to activations
        activations += [x]
        
        x = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        x = QQDropout(quantizer, rate=0.2, use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        for _ in range(4):

            # layer of separable depthwise conv2d
            # Separable consists of depthwise conv2d followed by conv2d with 1x1 kernels
            x = QQDepthwiseConv2D(quantizer, depth_multiplier=1, kernel_size=(3,3), 
                                        use_bias = use_bias,
                                        depthwise_quantizer = 'logit',
                                        bias_quantizer = 'logit',
                                        kernel_initializer='he_normal',
                                        bias_initializer='zeros',
                                        padding='same', 
                                        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                        use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]

            # QQApplyAlpha
            x = QQApplyAlpha(quantizer)(x)
            # Append to activations
            activations += [x]
            
            x = QQBatchNormalization(quantizer, beta_quantizer = 'po2', 
                                            gamma_quantizer = 'relu_po2', 
                                            mean_quantizer = 'po2', 
                                            variance_quantizer = 'relu_po2_quadratic',
                                            beta_initializer = 'zeros', 
                                            gamma_initializer = 'ones', 
                                            moving_mean_initializer = 'zeros', 
                                            moving_variance_initializer = 'ones', 
                                            use_constraint= use_constraint)(x)
            # Append to activations
            activations += [x]
            
            x = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]

            x = QQConv2D(quantizer, filters, (1,1), 
                            padding='same', 
                            use_bias = use_bias,
                            kernel_initializer='he_normal',
                            bias_initializer='zeros',
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                            use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]

            # QQApplyAlpha
            x = QQApplyAlpha(quantizer)(x)
            # Append to activations
            activations += [x]
            
            x = QQBatchNormalization(quantizer, beta_quantizer = 'po2', 
                                            gamma_quantizer = 'relu_po2', 
                                            mean_quantizer = 'po2', 
                                            variance_quantizer = 'relu_po2_quadratic',
                                            beta_initializer = 'zeros', 
                                            gamma_initializer = 'ones', 
                                            moving_mean_initializer = 'zeros', 
                                            moving_variance_initializer = 'ones', 
                                            use_constraint= use_constraint)(x)
            # Append to activations
            activations += [x]
            
            x = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]

        # Reduce size and apply final softmax
        x = QQDropout(quantizer, rate=0.4, use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        # TODO: This layer is erroring so let's use normal layer for now
        # x = QQAveragePooling2D(quantizer, pool_size=(25,5), 
        #                        average_quantizer='logit', 
        #                        use_constraint=use_constraint)(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(25,5))(x)

        # Append to activations
        activations += [x]

        x = QQFlatten(quantizer, use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        x = QQDense(quantizer, out_shape[0], 
                            use_bias = use_bias,
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        out = QQSoftmax(quantizer, use_constraint=use_constraint)(x)
        # Append to activations
        activations += [out]

        return ip, out, activations

    """ Override NN method because otherwise this will be too long """
    def create_model_name_by_architecture(self):
        return 'DSConv'


class SqueezeNet(QModel):
    def __init__(self, quantizer: 'QuantizationScheme', in_shape = (50, 50, 3), out_shape = (20,),
                    use_bypass=False, dropout_rate=None, compression=1.0, type = 'classification',
                    optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'], **kwargs):
        """."""
        super().__init__(quantizer, in_shape, out_shape, optimizer = optimizer, 
                         loss = loss, metrics = metrics, type = type, **kwargs)

        self.use_bypass = use_bypass
        self.dropout_rate = dropout_rate
        self.compression = compression

        # Now build 
        self.build_model(quantizer, in_shape, out_shape, 
                         loss = loss, metrics = metrics, type = type, **kwargs)
        
        # Make sure we set original metrics here
        self._metrics = metrics

    def build_model(self, quantizer: 'QuantizationScheme', in_shape, out_shape, **kwargs):
        super().build_model(*self.squeezenet(quantizer, in_shape, out_shape, **kwargs), **kwargs)

    """ Build model method """
    def squeezenet(self, quantizer: 'QuantizationScheme', in_shape, out_shape, 
                   use_bias = True, use_constraint = True, **kwargs):

        # Get params
        use_bypass = self.use_bypass
        dropout_rate = self.dropout_rate
        compression = self.compression

        # Init Activations
        activations = []
        
        # Input layer 
        ip = tf.keras.layers.Input(shape=in_shape)

        # Create layers
        x = QQConv2D(quantizer, int(96*self.compression), (7,7), 
                            strides=(2,2), 
                            padding='same',
                            use_bias = use_bias,
                            kernel_initializer='he_normal',
                            bias_initializer = 'zeros',
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                            use_constraint=use_constraint)(ip)
        # Append to activations
        activations += [x]

        # QQApplyAlpha
        x = QQApplyAlpha(quantizer)(x)
        # Append to activations
        activations += [x]
        
        x = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        x = QQMaxPooling2D(quantizer, pool_size=(3,3), strides=(2,2), 
                           name = 'maxpool1', use_constraint= use_constraint)(x)
        # Append to activations
        activations += [x]

        # Add Activation just for quantization purposes
        x = QQActivation(quantizer, 'logit', use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]
        
        x, _acts = self.create_fire_module(x, int(16*compression), 'fire2', quantizer, use_constraint = use_constraint, use_bypass = False, **kwargs)
        activations.extend(_acts)
        x, _acts = self.create_fire_module(x, int(16*compression), 'fire3', quantizer, use_constraint = use_constraint, use_bypass = use_bypass, **kwargs)
        activations.extend(_acts)
        x, _acts = self.create_fire_module(x, int(32*compression), 'fire4', quantizer, use_constraint = use_constraint, use_bypass = False, **kwargs)
        activations.extend(_acts)

        x = QQMaxPooling2D(quantizer, pool_size=(3,3), strides=(2,2), name = 'maxpool4',
                           use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        # ADd Activation just for quantization purposes
        x = QQActivation(quantizer, 'logit', use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        x, _acts = self.create_fire_module(x, int(32*compression), 'fire5', quantizer, use_constraint = use_constraint, use_bypass = use_bypass, **kwargs)
        activations.extend(_acts)
        x, _acts = self.create_fire_module(x, int(48*compression), 'fire6', quantizer, use_constraint = use_constraint, **kwargs)
        activations.extend(_acts)
        x, _acts = self.create_fire_module(x, int(48*compression), 'fire7', quantizer, use_constraint = use_constraint, use_bypass = use_bypass, **kwargs)
        activations.extend(_acts)
        #x = self.create_fire_module(x, int(64*compression), 'fire8', quantizer, **kwargs)

        #x = keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name = 'maxpool8')(x)

        #x = self.create_fire_module(x, int(64*compression), 'fire9', quantizer, use_bypass = use_bypass, **kwargs)

        if dropout_rate:
            x = QQDropout(quantizer, dropout_rate, use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]
        
        # Output
        #x = keras.layers.GlobalAveragePooling2D(name='avgpool10')(x)
        x = QQFlatten(quantizer, use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        # x = QQConv2D(quantizer, self.out_shape[0], (1,1), 
        #                     strides=(1,1), 
        #                     padding='valid',
        #                     use_bias = use_bias,
        #                     kernel_quantizer = quantizer,
        #                     bias_quantizer = quantizer,
        #                     kernel_regularizer=keras.regularizers.l2(1e-4),
        #                     name = 'conv10')(x)
        #x = keras.layers.GlobalAveragePooling2D(name='avgpool10')(x)

        x = QQDense(quantizer, out_shape[0], 
                                kernel_quantizer = 'logit',
                                bias_quantizer = 'logit',
                                kernel_initializer = 'he_normal',
                                bias_initializer = 'zeros',
                                use_bias = use_bias,
                                use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        out = QQSoftmax(quantizer, name='softmax', use_constraint=use_constraint)(x)
        # Append to activations
        activations += [out]

        return ip, out, activations
    
    """ Override NN method because otherwise this will be too long """
    def create_model_name_by_architecture(self):
        return 'squeezenet'

    """ Create fire module """
    def create_fire_module(self, x, nb_squeeze_filter, name, quantizer, use_constraint = True, use_bypass = False, use_bias = True, **kwargs):

        _acts = []

        nb_expand_filter = 4 * int(nb_squeeze_filter)

        squeeze = QQConv2D(quantizer, nb_squeeze_filter, (1,1), 
                            padding='same',
                            use_bias = use_bias,
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            kernel_initializer='he_normal',
                            bias_initializer='zeros',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                            use_constraint= use_constraint,
                            name=f'{name}_squeeze')(x)
        _acts += [squeeze]

        # QQApplyAlpha
        squeeze = QQApplyAlpha(quantizer)(squeeze)
        # Append to activations
        _acts += [squeeze]
        
        squeeze = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(squeeze)
        _acts += [squeeze]

        expand_1x1 = QQConv2D(quantizer, nb_expand_filter, (1,1), 
                            padding='same',
                            use_bias = use_bias,
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            kernel_initializer='he_normal',
                            bias_initializer='zeros',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                            use_constraint=use_constraint,
                            name=f'{name}_expand_1x1')(squeeze)
        _acts += [expand_1x1]

        # QQApplyAlpha
        expand_1x1 = QQApplyAlpha(quantizer)(expand_1x1)
        # Append to activations
        _acts += [expand_1x1]

        expand_1x1 = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(expand_1x1)
        _acts += [expand_1x1]

        expand_3x3 = QQConv2D(quantizer, nb_expand_filter, (3,3), 
                            padding='same',
                            use_bias = use_bias,
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            kernel_initializer='he_normal',
                            bias_initializer='zeros',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                            name=f'{name}_expand_3x3',
                            use_constraint=use_constraint)(squeeze)
        _acts += [expand_3x3]

        # QQApplyAlpha
        expand_3x3 = QQApplyAlpha(quantizer)(expand_3x3)
        # Append to activations
        _acts += [expand_3x3]

        expand_3x3 = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(expand_3x3)
        _acts += [expand_3x3]

        x_ret = tf.keras.layers.concatenate([expand_1x1, expand_3x3], axis=-1, name=f'{name}_concat')
        _acts += [x_ret]

        if use_bypass:
            x_ret = tf.keras.layers.add([x_ret, x], name=f'{name}_concatenate_bypass')
            _acts += [x_ret]
        
        return x_ret, _acts

""" SqueezeNet """
class SqueezeNet11(QModel):
    def __init__(self, quantizer: 'QuantizationScheme', in_shape = (50, 50, 3), out_shape = (20,),
                    use_bypass=False, dropout_rate=None, compression=1.0,
                    optimizer = 'adam', loss = 'categorical_crossentropy', 
                    metrics = ['accuracy'], type = 'classification', **kwargs):
        """."""
        super().__init__(quantizer, in_shape, out_shape, optimizer = optimizer, loss = loss, 
                         metrics = metrics, type = type, **kwargs)

        # Local
        self.use_bypass = use_bypass
        self.dropout_rate = dropout_rate
        self.compression = compression

        # Now build
        self.build_model(quantizer, in_shape, out_shape,
                         loss = loss, metrics = metrics, type = type, 
                         **kwargs)
        
        # Make sure we set original metrics here
        self._metrics = metrics
        
    def build_model(self, quantizer: 'QuantizationScheme', input_shape, output_shape, **kwargs):
        super().build_model(*self.squeezenet11(quantizer, input_shape, output_shape, **kwargs), **kwargs)

    """ Build model method """
    def squeezenet11(self, quantizer: 'QuantizationScheme', in_shape, out_shape, use_bias = True, 
                     use_constraint = True, **kwargs):

        # Get params
        use_bypass = self.use_bypass
        dropout_rate = self.dropout_rate
        compression = self.compression

        # Activations
        activations = []

        # Input layer 
        ip = tf.keras.layers.Input(shape=in_shape)

        # Create layers
        x = QQConv2D(quantizer, int(96*self.compression), (7,7), 
                            strides=(2,2), 
                            padding='same',
                            use_bias = use_bias,
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            kernel_initializer = 'he_normal',
                            bias_initializer = 'zeros',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                            use_constraint=use_constraint)(ip)
        # Append to activations
        activations += [x]

        # QQApplyAlpha
        x = QQApplyAlpha(quantizer)(x)
        # Append to activations
        activations += [x]
        
        x = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        x = QQMaxPooling2D(quantizer, pool_size=(3,3), strides=(2,2), name = 'maxpool1',
                           use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        # Add Activation just for quantization purposes
        x = QQActivation(quantizer, 'logit', use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]
        
        x, _acts = self.create_fire_module(x, int(16*compression), 'fire2', quantizer, use_constraint = use_constraint, use_bypass = use_bypass, **kwargs)
        activations.extend(_acts)

        x, _acts = self.create_fire_module(x, int(16*compression), 'fire3', quantizer, use_constraint = use_constraint, use_bypass = use_bypass, **kwargs)
        activations.extend(_acts)

        x = QQMaxPooling2D(quantizer, pool_size=(3,3), strides=(2,2), name = 'maxpool3', use_constraint = use_constraint)(x)
        # Append to activations
        activations += [x]

        # ADd Activation just for quantization purposes
        x = QQActivation(quantizer, 'logit', use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        x, _acts = self.create_fire_module(x, int(32*compression), 'fire4', quantizer, use_constraint = use_constraint, use_bypass = use_bypass, **kwargs)
        activations.extend(_acts)
        
        x, _acts = self.create_fire_module(x, int(32*compression), 'fire5', quantizer, use_constraint = use_constraint, use_bypass = use_bypass, **kwargs)
        activations.extend(_acts)

        x = QQMaxPooling2D(quantizer, pool_size=(3,3), strides=(2,2), name = 'maxpool5', use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        # Add Activation just for quantization purposes
        x = QQActivation(quantizer, 'logit', use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        x, _acts = self.create_fire_module(x, int(48*compression), 'fire6', quantizer, use_constraint = use_constraint, use_bypass = use_bypass, **kwargs)
        activations.extend(_acts)

        x, _acts = self.create_fire_module(x, int(48*compression), 'fire7', quantizer, use_constraint = use_constraint, use_bypass = use_bypass, **kwargs)
        activations.extend(_acts)
        
        x, _acts = self.create_fire_module(x, int(64*compression), 'fire8', quantizer, use_constraint = use_constraint, use_bypass = use_bypass, **kwargs)
        activations.extend(_acts)
        
        x, _acts = self.create_fire_module(x, int(64*compression), 'fire9', quantizer, use_constraint = use_constraint, use_bypass = use_bypass, **kwargs)
        activations.extend(_acts)

        if dropout_rate:
            x = QQDropout(quantizer, dropout_rate, use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]
        
        # Output
        # x = QQConv2D(quantizer, self.out_shape[0], (1,1), 
        #                     strides=(1,1), 
        #                     padding='valid',
        #                     use_bias = use_bias,
        #                     kernel_quantizer = quantizer,
        #                     bias_quantizer = quantizer,
        #                     kernel_regularizer=keras.regularizers.l2(1e-4),
        #                     name = 'conv10')(x)
        #x = keras.layers.GlobalAveragePooling2D(name='avgpool10')(x)

        x = QQFlatten(quantizer, use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        x = QQDense(quantizer, out_shape[0], 
                                kernel_quantizer = 'logit',
                                bias_quantizer = 'logit',
                                kernel_initializer='he_normal',
                                bias_initializer='zeros',
                                use_bias = use_bias,
                                use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]
                                
        out = QQSoftmax(quantizer, name='softmax')(x)
        # Append to activations
        activations += [out]

        return ip, out, activations


    """ Override NN method because otherwise this will be too long """
    def create_model_name_by_architecture(self):
        return 'squeezenet'

    
    """ Create fire module """
    def create_fire_module(self, x, nb_squeeze_filter, name, quantizer, use_constraint = True, use_bypass = False, use_bias = True, **kwargs):
        
        _acts = []

        nb_expand_filter = 4 * int(nb_squeeze_filter)

        squeeze = QQConv2D(quantizer, nb_squeeze_filter, (1,1), 
                            padding='same',
                            use_bias = use_bias,
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            kernel_initializer = 'he_normal',
                            bias_initializer = 'zeros',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                            name=f'{name}_squeeze',
                            use_constraint=use_constraint)(x)
        _acts += [squeeze]

        # QQApplyAlpha
        squeeze = QQApplyAlpha(quantizer)(squeeze)
        # Append to activations
        _acts += [squeeze]
        
        squeeze = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(squeeze)
        _acts += [squeeze]

        expand_1x1 = QQConv2D(quantizer, nb_expand_filter, (1,1), 
                            padding='same',
                            use_bias = use_bias,
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            kernel_initializer='he_normal',
                            bias_initializer='zeros',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                            name=f'{name}_expand_1x1',
                            use_constraint=use_constraint)(squeeze)
        _acts += [expand_1x1]

        # QQApplyAlpha
        expand_1x1 = QQApplyAlpha(quantizer)(expand_1x1)
        # Append to activations
        _acts += [expand_1x1]
        
        expand_1x1 = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(expand_1x1)
        _acts += [expand_1x1]

        expand_3x3 = QQConv2D(quantizer, nb_expand_filter, (3,3), 
                            padding='same',
                            use_bias = use_bias,
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            kernel_initializer='he_normal',
                            bias_initializer='zeros',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                            name=f'{name}_expand_3x3',
                            use_constraint=use_constraint)(squeeze)
        _acts += [expand_3x3]

        # QQApplyAlpha
        expand_3x3 = QQApplyAlpha(quantizer)(expand_3x3)
        # Append to activations
        _acts += [expand_3x3]
        
        expand_3x3 = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(expand_3x3)
        _acts += [expand_3x3]

        x_ret = tf.keras.layers.concatenate([expand_1x1, expand_3x3], axis=-1, name=f'{name}_concat')
        _acts += [x_ret]

        if use_bypass:
            x_ret = tf.keras.layers.add([x_ret, x], name=f'{name}_concatenate_bypass')
            _acts += [x_ret]
        
        return x_ret, _acts


""" Register models """
LEGACYMODELS = {'LeNet5': LeNet5, 
                'MobileNetV1': MobileNetV1, 
                'DSConv': DSConv, 
                'SqueezeNet': SqueezeNet, 
                'SqueezeNet11': SqueezeNet11}