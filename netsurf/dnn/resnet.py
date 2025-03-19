""" Numpy """
import numpy as np

""" Tensorflow """
import tensorflow as tf

# Custom layers 
from .models import QModel

""" Import custom layers """
from .layers import QQDense, QQConv1D, QQConv2D, QQDepthwiseConv2D, QQSeparableConv2D, \
                    QQGlobalAveragePooling2D, QQMaxPooling2D, QQAveragePooling2D, \
                    QQActivation, QQSoftmax, \
                    QQBatchNormalization, \
                    QQFlatten, QQDropout, QQReshape, \
                    QQLSTM, QQApplyAlpha


""" Resnet Basic blocks """
class ResnetBasicBlock(tf.keras.layers.Layer):
    expansion = 1
    def __init__(self, planes, quantizer, strides=1, use_constraint = True, **kwargs):
        
        super().__init__()

        self.planes = planes 
        self.strides = strides
        self.quantizer = quantizer
        self.use_constraint = use_constraint
        
    def __call__(self, x):
        
        # get params
        planes = self.planes 
        strides = self.strides
        quantizer = self.quantizer
        use_constraint = self.use_constraint

        _acts = []

        # Create layers 
        out = QQConv2D(quantizer, planes, kernel_size=3, 
                                            strides=strides, padding='same', 
                                            use_bias=False,
                                            kernel_quantizer = 'logit',
                                            bias_quantizer = 'logit',
                                            kernel_initializer='he_normal',
                                            bias_initializer='zeros',
                                            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                            use_constraint=use_constraint)(x)
        _acts += [out]

        out = QQBatchNormalization(quantizer, beta_quantizer = 'po2', 
                                            gamma_quantizer = 'relu_po2', 
                                            mean_quantizer = 'po2', 
                                            variance_quantizer = 'relu_po2_quadratic',
                                            beta_initializer = 'zeros', 
                                            gamma_initializer = 'ones', 
                                            moving_mean_initializer = 'zeros', 
                                            moving_variance_initializer = 'ones',
                                            use_constraint=use_constraint)(out)
        _acts += [out]
        
        out = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(out)
        _acts += [out]

        out = QQConv2D(quantizer, planes, kernel_size=3, 
                            strides=1, padding='same', 
                            use_bias=False,
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            kernel_initializer='he_normal',
                            bias_initializer='zeros',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                            use_constraint=use_constraint)(out)
        _acts += [out]

        out = QQBatchNormalization(quantizer, beta_quantizer = 'po2', 
                                            gamma_quantizer = 'relu_po2', 
                                            mean_quantizer = 'po2', 
                                            variance_quantizer = 'relu_po2_quadratic',
                                            beta_initializer = 'zeros', 
                                            gamma_initializer = 'ones', 
                                            moving_mean_initializer = 'zeros', 
                                            moving_variance_initializer = 'ones',
                                            use_constraint=use_constraint)(out)
        _acts += [out]

        out = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(out)
        _acts += [out]

        # Shortcut 
        if strides != 1 or x.shape[-1] != planes*self.expansion:
            x = QQConv2D(quantizer, planes, kernel_size=1, strides=strides, use_bias=False,
                                    kernel_quantizer = 'logit',
                                    bias_quantizer = 'logit',
                                    kernel_initializer='he_normal',
                                    bias_initializer='zeros',
                                    kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                    use_constraint=use_constraint)(x)
            _acts += [x]

            x = QQBatchNormalization(quantizer, beta_quantizer = 'po2', 
                                            gamma_quantizer = 'relu_po2', 
                                            mean_quantizer = 'po2', 
                                            variance_quantizer = 'relu_po2_quadratic',
                                            beta_initializer = 'zeros', 
                                            gamma_initializer = 'ones', 
                                            moving_mean_initializer = 'zeros', 
                                            moving_variance_initializer = 'ones',
                                            use_constraint=use_constraint)(x)
            _acts += [x]
        
        out = tf.keras.layers.add([out, x])
        _acts += [out]

        out = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(out)
        _acts += [out]
        return out, _acts


""" Build ResnetV1 """
class ResNetV1(QModel):
    def __init__(self, quantizer: 'QuantizationScheme', in_shape = (32, 32, 3), out_shape = (10,),
                    loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], 
                    type = 'classification',
                    **kwargs):
        """."""
        super().__init__(quantizer, in_shape, out_shape, loss = loss, optimizer = optimizer,
                        metrics = metrics, type = type, **kwargs)
        
        # Build model now
        self.build_model(quantizer, in_shape, out_shape, loss = loss, metrics = metrics, type = type, **kwargs)

        # Make sure we set original metrics here
        self._metrics = metrics

    def build_model(self, quantizer : 'QuantizationScheme', in_shape, out_shape, **kwargs):
        super().build_model(*self.resnetv1(quantizer, in_shape, out_shape, **kwargs), **kwargs)

    """ Build model method """
    def resnetv1(self, quantizer: 'QuantizationScheme', in_shape, out_shape, use_constraint = True, use_bias = True, **kwargs):

        # Input layer, change kernel size to 7x7 and strides to 2 for an official resnet
        ip = tf.keras.layers.Input(shape=in_shape)
        # Init acts
        activations = []

        # First conv group
        x = QQConv2D(quantizer, 16,
                            kernel_size=3,
                            strides=1,
                            use_bias = use_bias,
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            padding='same',
                            kernel_initializer='he_normal',
                            use_constraint=use_constraint,
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4))(ip)
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

        #x = MaxPooling2D(pool_size=(2, 2))(x) # uncomment this for official resnet model

        # Loop thru stacks
        params =  [ (16,3,3,1,1), (32,3,3,2,1), (64,3,3,2,1) ]

        for f, ks1, ks2, st1, st2 in params: 
            # Weight layers
            y = QQConv2D(quantizer, f,
                            kernel_size=ks1,
                            strides=st1,
                            padding='same',
                            kernel_initializer='he_normal',
                            use_bias = use_bias,
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            use_constraint=use_constraint,
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
            # Append to activations
            activations += [y]

            # QQApplyAlpha
            y = QQApplyAlpha(quantizer)(y)
            # Append to activations
            activations += [y]
            
            y = QQBatchNormalization(quantizer, beta_quantizer = 'po2', 
                                            gamma_quantizer = 'relu_po2', 
                                            mean_quantizer = 'po2', 
                                            variance_quantizer = 'relu_po2_quadratic',
                                            beta_initializer = 'zeros', 
                                            gamma_initializer = 'ones', 
                                            moving_mean_initializer = 'zeros', 
                                            moving_variance_initializer = 'ones', 
                                            use_constraint= use_constraint)(y)
            # Append to activations
            activations += [y]
            
            y = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(y)
            # Append to activations
            activations += [y]

            y = QQConv2D(quantizer, f,
                            kernel_size=ks1,
                            strides=st2,
                            padding='same',
                            use_bias = use_bias,
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            kernel_initializer='he_normal',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                            use_constraint=use_constraint)(y)
            # Append to activations
            activations += [y]
            
            y = QQBatchNormalization(quantizer, beta_quantizer = 'po2', 
                                            gamma_quantizer = 'relu_po2', 
                                            mean_quantizer = 'po2', 
                                            variance_quantizer = 'relu_po2_quadratic',
                                            beta_initializer = 'zeros', 
                                            gamma_initializer = 'ones', 
                                            moving_mean_initializer = 'zeros', 
                                            moving_variance_initializer = 'ones', 
                                            use_constraint= use_constraint)(y)
            # Append to activations
            activations += [y]

            # Adjust for change in dimension due to stride in identity
            if st1 > 1:
                x = QQConv2D(quantizer, f,
                            kernel_size=1,
                            strides=st1,
                            padding='same',
                            use_bias = use_bias,
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            kernel_initializer='he_normal',
                            use_constraint=use_constraint,
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
                # Append to activations
                activations += [x]

                # QQApplyAlpha
                x = QQApplyAlpha(quantizer)(x)
                # Append to activations
                activations += [x]
    
            # Overall residual, connect weight layer and identity paths
            x = tf.keras.layers.add([x, y]) 
            # Append to activations
            activations += [x]

            x = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]


        # Final classification layer.
        pool_size = int(np.amin(x.shape[1:3]))
        # TODO: THIS LAYER IS RAISING A DTYPE ERROR, VERY ANNOYING
        ##x = QQAveragePooling2D(quantizer, pool_size=pool_size, use_constraint=use_constraint, dtype = tf.float32)(tf.cast(x, tf.float32))
        x = tf.keras.layers.AveragePooling2D(pool_size=pool_size)(x)
        
        # Append to activations
        activations += [x]

        y = QQFlatten(quantizer, use_constraint=use_constraint)(x)
        # Append to activations
        activations += [y]

        y = QQDense(quantizer, out_shape[0],
                                use_bias = use_bias,
                                kernel_quantizer = 'logit',
                                bias_quantizer = 'logit',
                                kernel_initializer='he_normal',
                                use_constraint=use_constraint)(y)
        # Append to activations
        activations += [y]

        out = QQSoftmax(quantizer, use_constraint=use_constraint)(y)
        # Append to activations
        activations += [out]

        return ip, out, activations
    
    """ Override NN method because otherwise this will be too long """
    def create_model_name_by_architecture(self):
        return 'resnetv1'

""" RESNET BLOCK, used for ResNet18 and ResNet34 """
def ResNet(block, num_blocks, quantizer, in_shape, out_shape,
           use_bias = True, use_constraint = True, **kwargs):

    # Init activations
    activations = []

    # Input layer 
    ip = tf.keras.layers.Input(shape=in_shape)

    # Create layers
    x = QQConv2D(quantizer, 64, kernel_size=3, strides=1, padding = 'same', use_bias=False,
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            kernel_initializer='he_normal',
                            bias_initializer='zeros',
                            use_constraint=use_constraint,
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4))(ip)
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

    x, _acts = _make_layer(x, block, 64, num_blocks[0], quantizer, use_constraint = use_constraint, strides=1, **kwargs)
    activations.extend(_acts)

    x, _acts = _make_layer(x, block, 128, num_blocks[1], quantizer, use_constraint=use_constraint, strides=2, **kwargs)
    activations.extend(_acts)

    x, _acts = _make_layer(x, block, 256, num_blocks[2], quantizer, use_constraint=use_constraint, strides=2, **kwargs)
    activations.extend(_acts)

    x, _acts = _make_layer(x, block, 512, num_blocks[3], quantizer, use_constraint=use_constraint, strides=2, **kwargs)
    activations.extend(_acts)

    x = QQGlobalAveragePooling2D(quantizer)(x)
    # Append to activations
    activations += [x]

    #x = keras.layers.AveragePooling2D(pool_size=(4,4))(x)
    #x = keras.layers.Flatten()(x)

    # Reshape here??? before dense ??
    x = QQDense(quantizer, out_shape[0], 
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            kernel_initializer='he_normal',
                            bias_initializer='zeros',
                            use_constraint=use_constraint,
                            use_bias = use_bias)(x)
    # Append to activations
    activations += [x]

    out = QQSoftmax(quantizer, use_constraint=use_constraint)(x)
    # Append to activations
    activations += [out]

    return ip, out, activations

def _make_layer(x, block, planes, num_blocks, quantizer, use_constraint = True, strides = 1, **kwargs):
    _strides = [strides] + [1]*(num_blocks-1)
    _acts = []
    for stride in _strides:
        x, _act = block(planes, quantizer, stride, use_constraint = use_constraint, **kwargs)(x)
        planes = planes*block.expansion
        _acts.extend(_act)
    return x, _acts
    
""" resnet 18 """
class ResNet18(QModel):
    def __init__(self, quantizer, in_shape = (32, 32, 3), out_shape = (10,),
                    optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'], 
                    type = 'classification', **kwargs):
        """."""
        super().__init__(quantizer, in_shape, out_shape, optimizer = optimizer, loss = loss, 
                         metrics = metrics, type = type, **kwargs)
        
        # Now build
        self.build_model(quantizer, in_shape, out_shape, loss = loss, metrics = metrics, type = type, **kwargs)

        # Make sure we set original metrics here
        self._metrics = metrics

    # Build model now 
    def build_model(self, quantizer, in_shape, out_shape, **kwargs):
        super().build_model(*self.resnet18(quantizer, in_shape, out_shape, **kwargs), **kwargs)
    
    """ build model method """
    def resnet18(self, in_shape, out_shape, *args, **kwargs):
        # Call model composer 
        return ResNet(ResnetBasicBlock, [2, 2, 2, 2], in_shape, out_shape, *args, **kwargs)
    
    """ Override NN method because otherwise this will be too long """
    def create_model_name_by_architecture(self):
        return 'resnet18'
    
     
""" resnet 34 """
class ResNet34(QModel):
    def __init__(self, quantizer, in_shape = (32, 32, 3), out_shape = (10,),
                    optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'], 
                    type = 'classification', **kwargs):
        """."""
        super().__init__(quantizer, in_shape, out_shape, optimizer = optimizer, loss = loss, 
                         metrics = metrics, type = type, **kwargs)
        
        # Now build
        self.build_model(quantizer, in_shape, out_shape, loss = loss, metrics = metrics, type = type, **kwargs)

        # Make sure we set original metrics here
        self._metrics = metrics

    # Build model now 
    def build_model(self, quantizer: 'QuantizationScheme', in_shape, out_shape, **kwargs):
        super().build_model(*self.resnet34(quantizer, in_shape, out_shape, **kwargs), **kwargs)

    """ build model method """
    def resnet34(self, in_shape, out_shape, *args, **kwargs):
        # Call model composer 
        return ResNet(ResnetBasicBlock, [3, 4, 6, 3], in_shape, out_shape, *args, **kwargs)
    
    """ Override NN method because otherwise this will be too long """
    def create_model_name_by_architecture(self):
        return 'resnet34'

RESNETMODELS = {'ResNet18': ResNet18, 'ResNet34': ResNet34, 'ResNetV1': ResNetV1}
