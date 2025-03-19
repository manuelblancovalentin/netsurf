import tensorflow as tf
NUM_CLASSES = 3

from .models import QModel

""" Import custom layers """
from .layers import QQDense, QQConv1D, QQConv2D, QQDepthwiseConv2D, QQSeparableConv2D, \
                    QQGlobalAveragePooling2D, QQMaxPooling2D, \
                    QQActivation, QQSoftmax, \
                    QQBatchNormalization, \
                    QQFlatten, QQDropout, QQReshape, \
                    QQLSTM, QQApplyAlpha

""" EXTRACTED FROM https://github.com/KastnerRG/edge-nns/blob/main/smart-pixel/models.py"""

""" SmartPixel base """
class SmartPixel(QModel):
    def __init__(self, quantizer, in_shape = (13,), out_shape = (NUM_CLASSES,), *args,
                    use_bypass=False, dropout_rate=None, compression=1.0,
                    loss = 'categorical_crossentropy', metrics = ['accuracy'], 
                    type = 'classification', **kwargs):
        """."""
        super().__init__(quantizer, in_shape, out_shape, *args, loss = loss, metrics = metrics, 
                         type = type,
                         **kwargs)

        # Set flags 
        self.use_bypass = use_bypass
        self.dropout_rate = dropout_rate
        self.compression = compression

        # Build model
        self.build_model(quantizer, in_shape, out_shape, loss = loss, metrics = metrics, type = type, **kwargs)
    
        # Make sure we set original metrics here
        self._metrics = metrics

    def __evaluate_classification(self, x, y, **kwargs):
        """."""
        return super().evaluate_classification(x, y, xlog = False, ylog = False, **kwargs)

        
""" Dense baseline """
class SmartPixelDense(SmartPixel):
    def build_model(self, quantizer: 'QuantizationScheme', in_shape, out_shape, **kwargs):
        """."""
        super().build_model(*self.qkeras_dense_model(quantizer, in_shape, out_shape, **kwargs), **kwargs)
    
    # Dense baseline
    def qkeras_dense_model(self, quantizer, in_shape, out_shape, dense_width = 58, use_constraint = True, **kwargs):
        """
        QKeras model
        """
        x = ip = tf.keras.layers.Input(in_shape, name="input1")

        # Activations
        activations = []
        
        x = QQDense(quantizer,
                dense_width,
                kernel_quantizer='logit',
                bias_quantizer='logit',
                kernel_initializer='glorot_normal',
                bias_initializer='zeros',
                name="dense1",
                use_constraint=use_constraint)(ip)
        # Append to activations
        activations += [x]

        # Alpha
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
        
        x = QQDense(quantizer, 
            out_shape[0],
            kernel_quantizer='logit',
            bias_quantizer='logit',
            name="dense2",
            use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        # Softmax
        out = QQSoftmax(quantizer, use_constraint=use_constraint)(x)
        # Append to activations
        activations += [out]

        return ip, out, activations

    def create_model_name_by_architecture(self):
        return 'smartpixeldense'

""" Dense large """
class SmartPixelDenseLarge(SmartPixel):
    def build_model(self, quantizer: 'QuantizationScheme', in_shape, out_shape, **kwargs):
        super().build_model(*self.dense_model_large(quantizer, in_shape, out_shape, **kwargs), **kwargs)

    # Dense large
    def dense_model_large(self, quantizer: 'QuantizationScheme', in_shape, out_shape, dense_width=32, use_constraint=True, **kwargs):
        """
        QKeras model
        """
        x = ip = tf.keras.layers.Input(in_shape, name="input1")

        # Init activations
        activations = []

        # 3 blocks
        for i in range(3):

            x = QQDense(quantizer, 
                dense_width,
                kernel_quantizer='logit',
                bias_quantizer='logit',
                name=f"dense{i}",
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

        # Final output
        x = QQDense(quantizer, out_shape[0],
                    kernel_quantizer = 'logit',
                    bias_quantizer = 'logit',
                    name = f'dense_out',
                    use_constraint=use_constraint)(x)
        
        # Append to activations
        activations += [x]

        # Softmax
        out = QQSoftmax(quantizer, use_constraint=use_constraint)(x)
        # Append to activations
        activations += [out]
        
        return ip, out, activations

    def create_model_name_by_architecture(self):
        return 'smartpixeldenselarge'


SMARTPIXMODELS = {'SmartPixelDense': SmartPixelDense, 'SmartPixelDenseLarge': SmartPixelDenseLarge} 
