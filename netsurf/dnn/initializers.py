import tensorflow as tf
from tensorflow.keras.initializers import Initializer


def get_initializers(quantizer: 'QuantizationScheme'):
    # Get initializers but do it
    init_map = {}
    init_map['constant']  = QConstant(quantizer)
    init_map['zeros']     = QZeros(quantizer)
    init_map['ones']      = QOnes(quantizer)
    init_map['random_normal'] = QRandomNormal(quantizer)
    init_map['random_uniform'] = QRandomUniform(quantizer)
    init_map['he_uniform'] = QHeUniform(quantizer)
    init_map['he_normal'] = QHeNormal(quantizer)
    init_map['glorot_uniform'] = QGlorotUniform(quantizer)
    init_map['glorot_normal'] = QGlorotNormal(quantizer)
    init_map['lecun_uniform'] = QLecunUniform(quantizer)
    init_map['lecun_normal'] = QLecunNormal(quantizer)
    return init_map

class QuantizedInitializer(Initializer):
    def __init__(self, quantizer: 'QuantizationScheme'):
        self.quantizer = quantizer

        # From the quantizer we can get the min and max values
        self.min_val = quantizer.min_value
        self.max_val = quantizer.max_value
    
    def __call__(self, weights):
        return tf.clip_by_value(weights, self.min_val, self.max_val)  # Clip values to the specified range

    def __repr__(self):
        return f'{self.__class__.__name__} <QuantizedInitializer>'
        

# This is only to be used internally
class __QHe(QuantizedInitializer):
    def __init__(self, quantizer: 'QuantizationScheme'):
        # Super sets max and min values
        super().__init__(quantizer)
    
    def __call__(self, fcn, factor, shape, dtype=None):
        fan_in = shape[0] if len(shape) > 1 else shape[-1]  # Calculate fan-in for He initialization
        stddev = tf.sqrt(factor / fan_in)  # He-normal stddev or limit in uniform
        # Call function 
        weights = fcn(stddev, dtype=dtype)
        # Call super for clipping
        return super().__call__(weights)

class QHeUniform(__QHe):
    def __init__(self, quantizer: 'QuantizationScheme'):
        # Super sets max and min values
        super().__init__(quantizer)

    def __call__(self, shape, dtype=None):
        fcn = lambda stddev, dtype=dtype: tf.random.uniform(shape, minval=self.min_val, maxval=self.max_val, dtype=dtype or tf.float32)
        return super().__call__(fcn, 6.0, shape, dtype)

class QHeNormal(__QHe):
    def __init__(self, quantizer: 'QuantizationScheme'):
        # Super sets max and min values
        super().__init__(quantizer)

    def __call__(self, shape, dtype=None):
        fcn = lambda stddev, dtype=dtype: tf.random.normal(shape, mean=(self.max_val + self.min_val)/2, stddev=stddev, dtype=dtype or tf.float32)
        return super().__call__(fcn, 2.0, shape, dtype)


""" This is to be used internally """
class __QGlorot(QuantizedInitializer):
    def __init__(self, quantizer: 'QuantizationScheme'):
        # Super sets max and min values
        super().__init__(quantizer)
    
    def __call__(self, fcn, factor, shape, dtype=None):
        fan_in = shape[0] if len(shape) > 1 else shape[-1]  # Calculate fan-in for He initialization
        fan_out = shape[1] if len(shape) > 1 else shape[-1]  # Calculate fan-out for He initialization
        fan_avg = (fan_in + fan_out) / 2
        stddev = tf.sqrt(factor / fan_avg)  # He-normal stddev or limit in uniform
        # Call function 
        weights = fcn(stddev, dtype=dtype)
        # Call super for clipping
        return super().__call__(weights)

""" Glorot Normal """
class QGlorotNormal(__QGlorot):
    def __init__(self, quantizer: 'QuantizationScheme'):
        # Super sets max and min values
        super().__init__(quantizer)

    def __call__(self, shape, dtype=None):
        fcn = lambda stddev, dtype=dtype: tf.random.normal(shape, mean=(self.max_val + self.min_val)/2, stddev=stddev, dtype=dtype or tf.float32)
        return super().__call__(fcn, 2.0, shape, dtype)

""" Glorot Uniform """
class QGlorotUniform(__QGlorot):
    def __init__(self, quantizer: 'QuantizationScheme'):
        # Super sets max and min values
        super().__init__(quantizer)

    def __call__(self, shape, dtype=None):
        fcn = lambda stddev, dtype=dtype: tf.random.uniform(shape, minval=self.min_val, maxval=self.max_val, dtype=dtype or tf.float32)
        return super().__call__(fcn, 6.0, shape, dtype)
    
""" Lecun Normal """
class QLecunNormal(__QHe):
    def __init__(self, quantizer: 'QuantizationScheme'):
        # Super sets max and min values
        super().__init__(quantizer)

    def __call__(self, shape, dtype=None):
        fcn = lambda stddev, dtype=dtype: tf.random.normal(shape, mean=(self.max_val + self.min_val)/2, stddev=stddev, dtype=dtype or tf.float32)
        return super().__call__(fcn, 1.0, shape, dtype)

""" Lecun Uniform """
class QLecunUniform(__QHe):
    def __init__(self, quantizer: 'QuantizationScheme'):
        # Super sets max and min values
        super().__init__(quantizer)

    def __call__(self, shape, dtype=None):
        fcn = lambda stddev, dtype=dtype: tf.random.uniform(shape, minval=self.min_val, maxval=self.max_val, dtype=dtype or tf.float32)
        return super().__call__(fcn, 3.0, shape, dtype)



""" Random Initializers """
class QRandomUniform(QuantizedInitializer):
    def __init__(self, quantizer: 'QuantizationScheme'):
        # Super sets max and min values
        super().__init__(quantizer)

    def __call__(self, shape, dtype=None):
        weights = tf.random.uniform(shape, minval=self.min_val, maxval=self.max_val, dtype=dtype or tf.float32)
        return super().__call__(weights)

""" Random Normal (note that this is exactly the same as TruncatedNormal) """
class QRandomNormal(QuantizedInitializer): 
    def __init__(self, quantizer: 'QuantizationScheme', stddev = 0.05):
        # Super sets max and min values
        super().__init__(quantizer)
        self.stddev = stddev

    def __call__(self, shape, dtype=None):
        weights = tf.random.normal(shape, mean=(self.max_val + self.min_val)/2, stddev=self.stddev, dtype=dtype or tf.float32)
        return super().__call__(weights)


""" Zeros (a bit of an overkill... but whatever) """
class QZeros(QuantizedInitializer):
    def __init__(self, quantizer: 'QuantizationScheme'):
        # Super sets max and min values
        super().__init__(quantizer)

    def __call__(self, shape, dtype=None):
        weights = tf.zeros(shape, dtype=dtype or tf.float32)
        return super().__call__(weights)

""" Ones (a bit of an overkill... but whatever) """
class QOnes(QuantizedInitializer):
    def __init__(self, quantizer: 'QuantizationScheme'):
        # Super sets max and min values
        super().__init__(quantizer)

    def __call__(self, shape, dtype=None):
        weights = tf.ones(shape, dtype=dtype or tf.float32)
        return super().__call__(weights)

""" Constant """
class QConstant(QuantizedInitializer):
    def __init__(self, quantizer: 'QuantizationScheme', value = 0.0):
        # Super sets max and min values
        super().__init__(quantizer)
        self.value = value

    def __call__(self, shape, dtype=None):
        weights = tf.constant(self.value, shape=shape, dtype=dtype or tf.float32)
        return super().__call__(weights)

