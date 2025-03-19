import tensorflow as tf 

""" Import qkeras """
import qkeras

""" Import initializers and quantizers """
from .quantizers import get_quantizers
from .initializers import get_initializers

# Custom plots
from .. import utils

""" Import pergamos """
import pergamos as pg

""" Import pandas """
import pandas as pd

""" Import numpy """
import numpy as np

""" Matplotlib """
import matplotlib.pyplot as plt

class QConstraint(tf.keras.constraints.Constraint):
    _ICON = "🔗"
    """Clips layer weights to a given min/max range after each training step."""
    def __init__(self, quantizer: 'QuantizationScheme'):
        self.quantizer = quantizer
        self.min_val = quantizer.min_value
        self.max_val = quantizer.max_value

    def __call__(self, weights):
        return tf.clip_by_value(weights, self.min_val, self.max_val)

    def get_config(self):
        return {"min_val": self.min_val, "max_val": self.max_val}

    def __repr__(self):
        s = f'{self._ICON} <QConstraint ({self.quantizer._scheme_str})> obj @ ({hex(id(self))})'
        s += f'    Min value: {self.min_val}\n'
        s += f'    Max value: {self.max_val}\n'
        return s
    
    def __str__(self):
        return f'QConstraint ({self.quantizer._scheme_str}) [min: {self.min_val}, max: {self.max_val}]'

    @pg.printable
    def html(self, include_emojis: bool = True):
        return str(self)


""" CUSTOM IMPLEMENTATION OF QKERAS LAYERS, 
        This is because we found out that qkeras DOES NOT ENFORCE quantization,
        it simply calculates the alpha values. 

        WE NEED THIS TO BE ENFORCED THO, SO LET'S WRAP THE QKERAS LAYERS IN
        OUR CUSTOM WRAPPERS THAN ENSURE THAT:
        (1) WE USE THE CORRECT KERNEL QUANTIZER
        (2) WE USE THE CORRECT BIAS QUANTIZER
        (3) WE USE THE CORRECT ACTIVATION QUANTIZER
        (4) WE USE THE CORRECT BATCH NORMALIZATION QUANTIZERS
        (5) WE USE THE CORRECT KERNEL INITIALIZER
        (6) WE USE THE CORRECT BIAS INITIALIZER
        (7) WE USE THE CORRECT REGULARIZERS (? NOT IMPLEMENTED YET)

"""
# This class is only supposed to be used internally
class _QQLayer:
    _QPROPS = []
    _ICON = "📓"
    def __init__(self, quantizer: 'QuantizationScheme', use_constraint: bool = True, **kwargs):
        super().__init__()
        self.quantizer = quantizer
        self.max_value = self.quantizer.max_value
        self.min_value = self.quantizer.min_value
        self.range = self.max_value - self.min_value
        self.use_constraint = use_constraint

        # Get maps
        quantizer_map = get_quantizers(quantizer)
        init_map = get_initializers(quantizer)

        maps = {'initializer': init_map, 'quantizer': quantizer_map}

        # Initialize the quantizers and initializers according to _PROPS
        kws = {}
        for prop in self._QPROPS:
            for category, qmap in maps.items():
                cat_name = f'{prop}_{category}'
                if cat_name in kwargs:
                    kws[cat_name] = qmap[kwargs.pop(cat_name)]
                    
        if use_constraint:
            for prop in self._QPROPS:
                kws[f'{prop}_constraint'] = QConstraint(quantizer)

        # Set in the object as kwargs
        self._qkwargs = kws
    
    def __repr__(self):
        cname = f"{self.__class__.__name__}"
        cname += " (" + self.quantizer._scheme_str + ")" if (self.quantizer.__class__.__name__ == 'QuantizationScheme') else ""
        s = f'{self._ICON} <QQLayer {cname} obj @ ({hex(id(self))})\n'
        s += f'    Built: {"✅" if self.built else "❌"}\n'
        s += f'    Dtype: {self.dtype}\n'
        s += f'    Max value: {self.max_value}\n'
        s += f'    Min value: {self.min_value}\n'
        s += f'    Range: {self.range}\n'
        if hasattr(self, 'activity_regularizer'):
            s += f'    Activity Regularizer: {self.activity_regularizer}\n'
        if hasattr(self, 'losses') and self.losses:
            s += f'    Losses: {self.losses}\n'
        if hasattr(self, 'metrics') and self.metrics:
            s += f'    Metrics: {self.metrics}\n'

        # Layer specific stuff here 
        for prop in self._QPROPS:
            if hasattr(self, prop):
                s += f'    {prop}:\n'
                for k in ['initializer', 'quantizer', 'constraint', 'regularizer', 'range']:
                    if hasattr(self, f'{prop}_{k}'):
                        s += f'        {k}: {getattr(self, f"{prop}_{k}")}\n'

        return s
    
    
    # Only to be called by children
    def _serialize(self, include_emojis: bool = True, flatten = True):

        d = {}
        if hasattr(self, 'losses'):
            if self.losses:
                if len(self.losses) > 0:
                    d['❌ Losses'] = ", ".join([str(l) for l in self.losses])
        if hasattr(self, 'metrics'):
            if self.metrics:
                if len(self.metrics) > 0:
                    d['📏 Metrics'] = ", ".join([str(m) for m in self.metrics])
        if hasattr(self, 'activity_regularizer'):
            d['📏 Activity Regularizer'] = str(self.activity_regularizer)

        # Get layers input and output shape 
        def get_ly_shape_str(layer, prop = 'input'):
            if hasattr(layer,f'{prop}_shape'):
                in_shape = layer.input_shape
                if not isinstance(in_shape, list):
                    in_shape = [in_shape]
                in_shape = ["(?," + ",".join(list(map(str,sh[1:]))) + ")" if sh else ("?",) for sh in in_shape ]
                if len(in_shape) == 1:
                    in_shape = str(in_shape[0])
                else:
                    in_shape = ", ".join(in_shape)
            else:
                in_shape = "(?,)"
            return in_shape
        
        # Input
        in_shape = get_ly_shape_str(self, 'input')
        # Same for output
        out_shape = get_ly_shape_str(self, 'output')

        d = dict({
            "❡ Name": self.name,
            "📓 Class": self.__class__.__name__,
            "▶ Input shape": in_shape,
            "◀ Output shape": out_shape,
            "🛠️ Built": "✅" if self.built else "❌",
            "👾 Dtype": self.dtype,
            "🔗 Min value": self.min_value,
            "🔗 Max value": self.max_value}, 
            **d)
        

        props = pd.DataFrame([d]).T
        props = {'Layer properties': props, 
                 '__options': {'collapsible': False}}
        content = [props]

        # Add layer specific containers here 
        cats = []
        for prop in self._QPROPS:
            subcat = {'__options': {'collapsible': True}}
            _conts = []
            if hasattr(self, prop):
                subcat_prop_table = {}
                arr = getattr(self, prop)
                if hasattr(arr, 'numpy'):
                    arr = arr.numpy()
                if hasattr(arr, 'shape'):
                    subcat_prop_table = {f'📦 {prop}_size': getattr(self, prop).shape}
                for k, ik in zip(['initializer', 'quantizer', 'constraint', 'regularizer', 'range'],
                                 ['🏁 ', '🧮 ', '🔗 ', '🎚️ ', '📐 ']):
                    if hasattr(self, f'{prop}_{k}'):
                        subcat_prop_table[f'{ik}{prop}_{k}'] = str(getattr(self, f"{prop}_{k}"))
                subcat_prop_table = pd.DataFrame([subcat_prop_table]).T
                # Now the numpy array itself 
                if isinstance(arr, np.ndarray):
                    datatable = {'Values': arr, 
                                '__options': {'collapsible': True}}
                    
                    # We can also add a plot of the weight's histogram 
                    fig, ax = plt.subplots(1,1, figsize = (7,7))
                    fig, ax = utils.plot.plot_quantized_histogram(arr, self.quantizer, ax = ax,
                                                                figsize = (7,7), bins = None, 
                                                                min_value = None, max_value = None,
                                                                title = None, legend = True, 
                                                                xlabel = None, ylabel = None, flatten = flatten)
                    
                    # pg.Plot inside container
                    wplot = {'📊 Histogram': pg.Plot(fig),
                            '__options': {'collapsible': True}}

                    _conts = [subcat_prop_table, datatable, wplot]
            subcat[prop.capitalize()] = _conts
            cats.append(subcat)

        #content.extend(cats)

        return content, cats

    def serialize(self, include_emojis: bool = True, **kwargs):
        r = self._serialize(include_emojis = include_emojis, **kwargs)
        r[0].extend(r[1])
        return r[0]
        # # Concatenate into a single list
        #r[0]['Layer properties'] = pd.concat([r[0]['Layer properties'], r[1]])
        
        # # Concatenate into a single list
        # conts.extend(cats)

        #return r[0].extend(r[1])
        #return self._serialize(include_emojis = include_emojis)

    @pg.printable
    def html(self, include_emojis: bool = True, **kwargs):
        return self.serialize(include_emojis=True, **kwargs)
    
""" This layer learns separate alpha/beta parameters for each channel (or dimension), which
    will be used later to ensure that activations stay within quantization range.
    The output of this layer is:
        output = input * alpha + beta
    where alpha and beta are learned parameters.
"""
class QQApplyAlpha(tf.keras.layers.Layer):
    _QPROPS = ['alpha', 'beta']
    _ICON = "🧿"
    def __init__(self, quantizer: 'QuantizationScheme', alpha=1.0, beta=0.0, reg_factor=1.0, axis=-1, **kwargs):
        """
        A layer that learns separate alpha/beta parameters for each channel (or dimension).
        
        Args:
            initial_alpha (float): Initial value for the scaling parameter.
            initial_beta (float): Initial value for the bias parameter.
            axis (int): The axis (dimension) along which alpha/beta are applied.
                        Typically -1 for 'channels last'.
            **kwargs: Additional keyword arguments passed to tf.keras.layers.Layer.
        """
        super().__init__(**kwargs)
        self.initial_alpha = alpha
        self.initial_beta = beta
        self.axis = axis
        self.quantizer = quantizer
        self.target_range = (self.quantizer.min_value, self.quantizer.max_value)
        self.reg_factor = reg_factor

    def build(self, input_shape):
        """
        Create the trainable weights. If axis = -1 and input_shape = (batch, ..., channels),
        then alpha and beta will each have shape = (channels,).
        """
        # For a typical NHWC format, axis = -1 => shape = (# of channels,).
        # If your input is 2D (batch, features), shape would be (features,).
        param_dim = input_shape[self.axis]

        # param_dim could be None if it's not known at build time; in that case, 
        # you need dynamic shape handling. For a typical static shape, we proceed:
        if param_dim is None:
            raise ValueError("The size of the specified axis must be known (not None).")

        # Create alpha/beta with shape (param_dim,).
        self.alpha = self.add_weight(
            name="alpha",
            shape=(param_dim,),
            initializer = tf.keras.initializers.Constant(self.initial_alpha),
            trainable=True
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(param_dim,),
            initializer = tf.keras.initializers.Constant(self.initial_beta),
            trainable=True
        )

        super().build(input_shape)

    def call(self, inputs):
        """
        Multiply inputs by alpha and add beta, broadcasting across all other axes.
        """
        # Compute the scaled and shifted output.
        output = self.alpha * inputs + self.beta
        
        # Retrieve the desired output range.
        target_min, target_max = self.target_range
        
        # Compute a penalty for values above target_max or below target_min.
        penalty_upper = tf.maximum(0.0, output - target_max)
        penalty_lower = tf.maximum(0.0, target_min - output)
        loss_penalty = tf.reduce_mean(penalty_upper + penalty_lower)
        
        # Add the penalty term (scaled by reg_factor) to the overall model loss.
        self.add_loss(self.reg_factor * loss_penalty)
        
        return output

    def get_config(self):
        """
        Allows the layer to be serialized (e.g., when saving a model).
        """
        config = super().get_config()
        config.update({
            "alpha": self.initial_alpha,
            "beta": self.initial_beta,
            "axis": self.axis,
            "quantizer": self.quantizer._scheme_str,
            "reg_factor": self.reg_factor,
        })
        return config
    
    def serialize(self, include_emojis: bool = True, flatten = True):
        
        # Get layers input and output shape 
        def get_ly_shape_str(layer, prop = 'input'):
            if hasattr(layer,f'{prop}_shape'):
                in_shape = layer.input_shape
                if not isinstance(in_shape, list):
                    in_shape = [in_shape]
                in_shape = ["(?," + ",".join(list(map(str,sh[1:]))) + ")" if sh else ("?",) for sh in in_shape ]
                if len(in_shape) == 1:
                    in_shape = str(in_shape[0])
                else:
                    in_shape = ", ".join(in_shape)
            else:
                in_shape = "(?,)"
            return in_shape
       
        # Input
        in_shape = get_ly_shape_str(self, 'input')
        # Same for output
        out_shape = get_ly_shape_str(self, 'output')

        d = {
            "❡ Name": self.name,
            "📓 Class": self.__class__.__name__,
            "▶ Input shape": in_shape,
            "◀ Output shape": out_shape,
            "🛠️ Built": "✅" if self.built else "❌",
            "👾 Dtype": self.dtype,
            "🎲 Initial alpha": self.initial_alpha,
            "🎲 Initial beta": self.initial_beta}
        
        props = pd.DataFrame([d]).T
        props = {'Layer properties': props, 
                 '__options': {'collapsible': False}}

        # Add layer specific containers here 
        cats = []
        for prop in ['alpha', 'beta']:
            subcat = {'__options': {'collapsible': True}}
            _conts = []
            if hasattr(self, prop):
                subcat_prop_table = {}
                arr = getattr(self, prop)
                if hasattr(arr, 'numpy'):
                    arr = arr.numpy()
                if hasattr(arr, 'shape'):
                    subcat_prop_table = {f'📦 {prop}_size': getattr(self, prop).shape}
                subcat_prop_table = pd.DataFrame([subcat_prop_table]).T
                # Now the numpy array itself 
                if isinstance(arr, np.ndarray):
                    datatable = {'Values': arr, 
                                '__options': {'collapsible': True}}
                    
                    # We can also add a plot of the weight's histogram 
                    fig, ax = plt.subplots(1,1, figsize = (7,7))
                    fig, ax = utils.plot.plot_quantized_histogram(arr, self.quantizer, ax = ax,
                                                                figsize = (7,7), bins = None, 
                                                                min_value = None, max_value = None,
                                                                title = None, legend = True, 
                                                                xlabel = None, ylabel = None, flatten = flatten)
                    
                    # pg.Plot inside container
                    wplot = {'📊 Histogram': pg.Plot(fig),
                            '__options': {'collapsible': True}}

                    _conts = [subcat_prop_table, datatable, wplot]
            subcat[prop.capitalize()] = _conts
            cats.append(subcat)

        content = [props, cats]

        return content

    @pg.printable
    def html(self, include_emojis: bool = True, **kwargs):
        return self.serialize(include_emojis=True, **kwargs)

  
class QQDense(qkeras.QDense, _QQLayer):
    _QPROPS = ['kernel', 'bias']
    _ICON = "📕"
    def __init__(self, quantizer, *args, kernel_quantizer: str = 'logit', bias_quantizer: str = 'logit', 
                 kernel_initializer: str = 'glorot_normal', bias_initializer: str = 'zeros', 
                 use_constraint: bool = True,
                 **kwargs):
        # After calling this, we will have the actual quantizer objects in "self._qkwargs"
        _QQLayer.__init__(self, quantizer,  
                          kernel_quantizer = kernel_quantizer, bias_quantizer = bias_quantizer, 
                          kernel_initializer = kernel_initializer, bias_initializer = bias_initializer,
                          use_constraint = use_constraint)

        # Finally just call the parent class
        qkeras.QDense.__init__(self, *args, **self._qkwargs, **kwargs)
    
    def serialize(self, include_emojis: bool = True, **kwargs):
        conts, cats = super()._serialize(include_emojis = include_emojis, **kwargs)
        # Add this layer specific stuff to conts
        # layer_specific = {'dense': ['units', 'activation', 'use_bias', 'use_constraint',]}

        d = dict({
            "🎱 Units": self.units,
            "📗 Activation": self.activation,
            "⚓️ Use bias": "✅" if self.use_bias else "❌",
            "🔗 Use constraint": "✅" if self.use_constraint else "❌"})
        
        props = pd.DataFrame([d]).T
        conts[0]['Layer properties'] = pd.concat([conts[0]['Layer properties'], props])
        
        # Concatenate into a single list
        conts.extend(cats)
        return conts

class QQBatchNormalization(qkeras.QBatchNormalization, _QQLayer):
    _QPROPS = ['beta', 'gamma']
    _ICON = "📒"
    def __init__(self, quantizer, *args, 
                 beta_quantizer: str = 'po2', gamma_quantizer: str = 'relu_po2', 
                 mean_quantizer: str = 'po2', variance_quantizer: str = 'relu_po2_quadratic', 
                 beta_initializer: str = 'zeros', gamma_initializer: str = 'ones',
                 moving_mean_initializer: str = 'zeros', moving_variance_initializer: str = 'ones',
                 use_constraint: bool = True, 
                 **kwargs):
        
        # After calling this, we will have the actual quantizer objects in "self._qkwargs"
        _QQLayer.__init__(self, quantizer, beta_quantizer = beta_quantizer, gamma_quantizer = gamma_quantizer,
                         beta_initializer = beta_initializer, gamma_initializer = gamma_initializer,
                         use_constraint = use_constraint)

        # Unfortunalety, and for this specific case, qkeras is NOT consitent enough,
        # and they called "variance_quantizer" and "mean_quantizer", and then
        # "moving_mean_initializer" and "moving_variance_initializer"...
        # So let's set those two up manually

        # Get quantizers map
        quantizer_map = get_quantizers(quantizer)
        init_map = get_initializers(quantizer)

        mean_quantizer = quantizer_map[mean_quantizer]
        variance_quantizer = quantizer_map[variance_quantizer]
        moving_mean_initializer = init_map[moving_mean_initializer]
        moving_variance_initializer = init_map[moving_variance_initializer]

        qkeras.QBatchNormalization.__init__(self, *args, **self._qkwargs,
                                            mean_quantizer = mean_quantizer, variance_quantizer = variance_quantizer,
                                            moving_mean_initializer = moving_mean_initializer, moving_variance_initializer = moving_variance_initializer,
                                            **kwargs)


class QQConv1D(qkeras.QConv1D, _QQLayer):
    _QPROPS = ['kernel', 'bias']
    _ICON = "📕"
    def __init__(self, quantizer, *args, kernel_quantizer: str = 'logit', bias_quantizer: str = 'logit', 
                 kernel_initializer: str = 'glorot_normal', bias_initializer = 'zeros', 
                 use_constraint: bool = True, **kwargs):
        # After calling this, we will have the actual quantizer objects in "self._qkwargs"
        _QQLayer.__init__(self, quantizer, 
                          kernel_quantizer = kernel_quantizer, 
                          bias_quantizer = bias_quantizer, 
                          kernel_initializer = kernel_initializer, 
                          bias_initializer = bias_initializer,
                          use_constraint = use_constraint)
        # Finally just call the parent class
        qkeras.QConv1D.__init__(self, *args, **self._qkwargs, **kwargs)


class QQConv2D(qkeras.QConv2D, _QQLayer):
    _QPROPS = ['kernel', 'bias']
    _ICON = "📕"
    def __init__(self, quantizer, *args, kernel_quantizer: str = 'logit', bias_quantizer: str = 'logit', 
                 kernel_initializer: str = 'glorot_normal', bias_initializer = 'zeros', 
                 use_constraint: bool = True, **kwargs):
        # After calling this, we will have the actual quantizer objects in "self._qkwargs"
        _QQLayer.__init__(self, quantizer, 
                          kernel_quantizer = kernel_quantizer, 
                          bias_quantizer = bias_quantizer, 
                          kernel_initializer = kernel_initializer, 
                          bias_initializer = bias_initializer,
                          use_constraint = use_constraint)

        # Finally just call the parent class
        qkeras.QConv2D.__init__(self, *args, **self._qkwargs, **kwargs)
        

class QQConv3D(tf.keras.layers.Conv3D, _QQLayer):
    _QPROPS = ['kernel', 'bias']
    _ICON = "📕"
    def __init__(self, quantizer, *args, kernel_quantizer: str = 'logit', bias_quantizer: str = 'logit', 
                 kernel_initializer: str = 'glorot_normal', bias_initializer: str = 'zeros', 
                 use_constraint: bool = True, **kwargs):
        # After calling this, we will have the actual quantizer objects in "self._qkwargs"
        _QQLayer.__init__(self, quantizer,
                        kernel_quantizer = kernel_quantizer, 
                        bias_quantizer = bias_quantizer, 
                        kernel_initializer = kernel_initializer, 
                        bias_initializer = bias_initializer,
                        use_constraint = use_constraint)

        # Not implemented in qkeras yet
        tf.keras.layers.Conv3D.__init__(self, *args, **kwargs)



class QQConv1DTranspose(tf.keras.layers.Conv1DTranspose, _QQLayer):
    _QPROPS = ['kernel', 'bias']
    _ICON = "📕"
    def __init__(self, quantizer, *args, kernel_quantizer: str = 'logit', bias_quantizer: str = 'logit', 
                 kernel_initializer: str = 'glorot_normal', bias_initializer: str = 'zeros', 
                 use_constraint: bool = True, **kwargs):
        _QQLayer.__init__(self, quantizer, 
                          kernel_quantizer = kernel_quantizer, 
                          bias_quantizer = bias_quantizer, 
                          kernel_initializer = kernel_initializer, 
                          bias_initializer = bias_initializer,
                          use_constraint = use_constraint)
        
        # Not implemnted in qkeras yet
        tf.keras.layers.Conv1DTranspose.__init__(self, *args, **kwargs)

class QQConv2DTranspose(qkeras.QConv2DTranspose, _QQLayer):
    _QPROPS = ['kernel', 'bias']
    _ICON = "📕"
    def __init__(self, quantizer, *args, kernel_quantizer: str = 'logit', bias_quantizer: str = 'logit', 
                 kernel_initializer: str = 'glorot_normal', bias_initializer: str = 'zeros', 
                 use_constraint: bool = True, **kwargs):
        # After calling this, we will have the actual quantizer objects in "self._qkwargs"
        _QQLayer.__init__(self, quantizer, 
                          kernel_quantizer = kernel_quantizer, 
                          bias_quantizer = bias_quantizer, 
                          kernel_initializer = kernel_initializer, 
                          bias_initializer = bias_initializer,
                          use_constraint = use_constraint)
        # Finally just call the parent class
        qkeras.QConv2DTranspose.__init__(self, *args, **self._qkwargs, **kwargs)


class QQConv3DTranspose(tf.keras.layers.Conv3DTranspose, _QQLayer):
    _QPROPS = ['kernel', 'bias']
    _ICON = "📕"
    def __init__(self, quantizer, *args, kernel_quantizer: str = 'logit', bias_quantizer: str = 'logit', 
                 kernel_initializer: str = 'glorot_normal', bias_initializer: str = 'zeros', 
                 use_constraint: bool = True, **kwargs):
        # After calling this, we will have the actual quantizer objects in "self._qkwargs"
        _QQLayer.__init__(self, quantizer, 
                          kernel_quantizer = kernel_quantizer, 
                          bias_quantizer = bias_quantizer, 
                          kernel_initializer = kernel_initializer, 
                          bias_initializer = bias_initializer,
                          use_constraint = use_constraint)

        # Not implemnted in qkeras yet
        tf.keras.layers.Conv3DTranspose.__init__(self, *args, **kwargs)


class QQDepthwiseConv2D(qkeras.QDepthwiseConv2D, _QQLayer):
    _QPROPS = ['depthwise', 'bias']
    _ICON = "📕"
    def __init__(self, quantizer, *args, 
                 depthwise_quantizer: str = 'logit', 
                 depthwise_initializer: str = 'glorot_normal',
                 bias_quantizer: str = 'logit', bias_initializer: str = 'zeros', 
                 use_constraint: bool = True, **kwargs):
        # After calling this, we will have the actual quantizer objects in "self._qkwargs"
        _QQLayer.__init__(self, quantizer,
                          depthwise_quantizer = depthwise_quantizer, 
                          depthwise_initializer = depthwise_initializer,
                          bias_quantizer = bias_quantizer, 
                          bias_initializer = bias_initializer,
                          use_constraint = use_constraint, **kwargs)
        
        # Finally just call the parent class
        qkeras.QDepthwiseConv2D.__init__(self, *args, **self._qkwargs, **kwargs)
        

class QQSeparableConv2D(qkeras.QSeparableConv2D, _QQLayer):
    _QPROPS = ['depthwise', 'pointwise','bias']
    _ICON = "📕"
    def __init__(self, quantizer, *args, depthwise_quantizer: str = 'logit', pointwise_quantizer: str = 'logit', 
                 depthwise_initializer: str = 'glorot_normal', pointwise_initializer: str = 'zeros', 
                 bias_quantizer: str = 'logit', bias_initializer: str = 'zeros', 
                 use_constraint: bool = True, **kwargs):
        # After calling this, we will have the actual quantizer objects in "self._qkwargs"
        _QQLayer.__init__(self, quantizer,
                          depthwise_quantizer = depthwise_quantizer, 
                          pointwise_quantizer = pointwise_quantizer, 
                          depthwise_initializer = depthwise_initializer, 
                          pointwise_initializer = pointwise_initializer,
                            bias_quantizer = bias_quantizer, bias_initializer = bias_initializer,
                          use_constraint = use_constraint, **kwargs)
        
        # Finally just call the parent class
        qkeras.QSeparableConv2D.__init__(self, *args, **self._qkwargs, **kwargs)
        

class QQActivation(qkeras.QActivation, _QQLayer):
    _QPROPS = ['activation']
    _ICON = "📗"
    def __init__(self, quantizer, activation: str = 'relu', *args, 
                 use_constraint: bool = True, **kwargs):
        # After calling this, we will have the actual quantizer objects in "self._qkwargs"
        _QQLayer.__init__(self, quantizer, 
                          activation_quantizer = activation, 
                          use_constraint = use_constraint)
        
        # We need to translate the "activation_quantizer" name in _qkwargs. For instance, 
        # if we have "relu", we actually need to pass the name "quantized_relu" to 

        # Finally just call the parent class
        qkeras.QActivation.__init__(self, self._qkwargs['activation_quantizer'], **kwargs)

    def call(self, inputs):
        return super().call(inputs)

class QQMaxPooling2D(tf.keras.layers.MaxPooling2D, _QQLayer):
    _QPROPS = []
    _ICON = "🗳️"
    def __init__(self, quantizer, *args, use_constraint: bool = False, **kwargs):
        _QQLayer.__init__(self, quantizer, use_constraint = False) # No constraint for this layer
        tf.keras.layers.MaxPooling2D.__init__(self, *args, **kwargs)

class QQAveragePooling2D(qkeras.QAveragePooling2D, _QQLayer):
    _QPROPS = ['average']
    _ICON = "🗳️"
    def __init__(self, quantizer, *args, use_constraint: bool = False, dtype = None, average_quantizer = 'po2', 
                 **kwargs):
        # After calling this, we will have the actual quantizer objects in "self._qkwargs"
        _QQLayer.__init__(self, quantizer, 
                          average_quantizer = average_quantizer, 
                          use_constraint = False) # No constraint for this layer
        # Finally just call the parent class
        qkeras.QAveragePooling2D.__init__(self, *args, **self._qkwargs, dtype = tf.float64, **kwargs)
        

class QQGlobalAveragePooling2D(qkeras.QGlobalAveragePooling2D, _QQLayer):
    _QPROPS = ['average']
    _ICON = "🗳️"
    def __init__(self, quantizer, *args, average_quantizer = 'po2', use_constraint: bool = False,  **kwargs):
        # After calling this, we will have the actual quantizer objects in "self._qkwargs"
        _QQLayer.__init__(self, quantizer, 
                          average_quantizer = average_quantizer,
                          use_constraint = False) # No constraint for this layer
                
        # Finally just call the parent class
        qkeras.QGlobalAveragePooling2D.__init__(self, *args, **self._qkwargs, **kwargs)


""" REDUNDANT LAYERS (NOTHING CHANGES RIGHT NOW )"""
class QQFlatten(tf.keras.layers.Flatten, _QQLayer):
    _QPROPS = []
    _ICON = "🫓"
    def __init__(self, quantizer, *args, use_constraint: bool = False, **kwargs):
        _QQLayer.__init__(self, quantizer, use_constraint = False) # No constraint for this layer
        tf.keras.layers.Flatten.__init__(self, *args, **kwargs)

class QQReshape(tf.keras.layers.Reshape, _QQLayer):
    _QPROPS = []
    _ICON = "🌀"
    def __init__(self, quantizer, *args, use_constraint: bool = True, **kwargs):
        _QQLayer.__init__(self, quantizer, use_constraint = False) # No constraint for this layer
        tf.keras.layers.Reshape.__init__(self, *args, **kwargs)

class QQDropout(qkeras.Dropout, _QQLayer):
    _QPROPS = []
    _ICON = "❌"
    def __init__(self, quantizer, *args, use_constraint: bool = True, **kwargs):
        _QQLayer.__init__(self, quantizer, use_constraint = False) # No constraint for this layer
        qkeras.Dropout.__init__(self, *args, **kwargs)

class QQSoftmax(tf.keras.layers.Softmax, _QQLayer):
    _QPROPS = []
    _ICON = "📗"
    def __init__(self, quantizer, *args, use_constraint: bool = True, **kwargs):
        _QQLayer.__init__(self, quantizer, use_constraint = False) # No constraint for this layer
        tf.keras.layers.Softmax.__init__(self, *args, **kwargs)

class QQSigmoid(tf.keras.layers.Activation, _QQLayer):
    _QPROPS = []
    _ICON = "🦎"
    def __init__(self, quantizer, *args, use_constraint: bool = True, **kwargs):
        _QQLayer.__init__(self, quantizer, use_constraint = False) # No constraint for this layer
        tf.keras.layers.Activation.__init__(self, 'sigmoid', *args, **kwargs)




class QQLSTM(qkeras.QLSTM, _QQLayer):
    _QPROPS = ['kernel', 'recurrent', 'bias', 'state']
    _ICON = "🔮"
    def __init__(self, quantizer, *args, 
                 kernel_quantizer: str = 'logit', 
                 recurrent_quantizer: str = 'logit',
                 bias_quantizer: str = 'logit', 
                 state_quantizer: str = 'logit',
                 kernel_initializer: str = 'glorot_normal', 
                 recurrent_initializer: str = 'glorot_normal',
                 bias_initializer: str = 'zeros',
                 recurrent_activation: str = 'hard_sigmoid',
                 use_constraint: bool = True,
                 **kwargs):
        
        # recurrent_regularizer: Any, None = None, bias_regularizer: Any, None = None, activity_regularizer: Any, None = None, kernel_constraint: Any, None = None, recurrent_constraint: Any, None = None, bias_constraint: Any, None = None, 
        # dropout: float = 0, recurrent_dropout: float = 0, unit_forget_bias: bool = True, 
        # implementation: int = 1, 
        # return_sequences: bool = False, return_state: bool = False, 
        # go_backwards: bool = False, stateful: bool = False, unroll: bool = False, **kwargs: Any
        # use_bias: bool = True, 
        _QQLayer.__init__(self, quantizer, 
                          kernel_quantizer = kernel_quantizer, 
                          recurrent_quantizer = recurrent_quantizer,
                          bias_quantizer = bias_quantizer, 
                          state_quantizer = state_quantizer,
                          kernel_initializer = kernel_initializer, 
                          recurrent_initializer = recurrent_initializer,
                          bias_initializer = bias_initializer,
                          recurrent_activation = recurrent_activation,
                          use_constraint = use_constraint)
        
        if 'state_constraint' in self._qkwargs:
            # Pop
            self.state_constraint = self._qkwargs.pop('state_constraint')
        
        qkeras.QLSTM.__init__(self, *args, **self._qkwargs, **kwargs)

class QQUpSampling2D(tf.keras.layers.UpSampling2D, _QQLayer):
    _QPROPS = []
    _ICON = "🗳️"
    def __init__(self, quantizer, *args, use_constraint: bool = True, **kwargs):
        _QQLayer.__init__(self, quantizer, use_constraint = use_constraint)
        tf.keras.layers.UpSampling2D.__init__(self, *args, **kwargs)


# Custom layers
QQLAYERS = {'QQDense': QQDense,
            'QQBatchNormalization': QQBatchNormalization,
            'QQConv1D': QQConv1D,
            'QQConv2D': QQConv2D,
            'QQConv3D': QQConv3D,
            'QQConv1DTranspose': QQConv1DTranspose,
            'QQConv2DTranspose': QQConv2DTranspose,
            'QQConv3DTranspose': QQConv3DTranspose,
            'QQDepthwiseConv2D': QQDepthwiseConv2D,
            'QQSeparableConv2D': QQSeparableConv2D,
            'QQActivation': QQActivation,
            'QQMaxPooling2D': QQMaxPooling2D,
            'QQAveragePooling2D': QQAveragePooling2D,
            'QQGlobalAveragePooling2D': QQGlobalAveragePooling2D,
            'QQFlatten': QQFlatten,
            'QQReshape': QQReshape,
            'QQDropout': QQDropout,
            'QQSoftmax': QQSoftmax,
            'QQLSTM': QQLSTM,
            'QQUpSampling2D': QQUpSampling2D,
            'QQSigmoid': QQSigmoid}
