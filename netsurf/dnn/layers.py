import tensorflow as tf 

""" Import qkeras """
import qkeras

""" Import initializers and quantizers """
from .quantizers import get_quantizers
from .initializers import get_initializers

""" Attack from qpolar """
#from .qpolar import Attack
from ..core.injection import Attack

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
        self.has_delta = False
        self.prune = False

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

    def build(self, input_shape):
        pass

    def call(self, inputs, training = False):
        if self.prune: self.apply_pruning() # Apply pruning masks before computation
        return super().call(inputs, training = training)

    def attack(self, X, **kwargs):
        return self(X)
    
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
    


class PrunableLayer:
    """
    A mixin class that provides pruning functionality for layers with trainable parameters.

    This class should be inherited by layers that contain weights (e.g., Dense, Conv2D)
    and require structured or unstructured pruning during training. It introduces a 
    pruning mask for each prunable weight and provides methods to apply and update 
    pruning dynamically.

    Attributes:
        prune_masks (dict): A dictionary mapping prunable weight names to their respective
            pruning masks. These masks determine which weights are set to zero.

    Methods:
        build(input_shape):
            Initializes the pruning masks for each prunable weight in the layer.
        
        apply_pruning():
            Applies the pruning masks to the corresponding weights, effectively zeroing
            out pruned connections during training and inference.
        
        update_pruning_mask(sparsity: float):
            Updates the pruning masks dynamically by setting the lowest `sparsity` fraction
            of weights to zero, based on their magnitude.

        prune_summary():
            Logs the sparsity level of each pruned weight tensor for debugging and monitoring.

    Usage:
        - Inherit this class alongside `_QQLayer` for prunable layers.
        - Call `apply_pruning()` inside the `call()` method to enforce pruning.
        - Use `update_pruning_mask(sparsity)` within a callback to schedule pruning
          over time.

    Example:
        >>> class QQDense(qkeras.QDense, _QQLayer, PrunableLayer):
        >>>     _PRUNED_QPROPS = ['kernel']
        >>>     def __init__(self, quantizer, *args, **kwargs):
        >>>         _QQLayer.__init__(self, quantizer, **kwargs)
        >>>         PrunableLayer.__init__(self)
        >>>         qkeras.QDense.__init__(self, *args, **kwargs)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prune_masks = {}  # Dictionary to store pruning masks
        # Override _QQLayer "prune" attribute to enable pruning
        self.prune = True

    def build(self, input_shape):
        """Initialize pruning masks only for trainable properties in `_QPROPS`."""
        for prop in self._PRUNED_QPROPS:
            if hasattr(self, prop):
                param = getattr(self, prop)
                self.prune_masks[prop] = self.add_weight(
                    name=f"{prop}_prune_mask",
                    shape=param.shape,
                    initializer=tf.keras.initializers.Ones(),  # Initialize with ones (no pruning initially)
                    trainable=False  # Ensure masks are not updated via gradients
                )

    def apply_pruning(self):
        """Applies pruning masks to weights."""
        for prop in self._PRUNED_QPROPS:
            if prop in self.prune_masks and hasattr(self, prop):
                weight = getattr(self, prop)
                mask = self.prune_masks[prop]
                
                if isinstance(weight, tf.Variable):  # Ensure it's a TensorFlow variable
                    weight.assign(weight * mask)  # ✅ Use .assign() instead of setattr()

    def update_pruning_mask(self, sparsity: float):
        """Dynamically updates pruning masks based on sparsity."""
        for prop in self._PRUNED_QPROPS:
            if prop in self.prune_masks and hasattr(self, prop):
                weights = getattr(self, prop)
                if weights is not None:
                    threshold = np.percentile(np.abs(weights.numpy()), sparsity * 100)
                    mask = np.where(np.abs(weights.numpy()) > threshold, 1, 0)

                    # ✅ Use .assign() to update the mask
                    self.prune_masks[prop].assign(mask)

    def prune_summary(self):
        """Print pruning stats for debugging."""
        for prop in self._PRUNED_QPROPS:
            if prop in self.prune_masks:
                mask = self.prune_masks[prop].numpy()
                sparsity = 1.0 - np.mean(mask)
                print(f"[INFO] - {self.name}.{prop} sparsity: {sparsity:.2%}")



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
        self.has_delta = True

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

        # Now let's add the bit-flipping variables 
        self.alpha_delta = self.add_weight('alpha_delta',
            shape = self.alpha.shape + (self.quantizer.m,),
            initializer = 'zeros',
            trainable = False)
        
        self.alpha_N = self.add_weight('alpha_N',
            shape = self.alpha.shape + (self.quantizer.m,),
            initializer = 'zeros',
            trainable = False)
        
        # Same for bias 
        self.beta_delta = self.add_weight('beta_delta',
            shape = self.beta.shape + (self.quantizer.m,),
            initializer = 'zeros',
            trainable = False)
        
        self.beta_N = self.add_weight('beta_N',
            shape = self.beta.shape + (self.quantizer.m,),
            initializer = 'zeros',
            trainable = False)
        
        # 🔹 Add a non-trainable variable to store the loss per batch
        self.regularization_loss_value = self.add_weight(
            name="reg_loss_tracker",
            shape=(),
            initializer="zeros",
            trainable=False
        )

        super().build(input_shape)

    """ Method to compute the deltas """
    def compute_deltas(self):
        # Compute the deltas
        alpha_q = self.quantizer(self.alpha)
        alpha_delta = self.quantizer.compute_delta_matrix(alpha_q)
        self.alpha_delta.assign(alpha_delta)

        # Now bias 
        beta_q = self.quantizer(self.beta)
        beta_delta = self.quantizer.compute_delta_matrix(beta_q)
        self.beta_delta.assign(beta_delta)

    def attack(self, X, N = None, clamp = False):
        # If N is not an attack object, turn it into one 
        if isinstance(N, Attack):
            # Apply the "N" values to the variables 
            if hasattr(N, 'alpha'):
                self.alpha_N.assign(N.alpha)
            if hasattr(N, 'beta'):
                self.beta_N.assign(N.beta)
            # Now apply the deltas 
            alpha_d = self.alpha + tf.reduce_sum(self.alpha_delta * self.alpha_N, axis = -1)
            beta_d = self.beta + tf.reduce_sum(self.beta_delta * self.beta_N, axis = -1)
            return self(X, training = False, alpha = alpha_d, beta = beta_d, clamp = clamp)
        elif isinstance(N, dict) or isinstance(N, int) or isinstance(N, float):
            # Turn into Attack object 
            N = Attack(N = N, variables = {'alpha': self.alpha_delta, 'beta': self.beta_delta})
            return self.attack(X, N, clamp = clamp)
        else:
            return self(X)

    def call(self, inputs, alpha = None, beta = None, clamp = False):
        """
        Multiply inputs by alpha and add beta, broadcasting across all other axes.
        """
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta

        # Compute the scaled and shifted output.
        output = alpha * inputs + beta
        
        # Retrieve the desired output range.
        target_min, target_max = self.target_range
        
        # Compute a penalty for values above target_max or below target_min.
        penalty_upper = tf.maximum(0.0, output - target_max)
        penalty_lower = tf.maximum(0.0, target_min - output)
        loss_penalty = tf.reduce_mean(penalty_upper + penalty_lower)
        
        # Add the penalty term (scaled by reg_factor) to the overall model loss.
        self.add_loss(self.reg_factor * loss_penalty)

        # 🔹 Store the loss in the non-trainable variable
        self.regularization_loss_value.assign(self.reg_factor * loss_penalty)

        # Clamp the output to the target range if desired.
        if clamp:
            output = tf.clip_by_value(output, target_min, target_max)
        
        return output

    # Compute impact 
    def compute_impact(self, X, N = None, batch_size = None, **kwargs):
        if N is None:
            N = Attack(N = 1, variables = {self.alpha.name: self.alpha_delta, 
                                              self.beta.name: self.beta_delta})
        
        if batch_size is None:
            batch_size = tf.shape(X)[0]
        # Compute the deltas
        self.compute_deltas()

        # compute num batches
        num_batches = int(tf.math.ceil(tf.cast(tf.shape(X)[0], tf.float32) / batch_size))

        # Calculate P
        P = {}
        for prop in self._QPROPS:
            # Check if prop exists
            if hasattr(self, prop):
                prop_val = getattr(self, prop)
                prop_shape = prop_val.shape
                # Compute the impact
                if prop == 'alpha':
                    P_d = []
                    # Loop over batch_size
                    for i in range(self.quantizer.m):
                        P_bit = []
                        # Loop over the bits 
                        for b in range(num_batches):
                            # Compute the impact of the attack
                            P_bit.append(X[b*batch_size:(b+1)*batch_size] * (self.alpha_delta[...,i] * N[self.alpha.name][...,i]))
                        # Concat over 0 
                        P_d.append(tf.concat(P_bit, axis = 0))

                    # Stack
                    P_d = tf.stack(P_d, axis = len(X.shape))

                    # We need to average over all dimensions but last 
                    if len(P_d.shape) > len(self.alpha.shape):
                        P_d = tf.reduce_mean(P_d, axis = tuple(range(0, len(X.shape)-1)))

                    pname = self.alpha.name
                elif prop == 'beta':
                    # Compute the impact of the attack
                    P_d = self.beta_delta * N[self.beta.name]
                    pname = self.beta.name
                P[pname] = P_d
        
        return P
            

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


""" Weight Layer """
class WeightLayer:
    def build(self):
        # Now let's add the bit-flipping variables for _QPROPS
        for prop in self._QPROPS:
            # Check if prop exists
            if hasattr(self, prop):
                prop_val = getattr(self, prop)
                prop_shape = prop_val.shape
                # Deltas and N
                for suffix in ['delta', 'N']:
                    attr_name = f'{prop}_{suffix}'
                    
                    # Set the attribute
                    setattr(self, attr_name, self.add_weight(attr_name,
                        shape = prop_shape + (self.quantizer.m,),
                        initializer = 'zeros',
                        trainable = False))
                    
            # Now the N variables
            attr_name = f'{prop}_N'

    """ Method to compute the deltas """
    def compute_deltas(self):
        # Compute the deltas for qprops
        for prop in self._QPROPS:
            # Check if prop exists
            if hasattr(self, prop):
                prop_val = getattr(self, prop)
                prop_q = self.quantizer(prop_val)
                prop_delta = self.quantizer.compute_delta_matrix(prop_q)
                getattr(self, f'{prop}_delta').assign(prop_delta)

    def _dense_call(self, X, training = False, W = None, b = None):
        if self.prune: self.apply_pruning() # Apply pruning masks before computation
        if W is None:
            W = self.kernel
        if b is None:
            b = self.bias
        # Matmul
        output = tf.matmul(X, W)
        # Add bias
        output = tf.nn.bias_add(output, b)
        return output

    def _conv_call(self, function, X, training = False, W = None, b = None, **kwargs):
        # Apply pruning masks before computation
        if self.prune: self.apply_pruning() 
        # Get the kernel and bias
        if W is None: W = self.kernel
        if b is None: b = self.bias
        # Perform convolution (assumes self.strides, self.padding, etc. are defined)
        output = function(X, W, **kwargs)
        # Add bias
        output = tf.nn.bias_add(output, b)
        return output

    def _sep_conv_2d_call(self, X, training = False, depthwise = None, pointwise = None, b = None, **kwargs):
        # Apply pruning masks before computation
        if self.prune: self.apply_pruning() 
        # Get the kernel and bias
        if depthwise is None: depthwise = self.depthwise
        if pointwise is None: pointwise = self.pointwise
        if b is None: b = self.bias
        # Perform convolution (assumes self.strides, self.padding, etc. are defined)
        depthwise_out = tf.nn.depthwise_conv2d(X, depthwise, **kwargs)
        pointwise_out = tf.nn.conv2d(depthwise_out, pointwise, strides=[1,1,1,1], padding='SAME')
        # Add bias
        output = tf.nn.bias_add(pointwise_out, b)
        return output

    """ This attack is valid for all normal conv layers and dense layer """
    def _attack(self, X, N = None, **kwargs):
        # If N is not an attack object, turn it into one 
        if isinstance(N, Attack):
            # Apply the "N" values to the variables 
            if self.kernel.name in N:
                self.kernel_N.assign(N[self.kernel.name])
            if self.bias.name in N:
                self.bias_N.assign(N[self.bias.name])
            # Now apply the deltas 
            W_d = self.kernel + tf.reduce_sum(self.kernel_delta * self.kernel_N, axis = -1)
            b_d = self.bias + tf.reduce_sum(self.bias_delta * self.bias_N, axis = -1)
            return self(X, training = False, W = W_d, b = b_d)
        elif isinstance(N, dict) or isinstance(N, int) or isinstance(N, float):
            # Turn into Attack object 
            N = Attack(N = N, variables = {self.kernel.name: self.kernel_delta, self.bias.name: self.bias_delta})
            return self.attack(X, N, **kwargs)
        else:
            return self(X)
    
    """ Base method to compute the impact (this has to be implemented by child) """
    def compute_impact(self, X, N = None, batch_size = None, **kwargs):
        raise NotImplementedError("This method has to be implemented by the child class")

    # Serialize methods 
    def _serialize_conv(self, include_emojis: bool = True, **kwargs):
        conts, cats = _QQLayer._serialize(self, include_emojis=include_emojis, **kwargs)
        # Base display for all convolutional layers
        d = {
            "📏 Kernel size": self.kernel_size,
            "🎯 Strides": self.strides,
            "🔁 Padding": self.padding,
            "📚 Dilation rate": getattr(self, 'dilation_rate', None),
            "⚓️ Use bias": "✅" if getattr(self, 'use_bias', False) else "❌",
            "🔗 Use constraint": "✅" if getattr(self, 'use_constraint', False) else "❌"
        }

        # Add only if present in the layer
        if hasattr(self, 'activation'):
            d["📗 Activation"] = self.activation
        if hasattr(self, 'groups'):
            d["🧪 Groups"] = self.groups
        if hasattr(self, 'depth_multiplier'):
            d["🌀 Depth multiplier"] = self.depth_multiplier
        if hasattr(self, 'output_padding'):
            d["🧵 Output padding"] = self.output_padding
        if hasattr(self, 'data_format'):
            d["🧭 Data format"] = self.data_format
        if hasattr(self, 'filters'):
            d["🎚️ Filters"] = self.filters

        props = pd.DataFrame([d]).T
        conts[0]['Layer properties'] = pd.concat([conts[0]['Layer properties'], props])

        conts.extend(cats)
        return conts

    def _serialize_dense(self, include_emojis: bool = True, **kwargs):
        conts, cats = _QQLayer._serialize(self, include_emojis = include_emojis, **kwargs)
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

    def _serialize_batchnorm(self, include_emojis: bool = True, **kwargs):
        conts, cats = _QQLayer._serialize(self, include_emojis = include_emojis, **kwargs)
        # Add this layer specific stuff to conts
        # layer_specific = {'batchnorm': ['axis', 'momentum', 'epsilon', 'center', 'scale', 'beta_initializer', 'gamma_initializer', 'moving_mean_initializer', 'moving_variance_initializer', 'beta_regularizer', 'gamma_regularizer', 'beta_constraint', 'gamma_constraint']}

        d = dict({
            "🔀 Axis": self.axis,
            "🔁 Momentum": self.momentum,
            "🔗 Epsilon": self.epsilon,
            "🔗 Center": "✅" if self.center else "❌",
            "🔗 Scale": "✅" if self.scale else "❌",
            "🏁 Beta initializer": self.beta_initializer,
            "🏁 Gamma initializer": self.gamma_initializer,
            "🏁 Moving mean initializer": self.moving_mean_initializer,
            "🏁 Moving variance initializer": self.moving_variance_initializer,
            "🎚️ Beta regularizer": self.beta_regularizer,
            "🎚️ Gamma regularizer": self.gamma_regularizer,
            "🔗 Beta constraint": self.beta_constraint,
            "🔗 Gamma constraint": self.gamma_constraint
        })
        
        props = pd.DataFrame([d]).T
        conts[0]['Layer properties'] = pd.concat([conts[0]['Layer properties'], props])
        
        # Concatenate into a single list
        conts.extend(cats)
        return conts

""" Dense / fully connected layer """
class QQDense(qkeras.QDense, _QQLayer, PrunableLayer, WeightLayer):
    _QPROPS = ['kernel', 'bias']
    _PRUNED_QPROPS = ['kernel']
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
        
        # Initialize pruning logic
        PrunableLayer.__init__(self)  
        
        # Finally just call the parent class
        qkeras.QDense.__init__(self, *args, **self._qkwargs, **kwargs)
        # Set self.has_delta = True
        self.has_delta = True

    # Modify self.build to include bit-flip vars 
    def build(self, input_shape):
        """Add bit-flip attack variables for kernel and bias."""
        super().build(input_shape)  # Call QKeras build method
        # First call super to build kernel/bias
        _QQLayer.build(self, input_shape)
        # Now build the pruning masks
        PrunableLayer.build(self, input_shape)
        # Finally build the bit-flip vars
        WeightLayer.build(self)
        
    # Call method to just do forward pass
    # USING TF/KERAS CONVENTION:
    # output = X @ W.T
    def call(self, X, **kwargs):
        return WeightLayer._dense_call(self, X, **kwargs)

    # Attack method
    def attack(self, X, N = None, **kwargs):
        return WeightLayer._attack(self, X, N = N, **kwargs)
    
    # Compute the impact of the attack
    def compute_impact(self, X, N = None, batch_size = None, **kwargs):
        # Compute the impact of the attack
        if N is None:
            N = Attack(N = 1, variables = {self.kernel.name: self.kernel_delta,
                                           self.bias.name: self.bias_delta})
        # Make sure the deltas have been calculated
        self.compute_deltas()

        # if batch_size is None:
        if batch_size is None:
            batch_size = tf.shape(X)[0]
        
        # Compute num_batches
        num_batches = tf.maximum(1, tf.shape(X)[0]//batch_size)

        # Impact for dense layer is just:
        # P_k = X @ delta_k*N 
        # P_b = delta_b*N
        P = {}
        for prop in self._QPROPS:
            # Check if prop exists
            if hasattr(self, prop):
                prop_val = getattr(self, prop)
                prop_shape = prop_val.shape
                # Compute the impact
                if prop == 'kernel':
                    P_d = []
                    # Loop over batch_size
                    for i in range(self.quantizer.m):
                        P_bit = []
                        # Loop over the bits 
                        for b in range(num_batches):
                            # Compute the impact of the attack
                            P_bit.append(X[b*batch_size:(b+1)*batch_size,...,None] * ((self.kernel_delta[...,i] * N[self.kernel.name][...,i])))
                        # Concat over 0 
                        P_d.append(tf.concat(P_bit, axis = 0))
                    # Stack
                    P_d = tf.stack(P_d, axis = len(self.kernel_delta.shape))
                    if len(X.shape) > 2:
                        # We need to average over all dimensions but 0 and last 
                        P_d = tf.reduce_mean(P_d, axis = tuple(range(0, len(self.kernel.shape)-1)))
                    pname = self.kernel.name
                elif prop == 'bias':
                    # Compute the impact of the attack
                    P_d = self.bias_delta * N[self.bias.name]
                    pname = self.bias.name
                
                P[pname] = P_d
        
        return P
        

    # Serialize (for display)
    def serialize(self, **kwargs):
        return WeightLayer._serialize_dense(self, **kwargs)
    

""" Convolution layers """
class QQConv1D(qkeras.QConv1D, _QQLayer, PrunableLayer, WeightLayer):
    _QPROPS = ['kernel', 'bias']
    _PRUNED_QPROPS = ['kernel']
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
        # Initialize pruning logic
        PrunableLayer.__init__(self) 

        # Finally just call the parent class
        qkeras.QConv1D.__init__(self, *args, **self._qkwargs, **kwargs)

        # 🔹 Enable bit-flip perturbation support
        self.has_delta = True
    
    def build(self, input_shape):
        """Add bit-flip attack variables for kernel and bias."""
        super().build(input_shape)  # Call QKeras build method
        # First call super to build kernel/bias
        _QQLayer.build(self, input_shape)
        # Now build the pruning masks
        PrunableLayer.build(self, input_shape)
        # Build the bit-flip attack variables
        WeightLayer.build(self)
    
    # Call method to just do forward pass
    def call(self, X, **kwargs):
        return WeightLayer._conv_call(self, tf.nn.conv1d, X, **kwargs, 
                                      strides=self.strides, padding=self.padding.upper(), dilations=self.dilation_rate)

    # Attack method
    def attack(self, X, N = None, **kwargs):
        return WeightLayer._attack(self, X, N = N, **kwargs)

    # Serialize
    def serialize(self, **kwargs):
        return WeightLayer._serialize_conv(self, **kwargs)


class QQConv2D(qkeras.QConv2D, _QQLayer, PrunableLayer, WeightLayer):
    _QPROPS = ['kernel', 'bias']
    _PRUNED_QPROPS = ['kernel']
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
        
        # Initialize pruning logic
        PrunableLayer.__init__(self) 

        # Finally just call the parent class
        qkeras.QConv2D.__init__(self, *args, **self._qkwargs, **kwargs)

        # 🔹 Enable bit-flip perturbation support
        self.has_delta = True
    
    # Modify self.build to include bit-flip vars 
    def build(self, input_shape):
        """Add bit-flip attack variables for kernel and bias."""
        super().build(input_shape)  # Call QKeras build method
        # First call super to build kernel/bias
        _QQLayer.build(self, input_shape)
        # Now build the pruning masks
        PrunableLayer.build(self, input_shape)
        # Build the bit-flip attack variables
        WeightLayer.build(self)
    
    # Call method to just do forward pass
    # USING TF/KERAS CONVENTION:
    # output = X @ W.T
    def call(self, X, **kwargs):
        # Add some kwargs
        return WeightLayer._conv_call(self, tf.nn.conv2d, X, **kwargs, 
                               strides=self.strides, padding=self.padding.upper(), dilations=self.dilation_rate)
    
    # Attack method
    def attack(self, X, N = None, **kwargs):
        return WeightLayer._attack(self, X, N = N, **kwargs)

    def compute_conv2d_bitwise_impact(self, X, batch_size, delta_kernel, N_kernel, 
                                   strides, padding, dilation_rate=(1, 1), 
                                   data_format="channels_last"):
        """
        Compute per-bit impact of each weight in a Conv2D layer.
        Returns a tensor of shape (kh, kw, in_channels, out_channels, n_bits)
        with the L2 norm of the output perturbation from each bit.
        """
        kh, kw, cin, cout, n_bits = delta_kernel.shape
        impact_map = np.zeros((kh, kw, cin, cout, n_bits), dtype=np.float32)

        num_batches = int(tf.math.ceil(tf.cast(tf.shape(X)[0], tf.float32) / batch_size))

        for i in range(kh):
            for j in range(kw):
                for inc in range(cin):
                    for outc in range(cout):
                        for k in range(n_bits):
                            # Construct a kernel with delta at one position
                            delta = np.zeros((kh, kw, cin, cout), dtype=np.float32)
                            delta[i, j, inc, outc] = delta_kernel[i, j, inc, outc, k] * N_kernel[i, j, inc, outc, k]
                            delta_tensor = tf.convert_to_tensor(delta, dtype=tf.float32)

                            for b in range(num_batches):
                                x_batch = X[b*batch_size:(b+1)*batch_size]
                                conv_out = tf.nn.conv2d(
                                    x_batch,
                                    delta_tensor,
                                    strides=[1, *strides, 1],
                                    padding=padding.upper(),
                                    dilations=[1, *dilation_rate, 1],
                                    data_format='NHWC' if data_format == 'channels_last' else 'NCHW'
                                )

                                # L2 norm = total impact of that bit
                                impact_map[i, j, inc, outc, k] += tf.norm(conv_out).numpy()
                            
                            # Normalize over num_batches
                            impact_map[i, j, inc, outc, k] /= num_batches

        return impact_map

    # Compute the impact of the attack
    def compute_impact(self, X, N = None, batch_size = None, **kwargs):
        # Compute the impact of the attack
        if N is None:
            N = Attack(N = 1, variables = {self.kernel.name: self.kernel_delta, 
                                           self.bias.name: self.bias_delta})
        # Make sure the deltas have been calculated
        self.compute_deltas()

        if batch_size is None:
            batch_size = tf.shape(X)[0]
        
        # Impact for conv layers is just:
        # P_k = X * delta_k*N  (where * represents convolution, remember conv is a distributive operator)
        # P_b = delta_b*N
        P = {}
        for prop in self._QPROPS:
            # Check if prop exists
            if hasattr(self, prop):
                prop_val = getattr(self, prop)
                prop_shape = prop_val.shape
                # Compute the impact
                if prop == 'kernel':
                    # Compute the impact of the attack as the conv (per bit)
                    P_d = self.compute_conv2d_bitwise_impact(X, batch_size, 
                                                             self.kernel_delta, 
                                                             self.kernel_N, 
                                                             self.strides,
                                                             self.padding, 
                                                             dilation_rate=self.dilation_rate,
                                                             data_format = self.data_format)

                    pname = self.kernel.name
                elif prop == 'bias':
                    # Compute the impact of the attack
                    P_d = self.bias_delta * N[self.bias.name]
                    pname = self.bias.name
                # Add to the dict
                P[pname] = P_d
        
        return P

    # Serialize
    def serialize(self, **kwargs):
        return WeightLayer._serialize_conv(self, **kwargs)
        

class QQConv3D(tf.keras.layers.Conv3D, _QQLayer, PrunableLayer, WeightLayer):
    _QPROPS = ['kernel', 'bias']
    _PRUNED_QPROPS = ['kernel']
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

        # Initialize pruning logic
        PrunableLayer.__init__(self) 

        # Not implemented in qkeras yet
        tf.keras.layers.Conv3D.__init__(self, *args, **kwargs)

        # 🔹 Enable bit-flip perturbation support
        self.has_delta = True
    
    def build(self, input_shape):
        """Add bit-flip attack variables for kernel and bias."""
        super().build(input_shape)  # Call QKeras build method
        # First call super to build kernel/bias
        _QQLayer.build(self, input_shape)
        # Now build the pruning masks
        PrunableLayer.build(self, input_shape)
        # Finally build the bit-flip vars
        WeightLayer.build(self)
    
    # Call method to just do forward pass
    # USING TF/KERAS CONVENTION:
    # output = X @ W.T
    def call(self, X, **kwargs):
        # Add some kwargs
        return WeightLayer._conv_call(self, tf.nn.conv3d, X, **kwargs, 
                               strides=self.strides, padding=self.padding.upper(), dilations=self.dilation_rate)
    
    # Attack method
    def attack(self, X, N = None, **kwargs):
        return WeightLayer._attack(self, X, N = N, **kwargs)

    # Serialize
    def serialize(self, **kwargs):
        return WeightLayer._serialize_conv(self, **kwargs)

""" 
    Deconvolutional layers
"""
class QQConv1DTranspose(tf.keras.layers.Conv1DTranspose, _QQLayer, PrunableLayer, WeightLayer):
    _QPROPS = ['kernel', 'bias']
    _PRUNED_QPROPS = ['kernel']
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
        
        # Initialize pruning logic
        PrunableLayer.__init__(self) 

        # Not implemnted in qkeras yet
        tf.keras.layers.Conv1DTranspose.__init__(self, *args, **kwargs)

        # 🔹 Enable bit-flip perturbation support
        self.has_delta = True
    
    def build(self, input_shape):
        """Add bit-flip attack variables for kernel and bias."""
        super().build(input_shape)  # Call QKeras build method
        # First call super to build kernel/bias
        _QQLayer.build(self, input_shape)
        # Now build the pruning masks
        PrunableLayer.build(self, input_shape)
        # Finally build the bit-flip vars
        WeightLayer.build(self)
    
    def call(self, X, **kwargs):
        raise NotImplementedError("Conv1DTranspose is not implemented in TensorFlow yet.")
    
    # Attack method
    def attack(self, X, N = None, **kwargs):
        return WeightLayer._attack(self, X, N = N, **kwargs)

    def serialize(self, **kwargs):
        return WeightLayer._serialize_conv(self, **kwargs)
    

class QQConv2DTranspose(qkeras.QConv2DTranspose, _QQLayer, PrunableLayer, WeightLayer):
    _QPROPS = ['kernel', 'bias']
    _PRUNED_QPROPS = ['kernel']
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
        
        # Initialize pruning logic
        PrunableLayer.__init__(self) 

        # Finally just call the parent class
        qkeras.QConv2DTranspose.__init__(self, *args, **self._qkwargs, **kwargs)

        # 🔹 Enable bit-flip perturbation support
        self.has_delta = True
    
    def build(self, input_shape):
        """Add bit-flip attack variables for kernel and bias."""
        super().build(input_shape)  # Call QKeras build method
        # First call super to build kernel/bias
        _QQLayer.build(self, input_shape)
        # Now build the pruning masks
        PrunableLayer.build(self, input_shape)
        # Finally build the bit-flip vars
        WeightLayer.build(self)
    
    def call(self, X, **kwargs):
        # Compute output shape manually
        output_shape = tf.keras.layers.Conv2DTranspose(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation_rate=self.dilation_rate
        ).compute_output_shape(X.shape)

        # Add some kwargs
        ## [@manuelbv]: NOTE: THE REASON WHY WE ARE USING tf.keras.backend.conv2d_transpose here, instead of 
        # tf.nn.conv2d_transpose is because the latter does not fully support the output_shape to be dynamic 
        # (that is, not knowing what the batch_size is gonna be). 
        return WeightLayer._conv_call(self, tf.keras.backend.conv2d_transpose, X, output_shape = output_shape, 
                                      **kwargs, strides=self.strides, padding=self.padding)
    
    # Attack method
    def attack(self, X, N = None, **kwargs):
        return WeightLayer._attack(self, X, N = N, **kwargs)
    
    def compute_conv2dtranspose_bitwise_impact(self, X, batch_size, delta_kernel, N_kernel, 
                                           kernel_size, strides, padding, filters,
                                           dilation_rate=(1, 1), data_format="channels_last"):
        """
        Compute per-bit impact of each weight in a Conv2DTranspose layer.
        Returns a tensor of shape (kh, kw, in_channels, out_channels, n_bits)
        with the L2 norm of the output perturbation from each bit.
        """
        kh, kw, cin, cout, n_bits = delta_kernel.shape
        impact_map = np.zeros((kh, kw, cin, cout, n_bits), dtype=np.float32)

        num_batches = int(tf.math.ceil(tf.shape(X)[0]/batch_size))

        for i in range(kh):
            for j in range(kw):
                for inc in range(cin):
                    for outc in range(cout):
                        for k in range(n_bits):
                            delta = np.zeros((kh, kw, cin, cout), dtype=np.float32)
                            delta[i, j, inc, outc] = delta_kernel[i, j, inc, outc, k] * N_kernel[i, j, inc, outc, k]
                            delta_tensor = tf.convert_to_tensor(delta, dtype=tf.float32)

                            for b in range(num_batches):
                                x_batch = X[b*batch_size:(b+1)*batch_size]
                                
                                # Compute output shape
                                height = (x_batch.shape[1] - 1) * strides[0] + kernel_size[0]
                                width = (x_batch.shape[2] - 1) * strides[1] + kernel_size[1]
                                output_shape = tf.stack([x_batch.shape[0], height, width, filters])

                                conv_out = tf.nn.conv2d_transpose(
                                    x_batch,
                                    delta_tensor,
                                    output_shape=output_shape,
                                    strides=[1, *strides, 1],
                                    padding=padding.upper(),
                                    dilations=[1, *dilation_rate, 1],
                                    data_format='NHWC' if data_format == 'channels_last' else 'NCHW'
                                )

                                # Store L2 norm of the resulting output change
                                impact_map[i, j, inc, outc, k] += tf.norm(conv_out).numpy()
                            
                            # Normalize the impact map
                            impact_map[i, j, inc, outc, k] /= num_batches

        return impact_map

    # Compute the impact of the attack
    def compute_impact(self, X, N = None, batch_size = None, **kwargs):
        if N is None:
            N = Attack(N = 1, variables = {self.kernel.name: self.kernel_delta,
                                             self.bias.name: self.bias_delta})
        # Make sure the deltas have been calculated
        self.compute_deltas()

        if batch_size is None:
            batch_size = tf.shape(X)[0]
        
        # Compute num_batches
        num_batches = int(tf.math.ceil(tf.shape(X)[0]/batch_size))

        P = {}
        for prop in self._QPROPS:
            # Check if prop exists
            if hasattr(self, prop):
                prop_val = getattr(self, prop)
                prop_shape = prop_val.shape
                # Compute the impact
                if prop == 'kernel':
                    P_d = self.compute_conv2dtranspose_bitwise_impact(X, 
                                                                      batch_size,
                                                                self.kernel_delta, 
                                                                N[self.kernel.name], 
                                                                kernel_size=self.kernel_size,
                                                                strides=self.strides,
                                                                padding='valid',
                                                                filters = self.filters,
                                                                dilation_rate=self.dilation_rate,
                                                                data_format=self.data_format)
                    pname = self.kernel.name
                elif prop == 'bias':
                    P_d = self.bias_delta * N[self.bias.name]
                    pname = self.bias.name
                
                P[pname] = P_d
        return P

    def serialize(self, **kwargs):
        return WeightLayer._serialize_conv(self, **kwargs)


class QQConv3DTranspose(tf.keras.layers.Conv3DTranspose, _QQLayer, PrunableLayer, WeightLayer):
    _QPROPS = ['kernel', 'bias']
    _PRUNED_QPROPS = ['kernel']
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

        # Initialize pruning logic
        PrunableLayer.__init__(self) 

        # Not implemnted in qkeras yet
        tf.keras.layers.Conv3DTranspose.__init__(self, *args, **kwargs)

        # 🔹 Enable bit-flip perturbation support
        self.has_delta = True
    
    def build(self, input_shape):
        """Add bit-flip attack variables for kernel and bias."""
        super().build(input_shape)  # Call QKeras build method
        # First call super to build kernel/bias
        _QQLayer.build(self, input_shape)
        # Now build the pruning masks
        PrunableLayer.build(self, input_shape)
        # Finally build the bit-flip vars
        WeightLayer.build(self)
    
    def call(self, X, **kwargs):
        # Compute output shape manually
        output_shape = tf.keras.layers.Conv3DTranspose(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding.upper(),
            output_padding=self.output_padding,
            dilation_rate=self.dilation_rate
        ).compute_output_shape(X.shape)

        # Add some kwargs
        return WeightLayer._conv_call(self, tf.nn.conv3d_transpose, X, output_shape = output_shape, 
                                      **kwargs, strides=self.strides, padding=self.padding.upper())
    
    # Attack method
    def attack(self, X, N = None, **kwargs):
        return WeightLayer._attack(self, X, N = N, **kwargs)

    def serialize(self, **kwargs):
        return WeightLayer._serialize_conv(self, **kwargs)


class QQDepthwiseConv2D(qkeras.QDepthwiseConv2D, _QQLayer, PrunableLayer):
    _QPROPS = ['depthwise', 'bias']
    _PRUNED_QPROPS = ['kernel']
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
        
        # Initialize pruning logic
        PrunableLayer.__init__(self) 
        
        # Finally just call the parent class
        qkeras.QDepthwiseConv2D.__init__(self, *args, **self._qkwargs, **kwargs)

        # 🔹 Enable bit-flip perturbation support
        self.has_delta = True
    
    def build(self, input_shape):
        """Add bit-flip attack variables for kernel and bias."""
        super().build(input_shape)  # Call QKeras build method
        # First call super to build kernel/bias
        _QQLayer.build(self, input_shape)
        # Now build the pruning masks
        PrunableLayer.build(self, input_shape)
        # Finally build the bit-flip vars
        WeightLayer.build(self)
    
    def call(self, X, **kwargs):
        # Add some kwargs
        return WeightLayer._conv_call(self, tf.nn.depthwise_conv2d, X, 
                                      **kwargs, strides=self.strides, padding=self.padding.upper())
    
    # Attack method
    def attack(self, X, N = None, **kwargs):
        return WeightLayer._attack(self, X, N = N, **kwargs)

    def serialize(self, **kwargs):
        return WeightLayer._serialize_conv(self, **kwargs)
        

class QQSeparableConv2D(qkeras.QSeparableConv2D, _QQLayer, PrunableLayer, WeightLayer):
    _QPROPS = ['depthwise', 'pointwise', 'bias']
    _PRUNED_QPROPS = ['depthwise', 'pointwise']
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
        
        # Initialize pruning logic
        PrunableLayer.__init__(self) 
        
        # Finally just call the parent class
        qkeras.QSeparableConv2D.__init__(self, *args, **self._qkwargs, **kwargs)

        # 🔹 Enable bit-flip perturbation support
        self.has_delta = True
    
    def build(self, input_shape):
        """Add bit-flip attack variables for kernel and bias."""
        super().build(input_shape)  # Call QKeras build method
        # First call super to build kernel/bias
        _QQLayer.build(self, input_shape)
        # Now build the pruning masks
        PrunableLayer.build(self, input_shape)
        # Finally build the bit-flip vars
        WeightLayer.build(self)
    
    def call(self, X, **kwargs):
        # Add some kwargs
        return WeightLayer._sep_conv_2d_call(self, X, **kwargs, strides=self.strides, padding=self.padding.upper())
    
    # Attack method
    """ This attack is valid for all normal conv layers and dense layer """
    def attack(self, X, N = None, **kwargs):
        # If N is not an attack object, turn it into one 
        if isinstance(N, Attack):
            # Apply the "N" values to the variables 
            if self.depthwise.name in N:
                self.depthwise_N.assign(N[self.depthwise.name])
            if self.pointwise.name in N:
                self.pointwise_N.assign(N[self.pointwise.name])
            if self.bias.name in N:
                self.bias_N.assign(N[self.bias.name])
            # Now apply the deltas 
            depthwise_d = self.depthwise + tf.reduce_sum(self.depthwise_delta * self.depthwise_N, axis = -1)
            pointwise_d = self.pointwise + tf.reduce_sum(self.pointwise_delta * self.pointwise_N, axis = -1)
            b_d = self.bias + tf.reduce_sum(self.bias_delta * self.bias_N, axis = -1)
            return self(X, training = False, depthwise = depthwise_d, pointwise = pointwise_d, b = b_d)
        elif isinstance(N, dict) or isinstance(N, int) or isinstance(N, float):
            # Turn into Attack object 
            N = Attack(N = N, variables = {self.depthwise.name: self.depthwise_delta, 
                                           self.pointwise.name: self.pointwise_delta,
                                           self.bias.name: self.bias_delta})
            return self.attack(X, N, **kwargs)
        else:
            return self(X)

    def serialize(self, **kwargs):
        return WeightLayer._serialize_conv(**kwargs)


""" 
    Batch norm 
"""
class QQBatchNormalization(qkeras.QBatchNormalization, _QQLayer, PrunableLayer, WeightLayer):
    _QPROPS = ['beta', 'gamma']
    _PRUNED_QPROPS = ['gamma']
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
        
        # 🔹 Enable bit-flip perturbation support
        self.has_delta = True

    def build(self, input_shape):
        """Add bit-flip attack variables to beta and gamma."""
        super().build(input_shape)  # Call QKeras build method
        # First call super to build kernel/bias
        _QQLayer.build(self, input_shape)
        # Now build the pruning masks
        PrunableLayer.build(self, input_shape)
        # Finally build the bit-flip vars
        WeightLayer.build(self)


    """ Override the call method to include bit-flip perturbations """
    def call(self, inputs, training=False, apply_attack=False):
        """
        Apply batch normalization with optional bit-flip attack on gamma/beta.
        """
        if self.prune: self.apply_pruning() # Apply pruning masks before computation
        if apply_attack:
            # Store original values
            original_gamma = tf.identity(self.gamma)
            original_beta = tf.identity(self.beta)

            # Apply bit-flip attack perturbations
            self.gamma.assign(self.gamma + tf.reduce_sum(self.gamma_delta * self.gamma_N, axis=-1))
            self.beta.assign(self.beta + tf.reduce_sum(self.beta_delta * self.beta_N, axis=-1))

            # Compute batch norm output
            output = super().call(inputs, training=training)

            # Restore original values
            self.gamma.assign(original_gamma)
            self.beta.assign(original_beta)

            return output
        else:
            return super().call(inputs, training=training)
        
    def attack(self, X, N=None, **kwargs):
        """
        Apply bit-flip attack on gamma and beta.
        """
        if isinstance(N, Attack):
            # Apply bit-flip noise to gamma and beta if present
            if self.gamma.name in N:
                self.gamma_N.assign(N[self.gamma.name])
            if self.beta.name in N:
                self.beta_N.assign(N[self.beta.name])

            return self(X, training=False, apply_attack=True)

        elif isinstance(N, dict) or isinstance(N, int) or isinstance(N, float):
            # Convert to Attack object
            N = Attack(N=N, variables={self.gamma.name: self.gamma_delta, self.beta.name: self.beta_delta})
            return self.attack(X, N, **kwargs)

        else:
            return self(X)
    
    def compute_impact(self, X, N=None, batch_size=None, **kwargs):
        # Impact for batchnorm is similar to dense layer, but even easier
        # P_k = X * delta_k*N
        # P_b = delta_b*N
        if N is None:
            N = Attack(N=1, variables={self.gamma.name: self.gamma_delta, 
                                       self.beta.name: self.beta_delta})
        # Make sure the deltas have been calculated
        self.compute_deltas()

        if batch_size is None:
            batch_size = tf.shape(X)[0]
        
        # Compute num_batches
        num_batches = tf.maximum(1, tf.shape(X)[0]//batch_size)

        # We need to remove mean and divide by stddev
        # Get the mean and std of the activations
        mean = np.mean(X, axis = 0)
        std = np.std(X, axis = 0)
        # Renormalize
        X = (X - mean)/std

        P = {}
        for prop in self._QPROPS:
            # Check if prop exists
            if hasattr(self, prop):
                prop_val = getattr(self, prop)
                prop_shape = prop_val.shape
                # Compute the impact
                if prop == 'gamma':
                    P_d = []
                    for i in range(self.gamma_delta.shape[-1]):
                        P_bit = []
                        for b in range(num_batches):
                            P_dd = X[b*batch_size:(b+1)*batch_size] * self.gamma_delta[...,i] * N[self.gamma.name][...,i]
                            P_bit.append(P_dd)
                        # Concat over axis 0
                        P_bit = tf.concat(P_bit, axis = 0)
                        P_d.append(P_bit)
                    # Stack
                    P_d = tf.stack(P_d, axis = len(self.gamma_delta.shape) - 1)
                    pname = self.gamma.name
                elif prop == 'beta':
                    P_d = self.beta_delta * N[self.beta.name]
                    pname = self.beta.name
                
                P[pname] = P_d
        
        return P


    def serialize(self, **kwargs):
        return WeightLayer._serialize_batchnorm(self, **kwargs)



"""
    OTHER LAYERS
"""
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


"""
    LSTM
"""
class QQLSTM(qkeras.QLSTM, _QQLayer, PrunableLayer):
    _QPROPS = ['kernel', 'recurrent', 'bias', 'state']
    _PRUNED_QPROPS = ['kernel', 'recurrent']
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
        
        # Initialize pruning logic
        PrunableLayer.__init__(self) 
        
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
