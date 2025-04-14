# Basic modules 
import inspect
import os

# Tensorflow and keras
import tensorflow as tf
from tensorflow import keras  
from qkeras.utils import _add_supported_quantized_objects

""" Numpy """
import numpy as np

""" Pandas """
import pandas as pd

""" Matplotlib for evaluation visualization """
import matplotlib.pyplot as plt

""" Custom tensorplot (@manuelbv) """
import tensorplot as tp

""" Custom utils """
import netsurf

""" Utils """
from ..utils import io

""" Import pergamos """
import pergamos as pg

""" Tensorplot """
import tensorplot as tp 

""" Import custom layers """
from .layers import _QQLayer, QQDense, QQConv1D, QQConv2D, QQDepthwiseConv2D, QQSeparableConv2D, \
                    QQGlobalAveragePooling2D, QQMaxPooling2D, \
                    QQActivation, QQSoftmax, \
                    QQBatchNormalization, \
                    QQFlatten, QQDropout, QQReshape, \
                    QQLSTM, QQApplyAlpha, PrunableLayer


""" Basic neural network """
class QModel(tf.keras.Model):
    metadata = ['_config']
    def __init__(self, quantizer: 'QuantizationScheme', in_shape, out_shape, 
                 optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'],
                 type = 'classification', use_bias = True, use_constraint = True,
                 ip = None, out = None, **kwargs):

        # ONLY call super init if we have ip and out (this only happens when we explicitly use the __init__
        # method from another function, see the build_model below, when we create the activation model)
        if ip is not None and out is not None:
            super().__init__(ip, out)
            self._built = True
        else:
            self._built = False

        # internal config (for netsurf)
        self._config = {'quantization_scheme': quantizer.get_config() if hasattr(quantizer, 'get_config') else quantizer,
                'quantizer': quantizer,
                'in_shape': in_shape, 
                'out_shape': out_shape, 
                'use_bias': use_bias, 
                'use_constraint': use_constraint,
                'type': type, 
                'loss': loss, 
                'total_num_params': 0,
                'optimizer': optimizer, 
                'metrics': metrics}

        # Set pruned masks
        self.pruned_masks = None
        self.bit_flip_registry = None

    def count_trainable_parameters(self):
        return np.sum([np.prod(v.shape) for v in self.trainable_variables])

    def clone(self):
        obj = self.__class__.from_config(self.get_config())
        w_idx_mapper = {w.name: i for i, w in enumerate(self.variables)}
        # Make sure we copy the weights over...
        for var in self.variables:
            # Find index 
            idx = w_idx_mapper[var.name]
            if idx < len(obj.variables):
                try:
                    obj.variables[idx].assign(var)
                except Exception as e:
                    print(f'Error copying variable {var.name}: {e}')
        return obj

    def get_config(self):
        """
        Returns the configuration of the model as a dictionary.
        This is used for serialization.
        """
        super_config = super(QModel, self).get_config()

        # Make a deep copy to avoid modifying the original config
        base_config = super_config.copy()

        base_config['custom_config'] = self._config
        # Now, we need to update the metrics to be strings
        if 'metrics' in base_config['custom_config']:
            base_config['custom_config']['metrics'] = [m.name if hasattr(m, 'name') else m for m in base_config['custom_config']['metrics']]
        # Convert to list
        base_config['custom_config']['metrics'] = list(base_config['custom_config']['metrics'])
        
        # Same for loss 
        if 'loss' in base_config['custom_config']:
            if hasattr(base_config['custom_config']['loss'], 'name'):
                base_config['custom_config']['loss'] = base_config['custom_config']['loss'].name
            elif hasattr(base_config['custom_config']['loss'], '__name__'):
                base_config['custom_config']['loss'] = base_config['custom_config']['loss'].__name__
        
        base_config['custom_config'] = dict(base_config['custom_config'])

        # Remove quantizer from config
        if 'quantizer' in base_config['custom_config']:
            base_config['custom_config'].pop('quantizer')

        # Now make "quantization_scheme"  -> quantizer in this dict
        base_config['custom_config']['quantizer'] = base_config['custom_config'].pop('quantization_scheme')

        return base_config
    
    @classmethod
    def from_config(cls, config):
        """
        Reconstructs the model from its configuration.
        Assumes that 'quantizer' is stored as a configuration dictionary.
        """
        # Custom config
        custom_config = config.pop('custom_config')
        # Pop quantizer config and create a quantizer 
        quantizer_config = custom_config.pop('quantizer') if 'quantizer' in custom_config else None
        # Reconstruct the quantizer using a helper function.
        quantizer = netsurf.QuantizationScheme.from_config(quantizer_config) if quantizer_config is not None else None
        # Now, pass the remaining config parameters to the constructor.
        return cls(quantizer, **custom_config, base_config = config)

    @property
    def in_shape(self):
        return self._config['in_shape']

    @property
    def out_shape(self):
        return self._config['out_shape']

    @property
    def quantizer(self):
        return self._config['quantizer']

    @property
    def use_bias(self):
        return self._config['use_bias']
    
    @property
    def use_constraint(self):
        return self._config['use_constraint']
    
    @property
    def type(self):
        return self._config['type']

    def count_trainable_parameters(self):
        return np.sum([np.prod(v.shape) for v in self.trainable_variables])
    
    def count_pruned_parameters(self):
        return np.sum([np.sum(v == 0) for v in self.trainable_variables])
    
    def count_non_trainable_parameters(self):
        return self.count_params() - self.trainable_parameters
    
    """ compute deltas """
    def compute_deltas(self, verbose = False):
        for ly in self.layers:
            if hasattr(ly, 'compute_deltas'):
                if verbose: print(f'Computing deltas for layer {ly.name}')
                ly.compute_deltas()

    """ Build the model """
    def build_model(self, ip, out, activations, **kwargs):
        # Finally we can call init method with this 
        # Parse kwargs
        loss = kwargs.pop('loss') if 'loss' in kwargs else self._config['loss']
        optimizer = kwargs.pop('optimizer') if 'optimizer' in kwargs else self._config['optimizer']
        metrics = kwargs.pop('metrics') if 'metrics' in kwargs else self._config['metrics']
        # Make sure to clean up metrics
        metrics = cleanup_metrics(metrics)
    
        type = kwargs.pop('type') if 'type' in kwargs else self.type
        use_bias = kwargs.pop('use_bias') if 'use_bias' in kwargs else self.use_bias
        use_constraint = kwargs.pop('use_constraint') if 'use_constraint' in kwargs else self.use_constraint

        # This first call is gonna be used to build the model, the normal model in -> out
        super().__init__(ip, out)

        # Now, let's create a new instance of QModel for the activations_model (intermediate model with
        # all the results in between layers). Because we are gonna be passing "ip" and "out" as arguments
        # this will call super itself without having to call "build_model" again.
        # activation_model = QModel(self.quantizer, self.in_shape, self.out_shape, 
        #                           ip = ip, out = activations, 
        #                           optimizer = optimizer, 
        #                           loss = loss, 
        #                           metrics = metrics,
        #                           type = type,
        #                           use_bias = use_bias,
        #                           use_constraint = use_constraint,
        #                           **kwargs)
        
        # Store 
        self._activations = activations

        # Finally, to set the loss and optimizer, and metrics, we can just compile the model
        self.compile(optimizer = optimizer, loss = loss, metrics = metrics)

        # Make sure we compute the deltas 
        self.compute_deltas()

        # Quantize weights 
        V_q = [self.quantizer(v) for v in self.trainable_variables]

        # Compute mean quantization error 
        quantization_error = tf.reduce_mean([tf.reduce_mean(tf.abs(v - v_q)) for v, v_q in zip(self.trainable_variables, V_q)]).numpy()

        # Get the deltas for each variable
        self.deltas = [self.quantizer.compute_delta_matrix(v) for v in V_q]
   
    # Plot model using tensorplot
    def plot_model(self, show_shapes=True, show_layer_names=True, expand_nested=True, display_params = True, path = None, verbose = True, 
                    **kwargs):
        """ Plot model """
        fileout = tp.plot_model(self, show_shapes = show_shapes, 
                             show_layer_names = show_layer_names, expand_nested = expand_nested,
                             display_params = display_params, path = path, verbose = False, 
                             display = False,
                             **kwargs)
        if isinstance(fileout, str):
            netsurf.utils.log._custom('MDL', f'Saving model plot @ {fileout}')
        return fileout
    
    # Create model name based on architecture 
    def create_model_name_by_architecture(self):
        name = f'{self.count_params()}_'
        for ly in self.layers:
            layer_name = ly.name.lower()
            layer_class = ly.__class__.__name__.lower()

            if 'input' in layer_class:
                continue
            elif 'conv' in layer_class:
                f = ly.filters
                k = ly.kernel_size
                s = ly.strides
                #p = ly.padding
                #d = ly.dilation_rate
                name += f'conv{f}f'
                if k[0] == k[1]:
                    name += f'{k[0]}k'
                else:
                    name += f'{k[0]}x{k[1]}k'
                if s[0] == s[1]:
                    name += f'{s[0]}s_'
                else:
                    name += f'{s[0]}s{s[1]}_'
            elif 'dense' in layer_class:
                u = ly.units
                name += f'dense{u}u_'
            elif 'pool' in layer_class:
                k = None
                s = None
                if hasattr(ly,'pool_size'):
                    k = ly.pool_size
                else:
                    name += 'global'
                if hasattr(ly, 'strides'):
                    s = ly.strides
                name += f'pool'
                if k is not None:
                    if k[0] == k[1]:
                        name += f'{k[0]}p'
                    else:
                        name += f'{k[0]}x{k[1]}p'
                if s is not None:
                    if s[0] == s[1]:
                        name += f'{s[0]}s_'
                    else:
                        name += f'{s[0]}s{s[1]}_'

            elif 'qactivation' in layer_class:
                if hasattr(ly.activation, '__class__'):
                    if hasattr(ly.activation.__class__, '__name__'):
                        act = ly.activation.__class__.__name__
                else:
                    act = 'qactivation'
                name += f'{act}_'
            elif 'activation' in layer_class:
                if hasattr(ly.activation,'__name__'):
                    act = ly.activation.__name__
                elif isinstance(ly.activation, str):
                    act = ly.activation.replace('quantized','').replace('_','').split('(')[0]
                name += f'{act}_'
            else:
                if layer_class.lower()[0] == 'q':
                    layer_class = layer_class[1:]
                # flatten, add, dropout, bnorm, etc
                name += f"{layer_class.lower().replace('batchnormalization','bn').replace('dropout','dp')}_"

        name = name[:-1]
        return name

    def compile(self, loss = None, optimizer = None, metrics = None, **kwargs):
        # Parse loss 
        loss = self._config['loss'] if loss is None else loss
        if isinstance(loss, str):
            loss = netsurf.dnn.losses.parse_loss(loss)
        
        # Metrics 
        metrics = self._config['metrics'] if metrics is None else metrics
        if metrics:
            if isinstance(metrics[0], str):
                metrics = netsurf.dnn.metrics.parse_metrics(metrics)
        
        # Optimizer
        optimizer = self._config['optimizer'] if optimizer is None else optimizer

        super().compile(loss = loss, optimizer = optimizer, metrics = metrics)

    """ Attack is calling but applying noise * deltas """
    def attack(self, inputs, N = None, return_activations = False, return_dict = False, clamp = False):
        # We will loop thru all layers and apply the N. This is not calling the attack, this is 
        # basically setting self.N, and then calling the predict method.
        """
        Performs a forward pass through the model, applying the attack method
        on layers that support it, while storing intermediate activations.

        Args:
            inputs: The input tensor(s) for the forward pass.
            N: Attack (dict, int, <Attack> obj)
            return_activations: If True, return the dictionary of all layer activations.

        Returns:
            - If return_activations=True: A dictionary mapping layer names to activations.
            - Otherwise: The final model output after the attack.

        Returns:
            layer_activations: A dictionary mapping layer names to their activations.
        """
        layer_activations = {}  # Store activations per layer

        # Go through layers following the original computational graph
        for layer in self.layers:
            # Get input tensor(s) for the current layer
            inbound_layers = []
            for node in layer._inbound_nodes:
                if isinstance(node.inbound_layers, list):
                    inbound_layers.extend(node.inbound_layers)  # Extend if list of layers
                else:
                    inbound_layers.append(node.inbound_layers)  # Append if single layer

            # Now safely extract activations
            inbound_tensors = [layer_activations[l.name] for l in inbound_layers if l.name in layer_activations]

            if len(inbound_tensors) == 1:
                x = inbound_tensors[0]  # Single input
            elif len(inbound_tensors) > 1:
                x = inbound_tensors  # Multiple inputs (residual connections, etc.)
            else:
                x = inputs  # First layer

            # Apply attack if the layer supports it, otherwise just do the forward pass
            if hasattr(layer, "attack"):
                layer_activations[layer.name] = layer.attack(x, N = N, clamp=clamp)
            else:
                layer_activations[layer.name] = layer(x)

        # Get the activation at the output of the model 
        # Extract layer names from output tensors
        output_layer_names = [tensor._keras_history[0].name for tensor in self.outputs]  # ✅ Gets layer names
        # Retrieve the activation at the output of the model
        
        if return_dict:
            output_activations = {name: layer_activations[name] for name in output_layer_names if name in layer_activations}
        else:
            output_activations = [layer_activations[name] for name in output_layer_names if name in layer_activations]
            output_activations = output_activations if len(output_activations) > 1 else output_activations[0]

        if return_activations:
            return output_activations, layer_activations

        return output_activations
    
    @property
    def bit_flip_counter(self):
        total_bit_flips = 0
        for layer in self.layers:
            if hasattr(layer, 'bit_flip_counter'):
                total_bit_flips += int(layer.bit_flip_counter.numpy())
        return total_bit_flips

    def reset_bit_flip_counter(self):
        for layer in self.layers:
            if hasattr(layer, 'bit_flip_counter'):
                layer.bit_flip_counter.assign(0)

    """ Inject is like attack, but instead of passing a pre-defined N, just passing the BER ratio and the protection ratio """
    def inject(self, inputs, BER = 0.0, protection = 0.0, return_activations = False, return_dict = False,  clamp = False):
        layer_activations = {}  # Store activations per layer

        # Go through layers following the original computational graph
        for layer in self.layers:
            # Get input tensor(s) for the current layer
            inbound_layers = []
            for node in layer._inbound_nodes:
                if isinstance(node.inbound_layers, list):
                    inbound_layers.extend(node.inbound_layers)  # Extend if list of layers
                else:
                    inbound_layers.append(node.inbound_layers)  # Append if single layer

            # Now safely extract activations
            inbound_tensors = [layer_activations[l.name] for l in inbound_layers if l.name in layer_activations]

            if len(inbound_tensors) == 1:
                x = inbound_tensors[0]  # Single input
            elif len(inbound_tensors) > 1:
                x = inbound_tensors  # Multiple inputs (residual connections, etc.)
            else:
                x = inputs  # First layer

            # Apply inject if the layer supports it, otherwise just do the forward pass
            if hasattr(layer, "inject"):
                layer_activations[layer.name] = layer.inject(x, BER = BER, protection = protection, clamp=clamp)
            else:
                layer_activations[layer.name] = layer(x)

        # Get the activation at the output of the model 
        # Extract layer names from output tensors
        output_layer_names = [tensor._keras_history[0].name for tensor in self.outputs]  # ✅ Gets layer names
        # Retrieve the activation at the output of the model
        
        if return_dict:
            output_activations = {name: layer_activations[name] for name in output_layer_names if name in layer_activations}
        else:
            output_activations = [layer_activations[name] for name in output_layer_names if name in layer_activations]
            output_activations = output_activations if len(output_activations) > 1 else output_activations[0]

        if return_activations:
            return output_activations, layer_activations

        return output_activations

    """ Evaluation model """
    def evaluate(self, x, y, plot = False, batch_size = 32, **kwargs):
        # Call super method to get the predictions and evalaution
        vals = super().evaluate(x, y, verbose = False, return_dict = True, batch_size = batch_size)

        if not plot:
            return vals

        # If we are here, it means the specific model doesn't have a specific private 
        # method for evaluating it. So let's try to figure out a generic way to evaluate it.
        if hasattr(self, 'type') :
            if self.type == 'classification':
                return self.evaluate_classification(x, y, **kwargs)
            elif self.type == 'regression':
                return self.evaluate_regression(x, y, **kwargs)
            elif self.type == 'unsupervised':
                return self.evaluate_unsupervised(x, y, **kwargs)

        # Else, just print a warning
        print(f'[WARN] - Evaluation not implemented yet for model with problem of type {self.type}')
    
    def evaluate_classification(self, x, y, **kwargs):
        # Plot ROC 
        yhat = self.predict(x, verbose = False)
        netsurf.utils.plot.plot_ROC(y, yhat, **kwargs)
        # And get confusion matrix 
        netsurf.utils.plot.plot_confusion_matrix(y, yhat, **kwargs)

    def evaluate_regression(self, x, y, **kwargs):
        # Get predictions
        test_predictions = self.predict(np.array(x), verbose = False).flatten()
        
        # Get labels for each regressed property 
        if isinstance(y, pd.Series):
            labels = [y.name.capitalize()]
        elif isinstance(y, pd.DataFrame):
            labels = [str(c).capitalize() for c in y.columns]
        else:
            labels = None
        
        # Call scatter plotter
        netsurf.utils.plot.plot_scatter(y, test_predictions, 
                                      title = 'Predictions', 
                                      xlabel = 'True', 
                                      ylabel = 'Predicted', 
                                      labels = labels, 
                                      **kwargs)
    
    def evaluate_unsupervised(self, x, y, axs = None, num_samples = 10, filepath = None, **kwargs):
        # Get predictions
        x = x[:num_samples]
        y = y[:num_samples]
        test_predictions = self.predict(np.array(x), verbose = False)

        vmin = np.min([np.min(y), np.min(test_predictions)])
        vmax = np.max([np.max(y), np.max(test_predictions)])

        if axs is not None:
            if len(axs) != num_samples:
                # Warn the user and chang
                netsurf.utils.log._warn(f'Number of samples ({num_samples}) does not match number of axes ({len(axs)}). Restarting axs.')
                axs = None

        # Plot pairs of images (true vs predicted)
        show_me = axs is None
        if axs is None:
            fig, axs = plt.subplots(num_samples, 2, figsize=(4, 2*num_samples))
        else:
            fig = axs[0,0].figure

        for i in range(num_samples):
            axs[i,0].imshow(y[i], vmin = vmin, vmax = vmax)
            axs[i,0].set_title('True')
            axs[i,0].axis('off')

            axs[i,1].imshow(test_predictions[i], vmin = vmin, vmax = vmax)
            axs[i,1].set_title('Predicted')
            axs[i,1].axis('off')

        plt.tight_layout()
        if filepath is not None:
            plt.savefig(filepath)
            plt.close()
        elif show_me:
            plt.show()
        else:
            return fig, axs
        

    def __repr__(self):
        cname = '🧠 QModel'
        if self.__class__.__name__ != 'QModel':
            cname = f'🧠 {self.__class__.__name__} <QModel>'
        s = f'{cname} @ ({hex(id(self))})\n'
        s += f'   - 🧮 Quantizer: {self.quantizer._scheme_str}\n'
        s += f'   - ▶ Input shape: {self.in_shape}\n'
        s += f'   - ◀ Output shape: {self.out_shape}\n'
        s += f'   - 🏃‍♂️ Optimizer: {self.optimizer}\n'
        s += f'   - ⚖️ Loss: {self.loss}\n'
        s += f'   - 📏 Metrics: {self._metrics}\n'
        icon = {'classification':'📊', 'regression':'📈', 'unsupervised': '↻'}.get(self.type, "")
        s += f'   - {icon} Type: {self.type}\n'
        s += f'   - ⚓️ Use bias: {self.use_bias}\n'
        s += f'   - ⛓️ Use constraint: {self.use_constraint}\n'
        s += f'   - 📚 Number of layers: {len(self.layers)}\n'
        return s

    """ Serialize """
    def serialize(self, include_emojis = False):
        cname = '🧠 QModel'
        if self.__class__.__name__ != 'QModel':
            cname = f'🧠 {self.__class__.__name__} (QModel): {self.name}'

        props = pd.DataFrame([{"🧮 Quantizer": self.quantizer._scheme_str,
               "▶ Input shape": str(self.in_shape),
               "◀ Output shape": str(self.out_shape),
               "🏃‍♂️ Optimizer": self.optimizer,
               #"⚖️ Loss": self.loss,
               #"📏 Metrics": f"{', '.join(self._metrics)}",
               "📈 Type": self.type,
               "⚓️ Use bias": "✅" if self.use_bias else "❌",
               "⛓️ Use constraint": "✅" if self.use_constraint else "❌",
               }]).T
        
        props = {'Global properties': props, 
                 '__options': {'collapsible': False}}
        
        content = [props]

        # Add summary table 
        # _________________________________________________________________
        # Layer (type)                Output Shape              Param #   
        # =================================================================
        # input_1 (InputLayer)        [(None, 1)]               0                                                           
        # fc0 (QQDense)               (None, 64)                128       
        # act0 (QQActivation)         (None, 64)                0         
        # fc1 (QQDense)               (None, 64)                4160      
        # act1 (QQActivation)         (None, 64)                0         
        # fc_out (QQDense)            (None, 1)                 65        
        # =================================================================
        # Total params: 4,353
        # Trainable params: 4,353
        # Non-trainable params: 0
        # _________________________________________________________________

        summary_table = pg.Table()
        
        # Add table header 
        thead = pg.Thead()
        thr = pg.Tr()
        thr.append(pg.Th(content='Layer (type)'))
        thr.append(pg.Th(content='Output Shape'))
        thr.append(pg.Th(content='Param #'))
        thead.append(thr)

        # Loop thru layers
        tbody = pg.Tbody()
        for i,ly in enumerate(self.layers):
            row = pg.Tr()
            row.append(pg.Td(content=f'{ly.name} ({ly.__class__.__name__})'))
            row.append(pg.Td(content=str(ly.output_shape)))
            row.append(pg.Td(content=str(ly.count_params())))
            tbody.append(row)
        
        total_params = self.count_params()
        total_params_str = f'Total params: {self.count_params()}'
        trainable_params = np.sum([np.prod(v.shape) for v in self.trainable_variables])
        trainable_params_str = f'Trainable params: {trainable_params}'
        non_trainable_params_str = f'Non-trainable params: {total_params-trainable_params}'

        # Add footer
        tfoot = pg.Tfoot()
        tfr = pg.Tr()
        tfr.append(pg.Td(content=total_params_str, attributes={'style': "text-align:left;",
                                                                'colspan': '3'}))
        tfoot.append(tfr)
        tfr = pg.Tr()
        tfr.append(pg.Td(content=trainable_params_str, attributes={'style': "text-align:left;",
                                                                   'colspan': '3'}))
        tfoot.append(tfr)
        tfr = pg.Tr()
        tfr.append(pg.Td(content=non_trainable_params_str, attributes={'style': "text-align:left;",
                                                                       'colspan': '3'}))
        tfoot.append(tfr)
            
        summary_table.append(thead)
        summary_table.append(tbody)
        summary_table.append(tfoot)

        # Wrap around a container, it'll look better
        summary_table = {'Summary': summary_table, 
                 '__options': {'collapsible': False}}

        content += [summary_table]

        # Add losses 
        if self.loss:
            if not hasattr(self.loss, '__call__'):
                loss = netsurf.dnn.losses.parse_loss(self.loss)
            else:
                loss = self.loss
            if loss:
                # Get the actual definition (code) of "model.loss"
                try:
                    loss_source, line_number = inspect.getsourcelines(loss)  # ✅ Get source lines & starting line
                except:
                    loss_source, line_number = inspect.getsourcelines(loss.__class__)
                
                loss_source = ''.join(loss_source)  # Convert list of lines into a string

                # Get the file path where the function/class is defined
                try:
                    filepath = inspect.getfile(loss).replace(os.path.expanduser("~"), "~")
                except:
                    filepath = inspect.getfile(loss.__class__).replace(os.path.expanduser("~"), "~")

                # Create a Markdown object inside a container with a header
                header = f"📄 **Defined in:** `{filepath}` (Line {line_number})"
                ct = pg.Container(header, 'vertical')

                # Display the function/class source code
                loss_md = pg.Markdown(f'```python\n{loss_source}\n```')
                ct.append(loss_md)

                # Append to content
                content.append({'📉 Loss': ct})
        
        # Now metrics 
        if self.metrics:
            _mets = []
            for im, m in enumerate(netsurf.dnn.metrics.parse_metrics(self.metrics)):
                # Get the actual definition (code) of "model.loss"
                m_source, line_number = inspect.getsourcelines(m.__class__)  # ✅ Get source lines & starting line
                m_source = ''.join(m_source)  # Convert list of lines into a string

                # Get the file path where the function/class is defined
                filepath = inspect.getfile(m.__class__).replace(os.path.expanduser("~"), "~")

                # Create a Markdown object inside a container with a header
                if isinstance(self.metrics[im], str):
                    metric_name = self.metrics[im].capitalize()
                elif hasattr(self.metrics[im], '__name__'):
                    metric_name = self.metrics[im].__name__.capitalize()
                elif hasattr(self.metrics[im], 'name'):
                    metric_name = self.metrics[im].name.capitalize()
                elif hasattr(self.metrics[im], '__class__'):
                    metric_name = self.metrics[im].__class__.__name__.capitalize()
                else:
                    metric_name = str(self.metrics[im]).capitalize()
                header = f"📄 {metric_name} **Defined in:** `{filepath}` (Line {line_number})"
                ct = pg.Container(header, 'vertical')

                # Display the function/class source code
                m_md = pg.Markdown(f'```python\n{m_source}\n```')
                ct.append(m_md)

                _mets.append(ct)

            if len(_mets) > 0:
                # Append to content
                content.append({'📏 Metric': _mets})

        # Now loop layer by layer and call their serialize method
        layers = {}
        for i, ly in enumerate(self.layers):
            if hasattr(ly, 'serialize') and hasattr(ly, 'html'):
                entry_name = f'({i+1})'
                if hasattr(ly, '_ICON'):
                    entry_name += f' {ly._ICON}'
                entry_name += f' {ly.name} ({ly.__class__.__name__})'
                layers[entry_name] = ly.html(include_emojis, flatten = i < (len(self.layers)-1))
        
        content.append({'📚 Layers': layers})

        # Let's add a plot with the architecture of the model using tensorplot
        # First create a temporary random filename in the /tmp/ folder for the 
        # output image
        tmp_name = io.create_temporary_file(prefix = self.create_model_name_by_architecture(), 
                                            suffix = '_model_tensorplot', 
                                            ext = '.png')

        tp.plot_model(self, tmp_name, show_shapes=True, show_layer_names=True, show=False, verbose = False)

        # Now create a pergamos container with the image, inside a collapsible container
        img = pg.Image(tmp_name, embed = True)
        img = {'🏛️ Architecture': img, 
               '__options': {'collapsible': True}}

        # Now append to content
        content.append(img)

        # We can now delete the tmp image file
        os.remove(tmp_name)

        # We are missing the layers container 📚
        return {cname: content}

    @pg.printable
    def html(self, include_emojis = False):
        return self.serialize(include_emojis=True)


""" FNN neural network for testing """
class FNN(QModel):
    def __init__(self, quantizer: 'QuantizationScheme', 
                 in_shape = (1,), out_shape = (1,), 
                 loss='mse', metrics=['r2score','pearsoncorrelation'],
                 units = [64, 64],
                 type = 'regression', **kwargs):
        
        # First call super
        super().__init__(quantizer, in_shape, out_shape, loss = loss, metrics = metrics, type = type, **kwargs)

        # Finally build the model
        self.build_model(quantizer, in_shape, out_shape, loss = loss, metrics = metrics, type = type, units = units, **kwargs)

        # Set units to self._config
        self._config['units'] = units

        # Make sure we set original metrics here
        self._metrics = metrics

    def build_model(self, quantizer: 'QuantizationScheme', in_shape, out_shape, **kwargs):
        super().build_model(*self.FNNmodel(quantizer, in_shape, out_shape, **kwargs), **kwargs)
    
    """ Build the model """
    def FNNmodel(self, quantizer: 'QuantizationScheme', in_shape, out_shape, 
                   use_bias = True, use_activation = True, use_constraint = True,
                   units = [64, 64], alpha_reg_loss = 1.0, **kwargs):

        """ Build input layer """
        ip = keras.layers.Input(shape = in_shape)

        """ Build model """
        x = ip
        activations = []

        # if use_norm:
        #     self.norm = tf.keras.layers.Normalization(input_shape=in_shape, axis=-1)
        #     x = self.norm(x)

        # If shape_in is multidimensional, flatten
        if len(in_shape) > 1:
            x = QQFlatten(quantizer, name = 'flatten')(x)
            # Append to activations
            activations += [x]
        
        for i, u in enumerate(units):
            x = QQDense(quantizer, u,
                        name = f'fc{i}',
                        kernel_quantizer = 'logit',
                        bias_quantizer = 'logit',
                        kernel_initializer = 'glorot_uniform',
                        bias_initializer = 'zeros',
                        use_bias = use_bias, 
                        use_constraint = use_constraint)(x)
            
            # Append to activations
            activations += [x]

            # Add QQApplyAlpha
            x = QQApplyAlpha(quantizer, reg_factor = alpha_reg_loss, name = f'alpha{i}')(x)

            # Append to activations
            activations += [x]

            if use_activation:
                x = QQActivation(quantizer,'relu', name=f'act{i}', use_constraint = use_constraint)(x)
                
                # Append to activations
                activations += [x]

        """ Dense pass """
        out = QQDense(quantizer, out_shape[0] if len(out_shape) > 0 else 1,
                            name = 'fc_out',
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            kernel_initializer = 'glorot_uniform',
                            bias_initializer = 'zeros',
                            use_bias = use_bias,
                            use_constraint = use_constraint)(x)

        # Append to activations
        activations += [out]

        # If this is a classification problem, add Softmax
        if self.type == 'classification':
            out = QQSoftmax(quantizer, name = 'softmax')(out)
            # Append to activations
            activations += [out]

        # if out_shape has more than 1 dimension, remember to reshape back
        if len(out_shape) > 1:
            out = QQReshape(quantizer, out_shape, name = 'reshape')(out)
            # Append to activations
            activations += [out]
    
        return ip, out, activations
    
    @classmethod
    def from_config(cls, config):
        # Custom config
        custom_config = config.pop('custom_config')
        # Pop quantizer config and create a quantizer 
        quantizer_config = custom_config.pop('quantizer') if 'quantizer' in custom_config else None
        quantizer = netsurf.QuantizationScheme.from_config(quantizer_config) if quantizer_config is not None else None
        return cls(quantizer, **custom_config, base_config = config)

    """ Evaluation model """
    def evaluate(self, x, y, **kwargs):
        return super().evaluate(x, y, **kwargs)


""" LSTM model """
# Taken from: https://github.com/SiliconLabs/platform_ml_models/blob/master/eembc/human_activity_recognition/har_lstm.py
class LSTM(QModel):
    def __init__(self, quantizer, in_shape=(28, 1), out_shape=(10,), optimizer='adam', 
                 loss='categorical_crossentropy', metrics=['accuracy'], type='classification', **kwargs):
        # First call super
        super().__init__(quantizer, in_shape, out_shape, 
                         optimizer = optimizer, loss = loss, metrics = metrics, 
                         type = type, **kwargs)

        # Finally build the model
        self.build_model(quantizer, in_shape, out_shape, loss = loss, metrics = metrics, type = type, **kwargs)

        # Make sure we set original metrics here
        self._metrics = metrics

    def build_model(self, quantizer: 'QuantizerScheme', in_shape, out_shape, *args, **kwargs):
        super().build_model(*self.lstmmodel(quantizer, in_shape, out_shape, *args, **kwargs), **kwargs)

    """ Build model method """
    def lstmmodel(self, quantizer, in_shape, out_shape, use_bias = True, use_constraint = True, **kwargs):

        # Input layer
        ip = keras.layers.Input(shape=in_shape)

        # Acitvations
        activations = []

        # Build net
        x = QQDense(quantizer, 32,
                            activation = 'relu',
                            use_bias = use_bias,
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            kernel_initializer = 'glorot_uniform',
                            bias_initializer = 'zeros',
                            use_constraint=use_constraint)(ip)
        # Append to activations
        activations += [x]

        x = QQDropout(quantizer, 0.1, use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        x = QQLSTM(quantizer, 64, 
                        activation='tanh', 
                        recurrent_activation='sigmoid',
                        return_sequences=False, 
                        use_bias = use_bias,
                        kernel_quantizer = 'logit',
                        bias_quantizer = 'logit',
                        kernel_initializer = 'glorot_uniform',
                        bias_initializer = 'zeros',
                        kernel_regularizer=keras.regularizers.l2(0.00001),
                        recurrent_regularizer=keras.regularizers.l2(0.00001),
                        use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        x = QQDense(quantizer, 8,
                            use_bias = use_bias,
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            kernel_initializer = 'glorot_uniform',
                            bias_initializer = 'zeros',
                            use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]
        
        x = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        x = QQDropout(quantizer, 0.4, use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        x = QQDense(quantizer, 16,
                         use_bias = use_bias,
                         kernel_quantizer = 'logit',
                         bias_quantizer = 'logit',
                         kernel_initializer = 'glorot_uniform',
                         bias_initializer = 'zeros',
                         use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        x = QQActivation(quantizer,'relu', use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        x = QQDense(quantizer, out_shape[0],
                         use_bias = use_bias,
                         kernel_quantizer = 'logit',
                         bias_quantizer = 'logit',
                         kernel_initializer = 'glorot_uniform',
                         bias_initializer = 'zeros',
                         use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]
        
        out = QQSoftmax(quantizer, use_constraint=use_constraint)(x)
        # Append to activations
        activations += [out]

        return ip, out, activations 
    
    # Create model name based on architecture 
    def create_model_name_by_architecture(self):
        name = f'{self.count_params()}_'
        name += f'LSTM'
        return name
    
    def __evaluate(self, *args, **kwargs):
        raise NotImplementedError('This method is not implemented yet for this model')


""" Build the neural network now """
class CNN(QModel):
    def __init__(self, quantizer, in_shape=(28, 28, 1), out_shape=(10, ), 
                 optimizer='adam', loss='categorical_crossentropy', 
                 metrics=['accuracy'], type='classification', multiplier = 1,
                 **kwargs):
        # First call super
        super().__init__(quantizer, in_shape, out_shape, optimizer, loss, metrics, type, **kwargs)

        # Specific attrs
        self.multiplier = multiplier

        # Finally build the model
        self.build_model(quantizer, in_shape, out_shape, loss = loss, metrics = metrics, type = type, **kwargs)

        # Make sure we set original metrics here
        self._metrics = metrics

    def build_model(self, quantizer: 'QuantizationScheme', in_shape, out_shape, **kwargs):
        super().build_model(*self.cnnmodel(quantizer, in_shape, out_shape, **kwargs), **kwargs)

    """ Build the model """
    def cnnmodel(self, quantizer, in_shape = (28, 28, 1), out_shape = (10,), 
                    use_bias = True, lite = False, use_constraint = True, **kwargs):

        """ Build input layer """
        ip = keras.layers.Input(shape = in_shape)

        # Acitvations
        activations = []

        # Define filters 
        f = np.maximum(1,int(np.ceil(self.multiplier*18)))

        """ Stack convolutional layers"""
        x = QQConv2D(quantizer, f, (3, 3), 
                            name = "conv2d_1",
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            use_bias = use_bias,
                            kernel_initializer = 'lecun_uniform',
                            bias_initializer = 'zeros',
                            kernel_regularizer = keras.regularizers.l1(0.0001),
                            use_constraint=use_constraint)(ip)
        # Append to activations
        activations += [x]

        x = QQActivation(quantizer, 'relu', name="act_1", use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        if not lite:
            f = np.maximum(1,int(np.ceil(self.multiplier*32)))
            x = QQConv2D(quantizer, f, (3, 3), 
                                name = "conv2d_2",
                                kernel_quantizer = 'logit',
                                bias_quantizer = 'logit',
                                use_bias = use_bias,
                                kernel_initializer = 'lecun_uniform',
                                bias_initializer = 'zeros',
                                kernel_regularizer = keras.regularizers.l1(0.0001),
                                use_constraint=use_constraint)(x)
            
            # Append to activations
            activations += [x]

            x = QQActivation(quantizer, 'relu', name="act_2", use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]

        else:
            x = QQMaxPooling2D(quantizer, (2,2), name = 'pool_1', use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]

            # [@manuelbv]: Stack a quantization layer after this
            x = QQActivation(quantizer, 'logit', use_constraint= use_constraint)(x)
            # Append to activations
            activations += [x]

        """ Flatten """
        x = QQFlatten(quantizer, name = "flatten", use_constraint = use_constraint)(x)
        # Append to activations
        activations += [x]

        """ Dense pass """
        if not lite:
            x = QQDense(quantizer, 128,
                                name = 'fc1',
                                kernel_quantizer = 'logit',
                                bias_quantizer = 'logit',
                                use_bias = use_bias,
                                kernel_initializer = 'lecun_uniform',
                                bias_initializer = 'zeros',
                                kernel_regularizer = keras.regularizers.l1(0.0001),
                                use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]
        
            x = QQActivation(quantizer, 'relu', name="act_3", use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]

        x = QQDense(quantizer, out_shape[0],
                        name = 'fc2' if not lite else 'fc1',
                        kernel_quantizer = 'logit',
                        bias_quantizer = 'logit',
                        use_bias = use_bias,
                        kernel_initializer = 'lecun_uniform',
                        bias_initializer = 'zeros',
                        kernel_regularizer = keras.regularizers.l1(0.0001),
                        use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]
        

        """ Softmax """
        out = QQSoftmax(quantizer, name="softmax", use_constraint=use_constraint)(x)
        # Append to activations
        activations += [out]

        return ip, out, activations
    
    def __evaluate(self, *args, **kwargs):
        raise NotImplementedError('This method is not implemented yet for this model')
    

# As seen in here: https://github.com/fastmachinelearning/hls4ml-tutorial/blob/main/part6_cnns.ipynb
class hls4mlCNN(QModel):
    def __init__(self, quantizer, in_shape=(28, 28, 1), out_shape=(10, ),  
                 optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], 
                 type='classification', **kwargs):
        super().__init__(quantizer, in_shape, out_shape, optimizer = optimizer, loss = loss, 
                         metrics = metrics, type = type, **kwargs)

        # Finally build the model
        self.build_model(quantizer, in_shape, out_shape, 
                         optimizer = optimizer, 
                         loss = loss, 
                         metrics = metrics, 
                         type = type, 
                         **kwargs)

        # Make sure we set original metrics here
        self._metrics = metrics

    """ Build """
    def build_model(self, quantizer: 'QuantizationScheme', in_shape, out_shape, *args, **kwargs):
        super().build_model(*self.hls4mlmodel(quantizer, in_shape, out_shape, *args, **kwargs), **kwargs)
    
    """ Build the model """
    def hls4mlmodel(self, quantizer: 'QuantizationScheme', in_shape, out_shape,
                    use_bias = True, filters = [16, 16, 24], dense_units = [42, 64], 
                    use_constraint = True, **kwargs):


        """ Build input layer """
        x = ip = keras.layers.Input(shape = in_shape)

        # init activations
        activations = []

        """ Stack convolutional layers"""
        for i, f in enumerate(filters):
            x = QQConv2D(quantizer, int(f), (3, 3), strides = (1,1),
                                kernel_quantizer = 'logit',
                                bias_quantizer = 'logit',
                                use_bias = True,
                                kernel_initializer = 'lecun_uniform',
                                bias_initializer = 'zeros',
                                kernel_regularizer = keras.regularizers.l1(0.0001),
                                use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]

            # Add QQApplyAlpha
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
                                            use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]
            
            x = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]

            x = QQMaxPooling2D(quantizer, pool_size=(2, 2), use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]
            
            # Extra quantization
            x = QQActivation(quantizer, 'logit', use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]

        """ Flatten """
        x = QQFlatten(quantizer, use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]

        """ Stack dense layers """
        for i, n in enumerate(dense_units):
            x = QQDense(quantizer, n,
                        kernel_quantizer = 'logit',
                        bias_quantizer = 'logit',
                        use_bias = True,
                        kernel_initializer = 'lecun_uniform',
                        bias_initializer = 'zeros',
                        kernel_regularizer = keras.regularizers.l1(0.0001),
                        use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]

            # Add QQApplyAlpha
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
                                            use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]
            
            x = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]

        """ Last dense """
        x = QQDense(quantizer, out_shape[0],
                        kernel_quantizer = 'logit',
                        bias_quantizer = 'logit',
                        kernel_initializer = 'lecun_uniform',
                        bias_initializer = 'zeros',
                        use_bias = True,
                        use_constraint=use_constraint)(x)
        
        # Append to activations
        activations += [x]

        """ Softmax """
        out = QQSoftmax(quantizer, name="softmax", use_constraint=use_constraint)(x)
        # Append to activations
        activations += [out]

        return ip, out, activations
    
    """ Evaluation model """
    def __evaluate(self, x, y, filepath = None, show = True, **kwargs):
        
        """ Evaluate the model and plot ROC, etc. """
        predict_baseline = self.predict(x, verbose = False)

        # Invoke ROC
        num_bits = self.quantizer.m
        netsurf.utils.plots.plot_ROC(y, predict_baseline, num_bits = num_bits, filepath = filepath, show = show)
    
    # Create model name based on architecture 
    def create_model_name_by_architecture(self):
        name = f'{self.count_params()}_'
        name += 'hls4ml_cnn'
        return name


# """ Build the neural network now """
# class FNN(QModel):
#     def __init__(self, quantizer, in_shape=(28, 28, 1), out_shape=(10, ), 
#                  optimizer='adam', loss='categorical_crossentropy', 
#                  metrics=['accuracy'], type='classification', **kwargs):
#         # First call super
#         super().__init__(quantizer, in_shape, out_shape, optimizer = optimizer, loss = loss, 
#                          metrics = metrics, type = type, **kwargs)

#         # Finally build the model
#         self.build_model(quantizer, in_shape, out_shape, loss = loss, metrics = metrics, type = type, **kwargs)

#         # Make sure we set original metrics here
#         self._metrics = metrics

#     def build_model(self, quantizer: 'QuantizationScheme', in_shape, out_shape, *args, **kwargs):
#         super().build_model(*self.fnnmodel(quantizer, in_shape, out_shape, *args, **kwargs), **kwargs)
    
#     """ Build the model """
#     def fnnmodel(self, quantizer: 'QuantizationScheme', in_shape, out_shape,   
#                     use_bias = True, use_constraint = True, **kwargs):

#         """ Build input layer """
#         ip = keras.layers.Input(shape = in_shape)

#         # Init activations
#         activations = []

#         x = QQMaxPooling2D(quantizer, (2,2), name = 'pool_1', use_constraint=use_constraint)(ip)
#         # Append to activations
#         activations += [x]

#         """ Flatten """
#         x = QQFlatten(quantizer, name = "flatten", use_constraint=use_constraint)(x)
#         # Append to activations
#         activations += [x]

#         """ Dense pass """
#         x = QQDense(quantizer,np.prod(out_shape),
#                             name = 'fc1',
#                             kernel_quantizer = 'logit',
#                             bias_quantizer = 'logit',
#                             use_bias = use_bias,
#                             kernel_initializer = 'lecun_uniform',
#                             bias_initializer = 'zeros',
#                             kernel_regularizer = keras.regularizers.l1(0.0001),
#                             use_constraint=use_constraint)(x)
#         # Append to activations
#         activations += [x]

#         """ Softmax """
#         out = QQSoftmax(quantizer, name="softmax", use_constraint=use_constraint)(x)
#         # Append to activations
#         activations += [out]

#         return ip, out, activations
    
    


""" Build AE """
# Taken from: https://github.com/SiliconLabs/platform_ml_models/blob/master/eembc/ToyADMOS_FC_AE/toyadmos_autoencoder_eembc.py
class AE(QModel):
    def __init__(self, quantizer, in_shape = (28, 28, 1), out_shape = (28, 28, 1),
                 optimizer='adam', loss='mae', metrics=['r2score','pearsoncorrelation'], 
                 type='unsupervised', **kwargs):
        
        super().__init__(quantizer, in_shape, in_shape, optimizer = optimizer, loss = loss, 
                         metrics = metrics, type = type, **kwargs)
        
        # Finally build the model
        self.build_model(quantizer, in_shape, in_shape, loss = loss, metrics = metrics, type = type, **kwargs)

        # Make sure we set original metrics here
        self._metrics = metrics

    def build_model(self, quantizer: 'QuantizationScheme', in_shape, out_shape, **kwargs):
        super().build_model(*self.aemodel(quantizer, in_shape, **kwargs), **kwargs)

    """ Build model method """
    def aemodel(self, quantizer: 'QuantizationScheme', in_shape, use_bias = True, 
                use_constraint = True, **kwargs):

        # Input layer
        ip = keras.layers.Input(shape=in_shape)

        # Activations
        activations = []
        
        # Flatten input image
        x = QQFlatten(quantizer, use_constraint=use_constraint)(ip)
        # Append to activations
        activations += [x]

        # Encoding block
        for _ in range(4):
            # First encoder layer
            x = QQDense(quantizer, 128, 
                            use_bias = use_bias,
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            kernel_initializer = 'lecun_uniform',
                            bias_initializer = 'zeros',
                            use_constraint=use_constraint)(x)
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
                                            use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]
            
            x = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(x) 
            # Append to activations
            activations += [x]

        # Latent layer
        x = QQDense(quantizer, 8, 
                        kernel_quantizer='logit', 
                        bias_quantizer = 'logit',
                        kernel_initializer = 'lecun_uniform',
                        bias_initializer = 'zeros',
                        use_constraint=use_constraint)(x)
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
                                        use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]
        
        x = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(x) 
        # Append to activations
        activations += [x]

        # Decoder block 
        for _ in range(4):

            # First decoder layer
            x = QQDense(quantizer, 128,
                            use_bias = use_bias,
                            kernel_quantizer = 'logit',
                            bias_quantizer = 'logit',
                            kernel_initializer = 'lecun_uniform',
                            bias_initializer = 'zeros',
                            use_constraint=use_constraint)(x)
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
                                            use_constraint=use_constraint)(x)
            # Append to activations
            activations += [x]
            
            x = QQActivation(quantizer, 'relu', use_constraint=use_constraint)(x) 
            # Append to activations
            activations += [x]

        # Output layer
        x = QQDense(quantizer, np.prod(in_shape),
                        use_bias = use_bias,
                        kernel_quantizer = 'logit',
                        bias_quantizer = 'logit',
                        kernel_initializer = 'lecun_uniform',
                        bias_initializer = 'zeros',
                        use_constraint=use_constraint)(x)
        # Append to activations
        activations += [x]
        
        decoded = QQReshape(quantizer, in_shape)(x)
        # Append to activations
        activations += [decoded]

        return ip, decoded, activations
    
    def evaluate(self, x, y, **kwargs):
        return super().evaluate(x, y, batch_size = 3, **kwargs)

    """ Evaluate and plot """
    def __evaluate(self, x, x_val, verbose=True, show = True, **kwargs):
        
        xhat = self(x)
        xhat_val = self(x_val)

        mse = np.mean((x - xhat)**2)

        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        ax.scatter(x.flatten(), xhat.numpy().flatten(), linewidth = 0.3, edgecolor = (0,0,0,0.3), alpha = 0.5)
        ax.scatter(x_val.flatten(), xhat_val.numpy().flatten(), linewidth = 0.3, edgecolor = (0,0,0,0.3), alpha = 0.3, color = 'orange')

        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')

        if show:
            plt.show()
        else:
            plt.close(fig)

        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        _ = ax.hist(x.flatten(), bins = 128, alpha = 0.5, label = 'Original')
        _ = ax.hist(xhat.numpy().flatten(), bins = 128, alpha = 0.5, label = 'Reconstructed')
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.legend()

        if show:
            plt.show()
        else:
            plt.close(fig)
    
    # Create model name based on architecture 
    def create_model_name_by_architecture(self):
        name = f'{self.count_params()}_'
        name += f'AE_{"x".join(list(map(str,self.in_shape)))}_{len(self.layers)}layers'
        return name


# Register these models 
BASIC_MODELS = {'FNN': FNN, 'LSTM': LSTM, 'AE': AE, 'CNN': CNN, 'hls4mlCNN': hls4mlCNN}

from .legacy import *
from .resnet import *
from .econ import *
from .smart_pixel import *

# Register all models
netsurf.MODELS.update(BASIC_MODELS)
netsurf.MODELS.update(SMARTPIXMODELS)
netsurf.MODELS.update(RESNETMODELS)
netsurf.MODELS.update(ECONMODELS)
netsurf.MODELS.update(LEGACYMODELS)


""" TFLite model """
# Taken from: https://www.tensorflow.org/lite/examples
# Not implemented yet


""" Main get model function """
def get_model(quantization, model, *args, **kwargs):
    # Get models registered 
    models_map = netsurf.MODELS

    assert model.lower() in list([m.lower() for m in models_map]), f'Model {model} not found in registered models. Available models are: {models_map.keys()}'
    if 'verbose' in kwargs:
        kwargs.pop('verbose')
    return {kw.lower(): models_map[kw] for kw in models_map}[model.lower()](quantization, *args, verbose = False, **kwargs)


def cleanup_metrics(metrics):
    # Find unique
    metrics = list(set(metrics))
    # Keep only callable metrics
    metrics = [m for m in metrics if callable(m)]

    return metrics


def get_custom_objects(quantization_scheme, wrap = None):
    custom_objects = {}

    # (Optional) Add any additional quantized objects needed by your application.
    _add_supported_quantized_objects(custom_objects)

    # Add pruning wrapper support if you're using pruning.
    #custom_objects['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude

    # Register your custom model class.
    custom_objects['QModel'] = QModel

    # Now register all custom models 
    for name, model_cls in netsurf.MODELS.items():
        custom_objects[name] = model_cls
        
    # Register the base QQLayer.
    custom_objects['QQLayer'] = _QQLayer
    custom_objects['_QQLayer'] = _QQLayer

    # Also the prunablelayer
    custom_objects['PrunableLayer'] = PrunableLayer

    # All the rest 
    for name, layer_cls in netsurf.QQLAYERS.items():
        custom_objects[name] = layer_cls
        if name != layer_cls.__class__.__name__:
            custom_objects[layer_cls.__class__.__name__] = layer_cls

    # Register custom metrics and losses if needed.
    # Uncomment and modify the following lines if required:
    for name, metric_cls in netsurf.METRICS.items():
        custom_objects[name] = metric_cls
        # Also add with the lowercase name
        if name != metric_cls.__class__.__name__:
            custom_objects[metric_cls.__class__.__name__] = metric_cls

    # Same for losses 
    for name, loss_cls in netsurf.LOSSES.items():
        custom_objects[name] = loss_cls
        if name != loss_cls.__class__.__name__:
            custom_objects[loss_cls.__class__.__name__] = loss_cls

    return custom_objects

def load_model(path, quantization_scheme):
    """
    Loads a Keras model from the given path while automatically registering all
    custom objects. This includes:
      - The custom model class (QModel)
      - Custom layers (derived from QQLayer) that require a quantization_scheme as the first argument
      - Custom metrics, losses, and any additional quantized objects.
    
    Parameters:
        path (str): The file path to the saved model.
        quantization_scheme: Your custom quantization scheme object required by your layers/models.
    
    Returns:
        A Keras model instance with all custom objects properly registered.
    """
    custom_objects = get_custom_objects(quantization_scheme, wrap = True)
    
    # Finally, load the model with the custom objects.
    model = tf.keras.models.load_model(path, custom_objects=custom_objects)

    # clean up metrics 
    metrics = cleanup_metrics(model.metrics)
    
    # Recompile
    model.compile(optimizer = model.optimizer, loss = model.loss, metrics = metrics)
    model._metrics = metrics 

    # If pruning wrappers were used during training, strip them.
    model = strip_pruning(model)
    
    return model