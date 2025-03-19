import tensorflow as tf
import pandas as pd
import numpy as np

from .models import QModel
from .layers import QQDense, QQSigmoid


class BitFlipModel(QModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Now we need to make sure that only BitFlipDense layers are trainable, 
        # the rest should be non-trainable
        for layer in self.layers:
            if isinstance(layer, BitFlipDense):
                layer.trainable = True
            else:
                layer.trainable = False
    
    def call(self, inputs, use_P=True, full_P=True, use_frozen=False, **kwargs):
        """Forward pass with optional bit-flip injection."""
        x = inputs
        for layer in self.layers:
            if isinstance(layer, BitFlipDense):
                x = layer(x, use_P=use_P, full_P=full_P, use_frozen=use_frozen, **kwargs)
            else:
                x = layer(x, **kwargs)
        return x


class BitFlipDense(QQDense):
    """Custom Dense layer that injects bit flips based on trainable probabilities P, 
       with additional tracking for ranking, freezing, and iteration control."""
    
    def __init__(self, original_layer, quantizer, name=None, **kwargs):
        # Get in_shape and out_shape from original layer
        in_shape = original_layer.input_shape
        out_shape = original_layer.output_shape
        units = original_layer.units
        
        super().__init__(quantizer, units, name=name, **kwargs)
        
        self.units = original_layer.units
        self.activation = original_layer.activation
        self.quantizer = quantizer
        self.original_layer = original_layer
        self.built = False
        self.delta_computed = False
        self.rank_index = 0  # Iteration tracker

        # Create ranking dataframe
        self.ranking = pd.DataFrame(columns=['param', 'P_prime', 'P', 'i', 'j', 'bit', 'rank_index'])
        self.ranking = self.ranking.astype({'param': str, 'P_prime': float, 'P': float, 'i': int, 'j': int, 'bit': int, 'rank_index': int})


    def build(self, input_shape):
        # Make sure that neither kernel nor bias are trainable
        # Get the names for the kernel and bias
        kname = self.original_layer.kernel.name
        bname = self.original_layer.bias.name
        # Find these names in self.trainable_variables and set them to False
        vnames = [v.name for v in super().trainable_variables]

        # Unfortunately we cannot set the "trainable" flag directly for params,
        # so we will have to create a property that will return the correct values        
        self.kernel = super().trainable_variables[vnames.index(kname)]
        self.bias = super().trainable_variables[vnames.index(bname)]

        # Trainable probability tensor P'
        self.P_prime = self.add_weight(name="P_prime", shape=(self.kernel.shape[0], self.kernel.shape[1], 
                                                                self.quantizer.m),
                                        initializer=tf.keras.initializers.Constant(-4.0), trainable=True)

        # Freezing mask and indices
        self.frozen_mask = self.add_weight(name="frozen_mask", shape=self.P_prime.shape, initializer="zeros", trainable=False)
        self.frozen_index = self.add_weight(name="frozen_index", shape=self.P_prime.shape, initializer=tf.keras.initializers.Constant(-1), trainable=False)
        self.frozen_P_prime = self.add_weight(name="frozen_P_prime", shape=self.P_prime.shape, initializer="zeros", trainable=False)

        # Add deltas 
        self.delta = self.add_weight(name="delta", shape=self.P_prime.shape, initializer="zeros", trainable=False)
        self.bias_delta = self.add_weight(name="bias_delta", shape=(self.bias.shape[0], self.quantizer.m), initializer="zeros", trainable=False)

        self.built = True

    def compute_deltas(self):
        """
        Computes delta values for each bit in the quantization scheme.
        Delta represents how much a bit flip affects the weight value.
        """
        """Applies the quantization scheme using `tf.numpy_function` to handle NumPy-based operations."""
        def numpy_quantization(weight):
            w = self.quantizer(weight).astype(np.float32)  # Ensure float32
            delta = self.quantizer.compute_delta_matrix(w)
            return delta.astype(np.float32)

        delta = tf.numpy_function(numpy_quantization, [self.kernel], tf.float32)
        self.delta.assign(delta)

        # Same for bias 
        bias_delta = tf.numpy_function(numpy_quantization, [self.bias], tf.float32)
        self.bias_delta.assign(bias_delta)

        self.delta_computed = True


    def inject_bit_flips(self, full_P=True, use_frozen=False):
        """Applies bit flips based on trainable P values."""
        P = tf.keras.activations.sigmoid(self.frozen_P_prime if use_frozen else self.P_prime)

        if not self.delta_computed is None:
            self.compute_deltas()
        # Apply freezing mask
        if full_P:
            P = P * (1.0 - self.frozen_mask)
        else:
            mask = tf.equal(self.frozen_index, self.rank_index - 1)
            P = P * tf.cast(mask, tf.float32)

        # Compute bit-flip perturbations
        perturbed_kernel = self.kernel + tf.reduce_sum(self.delta * P, axis=-1)

        return perturbed_kernel

    def call(self, inputs, use_P=True, full_P=True, use_frozen=False):
        """Forward pass with optional bit-flip injection."""
        W = self.inject_bit_flips(full_P, use_frozen) if use_P else self.kernel
        output = tf.matmul(inputs, W) + self.bias
        return self.activation(output) if self.activation else output

    def rank_and_freeze(self, threshold=0.95):
        """Ranks and freezes bits that exceed the given probability threshold."""
        P_prime_vals = self.P_prime.numpy()
        P_vals = tf.keras.activations.sigmoid(self.P_prime).numpy()

        # Sort by P_prime values (highest first)
        ranked_indices = np.argsort(P_prime_vals.flatten())[::-1]
        ranked_indices = np.unravel_index(ranked_indices, P_prime_vals.shape)

        # Determine how many bits to freeze
        num_to_add = int((1 - threshold) * np.sum(self.frozen_mask.numpy() == 0))

        # Construct ranking DataFrame
        subdf = pd.DataFrame({'param': [self.name] * num_to_add,
                              'P_prime': P_prime_vals[ranked_indices][:num_to_add],
                              'P': P_vals[ranked_indices][:num_to_add],
                              'i': ranked_indices[0][:num_to_add],
                              'j': ranked_indices[1][:num_to_add],
                              'bit': ranked_indices[2][:num_to_add],
                              'rank_index': [self.rank_index] * num_to_add})

        self.ranking = pd.concat([self.ranking, subdf], ignore_index=True)

        # Freeze the selected bits
        for i in range(num_to_add):
            idx = (ranked_indices[0][i], ranked_indices[1][i], ranked_indices[2][i])
            self.frozen_mask[idx].assign(1)
            self.frozen_index[idx].assign(self.rank_index)
            self.frozen_P_prime[idx].assign(self.P_prime[idx])
            self.P_prime[idx].assign(-10.0)  # Push training P values towards 0

        print(f"({np.mean(self.frozen_mask.numpy()):.2%}) Freezing {num_to_add} weights in {self.name}")

        # Increment rank index
        self.rank_index += 1

        return subdf
    
    @property 
    def trainable_variables(self):
        return [self.P_prime]
    
    @property
    def non_trainable_variables(self):
        return [self.kernel, self.bias, self.frozen_mask, self.frozen_index, self.frozen_P_prime, self.delta, self.bias_delta]
    
class ModelWrapper:
    """Wraps a given model, replacing Dense layers with BitFlipDense for ranking."""
    
    def __init__(self, model, quantizer):
        self.original_model = model
        self.quantizer = quantizer
        
        # Set params 
        self.in_shape = model.input_shape
        self.out_shape = model.output_shape
        self.optimizer = model.optimizer 
        self.loss = model.loss
        self.metrics = model.metrics
        self.type = model.type
        self.use_bias = model.use_bias
        self.use_constraint = model.use_constraint

        self.wrapped_model = self._wrap_model()


    def _wrap_model(self):
        """Clones the original model and replaces Dense layers with BitFlipDense."""
        inputs = tf.keras.layers.Input(shape=self.original_model.input_shape[1:])
        x = inputs

        for layer in self.original_model.layers:
            if layer.__class__.__name__ == 'InputLayer':
                continue
            elif isinstance(layer, tf.keras.layers.Dense) or isinstance(layer, QQDense):
                x = BitFlipDense(layer, self.quantizer, name=layer.name)(x)
            else:
                # Generic wrapper (we need to redirect the trainable variables)
                x = layer(x)

        return BitFlipModel(self.quantizer, self.in_shape, self.out_shape, 
                        optimizer=self.optimizer, loss=self.loss, metrics=self.metrics,
                        type=self.type, use_bias=self.use_bias, use_constraint=self.use_constraint,
                        ip=inputs, out=x, name="wrapped_model")

    def train_P(self, X, Y, num_epochs=10, lr=0.01, lambda_reg=0.01):
        """Trains the P values using gradient descent."""
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        loss_fn = tf.keras.losses.MeanSquaredError()

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            # Make sure to watch the variables you want gradients for
            trainable_vars = self.wrapped_model.trainable_variables

            with tf.GradientTape() as tape:
                perturbed_output = self.wrapped_model(X, use_P=True)
                clean_output = self.wrapped_model(X, use_P=False)

                # Normalize loss
                mse_loss = loss_fn(clean_output, perturbed_output) / tf.size(perturbed_output, out_type=tf.float32)

                # L1 Regularization
                reg_loss = lambda_reg * (tf.reduce_sum([tf.reduce_sum(tf.keras.activations.sigmoid(var)) for var in self.wrapped_model.trainable_variables]))

                total_loss = -mse_loss + reg_loss

            # Compute gradients and update P
            grads = tape.gradient(total_loss, trainable_vars)
            optimizer.apply_gradients(zip(grads, trainable_vars))

            epoch_loss += total_loss.numpy()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}, Reg: {reg_loss.numpy():.6f}, MSE: {mse_loss.numpy():.6f}")

            # Iteratively freeze bits
            for layer in self.wrapped_model.layers:
                if isinstance(layer, BitFlipDense):
                    layer.rank_and_freeze()
