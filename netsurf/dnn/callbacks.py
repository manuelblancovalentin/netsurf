import os 

# typing
from typing import Iterable, Union

import tensorflow as tf 
import warnings

import numpy as np

from keras import backend

# netsurf
import netsurf
from .layers import QQApplyAlpha, PrunableLayer


class CustomModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
    
    def print_msg(self, msg):
        netsurf.utils.log._custom('MDL',msg)

    def _save_model(self, epoch, batch, logs):
        """Saves the model.

        Args:
            epoch: the epoch this iteration is in.
            batch: the batch this iteration is in. `None` if the `save_freq`
                is set to `"epoch"`.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        filepath = self._get_file_path(epoch, batch, logs)
        # Create host directory if it doesn't exist.
        dirname = os.path.dirname(filepath)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname, exist_ok=True)

        try:
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn(
                        f"Can save best model only with {self.monitor} "
                        "available, skipping.",
                        stacklevel=2,
                    )
                elif (
                    isinstance(current, np.ndarray)
                    or backend.is_tensor(current)
                ) and len(current.shape) > 0:
                    warnings.warn(
                        "Can save best model only when `monitor` is "
                        f"a scalar value. Received: {current}. "
                        "Falling back to `save_best_only=False`."
                    )
                    self.model.save(filepath, overwrite=True)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            self.print_msg(
                                f"Epoch {epoch}: {self.monitor} "
                                "improved "
                                f"from {self.best:.5f} to {current:.5f}, "
                                f"saving model to {filepath}"
                            )
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            self.print_msg(
                                f"Epoch {epoch}: "
                                f"{self.monitor} did not improve "
                                f"from {self.best:.5f}"
                            )
            else:
                if self.verbose > 0:
                    self.print_msg(
                        f"Epoch {epoch}: saving model to {filepath}"
                    )
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
        except IsADirectoryError:  # h5py 3.x
            raise IOError(
                "Please specify a non-directory filepath for "
                "ModelCheckpoint. Filepath used is an existing "
                f"directory: {filepath}"
            )
        except IOError as e:  # h5py 2.x
            # `e.errno` appears to be `None` so checking the content of
            # `e.args[0]`.
            if "is a directory" in str(e.args[0]).lower():
                raise IOError(
                    "Please specify a non-directory filepath for "
                    "ModelCheckpoint. Filepath used is an existing "
                    f"directory: f{filepath}"
                )
            # Re-throw the error for any other causes.
            raise e

    def _get_file_path(self, epoch, batch, logs):
        """Returns the file path for checkpoint."""

        try:
            # `filepath` may contain placeholders such as
            # `{epoch:02d}`,`{batch:02d}` and `{mape:.2f}`. A mismatch between
            # logged metrics and the path's placeholders can cause formatting to
            # fail.
            if batch is None or "batch" in logs:
                file_path = self.filepath.format(epoch=epoch + 1, **logs)
            else:
                file_path = self.filepath.format(
                    epoch=epoch + 1, batch=batch + 1, **logs
                )
        except KeyError as e:
            raise KeyError(
                f'Failed to format this callback filepath: "{self.filepath}". '
                f"Reason: {e}"
            )
        return file_path


class CustomPrinter(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        netsurf.utils.log._custom('MDL',f'Epoch {epoch} - {logs}')


def parse_callbacks(model, cbacks: dict, pruning_params: dict = {}):
    # Loop thru cbacks
    callbacks = []
    for cback in cbacks:
        # Get kws 
        kws = {}
        if cbacks[cback] is not None:
            kws = cbacks[cback]
        if cback.lower() == 'early_stopping' or cback.lower() == 'early_stopper':
            callbacks.append(netsurf.tf.keras.callbacks.EarlyStopping(**kws))
        elif cback.lower() == 'reduce_lr' or cback.lower() == 'reduce_learning_rate' or cback.lower() == 'reduce_learning_rate_on_plateau' or cback.lower() == 'reducelronplateau':
            callbacks.append(netsurf.tf.keras.callbacks.ReduceLROnPlateau(**kws))
        elif cback.lower() == 'print':
            callbacks.append(CustomPrinter())
        elif cback.lower() == 'checkpoint':
            callbacks.append(CustomModelCheckpoint(**kws))
        else:
            raise ValueError(f'Callback {cback} not recognized.')
        
    if len(pruning_params) > 0:
        callbacks.append(PruningScheduler(model, **pruning_params))
    
    return callbacks


# Helper: recursively get alpha and beta values from model layers.
def get_alpha_beta_values(model):
    total_alpha = []
    total_beta = []
    total_reg_loss = []
    for layer in model.layers:
        # Check if the layer is an instance of your custom alpha layer.
        if isinstance(layer, QQApplyAlpha):
            # Convert to numpy and take the mean (per channel) for each layer.
            total_alpha.append(np.mean(layer.alpha.numpy()))
            total_beta.append(np.mean(layer.beta.numpy()))
            total_reg_loss.append(np.mean(layer.regularization_loss_value.numpy()))
        # If the layer itself contains layers (e.g., nested models), recursively search.
        if hasattr(layer, 'layers') and layer.layers:
            a_vals, b_vals, l_vals = get_alpha_beta_values(layer, QQApplyAlpha)
            total_alpha.extend(a_vals)
            total_beta.extend(b_vals)
            total_reg_loss.extend(l_vals)

    return total_alpha, total_beta, total_reg_loss

class AlphaBetaTracker(tf.keras.callbacks.Callback):
    def __init__(self, log_every=1):
        """
        Args:
            alpha_layer_class: The class of your custom alpha layer (e.g., CustomApplyAlpha).
            log_every: How often (in epochs) to log the averages.
        """
        super().__init__()
        self.log_every = log_every
        self.alpha_history = []
        self.beta_history = []
        self.loss_alpha_reg = []


    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        total_alpha, total_beta, total_reg_loss = get_alpha_beta_values(self.model)
        if total_alpha:
            avg_alpha = np.mean(total_alpha)
            avg_beta = np.mean(total_beta)
            self.alpha_history.append(avg_alpha)
            self.beta_history.append(avg_beta)
            self.loss_alpha_reg.append(np.mean(total_reg_loss))
            # if (epoch + 1) % self.log_every == 0:
            #     print(f"Epoch {epoch+1}: Average alpha = {avg_alpha:.4f}, Average beta = {avg_beta:.4f}")
            # Optionally, add these to the logs dictionary so that they're available as metrics.
            if logs is not None:
                logs['avg_alpha'] = avg_alpha
                logs['avg_beta'] = avg_beta
                logs['loss_alpha_reg'] = np.mean(total_reg_loss)
            

""" Scheduler for pruning """
class PruningScheduler(tf.keras.callbacks.Callback):
    def __init__(self, model, final_sparsity=0.5, begin_epoch=2, step = 1, end_epoch=10):
        super().__init__()
        self.model = model
        self.final_sparsity = final_sparsity
        self.begin_epoch = begin_epoch
        self.end_epoch = end_epoch
        self.step = step

    def on_epoch_begin(self, epoch, logs=None):
        if (self.begin_epoch <= epoch <= self.end_epoch) and ((epoch - self.begin_epoch) % self.step == 0):
            progress = (epoch - self.begin_epoch) / (self.end_epoch - self.begin_epoch)
            current_sparsity = progress * self.final_sparsity
            netsurf.utils.log._custom('MDL', f"Applying pruning at {current_sparsity:.2%} sparsity")
            # Apply pruning only to PrunableLayer subclasses
            for layer in self.model.layers:
                if isinstance(layer, PrunableLayer):
                    layer.update_pruning_mask(current_sparsity)



