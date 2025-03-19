""" Tensorflow """
import tensorflow as tf

""" Scipy """
from scipy import stats

""" Numpy """
import numpy as np

""" Import wsbmr """
import wsbmr

# Litte snippet to parse metrics 
def parse_metrics(metrics):
    """ Get our custom metrics """
    mets = []
    for m in metrics:
        if isinstance(m, str):
            if m.lower() in METRICS:
                mets.append(METRICS[m.lower()])
                wsbmr.utils.log._custom('BMK',f'Adding custom metric {m} with definition {METRICS[m.lower()]}.')
            else:
                if hasattr(tf.keras.metrics, m):
                    mets.append(getattr(tf.keras.metrics, m)())
                    wsbmr.utils.log._custom('BMK',f'Adding standard keras metric {m} with definition {getattr(tf.keras.metrics,m)}.')
                else:
                    wsbmr.utils.log._warn(f'Metric {m} not found. Keeping as string.')
                    mets.append(m)
        elif isinstance(m, tf.keras.metrics.Metric):
            mets.append(m)
    
    return mets


@tf.function(reduce_retracing=True)
def calculate_emd(y_true, y_pred):
    _emd = np.mean(stats.wasserstein_distance(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1), u_weights=None, v_weights=None))
    return _emd.astype(np.float32)


class SerializableCustomMetric(tf.keras.metrics.Metric):
    def __init__(self, name='custom_metric', **kwargs):
        super().__init__(name=name, **kwargs)

        self.counter = self.add_weight(name='counter', shape=(), initializer='zeros')
        self.custom_metric = self.add_weight(name='custom_metric', shape=(), initializer='zeros')
    
    def __getstate__(self):
        variables = {v.name: v.numpy() for v in self.variables}
        state = {
            name: variables[var.name]
            for name, var in self._unconditional_dependency_names.items()
            if isinstance(var, tf.Variable)}
        state['name'] = self.name
        #state['num_classes'] = self.num_classes
        return state

    def __setstate__(self, state):
        self.__init__(name=state.pop('name'))
        for name, value in state.items():
            self._unconditional_dependency_names[name].assign(value)

""" Earth Mover's Distance Metric """
class EMDMetric(SerializableCustomMetric):
    def __init__(self, name='emd', **kwargs):
        super().__init__(name=name, **kwargs)
        self.counter = self.add_weight(name='counter', shape=(), initializer='zeros')
        self.emd = self.add_weight(name='emd', shape=(), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        _emd2 = tf.numpy_function(calculate_emd, [y_true, y_pred], tf.float32)
        self.emd.assign_add(_emd2)
        self.counter.assign_add(1)
    
    def result(self):
        return self.emd/self.counter
    
    def reset_state(self):
        self.counter.assign(0)
        # Reset the metric state at the start of each epoch.
        self.emd.assign(0.0)
    

""" Wasserstein Distance Metric """
class WDMetric(SerializableCustomMetric):
    def __init__(self, name='wd', **kwargs):
        super().__init__(name=name, **kwargs)
        self.counter = self.add_weight(name='counter', shape=(), initializer='zeros')
        self.wd = self.add_weight(name='wd', shape=(), initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        _wd = tf.reduce_mean(y_true*y_pred)
        self.wd.assign_add(_wd)
        self.counter.assign_add(1)
    
    def result(self):
        return self.wd/self.counter
    
    def reset_state(self):
        self.counter.assign(0)
        # Reset the metric state at the start of each epoch.
        self.wd.assign(0.0)
    
    

""" KL Divergence Metric """
class KLDivergenceMetric(SerializableCustomMetric):
    def __init__(self, name='kl_divergence', **kwargs):
        super().__init__(name=name, **kwargs)

        self.counter = self.add_weight(name='counter', shape=(), initializer='zeros')
        self.kl_divergence = self.add_weight(name='kl_divergence', shape=(), initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        
        # [@manuelbv]
        # y_true has zeros, so we have two options here: 
        # 1. Use tf.math.log(y_true/y_pred + 1e-8) to avoid log(0)
        # 2. add up all the values in y_true over axis = 0, and use that as our distribution

        y_true = tf.reduce_mean(y_true, axis=0)
        y_pred = tf.reduce_mean(y_pred, axis=0)

        _kl_divergence = tf.reduce_mean(y_true*tf.math.log(y_true/y_pred + 1e-8))

        # Average out 
        self.kl_divergence.assign_add(_kl_divergence)
        self.counter.assign_add(1)
    
    def result(self):
        return self.kl_divergence/self.counter
    
    def reset_state(self):
        self.counter.assign(0)
        # Reset the metric state at the start of each epoch.
        self.kl_divergence.assign(0.0)

""" Pearson correlation coefficient """
class PearsonCorrelation(tf.keras.metrics.Metric):
    def __init__(self, name='pearsoncorrelation', **kwargs):
        super(PearsonCorrelation, self).__init__(name=name, **kwargs)
        self.corr = self.add_weight(name='corr', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        
        n = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
        
        sum_x = tf.reduce_sum(y_true)
        sum_y = tf.reduce_sum(y_pred)
        sum_x_squared = tf.reduce_sum(tf.square(y_true))
        sum_y_squared = tf.reduce_sum(tf.square(y_pred))
        sum_xy = tf.reduce_sum(tf.multiply(y_true, y_pred))
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = tf.sqrt((n * sum_x_squared - tf.square(sum_x)) * (n * sum_y_squared - tf.square(sum_y)))
        
        result = numerator / denominator

        self.corr.assign(result)

    def result(self):
        return self.corr

    def reset_state(self):
        self.corr.assign(0.0)



""" R2 Score Metric """
class R2Score(tf.keras.metrics.Metric):
    def __init__(self, name='r2score', **kwargs):
        super(R2Score, self).__init__(name=name, **kwargs)
        self.r2 = self.add_weight(name='r2', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        mean_y_true = tf.reduce_mean(y_true)
        ss_tot = tf.reduce_sum(tf.square(y_true - mean_y_true))
        ss_res = tf.reduce_sum(tf.square(y_true - y_pred))

        result = 1 - ss_res / ss_tot

        self.r2.assign(result)
    
    def result(self):
        return self.r2
    
    def reset_state(self):
        self.r2.assign(0.0)
    

METRICS = {'emd': EMDMetric(), 
                    'wd': WDMetric(), 
                    'kl_div': KLDivergenceMetric(),
                    'categorical_accuracy': tf.keras.metrics.CategoricalAccuracy(),
                    'sparse_categorical_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
                    'accuracy': tf.keras.metrics.Accuracy(),
                    'mse': tf.keras.metrics.MeanSquaredError(),
                    'mean_squared_error': tf.keras.metrics.MeanSquaredError(),
                    'mean_square_error': tf.keras.metrics.MeanSquaredError(),
                    'mean_absolute_error': tf.keras.metrics.MeanAbsoluteError(),
                    'mae': tf.keras.metrics.MeanAbsoluteError(),
                    'r2score': R2Score(),
                    'pearsoncorrelation': PearsonCorrelation()}