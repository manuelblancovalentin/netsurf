
""" Basic modules """
import os, sys
from time import time
import datetime
import copy

""" Json """
import json

""" Numpy """
import numpy as np

""" Keras """
#import keras 
from tensorflow import keras

""" import pergamos """
import pergamos as pg

""" Pandas """
import pandas as pd

""" Matplotlib """
import matplotlib.pyplot as plt

""" netsurf """
from . import datasets

""" Custom utils """
import netsurf

""" Tensorflow (for pruning) """
import tensorflow as  tf 
#import tensorflow_model_optimization as tfmot
#from tensorflow_model_optimization.sparsity import keras as sparsity

""" Quantized models printing summary """
from qkeras.autoqkeras.utils import print_qmodel_summary

""" Load latest session method """
def load_session(session_dir, quantizer = None, session_file = None, latest = True):
    if latest and session_file is None:
        latest_pkl = netsurf.utils.find_latest_file(os.path.join(session_dir, 'training_session'), '*/training_session.*.pkl')
        if latest_pkl == '' or latest_pkl is None:
            netsurf.utils.log._warn(f'No session files found at {session_dir}')
            return None
        netsurf.utils.log._custom('BMK',f'Loading session from file {latest_pkl}')
        session_file = latest_pkl
    
    # Custom objects 
    custom_objects = netsurf.dnn.models.get_custom_objects(quantizer, wrap = False) if quantizer is not None else {}
    sess = netsurf.utils.load_object(session_file, custom_objects = custom_objects)
    return sess


""" Session class definition """
class Session:
    def __init__(self, session_type, session_name, session_date, session_path = None):
        self.name = session_name
        self.date = session_date
        self.type = session_type
        self.path = session_path
        self.logs = None
        self.config = {'batch_size': None, 'epochs': None, 'optimizer': None, 'loss': None, 'metrics': None, 'lr': None}

    def add_to_config(self, key, value):
        if key not in self.config:
            self.config[key] = None
        self.config[key] = value
    
    """ Function to plot the training history """
    def plot_training_history(self, logs = None, ylog = False, filename = None, to_file = False, show = True):
        if logs is None:
            logs = self.logs
        if to_file and filename is None:
            os.makedirs(self.path, exist_ok = True)
            filename = os.path.join(self.path, f'{self.name}_training_history.png')
        return netsurf.utils.plot.plot_training_history(logs, ylog = ylog, filename = filename, show = show)

    """ Save config and object to file """
    def save(self, filepath = None):
        """ Save the session object to file """
        if filepath is None:
            filepath = self.path
        filename = os.path.join(filepath, f'{self.name}.pkl')
        # Make sure directory exists 
        os.makedirs(filepath, exist_ok = True)
        try:
            netsurf.utils.save_object(self, filename)
            netsurf.utils.log._custom('BMK',f'Session saved to file {filename}')
        except Exception as e:
            netsurf.utils.log._error(f'Error saving session to file {filename}. Exception: {e}')
            pass

        """ Save the config to json file """
        config_filename = os.path.join(filepath, f'config_{self.name}.json')
        # dict to json
        d = self.config
        d['lr'] = str(d['lr'])
        try:
            with open(config_filename, 'w') as fp:
                json.dump(d, fp, indent=2)
            netsurf.utils.log._custom('BMK',f'Config saved to file {config_filename}')
        except:
            pass
    
    def html(self):
        # Create collapsible container for this dataset
        session_ct = pg.CollapsibleContainer(f"⏰ {self.name}", layout='vertical')

        # Create a container showing the basic information summary for this summary 
        summary_ct = pg.Container("Summary", layout='vertical')

        sum = {kv: ", ".join(list(map(str,self.config[kv]))) if isinstance(self.config[kv],list) else str(self.config[kv]) \
               for kv in self.config}
        sum = dict(**{kv: getattr(self, kv) for kv in ['name', 'path', 'date', 'type']},
                   **sum)
        # Create pandas dataframe 
        df = pd.DataFrame([sum]).T
        
        # Add to container
        summary_ct.append(pg.Table.from_data(df))
        
        # Add summary container to session container
        session_ct.append(summary_ct)

        filename = os.path.join(self.path, f'{self.name}_training_history.png')
        if not os.path.isfile(filename):
            self.plot_training_history(to_file = True, show = False)
        if os.path.isfile(filename):
            # Create collapsible container for the training history
            training_ct = pg.CollapsibleContainer("Training history", layout='vertical')
            training_ct.append(pg.Image(filename, embed=True))
            # Add to session container
            session_ct.append(training_ct)

        # Evaluation
        # Now evaluation
        for t in ['ROC','confusion_matrix', 'scatter', 'evaluation']:
            fpath = os.path.join(self.path, f'{t}.png')
            if os.path.isfile(fpath):
                eval_ct = pg.CollapsibleContainer(t.capitalize(), layout='vertical')
                eval_ct.append(pg.Image(fpath, embed=True))
                session_ct.append(eval_ct)
                

        return session_ct



""" Benchmark class definition """
class Benchmark:
    def __init__(self, name: str, dataset: str, model: str, quantization: 'QuantizationScheme', 
                 optimizer = None, loss = None, metrics = None, datasets_dir = '.', benchmarks_dir = '.', 
                 model_prefix = '', verbose = True, type = None,
                **kwargs):
        
        # Set default vars
        self.name = name
        self.dataset_name = dataset
        self.quantization = quantization 
        
        # Save kwargs for later 
        self.kwargs = kwargs
        # Optional 
        self.model_prefix = model_prefix
        self.model_name = model
        self.model_class = str(model)
        self._dirs_built = False

        # dirs
        self.benchmarks_dir = benchmarks_dir
        self.datasets_dir = datasets_dir
        self.verbose = verbose

        # At this point we don't really need the dataset, so let's just use the DATASETS_CONFIG to get the 
        # in/out shapes 
        dset_config = netsurf.config.DATASETS_CONFIG[dataset] if dataset in netsurf.config.DATASETS_CONFIG else None
        in_shape = dset_config['in_shape'][1:] if dset_config is not None else None
        out_shape = dset_config['out_shape'][1:] if dset_config is not None else None
        self.in_shape = in_shape
        self.out_shape = out_shape

        # mets = []
        # if 'ECON' not in name:
        #     mets += [netsurf.metrics.EMDMetric(), netsurf.metrics.WDMetric(), netsurf.metrics.KLDivergenceMetric()]


        # Keep original loss and metric names for later serialization (into pkl)
        if isinstance(loss, str):
            loss = loss.replace('mse','mean_squared_error').replace('mae','mean_absolute_error')
        self.loss_names = loss
        self.metric_names = metrics

        # Parse loss 
        self.loss_name = loss
        self.loss = netsurf.losses.parse_loss(loss)
        self.metric_names = metrics
        self.metrics = netsurf.dnn.metrics.parse_metrics(metrics)
        self.optimizer = optimizer

        # Parse the problem type (classification/rgression/unsupervised) if None
        if type is not None:
            if type not in ['classification', 'regression', 'unsupervised']:
                raise ValueError(f'Unknown type {type}. Should be one of [classification, regression, unsupervised]')
                type = None
        if type is None:
            if loss in ['categorical_crossentropy', 'sparse_categorical_crossentropy', 'binary_crossentropy']:
                self.type = 'classification'
            elif model in ['ae', 'econtae']:
                self.type = 'unsupervised'
            elif loss in ['mean_squared_error', 'mean_absolute_error', 'mse', 'mae']:
                self.type = 'regression'
            else:
                self.type = 'unknown'
        
        self.type = type

        """ Now get the model """
        self.init_tasks(self.quantization, **kwargs)

        
        # else:
        #     # Save initial weights to file
        #     self.save_weights(models_dir, model_name)
        
        # Compile the model
        #self.compile()
    
    def __repr__(self):
        return f'📦 {self.name} <netsurf.{self.__class__.__name__}> @ ({hex(id(self))})\n   - Model: {self.model_name}\n   - Dataset: {self.dataset_name}'


    """ Html element for pergamos """
    def html(self):

        # Create collapsible container for this benchmark
        bmk_ct = pg.CollapsibleContainer(f"📦 {self.name} (netsurf.{self.__class__.__name__})", layout='vertical')

        # Get model html
        model_ct = self.model.html()
        
        
        # Add last weight overview plots (pruning, pie, etc.)
        # Plot weight dist 
        filename = os.path.join(self.models_dir, 'weights_pie.png')
        if not os.path.exists(filename):
            self.plot_weight_pie(to_file = True, show = False)
        if os.path.exists(filename):
            ct = pg.CollapsibleContainer('🍕 Weights distribution', layout='vertical')
            ct.append(pg.Image(filename))
            model_ct[0].append(ct)

        # Plot sparsity
        filename = netsurf.utils.io.create_temporary_file(prefix='netsurf_sparsity', ext = '.png')
        netsurf.utils.plot.plot_sparsity(self.model, filepath = filename, show = False, separated = False, verbose = True)
        if os.path.exists(filename):
            ct = pg.CollapsibleContainer('🎨 Sparsity', layout='vertical')
            ct.append(pg.Image(filename, embed = True))
            model_ct[0].append(ct)
        # Delete temp file
        os.remove(filename)
        
        # Now separated
        filename = netsurf.utils.io.create_temporary_file(prefix='netsurf_sparsity_separated', ext = '.png')
        netsurf.utils.plot.plot_sparsity(self.model, filepath = filename, show = False, separated = True, verbose = True)
        if os.path.exists(filename):
            ct = pg.CollapsibleContainer('🎨 Sparsity (separated)', layout='vertical')
            ct.append(pg.Image(filename, embed = True))
            model_ct[0].append(ct)
        # Delete temp file
        os.remove(filename)

        # Add model html to container
        bmk_ct.extend(model_ct)

        # Add to benchmark container
        bmk_ct.append(self.dataset.html())

        ### Forward pass thru activations model. This will be a whole new container. 
        activations_ct = pg.CollapsibleContainer('🔥 Forward pass activations', layout='vertical')
         # Add to benchmark container
        bmk_ct.append(activations_ct)

        """ Add model container to doc report """
        # Build activation model to get the output at each single layer 
        activation_model = netsurf.QModel(self.quantization, self.model.in_shape, self.model.out_shape, 
                                        ip = self.model.input, out = self.model._activations)

        # Get xdata 
        xsample, ysample = self.get_dataset_sample(subset = 'validation', nsamples = 1000)

        # Plot the histogram of the activations at the output of each one of the layers
        fig, axs = netsurf.utils.plot.plot_histogram_activations(activation_model, X = xsample, 
                                                               show = False, sharex = False)

        # Create an image for the plot
        img = pg.Image(fig, embed=True)
        
        # Add to activations_ct
        activations_ct.append(img)
        # Close figure
        plt.close(fig)

        # Let's profile the model (similarly to what hls4ml does with the boxplot showing the limits)
        profile_ct = pg.CollapsibleContainer('⧯ Profiling', layout='vertical')
         # Add to benchmark container
        bmk_ct.append(profile_ct)

        fig_profile, ax_profile = netsurf.utils.plot.plot_model_profile(activation_model, X = xsample, show = False, sharex = False)
        # Create an image for the plot
        img_profile = pg.Image(fig_profile, embed=True)
        # Add to profile_ct
        profile_ct.append(img_profile)



        return bmk_ct

    def init_tasks(self, quantization, build_dirs = True, load_weights = True, pruning = None, **kwargs):
        
        # if 'model_params' in kwargs:
        model_params = kwargs.pop('model_params') if 'model_params' in kwargs else {}

        self.model = self.get_model(quantization, self.model_class, in_shape = self.in_shape, out_shape = self.out_shape, 
                                    optimizer = self.optimizer, loss = self.loss, metrics = self.metrics, 
                                    type = self.type, **model_params, **kwargs)
        
        
        # Now set the model name
        self.model_name = self.model.create_model_name_by_architecture()
        # Extract pruning factor from model name 
        if pruning is not None:
            if not isinstance(pruning, float):
                pruning = None
                netsurf.err('Pruning factor must be a float between 0.0 and 1.0, setting it to None')
            elif pruning < 0.0 or pruning > 1.0:
                pruning = None
                netsurf.err('Pruning factor must be a float between 0.0 and 1.0, setting it to None')
        
        self.pruning_factor, _ =  netsurf.utils.get_pruning_factor(self.model_name) if pruning is None else (pruning,None)
        self.total_num_params = self.model.count_trainable_parameters() - self.model.count_pruned_parameters()
        
        # if pruning_factor > 0.0 add to model full name
        if self.pruning_factor > 0.0:
            self.model_prefix += f'pruned_{self.pruning_factor}_'

        self.model_full_name = self.model_prefix + self.model_name

        # Let's set the model path 
        benchmark_dir = os.path.join(self.benchmarks_dir, self.name)
        quant_dir = os.path.join(benchmark_dir, f'{quantization._scheme_str.no_special_chars()}')
        model_dir = os.path.join(quant_dir, self.model_full_name)
        models_dir = os.path.join(model_dir, 'models')
        sessions_dir = os.path.join(model_dir, 'sessions')
        experiments_dir = os.path.join(model_dir, 'experiments')
        
        # Save keras model to file 
        model_path = os.path.join(models_dir, f'{self.model_full_name}.keras')
        #self.model.save(model_path)
        #netsurf.utils.log._custom('BMK',f'Model structure saved to file {model_path}')
        
        # Now set paths in structure 
        self.benchmark_dir = benchmark_dir
        self.quant_dir = quant_dir
        self.model_dir = model_dir
        self.models_dir = models_dir
        self.model_path = model_path
        self.sessions_dir = sessions_dir
        self.experiments_dir = experiments_dir

        if build_dirs:
            self.build_dirs()
            
        # Init sessions
        self.sessions = []

        # Let's try to load the weights (if they exist)
        if load_weights:
            self.load_weights(verbose = self.verbose)
        
        self._build_dirs = build_dirs
        self._load_weights = load_weights

    def build_dirs(self):
        # We need to build the directories level by level and initialize the metadata for each level. 
        # level 0: benchmarks_dir
        # level 1: benchmark 
        # level 2: quantization
        # level 3: model
        # level 4: method (not here yet)

        # Metadata for level 1
        bdir = self.benchmark_dir 
        os.makedirs(bdir, exist_ok = True)
        metadata0 = {'level': 'benchmark', 
                        'name': self.name, 
                        'creation_date': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                        'creation_user': os.getlogin(),
                        'creation_host': os.uname().nodename,
                        'config': netsurf.config.BENCHMARKS_CONFIG[self.name]}
        # metadata filepath 
        metadata0_filepath = os.path.join(bdir, '.metadata.netsurf')
        # Save metadata
        if not os.path.isfile(metadata0_filepath):
            netsurf.utils.log._custom('BMK',f'Saving benchmark metadata to file {metadata0_filepath}')
            with open(metadata0_filepath, 'w') as f:
                json.dump(metadata0, f, indent=2)


        # Metadata for level 2
        qdir = os.path.join(bdir, self.quantization._scheme_str.no_special_chars())
        os.makedirs(qdir, exist_ok = True)
        metadata1 = {'level': 'quantization', 
                    'name': self.quantization._scheme_str, 
                    'creation_date': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'creation_user': os.getlogin(),
                    'creation_host': os.uname().nodename,
                    'config': {'quantization': self.quantization._scheme_str}}
        # metadata filepath 
        metadata1_filepath = os.path.join(qdir, '.metadata.netsurf')
        if not os.path.isfile(metadata1_filepath):
            # Save metadata
            netsurf.utils.log._custom('BMK',f'Saving quantization metadata to file {metadata1_filepath}')
            with open(metadata1_filepath, 'w') as f:
                json.dump(metadata1, f, indent=2)
        
        # Metadata for level 3
        mdir = os.path.join(qdir, self.model_full_name)
        os.makedirs(mdir, exist_ok = True)
        metadata2 = {'level': 'model', 
                    'name': self.model_full_name, 
                    'creation_date': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'creation_user': os.getlogin(),
                    'creation_host': os.uname().nodename,
                    'config': {'model_name': self.model_name, 
                               'model_class': self.model_class,
                               'model_full_name': self.model_full_name, 
                               'model_prefix': self.model_prefix, 
                               'total_num_params': int(self.total_num_params),
                               'pruning_factor': self.pruning_factor,
                                'optimizer': str(self.optimizer), 'loss': self.loss_names, 'metrics': self.metric_names,
                                'in_shape': self.in_shape, 'out_shape': self.out_shape, 'dataset': self.dataset_name}}
        if 'sessions' in self.kwargs:
            if len(self.kwargs['sessions']) > 0:
                if 'optimizer_params' in self.kwargs['sessions'][0]:
                    metadata2['config']['optimizer_params'] = self.kwargs['sessions'][0]['optimizer_params']
        # metadata filepath 
        metadata2_filepath = os.path.join(mdir, '.metadata.netsurf')
        if not os.path.isfile(metadata2_filepath):
            # Save metadata
            netsurf.utils.log._custom('BMK',f'Saving model metadata to file {metadata2_filepath}')
            with open(metadata2_filepath, 'w') as f:
                json.dump(metadata2, f, indent=2)

        os.makedirs(self.benchmark_dir, exist_ok = True)
        os.makedirs(self.models_dir, exist_ok = True)
        os.makedirs(self.sessions_dir, exist_ok = True)
        os.makedirs(self.experiments_dir, exist_ok = True)
        self._dirs_built = True

    # save to file 
    def save(self, overwrite = False):
        filepath = os.path.join(self.model_dir, f'{self.name}.netsurf.bmk')
        if os.path.isfile(filepath) and not overwrite:
            netsurf.utils.log._warn(f'Benchmark file {filepath} already exists. Skipping. If you want to overwrite, set overwrite = True')
            return
        msg = netsurf.utils.save_object(self, filepath, meta_attributes = {'class': 'netsurf.Benchmark', 'benchmark': self.name, 
                                                                         'model': self.model_name, 'pruning': self.pruning_factor, 
                                                                         'model_full_name': self.model_full_name})
        netsurf.utils.log._custom('BMK',msg)

    def assert_dirs_built(self):
        if not self._dirs_built:
            self.build_dirs()

    def assert_dataset_is_loaded(self):
        self.assert_dirs_built()
        if not hasattr(self, 'dataset'):
            # Dataset not loaded, load now
            # Get dataset and model
            self.dataset = self.get_dataset(self.dataset_name, self.quantization, datasets_dir = self.datasets_dir, **self.kwargs)
            
            # Get shapes 
            if isinstance(self.dataset['train'], tuple):
                din = self.dataset['train'][0]
                dout = self.dataset['train'][1]
                if hasattr(self.model, 'preprocess_input'):
                    din = self.model.preprocess_input(din)
                if hasattr(self.model, 'preprocess_output'):
                    dout = self.model.preprocess_output(dout)
                in_shape = din.shape[1:]
                out_shape = dout.shape[1:]
            elif isinstance(self.dataset['train'], keras.preprocessing.image.DirectoryIterator):
                in_shape = self.dataset['train'].image_shape
                out_shape = (self.dataset['train'].num_classes,)
            elif isinstance(self.dataset['train'], tf.data.Dataset):
                in_shape = self.dataset['train'].element_spec[0].shape[1:]
                out_shape = self.dataset['train'].element_spec[1].shape[1:]
            else:
                raise ValueError('Unknown dataset type')

            
            # if either in our out_shpe has zero dims, at least add 1 (cause the output of the model will be 1)
            in_shape = np.atleast_2d(in_shape)
            out_shape = np.atleast_2d(out_shape)
            
            # If in_shape and out_shape have changed, rebuild the model 
            if (self.in_shape != in_shape).any() or (self.out_shape != out_shape).any():
                self.in_shape = in_shape
                self.out_shape = out_shape
                
                self._load_weights = False
                self._build_dirs = False
                if 'load_weights' in self.kwargs:
                    self._load_weights = self.kwargs.pop('load_weights')
                if 'verbose' in self.kwargs:
                    self.verbose = self.kwargs.pop('verbose')
                if 'build_dirs' in self.kwargs:
                    self._build_dirs = self.kwargs.pop('build_dirs')

                # Add a pretty big warning here!
                netsurf.utils.log._warn(f'Input/output shapes have changed. Rebuilding model with new shapes {in_shape} -> {out_shape}, note that this will override the weights!')
                self.init_tasks(self.quantization, build_dirs = self._build_dirs, load_weights = self._load_weights, verbose = self.verbose, **self.kwargs)

    """ Simple method to load weights """
    def load_weights(self, path = None, verbose = True):
        self.assert_dirs_built()
        # If path is None, find latest weights 
        if path is None:
            path = self.models_dir
            if verbose: netsurf.utils.log._custom('BMK',f'No path provided, looking for latest h5 file at default model path: {path}')
            if False:
                last_file = netsurf.utils.find_latest_file(path + '/' + self.model_full_name, '.weights.h5')
            else:
                #last_file = os.path.join(path,self.model_name + '.keras')
                last_file = os.path.join(path,self.model_full_name + '.keras.latest')
                if verbose: netsurf.utils.log._custom('BMK',f'Looking for file {last_file}')
            if last_file == '' or last_file is None:
                if verbose: netsurf.utils.log._warn(f'No weights files found at {path}. Skipping.')
                return
            # We found the file, let's try to load it
            path = last_file
        if os.path.isfile(path):
            try:
                self.model.load_weights(path)
                if verbose:
                    netsurf.utils.log._custom('BMK',f'Weights successfully loaded from file {path}')
            except Exception as e:
                if verbose: print(e)
                if verbose: netsurf.utils.log._error(f'Error loading weights from file {path}')
        else:
            if verbose: netsurf.utils.log._warn(f'Weights file {path} not found')
        
        # Now, if by any chance we have a model that has a "norm" property, this means that
        # we have to normalize the data. Thus, it means we have to make sure that the dataset 
        # is loaded.
        if hasattr(self.model, 'normalize'):
            self.assert_dataset_is_loaded()
            if verbose: netsurf.utils.log._custom('BMK',f'Applying norm.adapt to training data...')
            self.model.normalize(np.array(self.dataset.dataset['train'][0]))


    """ Simple method to save weights """
    def save_weights(self, path = None, prefix = None, verbose = True):
        self.assert_dirs_built()
        if path is None:
            path = self.models_dir
        
        if prefix is None:
            prefix = self.model_full_name
        
        # Generate filename for output file 
        filename = netsurf.utils.generate_filename(prefix, 'weights.h5')
        model_name = netsurf.utils.generate_filename(prefix, 'keras')
        filepath = os.path.join(path, filename)
        model_path = os.path.join(path, model_name)
        try:
            self.model.save(filepath, save_format = 'h5')
            if verbose: netsurf.utils.log._custom('BMK',f'Weights saved to h5 file {filepath}')
            self.model.save(model_path, save_format = 'tf')
            if verbose: netsurf.utils.log._custom('BMK',f'Model saved to keras file {model_path}')
        except:
            if verbose: netsurf.utils.log._error(f'Error saving weights to file {path}')
            return
        
        # Create a link to the latest weights
        latest_weights = os.path.join(path, f'{prefix}.weights.h5.latest')
        latest_model = os.path.join(path, f'{prefix}.keras.latest')
        if os.path.exists(latest_weights):
            os.remove(latest_weights)
        os.symlink(filepath, latest_weights)

        #[@max]
        if os.path.exists(latest_model):
            os.remove(latest_model)
        os.symlink(model_path, latest_model)
        if verbose: netsurf.utils.log._custom('BMK',f'Created a symlink to {latest_weights}')

    # Get dataset function
    def get_dataset(self, dataset, quantizer, *args, **kwargs):
        return datasets.load(dataset, quantizer, *args, **kwargs)
    
    # Get model function
    def get_model(self, quantization, model, *args, **kwargs):
        return netsurf.dnn.models.get_model(quantization, model, *args, **kwargs)
    
    def model_summary(self, to_file = False, quantized = False):
        self.assert_dirs_built()
        if not to_file:
            print(self.model.summary())
        else:
            summary_filename = os.path.join(self.models_dir, 'summary' if not quantized else 'summary_quantized')
            with open(summary_filename, 'w') as f:
                if not quantized:
                    self.model.summary(print_fn=lambda x: f.write(x + '\n'))
                else:
                    # [@manuelbv]: THIS IS EXTREMELY IMPORTANT!!!!!! EARLIER, I WAS REASSIGNING sys.stdout 
                    #   TO sys.__stdout__ WHICH WAS MESSING UP THE WHOLE IPYTHON ENVIRONMENT, AND I WOULDN'T
                    #   BE ABLE TO DEBUG ANYTHING AFTERWARDS. WE COULDNT' DEBUG ANYTHING AFTER THIS LINE. 
                    #   WE WOULD JUST GET A NOTIFICATION FROM VSCODE SAYING WE NEEDED IPYKERNEL SETUP WAS 
                    #   REQUIRED. SO THE TRICK IS TO STORE THE ORIGINAL sys.stdout AND THEN REASSIGN IT BACK!!!!!!!
                    # Store the original stdout
                    original_stdout = sys.stdout
                    sys.stdout = f  # Redirect stdout to the file
                    print_qmodel_summary(self.model)
                    # Reset the stdout
                    sys.stdout = original_stdout

    def plot_model(self, *args, filename = None, **kwargs):
        self.assert_dirs_built()
        if filename is None:
            filename = os.path.join(self.models_dir, 'architecture.png')
        return self.model.plot_model(*args, filename = filename, **kwargs)
    
    def dataset_summary(self):
        self.assert_dataset_is_loaded()
        self.dataset.display_data_stats()

    def plot_dataset(self, *args, num_samples = 12, filename = None, to_file = True, **kwargs):
        self.assert_dataset_is_loaded()
        filename1, filename2 = None, None
        
        if to_file:
            if filename is None:
                filename1 = os.path.join(self.benchmark_dir, 'dataset.png')
                filename2 = os.path.join(self.benchmark_dir, 'classes_distribution.png')
            else:
                if '.png' in filename:
                    filename1 = filename
                    filename2 = filename.replace('.png', '_classes_distribution.png')
                else:
                    filename1 = filename + '.png'
                    filename2 = filename + '_classes_distribution.png'
        # Images
        self.dataset.display_data(*args, num_samples = num_samples, filename = filename1, **kwargs)

        # Classes dist 
        self.dataset.display_classes_distribution(*args, filename = filename2, **kwargs)

    """ Plot weight pie """
    def plot_weight_pie(self, to_file = False, **kwargs):
        self.assert_dirs_built()
        filename = None
        if to_file:
            filename = os.path.join(self.models_dir, 'weights_pie.png')
        
        netsurf.utils.plot.plot_model_weights_pie(self.model, filepath = filename, **kwargs)

    """ Perform prunning """
    # def prune_model(self, batch_size = 32, final_sparsity = 0.5, step = 2, end_epoch = 10):
    #     self.assert_dataset_is_loaded()
    #     # Get train size from dataset 
    #     if isinstance(self.dataset.dataset['train'], tuple):
    #         train_size = self.dataset.dataset['train'][0].shape[0]
    #     elif isinstance(self.dataset.dataset['train'], keras.preprocessing.image.DirectoryIterator):
    #         train_size = self.dataset.dataset['train'].n
    #     elif isinstance(self.dataset.dataset['train'], tf.data.Dataset):
    #         train_size = self.dataset.dataset['train'].reduce(0, lambda x, _: x + 1).numpy()

    #     NSTEPS = int(train_size) // batch_size  # 90% train, 10% validation in 10-fold cross validation
        
    #     msg = f'[INFO] - Pruning conv/dense layers gradually, from 0% to {100*final_sparsity:3.2f}% every {step} epochs, ' + \
    #             f'ending by epoch number {end_epoch}; with batch_size {batch_size}, for a total of {NSTEPS} steps per epoch'
    #     print(msg)

    #     # Prune all convolutional and dense layers gradually from 0 to 50% sparsity every 2 epochs,
    #     # ending by the 10th epoch
    #     def pruneFunction(layer):
    #         pruning_params = {
    #             'pruning_schedule': sparsity.PolynomialDecay(
    #                 initial_sparsity=0.0, final_sparsity=final_sparsity, begin_step=NSTEPS * step, end_step=NSTEPS * end_epoch, 
    #                 frequency=NSTEPS
    #             )
    #         }
    #         if isinstance(layer, tf.keras.layers.Conv2D):
    #             return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
    #         if isinstance(layer, tf.keras.layers.Dense) and layer.name != 'output_dense':
    #             return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
    #         return layer

    #     # Perform actual cloning of the model for the pruning
    #     qmodel_pruned = tf.keras.models.clone_model(self.model, clone_function=pruneFunction)

    #     # Set the model back into place 
    #     raise ValueError('Not working yet')
    #     self.model.model = qmodel_pruned

    """ Function to compile model """
    def compile(self, *args, batch_size = 32, opt_params = {}, **kwargs):
        
        # If pruning_sparsity is not 0.0, then we prune the model
        # pruning_sparsity = pruning_params['final_sparsity'] if 'final_sparsity' in pruning_params else 0.0
        # if pruning_sparsity > 0.0:
        #     # Prune the model
        #     self.prune_model(batch_size = batch_size, **pruning_params)

        netsurf.utils.log._custom('MDL',f'Compiling model with parameters {", ".join([f"{k}={v}" for k, v in opt_params.items()])}')
        # Get optimizer first dynamically
        #opt = keras.optimizers.get(self.optimizer, **opt_params)
        # [@manuelbv]: NOTE: ORIGINALLY, I USED keras.optimizers.get BUT THE THING IS THAT 
        # IT DOES NOT ALLOW TO PASS PARAMETERS TO THE OPTIMIZER, SO I DECIDED TO USE THIS OTHER
        # APPROACH, JUST BY POINTING DIRECTLY TO THE OPTIMIZERS OBJECTS. 
        # I opened a github issue about this: https://github.com/keras-team/keras/issues/20251

        # so here's our dirty alternative:

        opt_options = {
            "adam": tf.keras.optimizers.Adam,
            "adagrad": tf.keras.optimizers.Adagrad,
            "ftrl": tf.keras.optimizers.Ftrl,
            "rmsprop": tf.keras.optimizers.RMSprop,
            "sgd": tf.keras.optimizers.SGD,
        }

        assert self.optimizer.lower() in opt_options, f'Optimizer {self.optimizer} not found. Available options are: {", ".join(list(opt_options.keys()))}'

        # Make sure opt_params are correctly parsed as numbers (if required)
        for k, v in opt_params.items():
            if isinstance(v, str):
                try:
                    opt_params[k] = eval(v)
                    netsurf.utils.log._custom('MDL',f'Parameter {k} parsed as number with value {opt_params[k]}')
                except:
                    netsurf.utils.log._warn(f'Parameter {k} COULD NOT BE parsed as number. Original value: {v}')
                    pass

        opt = opt_options[self.optimizer.lower()](**opt_params)

        return self.model.compile(*args, optimizer = opt, loss = self.loss, metrics = self.metrics)

    """ Function to train the model """
    def fit(self, epochs, batch_size, *args, callbacks = [], prune = False, pruning_params = {'final_sparsity': 0.5, 'step': 2, 'end_epoch': 10}, save_weights_checkpoint = True, 
            verbose = False, **kwargs):
        self.assert_dataset_is_loaded()
        netsurf.utils.log._custom('MDL',f'Fitting model with {epochs} epochs and batch_size {batch_size}')

        # Get fit session timestamp 
        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f'training_session.{current_datetime}'
        # Get session path 
        session_path = os.path.join(self.sessions_dir, session_name)

        # Initialize a session object 
        sess = Session('training', session_name, current_datetime, session_path = session_path)

        # If activated, save the model every epoch
        if False:

            # Generate filename for output file 
            filename = netsurf.utils.generate_filename(self.model_full_name, 'epoch_{epoch}.weights.h5')
            filepath = os.path.join(self.models_dir, filename)

            """ Create a custom callback to save weights every epoch,
                the only reason it has to be custom is because we want a
                different printer function
            """
            checkpoint_callback = netsurf.dnn.callbacks.CustomModelCheckpoint(
                filepath=filepath,  # Filepath pattern with epoch number
                save_weights_only=True,  # Save only weights, not the whole model
                verbose=1  # Print a message when saving weights
            )

            # Add to callbacks
            callbacks += [checkpoint_callback]

        # Add AlphaBetaTracker callback
        alpha_tracker = netsurf.dnn.callbacks.AlphaBetaTracker()
        callbacks += [alpha_tracker]

        
        """ Fit the model """
        start = time()

        # Get the data
        if isinstance(self.dataset['train'], tuple):
            XTrain, YTrain = self.dataset['train']
            XTest, YTest = self.dataset['validation']

            # Make sure the data is a multiple of 48 if this benchmark is ECON
            if 'ECON' in self.name:
                n = 48
                XTrain = XTrain[:-(XTrain.shape[0] % n)]
                YTrain = YTrain[:-(YTrain.shape[0] % n)]
                XTest = XTest[:-(XTest.shape[0] % n)]
                YTest = YTest[:-(YTest.shape[0] % n)]

            # If the model has a preprocess method, call it now 
            if hasattr(self.model, 'preprocess_input'):
                s0 = time()
                netsurf.utils.log._custom('BMK',f'Preprocessing input data...', end = '')
                XTrain = self.model.preprocess_input(XTrain)
                XTest = self.model.preprocess_input(XTest)
                print(f'done in {time() - s0:.2f} seconds')
            # Same for output 
            if hasattr(self.model, 'preprocess_output'):
                s0 = time()
                netsurf.utils.log._custom('BMK',f'Preprocessing output data...', end = '')
                YTrain = self.model.preprocess_output(YTrain)
                YTest = self.model.preprocess_output(YTest)
                print(f'done in {time() - s0:.2f} seconds')
            
            # Make sure if we have a normalizing layer, apply norm here now
            if hasattr(self.model, 'normalize'):
                s0 = time()
                netsurf.utils.log._custom('BMK',f'Applying norm.adapt to training data...', end = '')
                self.model.normalize(np.array(XTrain))
                print(f'done in {time() - s0:.2f} seconds')
            
            # Make sure the model has been compiled
            try:
                self.model.fit(XTrain[:batch_size], YTrain[:batch_size], verbose = 0, batch_size = batch_size)
            except Exception as e:
                # Recompile
                netsurf.utils.log._error(f'Model looks uncompiled. Recompiling now...')
                self.model.compile(self.optimizer, loss = self.loss, metrics = self.metrics)


            # Add fisher tracker 
            fisher_tracker = netsurf.dnn.callbacks.FisherTrackingCallback(self.model, 
                                                                          (np.array(XTest), np.array(YTest)),
                                                                          loss_fn = self.model.loss, 
                                                                          n_batches = 10)
            callbacks += [fisher_tracker]

            """ Create custom callback to print on every epoch using our own printer"""
            # THIS HAS TO BE THE LAST TO BE ADDED
            # Add custom printer callback
            callbacks += [netsurf.dnn.callbacks.CustomPrinter()]

            logs = self.model.fit(np.array(XTrain), np.array(YTrain),
                epochs = epochs,
                batch_size = batch_size,
                validation_data = (np.array(XTest), np.array(YTest)) if not 'ECON' in self.name else None, 
                callbacks = callbacks,
                verbose = False,
                **kwargs #,callbacks=[pruning_callbacks.UpdatePruningStep()]
            )

        elif isinstance(self.dataset['train'], keras.preprocessing.image.DirectoryIterator) or isinstance(self.dataset['train'], tf.data.Dataset):

            # Add fisher tracker 
            fisher_tracker = netsurf.dnn.callbacks.FisherTrackingCallback(self.model, 
                                                                          self.dataset['validation'],
                                                                          loss_fn = self.model.loss, 
                                                                          n_batches = 10)
            callbacks += [fisher_tracker]

            """ Create custom callback to print on every epoch using our own printer"""
            # THIS HAS TO BE THE LAST TO BE ADDED
            # Add custom printer callback
            callbacks += [netsurf.dnn.callbacks.CustomPrinter()]

            logs = self.model.fit(self.dataset['train'], 
                                        epochs = epochs,
                                        validation_data = self.dataset['validation'],
                                        callbacks = callbacks,
                                        verbose = False,
                                        **kwargs)
        else:
            raise ValueError('Unknown dataset type')

        end = time()
        elapsed_time = end - start

        # Make sure to update the epochs value because we might have stopped earlier if we used early stopping
        epochs = logs.params['epochs']

        netsurf.utils.log._custom('MDL',f'Model fitted in {elapsed_time:.2f} seconds after {epochs} epochs')
        
        if prune:
            # Get the supported weights pruned masks
            results = netsurf.dnn.models.get_supported_weights(self.model, verbose = False)
            pruned_masks, supported_weights, supported_layers, params_num = results
            
            # Set in place
            self.model.pruned_masks = pruned_masks 

        # Divide the elapsed_time by the epochs (this is not accurate but it's a good approximation just for now)
        stamps = np.linspace(0, elapsed_time, epochs)
        stamps_hms = [netsurf.utils.seconds_to_hms(s) for s in stamps]
        
        # Add to logs
        logs.history['time'] = stamps_hms

        # Add logs to session
        sess.logs = logs

        # Add config to session
        sess.add_to_config('batch_size', batch_size)
        sess.add_to_config('epochs', epochs)
        sess.add_to_config('optimizer', self.optimizer)
        sess.add_to_config('loss', self.loss_names)
        sess.add_to_config('metrics', self.metric_names)
        sess.add_to_config('lr', self.model.optimizer.lr.numpy())

        # Add session to benchmark 
        self.sessions.append(sess)

        return sess, logs

    def get_dataset_sample(self, subset = 'validation', nsamples = -1):
        # Get the data
        xsample = (self.dataset[subset])[0][:nsamples] if subset in self.dataset else None
        ysample = (self.dataset[subset])[1][:nsamples] if subset in self.dataset else None

        # If the model has a preprocess method, call it now 
        if hasattr(self.model, 'preprocess_input'):
            s0 = time()
            netsurf.utils.log._custom('MDL',f'Preprocessing input data...', end = '')
            xsample = self.model.preprocess_input(xsample)
            print(f'done in {time() - s0:.2f} seconds')
        # Same for output 
        if hasattr(self.model, 'preprocess_output'):
            s0 = time()
            netsurf.utils.log._custom('MDL',f'Preprocessing output data...', end = '')
            ysample = self.model.preprocess_output(ysample)
            print(f'done in {time() - s0:.2f} seconds')

        if hasattr(self.model, 'normalize'):
            s0 = time()
            netsurf.utils.log._custom('BMK',f'Applying norm.adapt to training data...', end = '')
            self.model.normalize(np.array(xsample))
            print(f'done in {time() - s0:.2f} seconds')
        
        return xsample, ysample


    """ Function to evaluate the model and plot it """
    def evaluate(self, sess, *args, nsamples = -1, to_file = False, filepath = None, show = True, **kwargs):
        self.assert_dataset_is_loaded()

        xsample, ysample = self.get_dataset_sample(subset = 'validation', nsamples = nsamples)
        
        if to_file:
            if filepath is None:
                session_dir = sess.path
                if self.model.type == 'classification':
                    filepath = os.path.join(session_dir, '{}.png')
                elif self.model.type == 'regression':
                    filepath = os.path.join(session_dir, f'scatter.png')
                else:
                    filepath = os.path.join(session_dir, f'evaluation.png')
    
        self.model.evaluate(xsample, ysample, *args, filepath = filepath, show = show, **kwargs)
    
    """ Function to plot the sparsity of weights """
    def plot_sparsity(self, sess, *args, to_file = False, filepath = None, show = True, separated = False, **kwargs):
        self.assert_dirs_built()
        if to_file:
            if filepath is None:
                session_dir = sess.path
                filepath = os.path.join(session_dir, 'sparsity.png' if not separated else 'sparsity_separated.png')
        
        netsurf.utils.plot.plot_sparsity(self.model, filepath = filepath, show = show, separated = separated)


""" Assertions """
def assert_benchmark(benchmark):
    if benchmark is not None:
        if benchmark not in netsurf.config.AVAILABLE_BENCHMARKS:
            netsurf.utils.log._error(f'Benchmark {benchmark} not available. Available options: {", ".join(netsurf.config.AVAILABLE_BENCHMARKS)}. Returning now.')
            benchmark = None
    return benchmark

""" Get benchmark """
def get_benchmark(benchmark: str, quantization: 'QuantizationScheme', *args, **kwargs) :
    
    """ Assert benchmark is available """
    benchmark = assert_benchmark(benchmark)
    if benchmark is None: return None   

    # Get benchmark config
    benchmarks_config = netsurf.config.BENCHMARKS_CONFIG

    # Copy config (so we don't modify the original)
    benchmarks_config_copy = copy.deepcopy(benchmarks_config)

    # Make sure we have no duplicate kwargs between kwargs and benchmarks_config_copy.
    # If we do, always take the one from kwargs
    mixed_kwargs = {}
    for k, v in benchmarks_config_copy[benchmark].items():
        if k in kwargs:
            mixed_kwargs[k] = kwargs[k]
        else:
            mixed_kwargs[k] = v
    
    for k, v in kwargs.items():
        if k not in mixed_kwargs:
            mixed_kwargs[k] = v

        # Pop dataset name and model 
    dataset_name = mixed_kwargs.pop('dataset')
    model_name = mixed_kwargs.pop('model')
    
    # Print benchmark 
    netsurf.utils.log._custom('BMK', f'Initializing benchmark object {benchmark}')

    """ Build benchmark object """
    bmk = Benchmark(benchmark, dataset_name, model_name, quantization,
                              *args, **mixed_kwargs)
    
    netsurf.utils.log._custom('BMK', f'Benchmark object {benchmark} initialized')
    return bmk


""" Get training session (load or train) """
def get_training_session(bmk, train_model = False, pruning = None, show_plots = False, plot = True):

    # Assert dataset is loaded 
    bmk.assert_dataset_is_loaded()

    # Get benchmark config
    benchmarks_config = netsurf.config.BENCHMARKS_CONFIG

    # Parse config
    session_params = netsurf.utils.parse_config(benchmarks_config[bmk.name])

    # Try to find the latest session and load 
    sess = netsurf.load_session(bmk.sessions_dir, quantizer = bmk.quantization, latest = True)

    """ Fit params """
    if train_model or sess is None:

        # Loop thru sessions
        for sp in session_params:
            
            """ Optimizer params """
            opt_params = sp['optimizer_params']
            batch_size = sp['batch_size']
            epochs = sp['epochs']
            pruning_params = {}
            if pruning:
                if isinstance(pruning, float) and pruning > 0.0 and pruning <= 1.0:
                    pruning_params = sp['pruning_params']
                    pruning_params['final_sparsity'] = pruning
            else:
                if bmk.pruning_factor > 0.0:
                    pruning_params = sp['pruning_params']
                    pruning_params['final_sparsity'] = bmk.pruning_factor

            # Print info 
            netsurf.utils.log._custom('MDL', f'Running session with batch_size = {batch_size}, epochs = {epochs}, opt_params = {opt_params}, pruning_params = {pruning_params}')

            # Parse callbacks 
            callbacks = netsurf.dnn.callbacks.parse_callbacks(bmk.model, sp['callbacks'], pruning_params = pruning_params)

            # Compile model 
            bmk.compile(opt_params = opt_params, batch_size = batch_size)

            # Run fitting
            sess, logs = bmk.fit(batch_size = batch_size, epochs = epochs, callbacks = callbacks) #callbacks=[pruning_callbacks.UpdatePruningStep()]

            # Save session config and object
            sess.save()

            # Save weights into file
            bmk.save_weights(prefix = bmk.model_full_name)   

    else:

        # Parse 
        opt_params = {}
        pruning_params = {}
        batch_size = 32
        if len(session_params) > 0:
            # Last session
            sp = session_params[-1]

            """ Optimizer params """
            opt_params = sp['optimizer_params']
            batch_size = sp['batch_size']
            epochs = sp['epochs']
            pruning_params = {}
            if pruning:
                if isinstance(pruning, float) and pruning > 0.0 and pruning <= 1.0:
                    pruning_params = sp['pruning_params']
                    pruning_params['final_sparsity'] = pruning
            else:
                if bmk.pruning_factor > 0.0:
                    pruning_params = sp['pruning_params']
                    pruning_params['final_sparsity'] = bmk.pruning_factor
            
            # Print info 
            netsurf.utils.log._custom('MDL', f'Loading session with batch_size = {batch_size}, epochs = {epochs}, opt_params = {opt_params}, pruning_params = {pruning_params}')

            # Parse callbacks 
            callbacks = netsurf.dnn.callbacks.parse_callbacks(bmk.model, sp['callbacks'], pruning_params = pruning_params)
            
        # Compile model 
        bmk.compile(opt_params = opt_params, batch_size = batch_size)

        # Get logs 
        logs = sess.logs

    """ Plots before leaving """
    if plot:
        # Plot training history
        sess.plot_training_history(logs, to_file = True, show = show_plots)

        # Plot weight dist 
        bmk.plot_weight_pie(to_file = True, show = show_plots)

        # Plot sparsity
        bmk.plot_sparsity(sess, to_file = True, show = show_plots)
        bmk.plot_sparsity(sess, separated = True, to_file = True, show = show_plots)

        # Evaluate model 
        bmk.evaluate(sess, to_file = True, show = show_plots, plot = True)
        
    return sess