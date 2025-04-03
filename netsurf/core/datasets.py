""" 
    This code builds, compiles and trains the MNIST architectures used to test our methods 
"""

""" Main modules"""
import os, sys
import yaml
import re

""" Numpy """
import numpy as np

""" Pandas """
import pandas as pd

""" File processing, handling and download """
import zipfile, tarfile
from glob import glob

""" Tqdm """
from tqdm import tqdm

""" Tensorflow and keras """
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras  
from keras import backend as K # To get intermediate activations between layers

""" Matplotlib """
import matplotlib.pyplot as plt
import seaborn as sns

""" Seaborn """
import seaborn as sns

""" Custom utils """
from netsurf import utils

import skimage.io as io
from skimage.color import gray2rgb
from skimage.transform import resize

""" Import netsurf """
import netsurf

""" Import pergamos """
import pergamos as pg

import numpy as np

class Normalizer:
    def __init__(self, target_range=(0, 1)):
        """
        Initialize the Normalizer with a given target range.
        """
        self.target_range = target_range
        self.reset()
    
    def reset(self):
        """
        Reset the learned coefficients (min and max) and training call count.
        """
        self._data_min = None
        self._data_max = None
        self._train_calls = 0
    
    def train(self, data):
        """
        Train (or update) the normalizer with new data.
        The coefficients (min and max) are computed per feature and averaged
        with previous values if already set.
        
        Parameters:
            data (np.ndarray): Array of shape (samples, features) or 1D array.
        """
        # Compute current per-feature min and max
        # Flatten data by all but last dimension (features, channels, etc.)
        data = np.atleast_2d(data)
        data = np.reshape(data, (np.prod(data.shape[:-1]), -1))

        current_min = np.min(data, axis=0)
        current_max = np.max(data, axis=0)
        
        if self._train_calls == 0:
            self._data_min = current_min
            self._data_max = current_max
        else:
            # Update as a running average: new_avg = (old_avg * n + new_value) / (n + 1)
            n = self._train_calls
            self._data_min = (self._data_min * n + current_min) / (n + 1)
            self._data_max = (self._data_max * n + current_max) / (n + 1)
        
        self._train_calls += 1
    
    """ Call method to normalize or denormalize data """
    def __call__(self, x: np.ndarray, invert = False):
        if not invert:
            return self.normalize(x)
        else:
            return self.denormalize(x)

    def normalize(self, data):
        """
        Normalize the given data using the learned coefficients.
        
        Parameters:
            data (np.ndarray): Data to normalize.
            
        Returns:
            np.ndarray: Normalized data within the target range.
        """
        if self._data_min is None or self._data_max is None:
            raise ValueError("Normalizer has not been trained yet. Call train() first.")
        
        target_min, target_max = self.target_range
        range_data = self._data_max - self._data_min
        
        # Prevent division by zero (if any feature has no variation)
        range_data = np.where(range_data == 0, 1, range_data)

        # self._data_min and range_data have the same size as data.shape[-1], which means we have to
        # expand them to the same shape as data. This is done by broadcasting the arrays to the same shape.

        dmin = np.broadcast_to(self._data_min, data.shape)
        dmax = np.broadcast_to(self._data_max, data.shape)
        rdata = np.broadcast_to(range_data, data.shape)
        
        normalized = target_min + (data - dmin) * (target_max - target_min) / rdata
        return normalized
    
    def denormalize(self, normalized_data):
        """
        Denormalize the data back to the original scale.
        
        Parameters:
            normalized_data (np.ndarray): Data in the target normalized range.
            
        Returns:
            np.ndarray: Data mapped back to the original scale.
        """
        if self._data_min is None or self._data_max is None:
            raise ValueError("Normalizer has not been trained yet. Call train() first.")
        
        target_min, target_max = self.target_range
        range_target = target_max - target_min
        range_data = self._data_max - self._data_min
        
        # Prevent division by zero
        if range_target == 0:
            range_target = 1
        
        # broadcast
        dmin = np.broadcast_to(self._data_min, normalized_data.shape)
        dmax = np.broadcast_to(self._data_max, normalized_data.shape)
        rdata = np.broadcast_to(range_data, normalized_data.shape)
        
        original = dmin + (normalized_data - target_min) * rdata / range_target
        return original
    
    def __repr__(self):
        return f"Normalizer(target_range={self.target_range}) <- ({self._data_min}, {self._data_max})"




""" Generic dataset class """
class Dataset:
    def __init__(self, quantizer: 'QuantizationScheme', verbose=True, 
                 datasets_dir = '.', **kwargs):

        # Set quantizer and dir
        self.quantizer = quantizer
        self.datasets_dir = datasets_dir
        self.cmap = 'gray'

    def get_figsize(self, nrows, ncols):
        return (ncols * 4, nrows * 4)

    def build_dataset(self, dataset: dict, types: dict = None, **kwargs):
        # assert dataset has at least train or validation
        assert('train' in dataset or 'validation' in dataset or 'test' in dataset)
        key = 'train' if 'train' in dataset else 'validation' if 'validation' in dataset else 'test'

        # Assert types 
        if types is not None:
            if 'input' in types:
                if not self.assert_type(types['input']):
                    raise ValueError(f"Invalid type for input: {types['input']}")
            if 'output' in types:
                if not self.assert_type(types['output']):
                    raise ValueError(f"Invalid type for output: {types['output']}")
        
        if types is None:
            types = {'input': f'{dataset[key][0].ndim}d', 
                     'output': f'{dataset[key][1].ndim}d'}

        # Make sure we normalize the dataset
        self.dataset, self.normalizer = self.norm(dataset, self.quantizer)
        self.types = types

    def assert_type(self, t: str):
        # regex for "<int>d"
        if re.match(r'^\d+d$', t):
            return True
        else:
            if t == 'img' or t == 'volume' or t == 'class':
                return True
            return False

    
    """ Method to normalize the data (EVERY DATASET HAS TO BE NORMALIZED!!) """
    def norm(self, dataset: dict, quantizer: 'QuantizerScheme', normalizer = None, verbose = True):
        
        # We need to normalize the dataset using the quantizer range.
        # Build a normalizer obj for input and one for output
        if normalizer is None:
            if not hasattr(self, 'normalizer'):
                normalizer = Normalizer(target_range=(quantizer.min_value, quantizer.max_value))
            else:
                normalizer = self.normalizer

        # Train normalizer
        if verbose: netsurf.utils.log._custom('DATA', f"Normalizing dataset (input) using quantizer range ({quantizer.min_value}, {quantizer.max_value})")
        normalizer.train(dataset['train'][0])

        # Normalize the dataset
        dataset['train'] = (normalizer(dataset['train'][0]), dataset['train'][1])
        dataset['validation'] = (normalizer(dataset['validation'][0]), dataset['validation'][1])

        return dataset, normalizer
    
    @property
    def in_shape(self):
        return self.dataset['train'][0].shape[1:]
    
    @property
    def out_shape(self):
        return self.dataset['train'][1].shape[1:]
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.dataset[key]
        elif isinstance(key, int):
            return list(self.dataset.keys())[key]
        return None
    
    def __setitem__(self, key, value):
        if isinstance(key, str):
            self.dataset[key] = value
        else:
            raise ValueError("Key must be a string")

    def html(self):

        # Create collapsible container for this dataset
        dataset_ct = pg.CollapsibleContainer(f"🗄️ {self.__class__.__name__} (Dataset)", layout='vertical')

        # Create a container showing the basic information summary for this summary 
        summary_ct = pg.Container("Summary", layout='vertical')
        # Create pandas dataframe 
        df = pd.DataFrame({'input_shape': [self.in_shape], 
                           'input_type': str(self.types['input']),
                           'output_shape': str(self.out_shape),
                           'output_type': [self.types['output']],
                           'subsets': "(" + ", ".join(map(str,list(self.dataset.keys()))) + ")",
                           'samples': "(" + ", ".join([str(self.dataset[subset][0].shape[0]) for subset in self.dataset.keys()]) + ")",
                           'normalizer stats (max,min)': f"({self.normalizer._data_max[0]}, {self.normalizer._data_min[0]})"
                           }).T
        
        # Add to container
        summary_ct.append(pg.Table.from_data(df))

        # Now plot distributions
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        self.plot_data_distribution(subset = 'train', title = 'Train data distribution', axs = [axs[0][0], axs[0][1]], types = self.types)
        self.plot_data_distribution(subset = 'validation', title = 'Validation data distribution', axs = [axs[1][0], axs[1][1]], types = self.types)

        # Create a Container for this image 
        data_dist_ct = pg.Container("Data distribution", layout='vertical')
        data_dist_ct.append(pg.Image(fig, embed=True))

        # If this is an image dataset, display some samples 
        num_rows = 0
        num_samples = 10
        if self.types['input'] == 'img': num_rows += len(self.dataset.keys())
        if self.types['output'] == 'img': num_rows += len(self.dataset.keys())

        if num_rows > 0:
            # Init figure
            fig, axs = plt.subplots(num_rows, num_samples, figsize=self.get_figsize(num_rows, num_samples))
            # Make sure axs is a list
            if isinstance(axs, np.ndarray):
                axs = list(axs)
            if not isinstance(axs, (list, tuple)):
                axs = [axs]

            k = 0
            for i, subset in enumerate(self.dataset.keys()):
                if self.types['input'] == 'img':
                    self.display_image_samples(subset = subset, index = 0, random = True, num_samples = num_samples, axs = axs[k],
                                                cmap = self.cmap, show = False)
                    k += 1
                if self.types['output'] == 'img':
                    self.display_image_samples(subset = subset, index = 1, random = True, num_samples = num_samples, axs = axs[k],
                                                cmap = self.cmap, show = False)
                    k += 1

            # Create a container for this image
            img_ct = pg.Container("Image samples", layout='vertical')
            img_ct.append(pg.Image(fig, embed=True))
            # close fig
            plt.close(fig)

        # Add to dataset_ct
        dataset_ct.append(summary_ct)
        dataset_ct.append(data_dist_ct)
        if num_rows > 0:
            dataset_ct.append(img_ct)

        return dataset_ct
        

    def plot_data_distribution(self, subset = 'train', title = None, filename = None, show = True, axs = None, types = {'input': None, 'output': None}, 
                               max_num_samples = 10000, **kwargs):
        if axs is not None:
            if not isinstance(axs, (list, tuple)):
                print("axs must be a list or tuple of matplotlib axes")
                axs = None
            if len(axs) != 2:
                print("axs must have exactly 2 axes")
                axs = None
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            
        utils.plot.plot_quantized_histogram(self.dataset[subset][0][:max_num_samples], self.quantizer, ax = axs[0], title = title + ' X', filename = filename, show = show, 
                                            type = types['input'], **kwargs)
        utils.plot.plot_quantized_histogram(self.dataset[subset][1][:max_num_samples], self.quantizer, ax = axs[1], title = title + ' Y', filename = filename, show = show, 
                                            type = types['output'], **kwargs)

    
    def display_image_samples(self, subset = 'train', index = 0, random = True, num_samples = 10, 
                              axs = None, preprocess_fcn = None, **kwargs):
        assert subset in self.dataset

        # Get subsetdata 
        if isinstance(self.dataset[subset], tuple):
            d = self.dataset[subset][index]
        elif isinstance(self.dataset[subset], keras.preprocessing.image.DirectoryIterator):
            d = self.dataset[subset]
            num_samples = min(num_samples, d.n)
            d.reset()
            d = d.next()[index]
        
        # preprocess
        if preprocess_fcn is not None:
            d = preprocess_fcn(d)
        
        # Get random permutation
        if random:
            idxs = np.random.permutation(d.shape[0])[:num_samples]
        else:
            idxs = np.arange(num_samples)
        
        # Get samples
        X = d[idxs]

        if axs is not None:
            # assert axs shape is the same as num_samples
            if len(axs) != num_samples:
                netsurf.utils.log._warn(f"Number of axes ({len(axs)}) is different from number of samples ({num_samples}). Creating new axes")
                axs = None

        if axs is None:
            fig, axs = plt.subplots(1, num_samples, figsize=(7, 7))
        
        # ensure axs is a list
        if num_samples == 1:
            axs = [axs]
        axs = list(axs)
        vmin, vmax = self.normalizer.target_range

        utils.plot.display_data_img(X, axs = axs, vmin = vmin, vmax = vmax, **kwargs)
        
        
    
    """ Display distribution of labels and classes """
    def display_classes_distribution(self, subset = 'train', title = None, filename = None, show = True, overwrite = False, **kwargs):
        assert(subset in list(self.dataset.keys()) )

        if filename is not None:
            if '.png' in filename:
                # Add subset
                filename = filename.replace('.png', f'_{subset}.png')
        
        # Check if exists 
        if os.path.isfile(filename) and not overwrite:
            netsurf.utils.log._custom('DATA', f'File {filename} already exists. Skipping')
            return

        # Get subsetdata 
        if isinstance(self.dataset[subset], tuple):
            d = self.dataset[subset][1]
            nb_classes = self.dataset['train'][1].shape[1]
        elif isinstance(self.dataset[subset], keras.preprocessing.image.DirectoryIterator):
            d = self.dataset[subset]
            d.reset()
            d = d.next()[1]
            nb_classes = d.shape[1]

        # Get distribution
        dist = np.sum(d, axis=0)
        
        fig = plt.figure()
        plt.bar(np.arange(nb_classes), dist)
        _ = plt.xticks(np.arange(nb_classes))
        plt.xlabel("Classes")
        plt.ylabel("Frequency")
        if title is not None:
            plt.title(title)
        if filename is not None:
            fig.savefig(filename)
            netsurf.utils.log._custom('DATA', f'Saved classes distribution to {filename}')

            # Close 
            plt.close(fig)

        if show:
            plt.show()
    
    """ Display data statistics """
    def display_data_stats(self, **kwargs):
        for subset in self.dataset.keys():
        
            # Get subsetdata 
            if isinstance(self.dataset[subset], tuple):
                d = self.dataset[subset][0]
            elif isinstance(self.dataset[subset], keras.preprocessing.image.DirectoryIterator):
                d = self.dataset[subset]
                d.reset()
                d = d.next()[0]
            elif isinstance(self.dataset[subset], tf.data.Dataset):
                d = self.dataset[subset]
                d = d.as_numpy_iterator()
                d = d.next()[0]
            else:
                raise ValueError('Unknown dataset type')
            
            try:
                netsurf.utils.log._custom('DATA', f"Subset: {subset}")
                netsurf.utils.log._custom('DATA', f"\tData shape: {d.shape}")
                netsurf.utils.log._custom('DATA', f"\tData mean: {np.mean(d)}")
                netsurf.utils.log._custom('DATA', f"\tData std: {np.std(d)}")
                netsurf.utils.log._custom('DATA', f"\tData min: {np.min(d)}")
                netsurf.utils.log._custom('DATA', f"\tData max: {np.max(d)}")
            except:
                print('')
                print(f"[INFO] - Subset: {subset}")
                print(f"\tData shape: {d.shape}")
                print(f"\tData mean: {np.mean(d)}")
                print(f"\tData std: {np.std(d)}")
                print(f"\tData min: {np.min(d)}")
                print(f"\tData max: {np.max(d)}")
    


""" Dummy dataset for testing """
class dummy(Dataset):
    def __init__(self, quantizer: 'QuantizationScheme', **kwargs):
        """ Init super """
        super().__init__(quantizer, **kwargs)

        # Build data
        dataset = self.build_dataset(**kwargs)
        types = {'input': '1d', 'output': '1d'}

        # Now call super 
        super().build_dataset(dataset, types = types, **kwargs)
        
    """ Function to build the dataset """
    def build_dataset(self, verbose = True, **kwargs):
        # For reproducibility, we will store the data in a csv file and load it
        filepath = os.path.join(self.datasets_dir, 'dummy_data.npy')
        if os.path.isfile(filepath):
            if verbose: netsurf.utils.log._custom('DATA', f"Loading dataset from {filepath}")
            dataset = np.load(filepath, allow_pickle=True).item()
            return dataset
        else:
            # Set random seed for reproducibility
            np.random.seed(0)

            """ Generate random data """
            nfeats = 1
            X = np.random.rand(1000, nfeats)
            Y = 2.5*X[:,0] + 1.3 + np.random.normal(0, 0.3, X.shape[0])

            # Divide into training and validation
            ntrain = int(0.8*X.shape[0])
            XTrain, YTrain = X[:ntrain], Y[:ntrain]
            XVal, YVal = X[ntrain:], Y[ntrain:]

             # Set everything in place 
            dataset = {'train': (XTrain, YTrain), 'validation': (XVal, YVal)}

            # Save to file
            if verbose: netsurf.utils.log._custom('DATA', f"Saving dataset to {filepath}")
            np.save(filepath, dataset)
       
        return dataset
    
    """ Display functions """
    def display_data(self, subset = 'training', show = True, filename = None, overwrite = False, **kwargs):
        
        if filename is not None:
            if '.png' in filename:
                # Add subset
                filename = filename.replace('.png', f'_{subset}.png')

        if os.path.isfile(filename) and not overwrite:
            netsurf.utils.log._custom('DATA', f'File {filename} already exists. Skipping')
            return

        X = self.dataset[subset][0]
        Y = self.dataset[subset][1]
        nfeats = X.shape[1]
        #coeffs = self.coeffs

        fig, axs = plt.subplots(nrows = nfeats, ncols = 1, figsize = (6, 4*nfeats))
        if nfeats == 1:
            axs = [axs]
        for ax in range(nfeats):
            axs[ax].scatter(X[:,ax], Y)
            axs[ax].set_xlabel(f"Feature {ax}") #- {coeffs[ax]:.2f}")
            axs[ax].set_ylabel("Target")

        plt.tight_layout()
        
        if filename is not None:
            fig.savefig(filename)
            netsurf.utils.log._custom('DATA', f'Saved dataset sample to {filename}')

            # close fig
            plt.close(fig)

        if show:
            plt.show()
    

    
""" Build MNIST Dataset """
class MNIST(Dataset):
    def __init__(self, quantizer: 'QuantizationScheme', **kwargs):
        """ Init super """
        super().__init__(quantizer, **kwargs)

        # Build data
        dataset = self.build_dataset(**kwargs)
        types = {'input': 'img', 'output': '1d'}

        # Now call super 
        super().build_dataset(dataset, types = types, **kwargs)
    
    def get_figsize(self, nrows, ncols):
        return (3*nrows, 3*ncols)
        
    """ Function to build the dataset """
    def build_dataset(self, **kwargs):

        """ Get MNIST data """
        (XTrain, YTrain), (XTest, YTest) = keras.datasets.mnist.load_data()

        # Reshape 
        XTrain = XTrain.reshape(XTrain.shape + (1,)).astype("float32")
        XTest = XTest.reshape(XTest.shape + (1,)).astype("float32")

        # Convert labels to one-hot
        nb_classes = np.max(YTrain)+1
        YTrain = keras.utils.to_categorical(YTrain, nb_classes)
        YTest = keras.utils.to_categorical(YTest, nb_classes)

        # Set everything in place 
        dataset = {'train': (XTrain, YTrain), 'validation': (XTest, YTest)}

        return dataset
    

""" Build Fashion MNIST Dataset """
class FashionMNIST(Dataset):
    def __init__(self, quantizer: 'QuantizationScheme', **kwargs):
        """ Init super """
        super().__init__(quantizer, **kwargs)

        # Build data
        dataset = self.build_dataset(**kwargs)
        types = {'input': 'img', 'output': '1d'}

        # Now call super 
        super().build_dataset(dataset, types = types, **kwargs)
    
    def get_figsize(self, nrows, ncols):
        return (3*nrows, 3*ncols)
        
    """ Function to build the dataset """
    def build_dataset(self, verbose = True, **kwargs):

        """ Get MNIST data """
        (XTrain, YTrain), (XTest, YTest) = keras.datasets.fashion_mnist.load_data()

        # Reshape 
        XTrain = XTrain.reshape(XTrain.shape + (1,)).astype("float32")
        XTest = XTest.reshape(XTest.shape + (1,)).astype("float32")

        # Convert labels to one-hot
        nb_classes = np.max(YTrain)+1
        YTrain = keras.utils.to_categorical(YTrain, nb_classes)
        YTest = keras.utils.to_categorical(YTest, nb_classes)

        # Set everything in place 
        dataset = {'train': (XTrain, YTrain), 'validation': (XTest, YTest)}

        return dataset



""" Build CIFAR10 Dataset """
class CIFAR10(Dataset):
    def __init__(self, quantizer: 'QuantizationScheme', **kwargs):
        """ Init super """
        super().__init__(quantizer, **kwargs)

        # Build data
        dataset = self.build_dataset(**kwargs)
        types = {'input': 'img', 'output': '1d'}

        # Now call super 
        super().build_dataset(dataset, types = types, **kwargs)
        
    """ Function to build the dataset """
    def build_dataset(self, verbose = True, **kwargs):

        """ Get MNIST data """
        (XTrain, YTrain), (XTest, YTest) = keras.datasets.cifar10.load_data()

        # Reshape 
        XTrain = XTrain.astype("float32")/255.0
        XTest = XTest.astype("float32")/255.0
        
        # Convert labels to one-hot
        nb_classes = np.max(YTrain)+1
        YTrain = keras.utils.to_categorical(YTrain, nb_classes)
        YTest = keras.utils.to_categorical(YTest, nb_classes)

        # Set everything in place 
        dataset = {'train': (XTrain, YTrain), 'validation': (XTest, YTest)}

        return dataset


"""
    This code builds, compiles, and trains the SVHN architectures used to test our methods.
"""

""" Build SVHN Dataset """
class SVHN(Dataset):
    def __init__(self, quantizer : 'QuantizationScheme', verbose=True, **kwargs):
        """ Init super """
        super().__init__(quantizer, **kwargs)

        # Build data
        dataset = self.build_dataset(**kwargs)
        types = {'input': 'img', 'output': '1d'}

        # Now call super 
        super().build_dataset(dataset, types = types, **kwargs)

    def preprocess(self, image, label, nclasses=10):
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(tf.squeeze(label), nclasses)
        return image, label
    
    def dataset_to_numpy(self, ds):
        """
        Convert tensorflow dataset to numpy arrays
        """
        images = []
        labels = []

        # Iterate over a dataset
        for i, (image, label) in enumerate(tfds.as_numpy(ds)):
            images.append(image)
            labels.append(label)
        
        images = np.concatenate(images, axis=0)
        labels = np.vstack(labels)

        return images, labels

    """ Function to build the dataset """
    def build_dataset(self, verbose=True, **kwargs):
        # Load dataset splits
        ds_train, info = tfds.load('svhn_cropped', split='train[:90%]', with_info=True, as_supervised=True)
        ds_test = tfds.load('svhn_cropped', split='test', shuffle_files=True, as_supervised=True)
        ds_val = tfds.load('svhn_cropped', split='train[-10%:]', shuffle_files=True, as_supervised=True)

        assert isinstance(ds_train, tf.data.Dataset)
        train_size = int(info.splits['train'].num_examples)
        input_shape = info.features['image'].shape
        n_classes = info.features['label'].num_classes

        netsurf.utils.log._custom('DATA', f'Training on {train_size} samples of input shape {input_shape}, belonging to {n_classes} classes')

        # Preprocess data
        batch_size = 1024
        train_data = ds_train.map(lambda x, y: self.preprocess(x, y, n_classes))
        test_data = ds_test.map(lambda x, y: self.preprocess(x, y, n_classes))
        val_data = ds_val.map(lambda x, y: self.preprocess(x, y, n_classes))

        train_data = train_data.shuffle(4096).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_data = test_data.shuffle(4096).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_data = val_data.shuffle(4096).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Convert dataset to numpy arrays for easy handling
        XTrain, YTrain = self.dataset_to_numpy(train_data)
        XTest, YTest = self.dataset_to_numpy(test_data)
        XVal, YVal = self.dataset_to_numpy(val_data)        

        # Store the dataset and statistics
        dataset = {'train': (XTrain, YTrain), 'test': (XVal, YVal), 'validation': (XTest, YTest)}

        return dataset


""" ToyADMOS (from ML Tiny perf) """
class ToyADMOS(Dataset):
    def __init__(self, quantizer : 'QuantizationScheme', **kwargs):
        """ Init super """
        super().__init__(quantizer, **kwargs)

        # Build data
        dataset = self.build_dataset(**kwargs)
        types = {'input': '1d', 'output': '1d'}

        # Now call super 
        super().build_dataset(dataset, types = types, **kwargs)

    def display_data(self, *args, **kwargs):
        print("[WARN] - Displaying data not implemented for ToyADMOS")

    def display_classes_distribution(self, *args, **kwargs):
        print("[WARN] - Displaying classes distribution not implemented for ToyADMOS")

    """ Function to build the yaml config for the data """
    def build_yaml_config(self, datasets_dir = "./datasets", **kwargs):
        
        # Create yaml dict
        yaml_config = {
            'dev_directory': os.path.join(datasets_dir, "ToyADMOS", "train"),
            'eval_directory' : os.path.join(datasets_dir, "ToyADMOS", "test"),
            'max_fpr' : 0.1,
            'feature': {
                'n_mels': 128,
                'frames' : 5,
                'n_fft': 1024,
                'hop_length': 512,
                'power': 2.0
            },
            'fit': {
                'compile': {
                    'optimizer' : 'adam',
                    'loss' : 'mean_squared_error'
                },
                'epochs' : 100,
                'batch_size' : 512,
                'shuffle' : True,
                'validation_split' : 0.1,
                'verbose' : 1
            }
        }

        # Save yaml config
        if not os.path.isfile(os.path.join(datasets_dir, "ToyADMOS", 'config.yml')):
            with open(os.path.join(datasets_dir,"ToyADMOS", 'config.yml'), 'w') as outfile:
                yaml.dump(yaml_config, outfile, default_flow_style=False)

        return yaml_config


    """ 
        Directly taken from: https://github.com/mlcommons/tiny/blob/11d3beb0e03418a1e83903e42865f081d55a48b5/benchmark/training/anomaly_detection/common.py
    """
    def file_load(self, file_name, mono=False):
        """ Audio processing """
        import librosa
        import librosa.core
        import librosa.feature
        #import librosa.display

        try:
            return librosa.load(file_name, sr=None, mono=mono)
        except:
            raise ValueError("file_broken or not exists!! : {}".format(file_name))

    ########################################################################
    # feature extractor
    #   Direcly extracted from: https://github.com/mlcommons/tiny/blob/master/benchmark/training/anomaly_detection/common.py#L135
    ########################################################################
    def file_to_vector_array(self, file_name,
                            n_mels=64,
                            frames=5,
                            n_fft=1024,
                            hop_length=512,
                            power=2.0,
                            method="librosa",
                            save_png=False,
                            save_hist=False,
                            save_bin=False,
                            save_parts=False):
        """
        convert file_name to a vector array.

        file_name : str
            target .wav file

        return : np.array( np.array( float ) )
            vector array
            * dataset.shape = (dataset_size, feature_vector_length)
        """
        # 01 calculate the number of dimensions
        dims = n_mels * frames

        # 02 generate melspectrogram
        y, sr = self.file_load(file_name)
        if method == "librosa":
            # 02a generate melspectrogram using librosa
            mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                            sr=sr,
                                                            n_fft=n_fft,
                                                            hop_length=hop_length,
                                                            n_mels=n_mels,
                                                            power=power)

            # 03 convert melspectrogram to log mel energy
            log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)


        else:
            raise ValueError("spectrogram method not supported: {}".format(method))
            return np.empty((0, dims))

        # 3b take central part only
        log_mel_spectrogram = log_mel_spectrogram[:,50:250]

        # 04 calculate total vector size
        vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1

        # 05 skip too short clips
        if vector_array_size < 1:
            return np.empty((0, dims))

        # 06 generate feature vectors by concatenating multiframes
        vector_array = np.zeros((vector_array_size, dims))
        for t in range(frames):
            vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T

        # 07 (optional) save histogram in png
        # if save_png:
        #     save_path = file_name.replace('.wav', '_hist_' + method + '.png')
        #     librosa.display.specshow(log_mel_spectrogram)
        #     pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
        #     pylab.close()

        # 08 (optional) save histogram
        if save_hist:
            save_path = file_name.replace('.wav', '_hist_' + method + '.txt')
            # transpose to obtain correct order
            np.swapaxes(log_mel_spectrogram, 0, 1).tofile(save_path, sep=",")

        # 08 (optional) save bin
        if save_bin:
            save_path = file_name.replace('.wav', '_hist_' + method + '.bin')
            # transpose to obtain correct order
            np.swapaxes(log_mel_spectrogram, 0, 1).astype('float32').tofile(save_path)

        # 08 (optional) save parts (sliding window)
        if save_parts:
            for i in range(vector_array_size):
                save_path = file_name.replace('.wav', '_hist_' + method + '_part{0:03d}'.format(i) + '.bin')
                # transpose to obtain correct order?
                vector_array[i].astype('float32').tofile(save_path)

        return vector_array


    """ Function to convert wav files to numpy arrays 
            This code was directly taken from here -> https://github.com/mlcommons/tiny/blob/master/benchmark/training/anomaly_detection/common.py#L372
            Original function name: list_to_vector_array
    """
    def wav_to_numpy(self, wav_files,
                        n_mels=64,
                        frames=5,
                        n_fft=1024,
                        hop_length=512,
                        power=2.0, **kwargs):
        """
        convert the file_list to a vector array.
        file_to_vector_array() is iterated, and the output vector array is concatenated.

        wav_files : list [ str ]
            .wav filename list of dataset
        msg : str ( default = "calc..." )
            description for tqdm.
            this parameter will be input into "desc" param at tqdm.

        return : np.array( np.array( float ) )
            vector array for training (this function is not used for test.)
            * dataset.shape = (number of feature vectors, dimensions of feature vectors)
        """
        # calculate the number of dimensions
        dims = n_mels * frames

        # iterate file_to_vector_array()
        for idx in tqdm(range(len(wav_files)), desc='Converting wav files to numpy array', ncols = 100):
            vector_array = self.file_to_vector_array(wav_files[idx],
                                                    n_mels=n_mels,
                                                    frames=frames,
                                                    n_fft=n_fft,
                                                    hop_length=hop_length,
                                                    power=power)
            if idx == 0:
                dataset = np.zeros((vector_array.shape[0] * len(wav_files), dims), float)
            dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array

        return dataset

    """ Function to build the dataset """
    def build_dataset(self, datasets_dir = "./datasets", verbose=True, **kwargs):
        # Make sure we have write permission in this directory
        if not os.access(".", os.W_OK):
            raise ValueError("Cannot write to current directory. Aborting download of dataset")
        
        # Create output directory
        os.makedirs(datasets_dir, exist_ok=True)

        # Download the dataset
        output_dir = os.path.join(datasets_dir, "ToyADMOS")
        if not os.path.isdir(output_dir):
            output_file = os.path.join(datasets_dir, "anomaly_detection_data_train.zip")
            if not os.path.exists(output_file):
                utils.download_file("https://zenodo.org/record/3678171/files/dev_data_ToyADMOS.zip?download=1", output_file)

            # Unzip the dataset
            with zipfile.ZipFile(output_file, 'r') as zip_ref:
                zip_ref.extractall(datasets_dir)
            
            # Remove zip file
            os.remove(output_file)
        
        # Validation data
        output_dir = os.path.join(datasets_dir, "ToyADMOS")
        if not os.path.isdir(output_dir):
            output_file = os.path.join(datasets_dir, "anomaly_detection_data_val.zip")
            if not os.path.exists(output_file):
                utils.download_file("https://zenodo.org/record/3727685/files/eval_data_train_ToyADMOS.zip?download=1", output_file)

            # Unzip the dataset
            with zipfile.ZipFile(output_file, 'r') as zip_ref:
                zip_ref.extractall(datasets_dir)
            
            # Remove zip file
            os.remove(output_file)


        # Build yaml config
        yaml_config = self.build_yaml_config(datasets_dir = datasets_dir, **kwargs)

        # Load wav files into numpy arrays 
        train_numpy_file = os.path.join(datasets_dir, "ToyADMOS", "train", "train_ds.npy")
        if not os.path.isfile(train_numpy_file):
            training_list_path = os.path.abspath(os.path.join(datasets_dir, "ToyADMOS", "train", "*.wav"))
            files = sorted(glob(training_list_path))
            train_ds = self.wav_to_numpy(files, **yaml_config)
            np.save(train_numpy_file, train_ds)
            netsurf.utils.log._custom('DATA', f"Saved train numpy wavs to {train_numpy_file}")
        else:
            netsurf.utils.log._custom('DATA', f"Loading train numpy wavs from {train_numpy_file}")
            train_ds = np.load(train_numpy_file)

        test_numpy_file = os.path.join(datasets_dir, "ToyADMOS", "test", "test_ds.npy")
        if not os.path.isfile(test_numpy_file):
            val_list_path = os.path.abspath(os.path.join(datasets_dir, "ToyADMOS", "test", "*.wav"))
            files = sorted(glob(val_list_path))
            val_ds = self.wav_to_numpy(files, **yaml_config)
            np.save(test_numpy_file, val_ds)
            netsurf.utils.log._custom('DATA', f"Saved test numpy wavs to {test_numpy_file}")
        else:
            netsurf.utils.log._custom('DATA', f"Loading test numpy wavs from {test_numpy_file}")
            val_ds = np.load(test_numpy_file)


        XTrain, YTrain = train_ds[:137200], train_ds[:137200]
        XTest, YTest = val_ds[:137200], val_ds[:137200]

        # Store the dataset and statistics
        dataset = {'train': (XTrain, YTrain), 'validation': (XTest, YTest)}

        return dataset



""" coco dataset """
class COCO(Dataset):
    def __init__(self, problem_type, quantizer: 'QuantizationScheme', **kwargs):
        """ Init super """
        super().__init__(quantizer, **kwargs)

        # Type of problem
        self.problem_type = problem_type
        self.parameters = {'person_detection': {'wakeword':'person', 'areaRatio':.025, 'useBoundingBoxArea':True, 'imgSize':96}}

        # Build data
        dataset = self.build_dataset(**kwargs)
        types = {'input': 'img', 'output': '1d'}

        # Now call super 
        super().build_dataset(dataset, types = types, **kwargs)

    @property
    def _urls(self):
        return {
            '2014': {
                'train': 'http://images.cocodataset.org/zips/train2014.zip',
                'test': 'http://images.cocodataset.org/zips/test2014.zip',
                'val': 'http://images.cocodataset.org/zips/val2014.zip',
                'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
            },
            '2017': {
                'train': 'http://images.cocodataset.org/zips/train2017.zip',
                'test': 'http://images.cocodataset.org/zips/test2017.zip',
                'val': 'http://images.cocodataset.org/zips/val2017.zip',
                'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
            }
        }

    # Generation function for each dataset portion
    #   Taken directly from: https://github.com/SiliconLabs/platform_ml_models/blob/master/eembc/Person_detection/buildPersonDetectionDatabase.py
    def _generate_coco_instance(self, data_type, output_data_dir,  
                                    wakeword='person', areaRatio=.025, useBoundingBoxArea=True, imgSize=96):
        
        """ Coco tools """
        from pycocotools.coco import COCO as pycoco

        # Parameters, change directories for a local run
        useLocalCoco=True
        debugCats=False
        annotations_dir = os.path.join(output_data_dir, data_type.replace('train','annotations').replace('val','annotations'))

        # initialize COCO api for instance annotations
        annotation_file = '{}/instances_{}.json'.format(annotations_dir,data_type)
        #print(data_type + ', starting processing')
        coco = pycoco(annotation_file)

        # display COCO categories and supercategories
        if( debugCats ):
            cats = coco.loadCats(coco.getCatIds())
            nms=[cat['name'] for cat in cats]
            netsurf.utils.log._custom('DATA', 'COCO categories: \n{}\n'.format(' '.join(nms)))

            nms = set([cat['supercategory'] for cat in cats])
            netsurf.utils.log._custom('DATA', 'COCO supercategories: \n{}'.format(' '.join(nms)))

        # Get all images containing given categories
        catIds = coco.getCatIds(catNms=wakeword)

        # Create the output directories
        if not os.path.exists(output_data_dir):
            os.makedirs(output_data_dir)
        if not os.path.exists('%s/%s'%(output_data_dir,data_type)):
            os.makedirs('%s/%s'%(output_data_dir,data_type))
        if not os.path.exists('%s/%s/%s'%(output_data_dir,data_type,wakeword)):
            os.makedirs('%s/%s/%s'%(output_data_dir,data_type,wakeword))
        if not os.path.exists('%s/%s/non_%s'%(output_data_dir,data_type,wakeword)):
            os.makedirs('%s/%s/non_%s'%(output_data_dir,data_type,wakeword))

        # Loop over all images and process
        #index = 0
        if len(glob('%s/%s/%s/*.jpg'%(output_data_dir,data_type,wakeword))) == 0:
            for image in tqdm(coco.imgs, miniters=int(len(coco.imgs)/100), desc = f"Processing {data_type} images"):
                # Read image
                img = coco.loadImgs(image)[0]
                if( useLocalCoco ):
                    I = io.imread('%s/%s/%s'%(output_data_dir, data_type, img['file_name']))
                else:
                    I = io.imread(img['coco_url'])
                
                # Convert to RGB if needed
                if( I.ndim == 2 ):
                    I = gray2rgb(I)

                # Get annotation
                annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=False)
                anns = coco.loadAnns(annIds)

                # Debug plot
                if( False ):
                    plt.imshow(I)
                    plt.axis('off')
                    coco.showAnns(anns)
                    plt.show()

                # Area of image for ratio calculations
                imageArea = I.shape[0]*I.shape[1]

                # Check if at least a person is present
                fullFileName = ''
                if( len(anns) > 0 ):
                    for ann in anns:
                        # Check if annotation's area is large enough
                        # If not skip image altogether
                        if( useBoundingBoxArea ):
                            annArea = ann['bbox'][2]*ann['bbox'][3]
                        else:
                            annArea = ann['area']

                        if( annArea/imageArea > areaRatio ):
                            fullFileName = '%s/%s/%s/%s'%(output_data_dir, data_type, wakeword, img['file_name'])
                            break
                else:
                    fullFileName = '%s/%s/non_%s/%s'%(output_data_dir, data_type, wakeword, img['file_name'])

                # Resize and write only we didn't skip
                if( len(fullFileName) ):
                    I = resize(I, (imgSize, imgSize), anti_aliasing=True)
                    io.imsave(fullFileName, (255*I).astype(np.uint8), check_contrast=False)
                            
                # Show progress
                # index += 1
                # if( (index % 100) == 0 ):
                #     pbar.update(100)
                    #print(data_type + ', index=' + str(index))

    """ Download subsets and uncompress """
    def _download_and_uncompress(self, data_type, output_data_dir):

        # Parse data_type %[train/val/annotations]%year
        data_type = data_type.lower()
        tt = None
        if 'train' in data_type:
            tt = 'train'
        elif 'val' in data_type:
            tt = 'val'
        elif 'annotations' in data_type:
            tt = 'annotations'
        else:
            raise ValueError(f'Invalid data_type: {data_type}. Valid types are: train2014, val2014, test2014, annotations2014, train2017, val2017, test2017, annotations2017')

        yy = data_type.replace(tt,'')

        # Get appropriate urls
        url = self._urls[yy][tt.lower()]

        # Make sure we have write permission in this directory
        if not os.access(".", os.W_OK):
            raise ValueError("Cannot write to current directory. Aborting download of dataset")
        
        if not os.path.isdir(os.path.join(output_data_dir, data_type)):
            # Create output directory
            os.makedirs(output_data_dir, exist_ok=True)

            output_file = os.path.join(output_data_dir, f"{data_type}.zip")
            if not os.path.exists(output_file):
                utils.download_file(url, output_file)

            # Unzip the dataset
            with zipfile.ZipFile(output_file, 'r') as zip_ref:
                zip_ref.extractall(output_data_dir)
            
            if tt == 'annotations':
                os.rename(os.path.join(output_data_dir, 'annotations'), os.path.join(output_data_dir, data_type))
            
            # Remove zip file
            os.remove(output_file)

    """ Function to build the dataset """
    def build_dataset(self, datasets_dir = "./datasets", verbose=True, **kwargs):
        # Make sure we have write permission in this directory
        if not os.access(".", os.W_OK):
            raise ValueError("Cannot write to current directory. Aborting download of dataset")
        
        # Looking for people as wakeword, 2.5% area of image for a person, be careful with other parameters
        output_dataset_dir = os.path.join(datasets_dir, 'coco', 'vw_coco2014_96_2p5b')
        self._download_and_uncompress('train2014', output_dataset_dir)
        self._download_and_uncompress('annotations2014', output_dataset_dir)
        self._generate_coco_instance('train2014', output_dataset_dir, **self.parameters[self.problem_type])

        # In order to use coco, we use an ImageDataGenerator pointing to the data directory
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=.1,
            horizontal_flip=True,
            validation_split=0.1,
            rescale=1. / 255)

        train_generator = datagen.flow_from_directory(
            os.path.join(output_dataset_dir,'train2014'),
            target_size=(96, 96),
            batch_size=128,
            subset='training',
            color_mode='rgb')

        # Looking for people as wakeword, 2.5% area of image for a person, be careful with other parameters
        self._download_and_uncompress('val2014', output_dataset_dir)
        self._download_and_uncompress('annotations2014', output_dataset_dir)
        self._generate_coco_instance('val2014', output_dataset_dir, **self.parameters[self.problem_type])

        val_generator = datagen.flow_from_directory(
            os.path.join(output_dataset_dir,'val2014'),
            target_size=(96, 96),
            batch_size=128,
            subset='validation',
            color_mode='rgb')
        
        # Store the dataset and statistics
        dataset = {'train': train_generator, 'validation': val_generator}
        return dataset


""" Human Activity recognition dataset """
class UCI_HAR(Dataset):
    def __init__(self, quantizer: 'QuantizationScheme', verbose=True, **kwargs):
        """ Init super """
        super().__init__(quantizer, **kwargs)

        # Build data
        dataset = self.build_dataset(**kwargs)
        types = {'input': 'img', 'output': '1d'}

        # Now call super 
        super().build_dataset(dataset, types = types, **kwargs)
    
    def display_data(self, *args, **kwargs):
        print("[WARN] - Displaying data not implemented for UCI HAR dataset")

    def display_classes_distribution(self, *args, **kwargs):
        print("[WARN] - Displaying classes distribution not implemented for UCI HAR dataset")

    """ Build dataset """
    def build_dataset(self, datasets_dir = "./datasets", verbose=True, **kwargs):
        # Make sure we have write permission in this directory
        if not os.access(".", os.W_OK):
            raise ValueError("Cannot write to current directory. Aborting download of dataset")
        
        # Download and unzip dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
        output_dataset_dir = os.path.join(datasets_dir, 'HAR')
        if not os.path.isdir(output_dataset_dir):
            # Create output directory
            os.makedirs(output_dataset_dir, exist_ok=True)

            output_file = os.path.join(output_dataset_dir, f"UCI_HAR.zip")
            if not os.path.exists(output_file):
                utils.download_file(url, output_file)

            # Unzip the dataset
            with zipfile.ZipFile(output_file, 'r') as zip_ref:
                zip_ref.extractall(output_dataset_dir)
            
            os.rename(os.path.join(output_dataset_dir, 'UCI HAR Dataset'), os.path.join(output_dataset_dir, 'UCI_HAR'))
            
            # Remove zip file
            os.remove(output_file)

        # Train data 
        train_dir = os.path.join(output_dataset_dir, 'UCI_HAR', 'train')

        dats = ()
        for measure in ['total_acc', 'body_acc', 'body_gyro']:
            for axis in ['_x', '_y', '_z']:
                netsurf.utils.log._custom('DATA', 'Loading ' + measure + axis + '_train.txt')
                dats += (np.loadtxt(os.path.join(train_dir, 'Inertial Signals', measure + axis + '_train.txt')).astype('float16'),)

        XTrain = np.stack(dats, axis=2)

        y_train_raw = np.loadtxt(os.path.join(train_dir,'y_train.txt')).astype('int8')
        YTrain = keras.utils.to_categorical(y_train_raw-1)

        # Test data 
        test_dir = os.path.join(output_dataset_dir, 'UCI_HAR', 'test')

        dats = ()
        for measure in ['total_acc', 'body_acc', 'body_gyro']:
            for axis in ['_x', '_y', '_z']:
                netsurf.utils.log._custom('DATA', 'Loading', measure + axis + '_test.txt')
                dats += (np.loadtxt(os.path.join(test_dir, 'Inertial Signals', measure + axis + '_test.txt')).astype('float16'),)

        XTest = np.stack(dats, axis=2)

        y_test_raw = np.loadtxt(os.path.join(test_dir,'y_test.txt')).astype('int8')
        YTest = keras.utils.to_categorical(y_test_raw-1)

        if verbose:
            netsurf.utils.log._custom('DATA', f"UCI HAR Training set {XTrain.shape}, {YTrain.shape}")
            netsurf.utils.log._custom('DATA', f"UCI HAR Test set {XTest.shape}, {YTest.shape}")

        # Store the dataset and statistics
        self.dataset = {'train': (XTrain, YTrain), 'validation': (XTest, YTest)}
        self.stats = {'train': {'mean': np.nan, 'std': np.nan, 'num_classes': np.max(y_train_raw)}}



""" GTSDB dataset """
class GTSDB(Dataset):
    def __init__(self, quantizer: 'QuantizationScheme', **kwargs):
        """ Init super """
        super().__init__(quantizer, **kwargs)
        # url
        self.url = "https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip"

        # Build data
        dataset = self.build_dataset(**kwargs)
        types = {'input': 'img', 'output': '1d'}

        # Now call super 
        super().build_dataset(dataset, types = types, **kwargs)


    """ Download subsets and uncompress """
    def _download_and_uncompress(self, output_data_dir):

        # Make sure we have write permission in this directory
        if not os.access(".", os.W_OK):
            raise ValueError("Cannot write to current directory. Aborting download of dataset")
        
        if not os.path.isdir(output_data_dir):
            # Create output directory
            os.makedirs(output_data_dir, exist_ok=True)

            output_file = os.path.join(output_data_dir, "GTSDB.zip")
            if not os.path.exists(output_file):
                utils.download_file(self.url, output_file)

            # Unzip the dataset
            with zipfile.ZipFile(output_file, 'r') as zip_ref:
                zip_ref.extractall(output_data_dir)
            
            # Remove zip file
            os.remove(output_file)

            os.rename(os.path.join(output_data_dir, 'FullIJCNN2013'), os.path.join(output_data_dir, 'full'))

    """ Function to build the dataset """
    def build_dataset(self, datasets_dir = "./datasets", verbose=True, **kwargs):
        # Make sure we have write permission in this directory
        if not os.access(".", os.W_OK):
            raise ValueError("Cannot write to current directory. Aborting download of dataset")
        
        """ Open cv """
        import cv2

        # Looking for people as wakeword, 2.5% area of image for a person, be careful with other parameters
        output_dataset_dir = os.path.join(datasets_dir, 'GTSBR')
        self._download_and_uncompress(output_dataset_dir)

        classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }

        # #ImgNo#.ppm;#leftCol#;##topRow#;#rightCol#;#bottomRow#;#ClassID#
        # read file gt.txt
        with open(os.path.join(output_dataset_dir, 'full', 'gt.txt')) as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            lines = [line.split(';') for line in lines]
            lines = np.array(lines)
            df = pd.DataFrame(lines, columns = ['ImgNo','leftCol','topRow','rightCol','bottomRow','classID'])

        # Load images
        images = []
        labels = []

        for i, row in tqdm(df.iterrows(), total = len(df), desc = "Loading GTSDB images"):
            img = cv2.imread(os.path.join(output_dataset_dir, 'full', row['ImgNo']))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Crop image to region of interest
            roc = list(row[1:].to_numpy().astype(int))
            img = img[roc[1]:roc[3], roc[0]:roc[2]]
            img = cv2.resize(img, (50, 50), interpolation = cv2.INTER_LINEAR)
            img = img / 255.0
            images.append(img)
            labels.append(roc)
        
        # Convert to numpy
        images = np.transpose(np.stack(images, axis = 3),(3,0,1,2))
        labels = np.array(labels)
        # Labels to one hot
        labels = keras.utils.to_categorical(labels[:,-1], num_classes = len(classes))
        
        # Split data into training and validation
        ntrain = int(0.8*images.shape[0])
        XTrain, YTrain = images[:ntrain], labels[:ntrain]
        XVal, YVal = images[ntrain:], labels[ntrain:]

        # Store the dataset and statistics
        dataset = {'train': (XTrain, YTrain), 'validation': (XVal, YVal)}
        self.classes = classes

        return dataset



""" AutoMPG dataset """
class AutoMPG(Dataset):
    def __init__(self, quantizer: 'QuantizationScheme', **kwargs):
        """ Init super """
        super().__init__(quantizer, **kwargs)

        # Build data
        dataset = self.build_dataset(**kwargs)
        types = {'input': '2d', 'output': '1d'}

        # Now call super 
        super().build_dataset(dataset, types = types, **kwargs)
    
    def build_dataset(self, datasets_dir = "./datasets", verbose=True, **kwargs):
        # Make sure we have write permission in this directory
        if not os.access(".", os.W_OK):
            raise ValueError("Cannot write to current directory. Aborting download of dataset")
        
        # Download and unzip dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
        output_dataset_dir = os.path.join(datasets_dir, 'AutoMPG')
        output_file = os.path.join(output_dataset_dir, f"auto-mpg.data")
        if not os.path.isdir(output_dataset_dir):
            # Create output directory
            os.makedirs(output_dataset_dir, exist_ok=True)

            if not os.path.exists(output_file):
                utils.download_file(url, output_file)

        """ Directly obtained from https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/regression.ipynb#scrollTo=CiX2FI4gZtTt """
        column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

        raw_dataset = pd.read_csv(output_file, names=column_names,
                                na_values='?', comment='\t',
                                sep=' ', skipinitialspace=True)
        
        dataset = raw_dataset.copy()
        # Drop nan
        dataset = dataset.dropna()

        # The "Origin" column is categorical, not numeric. So the next step is to one-hot encode the values in the column with pd.get_dummies.
        # Neglecting to specify a data type by way of a dtype argument will leave you with boolean values, causing errors during normalization when instantiating the Tensor object if the feature values are not cast to a uniform type when passing the array into tf.keras.layers.Normalization.adapt(). Tensor objects must house uniform data types.
        # You can set up the tf.keras.Model to do this kind of transformation for you but that's beyond the scope of this tutorial. Check out the Classify structured data using Keras preprocessing layers or Load CSV data tutorials for examples.
        dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'}).astype(str)
        dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='', dtype=float)

        # We can store this into df 
        self.df = dataset

        train_dataset = dataset.sample(frac=0.8, random_state=0)
        test_dataset = dataset.drop(train_dataset.index)


        XTrain = train_dataset.copy()
        XVal = test_dataset.copy()

        YTrain = XTrain.pop('MPG')
        YVal = XVal.pop('MPG')

        # Store the dataset and statistics
        dataset = {'train': (XTrain, YTrain), 'validation': (XVal, YVal)}
        return dataset

    def display_data(self, *args, **kwargs):
        train_dataset = self.df
        sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
    
    def display_classes_distribution(self, *args, **kwargs):
        print("[WARN] - Displaying classes distribution not implemented for AutoMPG dataset")


""" HGCal dataset """
class HGCal(Dataset):
    def __init__(self, quantizer: 'QuantizationScheme', extra_args = {}, **kwargs):
        """ Init super """
        super().__init__(quantizer, **kwargs)

        # Set flags
        self.extra_args = extra_args

        self.params = netsurf.dnn.econ.EconParams()
        
        # Build data
        dataset = self.build_dataset(**kwargs)
        types = {'input': 'img', 'output': 'img'}

        # Now call super 
        super().build_dataset(dataset, types = types, **kwargs)

    def get_figsize(self, nrows, ncols):
        return (3*nrows, 3*ncols)
    
    """ Function to build the dataset """
    def build_dataset(self, datasets_dir = "./datasets", extra_args = {}, **kwargs):
        
        # Make sure the datset exists 
        data_dir = os.path.join(datasets_dir, 'hgcal22data', 'hgcal22data_signal_driven_ttbar_v11', 'nElinks_5')
        if not os.path.isdir(data_dir):
            raise ValueError("[ERR] - HGCal dataset not found. Aborting.")        

        # Load data
        (data_values,phys_values) = self.load_data(inputFile = data_dir, **extra_args, **kwargs)

        # normalize 
        normdata, maxdata, sumdata = self.normalize_data(data_values, **extra_args)
        maxdata = maxdata / 35.0  # normalize to units of transverse MIPs
        sumdata = sumdata / 35.0  # normalize to units of transverse MIPs

        # Too much data, take a subset cause kernel is crashing everytime
        normdata = normdata[:100000]
        maxdata = maxdata[:100000]
        sumdata = sumdata[:100000]

        # Preprocess data into image 
        normdata = self.preprocess_input(normdata, in_shape = self.params.shape)

        # Split data into training and validation
        ntrain = int(0.8*normdata.shape[0])
        XTrain, YTrain = normdata[:ntrain], normdata[:ntrain]
        XVal, YVal = normdata[ntrain:], normdata[ntrain:]

        # Same for phys values 
        XTrainPhys, YTrainPhys = phys_values[:ntrain], phys_values[:ntrain]
        XValPhys, YValPhys = phys_values[ntrain:], phys_values[ntrain:]

        # same for maxdata and sum data
        XTrainMax, YTrainMax = maxdata[:ntrain], maxdata[:ntrain]
        XValMax, YValMax = maxdata[ntrain:], maxdata[ntrain:]

        XTrainSum, YTrainSum = sumdata[:ntrain], sumdata[:ntrain]
        XValSum, YValSum = sumdata[ntrain:], sumdata[ntrain:]

        # Store the dataset and statistics
        dataset = {'train': (XTrain, YTrain), 'validation': (XVal, YVal)}
        self.stats = {'train': {'mean': np.nan, 'std': np.nan, 'num_classes': 0}}
        self.phys_dataset = {'train': (XTrainPhys, YTrainPhys), 'validation': (XValPhys, YValPhys)}
        self.max_dataset = {'train': (XTrainMax, YTrainMax), 'validation': (XValMax, YValMax)}
        self.sum_dataset = {'train': (XTrainSum, YTrainSum), 'validation': (XValSum, YValSum)}

        return dataset

    def normalize_data(self, data_values, rescaleInputToMax=False, **kwargs):
        # normalize input charge data rescaleInputToMax: normalizes charges to
        # maximum charge in module sumlog2 (default): normalizes charges to
        # 2**floor(log2(sum of charge in module)) where floor is the largest scalar
        # integer: i.e. normalizes to MSB of the sum of charges (MSB here is the
        # most significant bit) rescaleSum: normalizes charges to sum of charge in
        # module
        norm_data, max_data, sum_data = self._normalize(
            data_values.copy(), rescaleInputToMax=rescaleInputToMax, sumlog2=True
        )

        return norm_data, max_data, sum_data

    def _normalize(self, data, rescaleInputToMax=False, sumlog2=True):
        maxes =[]
        sums =[]
        sums_log2=[]
        for i in range(len(data)):
            maxes.append( data[i].max() )
            sums.append( data[i].sum() )
            sums_log2.append( 2**(np.floor(np.log2(data[i].sum()))) )
            if sumlog2:
                data[i] = 1.*data[i]/(sums_log2[-1] if sums_log2[-1] else 1.)
            elif rescaleInputToMax:
                data[i] = 1.*data[i]/(data[i].max() if data[i].max() else 1.)
            else:
                data[i] = 1.*data[i]/(data[i].sum() if data[i].sum() else 1.)
        if sumlog2:
            return  data,np.array(maxes),np.array(sums_log2)
        else:
            return data,np.array(maxes),np.array(sums)

    def unnormalize(self, norm_data,maxvals,rescaleOutputToMax=False, sumlog2=True):
        for i in range(len(norm_data)):
            if rescaleOutputToMax:
                norm_data[i] =  norm_data[i] * maxvals[i] / (norm_data[i].max() if norm_data[i].max() else 1.)
            else:
                if sumlog2:
                    sumlog2 = 2**(np.floor(np.log2(norm_data[i].sum())))
                    norm_data[i] =  norm_data[i] * maxvals[i] / (sumlog2 if sumlog2 else 1.)
                else:
                    norm_data[i] =  norm_data[i] * maxvals[i] / (norm_data[i].sum() if norm_data[i].sum() else 1.)
        return norm_data

    """ Method to preprocess input data (gets evaluated at fit/evaluate time) """
    def preprocess_input(self, x, in_shape = None):
        shape = self.params.shape if in_shape is None else in_shape

        if len(self.params.array)>0:
            arrange = self.params.array
            x = x[:,arrange]
        if len(self.params.arrMask)>0:
            arrMask = self.params.arrMask
            x[:,arrMask==0]=0  #zeros out repeated entries

        shaped_data = x.reshape(len(x),shape[0],shape[1],shape[2])

        if self.params.n_copy>0:
            n_copy  = self.params.n_copy
            occ_low = self.params.occ_low
            occ_hi = self.params.occ_hi
            shaped_data = self.cloneInput(shaped_data,n_copy,occ_low,occ_hi)
        
        return shaped_data
    
    def preprocess_output(self, y):
        return self.preprocess_input(y)

    # This was extracted directly from: https://github.com/oliviaweng/fastml-science/blob/quantized-autoencoder/sensor-data-compression/train.py#L139
    def load_data(self, inputFile = None, AEonly = 1, nELinks = 5, models = '8x8_c8_S2_tele_fqK_6bit', 
                    nrowsPerFile = 4500000, noHeader = True, num_val_inputs = 512, 
                    evalOnly = False, rescaleInputToMax = False, rescaleOutputToMax = False,
                    maskPartials = False, maskEnergies = False, saveEnergy = False, double = False,
                    occReweight = False,
                    **kwargs):
        # charge data headers of 48 Input Trigger Cells (TC) 
        CALQ_COLS = ['CALQ_%i'%c for c in range(0, 48)]
        
        #Keep track of phys data
        COORD_COLS=['tc_eta','tc_phi']
        
        def mask_data(data, maskPartials, maskEnergies):
            # mask rows where occupancy is zero
            mask_occupancy = (data[CALQ_COLS].astype('float64').sum(axis=1) != 0)
            data = data[mask_occupancy]
            
            if maskPartials:
                mask_isFullModule = np.isin(data.ModType.values,['FI','FM','FO'])
                #_logger.info('Mask partial modules from input dataset')
                data = data[mask_isFull]
            if maskEnergies:
                try:
                    mask_energy = data['SimEnergyFraction'].astype('float64') > 0.05
                    data = data[mask_energy]
                except:
                    #_logger.warning('No SimEnergyFraction array in input data')
                    pass
            return data
        
        if os.path.isdir(inputFile):
            df_arr = []
            phy_arr=[]
            for infile in os.listdir(inputFile):
                if os.path.isdir(inputFile+infile): continue
                infile = os.path.join(inputFile,infile)
                if noHeader:
                    df_arr.append(pd.read_csv(infile, dtype=np.float64, header=0, nrows = nrowsPerFile, usecols=[*range(0,48)], names=CALQ_COLS))
                    phy_arr.append(pd.read_csv(infile, dtype=np.float64, header=0, nrows = nrowsPerFile, usecols=[*range(55,57)], names=COORD_COLS))
                else:
                    df_arr.append(pd.read_csv(infile, nrows=nrowsPerFile))
            data = pd.concat(df_arr)
            phys = pd.concat(phy_arr)
        else:
            data = pd.read_csv(inputFile, nrows=nrowsPerFile)
        
        data = mask_data(data, maskPartials, maskEnergies)

        if saveEnergy:
            try:
                simEnergyFraction = data['SimEnergyFraction'].astype('float64') # module simEnergyFraction w. respect to total event's energy
                simEnergy = data['SimEnergyTotal'].astype('float64') # module simEnergy
                simEnergyEvent = data['EventSimEnergyTotal'].astype('float64') # event simEnergy
            except:
                simEnergyFraction = None
                simEnergy = None
                simEnergyEvent = None
                #_logger.warning('No SimEnergyFraction or SimEnergyTotal or EventSimEnergyTotal arrays in input data')

        data = data[CALQ_COLS].astype('float64')
        phys = phys[COORD_COLS]
        data_values = data.values
        phys_values = phys.values
        #_logger.info('Input data shape')
        #print(data.shape)
        #data.describe()

        # duplicate data (e.g. for PU400?)
        if double:
            def double_data(data):
                doubled=[]
                i=0
                while i<= len(data)-2:
                    doubled.append( data[i] + data[i+1] )
                    i+=2
                return np.array(doubled)
            doubled_data = double_data(data_values.copy())
            #_logger.info('Duplicated the data, the new shape is:')
            #print(doubled_data.shape)
            data_values = doubled_data

        return (data_values,phys_values)
    
    """ Display functions """
    def display_data(self, *args, filename = None, **kwargs):

        occupancy_all = {subset: np.count_nonzero(self.dataset[subset][0],axis=1) for subset in self.dataset.keys()}
        occupancy_all_1MT = {subset: np.count_nonzero(self.dataset[subset][0]>35,axis=1) for subset in self.dataset.keys()}
        
        for subset in self.dataset.keys():
            
            self.plot_hist(occupancy_all[subset].flatten(), 
                    title = f"occ_all_{subset}", xlabel = "occupancy (all cells)", ylabel = "evts",
                    stats = False, logy = True, nbins = 50, lims = [0,50], filename = filename, **kwargs)
            self.plot_hist(occupancy_all_1MT[subset].flatten(), 
                    title = f"occ_1MT_{subset}", xlabel = r"occupancy (1 MIP$_{\mathrm{T}}$ cells)", ylabel = "evts",
                    stats = False, logy = True, nbins = 50, lims = [0,50], filename = filename, **kwargs)
            self.plot_hist(np.log10(self.max_dataset[subset][0].flatten()), 
                    title = f"maxQ_all_{subset}", xlabel = self.params.logMaxTitle, ylabel = "evts",
                    stats = False, logy = True, nbins = 50, lims = [0,2.5], filename = filename, **kwargs)
            self.plot_hist(np.log10(self.sum_dataset[subset][0].flatten()), 
                    title = f"sumQ_all_{subset}", xlabel = self.params.logTotTitle, ylabel = "evts",
                    stats = False, logy = True, nbins = 50, lims = [0,2.5], filename = filename, **kwargs)
    
    def display_classes_distribution(self, *args, **kwargs):
        print("[WARN] - Displaying classes distribution not implemented for HGCal dataset")

    # Taken from: https://github.com/oliviaweng/fastml-science/blob/main/sensor-data-compression/utils/plot.py#L14
    def plot_hist(self, vals, subset = 'train', title = None, random = False, show = True, filename = None, 
                    xlabel = "", ylabel = "", nbins = 40, lims = None, stats = True, logy = False, 
                    leg = None, overwrite = False, **kwargs):

        assert(subset in list(self.dataset.keys()) )
        
        if filename is not None:
            if '.png' in filename:
                # Add subset
                filename = filename.replace('.png', f'{title}_{subset}.png')

        # Check if exists 
        if os.path.isfile(filename) and not overwrite:
            netsurf.utils.log._custom('DATA', f'File {filename} already exists. Skipping')
            return

        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(6,4))
        if leg:
            n, bins, patches = ax.hist(vals, nbins, range=lims, label=leg)
        else:
            n, bins, patches = ax.hist(vals, nbins, range=lims)

        plt.text(0.1, 0.9, title, transform=ax.transAxes)
        if stats:
            mu = np.mean(vals)
            std = np.std(vals)
            plt.text(0.1, 0.8, r'$\mu=%.3f,\ \sigma=%.3f$'%(mu,std),transform=ax.transAxes)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel if ylabel else 'Entries')
        if logy: 
            ax.set_yscale('log')
        
        ax.grid(True)
        
        if title is not None:
            fig.suptitle(title)
        plt.tight_layout()

        if filename is not None:
            fig.savefig(filename)
            netsurf.utils.log._custom('DATA', f'Saved dataset sample to {filename}')

            # close fig
            plt.close(fig)

        if show:
            plt.show()


""" Smart Pixel dataset """
class SmartPixel(Dataset):
    def __init__(self, quantizer: 'QuantizationScheme', **kwargs):
        """ Init super """
        super().__init__(quantizer, **kwargs)

        # Build data
        dataset = self.build_dataset(**kwargs)
        types = {'input': '1d', 'output': 'class'}

        # Now call super 
        super().build_dataset(dataset, types = types, **kwargs)
    
    """ Function to build the dataset """
    def build_dataset(self, datasets_dir = "./datasets", verbose=True, **kwargs):
        # Link to dataset
        url = "https://cseweb.ucsd.edu/~oweng/smart_pixel_dataset/ds8_only.tar.gz"
        output_dataset_dir = os.path.join(datasets_dir, 'SmartPixel')
        output_file = os.path.join(output_dataset_dir, f"ds8_only.tar.gz")
        if not os.path.isdir(output_dataset_dir):
            # Create output directory
            os.makedirs(output_dataset_dir, exist_ok=True)

            if not os.path.exists(output_file):
                utils.download_file(url, output_file)

            # Unzip the dataset
            with tarfile.open(output_file, 'r:gz') as tar:
                tar.extractall(output_dataset_dir)

            # Remove zip file
            os.remove(output_file)
        
        # Load data
        data_dir = os.path.join(output_dataset_dir, 'ds8_only')
        
        # Normalize
        local_id=0
        train_data = os.path.join(data_dir,"dec6_ds8_quant",f"QuantizedInputTrainSetLocal{local_id}.csv")
        train_label = os.path.join(data_dir,"dec6_ds8_quant",f"TrainSetLabelLocal{local_id}.csv")
        test_data = os.path.join(data_dir,"dec6_ds8_quant",f"QuantizedInputTestSetLocal{local_id}.csv")
        test_label = os.path.join(data_dir,"dec6_ds8_quant",f"TestSetLabelLocal{local_id}.csv")

        df1 = pd.read_csv(train_data)
        df2 = pd.read_csv(train_label)
        df3 = pd.read_csv(test_data)
        df4 = pd.read_csv(test_label)

        # Convert df4 to one-hot
        df2 = 1.0*pd.get_dummies(df2, columns = ['ptLabel'])
        df4 = 1.0*pd.get_dummies(df4, columns = ['ptLabel'])

        XTrain = df1.values
        XVal = df3.values
        YTrain = df2.values
        YVal = df4.values

        # Store column values 
        self.columns = {'x': df1.columns, 'y': df2.columns}

        # Set everything in place 
        dataset = {'train': (XTrain, YTrain), 'validation': (XVal, YVal)}
        return dataset
    
    def display_data(self, *args, subset = 'training', title = 'SmartPixel data', show = True, 
                     filename = None, num_samples = 1000, overwrite = False, **kwargs):
        
        if filename is not None:
            if '.png' in filename:
                # Add subset
                filename = filename.replace('.png', f'{title}_{subset}.png')

        if os.path.isfile(filename) and not overwrite:
            netsurf.utils.log._custom('DATA', f'File {filename} already exists. Skipping')
            return
        
        Z = self.dataset[subset]

        # Convert Z[1] from one_hot to sparse categorical
        Z = (Z[0], np.argmax(Z[1], axis = 1)[:,None])

        # Cat 
        Z = np.concatenate(Z, axis = 1)

        # Convert to pandas 
        df = pd.DataFrame(Z, columns = list(self.columns['x']) + ['ptLabel'])

        plot = sns.pairplot(df[:num_samples], hue = 'ptLabel', corner = True)
        
        if filename is not None:
            plot.savefig(filename)
            netsurf.utils.log._custom('DATA', f'Saved dataset sample to {filename}')

            # Now close 
            plt.close(plot.fig)

        if show:
            plt.show()
    



""" Build KeywordSpotting Dataset """
class KeywordSpotting(Dataset):
    def __init__(self, quantizer: 'QuantizationScheme', **kwargs):
        """ Init super """
        super().__init__(quantizer, **kwargs)

        # Build data
        dataset = self.build_dataset(**kwargs)
        types = {'input': '3d', 'output': '2d'}

        # Now call super 
        super().build_dataset(dataset, types = types, **kwargs)
        
    def cast_and_pad(self, sample_dict):
        audio = sample_dict['audio']
        label = sample_dict['label']
        paddings = [[0, 16000-tf.shape(audio)[0]]]
        audio = tf.pad(audio, paddings)
        audio16 = tf.cast(audio, 'int16')
        return audio16, label
    
    def get_preprocess_audio_func(self, model_settings, is_training=False, background_data = []):
        def prepare_processing_graph(next_element):
            """Builds a TensorFlow graph to apply the input distortions.
            Creates a graph that loads a WAVE file, decodes it, scales the volume,
            shifts it in time, adds in background noise, calculates a spectrogram, and
            then builds an MFCC fingerprint from that.
            This must be called with an active TensorFlow session running, and it
            creates multiple placeholder inputs, and one output:
            - wav_filename_placeholder_: Filename of the WAV to load.
            - foreground_volume_placeholder_: How loud the main clip should be.
            - time_shift_padding_placeholder_: Where to pad the clip.
            - time_shift_offset_placeholder_: How much to move the clip in time.
            - background_data_placeholder_: PCM sample data for background noise.
            - background_volume_placeholder_: Loudness of mixed-in background.
            - mfcc_: Output 2D fingerprint of processed audio.
            Args:
            model_settings: Information about the current model being trained.
            """
            desired_samples = model_settings['desired_samples']
            background_frequency = model_settings['background_frequency']
            background_volume_range_= model_settings['background_volume_range_']

            wav_decoder = tf.cast(next_element['audio'], tf.float32)
            if model_settings['feature_type'] != "td_samples":
                wav_decoder = wav_decoder/tf.reduce_max(wav_decoder)
            else:
                wav_decoder = wav_decoder/tf.constant(2**15,dtype=tf.float32)
            #Previously, decode_wav was used with desired_samples as the length of array. The
            # default option of this function was to pad zeros if the desired samples are not found
            wav_decoder = tf.pad(wav_decoder,[[0,desired_samples-tf.shape(wav_decoder)[-1]]]) 
            # Allow the audio sample's volume to be adjusted.
            foreground_volume_placeholder_ = tf.constant(1,dtype=tf.float32)
            
            scaled_foreground = tf.multiply(wav_decoder,
                                            foreground_volume_placeholder_)
            # Shift the sample's start position, and pad any gaps with zeros.
            time_shift_padding_placeholder_ = tf.constant([[2,2]], tf.int32)
            time_shift_offset_placeholder_ = tf.constant([2],tf.int32)
            scaled_foreground.shape
            padded_foreground = tf.pad(scaled_foreground, time_shift_padding_placeholder_, mode='CONSTANT')
            sliced_foreground = tf.slice(padded_foreground, time_shift_offset_placeholder_, [desired_samples])
        
            if is_training and background_data != []:
                background_volume_range = tf.constant(background_volume_range_,dtype=tf.float32)
                background_index = np.random.randint(len(background_data))
                background_samples = background_data[background_index]
                background_offset = np.random.randint(0, len(background_samples) - desired_samples)
                background_clipped = background_samples[background_offset:(background_offset + desired_samples)]
                background_clipped = tf.squeeze(background_clipped)
                background_reshaped = tf.pad(background_clipped,[[0,desired_samples-tf.shape(wav_decoder)[-1]]])
                background_reshaped = tf.cast(background_reshaped, tf.float32)
                if np.random.uniform(0, 1) < background_frequency:
                    background_volume = np.random.uniform(0, background_volume_range_)
                else:
                    background_volume = 0
                background_volume_placeholder_ = tf.constant(background_volume,dtype=tf.float32)
                background_data_placeholder_ = background_reshaped
                background_mul = tf.multiply(background_data_placeholder_,
                                    background_volume_placeholder_)
                background_add = tf.add(background_mul, sliced_foreground)
                sliced_foreground = tf.clip_by_value(background_add, -1.0, 1.0)
            
            if model_settings['feature_type'] == 'mfcc':
                stfts = tf.signal.stft(sliced_foreground, frame_length=model_settings['window_size_samples'], 
                                    frame_step=model_settings['window_stride_samples'], fft_length=None,
                                    window_fn=tf.signal.hann_window
                                    )
                spectrograms = tf.abs(stfts)
                num_spectrogram_bins = stfts.shape[-1]
                # default values used by contrib_audio.mfcc as shown here
                # https://kite.com/python/docs/tensorflow.contrib.slim.rev_block_lib.contrib_framework_ops.audio_ops.mfcc
                lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20.0, 4000.0, 40 
                linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix( num_mel_bins, num_spectrogram_bins,
                                                                                    model_settings['sample_rate'],
                                                                                    lower_edge_hertz, upper_edge_hertz)
                mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
                mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
                # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
                log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
                # Compute MFCCs from log_mel_spectrograms and take the first 13.
                mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :model_settings['dct_coefficient_count']]
                mfccs = tf.reshape(mfccs,[model_settings['spectrogram_length'], model_settings['dct_coefficient_count'], 1])
                next_element['audio'] = mfccs
                #next_element['label'] = tf.one_hot(next_element['label'],12)

            elif model_settings['feature_type'] == 'lfbe':
                # apply preemphasis
                preemphasis_coef = 1 - 2 ** -5
                power_offset = 52
                num_mel_bins = model_settings['dct_coefficient_count']
                paddings = tf.constant([[0, 0], [1, 0]])
                # for some reason, tf.pad only works with the extra batch dimension, but then we remove it after pad
                sliced_foreground = tf.expand_dims(sliced_foreground, 0)
                sliced_foreground = tf.pad(tensor=sliced_foreground, paddings=paddings, mode='CONSTANT')
                sliced_foreground = sliced_foreground[:, 1:] - preemphasis_coef * sliced_foreground[:, :-1]
                sliced_foreground = tf.squeeze(sliced_foreground) 
                # compute fft
                stfts = tf.signal.stft(sliced_foreground,  frame_length=model_settings['window_size_samples'], 
                                        frame_step=model_settings['window_stride_samples'], fft_length=None,
                                        window_fn=functools.partial(
                                        tf.signal.hamming_window, periodic=False),
                                        pad_end=False,
                                        name='STFT')
                
                # compute magnitude spectrum [batch_size, num_frames, NFFT]
                magspec = tf.abs(stfts)
                num_spectrogram_bins = magspec.shape[-1]
                
                # compute power spectrum [num_frames, NFFT]
                powspec = (1 / model_settings['window_size_samples']) * tf.square(magspec)
                powspec_max = tf.reduce_max(input_tensor=powspec)
                powspec = tf.clip_by_value(powspec, 1e-30, powspec_max) # prevent -infinity on log
                
                def log10(x):
                    # Compute log base 10 on the tensorflow graph.
                    # x is a tensor.  returns log10(x) as a tensor
                    numerator = tf.math.log(x)
                    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
                    return numerator / denominator
                
                # Warp the linear-scale, magnitude spectrograms into the mel-scale.
                lower_edge_hertz, upper_edge_hertz = 0.0, model_settings['sample_rate'] / 2.0
                linear_to_mel_weight_matrix = (
                    tf.signal.linear_to_mel_weight_matrix(
                        num_mel_bins=num_mel_bins,
                        num_spectrogram_bins=num_spectrogram_bins,
                        sample_rate=model_settings['sample_rate'],
                        lower_edge_hertz=lower_edge_hertz,
                        upper_edge_hertz=upper_edge_hertz))

                mel_spectrograms = tf.tensordot(powspec, linear_to_mel_weight_matrix,1)
                mel_spectrograms.set_shape(magspec.shape[:-1].concatenate(
                    linear_to_mel_weight_matrix.shape[-1:]))

                log_mel_spec = 10 * log10(mel_spectrograms)
                log_mel_spec = tf.expand_dims(log_mel_spec, -1, name="mel_spec")
                
                log_mel_spec = (log_mel_spec + power_offset - 32 + 32.0) / 64.0
                log_mel_spec = tf.clip_by_value(log_mel_spec, 0, 1)

                next_element['audio'] = log_mel_spec

            elif model_settings['feature_type'] == 'td_samples':
                ## sliced_foreground should have the right data.  Make sure it's the right format (int16)
                # and just return it.
                paddings = [[0, 16000-tf.shape(sliced_foreground)[0]]]
                wav_padded = tf.pad(sliced_foreground, paddings)
                wav_padded = tf.expand_dims(wav_padded, -1)
                wav_padded = tf.expand_dims(wav_padded, -1)
                next_element['audio'] = wav_padded
                
            return next_element
        
        return prepare_processing_graph

    """ Function to build background data """
    def prepare_background_data(self, bg_path, BACKGROUND_NOISE_DIR_NAME):
        """Searches a folder for background noise audio, and loads it into memory.
        It's expected that the background audio samples will be in a subdirectory
        named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
        the sample rate of the training data, but can be much longer in duration.
        If the '_background_noise_' folder doesn't exist at all, this isn't an
        error, it's just taken to mean that no background noise augmentation should
        be used. If the folder does exist, but it's empty, that's treated as an
        error.
        Returns:
            List of raw PCM-encoded audio samples of background noise.
        Raises:
            Exception: If files aren't found in the folder.
        """
        background_data = []
        background_dir = os.path.join(bg_path, BACKGROUND_NOISE_DIR_NAME)
        if not os.path.exists(background_dir):
            return background_data
        #with tf.Session(graph=tf.Graph()) as sess:
        #    wav_filename_placeholder = tf.placeholder(tf.string, [])
        #    wav_loader = io_ops.read_file(wav_filename_placeholder)
        #    wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
        search_path = os.path.join(bg_path, BACKGROUND_NOISE_DIR_NAME,'*.wav')
        #for wav_path in gfile.Glob(search_path):
        #    wav_data = sess.run(wav_decoder, feed_dict={wav_filename_placeholder: wav_path}).audio.flatten()
        #    self.background_data.append(wav_data)
        for wav_path in gfile.Glob(search_path):
            #audio = tfio.audio.AudioIOTensor(wav_path)
            raw_audio = tf.io.read_file(wav_path)
            audio = tf.audio.decode_wav(raw_audio)
            background_data.append(audio[0])
        if not background_data:
            raise Exception('No background wav files were found in ' + search_path)
        return background_data

    def convert_dataset(self, item):
        """Puts the mnist dataset in the format Keras expects, (features, labels)."""
        audio = item['audio']
        label = item['label']
        return audio, label

    """ Function to build the dataset """
    def build_dataset(self, get_waves=False, val_cal_subset = False, verbose = True, **kwargs):


        bg_path = self.datasets_dir
        background_volume = 0.1
        background_frequency = 0.8
        silence_percentage = 10.0
        unknown_percentage = 10.0
        time_shift_ms = 100.0
        sample_rate = 16000
        clip_duration_ms = 1000
        window_size_ms = 30.0
        window_stride_ms = 20.0
        feature_type = "mfcc" #["mfcc", "lfbe", "td_samples"]
        dct_coefficient_count = 10
        num_train_samples = -1
        num_val_samples = -1
        num_test_samples = -1
        num_bin_files = 1000
        bin_file_path = os.path.join(os.getenv('HOME'), 'kws_test_files')
        label_count=12
        batch_size = 100

        desired_samples = int(sample_rate * clip_duration_ms / 1000)
        if feature_type == 'td_samples':
            window_size_samples = 1
            spectrogram_length = desired_samples
            dct_coefficient_count = 1
            window_stride_samples = 1
            fingerprint_size = desired_samples
        else:
            window_size_samples = int(sample_rate * window_size_ms / 1000)
            window_stride_samples = int(sample_rate * window_stride_ms / 1000)
            length_minus_window = (desired_samples - window_size_samples)
            if length_minus_window < 0:
                spectrogram_length = 0
            else:
                spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
                fingerprint_size = dct_coefficient_count * spectrogram_length
        
        model_settings = {
            'desired_samples': desired_samples,
            'window_size_samples': window_size_samples,
            'window_stride_samples': window_stride_samples,
            'feature_type': feature_type, 
            'spectrogram_length': spectrogram_length,
            'dct_coefficient_count': dct_coefficient_count,
            'fingerprint_size': fingerprint_size,
            'label_count': label_count,
            'sample_rate': sample_rate,
            'background_frequency': 0.8, # args.background_frequency
            'background_volume_range_': 0.1
        }


        splits = ['train', 'test', 'validation']
        (ds_train, ds_test, ds_val), ds_info = tfds.load('speech_commands', split = splits, data_dir = self.datasets_dir, with_info = True)

        BACKGROUND_NOISE_DIR_NAME='_background_noise_' 
        background_data = self.prepare_background_data(bg_path, BACKGROUND_NOISE_DIR_NAME)


        if val_cal_subset:  # only return the subset of val set used for quantization calibration
            with open("quant_cal_idxs.txt") as fpi:
                cal_indices = [int(line) for line in fpi]
            cal_indices.sort()
            # cal_indices are the positions of specific inputs that are selected to calibrate the quantization
            count = 0  # count will be the index into the validation set.
            val_sub_audio = []
            val_sub_labels = []
            for d in ds_val:
                if count in cal_indices:          # this is one of the calibration inpus
                    new_audio = d['audio'].numpy()  # so add it to a stack of tensors 
                    if len(new_audio) < 16000:      # from_tensor_slices doesn't work for ragged tensors, so pad to 16k
                        new_audio = np.pad(new_audio, (0, 16000-len(new_audio)), 'constant')
                    val_sub_audio.append(new_audio)
                    val_sub_labels.append(d['label'].numpy())
                count += 1
            # and create a new dataset for just the calibration inputs.
            ds_val = tf.data.Dataset.from_tensor_slices({"audio": val_sub_audio,
                                                        "label": val_sub_labels})
        
        if num_train_samples != -1:
            ds_train = ds_train.take(num_train_samples)
        if num_val_samples != -1:
            ds_val = ds_val.take(num_val_samples)
        if num_test_samples != -1:
            ds_test = ds_test.take(num_test_samples)
        
        if get_waves:
            ds_train = ds_train.map(self.cast_and_pad)
            ds_test  =  ds_test.map(self.cast_and_pad)
            ds_val   =   ds_val.map(self.cast_and_pad)
        else:
            # extract spectral features and add background noise
            ds_train = ds_train.map(self.get_preprocess_audio_func(model_settings, is_training = True,
                                                            background_data = background_data),
                                    num_parallel_calls = tf.data.experimental.AUTOTUNE)
            ds_test = ds_test.map(self.get_preprocess_audio_func(model_settings, is_training = False,
                                                            background_data = background_data),
                                    num_parallel_calls = tf.data.experimental.AUTOTUNE)
            ds_val = ds_val.map(self.get_preprocess_audio_func(model_settings, is_training = False,
                                                            background_data = background_data),
                                    num_parallel_calls = tf.data.experimental.AUTOTUNE)
            # change output from a dictionary to a feature,label tuple
            ds_train = ds_train.map(self.convert_dataset)
            ds_test = ds_test.map(self.convert_dataset)
            ds_val = ds_val.map(self.convert_dataset)
        
        # Now that we've acquired the preprocessed data, either by processing or loading,
        ds_train = ds_train.batch(batch_size)
        ds_test = ds_test.batch(batch_size)  
        ds_val = ds_val.batch(batch_size)
        
        
        # Set everything in place 
        dataset = {'train': ds_train, 'validation': ds_val, 'test': ds_test}
        return dataset
    
    def display_data(self, *args, **kwargs):
        print("[WARN] - Displaying data not implemented for keyword_spotting")

    def display_classes_distribution(self, *args, **kwargs):
        print("[WARN] - Displaying classes distribution not implemented for keyword_spotting")



""" Generic function to pick a dataset """
def load(dataset, quantizer, **kwargs):
    options = {'mnist': MNIST, 
               'fashion_mnist': FashionMNIST, 'fashionmnist': FashionMNIST, 'fashion': FashionMNIST,
                'svhn': SVHN,
                'toyadmos': ToyADMOS, 
                'cifar10': CIFAR10, 'cifar': CIFAR10,
                'uci_har': UCI_HAR,
                'pdcoco': lambda *args, **kwargs: COCO('person_detection', *args, **kwargs),
                'gtsdb': GTSDB,
                'hgcal': HGCal,
                'dummy': dummy,
                'autompg': AutoMPG,
                'keyword_spotting': KeywordSpotting,
                'smartpixel': SmartPixel} 
                
    ds = options.get(dataset.lower(), None)(quantizer, **kwargs)

    return ds

