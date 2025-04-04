""" Code for weight ranking according to different methods """

# Basic
from dataclasses import dataclass, field
from typing import Optional, List

""" Local imports """

""" Modules """
import os
import copy
import time
from glob import glob
import re
import json

""" Numpy and pandas """
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

""" Matplotlib & seaborn """
import matplotlib.pyplot as plt 
import seaborn as sns

""" Tensorflow """
import tensorflow as tf


""" netsurf modules """
import netsurf



"""
    Decorator to print the time it takes to run a function
"""
def timeranking(func):
    def wrapper(self, model, *args, verbose = True, **kwargs):
        # Start timer
        start_time = time.time()

        if self.ranking.is_empty:
            # Rank weights and get table back
            # Keep track of how long it takes for this ranker to rank the weights
            if verbose: netsurf.info(f'Ranking weights with method {self.method}... ', end = '')
            result = func(self, model, *args, out_dir = self.path, verbose = verbose, **kwargs)

            # Check if rank returned two or one values
            if isinstance(result, tuple):
                self.ranking, self.ranking_filename_format = result[0], result[1]
            else:
                self.ranking = result
            
        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        if verbose: netsurf.info(f'Ranking done in {elapsed_time:3.2f} seconds')

        # Store in global_metrics
        self.ranking_time = elapsed_time
        return self.ranking
    return wrapper






####################################################################################################################
#
# RANK OBJECT
#
####################################################################################################################
@dataclass
class Ranking:
    ranking: pd.DataFrame
    alias: str = None
    method: str = None
    filepath: Optional[str] = None
    loaded_from_file: Optional[bool] = False

    @property
    def is_empty(self):
        if self.ranking is None:
            return True
        return self.ranking.empty

    """
        Extend properties and methods so it looks like this object is just a dataframe
    """
    # Indexing should index the dataframe directly
    def __getitem__(self, item):
        return self.ranking.__getitem__(item)

    def __setitem__(self, key, value):
        self.ranking.__setitem__(key, value)

    def __len__(self):
        return len(self.ranking)
    
    def __iter__(self):
        return iter(self.ranking)
    
    def __next__(self):
        return next(iter(self.ranking))
    
    def iterrows(self):
        return self.ranking.iterrows()
    
    def items(self):
        return self.ranking.items()
    
    def values(self):
        return self.ranking.values()

    @staticmethod 
    def from_file(filepath, alias = None, method = None):
        # check file 
        if not netsurf.utils.path_exists(filepath):
            # Warn the user that the file doesn't exist and initialize and empty ranking
            netsurf.error(f"File {filepath} not found. Initializing empty ranking.")
            # Create an empty ranking
            return Ranking(pd.DataFrame(), alias = alias, method = method, filepath = filepath, loaded_from_file = False)
        
        # Load the ranking
        df = pd.read_csv(filepath)
        # If we can find a metadata object in the parent directory (where the folder "config1" sits), we can get the alias from there
        metadata = netsurf.utils.get_metadata(os.path.dirname(filepath))
        if metadata is not None:
            if 'alias' in metadata:
                alias = metadata['alias']
        # if alias is still None, we can try to infer it from df 
        if alias is None:
            if 'ranker' in df.columns:
                # get rows that are not nan, only strings
                alias = df.loc[df['ranker'].notna(), 'ranker'].astype(str).mode()[0]
                netsurf.info(f'Alias inferred from ranking data itself: {alias}')
        # Same for method
        if method is None:
            if 'method' in df.columns:
                # get rows that are not nan, only strings
                method = df.loc[df['method'].notna(), 'method'].astype(str).mode()[0]
                netsurf.info(f'Method inferred from ranking data itself: {method}')

        # Initialize 
        netsurf.info(f"Loading ranking from {filepath}")
        return Ranking(df, alias = alias, method = method, filepath = filepath, loaded_from_file = True)

    def save(self, filepath = None, overwrite = False):
        # We have to check if the filepath is a file or a folder. If it's a folder, we need to add the filename
        # to the folder. If it's a file, we need to check if it exists and if it does, we need to replace it.
        if filepath is None:
            if self.filepath is None:
                raise ValueError("No filepath provided. Cannot save ranking.")
            filepath = self.filepath
        # Check if the filepath is a file or a folder
        if not filepath.endswith('.csv'):
            if os.path.isdir(filepath):
                filepath = os.path.join(filepath, "ranking.csv")
        # if filepath is None, replace with this now
        if self.filepath is None:
            self.filepath = filepath
        if os.path.exists(filepath) and not overwrite:
            # Warn the user that the file already exists and ask if they want to overwrite it
            netsurf.warn(f"File {filepath} already exists. Use overwrite=True to overwrite it.")
            return 
        # Save the ranking
        self.ranking.to_csv(filepath, index = False)
        netsurf.info(f"Ranking saved to {filepath} Internal filepath definition in object updated.")
    
    def plot_ranking(self, axs = None, w = 300, show = True):
        
        # Fields and titles
        items = [('bit','Bit number', lambda x: x, 'green'), 
                 ('value', 'Param Value', lambda x: x, 'orange'),
                 ('binary', 'Num Ones (bin)', lambda x: [np.sum([int(i) for i in xx]) for xx in x] , 'blue'),
                 ('pruned', 'Pruned', lambda x: 1.0*x, 'red'),
                 ('variable_index', 'Variable Index (~Layer)', lambda x: x, 'black'),
                  ('susceptibility', 'Raw susceptibility', lambda x: x, 'purple'),
                  ('susceptibility', 'Absolute |Susceptibility|', lambda x: np.abs(x), 'purple')]
        # if impact in ranking, add it too
        if 'impact' in self.ranking.columns:
            items.append(('impact', 'Impact', lambda x: x, 'brown'))
        if 'gradient' in self.ranking.columns:
            items.append(('gradient', 'Gradient', lambda x: x, 'brown'))
        if 'hessian' in self.ranking.columns:
            items.append(('hessian', 'Hessian', lambda x: x, 'brown'))

        # available fields are
        # df = {'param': [], 'global_param_num': [], 'variable_index': [], 'internal_param_num': [],
        #       'coord': [], 'bit': [], 'value': [], 'rank': [], 'susceptibility': [], 
        #     'binary': [], 'pruned': []}
        
        num_axs = len(items)
        if axs is not None:
            # Make sure it's the right length
            if len(axs) != num_axs:
                netsurf.error(f'Expected {num_axs} axes, got {len(axs)}. Falling back to default.')
                axs = None

        # Plot indexes in ranking
        show_me = (axs is None) & show
        if axs is None:
            fig, axs = plt.subplots(num_axs, 1, figsize=(13, 13))
        else:
            fig = axs[0].figure

        # Plot bit number first 
        for i, (field, title, transform, color) in enumerate(items):
            netsurf.utils.plot.plot_avg_and_std(transform(self[field]), w, axs[i], shadecolor=color, ylabel=title)

        if show_me:
            plt.tight_layout()
            plt.show()
        else:
            return fig, axs


####################################################################################################################
#
# GENERIC RANKER (PARENT OF ALL)
#
####################################################################################################################

# Generic Weight Ranker (parent of rest)
@dataclass
class WeightRanker:
    quantization: 'QuantizationScheme' = field(default=None, repr=False)
    ranking: Optional[pd.DataFrame] = field(default=None, repr=False)
    ascending: Optional[bool] = False
    times_weights: Optional[bool] = False
    normalize_score: Optional[bool] = False
    use_delta_as_weight: Optional[bool] = False
    method_suffix: Optional[str] = None
    method_kws: Optional[str] = ""
    batch_size: Optional[int] = 1000

    complete_ranking: Optional[bool] = True
    parent_path: Optional[str] = None
    ranking_time: Optional[float] = None
    reload_ranking: Optional[bool] = True
    config_hash: Optional[str] = None
    config_tag: Optional[str] = None
    
    _ICON = "🏆"

    @property
    def config(self):
        d = {
            'quantization': self.quantization._scheme_str,
            'times_weights': self.times_weights,
            'normalize_score': self.normalize_score,
            'use_delta_as_weight': self.use_delta_as_weight,
            'alias': self.alias,
            'method': self.method,
            'ascending': self.ascending,
            'method_suffix': self.method_suffix,
            'method_kws': self.method_kws,
        }
        return d

    @property
    def method(self):
        return self.alias
    
    def __post_init__(self, *args, **kwargs):
        # Initialize the dataframe
        self.ranking = None
        # Initialize the path
        self.path = None

        # If parent_path is not None, try to get the "config" hash and tag for this ranker
        if self.parent_path is not None:
            self.config_hash, self.config_tag = self.get_config_hash(self.parent_path, self.config)

        # Now, if parent_path is not None and "reload_ranking" is True, we have to check if it exists and reload ranking
        if self.reload_ranking:
            if netsurf.utils.path_exists(self.parent_path):
                # First, we need to scout the path for all folders (they will look like <config1>, <config2>, etc.)
                if self.config_tag is None:
                    configs = self.matched_dirs(self.parent_path)
                    if len(configs) > 0:
                        # Pick latest (0)
                        path = configs[0]
                    else:
                        # inform the user there are no config folders
                        netsurf.warn(f"No config folders found in {self.parent_path} for ranker {self.alias}. Cannot reload ranking.")
                        return None
                else:
                    path = os.path.join(self.parent_path, self.config_tag)
                
                # Create 
                self.ranking = Ranking.from_file(os.path.join(path, "ranking.csv"), *args, alias = self.alias, method = self.method, **kwargs)
                # Update path 
                self.path = path
            else:
                # inform the user the directory doesn't exist
                netsurf.warn(f"Path {self.parent_path} does not exist for ranker {self.alias}. Cannot reload ranking.")

    @staticmethod
    def get_config_hash(path, config):
        config_hash = netsurf.utils.generate_config_hash(config)

        # Check if path exists. If it does, check if there are any config folders in there with files (.hash) that
        # have the same hash 
        config_dirs = WeightRanker.find_config_dirs(path)

        if len(config_dirs) > 0:
            # Check if the hash exists in any of the config folders
            for config_dir in config_dirs:
                # Check if the hash file exists
                hash_file = os.path.join(config_dir, ".hash")
                if netsurf.utils.path_exists(hash_file):
                    if os.path.isfile(hash_file):
                        # read it and check if it matches our "config_hash" variable
                        with open(hash_file, 'r') as f:
                            hash_value = f.read().strip()
                            if hash_value == config_hash:
                                # If it matches, we can use this folder as our parent path
                                netsurf.info(f"Found config hash {config_hash} in {config_dir}. Using this as parent path.")
                                return config_hash, os.path.basename(config_dir)
            else:
                # There are config dirs, but none of them match our hash, meaning we need to get the latest one and 
                # increase +1 in the name
                # Extract numbers that match the pattern "config(\d+)"
                numbers = []
                for name in config_dirs:
                    match = re.fullmatch(r'config(\d+)', name)
                    if match:
                        numbers.append(int(match.group(1)))
                
                tag = max(numbers) + 1 if numbers else 1
                tag = f'config{tag}'
        else:
            # No config dirs, so we can create a new one
            tag = 'config1'
        # Log 
        netsurf.info(f'Initializing new experiment config directory with hash {config_hash} @ {tag}')      
        # Create directory and print hash 
        config_dir = os.path.join(path, tag)
        os.makedirs(config_dir, exist_ok=True)
        # Write hash to file
        hash_file = os.path.join(config_dir, ".hash")
        with open(hash_file, 'w') as f:
            f.write(config_hash)
        # Save the config as well
        config_file = os.path.join(config_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        return config_hash, tag
        

    @staticmethod 
    def find_config_dirs(path):
        if not netsurf.utils.path_exists(path):
            return []
        all_dirs = glob(os.path.join(path, "config*"))
    
        # Filter using regex: must match exactly 'config' followed by digits
        pattern = re.compile(r"config\d+$")
        matched_dirs = [d for d in all_dirs if os.path.isdir(d) and pattern.fullmatch(os.path.basename(d))]
        # Sort by most recent first 
        if len(matched_dirs) > 0:
            matched_dirs.sort(key=os.path.getmtime, reverse=True)
        return matched_dirs

    @staticmethod
    # Function to create a weight ranker given a couple of flags 
    def build(method: str, *args, **kwargs):
        options = {'random': RandomWeightRanker, 
                    'weight_abs_value': AbsoluteValueWeightRanker,
                    'layerwise': LayerWeightRanker,
                    'bitwise': BitwiseWeightRanker,
                    'hirescam': HiResCamWeightRanker,
                    'hiresdelta': HiResDeltaRanker,
                    'recursive_uneven': RecursiveUnevenRanker,
                    'diffbitperweight': DiffBitsPerWeightRanker,
                    'hessian': HessianWeightRanker,
                    'hessiandelta': HessianDeltaWeightRanker,
                    'qpolar': QPolarWeightRanker,
                    'qpolargrad': QPolarGradWeightRanker,
                    'fisher': FisherWeightRanker,
                    'aiber': AIBerWeightRanker}
                    
        return options.get(method.lower(), WeightRanker)(*args, **kwargs)
    
    def save_ranking(self, filepath: str = None):
        # Save the ranking to file
        if self.ranking is not None:
            if filepath is None:
                if self.ranking.filepath is not None:
                    filepath = self.ranking.filepath
            self.ranking.save(filepath)
        else:
            raise ValueError("Ranking is None. Cannot save ranking.")

    """ Extracting the table of weights and creating the pandas DF is the same 
        for all methods, so we can define it inside the generic weightRanker obj 
    """
    def extract_weight_table(self, model, *args, verbose = True, **kwargs):
        # Get all variables for this model 
        variables = model.trainable_variables
        vnames = [v.name for v in model.variables]

        # Get quantization
        Q = self.quantization

        # This is what the table should look like:
        # | param name | index (in trainable_variables) | coord (inside variable) | value | rank | susceptibility | bit |
        # +------------+--------------------------------+--------------------------+-------+------+----------------+-----+
        # Create dictionary to store all the values
        df = {'param': [], 'global_param_num': [], 'variable_index': [], 'internal_param_num': [],
              'coord': [], 'bit': [], 'value': [], 'rank': [], 'susceptibility': [], 
            'binary': [], 'pruned': []}
        
        with netsurf.utils.ProgressBar(total = len(variables), prefix = f'Extracting base weight table for {model.name}...') as pbar:
            cum_index = 0
            for iv, v in enumerate(variables):
                # Get the indices for each dimension
                indices = np.indices(v.shape)
                # Reshape to get a list of coordinates
                # This creates an array of shape (num_dimensions, total_elements)
                coords = indices.reshape(indices.shape[0], -1).T
                # Flatten the values 
                values = v.numpy().flatten()
                # Repeat the name of the variable for all the values 
                names = np.array([v.name]*len(values))
                # Get the param number as np.arange 
                internal_param_num = np.arange(len(values))
                # The global_index is iv 
                variable_index = [iv]*len(values)
                # Get the binary representation of each value 
                binary = np.apply_along_axis("".join, 1, (Q.bin(values)*1).astype(str))

                # Finally, init the bit 
                bits = np.repeat(np.arange(Q.n + Q.s - 1, -Q.f-1, -1), len(values))

                # global_param_num
                global_param_num = cum_index + np.arange(len(values))

                # Add cum_index
                cum_index += len(values)

                # We need to repeat this num_bits times
                names = np.tile(names, Q.m)
                variable_index = np.tile(variable_index, Q.m)
                internal_param_num = np.tile(internal_param_num, Q.m)
                binary = np.tile(binary, Q.m)
                coords = np.tile(coords, (Q.m, 1))
                values = np.tile(values, Q.m)
                # Repeat global_param_num
                global_param_num = np.tile(global_param_num, Q.m)

                # Also, keep track of parameters that have been pruned
                # get the pruned mask 
                pruned_mask_vname = v.name.replace(':','_prune_mask:')
                if pruned_mask_vname in vnames:
                    pruned = model.variables[vnames.index(pruned_mask_vname)].numpy().flatten() == 0
                    # tile
                    pruned = np.tile(pruned, Q.m)
                else:
                    pruned = (values == 0)  

                # Now, extend the dictionary with the new values
                df['param'].extend(list(names))
                df['global_param_num'].extend(list(global_param_num))
                df['variable_index'].extend(list(variable_index))
                df['coord'].extend(list(coords))
                df['value'].extend(list(values))
                df['rank'].extend(list([0]*len(values)))
                df['susceptibility'].extend(list([0]*len(values)))
                df['bit'].extend(list(bits))
                df['internal_param_num'].extend(list(internal_param_num))
                df['binary'].extend(list(binary))
                df['pruned'].extend(list(pruned))

                # Update progress bar by 1
                pbar.update()

        # Build df 
        df = pd.DataFrame(df)
        # The index is the global parameter number. Explicitly set it as a column as well (in case we
        # re-sort and reindex by mistake or not later)
        df['param_num'] = df.index

        # Let's randomize before ranking so we get rid of locality 
        df = df.sample(frac=1)

        return df

    
    def plot_ranking(self, axs = None, w = 300, show = True):
        
        # Fields and titles
        items = [('bit','Bit number', lambda x: x, 'green'), 
                 ('value', 'Param Value', lambda x: x, 'orange'),
                 ('binary', 'Num Ones (bin)', lambda x: [np.sum([int(i) for i in xx]) for xx in x] , 'blue'),
                 ('pruned', 'Pruned', lambda x: 1.0*x, 'red'),
                 ('variable_index', 'Variable Index (~Layer)', lambda x: x, 'black'),
                  ('susceptibility', 'Raw susceptibility', lambda x: x, 'purple'),
                  ('susceptibility', 'Absolute |Susceptibility|', lambda x: np.abs(x), 'purple')]
        # Make sure that binary is actually a binary string 
        if not isinstance(self.ranking['binary'][0], str):
            self.ranking['binary'] = ["".join([str(xx) for xx in x]) for x in (1.0*self.quantization.bin(self.quantization(self.ranking['value'].values))).astype(int)]
        elif isinstance(self.ranking['binary'][0], str):
            # But length is not correct
            if len(self.ranking['binary'][0]) != self.quantization.m:
                self.ranking['binary'] = ["".join([str(xx) for xx in x]) for x in (1.0*self.quantization.bin(self.quantization(self.ranking['value'].values))).astype(int)]
        # if impact in ranking, add it too
        if 'impact' in self.ranking.columns:
            items.append(('impact', 'Impact', lambda x: x, 'brown'))
        if 'gradient' in self.ranking.columns:
            items.append(('gradient', 'Gradient', lambda x: x, 'brown'))
        if 'hessian' in self.ranking.columns:
            items.append(('hessian', 'Hessian', lambda x: x, 'brown'))

        # available fields are
        # df = {'param': [], 'global_param_num': [], 'variable_index': [], 'internal_param_num': [],
        #       'coord': [], 'bit': [], 'value': [], 'rank': [], 'susceptibility': [], 
        #     'binary': [], 'pruned': []}
        
        num_axs = len(items)
        if axs is not None:
            # Make sure it's the right length
            if len(axs) != num_axs:
                netsurf.error(f'Expected {num_axs} axes, got {len(axs)}. Falling back to default.')
                axs = None

        # Plot indexes in ranking
        show_me = (axs is None) & show
        if axs is None:
            fig, axs = plt.subplots(num_axs, 1, figsize=(13, 13))
        else:
            fig = axs[0].figure

        # Plot bit number first 
        for i, (field, title, transform, color) in enumerate(items):
            netsurf.utils.plot.plot_avg_and_std(transform(self.ranking[field]), w, axs[i], shadecolor=color, ylabel=title)

        if show_me:
            plt.tight_layout()
            plt.show()
        else:
            return fig, axs
    
    @property
    def alias(self):
        return 'generic'


####################################################################################################################
#
# RANDOM RANKER
#
####################################################################################################################

# Random weight Ranker (list all the weights in the structure and rank them randomly)
@dataclass
class RandomWeightRanker(WeightRanker):
    _ICON = "🎲"

    # Method to actually rank the weights
    @timeranking
    def rank(self, model, *args, base_df = None, **kwargs):
        # Call super method to obtain DF 
        df = self.extract_weight_table(model, *args, **kwargs) if base_df is None else base_df

        # Susceptibility here is considered uniform (hence the randomness assigning TMR)
        df['susceptibility'] = [1/(len(df))]*len(df)
        df['rank'] = np.random.permutation(np.arange(len(df)))
        # Sort by rank
        df = df.sort_values(by='rank')
        # Add column with the alias 
        df['alias'] = self.alias
        df['method'] = self.method

        # assign to self 
        self.ranking = Ranking(df, alias = self.alias, method = self.method, filepath = self.path, loaded_from_file = False)

        return self.ranking
    
    @property
    def alias(self):
        return 'random'


####################################################################################################################
#
# WEIGHT VALUE RANKERS (AbsoluteValue)
#
####################################################################################################################
""" Rank weights according to their absolute value (the larger, the most important) """
@dataclass
class AbsoluteValueWeightRanker(WeightRanker):
    _ICON = "🗿"

    # Method to actually rank the weights
    @timeranking
    def rank(self, model, *args, ascending = False, base_df = None, **kwargs):
        # Call super method to obtain DF 
        df = self.extract_weight_table(model, *args, **kwargs) if base_df is None else base_df

        # Susceptibility here is considered uniform (hence the randomness assigning TMR)
        #df = df.sort_values(by='value', key=abs, ascending=ascending)
        df['susceptibility'] = np.abs(df['value'].values)
        df = df.sort_values(by=['pruned','bit','susceptibility'], ascending = [True, False, ascending])
        df['rank'] = np.arange(len(df))
        # Add column with the alias 
        df['alias'] = self.alias
        df['method'] = self.method
        
        # assign to self 
        self.ranking = Ranking(df, alias = self.alias, method = self.method, filepath = self.path, loaded_from_file = False)

        return self.ranking
    
    @property
    def alias(self):
        return 'weight_abs_value'


####################################################################################################################
#
# POSITIONAL RANKERS (Bitwise MSB_LSB, layerwise, etc)
#
####################################################################################################################

""" Rank weights by layer (top to bottom, bottom to top or custom order) """
@dataclass
class BitwiseWeightRanker(WeightRanker):
    _ICON = "🔢"
    
    # Method to actually rank the weights
    @timeranking
    def rank(self, model, *args, ascending = False, base_df = None, **kwargs):
        # Call super method to obtain DF 
        df = self.extract_weight_table(model, *args, **kwargs) if base_df is None else base_df

        # Susceptibility here is considered uniform (hence the randomness assigning TMR)
        df['susceptibility'] = 2.0**df['bit']
        df = df.sort_values(by=['pruned','bit'], ascending = [True, ascending])
        df['rank'] = np.arange(len(df))
        # Add column with the alias 
        df['alias'] = self.alias
        df['method'] = self.method
        
        # assign to self 
        self.ranking = Ranking(df, alias = self.alias, method = self.method, filepath = self.path, loaded_from_file = False)

        return self.ranking
    
    @property
    def method(self):
        return 'bitwise'
    
    @property
    def alias(self):
        return f'bitwise_{"lsb" if self.ascending else "msb"}'


""" Rank weights by layer (top to bottom, bottom to top or custom order) """
@dataclass
class LayerWeightRanker(WeightRanker):
    _ICON = "🎞️"

    # Method to actually rank the weights
    @timeranking
    def rank(self, model, *args, ascending = True, **kwargs):
        # Call super method to obtain DF 
        df = self.extract_weight_table(model, *args, **kwargs)

        # Susceptibility here is considered uniform (hence the randomness assigning TMR)
        # (variable_index is almost like layer index, although there is a preference of kernel vs bias, cause 
        # when listing variables kernel always comes before bias, even for the same layer, but whatever).
        # If required, we could actually enforce layer index computation by grouping variables by their name 
        # (removing the "/..." part of the name) and then sorting by the order of appearance in the model, 
        # but I don't really think this is required right now. 
        df = df.sort_values(by=['pruned', 'bit', 'variable_index'], ascending = [True, False, ascending])
        df['rank'] = np.arange(len(df))
        df['susceptibility'] = 2.0**df['bit'] * (self.quantization.m - df['variable_index'] + 1)
        # Add column with the alias 
        df['alias'] = self.alias
        df['method'] = self.method

        # assign to self 
        self.ranking = Ranking(df, alias = self.alias, method = self.method, filepath = self.path, loaded_from_file = False)
        self.ascending = ascending

        return self.ranking
    
    @property
    def method(self):
        return 'layerwise'
    
    @property
    def alias(self):
        return f'layerwise_{"first" if self.ascending else "last"}'



####################################################################################################################
#
# COMPENSATIONAL RANKERS (DiffBitsPerWeight, RecursiveUnevenRanker)
#
####################################################################################################################

""" Rank weights with how different bits per weight"""
@dataclass
class DiffBitsPerWeightRanker(WeightRanker):
    _ICON = "🔢"

    def calculate_bit_differences(self, binary_str):
        # Initialize the sum of differences
        diff_sum = 0
        # Iterate through the binary string, excluding the sign bit
        for i in range(1, len(binary_str) - 1):
            # Calculate the difference between adjacent bits
            diff = int(binary_str[i + 1]) - int(binary_str[i])
            # Add the absolute value of the difference to the sum
            diff_sum += abs(diff)
        return diff_sum

    @timeranking
    def rank(self, model, *args, base_df = None, **kwargs):
        # Call super method to obtain DF 
        def process_value(value):
            b = netsurf.utils.float_to_binary(value,self.quantization.m)
            diff_sum = self.calculate_bit_differences(b)
            return diff_sum

        df = self.extract_weight_table(model, self.quantization.m) if base_df is None else base_df

        differences = np.vectorize(process_value)(df['value'].values)
        df['susceptibility'] = differences
        df = df.sort_values(by=['pruned','bit'], ascending = [True, False])
        df['rank'] = np.arange(len(df))
        # Add column with the alias 
        df['alias'] = self.alias
        df['method'] = self.method

        # assign to self 
        self.ranking = Ranking(df, alias = self.alias, method = self.method, filepath = self.path, loaded_from_file = False)

        return self.ranking

    
    @property
    def alias(self):
        return f'diff_bits_per_weight'


""" Rank weights by using proportion Recursively """
@dataclass
class RecursiveUnevenRanker(WeightRanker):
    _ICON = "🔄"

    # Method to actually rank the weights
    @timeranking
    def rank(self, model, *args, base_df = None, **kwargs):
        # Call super method to obtain DF 
        df = self.extract_weight_table(model, **kwargs) if base_df is None else base_df

        last_level = self.quantization.m - 1
        input_df = df[df['bit']==0]
        df = self.rec(input_df, last_level)
        df['rank'] = np.arange(len(df))
        # Add column with the alias 
        df['alias'] = self.alias
        df['method'] = self.method

        # assign to self 
        self.ranking = Ranking(df, alias = self.alias, method = self.method, filepath = self.path, loaded_from_file = False)

        return self.ranking

    def _rec(self, input_df, last_level, bit_level = 0, indexes_to_triplicate = []):

        # Calculate proportions 
        subdf = input_df['binary'].str[bit_level].astype(int)
        c0 = len(input_df[subdf == 0])
        c1 = len(input_df[subdf == 1])
        # check which one is greater (if c0 < c1 -> we want 1, otherwise get 0)
        next_bit = int(c0 < c1)
        # get the next subdff
        subdff = input_df[subdf == next_bit]

        # If this is not the last level, keep recursively
        if bit_level < last_level:
            indexes_to_triplicate = self._rec(subdff, bit_level + 1, last_level, indexes_to_triplicate = indexes_to_triplicate)
            #indexes_to_triplicate += list(indexes_to_triplicate2)
        else:
            # Now, we reached the last bit, which means we need to pick min(c0,c1) rows from 
            # whatever {c0,c1} has a greater proportion, and set them to triplicate.
            if min(c0,c1) > 0:
                indexes_to_triplicate = subdff.index[:min(c0,c1)]
            else:
                indexes_to_triplicate = subdff.index[:max(c0,c1)]
        return indexes_to_triplicate
            

    # Entry point function
    def rec(self, input_df, last_level, bit_level = 0, indexes_to_triplicate = []):
        rank = None
        total_weights = len(input_df)
        count = 0

        while (len(input_df) > 0):
            # Let's calculate all the proportions for all bits
            codes = np.stack(input_df['binary'].apply(lambda x: [int(i) for i in x]))
            ps = codes.mean(axis=0)
            
            w_left = len(input_df)
            msg = f'Weights left: {len(input_df)}  -->  '
            msg += '      '.join([f'({i}) {100*(1-p):3.2f}% {100*p:3.2f}%' for i,p in enumerate(ps)])
            print(msg)
            #pbar.set_postfix({'weights_left': len(input_df)})
            #count += len(indexes_to_triplicate)
            #pbar.update(count)
            if len(indexes_to_triplicate) > 0:
                # Remove from input_df
                sub = input_df.loc[indexes_to_triplicate]
                input_df = input_df.drop(indexes_to_triplicate)
                bits = np.repeat(np.arange(last_level+1)[:,None].T, len(sub), axis = 0).flatten()
                sub = sub.loc[sub.index.repeat(last_level+1)]
                sub['bit'] = bits
                # all_indexes = list(input_df.index)
                # indexes_not_to_triplicate = []
                # for i in indexes_to_triplicate:
                #     if indexes_to_triplicate in all_indexes:
                #         indexes_not_to_triplicate.append(i)
                # input_df = input_df.loc[indexes_not_to_triplicate]
                if rank is None:
                    rank = sub
                else:
                    # Append
                    rank = pd.concat([rank, sub], axis = 0)
                
                # Reset indexes_to_triplicate
                indexes_to_triplicate = []
                
            # Just call recursive method 
            indexes_to_triplicate = self._rec(input_df, bit_level, last_level, indexes_to_triplicate = indexes_to_triplicate)
        
        #rank = pd.concat([rank, sub], axis = 0)
        return rank
    
    @property
    def alias(self):
        return f'recursive_uneven'



####################################################################################################################
#
# FIRST ORDER GRADIENT RANKERS (GRADCAM++)
#
####################################################################################################################

""" Rank weights by using HiRes (gradcam++) (we don't really use this class directly, this is just a parent 
    for HiResCamRanker and HiResDeltaRanker) 
"""
@dataclass
class GradRanker(WeightRanker):
    _ICON = "💈"

    # Method to actually rank the weights
    @timeranking
    def rank(self, model, X, Y, ascending = False,  **kwargs):
        # Call super method to obtain DF 
        df = self.extract_weight_table(model, X, Y, **kwargs)

        # Finally, sort by susceptibility 
        df = df.sort_values(by=['pruned','bit','susceptibility'], ascending = [True, False, ascending])
        # Add column with the alias 
        df['alias'] = self.alias
        df['method'] = self.method

        # assign to self 
        self.ranking = Ranking(df, alias = self.alias, method = self.method, filepath = self.path, loaded_from_file = False)

        return self.ranking
    
    """ Extracting the table of weights and creating the pandas DF is the same 
        for all methods, so we can define it inside the generic weightRanker obj 
    """
    def extract_weight_table(self, model, X, Y, quantization: 'QuantizationScheme', 
                                batch_size = 1000, verbose = True, 
                                normalize_score = False, times_weights = False,
                                ascending = False, absolute_value = True, base_df = None,
                                bit_value = None, out_dir = ".", **kwargs):
        
        # Call super to get the basic df 
        df = super().extract_weight_table(model, quantization, verbose = verbose, 
                                          ascending = ascending,
                                          **kwargs) if base_df is None else base_df

        # Make sure susceptibility is float
        df['susceptibility'] = df['susceptibility'].astype(float)

        # If use_delta as weight, clone the model and replace the weights with the deltas for each bit
        use_delta_as_weight = self.use_delta_as_weight

        # Loop thru all bits (even if we are not using deltas as weights, this is so we can use the same loop
        # and code for both cases. If use_delta_as_weight is False we will break after the first loop iter)
        for ibit, bit in enumerate(np.arange(quantization.n + quantization.s - 1, -quantization.f-1, -1)):
            
            # Clone our model so we don't mess with the original one
            delta_model = model.clone()

            # Make sure we compile
            delta_model.compile(loss = delta_model.loss, optimizer = delta_model.optimizer, metrics = delta_model.metrics)

            # If use_delta_as_weight, then replace the weights with the deltas for the current bit
            if use_delta_as_weight:    
                deltas = delta_model.deltas

                # Replace 
                for iv, v in enumerate(delta_model.trainable_variables):
                    vname = v.name
                    assert deltas[iv].shape[:-1] == v.shape, f'Error at {iv} {vname} {deltas[iv].shape[:-1]} != {v.shape}'

                    # Replace the weights with the deltas (if use_delta_as_weight)
                    v.assign(deltas[iv][...,ibit])

            # Now let's get the gradients for all trainable variables
            # We will use the activation model to get the output for every layer 
            with tf.GradientTape(persistent = True) as tape:
                # Forward pass
                predictions = delta_model(X, training=True)
                
                # Calculate loss
                loss = delta_model.loss(Y, predictions)
                
                # Add regularization losses if any
                if delta_model.losses:
                    loss += tf.math.add_n(delta_model.losses)

            # Get gradients
            orig_gradients = tape.gradient(loss, delta_model.trainable_variables)

            # Copy gradients over
            gradients = [tf.identity(g) for g in orig_gradients]

            # Apply transformations required
            if times_weights:
                # Multiply gradients times variables 
                gradients = [g*v for g,v in zip(gradients, delta_model.trainable_variables)]

            if absolute_value:
                gradients = [tf.math.abs(g) for g in gradients]

            if normalize_score:
                # Normalize by range per variable (max-min) minus min
                gradients = [(g-tf.math.reduce_min(g))/(tf.math.reduce_max(g)-tf.math.reduce_min(g)) for g in gradients]

            # Now build the table for pandas
            for iv, (g, g0) in enumerate(zip(gradients, orig_gradients)):
                vname = model.trainable_variables[iv].name
                # Find the places in DF tht match this variable's name 
                # and set the susceptibility to the gradient value for this bit 
                idx = df[(df['param'] == vname) & (df['bit'] == bit)].index
                df.loc[idx,'susceptibility'] = g.numpy().flatten()
                # Set the original gradient also just in case 
                df.loc[idx, 'gradients'] = g0.numpy().flatten()
                
                # if not use_weight_as_delta, repeat this for all bits 
                if not use_delta_as_weight:
                    for i in np.arange(quantization.n + quantization.s - 1, -quantization.f-1, -1):
                        if i != bit:
                            idx = df[(df['param'] == vname) & (df['bit'] == i)].index
                            df.loc[idx,'susceptibility'] = g.numpy().flatten()
            
            # And finally, break if we are not using deltas as weights
            if not use_delta_as_weight:
                break

        return df


""" Rank weights by using HiRes (gradcam++) as a weight to compute, on average, how important each weight is """
@dataclass
class HiResCamWeightRanker(GradRanker):
    
    @property
    def method(self):
        return 'hirescam'

    @property
    def alias(self):
        alias = 'hirescam'
        if self.times_weights:
            alias += '_times_weights'
        if self.normalize_score:
            alias += '_norm'
        return alias

""" Rank weights by using HiRes (gradcam++) but instead of using the weights of the model, we use the DELTAS as weights """
@dataclass
class HiResDeltaRanker(GradRanker):
    _ICON="🔮"
    def __init__(self, *args, use_delta_as_weight = True, **kwargs):
        """."""
        super().__init__(*args, use_delta_as_weight = True, **kwargs)
    
    @property
    def method(self):
        return 'hiresdelta'

    @property
    def alias(self):
        alias = 'hiresdelta'
        if self.times_weights:
            alias += '_times_weights'
        if self.normalize_score:
            alias += '_norm'
        return alias


####################################################################################################################
#
# SECOND ORDER GRADIENT RANKERS (Hessian)
#
####################################################################################################################

""" This is the parent class for all Hessian-based rankers """
@dataclass
class HessianRanker(WeightRanker):
    eigen_k_top: Optional[float] = 4
    max_iter: Optional[float] = 1000
    
    _ICON = "👴🏻"

    # override extract_weight_table
    def extract_weight_table(self, model, X, Y, 
                             normalize_score = False,
                             ascending = False, absolute_value = True, base_df = None, 
                             verbose = True, 
                             **kwargs):
        
        # Get values 
        times_weights = self.times_weights
        normalize_score = self.normalize_score
        delta_ranking = False

        # Call super to get the basic df 
        df = super().extract_weight_table(model, verbose = verbose, 
                                          ascending = ascending,
                                          **kwargs) if base_df is None else base_df

        # get quantizer
        Q = self.quantization

        # Make sure susceptibility is float
        df['susceptibility'] = df['susceptibility'].astype(float)

        # If use_delta as weight, clone the model and replace the weights with the deltas for each bit
        use_delta_as_weight = self.use_delta_as_weight

        # Store the original variable values first
        original_vars = [v.numpy() for v in model.trainable_variables]

        # Loop thru all bits (even if we are not using deltas as weights, this is so we can use the same loop
        # and code for both cases. If use_delta_as_weight is False we will break after the first loop iter)
        for ibit, bit in enumerate(np.arange(Q.n + Q.s - 1, -Q.f-1, -1)):
            
            # Clone our model so we don't mess with the original one
            #delta_model = model.clone()

            # Make sure we compile
            #delta_model.compile(loss = delta_model.loss, optimizer = delta_model.optimizer, metrics = delta_model.metrics)

            # If use_delta_as_weight, then replace the weights with the deltas for the current bit
            if use_delta_as_weight:    
                deltas = model.deltas

                # Replace 
                for iv, v in enumerate(model.trainable_variables):
                    vname = v.name
                    assert deltas[iv].shape[:-1] == v.shape, f'Error at {iv} {vname} {deltas[iv].shape[:-1]} != {v.shape}'

                    # Replace the weights with the deltas (if use_delta_as_weight)
                    v.assign(deltas[iv][...,ibit])

            # Compute top eigenvalues/vectors
            eigenvalues, eigenvectors = self.top_eigenvalues(model, X, Y, k=self.eigen_k_top, max_iter = self.max_iter)

            # Get parameter sensitivity ranking
            sensitivity = self.parameter_sensitivity(model, eigenvalues, eigenvectors)

            if absolute_value:
                sensitivity = [tf.math.abs(g) for g in sensitivity]

            if normalize_score:
                gmax = tf.math.reduce_max([tf.math.reduce_max(g) for g in sensitivity])
                gmin = tf.math.reduce_min([tf.math.reduce_min(g) for g in sensitivity])
                # Normalize by range per variable (max-min) minus min
                sensitivity = [(g-gmin)/(gmax-gmin)for g in sensitivity]

            # Now build the table for pandas
            for iv, g in enumerate(sensitivity):
                vname = model.trainable_variables[iv].name
                # Find the places in DF tht match this variable's name 
                # and set the susceptibility to the gradient value for this bit 
                idx = df[(df['param'] == vname) & (df['bit'] == bit)].index
                df.loc[idx,'susceptibility'] = g.numpy().flatten()
                
                # if not use_weight_as_delta, repeat this for all bits 
                if not use_delta_as_weight:
                    for i in np.arange(Q.n + Q.s - 1, -Q.f-1, -1):
                        if i != bit:
                            idx = df[(df['param'] == vname) & (df['bit'] == i)].index
                            df.loc[idx,'susceptibility'] = g.numpy().flatten()
            
            # And finally, break if we are not using deltas as weights
            if not use_delta_as_weight:
                break
        
        # if use_delta_as_weight make sure to set the original values back 
        if use_delta_as_weight:
            for iv, v in enumerate(model.trainable_variables):
                v.assign(original_vars[iv])

        return df


    def _flatten_params(self, params):
        """Flatten a list of parameter tensors into a single 1D tensor"""
        return tf.concat([tf.reshape(p, [-1]) for p in params], axis=0)
    
    def _reshape_vector_to_param_shapes(self, vars, vector):
        """Reshape a flat vector back to the original parameter shapes"""
            
        param_shapes = [p.shape for p in vars]
        param_sizes = [tf.size(p).numpy() for p in vars]
        
        reshaped_params = []
        start_idx = 0
        for shape, size in zip(param_shapes, param_sizes):
            end_idx = start_idx + size
            reshaped_params.append(tf.reshape(vector[start_idx:end_idx], shape))
            start_idx = end_idx
            
        return reshaped_params

    def parameter_sensitivity(self, model, eigenvalues, eigenvectors, strategy="sum"):
        """
        Compute parameter sensitivity based on eigenvalues and eigenvectors
        
        Args:
            eigenvalues: List of eigenvalues
            eigenvectors: List of eigenvectors
            strategy: 'sum' or 'max' strategy for combining eigenvector contributions
            
        Returns:
            (parameter_ranking, sensitivity_scores)
        """
        if strategy not in ["sum", "max"]:
            raise ValueError("Strategy must be 'sum' or 'max'")
        
        # Get flattened parameter values
        params_flat = self._flatten_params(model.trainable_variables).numpy()
        
        # Flatten eigenvectors
        flat_eigenvectors = []
        for i, v in enumerate(eigenvectors):
            flat_v = self._flatten_params(v).numpy()
            if eigenvalues:
                flat_v *= eigenvalues[i].numpy()  # Scale by eigenvalue
            flat_eigenvectors.append(flat_v)
        
        # Compute sensitivity scores
        if strategy == "sum":
            scores = np.zeros_like(params_flat)
            for ev in flat_eigenvectors:
                # Compute contribution of this eigenvector
                contribution = np.abs(ev * params_flat)
                scores += contribution
        else:  # strategy == "max"
            stacked_evs = np.stack(flat_eigenvectors)
            abs_contributions = np.abs(stacked_evs * params_flat)
            scores = np.sum(abs_contributions, axis=0)
        
        # Reshape scores to parameter shapes
        scores = self._reshape_vector_to_param_shapes(model.trainable_variables, scores)

        # Rank parameters by score
        # param_ranking = np.flip(np.argsort(scores))
        # param_scores = scores[param_ranking]
        
        return scores

    @staticmethod
    @tf.function
    def _compute_hvp(model, x_batch, y_batch, v):
        """
        Compute the Hessian-vector product (HVP)
        
        Args:
            x_batch: Input batch
            y_batch: Target batch
            v: Vector to compute HVP with (list of tensors with same shapes as parameters)
            
        Returns:
            HVP and the inner product v^T * HVP (for eigenvalue computation)
        """
        
        with tf.GradientTape() as outer_tape:
            with tf.GradientTape() as inner_tape:
                y_pred = model(x_batch, training=True)
                loss = model.loss(y_batch, y_pred)
                
            grads = inner_tape.gradient(loss, model.trainable_variables)
            
            # Compute v^T * grads (needed for eigenvalue calculation)
            grads_v_product = tf.add_n([
                tf.reduce_sum(g * v_part) for g, v_part in zip(grads, v) 
                if g is not None and v_part is not None
            ])
            
        # Compute Hessian-vector product
        hvp = outer_tape.gradient(grads_v_product, model.trainable_variables)
        
        # Compute v^T * H * v (eigenvalue estimate)
        eigenvalue_estimate = tf.add_n([
            tf.reduce_sum(v_part * hvp_part) for v_part, hvp_part in zip(v, hvp)
            if v_part is not None and hvp_part is not None
        ])
        
        return hvp, eigenvalue_estimate
    
    def _generate_random_vector(self, vars, rademacher=True):
        """
        Generate a random vector with the same structure as model parameters
        
        Args:
            rademacher: If True, generate Rademacher random variables {-1, 1},
                       otherwise, generate standard normal random variables
                       
        Returns:
            List of random tensors with same shapes as parameters
        """
        
        if rademacher:
            return [tf.cast(2 * tf.random.uniform(p.shape, 0, 2, dtype=tf.int32) - 1, 
                           dtype=tf.float32) for p in vars]
        else:
            return [tf.random.normal(p.shape) for p in vars]
    
    
    def _normalize_vectors(self, v):
        """Normalize a list of vectors"""
        # Compute squared norm
        squared_norm = tf.add_n([tf.reduce_sum(tf.square(p)) for p in v])
        norm = tf.sqrt(squared_norm) + tf.keras.backend.epsilon()
        
        # Normalize each part
        return [p / norm for p in v]
    
    def _make_vector_orthogonal(self, v, eigenvectors):
        """
        Make v orthogonal to all vectors in eigenvectors
        
        Args:
            v: Vector to make orthogonal
            eigenvectors: List of vectors to make v orthogonal to
            
        Returns:
            v made orthogonal to all vectors in eigenvectors
        """
        for evec in eigenvectors:
            # Compute dot product
            dot_product = tf.add_n([
                tf.reduce_sum(v_part * e_part) for v_part, e_part in zip(v, evec)
            ])
            
            # Subtract projection
            v = [v_part - dot_product * e_part for v_part, e_part in zip(v, evec)]
            
        return v

    def top_eigenvalues(self, model, x, y, k=1, max_iter=100, tol=1e-6, verbose=True):
        """
        Compute the top k eigenvalues and eigenvectors of the Hessian using power iteration
        
        Args:
            x: Input data
            y: Target data
            k: Number of eigenvalues/vectors to compute
            max_iter: Maximum number of iterations for power method
            tol: Convergence tolerance
            verbose: Whether to show progress bar
            
        Returns:
            (eigenvalues, eigenvectors)
        """
        # Start timing
        start_time = time.time()
        
        eigenvalues = []
        eigenvectors = []
        
        # Set up progress tracking
        total_iterations = k * max_iter

        with netsurf.utils.ProgressBar(total=total_iterations, prefix=f'Computing eigenvalues (k={k}, max_iters={max_iter})') as pbar:

            # Compute each eigenvalue/vector pair
            for i in range(k):
                # Initialize random vector
                v = self._generate_random_vector(model.trainable_variables, rademacher=False)
                v = self._normalize_vectors(v)
                
                # Initial eigenvalue
                eigenvalue = None
                rel_error = float('inf')
                
                # Power iteration
                for j in range(max_iter):
                    # Make v orthogonal to previously computed eigenvectors
                    if eigenvectors:
                        v = self._make_vector_orthogonal(v, eigenvectors)
                        v = self._normalize_vectors(v)
                    
                    # Initialize HVP accumulators
                    _hvp, _eigenvalue = self._compute_hvp(model, x, y, v)
                    num_samples = len(x)
                    
                    # Normalize by number of samples
                    hvp = [h / tf.cast(num_samples, tf.float32) for h in _hvp]
                    current_eigenvalue = _eigenvalue / tf.cast(num_samples, tf.float32)
                    
                    # Normalize the HVP
                    v = self._normalize_vectors(hvp)
                    
                    # Check for convergence
                    if eigenvalue is not None:
                        rel_error = abs(current_eigenvalue - eigenvalue) / (abs(eigenvalue) + 1e-10)
                        if rel_error < tol:
                            #pbar.update(max_iter - j)
                            break
                    
                    eigenvalue = current_eigenvalue
                    pbar.update(j + i*max_iter + 1)
                    # pbar.update(1)
                    # if j % 5 == 0:
                    #     pbar.set_description(f"Eigenvalue {i+1}/{k}, Error: {rel_error:.2e}")
                
                # Store results
                eigenvalues.append(eigenvalue)
                eigenvectors.append(v)
                
                # Update progress description
                #pbar.set_description(f"Eigenvalue {i+1}/{k} = {eigenvalue:.4e}")
            
            #pbar.close()
        
        if verbose:
            print(f"Time taken: {time.time() - start_time:.2f} seconds")
            
        return eigenvalues, eigenvectors
    
    def trace_estimation(self, model, x, y, num_estimates=100, tol=1e-3, verbose=True):
        """
        Estimate the trace of the Hessian using Hutchinson's method
        
        Args:
            x: Input data
            y: Target data
            num_estimates: Maximum number of random vectors to use
            tol: Convergence tolerance
            verbose: Whether to show progress
            
        Returns:
            Estimated trace of the Hessian
        """
        # Start timing
        start_time = time.time()
        
        # Hutchinson's trace estimator
        trace_estimates = []
        current_mean = 0.0
        
        # Set up progress tracking
        with netsurf.utils.ProgressBar(total = num_estimates, prefix = 'Estimating trace') as pbar:
            for i in range(num_estimates):
                # Generate Rademacher random vector
                v = self._generate_random_vector(model.trainable_variables, rademacher=True)
                
                # Initialize accumulators
                num_samples = len(x)
                
                # Compute over batches
                _, vhv = self._compute_hvp(x, y, v)
                
                # Compute batch average
                vhv_estimate = vhv / tf.cast(num_samples, tf.float32)
                trace_estimates.append(vhv_estimate)
                
                # Calculate running mean
                prev_mean = current_mean
                current_mean = np.mean(trace_estimates)
                
                # Update progress
                pbar.update(1)
                pbar.set_description(f"Trace estimate: {current_mean:.4e}")
                
                # Check for convergence
                if i > 10:  # Need a minimum number of samples for stability
                    rel_change = abs(current_mean - prev_mean) / (abs(prev_mean) + 1e-10)
                    if rel_change < tol:
                        pbar.update(num_estimates - i - 1)  # Update remaining steps
                        break
        
        if verbose:
            netsurf.info(f"Time taken: {time.time() - start_time:.2f} seconds")
            netsurf.info(f"Final trace estimate: {current_mean:.6e}")
            
        return current_mean
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # In the original code they also ignore bias!!
        ignore_bias = True

        if 'normalize_score' in kwargs:
            normalize_score = kwargs.pop('normalize_score')
            self.normalize_score = normalize_score
        if 'times_weights' in kwargs:
            times_weights = kwargs.pop('times_weights')
            self.times_weights = times_weights
        if 'ascending' in kwargs:
            ascending = kwargs.pop('ascending')
            self.ascending = ascending

        """ Assertions """
        assert inner_ranking_method in ['same', 'hierarchical', 'msb'], 'Invalid ranking method'
        
        # Initialize dataframe table with all properties for all layers 
        df = []
        num_bits = quantization.m
        use_delta_as_weight = self.use_delta_as_weight
        print(f'[INFO] - {"U" if use_delta_as_weight else "NOT u"}sing delta as weights.')

        # Get supported weights and pruned masks 
        results = netsurf.models.get_supported_weights(model.model, numpy = False, pruned = True, verbose = False)
        supported_weights, supported_pruned_masks, supported_layers, weights_param_num = results
        
        if verbose:
            print(f'[INFO] - Found a total of {len(supported_weights)} supported weights: {", ".join(list(supported_weights.keys()))}')

        # Get layer index per layer in supported_layers
        supported_layers_idxs = {lname: model.model.layers.index(supported_layers[lname]) for lname in supported_layers}

        # Get deltas per weight
        deltas = {kw: netsurf.models.get_deltas(kv, num_bits = num_bits) for kw, kv in supported_weights.items()}
        is_bit_one = {kw: deltas[kw][1] for kw in supported_weights}
        deltas = {kw: deltas[kw][0] for kw in supported_weights}

        # Store the old weights 
        old_weights = copy.deepcopy(supported_weights)

        # Pick the right loss
        loss = model.loss
        if isinstance(loss, str):
            if loss == 'categorical_crossentropy':
                loss = tf.keras.losses.CategoricalCrossentropy()
            elif loss == 'mse' or loss == 'mean_squared_error':
                loss = tf.keras.losses.MeanSquaredError()
            else:
                raise ValueError(f'Loss {model.loss} not supported. Only categorical_crossentropy and mean_squared_error are supported for now.')

        # Loop thru bits and place the deltas 
        for i in range(num_bits):
            
            # Replace the weights with the deltas (if use_delta_as_weight)
            if use_delta_as_weight:
                for w in model.model.weights:
                    kw = w.name
                    if kw in deltas:
                        w.assign(deltas[kw][...,i])
            
            # Perform actual hessian ranking on our model
            hess = fkeras.metrics.HessianMetrics(
                model.model, 
                loss, 
                X, 
                Y,
                batch_size=480
            )

            hess_start = time.time()
            top_k = 8
            BIT_WIDTH = 8
            strategy = "sum"
            # Hessian model-wide sensitivity ranking
            eigenvalues, eigenvectors = hess.top_k_eigenvalues(k=top_k, max_iter=500, rank_BN=False, prefix=f"Bit {i} - " if use_delta_as_weight else "")

            print(f'Hessian eigenvalue compute time: {time.time() - hess_start} seconds\n')
            # eigenvalues = None
            rank_start_time = time.time()

            param_ranking, param_scores = hess.hessian_ranking_general(
                eigenvectors, eigenvalues=eigenvalues, k=top_k, strategy=strategy, iter_by=1
            )

            # First let's get the list of parameters per layer
            num_params_per_layer = []
            cumsum = 0 

            for ily, ly in enumerate(model.model.layers):
                if hasattr(ly, 'layer'):
                    ly = ly.layer
                if ly.__class__.__name__ in fkeras.fmodel.SUPPORTED_LAYERS:
                    #print(ly.name)
                    ps = [tuple(w.shape) for w in ly.trainable_variables]
                    pst = [np.prod(w.shape) for w in ly.trainable_variables]
                    if ignore_bias:
                        ps = [ps[0]]
                        pst = [pst[0]]
                    total = np.sum(pst)
                    cumsum += total
                    # Tuple output (will make debugging easier)
                    t = (ly.name, ily, total, cumsum, pst, ps)
                    # Append to list
                    num_params_per_layer.append(t)

            # Get the cumulative sum of parameters
            cumsum = np.array([t[3] for t in num_params_per_layer])

            # First, we will find to which layer each parameter belongs 
            # Get layers indexes 
            layer_idxs = np.array([t[1] for t in num_params_per_layer])
            param_ly = np.argmax(param_ranking[:,None] < cumsum[None,:], axis = 1)
                
            # Now, within the layer find the actual index 
            for rank, (p, score, ply) in enumerate(zip(param_ranking, param_scores, param_ly)):
                ly_t = num_params_per_layer[ply]
                ly_name = ly_t[0]

                # Remember to subtract the global cumsum (now tht we are looking inside the layer )
                ly_cumsum = 0
                if ply > 0:
                    ly_cumsum = num_params_per_layer[ply-1][3]    
                p_internal = p - ly_cumsum

                # Now get the number of params
                ly_num_params = ly_t[-2]

                # Get the shapes of the layer weights
                ly_shapes = ly_t[-1]

                # Get the cumsum internal to the layer 
                ly_internal_cumsum = np.cumsum(ly_num_params)

                # Get the index of the weight this param belongs to
                wi = np.argmax(p_internal < ly_internal_cumsum)

                # Get the shape of this weight 
                wi_idx = np.unravel_index(p_internal, ly_shapes[wi])

                # Form table entry 
                t = (ly_name, p, score, ply, ly_cumsum, p_internal, wi, wi_idx, ly_shapes[wi])
                #ranking.append(t)


                # Now let's build the table we want for all weights. This table should contain the following information:
                #
                #  | weight               | layer | coord        | value | rank | susceptibility | bit |
                #  +----------------------+-------+--------------+-------+------+----------------+-----+
                #  | conv2d_1[0][0][0][0] | 0     | [0][0][0][0] | 0.45  | ?    | ?              |  0  |
                #
                # Let's build the coordinate string 
                str_coord = '[' + ']['.join(list(map(str,wi_idx))) + ']'
                str_weight_name = f'{ly_name}{str_coord}'

                # Get weight value 
                global_layer_idx = layer_idxs[ply]
                w = model.model.layers[global_layer_idx].get_weights()[wi][wi_idx]
                str_value = str(w)

                # bits 
                str_bits = i

                # Get weight name 
                w_name = model.model.layers[global_layer_idx].weights[wi].name

                # Now build the "pruned" param
                pruned = supported_pruned_masks[w_name][wi_idx]
                str_pruned = pruned.numpy() if pruned is not None else False

                # Param num
                str_param_num = p

                # susceptibility
                suscept = score
                suscept_factor = 2.0**(-i)

                str_rank = rank

                if not use_delta_as_weight:
                    # We need to repeat everything num_bits times cause this is the last iteration (We'll break the loop after this)
                    str_weight_name = np.tile(str_weight_name, num_bits)
                    str_ily = np.tile(global_layer_idx, num_bits)
                    str_coords = np.tile(str_coord, num_bits)
                    str_value = np.tile(str_value, num_bits)
                    str_param_num = np.tile(str_param_num, num_bits)

                    # Redefine bits
                    bits = np.arange(num_bits)
                    str_bits = bits

                    # Redefine susceptibility
                    suscept_factor = 2.0**(-bits)
                    suscept = [score]*num_bits

                    # Str rank
                    str_rank = [rank]*num_bits
                
                # Scores
                # [@manuelbv]: we have two options here, 
                # 1: we just copy the score for all bits 
                # 2: we multiply the score times the delta of each bit (1 for MSB, 0.5 for MSB-1, etc.)
                if inner_ranking_method == 'same':
                    suscept = suscept
                elif inner_ranking_method == 'hierarchical':
                    suscept = suscept*suscept_factor

                # Now let's build the table we want for all weights. This table should contain the following information:
                #
                #  | weight               | layer | coord        | value | rank | susceptibility | bit |
                #  +----------------------+-------+--------------+-------+------+----------------+-----+
                #  | conv2d_1[0][0][0][0] | 0     | [0][0][0][0] | 0.45  | ?    | ?              |  0  |
                #
                # We'll add the rank and susceptibility later
                subT = {'weight': str_weight_name, 'layer': global_layer_idx, 'coord': str_coord, 
                        'value': str_value, 'bit': str_bits, 'pruned' : str_pruned, 'rank' : str_rank, 
                        'param_num': str_param_num, 'susceptibility': suscept}
                if use_delta_as_weight:
                    # We need to pass an index
                    subT = pd.DataFrame(subT, index = [rank])
                else:
                    subT = pd.DataFrame(subT)

                # Append to dfs structure 
                df.append(subT)

            # break if we are using deltas as weights
            if not use_delta_as_weight:
                break

        # concat all dfs 
        df = pd.concat(df, axis = 0).reset_index()

        # Finally, restore the weights of the model
        if use_delta_as_weight:
            for w in model.model.weights:
                kw = w.name
                if kw in deltas:
                    w.assign(old_weights[kw])
            
        self.df = df

        return df


    # Method to actually rank the weights
    @timeranking
    def rank(self, model, X, Y, ascending = False, **kwargs):
        # Call super method to obtain DF 
        df = self.extract_weight_table(model, X, Y, ascending = ascending, **kwargs)

        # Finally, sort by susceptibility 
        df = df.sort_values(by=['pruned','bit','susceptibility'], ascending = [True, False, ascending])
        # Add column with the alias 
        df['alias'] = self.alias
        df['method'] = self.method

        # assign to self 
        self.ranking = Ranking(df, alias = self.alias, method = self.method, filepath = self.path, loaded_from_file = False)
        return self.ranking
    



# Hessian based weight ranker 
@dataclass
class HessianWeightRanker(HessianRanker):
    @property
    def method(self):
        return 'hessian'
    
    @property
    def alias(self):
        alias = 'hessian'
        if self.normalize_score:
            alias += '_norm'
        return alias

""" Same, but using deltas as weights """
@dataclass
class HessianDeltaWeightRanker(HessianRanker):
    def __init__(self,*args, use_delta_as_weight = None, **kwargs):
        """."""
        super().__init__(*args, use_delta_as_weight = True, **kwargs)
    
    @property
    def method(self):
        return 'hessiandelta'

    @property
    def alias(self):
        alias = 'hessiandelta'
        if self.normalize_score:
            alias += '_norm'
        return alias
    

####################################################################################################################
#
# AI-BER RANKING 
#
####################################################################################################################
@dataclass
class AIBerWeightRanker(WeightRanker):
    _ICON = "🤖"
    def extract_weight_table(self, model, X, Y, ascending = False, verbose=False, base_df = None, **kwargs):
        Q = self.quantization
        
        # Call super to get the basic df 
        df = super().extract_weight_table(model, verbose = verbose, ascending = ascending, **kwargs) if base_df is None else base_df
        
        # Make sure susceptibility is float
        df['susceptibility'] = df['susceptibility'].astype(float)

        
        """ We are gonna create a new model that will hold a probability of each weight/bit being flipped.
                We will train this model while keeping the weights/biases of the original model frozen. 
                Because we don't want the model to use too many flips, we will impose a regularization term
                that will penalize the model for using too many flips.
                We'll do this iteratively like so: 
                    1) Train the model (wrt P) for a few epochs
                    2) Get the distribution of P and sort them by probability
                    3) Pick the top values and freeze them. Keep track of the global ranking. These
                        weights/bits will not be flipped from now on.
                    4) Repeat until all bits are frozen OR until there's no improvement in the loss.
        """
        # 1) Clone the model using the Wrapper
        wrapper = netsurf.dnn.aiber.ModelWrapper(model, Q)

        w_model = wrapper.wrapped_model
        
        # 2) Train the model for a few epochs
        history = wrapper.train_P(X[:100], Y[:100], num_epochs = 10)
        
        # Store in place and return 
        return df

    # Method to actually rank the weights
    @timeranking
    def rank(self, model, X, Y, ascending = False, **kwargs):
        # Call super method to obtain DF 
        df = self.extract_weight_table(model, X, Y, ascending = ascending, **kwargs)

        # Finally, sort by susceptibility 
        df = df.sort_values(by=['pruned','bit','susceptibility'], ascending = [True, False, ascending])
        # Add column with the alias 
        df['alias'] = self.alias
        df['method'] = self.method

        # assign to self 
        self.ranking = Ranking(df, alias = self.alias, method = self.method, filepath = self.path, loaded_from_file = False)
        return self.ranking
    
    @property
    def alias(self):
        return 'aiber'





####################################################################################################################
#
# DELTAPOX RANKING (WORK IN PROGRESS...)
#
####################################################################################################################
@dataclass
class QPolarWeightRanker(WeightRanker):
    _ICON = "🧲"

    def extract_weight_table(self, model, X, Y, ascending = False, verbose=False, batch_size = 1000, base_df = None, **kwargs):
        
        # Call super to get the basic df 
        df = WeightRanker.extract_weight_table(self, model, verbose = verbose,**kwargs) if base_df is None else base_df
        # Make sure susceptibility is float
        df['susceptibility'] = df['susceptibility'].astype(float)

        # Compute the impact for the model (considering uncorrupted activations)
        
        # 2) Get the deltas 
        deltas = model.deltas
        deltas = {v.name: deltas[i] for i, v in enumerate(model.trainable_variables)}

        num_batches = int(X.shape[0]//batch_size)

        # Let's get the activation for each layer BUT with full corruption (N=1)
        uncorrupted_output, uncorrupted_activations = model.attack(X, N = 0, return_activations = True)
        corrupted_output, corrupted_activations = model.attack(X, N = 1, return_activations = True)

        # Apply loss to get the error per output 
        loss_corrupted = model.loss(Y, corrupted_output)
        loss_uncorrupted = model.loss(Y, uncorrupted_output)

        # Total corruption in loss:
        delta_loss = loss_corrupted - loss_uncorrupted

        # Print some metrics 
        netsurf.info(f'Stats for maximum attack (N=1) for QModel for input data X ({X.shape}):')
        netsurf.info(f'Loss (corrupted): {loss_corrupted}')
        netsurf.info(f'Loss (uncorrupted): {loss_uncorrupted}')
        netsurf.info(f'Delta loss: {delta_loss}')

        # Get unique indexes for model.metrics_names
        unique_idxs = np.unique([model.metrics_names.index(mname) for mname in model.metrics_names if mname != 'loss'])

        for mname, met in zip(list(np.array(model.metrics_names)[unique_idxs]), 
                              list(np.array(model.metrics)[unique_idxs])):
            # Skip loss
            if mname == 'loss':
                continue
            
            # Compute metric 
            # Reset
            met.reset_states()
            _met = met(Y, corrupted_output).numpy()*1.0
            netsurf.info(f'{mname} (corrupted): {_met}')
            # Reset
            met.reset_states()
            _met = met(Y, uncorrupted_output).numpy()*1.0
            netsurf.info(f'{mname} (uncorrupted): {_met}')
            # Reset
            met.reset_states()
        
        # Plot distribution of error per activation 
        # fig, axs = plt.subplots(len(uncorrupted_activations), 1, figsize = (10,20))
        # for i, (unc, cor) in enumerate(zip(uncorrupted_activations.values(), corrupted_activations.values())):
        #     ax = axs[i]
        #     ax.hist(np.abs(unc - cor).flatten(), bins = 200)
        #     ax.set_title(f'Layer {i}')
        # plt.show()


        # Now compute the impact for each parameter as: act*delta
        P = {}
        for ily, ly in enumerate(model.layers):
            if not hasattr(ly, 'attack'):
                continue 

            # Get the input_tensor name 
            input_tensor = ly.input.name.rsplit('/',1)[0]

            # If we can find it in the activations, we can compute the impact
            if input_tensor not in corrupted_activations:
                continue

            act = corrupted_activations[input_tensor]

            if hasattr(ly, 'compute_impact'):
                # Print message
                netsurf.info(f'Computing impact for layer {ily} ({ly.name})')
                # Just compute the impact by directly calling the layer's method 
                impact = ly.compute_impact(act, batch_size = batch_size)

                # Store in P
                P = {**P, **impact}

        # Plot histogram for each P
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(len(P), 1, figsize = (10,20))
        # for i, (vname, impact) in enumerate(P.items()):
        #     ax = axs[i]
        #     ax.hist(impact.flatten(), bins = 100)
        #     ax.set_title(f'Impact for {vname}')
        # plt.show()
        
        # First, let's sort df by index (equivalent to sort_values(by='param_num') in our case)
        df = df.sort_index()

        # Now let's turn the P into the table we want and add it to the df
        for vname, impact in P.items(): 
            
            if impact.ndim == 5:
                # Make sure impact is a numpy array
                impact = np.array(impact)
                f_P = impact.transpose(3,0,1,2,4).flatten('F')
            elif impact.ndim == 4:
                # Make sure impact is a numpy array
                impact = np.array(impact)
                f_P = impact.transpose(2,0,1,3).flatten('F')
            elif impact.ndim == 3:
                impact = np.array(impact)
                f_P = impact.transpose(1,0,2).flatten('F')
            elif impact.ndim == 2:
                # Make sure impact is a numpy array
                impact = np.array(impact)
                f_P = impact.flatten('F')
            else:
                raise ValueError(f'Impact has invalid shape {impact.shape}.')
            
            # Sanity check
            # k = np.random.randint(len(df[df['param'] == vname])); print(k, ',', impact[tuple(df[df['param'] == vname]['coord'].iloc[k]) + (abs(df[df['param'] == vname]['bit'].iloc[k]),)], '?=', f_P[k])
            try:
                df.loc[df[df['param'] == vname].index, 'impact'] = f_P
            except:
                print('stop here')

        # Plot the dist of impact
        # fig, ax = plt.subplots(1,1, figsize = (10,10))
        # ax.hist(np.abs(df['impact']), bins = 100)
        # ax.set_title(f'Impact distribution')
        # plt.show()

        # Reshuffle
        # Let's randomize before ranking so we get rid of locality 
        df = df.sample(frac=1)

        """ For now, just sort by abs value """
        df['susceptibility'] = np.abs(df['impact'])
        df = df.sort_values(by=['pruned','bit','susceptibility'], ascending = [True, False, ascending])

        return df
    # Method to actually rank the weights
    @timeranking
    def rank(self, model, X, Y, ascending = False, **kwargs):
        # Make sure the model has extracted the deltas 
        model.compute_deltas()
        
        # Call super method to obtain DF 
        df = self.extract_weight_table(model, X, Y, ascending = ascending, **kwargs)

        # Finally, sort by susceptibility 
        df = df.sort_values(by=['pruned','bit','susceptibility'], ascending = [True, False, ascending])
        # Add column with the alias 
        df['alias'] = self.alias
        df['method'] = self.method

        # Create ranking object
        self.ranking = Ranking(df, alias = self.alias, method = self.method, filepath = self.path, loaded_from_file = False)

        return self.ranking

    @property
    def alias(self):
        return 'qpolar'


@dataclass
class QPolarGradWeightRanker(QPolarWeightRanker, GradRanker):
    _ICON = "🧲"
    def __init__(self, *args, **kwargs):
        super(GradRanker, self).__init__(*args, **kwargs)
        super(QPolarGradWeightRanker, self).__init__(*args, **kwargs)
        

    def extract_weight_table(self, model, X, Y, ascending = False, verbose=False, 
                             batch_size = 1000, **kwargs):
        # At this point we will have the susceptibility in terms of qpolar impact.
        df = QPolarWeightRanker.extract_weight_table(self, model, X, Y, verbose = verbose, 
                                                    ascending = ascending, batch_size = batch_size,
                                                    **kwargs)
        
        
        # Make sure we create a DEEP copy of df, cause this is pointing to self.df and it will be modified in the 
        # nxt call
        df_polar = df.copy()
        
        # Now let's get the gradients for all trainable variables
        df_grad = GradRanker.extract_weight_table(self, model, X, Y, verbose = verbose,
                                                    ascending = ascending, batch_size = batch_size, 
                                                    normalize_score = False, 
                                                    times_weights = False,
                                                    absolute_value = False, bit_value = None,
                                                    **kwargs)
        
        # Copy 
        df_grad = df_grad.copy()
        
        # Make sure self.df is None at this point (sanity check)
        self.ranking = None 

        # Now we will multiply the impact by the gradient (element-wise)
        # This will give us the final ranking
        # IT should not matter cause pandas takes care of this internally, BUT
        # just as a sanity check, let's sort indexes before multiplying
        df_polar = df_polar.sort_index()
        df_grad = df_grad.sort_index()

        df_polar['impact_times_gradient'] = df_polar['impact']*df_grad['susceptibility']

        # fig, axs  = plt.subplots(3,1, figsize = (10,10))
        # axs[0].hist(df_polar['impact'], bins = 100)
        # axs[0].set_title(f'Impact distribution')
        # axs[0].set_yscale('log')

        # axs[1].hist(df_grad['susceptibility'], bins = 100)
        # axs[1].set_title(f'Gradient susceptibility distribution')
        # axs[1].set_yscale('log')

        # axs[2].hist(df_polar['susceptibility'], bins = 100)
        # axs[2].set_title(f'Final susceptibility distribution')
        # # set y-axis to log scale
        # axs[2].set_yscale('log')
        # plt.show()

        # Get the absolute value 
        df_polar['susceptibility'] = np.abs(df_polar['impact_times_gradient'])
        # Add column with the alias 
        df_polar['alias'] = self.alias
        df_polar['method'] = self.method

        return df_polar

    def rank(self, model, X, Y, ascending=False, **kwargs):
        return QPolarWeightRanker.rank(self, model, X, Y, ascending = ascending, **kwargs)

    @property
    def alias(self):
        return 'qpolargrad'


"""
Fisher weight ranker
"""
@dataclass
class FisherWeightRanker(WeightRanker):
    _ICON = "🐟"


    def extract_weight_table(self, model, X, Y, ascending = False, verbose=False, batch_size = 1000, base_df = None, **kwargs):
        
        
        # Call super to get the basic df 
        df = WeightRanker.extract_weight_table(self, model, verbose = verbose, ascending = ascending, **kwargs) if base_df is None else base_df

        # Make sure susceptibility is float
        df['susceptibility'] = df['susceptibility'].astype(float)

        # MAKE SURE YOU ARE SORTED BY BIT AND INTERNAL PARAM NUM
        df = df.sort_values(by=['param','bit', 'internal_param_num'], ascending=[False,False, True])

        # Step 2: Compute Fisher diagonal (gradient^2 per parameter)
        with tf.GradientTape() as tape:
            preds = model(X, training=False)
            loss = model.loss(Y, preds)
            loss = tf.reduce_mean(loss)

            if model.losses:
                loss += tf.add_n(model.losses)

        grads = tape.gradient(loss, model.trainable_variables)
        fisher_diagonals = [tf.square(g).numpy().flatten() if g is not None else np.zeros_like(v.numpy().flatten()) 
                            for g, v in zip(grads, model.trainable_variables)]

        # Step 3: Assign to df
        for v, diag in zip(model.trainable_variables, fisher_diagonals):
            name = v.name
            # Repeat per bit
            for b in df['bit'].unique():
                df.loc[(df['param'] == name) & (df['bit'] == b), 'fisher'] = diag

        # Susceptibility is just fisher
        df['susceptibility'] = df['fisher']

        return df
    
    """ Rank """
    @timeranking
    def rank(self, model, X, Y, ascending=False, **kwargs):
        df = self.extract_weight_table(model, X, Y, ascending=ascending, **kwargs)
        df = df.sort_values(by=['pruned', 'bit', 'susceptibility'], ascending=[True, False, ascending])
        # Add column with the alias 
        df['alias'] = self.alias
        df['method'] = self.method

        self.ranking = self.ranking = Ranking(df, alias = self.alias, method = self.method, filepath = self.path, loaded_from_file = False)
        return self.ranking

    @property
    def alias(self):
        return 'fisher'




""" ###########################################################################################
# 
# RANKING COMPARISON, PROFILERS
#
### ###########################################################################################
""" 
@dataclass
class RankingComparator:
    """
    Class to compare different ranking methods
    """
    baseline: str = None
    granularity: float = 0.1
    rankers: List[WeightRanker] = field(default_factory=list)
    comparison: pd.DataFrame = field(init=False)

    def __post_init__(self):
        if not self.rankers:
            raise ValueError("No rankers provided.")
        # Check if baseline is in rankers
        if self.baseline and not any(r.alias == self.baseline for r in self.rankers):
            raise ValueError(f"Baseline '{self.baseline}' not found in rankers.")
        # Store _keys with aliases names 
        self._keys = [ranker.alias for ranker in self.rankers]


    """ Make sure we define iter so we can do stuff like 'if <str> in <RankingComparator>' """
    def __iter__(self):
        """ Iterate over rankers """
        for ranker in self.rankers:
            yield ranker.alias

    """ Length """
    def __len__(self):
        """ Get length """
        return len(self.rankers)

    """ Get item """
    def __getitem__(self, item):
        if isinstance(item, str):
            # Get the ranker by alias
            if item not in self._keys:
                raise KeyError(f"Ranker '{item}' not found.")
            idx = self._keys.index(item)
            return self.rankers[idx]
        elif isinstance(item, int):
            # Get the ranker by index
            if item < 0 or item >= len(self.rankers):
                raise IndexError(f"Ranker index '{item}' out of range.")
            return self.rankers[item]

    def save_rankings(self, dir):
        """
        Save the rankings to a directory.
        """
        if not os.path.exists(dir):
            os.makedirs(dir)
        for ranker in self.rankers:
            ranker.save_ranking(os.path.join(dir, ranker.alias))

    @staticmethod
    def from_directory(path: str, rankers: List[str] = None, granularity: float = 0.1, baseline: str = None, 
                       config_per_methods = {}, **kwargs) -> 'RankingComparator':
        # First check if the path exists. This should be the "experiments" inside the hierarchy benchmarks/<benchmark>/q<m>_<n>_<s>/<model_alias>/experiments
        # then we will pass this path when building the rankers 
        if rankers is None:
            rankers = ['random', 'hessian', 'hessiandelta', 'fisher', 'qpolar', 'qpolargrad', 'hirescam', 'hiresdelta', 'bitwise_msb', 'weight_abs_value']
        # We can now pass this path to the rankers and they will handle the reloading
        # Create rankers
        # Make sure we have an entry for all rankers in config_per_methods, even if it's empty
        config_per_methods = {ranker: config_per_methods.get(ranker, {}) for ranker in rankers}
        # Make sure we pop "method" from the configs, if in there
        config_per_methods = {ranker: {k: v for k, v in config_per_methods[ranker].items() if k != 'method'} for ranker in rankers}

        # We need to parse the "method_kws" here and unroll them
        for r in config_per_methods:
            if 'method_kws' in config_per_methods[r]:
                # This has the form method_kws = "argument_a=value_a argument_b=value_b"
                groups = config_per_methods[r]['method_kws'].split(' ')
                for k,v in [g.split('=') for g in groups]:
                    # try to eval
                    try:
                        v = eval(v)
                    except:
                        pass
                    # Add to config
                    config_per_methods[r][k] = v
                del config_per_methods[r]['method_kws']
        rankers = [WeightRanker.build(ranker, parent_path = os.path.join(path,ranker), **config_per_methods[ranker], **kwargs) for ranker in rankers]
        # Create the comparator
        return RankingComparator(rankers=rankers, granularity = granularity, baseline=baseline)

    @staticmethod 
    def from_object(rankers: List[str] = None, granularity: float = 0.1, baseline: str = None, **kwargs) -> 'RankingComparator':
        """
        Create a RankingComparator from a model and data.
        """
        if rankers is None:
            rankers = ['random', 'hessian', 'hessiandelta', 'fisher', 'qpolar', 'qpolargrad', 'hirescam', 'hiresdelta', 'bitwise_msb', 'weight_abs_value']
        # Create rankers
        rankers = [WeightRanker.build(ranker, **kwargs) for ranker in rankers]
        return RankingComparator(rankers=rankers, granularity=granularity, baseline=baseline)

    @staticmethod
    def from_rankers(rankers: List[WeightRanker], granularity: float = 0.1, baseline: str = None) -> 'RankingComparator':
        """
        Create a RankingComparator from a list of rankers.
        """
        return RankingComparator(rankers=rankers, granularity = granularity, baseline=baseline)
    
    def _bootstrap_corr_ci(self, x, y, method="kendall", n_boot=100, ci=0.95):
        """Bootstrap CI for Kendall/Spearman correlation."""
        stats = []
        x, y = np.asarray(x), np.asarray(y)
        for _ in range(n_boot):
            idx = np.random.choice(len(x), size=len(x), replace=True)
            if method == "kendall":
                val, _ = kendalltau(x[idx], y[idx])
            elif method == "spearman":
                val, _ = spearmanr(x[idx], y[idx])
            else:
                raise ValueError(f"Unsupported method: {method}")
            stats.append(val)
        stats = np.sort(stats)
        low = np.percentile(stats, (1 - ci) / 2 * 100)
        high = np.percentile(stats, (1 + ci) / 2 * 100)
        return low, high

    """ Extract weight table (for specified ranker methods or for all) """
    def rank(self, ranker: str, model, X, Y, ascending=False, **kwargs):
        # Make sure ranker is in self
        if ranker not in self._keys:
            raise KeyError(f"Ranker '{ranker}' not found.")

        # Check if rank already exists
        if self[ranker].ranking is not None:
            if not self[ranker].ranking.is_empty:
                return self[ranker].ranking
        
        # Rank
        base_df = WeightRanker.extract_weight_table(self[0], model, X, Y, ascending=ascending, **kwargs)
        return self[ranker].rank(model, X, Y, ascending=ascending, base_df = base_df, **kwargs)
 
    """ Perform the actual comparison of rankers against the baseline """
    def compare_rankers(self, granularity: float = 0.1) -> pd.DataFrame:

        """
        Computes a dataframe with Kendall, Spearman and Jaccard overlaps between rankers
        at different protection levels with respect to the baseline.
    
        Args:
            granularity: Step size for protection levels (between 0 and 1).
    
        Returns:
            DataFrame with columns: ['protection', 'method1', 'method2', 'baseline', 'kendall', 'spearman', 'jaccard']
        """
        if not self.baseline:
            if 'random' in self._keys:
                self.baseline = 'random'
                netsurf.info(f"Using 'random' as baseline, since no baseline was provided.")

        # Make sure that the rankers have ranked the weights
        not_yet_ranked = [ranker.alias for ranker in self.rankers if ranker.df is None]
        if len(not_yet_ranked) > 0:
            raise ValueError(f"Rankers {', '.join(not_yet_ranked)} have not been ranked yet.")

        baseline_ranker = next(r for r in self.rankers if r.alias == self.baseline)
        n_total = len(baseline_ranker.df)
        min_granularity = 1.0 / n_total
        if granularity < min_granularity:
            raise ValueError(f"Granularity too small. Minimum granularity is {min_granularity:.6f}.")

        rows = []

        protection_levels = np.arange(granularity, 1 + granularity, granularity)
        baseline_ranks = baseline_ranker.df.sort_values(by="rank").index.to_list()

        for method in self.rankers:
            if method.alias == self.baseline:
                continue

            method_ranks = method.df.sort_values(by="rank").index.to_list()

            for p in protection_levels:
                k = int(p * n_total)
                if k == 0 or k > n_total:
                    continue

                top_baseline = set(baseline_ranks[:k])
                top_method = set(method_ranks[:k])

                # Jaccard similarity
                intersection = len(top_baseline & top_method)
                union = len(top_baseline | top_method)
                jaccard = intersection / union if union > 0 else 0.0
                jaccard_sym = intersection / ((len(top_baseline) + len(top_method)) / 2)

                # Get full rank list for metrics that need order
                full_baseline_ranks = np.argsort(np.argsort(baseline_ranks))
                full_method_ranks = np.argsort(np.argsort(method_ranks))
                kendall_corr, _ = kendalltau(full_baseline_ranks, full_method_ranks)
                spearman_corr, _ = spearmanr(full_baseline_ranks, full_method_ranks)
                kendall_low, kendall_high = self._bootstrap_corr_ci(full_baseline_ranks, full_method_ranks, "kendall")
                spearman_low, spearman_high = self._bootstrap_corr_ci(full_baseline_ranks, full_method_ranks, "spearman")

                rows.append({
                    "protection": p,
                    "method1": method.alias,
                    "method2": self.baseline,
                    "baseline": self.baseline,
                    "kendall": kendall_corr,
                    "kendall_ci_low": kendall_low,
                    "kendall_ci_high": kendall_high,
                    "spearman": spearman_corr,
                    "spearman_ci_low": spearman_low,
                    "spearman_ci_high": spearman_high,
                    "jaccard": jaccard,
                    "jaccard_sym": jaccard_sym,
                })

        df = pd.DataFrame(rows)
        return df
    
    def plot_radar_overlap(self, protection_levels=[0.2, 0.4, 0.8, 1.0], axs = None, show = True):
        
        # Check if axs has the right length
        if axs is not None:
            if len(axs) != len(metrics):
                # set back to None
                print('axs has wrong length, creating new ones')
                axs = None

        # Check if we need to create figure
        show &= (axs is None)
        if axs is None:
            fig, axs = plt.subplots(len(metrics), 1, subplot_kw=dict(polar=True), figsize=(6, 12))
            fig.suptitle("Ranking Overlap", fontsize=16)
        else:
            fig = axs[0].figure
            fig.suptitle("Ranking Overlap", fontsize=16)

        methods = self.comparison['method1'].unique()
        metrics = ['kendall', 'spearman', 'jaccard', 'jaccard_sym']

        for p in protection_levels:
            dfp = self.comparison[self.comparison['protection'].round(2) == round(p, 2)]

            fig.suptitle(f"Ranking Overlap @ Protection {int(p):3.2%}", fontsize=16)

            for i, metric in enumerate(metrics):
                values = []
                labels = []
                for m in methods:
                    val = dfp[dfp['method1'] == m][metric].values[0]
                    values.append(val)
                    labels.append(m)
                values += values[:1]
                angles = [n / float(len(labels)) * 2 * np.pi for n in range(len(labels))]
                angles += angles[:1]

                axs[i].plot(angles, values, marker='o')
                axs[i].fill(angles, values, alpha=0.25)
                axs[i].set_title(metric.upper(), size=13)
                axs[i].set_xticks(angles[:-1])
                axs[i].set_xticklabels(labels)
                axs[i].set_yticks([0.25, 0.5, 0.75, 1.0])

            if show:
                plt.tight_layout()
                plt.show()
            
            return fig, axs
    
    def plot_overlap_curves(self, axs = None):

        # Check if axs has the right length
        if axs is not None:
            if len(axs) != 3:
                # set back to None
                print('axs has wrong length, creating new ones')
                axs = None

        # Check if we need to create figure
        show &= (axs is None)
        if axs is None:
            fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 12))
            fig.suptitle("Ranking Overlap", fontsize=16)
        else:
            fig = axs[0].figure
            fig.suptitle("Ranking Overlap", fontsize=16)
        
        metrics = ['kendall', 'spearman', 'jaccard', 'jaccard_sym']
        
        for i, metric in enumerate(metrics):
            sns.lineplot(data=self.comparison, x='protection', y=metric, hue='method1', ax=axs[i])
            axs[i].set_title(f'{metric.upper()} vs Protection')
            axs[i].set_ylim(0, 1)

        if show:
            plt.tight_layout()
            plt.show()
        
        return fig, axs
    
    def plot_cumulative_overlap(self, ax = None):
        
        # Check if we need to create figure
        show &= (ax is None)
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            fig.suptitle("Ranking Overlap", fontsize=16)
        else:
            fig = ax.figure
            fig.suptitle("Ranking Overlap", fontsize=16)
        
        avg_df = self.comparison.groupby('method1')[['kendall', 'spearman', 'jaccard', 'jaccard_sym']].mean().reset_index()
        melted = avg_df.melt(id_vars='method1', var_name='metric', value_name='value')

        sns.barplot(data=melted, x='method1', y='value', hue='metric', ax = ax)
        plt.title('Average Overlap Across Protection Levels')
        plt.xticks(rotation=45)

        if show:
            plt.tight_layout()
            plt.show()
        
        return fig, ax

