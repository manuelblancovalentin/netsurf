""" Code for weight ranking according to different methods """

# Basic
from dataclasses import dataclass, field
from typing import Optional, List, Dict

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

""" pergamos"""
import pergamos as pg


# Define emojis per method 
_EMOJIS = {'random': '🎲',
            'bitwise': '💡',
            'layerwise': '📚',
            'weight_abs_value': '🗿',
            'grad': '💈',
            'graddelta': '',
            'recursive_uneven': '🔄',
            'diffbitperweight': '🔢',
            'hessian': '👴🏻',
            'hessiandelta': '👵🏻',
            'qpolar': '🧲',
            'qpolargrad': '🔥',
            'fisher': '🐟',
            'aiber': "🤖"}

"""
    Decorator to print the time it takes to run a function
"""
def timeranking(func):
    def wrapper(self, model, *args, verbose = True, **kwargs):
        # Start timer
        start_time = time.time()

        if self.ranking.empty:
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
            # Store in global_metrics
            self.ranking.metrics['ranking_time'] = elapsed_time

            if verbose: netsurf.info(f'Ranking done in {elapsed_time:3.2f} seconds')
        else:
            if verbose: netsurf.info(f'Ranking already done. Skipping...')

        
        return self.ranking
    return wrapper



####################################################################################################################
#
# RANKING CONFIG OBJECT FOR EASY COMPARISON BETWEEN EXPS/RANKINGS
#
####################################################################################################################
@dataclass
class RankingConfig:
    config: dict = field(default_factory=dict)
    # Note that hash is a dynamic property that gets computed everytime we call it. This ensures we are always up-to-date

    # Class-level default values for all rankings
    DEFAULTS = {"method": "undefined",
                "quantization": "undefined",
                "suffix": "",
                "ascending": False,
                "normalize_score": False,
                "use_delta_as_weight": False,
                "batch_size": 1000,
                "absolute_value": True}
    
    def __post_init__(self):
        # Inject default values if not present
        for key, default_value in self.DEFAULTS.items():
            self.config.setdefault(key, default_value)
        # Make sure that alias is never in our config 
        # This is because we want to use it as a tag for the ranking
        self.config.pop("alias", None)

    @staticmethod
    def from_hash(hash_id: str):
        # Revese-engineer the hash to get the config and initialize a new object
        return RankingConfig(netsurf.utils.compressed_id_to_config(hash_id))

    def __getattr__(self, name):
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"'RankingConfig' object has no attribute '{name}'")

    @property
    def alias(self):
        s = f"{self.method}_{self.quantization}"
        # Method specific translation
        if self.method == 'bitwise':
            s += "_msb" if not self.ascending else "_lsb"
        elif self.method == 'layerwise':
            s += '_first' if self.ascending else "_last"
        else:
            s += "_ascending" if self.ascending else "_descending"

        # Now the common terms that only matter if they are true 
        if self.normalize_score:
            s += "_norm"
        if self.use_delta_as_weight:
            s += "_deltaweight"
        
        # If there are any other config options, add them to the alias
        for k, v in self.config.items():
            # Skip the ones we already added
            if k in ["method", "quantization", "suffix", "alias", "ascending", "normalize_score", "use_delta_as_weight", "batch_size"]:
                if self.method in ['layerwise', 'random', 'bitwise', 'weight_abs_value', 'fisher', 'aiber']:
                    # Skip the absolute value flag
                    if k == "absolute_value":
                        continue
                continue
            # Add them to the alias
            s += f"_{k}_{v}"
        
        # Add suffix if any
        s += self.suffix

        return s
        
    @property 
    def hash_id(self) -> str:
        """ Dynamically compute hash based on current config """
        return netsurf.utils.config_to_compressed_id(self.config)
        
    def __eq__(self, other):
        if not isinstance(other, RankingConfig):
            return False
        return self.core_config == other.core_config

    def __hash__(self):
        return hash(self.hash_id)
    
    def to_dict(self):
        return {
            "config": {**self.config, "alias": self.alias},
            "hash_id": self.hash_id,

        }
    
    @classmethod
    def from_dict(cls, d):
        cfg = d["config"]
        # pop alias
        cfg.pop("alias", None)
        return cls(cfg)

    def matches(self, other_dict):
        other_dict = {k: v for k,v in other_dict.items() if k != "alias"}
        normalized = {**self.DEFAULTS, **other_dict}
        return self.config == normalized

    def save_json(self, path):
        # Make sure dir exists
        if not os.path.exists(os.path.dirname(path)):
            netsurf.info(f"Creating directory {os.path.dirname(path)} for config file")
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        # Let user know
        netsurf.info(f"Config {self} saved to {path}")

    @staticmethod
    def load_json(path):
        with open(path, "r") as f:
            return RankingConfig.from_dict(json.load(f))

    def __repr__(self):
        return (f"<RankingConfig hash={self.hash_id}, "
                f"config={json.dumps(self.config, sort_keys=True)}>")

    def short_repr(self):
        return self.alias
    
    """
        Defining __iter__, items, keys, values and __getitem__ makes it possible for us to pass config objects like so:
                func(**<RankingConfig>)
            without explicitly doing:
                func(**<RankingConfig>.config)
    """
    def __iter__(self):
        return iter(self.config)

    def items(self):
        return self.config.items()

    def keys(self):
        return self.config.keys()

    def values(self):
        return self.config.values()

    def __getitem__(self, key):
        return self.config[key]
    


####################################################################################################################
#
# RANK OBJECT
#
####################################################################################################################
@dataclass
class Ranking:
    ranking: pd.DataFrame
    config: RankingConfig
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Internal props
    filepath: Optional[str] = None
    loaded_from_file: Optional[bool] = False
    _ICON = "🏆"

    # Class-level default values for all rankings
    DEFAULTS = {"ranking_time": "undefined"}
    
    def __post_init__(self):
        # Inject default values if not present
        for key, default_value in self.DEFAULTS.items():
            self.metrics.setdefault(key, default_value)
        # Try to infer the _ICON from the method (in config)
        self._ICON = _EMOJIS.get(self.config.method.lower(), "🏆")

    @property
    def empty(self):
        if self.ranking is None:
            return True
        return self.ranking.empty

    @property
    def hash(self):
        """Human-friendly config hash alias (e.g., for filenames or logs)
            This doesn't collide with __hash__().
        """
        return self.config.hash_id

    # Get attributes directly from config
    def __getattr__(self, name):
        # Only called if the attribute wasn't found the usual way
        if hasattr(self.config, name):
            return getattr(self.config, name)
        raise AttributeError(f"'Ranking' object has no attribute '{name}'")

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
    def read_csv(file, comment_char="#", metadata_required_keys=None):
        """
        Reads a CSV file with JSON metadata embedded in comment lines at the top.

        Parameters:
            file (str): Path to the CSV file.
            comment_char (str): Character used to mark comment lines (default: "#").
            metadata_required_keys (list[str], optional): If provided, checks that these keys exist in metadata.

        Returns:
            metadata (dict): Parsed metadata dictionary.
            df (pandas.DataFrame): DataFrame with CSV contents.
        """
        with open(file, "r") as f:
            lines = f.readlines()

        # Separate metadata lines and data lines
        meta_lines = []
        data_start_idx = 0
        for i, line in enumerate(lines):
            if line.startswith(comment_char):
                meta_lines.append(line[len(comment_char):].strip())
            else:
                data_start_idx = i
                break

        # Parse metadata
        comments = json.loads("\n".join(meta_lines)) if meta_lines else {}

        # Optional metadata validation
        if metadata_required_keys:
            for key in metadata_required_keys:
                if key not in comments:
                    raise ValueError(f"Missing required metadata key: {key}")

        # Read CSV data (ignore comment lines automatically)
        df = pd.read_csv(file, comment=comment_char)

        return comments.get("config", {}), df, comments.get("metrics", {})

    @staticmethod 
    def from_file(filepath, config = None):
        # check file 
        if not netsurf.utils.path_exists(filepath):
            # Warn the user that the file doesn't exist and initialize and empty ranking
            netsurf.error(f"File {filepath} not found. Initializing empty ranking.")
            # Create an empty ranking
            return Ranking(pd.DataFrame(), config = RankingConfig(), metrics = {}, filepath = filepath, loaded_from_file = False)
        
        # Load the ranking
        config, df, metrics = Ranking.read_csv(filepath, comment_char="#")
        # Initialize Config object 
        config = RankingConfig(config)

        # Initialize 
        netsurf.info(f"Loading ranking from {filepath}")
        return Ranking(df, config = config, metrics = metrics, filepath = filepath, loaded_from_file = True)

    def save(self, filepath = None, overwrite = False):
        # We have to check if the filepath is a file or a folder. If it's a folder, we need to add the filename
        # to the folder. If it's a file, we need to check if it exists and if it does, we need to replace it.
        if filepath is None:
            if self.filepath is None:
                raise ValueError("No filepath provided. Cannot save ranking.")
            filepath = self.filepath
        # Check if the filepath is a file or a folder
        if not filepath.endswith('.csv'):
            if os.path.exists(filepath):
                if os.path.isdir(filepath):
                    filepath = os.path.join(filepath, "ranking.csv")
            else:
                # This is just a directory name
                filepath = os.path.join(filepath, "ranking.csv")
        # if filepath is None, replace with this now
        if os.path.exists(filepath) and not overwrite:
            # Update filepath
            self.filepath = filepath
            # Warn the user that the file already exists and ask if they want to overwrite it
            netsurf.warn(f"File {filepath} already exists. Use overwrite=True to overwrite it.")
            return 
        
        # Save the config as well
        config_file = os.path.join(os.path.dirname(filepath), "config.json")
        self.config.save_json(config_file)
        # Save the ranking
        Ranking.to_csv(self, filepath, index = False)
        netsurf.info(f"Ranking saved to {filepath} ... Internal filepath definition in object updated.")
    

    @staticmethod 
    def to_csv(ranking: 'Ranking', filepath, index=False, comment_char="#"):
        """
        Saves a Ranking to a CSV file with metadata (as JSON) prepended in comment lines.
        """
        if not isinstance(ranking, Ranking):
            raise ValueError("ranking must be a Ranking object")
        # Check if the directory exists
        if not os.path.exists(os.path.dirname(filepath)):
            # Create the directory
            netsurf.info(f"Creating directory {os.path.dirname(filepath)} for ranking file")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
        with open(filepath, "w", newline="") as f:
            # ✅ Fix: serialize config dict, not the object
            metadata = {
                "config": ranking.config.to_dict()["config"],
                "metrics": ranking.metrics
            }
            json_str = json.dumps(metadata, indent=2)
            for line in json_str.splitlines():
                f.write(f"{comment_char} {line}\n")

            # Save DataFrame
            ranking.ranking.to_csv(f, index=index)


    def plot_ranking(self, axs = None, w = 300, show = True):
        
        # Fields and titles
        items = [('bit','Bit number', lambda x: x, 'green'), 
                 ('value', 'Param Value', lambda x: x, 'orange'),
                 #('binary', 'Num Ones (bin)', lambda x: [np.sum([int(i) for i in xx]) for xx in x] , 'blue'),
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
    config: RankingConfig = field(default_factory=RankingConfig, repr=False)
    ranking: Optional[pd.DataFrame] = pd.DataFrame()

    # Internal things just for the weight ranker state
    complete_ranking: Optional[bool] = True
    path: Optional[str] = "."
    ranking_time: Optional[float] = None
    reload_ranking: Optional[bool] = True
    
    _ICON = "🏆"


    def __post_init__(self, *args, **kwargs):
        # Initialize the dataframe
        self.ranking = pd.DataFrame()
        
        if self.path is None:
            self.path = "."

        # If the path is not None, make sure it contains the hash at the end.
        hash = self.config.hash_id
        if not self.path.endswith(hash):
            # Check if the path already contains the hash
            if hash not in self.path:
                # Add the hash to the path
                self.path = os.path.join(self.path, hash)
        
        # If not reload, leave
        if not self.reload_ranking:
            # Nothing else to do here
            netsurf.info(f"Ranker {self.method} initialized and linked @ {self.path}")
            return

        # Check path is a folder 
        if not netsurf.utils.is_valid_directory(self.path):
            # Directory doesn't exist, so there's nothing to load 
            netsurf.warn(f"Path {self.path} is not a valid directory. Cannot load ranking. Skipping.")
            return
        
        # IF self.reload_ranking, we can try to check if the file exists and if we can reload it. 
        # Check if the ranking file exists
        ranking_path = self.path
        if not self.path.endswith('.csv'):
            ranking_path = os.path.join(self.path, "ranking.csv")
        r = Ranking.from_file(ranking_path, config = {k: v for k,v in self.config.items() if k!='hash_id' and k != 'alias'})
        # Hash will not coincide because we initialized our config to something random. Now we are loading it from file.
        # It only has to coincide if pd.Dataframe is not empty. 
        if not self.ranking.empty and not r.empty:
            # make sure hash matches 
            if r.hash != self.config.hash_id:
                # Hashes don't match, ignore file
                netsurf.error(f"Hash {r.hash} from loaded file does not match current config hash {self.config.hash_id}. Ignoring file. This probably indicates a possible corrupted file!")
                return 
        
        # Else, set the ranking to the loaded one
        self.ranking = r
        netsurf.info(f"Ranker {self.method} initialized and linked @ {self.path} with loaded ranking from file.")


    # Get attributes directly from config
    def __getattr__(self, name):
        # Only called if the attribute wasn't found the usual way
        if hasattr(self.config, name):
            return getattr(self.config, name)
        raise AttributeError(f"'Ranker' object has no attribute '{name}'")

    
    @staticmethod
    # Function to create a weight ranker given a couple of flags 
    def build(method: str, quantization: 'QuantizationScheme', *args, config = {}, **kwargs):
        options = {'random': RandomWeightRanker, 
                    'weight_abs_value': AbsoluteValueWeightRanker,
                    'layerwise': LayerWeightRanker,
                    'bitwise': BitwiseWeightRanker,
                    'grad': GradWeightRanker,
                    'graddelta': GradDeltaWeightRanker,
                    'recursive_uneven': RecursiveUnevenRanker,
                    'diffbitperweight': DiffBitsPerWeightRanker,
                    'hessian': HessianWeightRanker,
                    'hessiandelta': HessianDeltaWeightRanker,
                    'qpolar': QPolarWeightRanker,
                    'qpolargrad': QPolarGradWeightRanker,
                    'fisher': FisherWeightRanker,
                    'aiber': AIBerWeightRanker}
        
        # Parse config
        if 'method_kws' in config:
            # This has the form method_kws = "argument_a=value_a argument_b=value_b"
            groups = config['method_kws'].split(' ')
            for k,v in [g.split('=') for g in groups]:
                # try to eval
                try:
                    v = eval(v)
                except:
                    pass
                # Add to config
                config[k] = v
            del config['method_kws']
        
        # Same for method_suffix:
        if 'method_suffix' in config:
            config['suffix'] = config['method_suffix'] if config['method_suffix'] is not None else ""
            del config['method_suffix']
            
        # Make sure to add the quantization scheme string to the config
        config['quantization'] = quantization._scheme_str

        # Create ranking config
        ranker_config = RankingConfig(config)

        # if method in options, this is the actual name (e.g. bitwise) we need to use to pick the right class, not "method" (which is the alias, e.g. bitwise_msb)
        if 'method' in config:
            method = config['method']
        return options.get(method.lower(), WeightRanker)(quantization, *args, config = ranker_config, **kwargs)
    
    """
        Save rankng into csv file (with config as header comments and also save config.json)
    """
    def save_ranking(self, filepath: str = None):
        # Save the ranking to file
        if self.ranking is not None:
            if filepath is None:
                if self.ranking.filepath is None:
                    # Nothing to do here, return 
                    raise ValueError("No filepath provided and no internal filepath defined. Cannot save ranking.")
                
                filepath = self.ranking.filepath
            # Update the internal filepath to make sure it's always up to date
            self.ranking.filepath = filepath
            # make sure directory exists
            if not os.path.exists(os.path.dirname(filepath)):
                # Create the directory
                netsurf.info(f"Creating directory {os.path.dirname(filepath)}")
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
            # Finally save
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

        # NOTE: VERY IMPORTANT!!! Our global_param_num does NOT take into consideration the bits. That is, it's just the
        # number of the parameter (for all bits). This will be an issue later, when we compare the rankings. 
        # So this is what we will do: We will use fractional numbers as a global_param_num_bit. Say we have 4 bits. 
        # Say a weight has a global_param_num of 13200. Then:
        #  - bit 0 -> global_param_num_bit = 13200 + (1/4)*0 
        #  - bit 1 -> global_param_num_bit = 13200 + (1/4)*1
        #  - bit 2 -> global_param_num_bit = 13200 + (1/4)*2
        #  - bit 3 -> global_param_num_bit = 13200 + (1/4)*3
        # You see this does not alter the global structure because there are no more bits  for this weight.
        # Also, you can retrieve the original ones by just doing np.floor(global_param_num_bit)
        # Because our bits go from Q.n + Q.s - 1 to -Q.f - 1, we need to subtract -Q.n + Q.s - 1 to the bit number
        df['global_param_num_bit'] = df['global_param_num'] + (df['bit'] - Q.n - Q.s + 1)/Q.m

        # Let's randomize before ranking so we get rid of locality 
        df = df.sample(frac=1)

        return df

    
    def plot_ranking(self, axs = None, w = 300, show = True):
        
        # Fields and titles
        items = [('bit','Bit number', lambda x: x, 'green'), 
                 ('value', 'Param Value', lambda x: x, 'orange'),
                 #('binary', 'Num Ones (bin)', lambda x: [np.sum([int(i) for i in xx]) for xx in x] , 'blue'),
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
        df = self.extract_weight_table(model, *args, **self.config) if base_df is None else base_df

        # Susceptibility here is considered uniform (hence the randomness assigning TMR)
        df['susceptibility'] = [1/(len(df))]*len(df)
        df['rank'] = np.random.permutation(np.arange(len(df)))
        # Sort by rank
        df = df.sort_values(by='rank').reset_index(drop=True)
        # Add column with the alias 
        df['alias'] = self.alias
        df['method'] = self.method

        # assign to self 
        self.ranking = Ranking(df, config = self.config, filepath = self.path, loaded_from_file = False)

        return self.ranking


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
    def rank(self, model, *args,  base_df = None, **kwargs):
        # Call super method to obtain DF 
        df = self.extract_weight_table(model, *args, **self.config) if base_df is None else base_df

        # Susceptibility here is considered uniform (hence the randomness assigning TMR)
        df['susceptibility'] = np.abs(df['value'].values)
        df = df.sort_values(by=['pruned','bit','susceptibility'], ascending = [True, False, self.ascending])
        df['rank'] = np.arange(len(df))
        # Add column with the alias 
        df['alias'] = self.alias
        df['method'] = self.method
        # reset index
        df = df.reset_index(drop=True)
        
        # assign to self 
        self.ranking = Ranking(df, config = self.config, filepath = self.path, loaded_from_file = False)

        return self.ranking


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
    def rank(self, model, *args, base_df = None, **kwargs):
        # Call super method to obtain DF 
        df = self.extract_weight_table(model, *args, **self.config) if base_df is None else base_df

        # Susceptibility here is considered uniform (hence the randomness assigning TMR)
        df['susceptibility'] = 2.0**df['bit']
        df = df.sort_values(by=['pruned','bit'], ascending = [True, self.ascending])
        df['rank'] = np.arange(len(df))
        # Add column with the alias 
        df['alias'] = self.alias
        df['method'] = self.method
        # reset index
        df = df.reset_index(drop=True)
        
        # assign to self 
        self.ranking = Ranking(df, config = self.config, filepath = self.path, loaded_from_file = False)

        return self.ranking


""" Rank weights by layer (top to bottom, bottom to top or custom order) """
@dataclass
class LayerWeightRanker(WeightRanker):
    _ICON = "🎞️"

    # Method to actually rank the weights
    @timeranking
    def rank(self, model, *args, **kwargs):
        # Call super method to obtain DF 
        df = self.extract_weight_table(model, *args, **self.config)

        # Susceptibility here is considered uniform (hence the randomness assigning TMR)
        # (variable_index is almost like layer index, although there is a preference of kernel vs bias, cause 
        # when listing variables kernel always comes before bias, even for the same layer, but whatever).
        # If required, we could actually enforce layer index computation by grouping variables by their name 
        # (removing the "/..." part of the name) and then sorting by the order of appearance in the model, 
        # but I don't really think this is required right now. 
        df = df.sort_values(by=['pruned', 'bit', 'variable_index'], ascending = [True, False, self.ascending])
        df['rank'] = np.arange(len(df))
        df['susceptibility'] = 2.0**df['bit'] * (self.quantization.m - df['variable_index'] + 1)
        # Add column with the alias 
        df['alias'] = self.alias
        df['method'] = self.method
        df = df.reset_index(drop=True)

        # assign to self 
        self.ranking = Ranking(df, config = self.config, filepath = self.path, loaded_from_file = False)

        return self.ranking
    


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

        df = self.extract_weight_table(model, **self.config) if base_df is None else base_df

        differences = np.vectorize(process_value)(df['value'].values)
        df['susceptibility'] = differences
        df = df.sort_values(by=['pruned','bit'], ascending = [True, False])
        df['rank'] = np.arange(len(df))
        # Add column with the alias 
        df['alias'] = self.alias
        df['method'] = self.method
        df = df.reset_index(drop=True)

        # assign to self 
        self.ranking = Ranking(df, config = self.config, filepath = self.path, loaded_from_file = False)

        return self.ranking


""" Rank weights by using proportion Recursively """
@dataclass
class RecursiveUnevenRanker(WeightRanker):
    _ICON = "🔄"

    # Method to actually rank the weights
    @timeranking
    def rank(self, model, *args, base_df = None, **kwargs):
        # Call super method to obtain DF 
        df = self.extract_weight_table(model, **self.config) if base_df is None else base_df

        last_level = self.quantization.m - 1
        input_df = df[df['bit']==0]
        df = self.rec(input_df, last_level)
        df['rank'] = np.arange(len(df))
        # Add column with the alias 
        df['alias'] = self.alias
        df['method'] = self.method
        df = df.reset_index(drop=True)

        # assign to self 
        self.ranking = Ranking(df, config = self.config, filepath = self.path, loaded_from_file = False)

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


####################################################################################################################
#
# FIRST ORDER GRADIENT RANKERS 
#
####################################################################################################################

""" Rank weights by first order gradient 
"""
@dataclass
class GradWeightRanker(WeightRanker):
    _ICON = "💈"
    use_delta_as_weight: Optional[bool] = False

    # Method to actually rank the weights
    @timeranking
    def rank(self, model, X, Y, **kwargs):

        # Call super method to obtain DF 
        df = self.extract_weight_table(model, X, Y, **self.config)

        # Finally, sort by susceptibility 
        df = df.sort_values(by=['pruned','bit','susceptibility'], ascending = [True, False, self.ascending])
        # set rank
        df['rank'] = np.arange(len(df))
        # Add column with the alias 
        df['alias'] = self.alias
        df['method'] = self.method
        df = df.reset_index(drop=True)

        # assign to self 
        self.ranking = Ranking(df, config = self.config, filepath = self.path, loaded_from_file = False)

        return self.ranking
    
    """ Extracting the table of weights and creating the pandas DF is the same 
        for all methods, so we can define it inside the generic weightRanker obj 
    """
    def extract_weight_table(self, model, X, Y, verbose = True, 
                                normalize_score = False, 
                                absolute_value = True, base_df = None,
                                bit_value = None, out_dir = ".", **kwargs):
        
         # Get quantization
        Q = self.quantization

        # Call super to get the basic df 
        df = WeightRanker.extract_weight_table(self, model, verbose = verbose, **self.config) if base_df is None else base_df

        # Make sure susceptibility is float
        df['susceptibility'] = df['susceptibility'].astype(float)

        # If use_delta as weight, clone the model and replace the weights with the deltas for each bit
        use_delta_as_weight = self.use_delta_as_weight

        # Loop thru all bits (even if we are not using deltas as weights, this is so we can use the same loop
        # and code for both cases. If use_delta_as_weight is False we will break after the first loop iter)
        for ibit, bit in enumerate(np.arange(Q.n + Q.s - 1, -Q.f-1, -1)):
            
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
            # if times_weights:
            #     # Multiply gradients times variables 
            #     gradients = [g*v for g,v in zip(gradients, delta_model.trainable_variables)]

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
                    for i in np.arange(Q.n + Q.s - 1, -Q.f-1, -1):
                        if i != bit:
                            idx = df[(df['param'] == vname) & (df['bit'] == i)].index
                            df.loc[idx,'susceptibility'] = g.numpy().flatten()
            
            # And finally, break if we are not using deltas as weights
            if not use_delta_as_weight:
                break

        return df



""" Rank weights by using HiRes (gradcam++) but instead of using the weights of the model, we use the DELTAS as weights """
@dataclass
class GradDeltaWeightRanker(GradWeightRanker):
    _ICON="🔮"
    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        # Make sure to set use_delta_as_weight to True
        self.use_delta_as_weight = True



####################################################################################################################
#
# SECOND ORDER GRADIENT RANKERS (Hessian)
#
####################################################################################################################

""" This is the parent class for all Hessian-based rankers """
@dataclass
class HessianWeightRanker(WeightRanker):
    eigen_k_top: Optional[float] = 4
    max_iter: Optional[float] = 1000
    use_delta_as_weight: Optional[bool] = False
    
    _ICON = "👴🏻"

    # override extract_weight_table
    def extract_weight_table(self, model, X, Y, 
                             normalize_score = False, absolute_value = True,
                             base_df = None, eigen_k_top = 8, max_iter = 1000,
                             verbose = True, 
                             **kwargs):
        
        # Call super to get the basic df 
        df = WeightRanker.extract_weight_table(self, model, verbose = verbose, **self.config) if base_df is None else base_df

        # get quantizer
        Q = self.quantization

        # Make sure susceptibility is float
        df['susceptibility'] = df['susceptibility'].astype(float)

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
            if self.use_delta_as_weight:    
                deltas = model.deltas

                # Replace 
                for iv, v in enumerate(model.trainable_variables):
                    vname = v.name
                    assert deltas[iv].shape[:-1] == v.shape, f'Error at {iv} {vname} {deltas[iv].shape[:-1]} != {v.shape}'

                    # Replace the weights with the deltas (if use_delta_as_weight)
                    v.assign(deltas[iv][...,ibit])

            # Compute top eigenvalues/vectors
            eigenvalues, eigenvectors = self.top_eigenvalues(model, X, Y, k=eigen_k_top, max_iter = max_iter)

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
                if not self.use_delta_as_weight:
                    for i in np.arange(Q.n + Q.s - 1, -Q.f-1, -1):
                        if i != bit:
                            idx = df[(df['param'] == vname) & (df['bit'] == i)].index
                            df.loc[idx,'susceptibility'] = g.numpy().flatten()
            
            # And finally, break if we are not using deltas as weights
            if not self.use_delta_as_weight:
                break
        
        # if use_delta_as_weight make sure to set the original values back 
        if self.use_delta_as_weight:
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
    def rank(self, model, X, Y, **kwargs):
        # Call super method to obtain DF 
        df = self.extract_weight_table(model, X, Y, **self.config)

        # Finally, sort by susceptibility 
        df = df.sort_values(by=['pruned','bit','susceptibility'], ascending = [True, False, self.ascending])
        # Set rank
        df['rank'] = np.arange(len(df))
        # Add column with the alias 
        df['alias'] = self.alias
        df['method'] = self.method
        df = df.reset_index(drop=True)

        # assign to self 
        self.ranking = Ranking(df, config = self.config, filepath = self.path, loaded_from_file = False)
        return self.ranking
    


""" Same, but using deltas as weights """
@dataclass
class HessianDeltaWeightRanker(HessianWeightRanker):
    _ICON = "👵🏻"
    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        # Make sure to set use_delta_as_weight to True
        self.use_delta_as_weight = True

####################################################################################################################
#
# AI-BER RANKING 
#
####################################################################################################################
@dataclass
class AIBerWeightRanker(WeightRanker):
    _ICON = "🤖"
    def extract_weight_table(self, model, X, Y,  verbose=False, base_df = None, **kwargs):
        Q = self.quantization
        
        # Call super to get the basic df 
        df = WeightRanker.extract_weight_table(self, model, verbose = verbose, **self.config) if base_df is None else base_df
        
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
    def rank(self, model, X, Y, **kwargs):
        # Call super method to obtain DF 
        df = self.extract_weight_table(model, X, Y, **self.config)

        # Finally, sort by susceptibility 
        df = df.sort_values(by=['pruned','bit','susceptibility'], ascending = [True, False, self.ascending])
        # set rank
        df['rank'] = np.arange(len(df))
        # Add column with the alias 
        df['alias'] = self.alias
        df['method'] = self.method
        df = df.reset_index(drop=True)

        # assign to self 
        self.ranking = Ranking(df, config = self.config, filepath = self.path, loaded_from_file = False)
        return self.ranking





####################################################################################################################
#
# DELTAPOX RANKING (WORK IN PROGRESS...)
#
####################################################################################################################
@dataclass
class QPolarWeightRanker(WeightRanker):
    _ICON = "🧲"

    def extract_weight_table(self, model, X, Y, verbose=False, batch_size = 1000, base_df = None, **kwargs):
        
        # Call super to get the basic df 
        df = WeightRanker.extract_weight_table(self, model, verbose = verbose, **self.config) if base_df is None else base_df
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
        df = df.sort_values(by=['pruned','bit','susceptibility'], ascending = [True, False, self.ascending])

        return df
    # Method to actually rank the weights
    @timeranking
    def rank(self, model, X, Y, **kwargs):
        # Make sure the model has extracted the deltas 
        model.compute_deltas()
        
        # Call super method to obtain DF 
        df = self.extract_weight_table(model, X, Y, **self.config, **kwargs)

        # Finally, sort by susceptibility 
        df = df.sort_values(by=['pruned','bit','susceptibility'], ascending = [True, False, self.ascending])
        # Set rank
        df['rank'] = np.arange(len(df))
        # Add column with the alias 
        df['alias'] = self.alias
        df['method'] = self.method
        df = df.reset_index(drop=True)

        # Create ranking object
        self.ranking = Ranking(df, config = self.config, filepath = self.path, loaded_from_file = False)

        return self.ranking



@dataclass
class QPolarGradWeightRanker(QPolarWeightRanker, GradWeightRanker):
    _ICON = "🧲"
    def __init__(self, *args, **kwargs):
        super(GradWeightRanker, self).__init__(*args, **kwargs)
        super(QPolarGradWeightRanker, self).__init__(*args, **kwargs)
        

    def extract_weight_table(self, model, X, Y, verbose=False, **kwargs):
        # At this point we will have the susceptibility in terms of qpolar impact.
        df = QPolarWeightRanker.extract_weight_table(self, model, X, Y, verbose = verbose, **self.config)
        
        
        # Make sure we create a DEEP copy of df, cause this is pointing to self.df and it will be modified in the 
        # nxt call
        df_polar = df.copy()
        
        # Now let's get the gradients for all trainable variables
        df_grad = GradWeightRanker.extract_weight_table(self, model, X, Y, verbose = verbose, **self.config)
        
        # Copy 
        df_grad = df_grad.copy()
        

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

    def rank(self, model, X, Y, **kwargs):
        return QPolarWeightRanker.rank(self, model, X, Y, **kwargs)


"""
Fisher weight ranker
"""
@dataclass
class FisherWeightRanker(WeightRanker):
    _ICON = "🐟"

    def extract_weight_table(self, model, X, Y, verbose=False, base_df = None, **kwargs):
        
        
        # Call super to get the basic df 
        df = WeightRanker.extract_weight_table(self, model, verbose = verbose, **self.config) if base_df is None else base_df

        # Make sure susceptibility is float
        df['susceptibility'] = df['susceptibility'].astype(float)

        # MAKE SURE YOU ARE SORTED BY BIT AND INTERNAL PARAM NUM
        df = df.sort_values(by=['param','bit', 'internal_param_num'], ascending=[False, False, self.ascending])

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
        df = self.extract_weight_table(model, X, Y, **self.config, **kwargs)
        df = df.sort_values(by=['pruned', 'bit', 'susceptibility'], ascending=[True, False, self.ascending])
        # Set rank
        df['rank'] = np.arange(len(df))
        # Add column with the alias 
        df['alias'] = self.alias
        df['method'] = self.method
        df = df.reset_index(drop=True)

        self.ranking = self.ranking = Ranking(df, config = self.config, filepath = self.path, loaded_from_file = False)
        return self.ranking




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
    baseline: Optional[str] = None
    granularity: Optional[float] = 0.01
    rankers: List[WeightRanker] = field(default_factory=list)
    comparison: pd.DataFrame = field(init=False)

    def __post_init__(self):
        if not self.rankers:
            netsurf.info("Initialized RankingComparator with no rankers.")

    @property
    def _keys(self):
        if not self.rankers:
            return []
        else:
            return [ranker.alias for ranker in self.rankers]

    """ Method to create a ranker """
    def create(self, ranker: str, **kwargs):
        """ Create a ranker """
        if ranker not in self._keys:
            raise KeyError(f"Ranker '{ranker}' not found.")
        ranker = self[ranker]
        return ranker.create(**kwargs)

    """ Methods to attach rankers """
    def append(self, ranker: WeightRanker):
        if not isinstance(ranker, WeightRanker):
            raise TypeError("Only WeightRanker instances can be added.")
        self.rankers.append(ranker)
        netsurf.info(f"Ranker {ranker.alias} added.")
    
    def extend(self, rankers: List[WeightRanker]):
        if not all(isinstance(r, WeightRanker) for r in rankers):
            raise TypeError("Only WeightRanker instances can be added.")
        self.rankers.extend(rankers)
        netsurf.info(f"Rankers {', '.join([r.alias for r in rankers])} added.")

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

    def save_to_csv(self, filepath):
        """
        Save the RankingComparator to a file.
        """
        # if ranking is not None
        if isinstance(self.comparison, pd.DataFrame):
            # Save the comparison
            netsurf.info(f"Saving comparison to {filepath}")
            self.comparison.to_csv(filepath, index=False)
        else:
            raise ValueError("Comparison not found. Run compare() first.")

    @staticmethod
    def load_from_csv(filepath: str) -> 'RankingComparator':
        """
        Load a RankingComparator from a file.
        """
        # Load csv, if it exists 
        if not netsurf.utils.file_exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found.")
        df = pd.read_csv(filepath)
        
        # Create an empty comparator
        comparator = RankingComparator()
        # Add this as comparison
        comparator.comparison = df
        return comparator

    @staticmethod
    def from_directory(path: str, rankers: List[str] = None, granularity: float = 0.1, baseline: str = None, 
                       config_per_methods = {}, **kwargs) -> 'RankingComparator':
        # First check if the path exists. This should be the "experiments" inside the hierarchy benchmarks/<benchmark>/q<m>_<n>_<s>/<model_alias>/experiments
        # then we will pass this path when building the rankers 
        if rankers is None:
            rankers = ['random', 'hessian', 'hessiandelta', 'fisher', 'qpolar', 'qpolargrad', 'grad', 'hiresdelta', 'bitwise', 'weight_abs_value']
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


    """
        Build a ranker and automatically append it to the list of rankers
    """
    def build_ranker(self, ranker: str, *args, is_baseline = False, overwrite = False, **kwargs) -> 'RankingComparator':
        """
        Create a RankingComparator from a model and data.
        """
        
        # Check if ranker already exists in structure 
        if (ranker in self._keys) and not overwrite:
            raise KeyError(f"Ranker '{ranker}' already exists in the comparator. Use overwrite=True to replace it and get rid of this error.")
        
        # Create rankers
        r = WeightRanker.build(ranker, *args, **kwargs)

        # Append to the list of rankers
        self.rankers.append(r)
        # If this is a baseline, set it
        if is_baseline:
            self.baseline = r.alias
            netsurf.info(f"Ranker '{ranker}' set as baseline.")
        return r

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
    def rank(self, ranker: str, model, X, Y, **kwargs):
        # Make sure ranker is in self
        if ranker not in self._keys:
            raise KeyError(f"Ranker '{ranker}' not found.")

        # Check if rank already exists
        if self[ranker].ranking is not None:
            if not self[ranker].ranking.is_empty:
                return self[ranker].ranking
        
        # Rank
        base_df = WeightRanker.extract_weight_table(self[0], model, X, Y, **self.config)
        return self[ranker].rank(model, X, Y, base_df = base_df, **kwargs)
 
    """ Set baseline """
    def set_baseline(self, ranker):
        # If ranker is string, check it exists 
        if isinstance(ranker, str):
            if ranker not in self._keys:
                # check if they match any alias 
                if ranker in [r.alias for r in self.rankers]:
                    ranker = next(r for r in self.rankers if r.alias == ranker)
                raise KeyError(f"Ranker '{ranker}' not found.")
            self.baseline = ranker
        elif isinstance(ranker, WeightRanker):
            # Check if ranker is in self
            if ranker.alias not in self._keys:
                raise KeyError(f"Ranker '{ranker.alias}' not found.")
            self.baseline = ranker.alias
        else:
            raise TypeError(f"Ranker '{ranker}' is not a string or WeightRanker instance.")


    """ Perform the actual comparison of rankers against the baseline """
    def compare_rankers(self, granularity: float = 0.1, bootstrap = False) -> pd.DataFrame:

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
        not_yet_ranked = [ranker.alias for ranker in self.rankers if ranker.ranking is None]
        if len(not_yet_ranked) > 0:
            raise ValueError(f"Rankers {', '.join(not_yet_ranked)} have not been ranked yet.")

        baseline_ranker = next(r for r in self.rankers if r.alias == self.baseline)
        n_total = len(baseline_ranker.ranking)
        min_granularity = 1.0 / n_total
        if granularity < min_granularity:
            raise ValueError(f"Granularity too small. Minimum granularity is {min_granularity:.6f}.")

        rows = []

        protection_levels = np.arange(granularity, 1 + granularity, granularity)
        baseline_params_ids = baseline_ranker.ranking['global_param_num_bit'].to_list()

        # Build lookup dictionaries: param ID → rank index
        baseline_rank_dict = {pid: i for i, pid in enumerate(baseline_params_ids)}
        
        #Q = baseline_ranker.quantization
        #baseline_ranks = baseline_ranks + abs(baseline_ranker.ranking['bit'] - Q.n - Q.s + 1)/Q.m

        with netsurf.utils.ProgressBar(total = (len(self.rankers)-1)*len(protection_levels), prefix = 'Computing ranking overlaps') as pbar:
            for method in self.rankers:
                if method.alias == self.baseline:
                    continue

                # Get full ordered parameter ID lists from each ranking
                method_param_ids   = method.ranking['global_param_num_bit'].to_list()

                # Build lookup dictionaries: param ID → rank index
                method_rank_dict   = {pid: i for i, pid in enumerate(method_param_ids)}

                for p in protection_levels:

                    k = int(p * n_total)
                    if k == 0 or k > n_total:
                        continue
                    
                    top_baseline = set(baseline_params_ids[:k])
                    top_method   = set(method_param_ids[:k])
                    top_common   = top_baseline & top_method  # only the ones in both

                    # --- Jaccard ---
                    intersection = len(top_common)
                    union = len(top_baseline | top_method)
                    jaccard = intersection / union if union > 0 else 0.0
                    jaccard_sym = intersection / ((len(top_baseline) + len(top_method)) / 2)

                    # --- Rank Vectors (aligned) ---
                    aligned_ids = sorted(top_common)  # any order is fine as long as it's the same
                    baseline_vec = [baseline_rank_dict[pid] for pid in aligned_ids]
                    method_vec   = [method_rank_dict[pid] for pid in aligned_ids]

                    # --- Correlations ---
                    if len(aligned_ids) < 2:
                        kendall_corr, spearman_corr = np.nan, np.nan
                        kendall_low, kendall_high = None, None
                        spearman_low, spearman_high = None, None
                    else:
                        kendall_corr, _ = kendalltau(baseline_vec, method_vec)
                        spearman_corr, _ = spearmanr(baseline_vec, method_vec)

                    if bootstrap:
                        # Bootstrap confidence intervals
                        # Note: This is a simplified version, you might want to improve it
                        # by using a more robust bootstrap method.
                        kendall_low, kendall_high = self._bootstrap_corr_ci(baseline_vec, method_vec, "kendall")
                        spearman_low, spearman_high = self._bootstrap_corr_ci(baseline_vec, method_vec, "spearman")
                    else:
                        kendall_low, kendall_high = None, None
                        spearman_low, spearman_high = None, None

                    rows.append({
                        "protection": p,
                        "method1": method.alias.split('_q')[0],
                        "method2": self.baseline.split('_q')[0],
                        "baseline": self.baseline.split('_q')[0],
                        "kendall": kendall_corr,
                        "kendall_ci_low": kendall_low,
                        "kendall_ci_high": kendall_high,
                        "spearman": spearman_corr,
                        "spearman_ci_low": spearman_low,
                        "spearman_ci_high": spearman_high,
                        "jaccard": jaccard,
                        "jaccard_sym": jaccard_sym,
                    })

                    # Pbar
                    pbar.update()

            df = pd.DataFrame(rows)

        # Update structure 
        self.comparison = df

        return df
    
    def plot_radar_overlap(self, protection_levels=[0.2, 0.4, 0.8, 1.0], axs = None, show = True):
        
        metrics = ['kendall', 'spearman', 'jaccard', 'jaccard_sym']

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

        fig.suptitle(f"Ranking Overlap", fontsize=16)

        for ip, p in enumerate(protection_levels):
            dfp = self.comparison[self.comparison['protection'].round(2) == round(p, 2)]

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
                axs[i].fill(angles, values, alpha=0.25, label = f'{p:3.2%}')
                axs[i].set_title(metric.upper(), size=13)
                axs[i].set_xticks(angles[:-1])
                axs[i].set_xticklabels(labels)
                axs[i].set_yticks([0.25, 0.5, 0.75, 1.0])
                if i == 0:
                    axs[i].legend(loc='best', bbox_to_anchor=(0.1, 0.1), fontsize=8)

        plt.tight_layout()
        if show:
            plt.show()
            
        return fig, axs
    
    def plot_overlap_curves(self, axs = None, show = True):

        metrics = ['kendall', 'spearman', 'jaccard', 'jaccard_sym']

        # Check if axs has the right length
        if axs is not None:
            if len(axs) != len(metrics):
                # set back to None
                print('axs has wrong length, creating new ones')
                axs = None

        # Check if we need to create figure
        show &= (axs is None)
        if axs is None:
            fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 12), sharex = True)
            fig.suptitle("Ranking Overlap", fontsize=16)
        else:
            fig = axs[0].figure
            fig.suptitle("Ranking Overlap", fontsize=16)
        
        for i, metric in enumerate(metrics):
            sns.lineplot(data=self.comparison, x='protection', y=metric, hue='method1', ax=axs[i])
            #axs[i].set_title(f'{metric.upper()} vs Protection')
            #axs[i].set_ylim(0, 1)
            # Place legend outside with title inside legend
            axs[i].legend(loc='upper left', bbox_to_anchor=(1, 1), title=f'{metric.upper()}')
            # Remove xlabel unless this is the last one 
            if i != len(metrics) - 1:
                axs[i].set_xlabel('')
        
        plt.tight_layout()
        if show:
            plt.show()
        
        return fig, axs
    
    def plot_cumulative_overlap(self, ax = None, show = True):
        
        # Check if we need to create figure
        show &= (ax is None)
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            fig.suptitle("Ranking Overlap", fontsize=16)
        else:
            fig = ax.figure
            fig.suptitle("Ranking Overlap", fontsize=16)
        
        avg_df = self.comparison.groupby('method1')[['kendall', 'spearman', 'jaccard', 'jaccard_sym']].mean().reset_index()
        # Add row with the baseline for reference 
        avg_df.loc[len(avg_df)] = [self.comparison['baseline'][0], 1, 1, 1, 1]

        melted = avg_df.melt(id_vars='method1', var_name='metric', value_name='value')

        sns.barplot(data=melted, x='method1', y='value', hue='metric', ax = ax)
        plt.title('Average Overlap Across Protection Levels')
        plt.xticks(rotation=45)
        # Place legend on the outside
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        if show:
            plt.show()
        
        return fig, ax

    # HTML
    def html(self):
        # Add a container to hold this whole comparison 
        ct = pg.CollapsibleContainer('⚖️ Ranking Comparison', layout = 'vertical')

        # Create container for table data 
        c1 = pg.CollapsibleContainer('Comparison Data', layout='vertical')
        # Now add the comparison table 
        t = pg.Table.from_data(self.comparison)
        c1.append(t)
        # Add to container
        ct.append(c1)

        # Now add the plots
        c2 = pg.CollapsibleContainer('〰️ Overlap curves', layout='vertical')
        # First the overlap_curves altogether
        fig, axs = self.plot_overlap_curves(show = False)
        for ax in axs:
            netsurf.utils.plot.turn_grids_on(ax)
        # Add to container
        p2 = pg.Image(fig, embed = True)
        c2.append(p2)
        ct.append(c2)
        # Close fig
        plt.close(fig)

        # Now add the radar plot
        c3 = pg.CollapsibleContainer('🕷️ Radar plot', layout='vertical')
        # First the radar plot
        fig, axs = self.plot_radar_overlap(show = False)
        # Add to container
        p3 = pg.Image(fig, embed = True)
        c3.append(p3)
        ct.append(c3)
        # close fig
        plt.close(fig)

        # Now add the cumulative plot
        c4 = pg.CollapsibleContainer('📊 Cumulative plot', layout='vertical')
        # First the cumulative plot
        fig, ax = self.plot_cumulative_overlap(show = False)
        # Add to container
        p4 = pg.Image(fig, embed = True)
        c4.append(p4)
        ct.append(c4)
        # close fig
        plt.close(fig)

        # Also, add the individual plots for avg/std for the overlap, which look very cool. 
        c5 = pg.CollapsibleContainer('🎨 Individual dispersion', layout='vertical')
        # Loop thru methods 
        g = self.comparison.groupby('method1')
        w = np.maximum(3,int(len(self.comparison)*0.15))
        for i,(name, group) in enumerate(g):
            # Create a container 
            _c = pg.CollapsibleContainer(name, layout='vertical')
            # Create a plot
            fig, ax = plt.subplots(figsize=(10, 5))
            netsurf.utils.plot.plot_avg_and_std(group['kendall'], w, shadecolor = f'C{4*i}', ax = ax)
            netsurf.utils.plot.plot_avg_and_std(group['spearman'], w, shadecolor = f'C{4*i+1}', ax = ax, show_legend = False)
            netsurf.utils.plot.plot_avg_and_std(group['jaccard'], w, shadecolor = f'C{4*i+2}', ax = ax, show_legend=False)
            netsurf.utils.plot.plot_avg_and_std(group['jaccard_sym'], w, shadecolor = f'C{4*i+3}', ax = ax, show_legend=False)
            # Keep only first legend
            handles, labels = ax.get_legend_handles_labels()
            handles = handles[:4]
            labels = labels[:4]
            ax.legend(handles, labels)
            ax.set_title(name)
            # Embed picture 
            _p = pg.Image(fig, embed = True)
            _c.append(_p)
            c5.append(_c)
            # close fig
            plt.close(fig)
        
        # Add to container
        ct.append(c5)
        # Return container
        return ct

