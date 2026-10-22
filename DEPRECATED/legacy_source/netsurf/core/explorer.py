""" Basic imports """
import os
from glob import glob
from copy import deepcopy
import re
from time import time

""" Typing """
from typing import Union, Iterable

# Time 
from datetime import datetime

""" json """
import json

""" Numpy """
import numpy as np

""" Pandas """
import pandas as pd

""" Matplotlib """
import matplotlib
import matplotlib.pyplot as plt

# Custom imports
import netsurf

####################################################################################################
# UTILITIES USED BY BUCKETS 
####################################################################################################


## Class to hold the hyperspace configuration for each bucket (instead of a dirty dict)
class HyperspaceConfig:
    def __init__(self, type, verbose = True, **kwargs):
        # Set the type (global, local)
        self.type = type

        # Set the rest of the parameters
        self._keys = self.init_hyperspace_config(**kwargs)

        if verbose: netsurf.utils.log._log(f"Initializing hyperspace global config with the following parameters: {self}")
    
    """ Property to get elements """
    def __getitem__(self, key):
        if key not in self._keys or key is None:
            return None
        return getattr(self, f'_{key}')

    # Set item 
    def __setitem__(self, key, value):
        # Add to keys
        if key not in self._keys:
            self._keys.append(key)
        setattr(self, f'_{key}', value)
    
    def items(self):
        ps = []
        for k in self._keys:
            ps += [(k, getattr(self, f'_{k}'))]
        return ps

    """ Init hyperspace config """
    def init_hyperspace_config(self, 
                               benchmarks_dir = netsurf.config.DEFAULT_BENCHMARKS_DIR,
                               datasets_dir = netsurf.config.DEFAULT_DATASETS_DIR,
                                map_level = netsurf.config.DEFAULT_LEVEL_MAP,
                                children_prop = netsurf.config.DEFAULT_CHILDREN_PROP, 
                                benchmarks = netsurf.config.DEFAULT_BENCHMARKS, 
                                quantizations = netsurf.config.DEFAULT_QUANTIZATIONS, 
                                pruning = netsurf.config.DEFAULT_PRUNINGS, 
                                methods = netsurf.config.DEFAULT_METHODS, 
                                protection = netsurf.config.DEFAULT_PROTECTION, 
                                ber = netsurf.config.DEFAULT_BER, 
                                num_reps = netsurf.config.DEFAULT_NUM_REPS, 
                                **kwargs):

        """ Initialize hyperspace global config """
        # Create placeholders
        hyperspace_global_config = {'benchmark': benchmarks, 
                                    'quantization': quantizations, 
                                    'pruning': pruning,
                                    'method': methods, 
                                    'protection': protection,
                                    'ber': ber,
                                    'num_reps': num_reps,
                                    'map_level': map_level,
                                    'children_prop': children_prop,
                                    'benchmarks_dir': benchmarks_dir,
                                    'datasets_dir': datasets_dir}
        
        _def_hyperspace_global_config = {'benchmark': netsurf.config.DEFAULT_BENCHMARKS,
                                        'quantization': netsurf.config.DEFAULT_QUANTIZATIONS,
                                        'pruning': netsurf.config.DEFAULT_PRUNINGS,
                                        'method': netsurf.config.DEFAULT_METHODS,
                                        'protection': netsurf.config.DEFAULT_PROTECTION,
                                        'ber': netsurf.config.DEFAULT_BER,
                                        'num_reps': netsurf.config.DEFAULT_NUM_REPS,
                                        'map_level': netsurf.config.DEFAULT_LEVEL_MAP,
                                        'children_prop': netsurf.config.DEFAULT_CHILDREN_PROP,
                                        'benchmarks_dir': netsurf.config.DEFAULT_BENCHMARKS_DIR,
                                        'datasets_dir': netsurf.config.DEFAULT_DATASETS_DIR}

        # Check if we have a map_level
        _keys = []
        for k, v in _def_hyperspace_global_config.items():
            if k not in hyperspace_global_config:
                hyperspace_global_config[k] = v
            elif hyperspace_global_config[k] is None:
                hyperspace_global_config[k] = v
            _keys.append(k)

            # Set internal property
            setattr(self, f'_{k}', hyperspace_global_config[k])
        
        return _keys

    def __repr__(self):
        ss = f'<{self.__class__.__name__} [{self.type}]>\n'
        for k in self._keys:
            ss += f'\t{k}: {getattr(self, f"_{k}")}\n'
        return ss


""" Custom class to keep track of progress recursively """
class RecursiveProgressTracker:
    def __init__(self, pbar, offset = 0.0, factor = 100):
        self.pbar = pbar
        self.offset = offset
        self.factor = factor
    def __call__(self, value, text):
        self.pbar(value, text)
    
    def get_next_factor(self, num_childs):
        if num_childs == 0:
            new_factor = 0
        if num_childs == 1:
            new_factor = self.factor * 1.0
        else:
            new_factor = self.factor * 1.0 / (num_childs - 1)
        
        return new_factor

""" Metadata object for easy use later """
class Metadata:
    def __init__(self, level, name, group = None, creation_date = None, creation_user = None, creation_host = None, config = {}):
        self.level = level
        self.name = name
        self.group = group if group is not None else name

        if creation_date is None: creation_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        if creation_user is None: creation_user = os.getlogin()
        if creation_host is None: creation_host = os.uname().nodename

        self.creation_date = creation_date
        self.creation_user = creation_user
        self.creation_host = creation_host
        self.config = config

    def _to_dict(self):
        return {'level': self.level, 'name': self.name, 'creation_date': self.creation_date, 
                'creation_user': self.creation_user, 'creation_host': self.creation_host, 'config': self.config}
    # Save to file 
    def save(self, path):
        metadata_filepath = os.path.join(path, '.metadata.netsurf')
        if not os.path.isfile(metadata_filepath):
            # Save metadata
            netsurf.utils.log._info(f'Saving experiment metadata to file {metadata_filepath}')
            with open(metadata_filepath, 'w') as f:
                json.dump(self._to_dict(), f, indent=2)
    
    def __repr__(self):
        s = f'<Metadata: {self.level} "{self.name}">\n'
        s += f'\tCreation Date: {self.creation_date}\n'
        s += f'\tCreation User: {self.creation_user}\n'
        s += f'\tCreation Host: {self.creation_host}\n'
        s += f'\tConfig: '
        if len(self.config) > 0:
            s += '{\n'
            s += netsurf.utils.recursive_dict_printer(self.config, tab = 2)
            s += '\t}'
        else:
            s += '{}'
        return s

# Basic get metadata method (initializes a metadata object )
def get_metadata(path):
    # First off, check if metadata exists by calling the utility 
    metadata = netsurf.utils.get_metadata(path)
    if metadata is None:
        return None
    return Metadata(**metadata)


""" Basic hierarchical container """
class HierarchicalContainer:
    def __gt__(self, other: 'HierarchicalContainer'):
        return self._LEVEL > other._LEVEL

    def __lt__(self, other: 'HierarchicalContainer'):
        return self._LEVEL < other._LEVEL
    
    def __eq__(self, other: 'HierarchicalContainer'):
        return self._LEVEL == other._LEVEL
    
    def __ge__(self, other: 'HierarchicalContainer'):
        return self._LEVEL >= other._LEVEL
    
    def __le__(self, other: 'HierarchicalContainer'):
        return self._LEVEL <= other._LEVEL
    
    def __ne__(self, other: 'HierarchicalContainer'):
        return self._LEVEL != other._LEVEL
    
    def __hash__(self):
        return hash(self.name)
    
    def __repr__(self):
        return f'<{self.__class__.__name__} [{self.type}] {self.name}>'
    
""" Some definitions for the levels """
# The structure we expect is: 
#   (0) - benchmarks_dir
#   (1)   - quantization (e.g., 6bits_0int)
#   (2)     - model (e.g., pruned_0.25_1095_hls4ml_cnn)
#   (3)       - experiments (folder containing the results for each method)
#   (4)         - method (e.g., bitwise_msb)
#   (5)           - run (e.g., run_20241215_103739)
#   (6)             - results.csv (file containing the results of the run)
#                   - config.json (file with the configuration of the run)
#                   - ranking.csv (file with the ranking)
#                   - global_metrics.json (file with the global metrics for this run)
#                   - job.log* (log file for the run)
#                   - job.sh* (job file for this run)
#                   - job.progress* (text file with the progress of the run)
#   (3)      - models (folder containing the models trained)
#   (3)      - sessions (folder containing the training sessions)
#   (4)         - training session (e.g. training_session.20241215_010614)
#   (5)           - config_training_session.*.json (config file with the training session parameters)
#   (5)           - training_session.*.log (log file with the training summary)
#   (5)           - training_session.*.pkl (binary file with the progress of the training session (can be loaded into netsurf))

""" Bucket container class (Recursive) 
        a Bucket is a container that will hold either other buckets or runs (at the last level)
        We will go thru the directory recursively creating other buckets or runs.
"""
class Bucket(HierarchicalContainer):
    _LEVEL = 0
    """ Init """
    def __init__(self, dir, name, level = 0, hyperspace_global_config = {}, structure_config = {}, metadata = None, **kwargs):
        # Set main vars
        self.dir = dir 
        self.name = name 
        self.level = level
        self.metadata = get_metadata(dir) if metadata is None else metadata

        # hyperspace global config 
        self.hyperspace_global_config = hyperspace_global_config if isinstance(hyperspace_global_config, HyperspaceConfig) else HyperspaceConfig('global', **kwargs)

        # Init hierarchical config
        self.structure_config = self.init_structure_config(structure_config)

        # Init 
        self.structural_local_metrics = {}

        # Init subs, keys
        self._subs = {}
        self._keys = []
        
        # Get the children of this bucket.
        self._subs, self._keys = self.get_children(**kwargs)

        # Init subcoverage 
        self.coverage = None


    """ Make it iterable """
    def __iter__(self):
        return iter(self._subs)
    
    def __next__(self):
        return next(self._subs)

    """ Make this class act as a list """
    def __getitem__(self, key):
        # if key is an int, find the corresponding key and return it
        if isinstance(key, int):
            return self._subs[self._keys[key]]
        return self._subs[key]

    # Set item is not allowed 
    def __setitem__(self, key, value):
        raise NotImplementedError("Setting items is not allowed in this class")

    """ Get descendants (children & children of children & children of children of children ... ) """
    @property
    def descendants(self):
        num_descendants = 0
        for k in self._keys:
            num_descendants += self._subs[k].descendants
        return num_descendants
    
    """ Implement function to be used by both get_results and get_coverage """
    # def _get_recursively(self, attr_name):
    #     """ Get results recursively """
    #     attr = f'get_{attr_name}'
    #     # Get results
    #     results = []
    #     for k in self._keys:
    #         r = getattr(self._subs[k],attr)()
    #         # deep copy 
    #         rr = deepcopy(r)
    #         results.append(r)

    #     # Create coveragepie
    #     netsurf.utils.plot.CoveragePie(b, verbose = False)


    #     # Concatenate
    #     if len(results) > 0:
    #         results = pd.concat(results)
    #         # Add type/name to results
    #         results[self.type] = self.name
    #     else:
    #         if attr_name == 'coverage':
                
    #             # If we have no coverage, then this means this is an empty 
    #             # method, with no run expriments yet. Thus, we will create
    #             # a dataframe with all combinations being zero. 
    #             protection_range = self.hyperspace_global_config['protection']
    #             ber_range = self.hyperspace_global_config['ber']
    #             num_reps = self.hyperspace_global_config['num_reps']

    #             # Init empty dataframe 
    #             cov = pd.DataFrame(columns=netsurf.core.experiments.Result.COLUMNS)
    #             # Create empty result
    #             r = netsurf.core.experiments.Result(cov, protection_range = protection_range, 
    #                                                 ber_range = ber_range,
    #                                                 num_reps = num_reps)
    #             # Call get_coverage (which will compute all combinations for us)
    #             cov, all_combs, combs, coverage, coverage_T = r.get_coverage()

    #             # Add experiment None
    #             cov['experiment'] = None

    #             # Add some columns now 
    #             ts = ['method', 'pruning', 'quantization', 'benchmark', 'root']
    #             # ns = [self.hyperspace_global_config['benchmark'], 
    #             #         self.hyperspace_global_config['quantization'],
    #             #         self.hyperspace_global_config['pruning'],
    #             #         self.hyperspace_global_config['method'],
    #             #         []]

    #             # Find the level we are at 
    #             i = ts.index(self.type)

    #             # experiments will be none, for the rest of the indices, we'll have to replicate
    #             # anything below the current level
    #             for ii in range(1, i+1):
    #                 # Get this level's property 
    #                 k = ts[ii]
    #                 # Get the property that we want to repeat 
    #                 krep = ts[ii-1]
    #                 ns = self.hyperspace_global_config[krep]

    #                 # Duplicate cov
    #                 cov_prev = deepcopy(cov)
    #                 all_combs_prev = deepcopy(all_combs)
    #                 new_covs = []
    #                 new_combs = []
    #                 for nn in ns:
    #                     # Add this type and name 
    #                     cov_tmp = cov_prev.copy()
    #                     cov_tmp[krep] = nn
    #                     new_covs.append(cov_tmp)
    #                     all_combs_tmp = all_combs_prev.copy()
    #                     all_combs_tmp[krep] = nn
    #                     new_combs.append(all_combs_tmp)
    #                 # Concatenate
    #                 cov = pd.concat(new_covs)
    #                 # Reset index 
    #                 cov.reset_index(drop = True, inplace = True)
    #                 # Same for all combs 
    #                 all_combs = pd.concat(new_combs)
    #                 all_combs.reset_index(drop = True, inplace = True)

    #             # Just reorder 
    #             # columns 
    #             cols = ts[:i][::-1] +  ['experiment', 'protection', 'ber', 'coverage', 'run_reps', 'total_num_reps']
    #             results = cov[cols]

    #         elif attr_name == 'results':
    #             results = pd.DataFrame(columns=netsurf.core.experiments.Result.COLUMNS)

    #             # Add type/name to results
    #             results[self.type] = self.name
    #     # Store local copy 
    #     if attr_name == 'coverage' or attr_name == 'all_combs':
    #         setattr(self, attr_name, deepcopy(results))
    #     return results
    
    """ Set recursively """
    def propagate_coverage_pie(self, **kwargs):
        """ Loop thru pies """
        for k in self._keys:
            if k in self.coverage_pie.subpies:
                # Set the pie in the child
                self._subs[k].coverage_pie = self.coverage_pie.subpies[k]
                self._subs[k].propagate_coverage_pie(**kwargs)

    """ Init structure config """
    def init_structure_config(self, structure_config):
        # Get the EXPECTED children for this bucket 
        children_type = self.hyperspace_global_config['map_level'][self.level + 1]
        children_property = self.hyperspace_global_config['children_prop'][self.type.capitalize()]
        expected_children = self.hyperspace_global_config[children_property]

        # Make a deep copy of the structure config
        sc = deepcopy(structure_config)

        # Put itself into the config 
        sc[self.type] =  self.name
        sc['children_type'] =  children_type
        sc['children_property'] = children_property
        sc['expected_children'] = expected_children

        return sc

    def _call_recursive(self, fcn: str, **kwargs):
        for k in self._keys:
            getattr(self._subs[k], fcn)(**kwargs)

    """ Get missing jobs """
    def get_missing_jobs(self):
        missing_jobs = []
        for k in self._keys:
            subjobs = self._subs[k].get_missing_jobs()
            # subjobs is a pd.DataFrame, add current level to it 
            subjobs[self.type] = self.name
            missing_jobs.append(subjobs)
        if len(missing_jobs) > 0:
            missing_jobs = pd.concat(missing_jobs, ignore_index=True)
            # Rearrange columns with self.type first 
            cols = missing_jobs.columns.tolist()
            cols = [self.type] + [c for c in cols if c != self.type]
            missing_jobs = missing_jobs[cols]
        else:
            missing_jobs = pd.DataFrame()
        return missing_jobs

    """ Get subcoverages for all children """
    def get_subcoverage(self):
        self._call_recursive('get_coverage')

    """ Get cover recursively """
    def get_coverage(self, groupby = None):
        
        # Get subcoverage for children
        self.get_subcoverage()

        # Get default groupby given this type and level 
        if groupby is None:
            groupby = ['root', 'benchmark', 'quantization', 'model', 'method']
            if self.type in groupby:
                i = groupby.index(self.type)
                if i < len(groupby) - 1:
                    groupby = groupby[i+1]
                else:
                    groupby = ['protection', 'ber']
            else:
                groupby = ['protection', 'ber']
            
        # Make sure groupby is a list 
        if not isinstance(groupby, list):
            groupby = [groupby]

        if self.coverage is None:
            out_coverage = self.results
            # Reorganize 
            out_coverage = out_coverage.rename(columns={'root':'bucket'})
            # Group
            try:
                gb = out_coverage.groupby(groupby)
            except Exception as e :
                print('stop here')

            run_reps = gb['loss'].apply(lambda x: x.notna().sum())
            total_num_reps = gb['loss'].apply(lambda x: x.shape[0])
            coverage = run_reps/total_num_reps

            # reset index 
            run_reps = run_reps.reset_index()
            # Rename metric to run_reps
            run_reps = run_reps.rename(columns = {'loss': 'run_reps'})
            # Add total number of reps 
            run_reps['total_num_reps'] = total_num_reps.reset_index()['loss']
            # Add coverage
            run_reps['coverage'] = coverage.reset_index()['loss']
            # And rearraange columns
            out_coverage = run_reps[groupby + ['coverage']]
            # Reset index 
            out_coverage.reset_index(drop = True, inplace = True)
            # Set to self
            self.coverage = out_coverage

        return self.coverage
    

    """ Get results recursively """
    def get_results(self, pbar = RecursiveProgressTracker(None)):

        # Before starting, keep the old factor and offsets of the pbar 
        old_factor = pbar.factor
        old_offset = pbar.offset
        # Now compute the new factor
        num_children = np.maximum(1,len(self._keys))
        new_factor = pbar.get_next_factor(num_children)
        children_type = self.structure_config['children_type']

        """ Get results recursively """
        # Get results
        results = []
        for ichild, children_name in enumerate(self._keys):

            # Create a new pbar for this child, with new offset 
            new_offset = old_offset + (ichild * new_factor)
            subpbar = RecursiveProgressTracker(pbar.pbar, offset = new_offset, factor = new_factor)

            # Init message to be displayed
            tabs = '  ' * (self.level + 1)
            msg = f"{tabs}({ichild+1}/{num_children}) - [{children_type.capitalize()}Container]: {children_name} - Getting results"
            # Call subpbar to update the progress
            subpbar(new_offset, f"Processing {msg}")

            r = self._subs[children_name].get_results(pbar = subpbar).astype(object)
            # Reset index 
            #r.reset_index(drop = True, inplace = True)
            rr = deepcopy(r)
            results.append(rr)
        # Concatenate
        data = None
        cols = deepcopy(netsurf.core.experiments.ResultSpace.COLUMNS)
        # Make sure we have all extra columns tht we need for this level 
        extra_cols = ['root','benchmark','quantization','model','model_name','pruning','method']
        extra_cols = extra_cols[extra_cols.index(self.type)+1:]
        # Add to cols 
        cols += extra_cols
        # Turn cols into set to avoid duplications
        cols = list(set(cols))
        if len(results) > 0:
            data = pd.concat(results, ignore_index = True)

        # Add to results  
        loss_name = self.structure_config.get('loss', 'categorical_crossentropy')
        metrics_names = self.structure_config.get('metrics', ['categorical_accuracy'])
        total_num_params = self.structure_config.get('total_num_params', 0)
        self.results = netsurf.core.experiments.ResultSpace(loss_name, metrics_names,
                                                          protection=self.hyperspace_global_config['protection'], 
                                                        ber=self.hyperspace_global_config['ber'], 
                                                        num_reps=-1,#num_reps=self.hyperspace_global_config['num_reps'],
                                                        total_num_params=total_num_params,
                                                        data = data,
                                                        columns = cols)

        # #  results = ResultSpace(loss_name, metric_names, self.protection_range, self.ber_range, self.num_reps, total_num_params, data=results, columns=columns)
        #          protection: Union[Iterable[float], np.ndarray], 
        #         ber: Union[Iterable[float], np.ndarray], 
        #         num_reps: int, 
        #         total_num_params: int, 
        #         data: pd.DataFrame = None,
        

        # Add type/name to results
        self.results[self.type] = self.name
        # Make sure the rest of the extra cols are in the results, in case they
        # are attributes in self 
        for e in extra_cols:
            if hasattr(self, e):
                if e not in self.results.columns:
                    self.results[e] = getattr(self, e)

        return self.results
    
    """ This function is DUMMY in general, but it's important at the Quantization/Model level """
    def get_children_name_and_metadata(self, x, dir, *args, **kwargs):
        # try to get metadata 
        metadata = get_metadata(os.path.join(dir,x))
        return x, {}, metadata

    """ Get the children of this bucket """
    def get_children(self, verbose = True, pbar = RecursiveProgressTracker(None), **kwargs):
        # Print if verbose
        tabs = '  ' * (self.level)
        if verbose and self.level == 0: netsurf.utils.log._info(f"{tabs} [{self.hyperspace_global_config['map_level'][self.level]}Container]: {self.name} @ {self.dir}", tab = 0)
        
        """ The children of this bucket are NOT only the directories in dir directory, 
            but also the EXPECTED children. That info is in the hyperspace_global_config 
            If the children are in the directory, great, we will create them and fill them 
            with whatever we find and go down the path recursively. If they are not, we will
            still create a placeholder for them, but we will not fill them with anything.
            This placeholder will allow us later on to keep track of the jobs that are missing
            and the coverage of the whole structure.
        """
        # Get the EXPECTED children for this bucket 
        children_type = self.structure_config['children_type']
        expected_children = self.structure_config['expected_children']

        # Get children Bucket class
        children_bucket_class = BUCKET_CLASSES.get(children_type, Bucket)
        
        # Before starting, keep the old factor and offsets of the pbar 
        old_factor = pbar.factor
        old_offset = pbar.offset
        # Now compute the new factor
        num_children = len(expected_children) if expected_children is not None else 1
        new_factor = pbar.get_next_factor(num_children)
        
        # Init children and children names
        children = {}
        children_names = []

        """ Loop thru expected children """
        for ichild, child in enumerate(expected_children):
            # Get the expected children name. This is often just the same as "child", but
            # for the model level we are actually looping thru pruning values <0.0, 0.125, etc.>,
            # but what we will have on disk is some directory like "pruned_0.0_<model_name>", etc.
            # Thus, we'll need to extract that model name at the QuantizationContainer level. 
            try:
                children_name, extra_structure_config_args, children_metadata = self.get_children_name_and_metadata(child, self.dir)
            except Exception as e:
                print(f'Error getting children name and metadata for {child} in {self.dir}')
                print(e)
                children_name, extra_structure_config_args, children_metadata = self.get_children_name_and_metadata(child, self.dir)

            # Construct the full path expected for this child
            full_path = os.path.join(self.dir, children_name)

            # Create a new pbar for this child, with new offset 
            new_offset = old_offset + (ichild * new_factor)
            subpbar = RecursiveProgressTracker(pbar.pbar, offset = new_offset, factor = new_factor)

            # Init message to be displayed
            tabs = '  ' * (self.level + 1)
            p = f"{'…/'*(self.level+1)}{os.path.basename(full_path)}"
            msg = f"{tabs}({ichild+1}/{num_children}) - [{children_type.capitalize()}Container]: {children_name} @ {p}"
            if verbose: netsurf.utils.log._info(msg, tab = 0)

            """
                IMPORTANT: At this point we have two possibilities:
                    (1) If the directory exists AND it has a .metadata.netsurf file inside,
                        we have to follow whatever that metadata has inside. 
                        This is weird, but it allows us to have model buckets within other 
                        model buckets, etc. But if it is what the user wanted, then we need
                        to follow this. 
                    (2) If the directory doesn't exist OR if it exists but the .metadata.netsurf
                        doesn't, we have to proceed as usual, inferring the children type that
                        would be expected at this level and creating a placeholder for it.
            """
            if children_metadata is None:
                # Try to find it again
                children_metadata = get_metadata(full_path)

            if children_metadata is not None:
                # Log
                netsurf.utils.log._info(f"{tabs}Found metadata file for {children_name} @ {full_path} Following metadata.", tab = self.level + 1)
                # Get the children type from the metadata
                children_type = children_metadata.level
                # Assert the name is the same we were expecting (should be)
                # TODO:
                if False:
                    assert children_name == children_metadata.name, f"Expected children name {children_name} but found {children_metadata.name}"
                # Override for now 
                children_name = children_metadata.name
                full_path = os.path.join(self.dir, children_name)
                # Get children Bucket class
                children_bucket_class = BUCKET_CLASSES.get(children_type.capitalize(), Bucket)
                # Add the config in the metadata to extra_structure_config_args
                extra_structure_config_args = {**extra_structure_config_args, **children_metadata.config}

            # Call subpbar to update the progress
            subpbar(new_offset, f"Processing {msg}")

            # Create a copy of the structure config
            new_structure_config = deepcopy(self.structure_config)
            # Add the extra_structure_config_args 
            new_structure_config = {**new_structure_config, **extra_structure_config_args}

            # Create children REGARDLESS of whether it exists or not (it will be handled downstream)
            c = children_bucket_class(full_path, children_name, level = self.level + 1, 
                                        verbose = verbose, pbar = subpbar, 
                                        hyperspace_global_config = self.hyperspace_global_config, 
                                        structure_config = new_structure_config,
                                        metadata = children_metadata,
                                        **kwargs)

            # Apprend to the children and children names
            children_names.append(children_name)
            children[children_name] = c

        return children, children_names

    """ Reload method (to reload the children) """
    def reload(self, verbose = True, **kwargs):
        # Get the children of this bucket.
        self._subs, self._keys = self.get_children(verbose = verbose, **kwargs)

        # Propagate the global metrics down
        self.propagate_global_metrics()

        # Now get the results 
        self.get_results(**kwargs)

        # Coverage
        self.get_coverage()

        # Coverage pies 
        this_pie = netsurf.utils.plot.CoveragePie(self, verbose = False)
        self.coverage_pie = this_pie

        # Now propagate the coverage pies down
        self.propagate_coverage_pie()


    """ Propagate the global metrics down (to the leaves) """
    def propagate_global_metrics(self):
        # Propagate down
        for k in self._keys:
            # Before propagating, store the local old metrics 
            self._subs[k].structural_local_metrics = deepcopy({kw: kv for kw, kv in self._subs[k].hyperspace_global_config.items()})
            self._subs[k].hyperspace_global_config = self.hyperspace_global_config
            self._subs[k].propagate_global_metrics()

    """ Plotting functions (will delegate this to the plotter, but we need this access thru here) """
    def plot_curves(self, type, *args, x = 'ber', y = 'mean', hue = 'protection', axs = None, ylabel = None, xlabel = None, info_label = None, 
                        is_entry_point = True, standalone = True, **kwargs):
        
        # Assert type in ['2d','3d']
        if type not in ['2d','3d','barplot','boxplot']:
            netsurf.utils.log._error(f'Invalid type {type} for plot_curves. Must be one of ["2d","3d","barplot","boxplot"]', tab = self.level)
            return [], [], 0, [], [], []

        # If this is just a placeholder (with empty data), skip
        if len(self.coverage) == 0:
            netsurf.utils.log._warn(f'Empty run placeholder. Skipping plot_curves.', tab = self.level)
            return [], [], 0, [], [], []

        if is_entry_point: netsurf.utils.log._info(f'Entry point detected.')

        # if ax is None, create a new figure with as many axes as there are runs 
        num_descendants = self.descendants
        netsurf.utils.log._log(f'Plotting {num_descendants} descendants for object {self.name} of type {self.type}', tab = self.level)

         # If num_children is 0, just skip this 
        if num_descendants == 0:
            return [], [], 0, [], [], []

        # Assert ax has length num_runs
        if axs is not None and not standalone:
            if not isinstance(axs, list) and not isinstance(axs, np.ndarray):
                axs = [axs]
            if len(axs) < num_descendants:
                netsurf.utils.log._error(f'Number of axes ({len(axs)}) does not match number of descendants ({num_descendants})', tab = self.level)
                axs = None
        
        
        fcn_kwargs = {'2d': {}, '3d': {'subplot_kw': {"projection": "3d"}},
                      'boxplot': {}, 'barplot': {}}.get(type)

        num_rows = 1
        num_cols = 1
        # If standalone, figures will be created at the last level (RunContainer)
        fig = []
        if not standalone and axs is None:
            if num_descendants > 1:
                num_rows = int(np.ceil(np.sqrt(num_descendants)))
                num_cols = int(np.ceil(num_descendants / num_rows))
                netsurf.utils.log._log(f'Creating {num_rows}x{num_cols} grid for {num_descendants} descendants', tab = self.level)

            # Init figure
            fig, axs = plt.subplots(num_rows, num_cols, figsize = (7 * num_cols, 7 * num_rows), sharey = True, sharex = True, **fcn_kwargs)
            
            # Make sure ax is a list
            if num_descendants == 1 or standalone:
                axs = [axs]
            else:
                axs = axs.flatten()
            

        # Make sure ax is a flattened list 
        axs = np.array(axs).flatten()

        # Remove "root" from the structure config in any case
        info_label_fields = ['benchmark', 'quantization', 'model', 'method', 'run']
        if info_label is None:
            info_label = {k: self.structure_config[k] for k in info_label_fields if k != 'root' and k in self.structure_config}
        
        if num_descendants > 1 and not standalone: 
            # Remove type from info label only if this is not standalone
            info_label = {k: v for k,v in info_label.items() if k != 'root' and k != self.type}
        
        if standalone:
            # add type to info label
            info_label[self.type] = self.name

        # Loop thru runs and plot each one 
        i = 0
        j = 0
        t = []
        _figs = []
        _axs = []
        _lines = []
        _infos = []
        num_left = int(num_descendants * 1.0)

        fcn = {'2d': 'plot_2D_curves', 
               '3d': 'plot_3D_volumes',
               'barplot': 'plot_barplot',
               'boxplot': 'plot_boxplot'}.get(type)
        
        while i < num_left:
            if len(self._keys) > j:
                # Get key
                k = self._keys[j]

                # Call children method to plot
                metric = None 
                if 'loss' in self.hyperspace_global_config._keys:
                    metric = self.hyperspace_global_config['loss']
                if num_descendants > 1:
                    yl = metric if i % num_cols == 0 else None
                    xl = 'Bit-Error Rate (BER)' if i >= ((num_rows - 2) * num_cols) else None
                else:
                    yl = ylabel
                    xl = 'Bit-Error Rate (BER)'
                
                netsurf.utils.log._info(f'Plotting {k}[j](axs[{i}:]) for {self.name} of type {self.type}', tab = self.level + 1)

                # Returns:
                # _f: figure
                # _i: index of the last axis used
                # _t: [List] of text objects
                _f, _ax, _i, _t, __line, _info = getattr(self._subs[k],fcn)(axs = axs[i:], x = x, y = y, hue = hue, 
                                                ylabel = yl, xlabel = xl,
                                                info_label = info_label, 
                                                is_entry_point = False,
                                                standalone = standalone,
                                                **kwargs)
                
                _axs += _ax
                _figs += _f
                _lines += __line
                t += _t
                _infos += _info

                # Update i
                i += _i

            else:
                # If we didn't plot anything, increment i
                num_left -= 1
                netsurf.utils.log._log(f'Skipping plot for j-th [{j}](axs[{i}:]), children of {self.name} because it is empty', tab = self.level + 1)
            j += 1
        
        # Delete empty axes (from i to num_rows * num_cols)
        if is_entry_point and not standalone:
            for k in range(i, len(axs)):
                fig.delaxes(axs[k])
            # Same for t, delete empty text objects
            t = t[:i]
        print('out')

        return [fig] if not standalone else _figs, _axs, i, t, _lines, _infos
    

    """ 2D Curves plot """
    def plot_2D_curves(self, *args, **kwargs):
        return self.plot_curves('2d', *args, **kwargs)
    
    """ 3D plot """
    def plot_3D_volumes(self, *args, **kwargs):
        return self.plot_curves('3d', *args, **kwargs)

    """ Barplot """
    def plot_barplot(self, *args, **kwargs):
        return self.plot_curves('barplot', *args, **kwargs)

    """ Box plot """
    def plot_boxplot(self, *args, **kwargs):
        return self.plot_curves('boxplot', *args, **kwargs)

    """ Local print function with the appropriate tabbing according to the level """
    def print(self, *args, print_fcn = print, **kwargs):
        # get tabs 
        tabs = '  ' * (self.level + 1)
        print(f"{tabs}", end = '')
        print_fcn(*args, **kwargs)
    
    """ Print tree """
    def tree(self, tab = 0):
        # Print this node
        tabs = '   ' * tab
        p = self.dir if tab == 0 else f"{'…/'*tab}{os.path.basename(self.dir)}"
        if self.level == 0:
            ss = f'{self.map_level[self.level]} "{self.name}" @ {p}\n'
        else:
            ss = f'{tabs} ↪ {self.map_level[self.level]} "{self.name}" @ {p}\n'
        # Print children
        for k in self._keys:
            ss += self._subs[k].tree(tab = tab + 1)
        return ss

    """ Representation """
    def __repr__(self, tab = 0):
        # Add extra space according to the level
        tabs = '  ' * tab
        p = self.dir if tab == 0 else f"{'…/'*tab}{os.path.basename(self.dir)}"
        map_level = self.hyperspace_global_config['map_level']
        ss = f'{tabs}{map_level[self.level]} "{self.name}" @ {p}'
        # Add children keys 
        ks = ', '.join([f'"{k}"' for k in self._keys])
        ss += f'\n{tabs}    ↪ [List: {BUCKET_CLASSES.get(map_level[self.level + 1], Bucket).__name__}]'
        for k in self._keys:
            ss += f'\n{tabs}       ↪ {k}'
        return ss
    
    """ HTML representation """
    @property
    def html(self):
        map_level = self.hyperspace_global_config['map_level']
        ss = f'<h1>Bucket [{map_level[self.level]}Container] : "{self.name}"</h1>\n'
        ss += f'<p>Information about this bucket:</p>\n'
        # Info
        ss += f'\t<ul>\n'
        if True: # This if statement is just to prettify the code here, so we keep track of the indentation in the actual html
            
            # Directory with href link to directory
            ss += f'\t\t<li><b>Directory:</b> <a href="file:/{self.dir}">{self.dir}</a></li>\n'
            
            # Show the kind of children it contains:
            ss += f'\t\t<li><b>Children: List[{map_level[self.level+1]}Container]</b>\n'
            if True: # This if statement is just to prettify the code here, so we keep track of the indentation in the actual html
                ss += f'\t\t\t<ul>\n'
                # Put children names 
                for k in self._keys:
                    ss += f'\t\t\t\t<li>{k}</li>\n'
                ss += f'\t\t\t</ul>\n'
            ss += f'\t\t</li>\n' # Close children

            # Show total elapsed time in format days hh:mm:ss
            if 'runtime' in self.structural_local_metrics:
                ss += f'\t\t<li><b>Total Runtime:</b> {netsurf.utils.get_elapsed_time(self.structural_local_metrics["runtime"])}</li>\n'
            
            if 'loss' in self.structural_local_metrics:
                loss = self.structural_local_metrics['loss']
                ss += f'\t\t<li><b>Loss:</b> {loss}</li>\n'
                ss += f'\t\t<li><b>{loss.title()} range:</b> [{self.structural_local_metrics["losses"][1]:3.2f}, {self.structural_local_metrics["losses"][0]:3.2f}]</li>\n'
            
            # Also metrics
            for k in self.structure_config['metrics']:
                if k in self.structural_local_metrics:
                    # Metric is None
                    ss += f'\t\t<li><b>{k.title()} range:</b> [{self.structural_local_metrics[k][1]:3.2%}, {self.structural_local_metrics[k][0]:3.2%}]</li>\n'

            # Show coverage statistics
            ss += f'\t\t<li><b>Coverage statistics:</b>\n'
            ss += f'\t\t\t<ul>\n'
            if True: # This if statement is just to prettify the code here, so we keep track of the indentation in the actual html
                for k, v in self.structural_local_metrics.items():
                    if k.startswith('num_'):
                        if k not in ['runtime', 'accuracy', 'loss', 'losses'] + self.structure_config['metrics']:
                            ss += f'\t\t\t\t<li><b>{k.title().replace("num","Number of ").replace("_"," ")}:</b> {v}</li>\n'
            ss += f'\t\t\t</ul>\n' 
            ss += f'\t\t</li>\n' # Close coverage statistics

            # Add the TMR range along with their colors. Their colors are specified in matplotlib color names, so first we need to get 
            # the equivalent RGBA values. 
            protection = self.structural_local_metrics['protection']
            cols = self.hyperspace_global_config['protection_colors']
            ss += f'\t\t<li><b>TMR:</b>\n'
            ss += f'\t\t<ul>\n'
            for i, t in enumerate(protection):
                if True: # This if statement is just to prettify the code here, so we keep track of the indentation in the actual html
                    # transform color from matplotlib name to rgba 
                    if t in cols:
                        rgba = matplotlib.colors.to_rgba(cols[t])
                        # convert to str 
                        rgba = f'rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]})'

                        ss += f'\t\t\t<li>\n'
                        ss += f'\t\t\t\t<span style="margin-left: 5px; vertical-align: middle; width: 40px; display: inline-block;">{100*t:05.2f}%</span>\n'
                        ss += f'\t\t\t\t<span style="background-color: {rgba}; border: 1px solid #505050; width: 40px; height: 10px; display: inline-block; vertical-align: middle;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>\n'
                        # QTextBrowser doesn't handle divs and styles properly, so the "width" tag isn't working here :/ 
                        #ss += f'\t\t\t\t<div style="background-color: {rgba}; border: 1px solid #505050; width: 40px; height: 10px; display: inline-block; vertical-align: middle;"></div>\n'
                        ss += f'\t\t\t</li>\n'
            ss += f'\t\t</ul>\n' 
            ss += f'\t\t</li>\n' # Close TMR

        ss += f'\t</ul>\n' # Close info
        
        # Extra info
        ss += f'<p>You can add additional descriptive text here if needed.</p>\n'
        return ss
    

""" Root is just a normal bucket """
class RootContainer(Bucket):
    """ Init """
    def __init__(self, dir, name, level = 0, **kwargs):
        # Set type before calling super
        self.type = 'root'

        # Call super (this calls get_children too) -- This will set the hyperspace_global_config
        super().__init__(dir, name, level = level, **kwargs)

    def propagate_global_metrics(self):
        # Before passing down, we can get rid of the first argument of the tuple for each value, which is just the function
        # we used to merge the values up.
        self.structural_local_metrics = {kw: kv for kw, kv in self.hyperspace_global_config.items()}
        # Now call super 
        super().propagate_global_metrics()
    

    """ Get coverage """
    def get_coverage(self):
        # Propagate down
        super().get_coverage()

        if self.coverage is None:
            out_coverage = self.results
            # Reorganize 
            out_coverage = out_coverage.rename(columns={'root':'bucket'})
            # Group by benchmark
            gb = out_coverage.groupby(['benchmark'])
            
            run_reps = gb['loss'].apply(lambda x: x.notna().sum())
            total_num_reps = gb['loss'].apply(lambda x: x.shape[0])
            coverage = run_reps/total_num_reps

            # reset index 
            run_reps = run_reps.reset_index()
            # Rename metric to run_reps
            run_reps = run_reps.rename(columns = {'loss': 'run_reps'})
            # Add total number of reps 
            run_reps['total_num_reps'] = total_num_reps.reset_index()['loss']
            # Add coverage
            run_reps['coverage'] = coverage.reset_index()['loss']
            # And rearraange columns
            out_coverage = run_reps[['benchmark', 'coverage']]
            # Reset index 
            out_coverage.reset_index(drop = True, inplace = True)
            # Set to local coverage before passing up 
            self.coverage = deepcopy(out_coverage)
        
        return self.coverage

""" Benchmark class """
class BenchmarkContainer(Bucket):
    _LEVEL = 1
    """ Init """
    def __init__(self, dir, name, level = 1, **kwargs):
        # Set type before calling super
        self.type = 'benchmark'

        # Call super (this calls get_children too) -- This will set the hyperspace_global_config
        super().__init__(dir, name, level = level, **kwargs)
    
    """ The quantization names are "q<m,n,s> but we store the directories as q<m>_<n>_<s> """
    def get_children_name_and_metadata(self, x, dir, *args, **kwargs):
        # try to get metadata 
        if x.startswith('q<') and x.endswith('>'):
            x = x.replace('<','').replace('>','').replace(',','_')
        metadata = get_metadata(os.path.join(dir,x))
        return x, {}, metadata


""" Quantization class """
class QuantizationContainer(Bucket):
    _LEVEL = 2
    """ Init """
    def __init__(self, dir, name, level = 2, **kwargs):
        # Set type before calling super
        self.type = 'quantization'

        # MAKE SURE THAT <dir> is NOT in the format q<m,n,s> but in the format q_<m>_<n>_<s>
        if name.startswith('q<') and name.endswith('>'):
            #name = name.replace('<','').replace('>','').replace(',','_')
            dir = os.path.join(os.path.dirname(dir), name.replace('<','').replace('>','').replace(',','_'))

        # Call super (this calls get_children too) -- This will set the hyperspace_global_config
        super().__init__(dir, name, level = level, **kwargs)

    def get_children_name(self, *args, **kwargs):
        return self.get_children_name_and_metadata(*args, **kwargs)[:1]

    def get_children_name_and_metadata(self, x, dir):
        # Let's try to get the pruning factor out, 
        # if we cannot, then we'll have to create a benchmark object and 
        # get the name of the model. 
        # First, let's try to see if we find anything in dir in the format "pruned_<float>_<model_name>"

        # If we already found "model_name", it will be in the structure 
        # (WHY WOULD WE HAVE THIS HERE? THIS IS QUANTIZATION LVL)
        metadata = None
        if 'model_name' in self.structure_config:
            model_name = self.structure_config['model_name']
            return f'pruned_{x}_{model_name}', {'pruning': x}, metadata

        # If not, then we need to figure it out. 
        """ Test 1: Get the model name from some directory """
        # Get valid directories in dir 
        if netsurf.utils.is_valid_directory(dir):
            valid_dirs = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
            for name in valid_dirs:
                # Do we have metadata for this directory that matches the expected child?
                metadata = get_metadata(os.path.join(dir, name))
                if metadata is not None:
                    if metadata.level == 'model':
                        if 'model_full_name' in metadata.config:
                            # Let's get the expected name for the model we are trying to build
                            model_name = metadata.config['model_name']
                            expected_model_full_name = f'pruned_{x}_{model_name}'

                            if metadata.config['model_full_name'] == expected_model_full_name:
                                # Great!! set the model name in the structure config and return 
                                self.structure_config['model_full_name'] = metadata.config.get('model_full_name')
                                self.structure_config['model_name'] = metadata.config.get('model_name')
                                self.structure_config['model_prefix'] = metadata.config.get('model_prefix')
                                # check if we also have the pruning factor
                                x = metadata.config.get('pruning', 0.0)
                                self.structure_config['total_num_params'] = metadata.config.get('total_num_params', 0)
                                self.structure_config['model_class'] = metadata.config.get('model_class')
                                self.structure_config['model_optimizer'] = metadata.config.get('optimizer','adam')
                                self.structure_config['model_loss'] = metadata.config.get('loss','categorical_crossentropy')
                                self.structure_config['model_metrics'] = metadata.config.get('metrics',[])
                                self.structure_config['model_in_shape'] = metadata.config.get('in_shape',(1,))
                                self.structure_config['model_out_shape'] = metadata.config.get('out_shape',(1,))
                                self.structure_config['model_optimizer_params'] = metadata.config.get('optimizer_params',{})

                                return metadata.config.get('model_full_name'), {'pruning': x}, metadata
                            
                            elif metadata.config['model_full_name'] == model_name:
                                # Beautiful, we have the model name, but not the pruning factor (it's probably 0.0 and this
                                # is why it was ommitted in the name)
                                self.structure_config['model_name'] = metadata.config.get('model_full_name')
                                self.structure_config['model_full_name'] = metadata.config.get('model_name')
                                self.structure_config['model_prefix'] = metadata.config.get('model_prefix')
                                self.structure_config['total_num_params'] = metadata.config.get('total_num_params', 0)
                                x = metadata.config.get('pruning', 0.0)
                                self.structure_config['model_class'] = metadata.config.get('model_class')
                                self.structure_config['model_optimizer'] = metadata.config.get('optimizer','adam')
                                self.structure_config['model_loss'] = metadata.config.get('loss','categorical_crossentropy')
                                self.structure_config['model_metrics'] = metadata.config.get('metrics',[])
                                self.structure_config['model_in_shape'] = metadata.config.get('in_shape',(1,))
                                self.structure_config['model_out_shape'] = metadata.config.get('out_shape',(1,))
                                self.structure_config['model_optimizer_params'] = metadata.config.get('optimizer_params',{})
                                return model_name, {'pruning': x}, metadata
                            else:
                                # Well, this is probably a different version (of pruning) of the model
                                # we are trying to use, so let's just use the expected_model_full_name
                                # as the name, but the rest of the variables hold
                                self.structure_config['model_name'] = model_name
                                self.structure_config['model_full_name'] = expected_model_full_name
                                self.structure_config['model_prefix'] = f'pruned_{x}_'
                                self.structure_config['total_num_params'] = metadata.config.get('total_num_params', 0)
                                return expected_model_full_name, {'pruning': x}, None
                            
                # Check if name starts with pruned_
                elif name.startswith('pruned_'):
                    pruning_factor, model_name = netsurf.utils.get_pruning_factor(name)
                    # if the results make sense, great
                    if isinstance(pruning_factor, float) and isinstance(model_name, str):
                        # Great, set the model name in the structure config and return 
                        self.structure_config['model_name'] = model_name
                        # Inform the user 
                        tabs = '  ' * (self.level + 1)
                        netsurf.utils.log._info(f"{tabs}(?/?) Able to deduce the model name for this particular structure: {model_name} @ {dir}", prefix = f'', tab = 0)
                        return f'pruned_{x}_{model_name}', {'pruning': x}, metadata

        """ Test 2: Get the model name from a benchmark object initialized with this config """
        benchmark = self.structure_config['benchmark']
        quantization = self.structure_config['quantization']
        datasets_dir = None
        workdir = "/tmp/netsurf"
        model_prefix = ""

        try:
            # Convert quantization to object
            Q = netsurf.QuantizationScheme(quantization)
            
            # Create benchmark object
            bmk = netsurf.get_benchmark(benchmark, Q, 
                                      datasets_dir = datasets_dir, 
                                      benchmarks_dir = workdir, 
                                    load_weights = False, model_prefix = model_prefix, 
                                    **netsurf.config.BENCHMARKS_CONFIG[benchmark], 
                                    verbose = False)
                

            # Try to extract the model name from the benchmark object
            model_name = bmk.model_name
            # If we can, then we will return the quantization name
            return f'pruned_{x}_{model_name}', {'pruning': x}, metadata
        
        except Exception as e:
            # If we cannot, then we will consider this a failure and return the quantization name
            netsurf.utils.log._warn(f"Could not deduce the model name for this particular structure @ {dir}\nBenchmark objected created failed with code {e}\nCreating a placeholder for this structure.")
            return f'pruned_{x}', {'pruning': x}, metadata
    
    
    """ Set recursively """
    def propagate_coverage_pie(self, **kwargs):
        """ Loop thru pies """
        for k in self._keys:
            # Get pruning factor for this model 
            x = self._subs[k].pruning_factor
            if k in self.coverage_pie.subpies:
                # Set the pie in the child
                self._subs[k].coverage_pie = self.coverage_pie.subpies[k]
                self._subs[k].propagate_coverage_pie(**kwargs)
            elif x in self.coverage_pie.subpies:
                # Set the pie in the child
                self._subs[k].coverage_pie = self.coverage_pie.subpies[x]
                self._subs[k].propagate_coverage_pie(**kwargs)

    def get_coverage(self):
        self.coverage = super().get_coverage(groupby=['pruning', 'model_name'])
        return self.coverage

    def plot_vus_vs_pruning(self, x = 'pruning', y = 'vus', hue = 'method',
                     axs = None, colors = None, 
                     xrange = None, yrange = None, 
                     xlabel = 'Pruning Rate (%)', ylabel = 'VUS', 
                     info_label = None, standalone = True, **kwargs):

        # If this is just a placeholder (with empty data), skip
        if len(self.coverage) == 0:
            netsurf.utils.log._warn(f'Empty run placeholder. Skipping plot_curves.', tab = self.level)
            return [], [], 0, [], [], []

        # Pass the global metrics
        if colors is None:
            colors = self.hyperspace_global_config[f'{hue}_colors'] if f'{hue}_colors' in self.hyperspace_global_config._keys else None
        if xrange is None:
            xrange = self.hyperspace_global_config[x] if x in self.hyperspace_global_config._keys else None
        metric = self.hyperspace_global_config['loss'] if 'loss' in self.hyperspace_global_config._keys else 'mse'
        if yrange is None:
            yrange = self.hyperspace_global_config[metric] if metric in self.hyperspace_global_config._keys else None

        # Remove "root" from the structure config
        info_label_fields = ['benchmark', 'quantization', 'model', 'method', 'run']
        if info_label is None:
            info_label = {k: self.structure_config[k] for k in info_label_fields if k != 'root' and k in self.structure_config}
        
        # If ax exists, make sure to add this type to info_label
        if axs is not None:
            info_label[self.type] = self.name
            # make sure axs is a list
            if not isinstance(axs, list) and not isinstance(axs, np.ndarray):
                axs = [axs]

        # If ax doesn't exists, create figure 
        if axs is None or standalone:
            fig, axs = plt.subplots(1, 1, figsize = (5, 5))
            axs = [axs]
        else:
            fig = axs[0].figure
        
        # We need to gather the AUCs and VUSs for every model that's children to 
        # this bucket, so let's 
        subplotters = {}
        for pn in self:
            p = self._subs[pn]
            pf = p.pruning_factor
            pmn = p.model_name
            for cn in p:
                # Get the results
                child = p._subs[cn]
                # Get the plotter
                subplotters[(pf, pmn, cn)] = child.plotter
        
        # Call the plotter
        _fig, _ax, _t, _lines = netsurf.gui.plotter.plot_vus_vs_pruning(subplotters,
            ax = axs[0], y = y, metric = metric, 
            colors = colors,
            xrange = xrange, yrange = yrange,
            xlabel = xlabel, ylabel = ylabel,
            info_label = info_label, 
            standalone = standalone,
            xlog = False, ylog = False,
            **kwargs)
        
        if not isinstance(_ax, list) and not isinstance(_ax, np.ndarray):
            _ax = [_ax]

        # Make sure ax has keywords applied 
        if not standalone:
            fig_out = [fig]
            plt.close(_fig)
        else:
            fig_out = [_fig]
            plt.close(fig)

        return fig_out, _ax, 1, [_t], [_lines], [deepcopy(info_label)]


""" Model class """
class ModelContainer(Bucket):
    _LEVEL = 3
    """ Init """
    def __init__(self, dir, name, level = 3, **kwargs):
        # Set type before calling super
        self.type = 'model'

        # Call super (this calls get_children too) -- This will set the hyperspace_global_config
        super().__init__(dir, name, level = level, **kwargs)

        self.pruning_factor = self.structure_config['pruning'] if 'pruning' in self.structure_config else None
        self.model_name = self.structure_config['model_name'] if 'model_name' in self.structure_config else None
        self.model_full_name = self.structure_config['model_full_name'] if 'model_full_name' in self.structure_config else None
        self.model_prefix = self.structure_config['model_prefix'] if 'model_prefix' in self.structure_config else None
        self.total_num_params = self.structure_config['total_num_params'] if 'total_num_params' in self.structure_config else 0

        # Try to find the benchmark object in this dir, if we have it 
        self.benchmark_obj, self.benchmark_obj_path = self.find_benchmark_object(dir, name)

        # if we have the benchmark object, we can get the total_num_params from it
        if self.benchmark_obj is not None:
            self.total_num_params = self.benchmark_obj.model.count_trainable_parameters() - self.benchmark_obj.model.count_pruned_parameters(),

    """ Find benchmark object """
    def find_benchmark_object(self, dir, name):
        # Try to find a ".bmk" file in this dir
        # List all .netsurf.bmk files in this directory
        if not netsurf.utils.is_valid_directory(dir):
            netsurf.utils.log._warn(f'Invalid directory {dir} for model {name}. Skipping.')
            return None, None
        bmk_files = [f for f in os.listdir(dir) if f.endswith('.netsurf.bmk')]
        if len(bmk_files) == 0:
            netsurf.utils.log._warn(f'No benchmark object found in {dir} with extension .netsurf.bmk')
            return None, None
        # If we have more than one, loop thru them and look at the extra attributes with xattr
        if 'benchmark' not in self.structure_config:
            netsurf.utils.log._warn(f'Benchmark is not defined for this model {name} so we cannot find the equivalent benchmark object at this directory.')
            return None, None
        benchmark = self.structure_config['benchmark']
        for f in bmk_files:
            # Get the full path
            full_path = os.path.join(dir, f)
            # Get the attributes
            attrs = netsurf.utils.get_xattrs(full_path)
            # If the name is the same, return this object
            if 'name' in attrs and 'class' in attrs:
                if attrs['class'] == 'netsurf.Benchmark':
                    if attrs['benchmark'].lower() == benchmark.lower():
                        netsurf.utils.log._info(f'Loading benchmark object {name} @ {full_path}')
                        return netsurf.utils.load_object(full_path), full_path
        # Log 
        netsurf.utils.log._warn(f'No benchmark object found in {dir} with extra attributes (xattr) of type benchmark, matching the name {name}')
        return None, None
            
    # Get children is a bit different cause at this level we have 3 folders: experiments, models, sessions
    def get_children(self, verbose = True, pbar = RecursiveProgressTracker(None), **kwargs):
        # Get all experiments 
        experiments = os.path.join(self.dir, 'experiments')
        models = os.path.join(self.dir, 'models')
        sessions = os.path.join(self.dir, 'sessions')

        """ The children of this bucket are NOT only the directories in dir directory, 
            but also the EXPECTED children. That info is in the hyperspace_global_config 
            If the children are in the directory, great, we will create them and fill them 
            with whatever we find and go down the path recursively. If they are not, we will
            still create a placeholder for them, but we will not fill them with anything.
            This placeholder will allow us later on to keep track of the jobs that are missing
            and the coverage of the whole structure.
        """
        # Get the EXPECTED children for this bucket 
        children_type = self.structure_config['children_type']
        expected_children = self.structure_config['expected_children']

        # Get children Bucket class
        children_bucket_class = BUCKET_CLASSES.get(children_type, Bucket)
        
        # Before starting, keep the old factor and offsets of the pbar 
        old_factor = pbar.factor
        old_offset = pbar.offset
        # Now compute the new factor
        new_factor = pbar.get_next_factor(len(expected_children))
        num_children = len(expected_children)

        # Init children and children names
        children = {}
        children_names = []

        """ Loop thru expected children """
        for ichild, child in enumerate(expected_children):
            # Get the expected children name. This is often just the same as "child", but
            # for the model level we are actually looping thru pruning values <0.0, 0.125, etc.>,
            # but what we will have on disk is some directory like "pruned_0.0_<model_name>", etc.
            # Thus, we'll need to extract that model name at the QuantizationContainer level. 
            children_name, extra_structure_config_args, children_metadata = self.get_children_name_and_metadata(child, experiments)

            # Construct the full path expected for this child
            full_path = os.path.join(experiments, children_name)

            # Create a new pbar for this child, with new offset 
            new_offset = old_offset + (ichild * new_factor)
            subpbar = RecursiveProgressTracker(pbar.pbar, offset = new_offset, factor = new_factor)

            # Init message to be displayed
            tabs = '  ' * (self.level + 1)
            p = f"{'…/'*(self.level+1)}{os.path.basename(full_path)}"
            msg = f"{tabs}({ichild+1}/{num_children}) - {children_type} {children_name} @ {p}"
            if verbose: netsurf.utils.log._info(msg, prefix = f'', tab = 0)

            """
                IMPORTANT: At this point we have two possibilities:
                    (1) If the directory exists AND it has a .metadata.netsurf file inside,
                        we have to follow whatever that metadata has inside. 
                        This is weird, but it allows us to have model buckets within other 
                        model buckets, etc. But if it is what the user wanted, then we need
                        to follow this. 
                    (2) If the directory doesn't exist OR if it exists but the .metadata.netsurf
                        doesn't, we have to proceed as usual, inferring the children type that
                        would be expected at this level and creating a placeholder for it.
            """
            if children_metadata is None:
                # Try to find it again
                children_metadata = get_metadata(full_path)

            if children_metadata is not None:
                # Log
                netsurf.utils.log._info(f"{tabs}Found metadata file for {children_name} @ {full_path} Following metadata.", tab = self.level + 1)
                # Get the children type from the metadata
                children_type = children_metadata.level
                # Assert the name is the same we were expecting (should be)
                # TODO:
                if False:
                    assert children_name == children_metadata.name, f"Expected children name {children_name} but found {children_metadata.name}"
                
                children_name = children_metadata.name
                # Get children Bucket class
                children_bucket_class = BUCKET_CLASSES.get(children_type.capitalize(), Bucket)
                # Add the config in the metadata to extra_structure_config_args
                extra_structure_config_args = {**extra_structure_config_args, **children_metadata.config}

            
            # Call subpbar to update the progress
            subpbar(new_offset, f"Processing {msg}")

            # Create a copy of the structure config
            new_structure_config = deepcopy(self.structure_config)
            # Add the extra_structure_config_args 
            new_structure_config = {**new_structure_config, **extra_structure_config_args}

            # Create children REGARDLESS of whether it exists or not (it will be handled downstream)
            c = children_bucket_class(full_path, children_name, level = self.level + 1, 
                                        verbose = verbose, pbar = subpbar, 
                                        hyperspace_global_config = self.hyperspace_global_config, 
                                        structure_config = new_structure_config,
                                        **kwargs)

            # Apprend to the children and children names
            children_names.append(children_name)
            children[children_name] = c

        return children, children_names

    """ Add pruning to results """
    def get_results(self, pbar = RecursiveProgressTracker(None), **kwargs):
        # Get results
        results = super().get_results(pbar = pbar)
        # Add pruning_factor, model and model_name
        results['pruning_factor'] = self.pruning_factor
        results['model'] = self.model_full_name
        results['model_name'] = self.model_name
        # Rename pruning_factor to pruning
        results = results.rename(columns = {'pruning_factor':'pruning'})
        # Set to self 
        self.results = results
        return self.results

    def get_missing_jobs(self):
        missing_jobs = []
        for k in self._keys:
            subjobs = self._subs[k].get_missing_jobs()
            # subjobs is a pd.DataFrame, add current level to it 
            subjobs[self.type] = self.name
            # Add pruning factor
            subjobs['pruning'] = self.pruning_factor
            # Add model name
            subjobs['model'] = self.model_name
            missing_jobs.append(subjobs)
        if len(missing_jobs) > 0:
            missing_jobs = pd.concat(missing_jobs, ignore_index=True)
            # Rearrange columns with self.type first 
            cols = missing_jobs.columns.tolist()
            cols = [self.type] + [c for c in cols if c != self.type]
            missing_jobs = missing_jobs[cols]
        else:
            missing_jobs = pd.DataFrame()
        return missing_jobs

    def plot_boxplot(self, *args, **kwargs):
        return self.plot_methodwise('boxplot', *args, **kwargs)

    def plot_barplot(self, *args, **kwargs):
        return self.plot_methodwise('barplot', *args, **kwargs)

    def plot_methodwise(self, type, x = 'ber', y = 'mean', hue = 'protection', 
                     axs = None, colors = None, 
                     xrange = None, yrange = None, metric = None,
                     xlabel = 'Bit-Error Rate (BER)', ylabel = 'Accuracy', 
                     info_label = None, standalone = True, **kwargs):

        # If this is just a placeholder (with empty data), skip
        if len(self.coverage) == 0:
            netsurf.utils.log._warn(f'Empty run placeholder. Skipping plot_curves.', tab = self.level)
            return [], [], 0, [], [], []

        # Pass the global metrics
        if colors is None:
            colors = self.hyperspace_global_config[f'{hue}_colors'] if f'{hue}_colors' in self.hyperspace_global_config._keys else None
        if xrange is None:
            xrange = self.hyperspace_global_config[x] if x in self.hyperspace_global_config._keys else None
        
        if metric is not None:
            # check if exists in hyperspace_global_config
            if metric not in self.hyperspace_global_config._keys:
                metric = None
        if metric is None:
            metric = self.hyperspace_global_config['loss'] if 'loss' in self.hyperspace_global_config._keys else 'mse'
        if yrange is None:
            yrange = self.hyperspace_global_config[metric] if metric in self.hyperspace_global_config._keys else None

        # Remove "root" from the structure config
        info_label_fields = ['benchmark', 'quantization', 'model', 'method', 'run']
        if info_label is None:
            info_label = {k: self.structure_config[k] for k in info_label_fields if k != 'root' and k in self.structure_config}
        
        # If ax exists, make sure to add this type to info_label
        if axs is not None:
            info_label[self.type] = self.name
            # make sure axs is a list
            if not isinstance(axs, list) and not isinstance(axs, np.ndarray):
                axs = [axs]

        # If ax doesn't exists, create figure 
        if axs is None or standalone:
            fig, axs = plt.subplots(1, 1, figsize = (5, 5))
            axs = [axs]
        else:
            fig = axs[0].figure
        
        # We need to gather the AUCs and VUSs for every method that's children to 
        # this bucket, so let's 
        subplotters = {}
        for cn in self:
            # Get the results
            child = self._subs[cn]
            # Get the plotter
            subplotters[cn] = child.plotter
        
        # Make sure we have the attribute 
        if not hasattr(netsurf.gui.plotter, f'plot_{type}'):
            netsurf.utils.log._warn(f'Plotter plot_{type} not found in netsurf.gui.plotter. Skipping plot_curves.', tab = self.level)
            return [fig], [axs[0]], 0, [], [], []

        _fig, _ax, _t, _lines = getattr(netsurf.gui.plotter,f'plot_{type}')(subplotters,
            ax = axs[0], y = y, metric = metric, 
            colors = colors,
            xrange = xrange, yrange = yrange,
            xlabel = xlabel, ylabel = ylabel,
            info_label = info_label, 
            standalone = standalone,
            xlog = False, ylog = False,
            **kwargs)
        
        if not isinstance(_ax, list) and not isinstance(_ax, np.ndarray):
            _ax = [_ax]

        # Make sure ax has keywords applied 
        if not standalone:
            fig_out = [fig]
            plt.close(_fig)
        else:
            fig_out = [_fig]
            plt.close(fig)

        return fig_out, _ax, 1, [_t], [_lines], [deepcopy(info_label)]
        

    
""" Method container 
        Even though it's called a method container, this is actually the experiment itself. 
        Each run is just a run for this particular experiment.
        This class should leverage:
            - the ability to call new runs, etc.
            - the ability to combine all results from the runs
            - the ability to check what is the global coverage
            - the ability to check the ranking
            - the ability to check the global metrics
            - ...
"""
class MethodContainer(Bucket):
    _LEVEL = 4
    """ Init """
    def __init__(self, dir, name, level = 4, **kwargs):
        # Set type before calling super
        self.type = 'method'

        # Call super (this calls get_children too) -- This will set the hyperspace_global_config
        super().__init__(dir, name, level = level, **kwargs)

    """ Children are the runs """
    def get_children(self, verbose=True, pbar=RecursiveProgressTracker(None), **kwargs):
        # Print if verbose
        tabs = '  ' * (self.level)
        if verbose and self.level == 0: netsurf.utils.log._info(f"{tabs} [{self.hyperspace_global_config['map_level'][self.level]}Container]: {self.name} @ {self.dir}", tab = 0)
        
        """ The children of this bucket are NOT only the directories in dir directory, 
            but also the EXPECTED children. That info is in the hyperspace_global_config 
            If the children are in the directory, great, we will create them and fill them 
            with whatever we find and go down the path recursively. If they are not, we will
            still create a placeholder for them, but we will not fill them with anything.
            This placeholder will allow us later on to keep track of the jobs that are missing
            and the coverage of the whole structure.
        """
        # Get the EXPECTED children for this bucket 
        children_type = self.structure_config['children_type']
        # Expected children here are anything that's a directory that follows the pattern "config(\d+)"
        expected_children = []
        if netsurf.utils.is_valid_directory(self.dir):
            expected_children = [d for d in os.listdir(self.dir) if os.path.isdir(os.path.join(self.dir, d)) and re.match(r'config\d+', d)]
        # Update structure 
        self.structure_config['expected_children'] = expected_children
        
        # The children is technically not a bucket, but a wrapper around the Experiment object in core/experiments.py

        # Get children Bucket class
        children_bucket_class = BUCKET_CLASSES.get(children_type, Bucket)
        
        # Before starting, keep the old factor and offsets of the pbar 
        old_factor = pbar.factor
        old_offset = pbar.offset
        # Now compute the new factor
        num_children = len(expected_children) if expected_children is not None else 1
        new_factor = pbar.get_next_factor(num_children)

        # Init children and children names
        children = {}
        children_names = []

        """ Loop thru expected children """
        for ichild, child in enumerate(expected_children):
            # Get the expected children name. This is often just the same as "child", but
            # for the model level we are actually looping thru pruning values <0.0, 0.125, etc.>,
            # but what we will have on disk is some directory like "pruned_0.0_<model_name>", etc.
            # Thus, we'll need to extract that model name at the QuantizationContainer level. 
            try:
                children_name, extra_structure_config_args, children_metadata = self.get_children_name_and_metadata(child, self.dir)
            except:
                print('stop here')

            # Construct the full path expected for this child
            full_path = os.path.join(self.dir, children_name)

            # Create a new pbar for this child, with new offset 
            new_offset = old_offset + (ichild * new_factor)
            subpbar = RecursiveProgressTracker(pbar.pbar, offset = new_offset, factor = new_factor)

            # Init message to be displayed
            tabs = '  ' * (self.level + 1)
            p = f"{'…/'*(self.level+1)}{os.path.basename(full_path)}"
            msg = f"{tabs}({ichild+1}/{num_children}) - [{children_type.capitalize()}Container]: {children_name} @ {p}"
            if verbose: netsurf.utils.log._info(msg, tab = 0)

            """
                IMPORTANT: At this point we have two possibilities:
                    (1) If the directory exists AND it has a .metadata.netsurf file inside,
                        we have to follow whatever that metadata has inside. 
                        This is weird, but it allows us to have model buckets within other 
                        model buckets, etc. But if it is what the user wanted, then we need
                        to follow this. 
                    (2) If the directory doesn't exist OR if it exists but the .metadata.netsurf
                        doesn't, we have to proceed as usual, inferring the children type that
                        would be expected at this level and creating a placeholder for it.
            """
            if children_metadata is None:
                # Try to find it again
                children_metadata = get_metadata(full_path)

            if children_metadata is not None:
                # Log
                netsurf.utils.log._info(f"{tabs}Found metadata file for {children_name} @ {full_path} Following metadata.", tab = self.level + 1)
                # Get the children type from the metadata
                children_type = children_metadata.level
                # Assert the name is the same we were expecting (should be)
                # TODO:
                if False:
                    assert children_name == children_metadata.name, f"Expected children name {children_name} but found {children_metadata.name}"
                # Override for now 
                children_name = children_metadata.name
                full_path = os.path.join(self.dir, children_name)
                # Get children Bucket class
                children_bucket_class = BUCKET_CLASSES.get(children_type.capitalize(), Bucket)
                # Add the config in the metadata to extra_structure_config_args
                extra_structure_config_args = {**extra_structure_config_args, **children_metadata.config}

            
            # Call subpbar to update the progress
            subpbar(new_offset, f"Processing {msg}")

            # Create a copy of the structure config
            new_structure_config = deepcopy(self.structure_config)
            # Add the extra_structure_config_args 
            new_structure_config = {**new_structure_config, **extra_structure_config_args}

            # Create children REGARDLESS of whether it exists or not (it will be handled downstream)
            c = children_bucket_class(full_path, children_name, level = self.level + 1, 
                                        verbose = verbose, pbar = subpbar, 
                                        hyperspace_global_config = self.hyperspace_global_config, 
                                        structure_config = new_structure_config,
                                        metadata = children_metadata,
                                        **kwargs)

            # Apprend to the children and children names
            children_names.append(children_name)
            children[children_name] = c

        return children, children_names
    
    def get_missing_jobs(self):
        # if coverage is less than zero, simply return this 
        if self.coverage['coverage'].mean() < 1.0:
            return pd.DataFrame({'method':self.name}, index = [0])
        return pd.DataFrame()

    @property
    def plotter(self):
        # Gather children plotters
        children_plotters = {}
        for k in self._keys:
            children_plotters[k] = self._subs[k].plotter
        return children_plotters

    """ Print tree """
    def tree(self, tab = 0):
        # Print this node
        # Add extra space according to the level
        tabs = '   ' * (tab + 1)
        p = self.dir if tab == 0 else f"{'…/'*tab}{os.path.basename(self.dir)}"
        ss = f'{tabs} ↪ Method "{self.name}" @ {p}'
        # show status
        icon = "✅" if self.status == 'completed' else "❌" if self.status == 'error' else "🔄" if self.is_running else "❓"
        ss += f'\n{tabs}    ↪ Status: {icon} {self.status}\n'
        # Now loop thru kids
        for k in self._keys:
            ss += self._subs[k].tree(tab = tab + 1)
        return ss
    
    """ Representation showing the status of the run and some statitics """
    def __repr__(self, tab = 0):
        # Add extra space according to the level
        tabs = '   ' * (tab)
        p = self.dir if tab == 0 else f"{'…/'*tab}{os.path.basename(self.dir)}"
        ss = f'{tabs}Method "{self.name}" @ {p}'
        # show status
        #icon = "✅" if self.status == 'completed' else "❌" if self.status == 'error' else "🔄" if self.is_running else "❓"
        #ss += f'\n{tabs}    ↪ Status: {icon} {self.status}\n'
        # Now loop thru kids
        for k in self._keys:
            ss += self._subs[k].__repr__(tab = tab + 1)
        return ss
    
class ExperimentWrapper(HierarchicalContainer):
    _LEVEL = 5
    """ Init """
    def __init__(self, dir, name, level = 5, hyperspace_global_config = None, structure_config = {}, metadata = None, **kwargs):
        # Set variables before anything else
        self.dir = dir
        self.name = name
        self.level = level
        self.type = 'experiment'
        self.metadata = metadata
        self.is_running = False

        # Set hyperspace_global_config
        self.hyperspace_global_config = hyperspace_global_config if isinstance(hyperspace_global_config, HyperspaceConfig) else HyperspaceConfig('global', **kwargs)

        # # Init structure_config
        self.structure_config = self.init_structure_config(structure_config)

        # Try to initialize the ACTUAL experiment object here 
        self.experiment_obj, self.experiment_obj_path = self.find_experiment_object(dir, name)

        # Let's get the results now 
        self.results = self.get_results()

        # Now get the coverage, results and status of this experiment
        self.status, self.coverage, self.coverage_progress, self.coverage_table = self.update_experiment_status()

        # Get the plotter
        self.plotter = self.build_plotters(**kwargs)

        # In this particular case, the local metrics are the same as the global metrics
        self.structure_config = {**self.structure_config, **self.get_structural_global_metrics()}
        
    """ Init structure config """
    def init_structure_config(self, structure_config):
        
        # Make a deep copy of the structure config
        sc = deepcopy(structure_config)

        if self.metadata is not None:
            # Put the metadata in the structure config
            sc = {**sc, **self.metadata.config}

        # Put itself into the config 
        sc[self.type] =  self.name

        return sc

    """ Get the global metrics for this run """
    def get_structural_global_metrics(self):

        # Add metric
        loss = self.structure_config.get('loss')
        losses = [np.max(self.results[loss]), np.min(self.results[loss])]
        # Add 
        self.hyperspace_global_config['loss'] = loss
        self.hyperspace_global_config['losses'] = losses

        for m in self.structure_config.get('metrics', ['categorical_accuracy']):
            if m in self.results.columns:
                _m = self.results[m]
                self.hyperspace_global_config[m] = [np.max(_m), np.min(_m)]


        sgm = deepcopy({kw: kv for kw, kv in self.hyperspace_global_config.items()})

        return sgm

    def find_experiment_object(self, dir, name):
        # First of all, let's try to see if there's a nodus metadata object here,
        # which will point us to the right db entry. This entry will have almost all
        # the info we need to initialize the experiment object.
        nodus_metadata = netsurf.utils.get_metadata(dir, filename = '.metadata.nodus')
        # If nodus metadata is None, well, things we a bit complicated. 
        # Then, we need to try to initialize the experiment object from scratch.
        if nodus_metadata is not None:
            raise NotImplementedError('Not implemented yet, TODO')
            c = {}
            if 'method' in self.structure_config:
                c['method'] = self.structure_config['method']
            if 'reload_ranking' in self.structure_config:
                c['reload_ranking'] = self.structure_config['reload_ranking']
        
        if not netsurf.utils.is_valid_directory(dir):
            netsurf.utils.log._warn(f'Invalid directory {dir} for experiment {name}. Skipping.')
            return None, None

        # Try to find a ".exp" file in this dir
        # List all .netsurf.exp files in this directory
        exp_files = [f for f in os.listdir(dir) if f.endswith('.netsurf.exp')]
        if len(exp_files) == 0:
            netsurf.utils.log._warn(f'No experiment object found in {dir} with extension .netsurf.exp')
            return None, None
        # If we have more than one, loop thru them and look at the extra attributes with xattr
        for f in exp_files:
            # Get the full path
            full_path = os.path.join(dir, f)
            # Get the attributes
            attrs = netsurf.utils.get_xattrs(full_path)
            # If the name is the same, return this object
            if 'name' in attrs and 'type' in attrs:
                if attrs['name'] == name and attrs['type'] == 'experiment':
                    netsurf.utils.log._info(f'Loading experiment object {name} @ {full_path}')
                    return netsurf.utils.load_object(full_path), full_path
        # Log 
        netsurf.utils.log._warn(f'No experiment object found in {dir} with extra attributes (xattr) of type experiment, matching the name {name}')
        return None, None
    
    """ Get results """
    def get_results(self, *args, **kwargs):
        
        # Initialize a resultsSpace object 
        loss_name = self.structure_config.get('loss', 'categorical_crossentropy')
        metrics_names = self.structure_config.get('metrics', ['categorical_accuracy'])
        total_num_params = self.structure_config.get('total_num_params', 0)
        # self.results = netsurf.core.experiments.ResultSpace(loss_name, metrics_names,
        #                                                   protection=self.hyperspace_global_config['protection'], 
        #                                                 ber=self.hyperspace_global_config['ber'], 
        #                                                 num_reps=-1,#num_reps=self.hyperspace_global_config['num_reps'],
        #                                                 total_num_params=total_num_params,
        #                                                 data = data,
        #                                                 columns = cols)


        # Chck if we have the data already 
        data = None
        extra_args = {}
        results_file = os.path.join(self.dir, 'results.csv')

        # If the experiment object is not None, update the results with these values 
        if self.experiment_obj is not None:
            data = self.experiment_obj.results
            extra_args = {'columns': data.columns}

        elif netsurf.utils.path_exists(results_file):
            # Read it 
            data = pd.read_csv(results_file)
            # Get the data
            extra_args['columns'] = data.columns
            netsurf.log(f'Updating results from {results_file}...')
        else:
            netsurf.log(f'Initializing empty placeholder results object for {self.name}...')


        results = netsurf.core.experiments.ResultSpace(loss_name, metrics_names,
                                                        protection = self.hyperspace_global_config['protection'],
                                                    ber = self.hyperspace_global_config['ber'],
                                                    num_reps = -1, #self.hyperspace_global_config['num_reps'],
                                                    total_num_params = total_num_params,
                                                    groupby = ['protection','ber'],
                                                    data = data, **extra_args)
        
        # Add experiment to the results 
        results['experiment'] = self.name

        return results

        if False:
            # Update the results object
            start_time = time()
            netsurf.log(f'Updating results from {results_file}...')
            results.update(df)
            end_time = time()
            netsurf.log(f'Updated results from {results_file} in {end_time - start_time:.2f} seconds.')
            return results

        # If we're here, then we couldn't find the results.csv file
        netsurf.utils.log._warn(f'No results.csv file found in {self.dir}')
        return results

    def update_experiment_status(self):
        # First check if we have an experiment object loaded 
        if self.experiment_obj is None:
            # Then let's try to get the results directly from the .csv files
            coverage_df, coverage_progress, coverage_table = self.results.get_coverage()

            if coverage_progress >= 1:
                # This is a completed run, nothing to do
                status = 'completed'
            elif coverage_progress >= 0:
                # This is an incomplete run, we need to see what's left to run
                # and generate the job for it (in case we run it in the future)
                status = 'incomplete'
            else:
                # This is an error run, we need to see what's wrong
                # and generate the job for it (in case we run it in the future)
                status = 'error'
            
            # Return 
            return status, coverage_df, coverage_progress, coverage_table
            

        # If we do have it, pick the status from there
        raise NotImplementedError('Not implemented yet, TODO')
    
    """ Get coverage """
    def get_coverage(self):
        cov = deepcopy(self.coverage)
        if len(cov) > 0:
            # Extend the coverage_df with the type and name
            cov[self.type] = self.name
        else:
            # Empty results
            cov = pd.DataFrame()
        
        # Reorganize 
        cov = cov[['experiment', 'protection', 'ber', 'coverage', 'run_reps', 'total_num_reps']]

        # Set subcoverage 
        self.subcoverage = cov
        return cov

    """ Get descendants (children & children of children & children of children of children ... ) """
    @property
    def descendants(self):
        return 1

    """ Run is not iterable """
    def __iter__(self):
        pass

    def get_children(self, *args, **kwargs):
        return {}, []

    """ Propagate the global metrics down (to this end point) """
    def propagate_global_metrics(self):
        # pass because at this level we have no children
        pass

    """ Load the configuration of the run """
    def load_config(self):
        # Check if the config file exists
        config_file = os.path.join(self.dir, 'config.json')
        if not netsurf.utils.path_exists(config_file):
            netsurf.utils.log._warn(f'No config.json file found in {self.dir}')
            return None
        
        # Read json
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        return config

    """ Build plotter for results """
    def build_plotters(self, **kwargs):
        # Get plotters for loss + metrics
        metric = 'loss'
        if self.results._loss:
            metric = self.results._loss
            if metric not in self.results:
                metric = 'loss'
        elif 'loss' in self.results:
            if len(self.results['loss']) > 0:
                try:
                    metric = self.results['loss'].mode()[0]
                except Exception as e:
                    print(e)
        elif 'loss' in self.structure_config:
            metric = self.structure_config['loss']
        elif 'benchmark' in self.structure_config:
            b = self.structure_config['benchmark']
            if b in netsurf.config.BENCHMARKS_CONFIG:
                metric = netsurf.config.BENCHMARKS_CONFIG[b].get('loss', 'accuracy')
        # Build plotter
        loss_plotter = netsurf.gui.plotter.ExperimentsPlotter(self.results, metric = metric, structure_config = self.structure_config, **kwargs)

        # Now for metrics 
        plotters = {'loss': loss_plotter}
        for m in self.results._metrics:
            if m.lower() not in self.results:
                if m in self.results:
                    plotters[m] = netsurf.gui.plotter.ExperimentsPlotter(self.results, metric = m, structure_config = self.structure_config, **kwargs)
                else:
                    continue
            else:
                plotters[m.lower()] = netsurf.gui.plotter.ExperimentsPlotter(self.results, metric = m.lower(), structure_config = self.structure_config, **kwargs)
            
        return plotters

    """ Plotting functions (will delegate this to the plotter, but we need this access thru here) """
    def plot_curves(self, type, *args, x = 'ber', y = 'mean', hue = 'protection', axs = None, colors = None, 
                        xrange = None, yrange = None, metric = None,
                        xlabel = 'Bit-Error Rate (BER)', ylabel = 'Accuracy', 
                        info_label = None, standalone = True, **kwargs):
        
        # Assert type in ['2d','3d']
        if type not in ['2d','3d']:
            netsurf.utils.log._error(f'Invalid type {type} for plot_curves. Must be one of ["2d","3d"]', tab = self.level)
            return [], [], 0, [], [], []

        # If this is just a placeholder (with empty data), skip
        if len(self.coverage) == 0:
            netsurf.utils.log._warn(f'Empty run placeholder. Skipping plot_curves.', tab = self.level)
            return [], [], 0, [], [], []

        # Pass the global metrics
        if colors is None:
            colors = self.hyperspace_global_config[f'{hue}_colors'] if f'{hue}_colors' in self.hyperspace_global_config._keys else None
        if xrange is None:
            xrange = self.hyperspace_global_config[x] if x in self.hyperspace_global_config._keys else None
        
        if metric is not None:
            # check if exists in hyperspace_global_config
            if metric not in self.hyperspace_global_config._keys:
                metric = None
        if metric is None:
            metric = self.hyperspace_global_config['loss'] if 'loss' in self.hyperspace_global_config._keys else 'mse'
        if yrange is None:
            yrange = self.hyperspace_global_config[metric] if metric in self.hyperspace_global_config._keys else None

        # Remove "root" from the structure config
        info_label_fields = ['benchmark', 'quantization', 'model', 'method', 'run']
        if info_label is None:
            info_label = {k: self.structure_config[k] for k in info_label_fields if k != 'root' and k in self.structure_config}
        
        # If ax exists, make sure to add this type to info_label
        if axs is not None:
            info_label[self.type] = self.name
            # make sure axs is a list
            if not isinstance(axs, list) and not isinstance(axs, np.ndarray):
                axs = [axs]

        fcn = {'2d': 'plot_2D_curves', '3d': 'plot_3D_volumes'}.get(type)
        fcn_kwargs = {'2d': {}, '3d': {"projection": "3d"}}.get(type)

        # If ax doesn't exists, create figure 
        if axs is None or standalone:
            if type == '2d':
                fig, axs = plt.subplots(1, 1, figsize = (5, 5), **fcn_kwargs)
            elif type == '3d':
                fig = plt.figure(figsize = (5, 5))
                axs = fig.add_subplot(111, **fcn_kwargs)
            axs = [axs]
        else:
            fig = axs[0].figure

        # Make sure the plotter.aucs is not empty 
        if hasattr(self.plotter[metric], 'aucs'):
            if len(self.plotter[metric].aucs) > 0 and (~self.results['loss'].isna()).any():
                _fig, _ax, _t, _lines = getattr(self.plotter[metric],fcn)(ax = axs[0], x = 'ber', y = 'mean', hue = 'protection', colors = colors,
                                                        xrange = xrange, yrange = yrange,
                                                        xlabel = xlabel, ylabel = ylabel,
                                                        info_label = info_label, 
                                                        standalone = standalone,
                                                        **kwargs)
            else:
                netsurf.utils.log._warn(f'Empty run placeholder. Skipping plot_curves.', tab = self.level)
                return [fig], [axs[0]], 0, [], [], []
        else:
            netsurf.utils.log._warn(f'Empty run placeholder. Skipping plot_curves.', tab = self.level)
            return [fig], [axs[0]], 0, [], [], []
        
        if not isinstance(_ax, list) and not isinstance(_ax, np.ndarray):
            _ax = [_ax]

        # Make sure ax has keywords applied 
        if not standalone:
            fig_out = [fig]
            plt.close(_fig)
        else:
            fig_out = [_fig]
            plt.close(fig)

        return fig_out, _ax, 1, [_t], [_lines], [deepcopy(info_label)]

    def plot_2D_curves(self, *args, **kwargs):
        return self.plot_curves('2d', *args, **kwargs)
    
    def plot_3D_volumes(self, *args, **kwargs):
        return self.plot_curves('3d', *args, **kwargs)

    """ Print tree """
    def tree(self, tab = 0):
        # Print this node
        # Add extra space according to the level
        tabs = '   ' * (tab + 1)
        p = self.dir if tab == 0 else f"{'…/'*tab}{os.path.basename(self.dir)}"
        ss = f'{tabs} ↪ Run "{self.name}" @ {p}'
        # Run has no children, show status
        icon = "✅" if self.status == 'completed' else "❌" if self.status == 'error' else "🔄" if self.is_running else "❓"
        ss += f'\n{tabs}    ↪ Status: {icon} {self.status}\n'
        return ss
    
    """ Representation showing the status of the run and some statitics """
    def __repr__(self, tab = 0):
        # Add extra space according to the level
        tabs = '   ' * (tab)
        p = self.dir if tab == 0 else f"{'…/'*tab}{os.path.basename(self.dir)}"
        ss = f'{tabs}Run "{self.name}" @ {p}'
        # Run has no children, show status
        icon = "✅" if self.status == 'completed' else "❌" if self.status == 'error' else "🔄" if self.is_running else "❓"
        ss += f'\n{tabs}    ↪ Status: {icon} {self.status}\n'
        return ss
    
    """ HTML representation """
    @property
    def html(self):
        map_level = self.hyperspace_global_config['map_level']
        ss = f'<h1>Object [{map_level[self.level]}Container] : "{self.name}"</h1>\n'
        ss += f'<p>Information about this object:</p>\n'
        
        # Info
        ss += f'\t<ul>\n'
        if True: # This if statement is just to prettify the code here, so we keep track of the indentation in the actual html
            
            # Directory with href link to directory
            ss += f'\t\t<li><b>Directory:</b> <a href="file:/{self.dir}">{self.dir}</a></li>\n'
            
            # Show total elapsed time in format days hh:mm:ss
            #ss += f'\t\t<li><b>Total Runtime:</b> {netsurf.utils.get_elapsed_time(self.structural_local_metrics["runtime"])}</li>\n'
            ss += f'\t\t<li><b>Accuracy range:</b> [{self.structure_config["accuracy"][1]*100:3.2f}%, {self.structure_config["accuracy"][0]*100:3.2f}%]</li>\n'
            
            # Show coverage statistics
            ss += f'\t\t<li><b>Coverage statistics:</b>\n'
            ss += f'\t\t\t<ul>\n'
            if True: # This if statement is just to prettify the code here, so we keep track of the indentation in the actual html
                for k, v in self.structure_config.items():
                    if k.startswith('num_'):
                        if k not in ['runtime', 'accuracy']:
                            ss += f'\t\t\t\t<li><b>{k.title().replace("num","Number of ").replace("_"," ")}:</b> {v}</li>\n'
            ss += f'\t\t\t</ul>\n' 
            ss += f'\t\t</li>\n' # Close coverage statistics

            # Add the TMR range along with their colors. Their colors are specified in matplotlib color names, so first we need to get 
            # the equivalent RGBA values. 
            protection = self.structure_config['protection']
            cols = self.hyperspace_global_config['protection_colors']
            ss += f'\t\t<li><b>TMR:</b>\n'
            ss += f'\t\t<ul>\n'
            for i, t in enumerate(protection):
                if True: # This if statement is just to prettify the code here, so we keep track of the indentation in the actual html
                    # transform color from matplotlib name to rgba 
                    if t in cols:
                        rgba = matplotlib.colors.to_rgba(cols[t])
                        # convert to str 
                        rgba = f'rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]})'

                        ss += f'\t\t\t<li>\n'
                        ss += f'\t\t\t\t<span style="margin-left: 5px; vertical-align: middle; width: 40px; display: inline-block;">{100*t:05.2f}%</span>\n'
                        ss += f'\t\t\t\t<span style="background-color: {rgba}; border: 1px solid #505050; width: 40px; height: 10px; display: inline-block; vertical-align: middle;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>\n'
                        # QTextBrowser doesn't handle divs and styles properly, so the "width" tag isn't working here :/ 
                        #ss += f'\t\t\t\t<div style="background-color: {rgba}; border: 1px solid #505050; width: 40px; height: 10px; display: inline-block; vertical-align: middle;"></div>\n'
                        ss += f'\t\t\t</li>\n'
            ss += f'\t\t</ul>\n' 
            ss += f'\t\t</li>\n' # Close TMR

        ss += f'\t</ul>\n' # Close info
        
        # Extra info
        ss += f'<p>You can add additional descriptive text here if needed.</p>\n'
        return ss
    


""" Create a dict with the bucket classes """
BUCKET_CLASSES = {'Root': RootContainer, 'Benchmark': BenchmarkContainer, 
                  'Quantization': QuantizationContainer, 'Model': ModelContainer, 
                  'Method': MethodContainer, 
                  'Experiment': ExperimentWrapper}
                  #'Run': RunContainer}

""" create a function as an entry point to the recursive function """
def create_bucket(dir, name, level = 0, verbose = True, pbar = (lambda val, text: (val, text)) ,**kwargs):

    # Initialize the hyperspace global config
    hyperspace_global_config = HyperspaceConfig('global', verbose = verbose, **kwargs)

    # Create a progress tracker that allows us to keep track of the progress and update the progress bar in the GUI
    custom_pbar = RecursiveProgressTracker(pbar, offset = 0.0, factor = 100)
    
    # This will be a Root container
    b = BUCKET_CLASSES.get('Root',Bucket)(dir, name, level = level, verbose = verbose, 
                                          pbar = custom_pbar, 
                                          hyperspace_global_config = hyperspace_global_config, 
                                          **kwargs)

    # Now that we have all the global metrics, we can use them to color the plots, etc.
    protection_colors = netsurf.utils.plot.get_unique_colors(b.hyperspace_global_config['protection'])
    ber_colors = netsurf.utils.plot.get_unique_colors(b.hyperspace_global_config['ber'])
    method_colors = netsurf.utils.plot.get_unique_colors(b.hyperspace_global_config['method'])

    # Add this to the global metrics
    b.hyperspace_global_config['protection_colors'] = protection_colors
    b.hyperspace_global_config['ber_colors'] = ber_colors
    b.hyperspace_global_config['method_colors'] = method_colors

    # We are done with the initial exploration. Now we need to do another forward propagation
    # of the global metrics. This seems redundant, but think about it: we went from the leaves of
    # the tree to the root. At the root now we know the global metrics for all the leaves, but the
    # leaves have no idea about the global metrics of the root. So we need to go back down and 
    # propagate this information.
    b.propagate_global_metrics()

    # Get results 
    r = b.get_results(pbar = custom_pbar)

    # finally, make sure to get the coverage
    c = b.get_coverage()

    # Create the coverage pies 
    root_pie = netsurf.utils.plot.CoveragePie(b, verbose = False)
    # Set this pie 
    b.coverage_pie = root_pie
    # Now propagate the coverage pies down
    b.propagate_coverage_pie()

    return b