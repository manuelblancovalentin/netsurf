""" Basic modules """
import os 
import sys

""" Typing """
from typing import Iterable, Union


""" Regex """
import re 

import datetime
import json
import time
import csv

""" glob """
from glob import glob

""" Pandas """
import pandas as pd

""" Numpy """
import numpy as np

""" Matplotlib """
import matplotlib.pyplot as plt

""" Keras """
#import keras
#[@max]
from tensorflow import keras 
import tensorflow as tf

""" Custom netsurf utils """
import netsurf
from . import injection

""" Dynamic Table """
from DynamicTable import DynamicTable

""" Pergamos """
import pergamos as pg


""" Coverage class """
class Coverage:
    def __init__(self, obj: pd.DataFrame, groupby: Iterable = ['ber','protection']):
        self.obj = obj
        self.groupby = groupby
        self._update_coverage(self.groupby)
    
    def _update_coverage(self, groupby: Iterable = ['ber','protection']):
         # Get the number of combs that don't have "metric" as NaN for each (protection, rad) pair
        rg = self.obj.groupby(groupby)
        run_reps = rg['loss'].apply(lambda x: x.notna().sum())
        # reset index 
        run_reps = run_reps.reset_index()
        # Rename metric to run_reps
        run_reps = run_reps.rename(columns = {'loss': 'run_reps'})
        # Add total number of reps 
        run_reps['total_num_reps'] = self.obj.num_reps['num_reps']
        # Add coverage
        run_reps['coverage'] = run_reps['run_reps']/np.maximum(1,self.obj.num_reps['num_reps'])

        # Build the coverage table (cols are TMR, rows are BER, cells are missing reps)
        # Reshape the run_reps into a matrix of shape (len(protection), len(ber))
        try:
            # If shape is the same, just use reshape
            if len(run_reps['run_reps']) == (len(self.obj.protection)*len(self.obj.ber)):
                T = run_reps['run_reps'].values.reshape(len(self.obj.protection), len(self.obj.ber))
            else:
                # If not, we need to build the matrix manually
                T = np.zeros((len(self.obj.protection), len(self.obj.ber)))
                for _, row in run_reps.iterrows():
                    T[self.obj.protection.index(row['protection']), self.obj.ber.index(row['ber'])] = row['run_reps']
            # Get the coverage number (average)
            coverage = T.sum().sum()/(np.maximum(1,self.obj.num_reps['num_reps'].sum()))
        except:
            T = np.zeros((len(self.obj.protection), len(self.obj.ber)))
            coverage = 0.0
            # stop here

        self.coverage_df = run_reps
        self.coverage = coverage
        self.coverage_table = T

        
def get_num_reps_crossvalidation(total_num_variables, ber, protection, factor = 1):
    # get the number of bits we will be protecting 
    # (remember that pruned weights are not considered)
    num_bits_protected = np.floor(total_num_variables*protection).astype(int)
    # Num bits susceptible
    num_bits_susceptible = total_num_variables - num_bits_protected

    # Now, let's see how many bits we will flip per attack (ber)
    num_flips_per_attack = np.round(num_bits_susceptible*ber).astype(int)

    # Now compute how many reps we need to fill all the space
    num_reps = factor*(num_bits_susceptible + num_flips_per_attack - 1)//num_flips_per_attack
    return num_reps, num_flips_per_attack


""" Result class (extend pandas dataframe) """
class ResultSpace(pd.DataFrame):
    _metadata = ['protection', 'ber', 'num_reps', 'coverage', 'coverage_table', 'coverage_df', '_loss', '_metrics', 'stats']
    COLUMNS = ['protection', 'ber', 'true_ber', 'loss', 'elapsed_time', 'datetime', 'dataset', 'model_name', 'ranking_method', 'experiment_hash']
    def __init__(self, loss_name, metrics,
                 protection: Union[Iterable[float], np.ndarray], 
                ber: Union[Iterable[float], np.ndarray], 
                num_reps: int, 
                total_num_params: int, 
                data: pd.DataFrame = None,
                **kwargs):

        # Initialize the columns 
        if 'columns' in kwargs:
            columns = kwargs.pop('columns')
        else:
            columns = self.COLUMNS

            # Modify the columns to include the metrics, after the loss
            loss_index = columns.index('loss')
            columns = columns[:loss_index+1] + metrics + columns[loss_index+1:]

        # We need to make sure that we have a column with the loss itself (this is, that the column name is <loss_name>)
        if loss_name not in columns:
            columns.insert(columns.index('loss')+1, loss_name)
            

        init = False 
        if data is None:
            init = True 
        else:
            if data.empty:
                init = True

        # Create the Cartesian product of protection, ber, and repetitions
        _num_reps = None
        if init:
            extra_cols = {k: None for k in columns if k not in ['protection', 'ber']}
            data = []
            _num_reps = []
            # Initialize with all combinations filled with NaN
            for p in protection:
                for b in ber:
                    if num_reps == -1:
                        subreps, _ = get_num_reps_crossvalidation(total_num_params, b, p)
                        
                    else:
                        subreps = int(1.0*num_reps)
                    data.extend([dict(protection = p, ber = b, **extra_cols)]*subreps)
                    _num_reps.append(dict(protection =p, ber = b, num_reps = subreps))
            # Turn _num_reps into a DataFrame
            _num_reps = pd.DataFrame(_num_reps)

        else:
            extra_cols = {k: None for k in columns if k not in ['protection', 'ber'] and k not in data.columns}
            
            # Remove duplicated columns where one is all NaN
            data = self._remove_duplicate_nan_columns(data)

            _num_reps = []
            # Initialize with all combinations filled with NaN
            for p in protection:
                for b in ber:
                    if num_reps == -1:
                        subreps, _ = get_num_reps_crossvalidation(total_num_params, b, p)
                        
                    else:
                        subreps = int(1.0*num_reps)
                    _num_reps.append(dict(protection =p, ber = b, num_reps = subreps))
            # Turn _num_reps into a DataFrame
            _num_reps = pd.DataFrame(_num_reps)

        # If dataframe is not None, we can initialize the object
        super().__init__(data, columns = columns)

        # Now set num_reps in place 
        self.num_reps = _num_reps

        # Set the protection, ber and num_reps
        self.protection = list(protection)
        self.ber = list(ber)

        # We also need to ensure we don't have any combinations we DON'T need (outside of our range)
        self._discard_unnecessary_combinations()

        # If data is not None, make sure we fill the missing (protection,ber) combs with NaN
        # Ensure all (protection, ber, num_reps) combinations exist
        self._fill_missing_combinations(extra_cols)
        
        # Init coverage obj
        self.coverage = Coverage(self, **kwargs)

        # Let's get the stats for the results 
        # Get the metric field 
        self._loss = loss_name
        self._metrics = metrics
        if 'loss' in self.columns:
            if (~self['loss'].isna()).any():
                self._loss = self['loss'].mode()[0]
        # Init stats 
        self.stats = pd.DataFrame([], columns = ['mean', 'std'])
        self._get_stats()

    def _remove_duplicate_nan_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicated columns that have the same name, keeping the one with valid data.
        """
        # Get all column names (including duplicates)
        col_names = data.columns.tolist()

        # Iterate through unique column names
        for col in set(col_names):
            # Find all positions (indexes) where this column name occurs
            col_indices = [i for i, name in enumerate(col_names) if name == col]

            # If the column is duplicated
            if len(col_indices) > 1:
                # Extract all duplicated versions of the column
                col_variants = data.iloc[:, col_indices]

                # Identify the column with the most non-NaN values
                non_nan_counts = col_variants.notna().sum().values
                keep_idx = col_indices[np.argmax(non_nan_counts)]  # Index to keep

                # Drop all other versions (columns) except the one to keep
                drop_positions = [idx for idx in col_indices if idx != keep_idx]
                data = data.iloc[:, [i for i in range(data.shape[1]) if i not in drop_positions]]

        return data

    def _fill_missing_combinations(self, extra_cols):
        # Generate all expected combinations of (protection, ber, rep)
        full_combinations = pd.concat(
            [
                pd.DataFrame({
                    'protection': [row['protection']] * int(row['num_reps']),
                    'ber': [row['ber']] * int(row['num_reps']),
                    'rep': list(range(int(row['num_reps'])))
                })
                for _, row in self.num_reps.iterrows()
            ],
            ignore_index=True
        )

        # Convert protection and ber to float64 to avoid dtype mismatches
        full_combinations = full_combinations.astype({'protection': 'float64', 'ber': 'float64'})

        # Add a 'rep' column to self if it doesn't exist
        if 'rep' not in self.columns:
            self['rep'] = self.groupby(['protection', 'ber']).cumcount()

        # Convert self DataFrame to ensure 'protection' and 'ber' are also float64
        self = self.astype({'protection': 'float64', 'ber': 'float64'})

        # Identify missing combinations
        current_combinations = self[['protection', 'ber', 'rep']]
        missing_combinations = pd.merge(
            full_combinations,
            current_combinations,
            on=['protection', 'ber', 'rep'],
            how='left',
            indicator=True
        ).query('_merge == "left_only"').drop(columns=['_merge'])

        # If there are missing combinations, append them with NaN in extra columns
        if not missing_combinations.empty:
            for col in extra_cols:
                missing_combinations[col] = np.nan
            updated_df = pd.concat([self, missing_combinations], ignore_index=True)
            
            # In-place update
            self._update_inplace(updated_df)

    def _discard_unnecessary_combinations(self):
        # Identify combinations outside of our self.protection or self.ber range 
        mask = (
            ~self['protection'].isin(self.protection) |
            ~self['ber'].isin(self.ber)
        )

        # Drop unnecessary combinations
        if mask.any():
            updated_df = self[~mask].copy()
            self._update_inplace(updated_df)

    def update(self, data, **kwargs):
        # Update the data entries 
        # Ensure required columns are present
        required_cols = {'protection', 'ber'}
        if not required_cols.issubset(data.columns):
            missing = required_cols - set(data.columns)
            raise ValueError(f"Update DataFrame is missing columns: {missing}")
        
        # Collect rows to append 
        rows_to_append = []

        # Iterate through the update DataFrame
        for _, row in data.iterrows():
            p, b = row['protection'], row['ber']

            # Find the first available NaN slot for this (protection, ber) pair
            mask = (
                (self['protection'] == p) &
                (self['ber'] == b) & 
                (self['loss'].isna()) 
            )

            if mask.any():
                # Update the first available slot
                first_available_index = self[mask].index[0]
                self.loc[first_available_index, row.index] = row
            else:
                # If no available slot, append 
                # Collect the row for batch appending
                rows_to_append.append(row)
        
        # Batch append all new rows at once (if any)
        if rows_to_append:
            new_rows_df = pd.DataFrame(rows_to_append)
            # Ensure index is reset before concatenation
            new_rows_df.reset_index(drop=True, inplace=True)
            updated_df = pd.concat([self, new_rows_df], ignore_index=True)
            self._update_inplace(updated_df)
        
        # _discard_unnecessary_combinations
        self._discard_unnecessary_combinations()

        # Update coverage
        self._update_coverage(**kwargs)

    def _update_inplace(self, new_df: pd.DataFrame):
        """
        Safer in-place update while preserving metadata.
        """
        # Clear and replace all columns directly
        self.__dict__.update(new_df.__dict__)

    def _update_coverage(self, groupby = ['protection', 'ber']):
        # Create Coverage object 
        self.coverage._update_coverage(groupby = groupby)

    def get_coverage(self):
        return self.coverage.coverage_df, self.coverage.coverage, self.coverage.coverage_table

    def coverage_to_csv(self, filename):
        
        if self.num_reps is None:
            self.coverage.coverage_table.to_csv(filename, index = False)
        else:
            # Add num_reps to beginning of file 
            comments = ['# (protection, ber): num_reps']
            comments.extend([f"#\t({row['protection']}, {row['ber']}): {int(row['num_reps'])}" for _, row in self.num_reps.iterrows()])

            comments.extend(['# total number of reps: ' + str(self.num_reps['num_reps'].sum()),
                        '# total number of combinations: ' + str(len(self.protection)*len(self.ber)),
                        '# total number of reps run: ' + str(self.coverage.coverage_table.sum().sum()),
                        '# coverage: ' + str(100*self.coverage.coverage) + '%'])

            # Convert in a table with y-axis as protection and x-axis as ber, values should be the run_reps
            t = {}
            for p in self.protection:
                t[p] = {}
                for b in self.ber:
                    subt = self.coverage.coverage_df[(self.coverage.coverage_df['protection'] == p) & (self.coverage.coverage_df['ber'] == b)]
                    if subt.empty:
                        t[p][b] = 0
                    else:
                        t[p][b] = subt['run_reps'].iloc[0]
            
            t = pd.DataFrame(t).T

            header = [""] + list(t.columns)
            # Loop 
            with open(filename, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                for c in comments:
                    csv_writer.writerow([c])
                csv_writer.writerow(header)
                for row in t.iterrows():
                    csv_writer.writerow([row[0]] + list(row[1].values))

    def _get_stats(self):
        # Get the mean and std for each combination of (protection, ber) for each metric 
        # only if there's data 
        # check if there's anything else than NaN
        m = self._loss
        if self['loss'].notna().any():
            if m in self.columns:
                try:
                    # Get the mean and std for each combination of (protection, ber) for each metric 
                    mean = self.groupby(['protection', 'ber'])[m].mean()
                    std = self.groupby(['protection', 'ber'])[m].std()
                    # Add to global_metrics
                    self.stats['mean'] = mean
                    self.stats['std'] = std
                except Exception as e:
                    print(e)
                    pass
        

""" Experiment class """
class Experiment:
    def __init__(self, 
                 benchmark: 'Benchmark', 
                 ranking: 'Ranking', 
                 name = None, path = None, 
                 num_reps = 10, ber_range = np.arange(0.005, 0.055, step=0.005), protection_range = np.arange(0.0, 1.0, step = 0.1), 
                 **kwargs):

        # Set ranker object
        self.ranking = ranking

        # Set the experiment simulation vars
        self.ber_range = np.round(ber_range, 5)
        self.protection_range = protection_range
        self.num_reps = num_reps

        # Now let's compare this configuration hash with the existing ones (if any), and get the next 
        # available name for this experiment config dir (e.g., config1, config2, etc)
        self.name, self.alias, self.path = self.get_experiment_name(benchmark, name, path)

        # Get the loss and metrics
        (loss_name, formatted_loss_name, fmt_loss_name), (metric_names, metrics, fmt_metrics, _) = self.get_loss_and_metrics(benchmark)

        # Get total number of params of our model (in case we pass num_reps = -1)
        total_num_params = benchmark.model.count_trainable_parameters() - benchmark.model.count_pruned_parameters()

        # If we need to reload the ranking, we do it here
        self.results, self.global_metrics = self._reload_data(loss_name, metric_names, self.name, self.path, total_num_params)

        # Init progress table to None
        self.progress_table = None

    # Translation table from loss/metric to column name
    def _translate(self, name):
        name1 = {'mean_squared_error': 'Mean Squared Error',
            'mean_absolute_error': 'Mean Absolute Error',
            'accuracy': 'Accuracy (%)',
            'categorical_accuracy': 'Accuracy (%)',
            'categorical_crossentropy': 'Cat. XEntropy',
            'binary_crossentropy': 'Bin. XEntropy',
            'sparse_categorical_crossentropy': 'Sparse Cat. XEntropy',
            'kullback_leibler_divergence': 'KL Div',
            'cosine_similarity': 'Cosine Sim',
            'mean_absolute_percentage_error': 'Mean Abs. % Error',
            'mean_squared_logarithmic_error': 'Mean Squared Log Error',
            'poisson': 'Poisson',
            'pearsoncorrelation': 'Correlation (%)',
            'r2score': 'R2 Coefficient (%)',
            'kl_divergence': 'KL Div',
            'emd': 'EMD',
            'wd': 'WD'
            }.get(name.lower(), name)
        
        # Equivalent formatting and format fcn
        fmt_loss_name = {'mean_squared_error': '{:.4f}',
            'mean_absolute_error':'{:.4f}',
            'accuracy': '{:3.2%}',
            'categorical_accuracy': '{:3.2%}',
            'categorical_crossentropy': '{:.4f}',
            'binary_crossentropy': '{:.4f}',
            'sparse_categorical_crossentropy': '{:.4f}',
            'kullback_leibler_divergence': '{:.4f}',
            'cosine_similarity': '{:.4f}',
            'mean_absolute_percentage_error': '{:.4f}',
            'mean_squared_logarithmic_error': '{:.4f}',
            'poisson': '{:.4f}',
            'pearsoncorrelation': '{:3.2%}',
            'r2score': '{:3.2%}',
            'kl_divergence': '{:.4f}',
            'emd': '{:.4f}',
            'wd': '{:.4f}'
            }.get(name.lower(), name)
        return name1, fmt_loss_name

    def get_loss_and_metrics(self, benchmark):
        
        """ Get the loss and metric names """
        loss_name = benchmark.loss_name
        formatted_loss_name, fmt_loss_name  = self._translate(loss_name)
        metric_names = []
        metrics = []
        fmt_metrics = []
        metric_fcns = {}
        # Add the metrics 
        for metric in benchmark.metrics:
            if hasattr(metric, 'name'):
                metric_name = metric.name
                # Do not add if this metric is the loss
                if metric_name.lower() == loss_name.lower():
                    continue
                # Translate
                formatted_metric_name, fmt_metric_name = self._translate(metric_name)
                metrics.append(formatted_metric_name)
                fmt_metrics.append(fmt_metric_name)
                metric_names.append(metric_name)
                metric_fcns[metric_name] = metric
        return (loss_name, formatted_loss_name, fmt_loss_name), (metric_names, metrics, fmt_metrics, metric_fcns)
    
    # save to file 
    def save(self, overwrite = False):
        filepath = os.path.join(self.path, f'{self.name}.netsurf.exp')
        if os.path.isfile(filepath) and not overwrite:
            netsurf.utils.log._warn(f'Experiment file {filepath} already exists. Skipping. If you want to overwrite, set overwrite = True')
            return
        msg = netsurf.utils.save_object(self, filepath, meta_attributes = {'class': 'netsurf.Experiment', 'hash': self._config_hash.encode()})
        netsurf.utils.log._custom('EXP',msg)


    def get_experiment_name(self, benchmark, name, parent_dir):
        # Get all directories in parent_dir
        dirs = []
        config_tag = 'config1'
        if netsurf.utils.is_valid_directory(parent_dir):
            if name is None:
                dirs = netsurf.WeightRanker.find_config_dirs(parent_dir)
                if len(dirs) > 0:
                    # Get the last config dir
                    config_tag = dirs[0]
            else:
                dirs = [name]
        else:
            # Create parent's path for this experiment
            # Config dir can be obtained from ranking 
            if self.ranking.filepath is not None:
                parent_dir = os.path.dirname(self.ranking.filepath)
                config_tag = os.path.basename(os.path.dirname(self.ranking.filepath))
            else:
                parent_dir = os.path.join(benchmark.experiments_dir, self.ranking.method, config_tag)

        # Set test name 
        exp_alias = f'{self.ranking.alias}_{benchmark.quantization._scheme_str}_{config_tag}'

        exp_name = config_tag
        exp_path = parent_dir
        # Log 
        netsurf.utils.log._info(f'Creating new experiment directory with hash {exp_path}')

        # Create the directory
        os.makedirs(exp_path, exist_ok = True)

        # We need to create the metadata now 
        metadata = {'level': 'method', 
                    'name': self.ranking.alias, 
                    'group': self.ranking.method,
                    'creation_date': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'creation_user': os.getlogin(),
                    'creation_host': os.uname().nodename,
                    'config': {}}

        # metadata filepath 
        metadata_filepath = os.path.join(parent_dir, '.metadata.netsurf')
        if not os.path.isfile(metadata_filepath):
            # Save metadata
            netsurf.utils.log._info(f'Saving method metadata to file {metadata_filepath}')
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Now, we need to do the same with the experiment metadata 
        metadata_exp = {'level': 'experiment',
                        'name': exp_name,
                        'creation_date': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                        'creation_user': os.getlogin(),
                        'creation_host': os.uname().nodename,
                        'config': {}} #self.ranker.config}

        # metadata filepath
        metadata_exp_filepath = os.path.join(exp_path, '.metadata.netsurf')
        if not os.path.isfile(metadata_exp_filepath):
            # Save metadata
            netsurf.utils.log._info(f'Saving experiment metadata to file {metadata_exp_filepath}')
            with open(metadata_exp_filepath, 'w') as f:
                json.dump(metadata_exp, f, indent=2)

        return exp_name, exp_alias, exp_path

    def _reload_data(self, loss_name, metric_names, exp_name, exp_path, total_num_params, reload_ranking = True):
        # There are 4 things we keep in an experiment folder:
        # 1) results in csv
        # 2) metrics regarding the experiment (metrics.json)

        # Get protection and ber ranges 
        columns = ['protection', 'ber', 'true_ber', 'loss', loss_name] 
        # Append metrics 
        columns.extend(metric_names)
        # Append other columns
        columns.extend(['elapsed_time', 'datetime', 'dataset', 'model_name', 'ranking_method', 'experiment_hash'])
        
        # Check if results exist 
        results_file = os.path.join(exp_path, 'results.csv')
        if netsurf.utils.path_exists(results_file):
            results = pd.read_csv(results_file)
            # Convert to Result object
            results = ResultSpace(loss_name, metric_names, self.protection_range, self.ber_range, self.num_reps, total_num_params, data=results, columns=columns)
            # Log 
            netsurf.utils.log._info(f'Loaded results from {results_file}')
        else:
            # Create empty results
            results = ResultSpace(loss_name, metric_names, self.protection_range, self.ber_range, self.num_reps, total_num_params, data = None, columns=columns)
            # Log 
            netsurf.utils.log._info(f'No results found for experiment {exp_name}')

        # Global metrics (pick)
        metrics_file = os.path.join(exp_path, 'metrics.json')
        if netsurf.utils.path_exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            # Log 
            netsurf.utils.log._info(f'Loaded metrics from {metrics_file}')
        else:
            metrics = {}
            # Log 
            netsurf.utils.log._info(f'No metrics found for experiment {exp_name}')
        return results, metrics

    """ Run experiment"""
    def run_experiment(self, benchmark, XY = None,
                       batch_size = 10000, 
                       interlayer_mse = False,
                       rerun = False, verbose = True, 
                       **kwargs):
        
        # Start timer
        start_time = time.time()

        # Results name
        results_file = os.path.join(self.path, 'results.csv')
        # coverage file 
        coverage_file = os.path.join(self.path, 'coverage.csv')

        # Check which combinations we have left 
        coverage = 0.0
        results = self.results
        # Get coverage 
        coverage_df, coverage, coverage_table = results.get_coverage()

        (loss_name, formatted_loss_name, fmt_loss_name), (metric_names, metrics, fmt_metrics, metric_fcns) = self.get_loss_and_metrics(benchmark)

        if coverage <= 0.0:
            if rerun: 
                results = ResultSpace(loss_name, metric_names, self.protection_range, self.ber_range, self.num_reps, data = None)
                coverage_df, coverage, coverage_table = results.get_coverage()

            # NO!!!! --> We need to run all combinations, so initialize file <-- NO!!!!
            # We might have coverage 0.0 but this only means this coverage is FOR THE CURRENT
            # BER AND TMR RANGE. We might have other BER and TMR values in the file. NEVER
            # OVERWRITE THE RESULTS_FILE, IF IT EXISTS!!!
        
        if not os.path.exists(results_file):
            with open(results_file, 'w') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(results.columns)

        # Save coverage table 
        results.coverage_to_csv(coverage_file)

        # Get combs from coverage_df (combs are the ones that 
        # have not been run yet, and thus have NaN values in metric)
        g = coverage_df[coverage_df['coverage'] < 1.0][['protection','ber','run_reps']]
        combs = {tuple(r[['protection','ber']].to_list()): int(max(0,r['run_reps'])) for _, r in g.iterrows()}
        # If we have no combinations left, we can return the results
        if len(combs) == 0:
            # Log 
            netsurf.utils.log._info('Experiment is finished. Nothing else to run.')
            return results

        # Log
        netsurf.utils.log._info(f'Experiment {self.name} started. Coverage: {100*coverage:3.1f}%')

        model = benchmark.model

        # # Get data 
        if XY is None:
            dataset = benchmark.dataset.dataset['train']

            # Apply pre-processing to data, if needed 
            if hasattr(benchmark.model, 'preprocess_input'):
                s0 = time.time()
                netsurf.utils.log._info(f'Preprocessing input data...', end = '')
                dset = benchmark.model.preprocess_input(dataset[0])
                dataset = (dset, dataset[1])
                print(f'done in {time.time() - s0:.2f} seconds')

            if hasattr(benchmark.model, 'preprocess_output'):
                s0 = time.time()
                netsurf.utils.log._info(f'Preprocessing output data...', end = '')
                dset = benchmark.model.preprocess_output(dataset[1])
                dataset = (dataset[0], dset)
                print(f'done in {time.time() - s0:.2f} seconds')


            if isinstance(dataset, tuple):
                num_samples = len(dataset[0])
                X = dataset[0]
                Y = dataset[1]

                # MAke sure they are numpy arrays
                if not isinstance(X, np.ndarray):
                    X = np.array(X)
                if not isinstance(Y, np.ndarray):
                    Y = np.array(Y)

            elif isinstance(dataset, keras.preprocessing.image.DirectoryIterator):
                num_samples = dataset.n
                # Concatenate a couple of batches
                X, Y = dataset.next()
                for i in range(2):
                    Xb, Yb = dataset.next()
                    X = np.concatenate((X,Xb), axis = 0)
                    Y = np.concatenate((Y,Yb), axis = 0)
            elif isinstance(dataset, tf.data.Dataset):
                # For some reason the batch size I set in the dataset is not working, so let's get it manually by iterating a couple of times
                exp_batch_size = 0
                # concatenate some batches 
                X, Y = None, None
                max_num_samples = len(dataset)
                for it, (Xb, Yb) in enumerate(dataset):
                    if X is None or Y is None:
                        X, Y = Xb, Yb
                    else:
                        X = np.concatenate((X,Xb), axis = 0)
                        Y = np.concatenate((Y,Yb), axis = 0)
                    exp_batch_size = Xb.shape[0]
                    max_num_samples = len(dataset)*exp_batch_size
                    if len(X) >= batch_size:
                        break
                num_samples = len(X)
                num_iters = np.ceil(num_samples/batch_size).astype(int)
                X = X[:batch_size*num_iters]
                Y = Y[:batch_size*num_iters]
            else:
                raise ValueError('Dataset not recognized')
        else:
            # Get data
            X, Y = XY

            # Get number of samples
            num_samples = XY[0].shape[0]
        
        # if interlayer_mse also get interlayer activations
        interactivations = None 
        if interlayer_mse:
            pass

        # Calculate max num of iters (per batch_size)
        if batch_size:
            num_iters = np.ceil(num_samples/batch_size).astype(int)
        else:
            num_iters = 1

        # protection,ber,true_ber,metric,accuracy,mse,elapsed_time,datetime,dataset,model_name,ranking_method,experiment_hash
        T = []

        # There are some parameters that we add on every row, but that are the same for all rows, so let's
        # initalize them here 
        dataset_name = benchmark.dataset.__class__.__name__
        model_name = benchmark.model_name
        ranking_method = self.ranking.method
        #experiment_hash = self.ranker.config_hash
        experiment_hash = "null"

        common_args = [dataset_name, model_name, ranking_method, experiment_hash]

        # Reopen file in append mode 
        csvfile = open(results_file, 'a', newline='')
        csv_writer = csv.writer(csvfile)

        # Open file to printout the table to a txt file. If the file already exists, make sure to get the next 
        # available name (appending .0, .1, etc.)
        table_printout_filename = os.path.join(self.path, 'injection_progress.txt')
        table_printout_filename = netsurf.utils.io.get_latest_suffix(table_printout_filename, glob(table_printout_filename+'*'), 
                                         next = True, divider = '_', 
                                         return_index = False, 
                                         verbose = True, 
                                         next_if_empty = True)[0]
        # Undo escape \.
        table_printout_filename = table_printout_filename.replace('\\.','.')

        # Get quantization
        quantization = benchmark.quantization
        
        """ Initialize dynamic table """
        header = ['Iteration', 'Method', 'Protection(%)', 'Bit Error Rate(%)', 'Elapsed time', 'Remain. time']
        formatters = {'Iteration':'{:10s}', 'Method': '{:30s}', 'Protection(%)':'%$', 'Bit Error Rate(%)': '%$', 
                      'Elapsed time': '{:8s}', 'Remain. time': '{:8s}'}
        # Now add the loss
        header += [formatted_loss_name]
        formatters[formatted_loss_name] = '{:40s}'
        # Add the metrics 
        for formatted_metric_name, fmt_metric_name in zip(metrics, fmt_metrics):
            # Add to header and formatters
            header += [formatted_metric_name]
            formatters[formatted_metric_name] = '{:40s}'

        # Finally add the true BER and coverage
        header += ['True BER(%)', 'Coverage(%)']
        formatters['True BER(%)'] = '{:2.5%}'
        formatters['Coverage(%)'] = '{:3.1%}'

        num_tmrs = len(self.protection_range)
        num_rads = len(self.ber_range)

        # time per rep
        time_per_rep = [] if 'time_per_rep' not in self.global_metrics else self.global_metrics['time_per_rep']
        if 'elapsed_time' not in self.global_metrics:
            self.global_metrics['elapsed_time'] = []

        # Now create the ErrorInjector wrapper for this value of protection
        injector_time_start = time.time()
        injector = injection.ErrorInjector(model, self.ranking, quantization, 
                                            ber_range = self.ber_range, 
                                            protection_range = self.protection_range,
                                            verbose = False)
        
        build_time_start = time.time()
        # Build injection models 
        attack_container = injector.build_injection_models(combs, self.num_reps, verbose = False)
        
        if verbose:
            netsurf.utils.log._info(f'Injector created in {build_time_start - injector_time_start:.2f} seconds')
            netsurf.utils.log._info(f'Injector models built in {time.time() - build_time_start:.2f} seconds')

        # Hold a second before printing the table (for some reason it's printing the table before the 
        # log above)
        sys.stdout.flush()
        time.sleep(1)

        # Get total_num_reps
        total_num_reps = attack_container.stats['num_attacks'].sum()

        if total_num_reps > 0:
            """ Build table """
            #formatters = {'Epoch':'{:03d}', 'Type': '{:s}', 'Progress':'%$', 'loss_labels':'{:.3f}'}
            progress_table = DynamicTable(header, formatters)
            progress_table.print_header()
            # Print to output file 
            with open(table_printout_filename, 'a') as table_printout:
                table_printout.write(progress_table.header)
            
            """ Loop thru all cases """
            icase = 0
            for icomb, attack_scheme in enumerate(attack_container):
                # Get tmr, rep and rad from case 
                # itmr, tmr = case['protection_idx'], case['protection']
                # irad, rad = case['ber_idx'], case['ber']
                # irep = case['rep']
                # Get tmr, rad and rep from comb
                tmr, rad, true_ber = attack_scheme['protection'], attack_scheme['ber'], attack_scheme['true_ber']
                comb = (tmr, rad)

                # Get already run reps for this 
                already_run_reps = combs[(tmr, rad)]

                itmr = np.where(self.protection_range == tmr)[0][0]
                irad = np.where(self.ber_range == rad)[0][0]

                # Num reps for this attack
                num_reps = int(attack_scheme['num_attacks'])

                #already_run_reps = max(0,num_reps - missing_num_reps)
                missing_num_reps = max(0, num_reps - already_run_reps)

                # Get all coverage values from table, BUT this one 
                already_run_tmp = coverage_table.sum().sum()

                # Keep a running avg for loss and metrics 
                avg_loss = 0.0
                avg_metrics = {mn: 0.0 for mn in metric_names}
                std_loss = 0.0
                std_metrics = {mn: 0.0 for mn in metric_names}

                # loop thru reps
                for irep, attack in enumerate(attack_scheme):
                    # Get start time for this rep
                    rep_start_time = time.time()

                    # Do forward pass with injection of attack 
                    activations = injector(X, attack = attack, batch_size = batch_size, verbose = False)

                    # Evaluate with metrics, loss 
                    metrics_vals = injector.evaluate(Y, activations, metric_fcns = metric_fcns, verbose = False)
                    
                    # Calculate elapsed time and ETA
                    s0 = time.time()
                    elapsed_time = s0 - start_time
                    elapsed_time_hhmmss = netsurf.utils.seconds_to_hms(elapsed_time)
                    rep_elapsed_time = s0 - rep_start_time

                    # Calculate ETA
                    eta = elapsed_time * total_num_reps/(icase+1) 
                    eta_hhmmss = netsurf.utils.seconds_to_hms(eta)

                    # remaining time
                    remain_time = eta - elapsed_time
                    remain_time_hhmmss = netsurf.utils.seconds_to_hms(remain_time)

                    loss = metrics_vals.pop('loss')

                    # Update running avg and std
                    avg_loss = (avg_loss*irep + loss)/(irep+1)
                    std_loss = np.sqrt((std_loss**2*irep + (loss - avg_loss)**2)/(irep+1))

                    for mn in metric_names:
                        avg_metrics[mn] = (avg_metrics[mn]*irep + metrics_vals[mn])/(irep+1)
                        std_metrics[mn] = np.sqrt((std_metrics[mn]**2*irep + (metrics_vals[mn] - avg_metrics[mn])**2)/(irep+1))
                    
                    #fmt_loss_name, fmt_metric_name
                    """ Get updated values to be set into table """
                    vals = {'Iteration': f'{irep+already_run_reps}/{num_reps+already_run_reps-1}', 
                            'Method': self.ranking.method, 
                            'Protection(%)': (itmr+1)/num_tmrs, 
                            'Bit Error Rate(%)': (irad+1)/num_rads, 
                            'Elapsed time': elapsed_time_hhmmss, 
                            'Remain. time': remain_time_hhmmss, 
                            formatted_loss_name: fmt_loss_name.format(avg_loss) + ' ± ' + fmt_loss_name.format(std_loss),
                            **{mft: ffmt.format(avg_metrics[mn]) + ' ± ' + ffmt.format(std_metrics[mn]) for mn,mft,ffmt in zip(metric_names, metrics, fmt_metrics)},
                            'True BER(%)': true_ber,
                            'Coverage(%)': (icase/(total_num_reps-1))
                            }


                    """ Update and print line """
                    append = (irep == (num_reps - 1))
                    append |= (itmr+irad+irep == 0)
                    if (irep % 20) == 0 or append:
                        progress_table.update_line(vals, append = append, print = True)

                    # Get datetime
                    dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                    # Append to table
                    # protection,ber,true_ber,loss,**metrics,elapsed_time,datetime,dataset,model_name,ranking_method,experiment_hash,rep
                    row = [tmr, rad, true_ber, loss_name, loss] + list(metrics_vals.values()) + [rep_elapsed_time, dt] + common_args + [irep]
                    T.append(row)

                    # Append to csv file 
                    with open(results_file, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        # Append to csv file 
                        csv_writer.writerow(row)

                    if len(time_per_rep) > 0 :
                        time_per_rep.append(elapsed_time - time_per_rep[-1])
                    else:
                        time_per_rep.append(elapsed_time)
                    
                    # Update missing_tmp
                    already_run_tmp += 1

                    # Update icase
                    icase += 1
                
                # Printout 
                with open(table_printout_filename, 'a') as table_printout:
                    table_printout.write(progress_table.lines[-1]+'\n')

                # Update coverage table directly 
                combs[comb] += (num_reps - already_run_reps)
                i = coverage_df[(coverage_df['protection'] == tmr) & (coverage_df['ber'] == rad)].index[0]
                if combs[comb] == 0:
                    # We are done, avoid division by zero
                    coverage_df.loc[i, ['run_reps','coverage']] = [num_reps, 1.0]
                else:
                    coverage_df.loc[i, ['run_reps','coverage']] = [combs[comb], combs[comb]/num_reps]

                # Update coverage table too 
                coverage_table[itmr, irad] = combs[comb]

                # Save coverage table
                results.coverage.coverage_df = coverage_df
                results.coverage.coverage_table = coverage_table
                results.coverage.coverage = np.minimum(coverage_table, num_reps).sum().sum()/total_num_reps
                results.coverage_to_csv(coverage_file)

            df = pd.DataFrame(T, columns = results.columns)
            # merge results, keeping non-nan entries
            #self.results.set_index(['protection','ber'], inplace = True)
            #df.set_index(['protection','ber'], inplace = True)
            self.results = df.combine_first(self.results)
            
            # Print last line of table
            progress_table.print_sep()
            with open(table_printout_filename, 'a') as table_printout:
                table_printout.write(progress_table.bottom_border)
            
            # Set self.progress_table to this raw text 
            self.progress_table = open(table_printout_filename, 'r').read()

            # Add time per rep to global metrics
            self.global_metrics['elapsed_time'] += time_per_rep
            total_exp_time = self.results['elapsed_time'].sum()

            # Add to global metrics
            self.global_metrics['total_experiment_time'] = total_exp_time

            # Write global_metrics to file 
            filename_json = os.path.join(self.path, 'metrics.json')
            with open(filename_json, 'w') as f:
                json.dump(self.global_metrics, f, indent=2)
                
            print("--- %s seconds ---" % (s0 - start_time))


    """ Implement method for html (for pergamos) """
    def html(self):
       
        # Create a collapsible container for this 
        # First, we want a summary table with all the config parameters for this method 
        if hasattr(self.ranking, '_ICON'):
            cname = f'{self.ranking._ICON} {self.ranking.method}'
        else:
            cname = f'🔬 {self.ranking.method}'
        if self.__class__.__name__ != 'Experiment':
            cname = f'🔬 {self.__class__.__name__} (Experiment): {self.name}'

        g = pg.CollapsibleContainer(cname, layout = 'vertical')

        props = {"Experiment name": self.name,
            "🏆 Ranker": f'{self.ranking.method} ({self.ranking.__class__.__name__})',
            '🔗 Alias': self.alias,
            'Normalize': self.config['normalize']}

        if 'suffix' in self.config:
            props['Suffix'] = self.config['suffix']
        
        for kw in self.global_metrics:
            if self.global_metrics[kw] is not None and kw != 'elapsed_time':
                props[kw.replace('_',' ').capitalize()] = self.global_metrics[kw]
        
        # Add configs 
        for kw, kv in [ss.split('=') for ss in self.config['method_kws'].split(' ')]:
            if kw == 'ascending':
                kw = '🔺 Ascending'
            if kv in ['True', 'False']:
                kv = eval(kv)
                kv = '✅' if kv else '❌'
            props[kw.capitalize()] = kv

        props = pd.DataFrame([props]).T

        p = pg.Table.from_data(props)
        
        g.append(p)

        # Now let's add the ranking plot
        if self.ranking is not None:
            fig, axs = self.ranking.plot_ranking(axs = None, w = 300, show = False)
            # Create a container for the plot
            ct = pg.CollapsibleContainer('🏆 Ranking', layout = 'vertical')
            # Create plot img
            p = pg.Plot(fig)
            ct.append(p)
            # Append to group
            g.append(ct)
            # Close fig
            plt.close(fig)
            
        #     t = pg.Table.from_data(self.ranking)
        #     # Create a container for the table
        #     ct = pg.CollapsibleContainer('Ranking', layout = 'vertical')
        #     ct.append(t)
        #     # Append to group
        #     g.append(ct)
        
        # Now add the results
        
        # Plot 2D curve
        loss_name = self.results['loss'].mode()[0]
        # Find index of loss_name
        idx = self.results.columns.get_loc(loss_name)
        # Metrics is everything after loss_name and before "elapsed_time"
        metric_names = self.results.columns[idx+1:self.results.columns.get_loc('elapsed_time')]

        # number of axs will be loss + metrics
        n_axs = len(metric_names) + 1
        fig, axs = plt.subplots(n_axs, 1, figsize = (10, 6*n_axs), sharex = True)
        # And the 3d figure
        fig3d, axs3d = plt.subplots(n_axs, 1, figsize = (10, 6*n_axs), subplot_kw = {'projection': '3d'})

        # Plot loss
        self.plotter = netsurf.gui.plotter.ExperimentsPlotter(self.results, metric = loss_name)
        self.plotter.plot_2D_curves(x = 'ber', y = 'mean', hue = 'protection', style = 'protection', ax = axs[0], 
                                    standalone = False, xlabel = 'Bit Error Rate (%)', ylabel = f'Loss ({loss_name})')
        self.plotter.plot_3D_volumes(x = 'ber', y = 'mean', z = 'mean', ax = axs3d[0], 
                                        standalone = False, xlabel = 'Bit Error Rate (%)', ylabel = f'Loss ({loss_name})')
        # Loop thru metrics 
        for i, metric_name in enumerate(metric_names):
            # Get plotter 
            plotter = netsurf.gui.plotter.ExperimentsPlotter(self.results, metric = metric_name)
            # Plot 
            plotter.plot_2D_curves(x = 'ber', y = 'mean', hue = 'protection', style = 'protection', ax = axs[i+1],
                                    standalone = False, xlabel = 'Bit Error Rate (%)', ylabel = f'{metric_name}')
            # 3D
            plotter.plot_3D_volumes(x = 'ber', y = 'mean', z = 'mean', ax = axs3d[i+1], 
                                        standalone = False, 
                                        xlabel = 'Bit Error Rate (%)',
                                        zlabel = f'{metric_name}')

        # Create a container for the plot
        ct = pg.CollapsibleContainer('📈 2D Line', layout = 'vertical')
        # Create plot img 
        p = pg.Plot(fig)
        ct.append(p)
        # Append to group
        g.append(ct)
        # Close ax 
        plt.close(fig)

        # Create a container for the plot
        ct = pg.CollapsibleContainer('📊 3D Volume', layout = 'vertical')
        # Create plot img
        p = pg.Plot(fig3d)
        ct.append(p)
        # Append to group
        g.append(ct)
        # Close fig
        plt.close(fig3d)
        
        if self.results is not None:
            # t = pg.Table.from_data(self.results)
            # # Create a container for the table
            # ct = pg.CollapsibleContainer('Results', layout = 'vertical')
            # ct.append(t)
            # # Append to group
            # g.append(ct)

            # Plot the histogram of susceptibility
            fig, ax = plt.subplots(1,1, figsize = (10,6))

            # Compute histogram
            susceptibility = self.ranking['susceptibility']

            # Get mu, std
            mu, std = susceptibility.mean(), susceptibility.std()

            bins = np.linspace(mu - 2*std, mu + 2*std, 75)
            hist, bins = np.histogram(susceptibility, bins = bins)

            # Plot histogram
            ax.stairs(hist, bins, alpha = 0.2, color = 'black', lw = 1.5)
            ax.set_title('Susceptibility histogram')
            ax.set_yscale('log')

            # grids
            ax.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
            # Make sure grid is on
            ax.set_axisbelow(True)

            # Create container 
            ct = pg.CollapsibleContainer('Susceptibility Histogram', layout = 'vertical')
            # Create plot img
            p = pg.Plot(fig)
            ct.append(p)
            # Append to group
            g.append(ct)
            # Close fig
            plt.close(fig)
        
        # Also print the process table 
        if self.progress_table is not None:
            # Create container 
            ct = pg.CollapsibleContainer('🧾 Progress Table', layout = 'vertical')
            # Create plot img
            p = pg.Terminal(self.progress_table)
            ct.append(p)
            # Append to group
            g.append(ct)

        # Create 
        return g

    """ Implement repr method """
    def __repr__(self):
        msg = '\n'
        msg += f'+{"-"*80}+\n'
        msg += f'| METHOD: {self.ranking.method:70s} |\n'
        msg += f'| Experiment: {self.name:66s} |\n'
        msg += f'+{"-"*80}+\n'

        # Add config to msg
        # msg += f'| {"Config:":79s}|\n'
        # msg += f'+{"-"*80}+\n'
        # def print_dict(d, indent = 1):
        #     msg = ''
        #     for kw in d:
        #         if isinstance(d[kw], dict):
        #             msg += f'| {"    "*indent}{"-"*(79-indent*4)}+\n'
        #             N = 79 - len(kw) - 4*indent - 1
        #             msg += f'| {"    "*indent}{kw:s} {" "*N}|\n'
        #             msg += print_dict(d[kw], indent = indent + 1)
        #             msg += f'| {"    "*indent}{"-"*(79-indent*4)}+\n'
        #         else:
        #             N = 79 - (len(kw) + np.maximum(len(str(d[kw])),50) + 2) - 4*indent - 1
        #             msg += f'| {"    "*indent}{kw:s}: {str(d[kw]):50s} {" "*N}|\n'
        #     return msg
        # msg += print_dict(self.ranker.config)
        # msg += f'+{"-"*80}+\n'
        msg += '\n'

        return msg