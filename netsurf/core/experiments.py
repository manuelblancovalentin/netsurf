""" Basic modules """
import os 
import sys

""" Typing """
# Basic
from dataclasses import dataclass, field
from typing import Optional, List, Iterable, Union
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
#from matplotlib import cm
import matplotlib.ticker as mticker
from sklearn.metrics import auc as sklearn_auc
from scipy.interpolate import griddata

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
        self.name, self.path = self.get_experiment_name(benchmark, name, path)

        # Get the loss and metrics
        (loss_name, formatted_loss_name, fmt_loss_name), (metric_names, metrics, fmt_metrics, _) = self.get_loss_and_metrics(benchmark)

        # Get total number of params of our model (in case we pass num_reps = -1)
        total_num_params = benchmark.model.count_trainable_parameters() - benchmark.model.count_pruned_parameters()

        # If we need to reload the ranking, we do it here
        self.results, self.global_metrics = self._reload_data(loss_name, metric_names, self.name, self.path, total_num_params)

        # Init progress table to None
        self.progress_table = None


    def get_experiment_name(self, benchmark, name, parent_dir):
        
        # Exp alias
        if name is None:
            name = f'{benchmark.name}_{self.ranking.alias}'
        
        # Start creating metadatas
        if not netsurf.utils.is_valid_directory(parent_dir):
            netsurf.utils.log._info(f'Creating new experiment directory @ {parent_dir}')
            # Create the directory
            os.makedirs(parent_dir, exist_ok = True)

        # top dir:
        top_dir = benchmark.experiments_dir
        method_dir = os.path.join(top_dir, self.ranking.method)

        # We need to create the metadata now 
        metadata = {'level': 'method', 
                    'name': self.ranking.alias, 
                    'group': self.ranking.method,
                    'creation_date': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'creation_user': os.getlogin(),
                    'creation_host': os.uname().nodename,
                    'config': {}}

        # metadata filepath 
        metadata_filepath = os.path.join(method_dir, '.metadata.netsurf')
        if not os.path.isfile(metadata_filepath):
            # Save metadata
            netsurf.utils.log._info(f'Saving method metadata to file {metadata_filepath}')
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Now, we need to do the same with the experiment metadata 
        metadata_exp = {'level': 'experiment',
                        'name': name,
                        'creation_date': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                        'creation_user': os.getlogin(),
                        'creation_host': os.uname().nodename,
                        'config': self.ranking.config.to_dict()}

        # metadata filepath
        metadata_exp_filepath = os.path.join(parent_dir, '.metadata.netsurf')
        if not os.path.isfile(metadata_exp_filepath):
            # Save metadata
            netsurf.utils.log._info(f'Saving experiment metadata to file {metadata_exp_filepath}')
            with open(metadata_exp_filepath, 'w') as f:
                json.dump(metadata_exp, f, indent=2)

        return name, parent_dir



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
        #msg = netsurf.utils.save_object(self, filepath, meta_attributes = {'class': 'netsurf.Experiment', 'hash': self._config_hash.encode()})
        msg = netsurf.utils.save_object(self, filepath, meta_attributes = {'class': 'netsurf.Experiment'})
        netsurf.utils.log._custom('EXP',msg)



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
        experiment_hash = self.ranking.hash

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
            '🔗 Alias': self.ranking.alias}
            # 'Normalize': self.config['normalize']}

        if hasattr(self.ranking, 'config'):
            if 'suffix' in self.ranking.config:
                if self.ranking.config['suffix'] != '':
                    props['Suffix'] = self.ranking.config['suffix']
        
        for kw in self.global_metrics:
            if self.global_metrics[kw] is not None and kw != 'elapsed_time':
                props[kw.replace('_',' ').capitalize()] = self.global_metrics[kw]
        
        # Add configs 
        if hasattr(self.ranking, 'config'):
            for kw, kv in self.ranking.config.items():
                if kw == 'ascending':
                    kw = '🔺 Ascending'
                if kw == 'normalize':
                    kw = ' Normalize'
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
        # Get the number of characters the 
        exp_name_entry = f' Experiment: {self.name} '
        method_name_entry = f' Method: {self.ranking.method} '
        num_chars = len(exp_name_entry)
        msg = f'+{"-"*num_chars}+\n'
        msg += f'|{method_name_entry:{num_chars}}|\n'
        msg += f'|{exp_name_entry}|\n'
        msg += f'+{"-"*num_chars}+\n'

        # Add config to msg
        msg += f'|{" Ranking Config:":{num_chars}}|\n'
        msg += f'+{"-"*num_chars}+\n'
        def print_dict(d, indent = 1):
            msg = ''
            for kw in d:
                if isinstance(d[kw], dict):
                    msg += f'|{"    "*indent}{"-"*((num_chars-1)-indent*4)}+\n'
                    N = num_chars - len(kw) - 4*indent - 1
                    msg += f'|{"    "*indent}{kw:s} {" "*N}|\n'
                    msg += print_dict(d[kw], indent = indent + 1)
                    msg += f'|{"    "*indent}{"-"*((num_chars-1)-indent*4)}+\n'
                else:
                    N = num_chars - (len(kw) + np.maximum(len(str(d[kw])),50) + 2) - 4*indent - 1
                    msg += f'|{"    "*indent}{kw:s}: {str(d[kw]):50s} {" "*N}|\n'
            return msg
        msg += print_dict(self.ranking.config)
        msg += f'+{"-"*num_chars}+\n'

        return msg
    

""" ###########################################################################################
# 
# EXPERIMENT COMPARISON, PROFILERS
#
### ###########################################################################################
""" 
@dataclass
class ExperimentComparator:
    """
    Class to compare different ranking methods
    """
    experiments: List[Experiment] = field(default_factory=list)

    # Init the metrics (AUC/VUS,etc.)
    
    comparison: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        if not self.experiments:
            netsurf.info("Initialized ExperimentComparator with no experiments.")

    @property
    def _keys(self):
        if not self.experiments:
            return []
        else:
            return [exp.name for exp in self.experiments]

    """ Method to create a ranker """
    def create(self, *args, **kwargs):
        # Build the experiment
        exp = Experiment(*args, **kwargs)
        # Add the experiment to the list
        self.experiments.append(exp)
        # Return the experiment
        return exp

    """ Methods to attach rankers """
    def append(self, experiment: Experiment):
        if not isinstance(experiment, Experiment):
            raise TypeError("Only Experiment instances can be added.")
        self.experiments.append(experiment)
        netsurf.info(f"Experiment {experiment.name} added.")
    
    def extend(self, experiments: List[Experiment]):
        if not all(isinstance(r, Experiment) for r in experiments):
            raise TypeError("Only Experiment instances can be added.")
        self.experiments.extend(experiments)
        netsurf.info(f"Experiments {', '.join([r.name for r in experiments])} added.")
    
        # Set the keys
    """ Make sure we define iter so we can do stuff like 'if <str> in <RankingComparator>' """
    def __iter__(self):
        """ Iterate over rankers """
        for experiment in self.experiments:
            yield experiment.name

    """ Length """
    def __len__(self):
        """ Get length """
        return len(self.experiments)

    """ Get item """
    def __getitem__(self, item):
        if isinstance(item, str):
            # Get the ranker by alias
            if item not in self._keys:
                raise KeyError(f"Experiment '{item}' not found.")
            idx = self._keys.index(item)
            return self.experiments[idx]
        elif isinstance(item, int):
            # Get the ranker by index
            if item < 0 or item >= len(self.experiments):
                raise IndexError(f"Experiment index '{item}' out of range.")
            return self.experiments[item]

    @staticmethod
    def _trapezoidal_auc(x, y):
        if len(x) < 2:
            return 0.0, 0.0
        sorted_idx = np.argsort(x)
        x_sorted = np.array(x)[sorted_idx]
        y_sorted = np.array(y)[sorted_idx]
        area = sklearn_auc(x_sorted, y_sorted)
        max_area = (np.max(x_sorted) - np.min(x_sorted)) * (np.max(y_sorted) - 0)
        return area, area / max_area if max_area > 0 else 0.0

    @staticmethod
    def compute_auc(curves):
        aucs = []
        for group_col, sweep_col in [('ber', 'protection'), ('protection', 'ber')]:
            for val, group in curves.groupby(group_col):
                for stat in ['mean', 'max', 'min', 'std']:
                    x = group[sweep_col].values
                    y = group[stat].values
                    area, rel_area = ExperimentComparator._trapezoidal_auc(x, y)

                    # Use true_ber if available
                    if 'true_ber' in group.columns:
                        x_true = group['true_ber'].values
                        true_area, true_rel_area = ExperimentComparator._trapezoidal_auc(x_true, y)
                    else:
                        true_area = true_rel_area = 0.0

                    aucs.append({
                        group_col: val, sweep_col: 'all', 'auc': area, 'rel_auc': rel_area,
                        'true_auc': true_area, 'true_rel_auc': true_rel_area,
                        'x': sweep_col, 'y': stat
                    })
        return pd.DataFrame(aucs)

    @staticmethod
    def tetrahedron_volume(p):
        if len(p) != 4:
            return 0.0
        a, b, c, d = map(np.array, p)
        return np.abs(np.dot(a - d, np.cross(b - d, c - d))) / 6.0

    @staticmethod
    def compute_vus(curves):
        protections = np.sort(curves['protection'].unique())
        bers = np.sort(curves['ber'].unique())
        true_bers = np.sort(curves['true_ber'].unique()) if 'true_ber' in curves.columns else bers

        result = {}
        for stat in ['mean', 'max', 'min', 'std']:
            for use_true in [False, True]:
                zn = 'true_' if use_true else ''
                y_axis = true_bers if use_true else bers
                vol = 0.0
                for i in range(len(protections) - 1):
                    for j in range(len(y_axis) - 1):
                        grid_pts = []
                        for dx, dy in [(0, 0), (0, 1), (1, 1), (1, 0)]:
                            px = protections[i + dx]
                            py = y_axis[j + dy]
                            col_y = 'true_ber' if use_true else 'ber'
                            match = (curves['protection'] == px) & (curves[col_y] == py)
                            z = curves.loc[match, stat].values
                            if len(z) > 0:
                                grid_pts.append((px, py, z[0]))
                        if len(grid_pts) == 4:
                            b0 = (grid_pts[0][0], grid_pts[0][1], 0.0)
                            b3 = (grid_pts[3][0], grid_pts[3][1], 0.0)
                            vol += ExperimentComparator.tetrahedron_volume([grid_pts[0], grid_pts[1], grid_pts[2], b0])
                            vol += ExperimentComparator.tetrahedron_volume([grid_pts[1], grid_pts[2], grid_pts[3], b3])

                max_z = curves[stat].max()
                dx = protections[-1] - protections[0]
                dy = y_axis[-1] - y_axis[0]
                norm_vol = vol / (dx * dy * max_z) if max_z > 0 else 0.0

                if stat not in result:
                    result[stat] = {}
                result[stat][f'{zn}vus'] = vol
                result[stat][f'{zn}vus_rel'] = norm_vol

        return pd.DataFrame(result)
    
    """ Perform actual evaluation of the experiments """
    def evaluate_experiments(self):
        """
        Evaluate all experiments in the comparator and compute AUC and VUS for each.
        Returns a dictionary: 
            {
                experiment_name: {
                    'loss': <loss_name>,
                    'metrics': List[str],
                    <loss_name>: {
                        'auc': df_auc, 
                        'vus': df_vus, 
                    },
                    <metric1>: {
                        'auc': df_auc,
                        'vus': df_vus,
                    },
                    <metric2>: { 
                        'auc': df_auc,
                        'vus': df_vus,
                    }
                }
            }
        """
        results = {}
        for exp in self.experiments:
            if not hasattr(exp, 'results') or exp.results is None:
                continue

            df = exp.results
            loss_name = getattr(df, '_loss', 'loss')
            metric_names = getattr(df, '_metrics', [])

            exp_result = {
                'loss': loss_name,
                'metrics': metric_names
            }

            # Compute for the loss
            if loss_name in df.columns:
                loss_curves = df.groupby(['protection', 'ber'])[loss_name].agg(['mean', 'max', 'min', 'std']).reset_index()
                if 'true_ber' in df.columns:
                    true_curves = df.groupby(['protection', 'true_ber'])[loss_name].agg(['mean', 'max', 'min', 'std']).reset_index()
                    loss_curves['true_ber'] = true_curves['true_ber']
                exp_result[loss_name] = {
                    'auc': self.compute_auc(loss_curves),
                    'vus': self.compute_vus(loss_curves)
                }

            # Compute for each metric
            for metric in metric_names:
                if metric not in df.columns:
                    continue
                metric_curves = df.groupby(['protection', 'ber'])[metric].agg(['mean', 'max', 'min', 'std']).reset_index()
                if 'true_ber' in df.columns:
                    true_curves = df.groupby(['protection', 'true_ber'])[metric].agg(['mean', 'max', 'min', 'std']).reset_index()
                    metric_curves['true_ber'] = true_curves['true_ber']
                exp_result[metric] = {
                    'auc': self.compute_auc(metric_curves),
                    'vus': self.compute_vus(metric_curves)
                }

            results[exp.ranking.method] = exp_result

        return results
    
    @staticmethod
    def plot_2d_curves(experiments: List[Experiment], metric='loss', stat='mean', xlog = True, axs = None, sharex = True, show = True):
        """
        Plot 2D BER vs <metric> curves for each experiment.
        
        Args:
            metric (str): Name of the metric/loss to plot (e.g. 'loss', 'mse', 'accuracy').
            stat (str): Statistic to show ('mean', 'max', 'min', 'std').
        """
        if axs is not None:
            if len(axs) != len(experiments):
                netsurf.error(f'Number of axes ({len(axs)}) does not match number of experiments ({len(experiments)}).')
                axs = None
        
        show &= (axs is None)
        if axs is None:
            fig, axs = plt.subplots(len(experiments), 1, figsize=(8, 5 * len(experiments)), sharex=sharex)
            axs = axs.flatten()
        else:
            fig = axs[0].figure


        for i, exp in enumerate(experiments):
            df = exp.results
            if metric == 'loss':
                # Pick actual column
                metric = df._loss
            if metric not in df.columns:
                print(f"Metric '{metric}' not found in {exp.ranking.method}, skipping.")
                continue
            
            # Aggregate per (protection, ber) pair
            curves = df.groupby(['protection', 'ber'])[metric].agg(['mean', 'max', 'min', 'std']).reset_index()

            protections = sorted(curves['protection'].unique())
            for p in protections:
                subset = curves[curves['protection'] == p]
                label = f'Protection {int(p * 100)}%'
                x = subset['ber'].values
                y = subset[stat].values
                yerr = subset['std'].values if 'std' in subset.columns else None

                axs[i].plot(x, y, marker='o', label=label)
                if yerr is not None and stat == 'mean':
                    axs[i].fill_between(x, y - yerr, y + yerr, alpha=0.2)

            axs[i].set_title(f'{exp.ranking.method}: BER vs {metric} ({stat})')
            if i == (len(axs)-1): axs[i].set_xlabel('Bit Error Rate (BER)')
            axs[i].set_ylabel(f'{metric.capitalize()} ({stat})')
            axs[i].grid(True, linestyle='--', alpha=0.5)
            if xlog: axs[i].set_xscale('log')
            axs[i].legend()

        plt.tight_layout()
        if show:
            plt.show()
        return fig, axs

    @staticmethod
    def plot_3d_surface(experiments: List[Experiment], stat: str = "mean", metric: str = "loss", 
                        x='ber', y='protection', axs = None, 
                        title=None, xlabel=None, ylabel=None, zlabel=None,
                        xlims=None, ylims=None, zlims=None,
                        show: bool = True, xlog = True, ylog = False):
        """
        Plot a 3D surface for each experiment showing BER vs Protection vs Metric.

        Args:
            stat (str): Which statistic to plot ('mean', 'std', 'max', 'min', etc.).
            metric (str): Name of the metric to visualize (must match column in result).
            save_path (str): If provided, save figure to this path instead of displaying it.
        """
        if axs is not None:
            if len(axs) != len(experiments):
                netsurf.error(f'Number of axes ({len(axs)}) does not match number of experiments ({len(experiments)}).')
                axs = None
        
        show &= (axs is None)
        if axs is None:
            fig, axs = plt.subplots(len(experiments), 1, figsize=(8, 5 * len(experiments)), subplot_kw={'projection': '3d'})
            axs = axs.flatten()
        else:
            fig = axs[0].figure
        
        # Loop through each experiment
        for i, exp in enumerate(experiments):
            
            curves = exp.results
            if metric == 'loss':
                # Pick actual column
                metric = curves._loss
            if metric not in curves.columns:
                netsurf.warn(f"[!] Metric '{metric}' not found in experiment '{exp.ranking.method}'. Skipping.")
                continue

            df_grouped = curves.groupby([x, y])[metric].agg(['mean', 'max', 'min', 'std']).reset_index()
            X_, Y_, Z_ = df_grouped[x], df_grouped[y], df_grouped[stat]

            x_vals = np.log10(X_) if xlog else X_
            y_vals = np.log10(Y_) if ylog else Y_

            xi = np.linspace(x_vals.min(), x_vals.max(), 50)
            yi = np.linspace(y_vals.min(), y_vals.max(), 50)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata((x_vals, y_vals), Z_, (Xi, Yi), method='cubic')

            surf = axs[i].plot_surface(Xi, Yi, Zi, cmap='bwr', edgecolor='#0000000f', alpha = 0.8, lw=0.8)
            axs[i].contourf(Xi, Yi, Zi, zdir='z', offset=0, cmap='coolwarm', alpha=0.5)
            axs[i].contourf(Xi, Yi, Zi, zdir='x', offset=xi.min(), cmap='coolwarm', alpha=0.5)
            axs[i].contourf(Xi, Yi, Zi, zdir='y', offset=yi.max(), cmap='coolwarm', alpha=0.5)

            if xlog:
                axs[i].set_xlim(xi.min(), xi.max()) if xlims is None else axs[i].set_xlim(*xlims)
                axs[i].xaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f"$10^{{{int(val)}}}$"))
            if ylog:
                axs[i].set_ylim(yi.min(), yi.max()) if ylims is None else axs[i].set_ylim(*ylims)
                axs[i].yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f"$10^{{{int(val)}}}$"))

            axs[i].set_zlim(*zlims) if zlims else None
            axs[i].set_xlabel(xlabel or 'BER')
            axs[i].set_ylabel(ylabel or 'Protection')
            axs[i].set_zlabel(zlabel or f'{metric.capitalize()} ({stat})')
            axs[i].set_title(title or f'{exp.ranking.method} - VUS')
            axs[i].set_box_aspect([1, 1, 0.5])
            
        plt.tight_layout()

        if show:
            plt.show()
        return fig, axs

    @staticmethod
    def plot_barplot(evaluation_results: pd.DataFrame, metric='loss', kind = 'vus', stat: str = 'mean',
                        ax=None, standalone=True, xlog=False, ylog=False, ylims=None, cmap='viridis',
                        remove_baseline=False, baseline=None, single_out='random',
                        title=None, ylabel=None, filename=None, show=True):


        if metric == 'loss':
            metric = evaluation_results['loss']
        
        subdata = 'vus' if 'vus' in kind else 'auc'

        # Collect data
        data = []
        for method, res in evaluation_results.items():
            if metric not in res:
                raise ValueError(f'Metric {metric} not found in results keys. Valid values are: {res.keys()}')
            subdf = res[metric].get(subdata, None)
            if subdf is None:
                continue
            if kind not in subdf.index:
                raise ValueError(f'Kind {kind} not in index, valid values are: {", ".join(subdf.index)}')
            row = subdf.loc[kind]
            data.append({'method': method, subdata: row['mean'], 'min': row['min'], 'max': row['max'], 'std': row['std']})

        df = pd.DataFrame(data)

        # If df is empty, pass
        if df.empty:
            return None, None

        ascending = True
        if metric:
            ascending = metric.lower() not in ['accuracy', 'acc']
        df = df.sort_values(subdata, ascending=ascending)

        if remove_baseline and baseline in df['method'].values:
            base_val = df[df['method'] == baseline][subdata].values[0]
            df[subdata] -= base_val

        if ax is None or standalone:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        ylims = ylims if ylims is not None else (0.95 * min(df.min().values[1:]), 1.05 * max(df.max().values[1:])) 
        yrange = np.maximum(ylims[1] - ylims[0],0.01)

        # Setup color mapping
        cmap = plt.get_cmap(cmap)
        norm = plt.Normalize(df[subdata].min(), df[subdata].max())
        color_mapper = lambda v: cmap(norm(v))

        bar_w = 0.4
        bar_space = 0.1
        xticks, xticklabels = [], []

        old_hatch_lw = plt.rcParams['hatch.linewidth']
        plt.rcParams['hatch.linewidth'] = 0.3
        plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.3)

        for i, (_, row) in enumerate(df.iterrows()):
            x = i * (bar_w + bar_space)
            vus, min_, max_, std_ = row['vus'], row['min'], row['max'], row['std']
            color = color_mapper(vus)
            hatch = '//' if row['method'] != single_out else 'oo'

            ptop = ax.fill_between([x, x + bar_w], vus, max_, color=color[:-1] + (0.3,), edgecolor='k', linewidth=0.5)
            ptop.set_hatch(hatch)

            pbot = ax.fill_between([x, x + bar_w], min_, vus, color=color[:-1] + (0.7,), edgecolor='k', linewidth=0.5)
            pbot.set_hatch(hatch)

            ax.plot([x, x + bar_w], [vus, vus], color='k', linewidth=1.4)
            ax.text(x + bar_w / 2, max_ + 0.01*yrange, f'{max_:.3f}', ha='center', va='bottom', fontsize=9)
            ax.text(x + bar_w / 2, min_ - 0.01*yrange, f'{min_:.3f}', ha='center', va='top', fontsize=9)

            xticks.append(x + bar_w / 2)
            if row['method'] == 'weight_abs_value':
                label = r'$|\theta|$'
            else:
                label = row['method']
                if 'delta' in label:
                    label = '(' + label.replace('delta','') + ')' + r'$\Delta$'
                label = label.replace('_', '\n')
                label = label.replace('fisher', 'Fisher\n' + r'$\|\nabla\|^2$')
                label = label.replace('grad',r'$\nabla\theta$')
                label = label.replace('hessian', 'Hessian\n' + r'$\mathcal{{H}}\theta$')

        #     rf"$\mathcal{{H}}$ (entropy): {format_value(self.entropy)}",
        #     rf"$\sigma^2$ (variance): {format_value(self.variance)}",
        #     rf"$\gamma_1$ (skew): {format_value(self.skewness)}",
        #     rf"$\gamma_2$ (kurtosis): {format_value(self.kurtosis)}",
        #     rf"$\mu$ (mean): {format_value(self.mean)}",
        #     rf"$\sigma$ (std): {format_value(self.std)}",
        #     rf"$min$: {format_value(self.min)}",
        #     rf"$max$: {format_value(self.max)}",
        #     rf"$\|x\|_1$ (L1 energy): {format_value(self.l1_energy)}",
        #     rf"$\|x\|_2^2$ (L2 energy): {format_value(self.l2_energy)}",
        #     rf"$\mathcal{{G}}$ (gini): {format_value(self.gini)}",
        # ]

            xticklabels.append(label)

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=0, ha='center')
        ax.set_ylabel(ylabel or stat)
        ax.set_ylim(ylims)
        if title is None:
            title = f'{kind.upper()} ({metric}) - {stat}' if metric else f'{kind.upper()} - {stat}'
        if title:
            ax.set_title(title)
        if xlog:
            ax.set_xscale('log')
        if ylog:
            ax.set_yscale('log')

        ax.grid(True, which='major', linestyle='--')
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle=':')


        if filename:
            plt.savefig(filename, bbox_inches='tight')
        if show and standalone:
            plt.show()

        plt.rcParams['hatch.linewidth'] = old_hatch_lw

        return fig, ax

    @staticmethod
    def plot_radar_comparison(results_dict, stat='mean', normalized=False, show=True):
        """
        Plot radar chart comparing AUC and VUS stats for all methods for each metric.
        
        Args:
            results_dict (dict): Output of evaluate_experiments().
            stat (str): Statistic to use ('mean', 'max', 'min', 'std').
            normalized (bool): Whether to normalize each stat across methods.
            show (bool): Whether to display the figure.
        """
        raise NotImplementedError('Not Implemented Yet')


    @staticmethod
    def compute_ranking_distribution(results, metric='loss', stat='mean', n_samples=1000, seed=42):
        """
        Compute ranking distributions via resampling from experiment evaluation results.

        Args:
            results (dict): Output from `evaluate_experiments()`.
            metric (str): Metric to analyze.
            stat (str): Statistic to use (e.g., 'mean', 'std', 'max', ...).
            n_samples (int): Number of resampling iterations.
            seed (int): Random seed.

        Returns:
            DataFrame: Multi-indexed DataFrame with (protection, ber, sample) -> method -> rank.
        """
        rng = np.random.default_rng(seed)
        all_records = []

        for method, data in results.items():
            if metric == 'loss':
                metric = data[metric]

            if metric not in data:
                continue
            auc_df = data[metric]['auc']
            for _, row in auc_df.iterrows():
                x = row['x']  # 'protection' or 'ber'
                y = row['y']  # e.g. 'mean'
                key = row[x]  # the value of protection or ber
                val = row[stat]  # the stat value
                all_records.append({
                    'method': method,
                    'sweep': x,
                    'value': key,
                    'stat': y,
                    'score': val
                })
        df = pd.DataFrame(all_records)

        # Focus on only one sweep (e.g., protection)
        sweep = df['sweep'].iloc[0]
        values = sorted(df['value'].unique())
        methods = sorted(df['method'].unique())

        result = []

        for v in values:
            df_sub = df[(df['value'] == v) & (df['stat'] == stat)]
            method_scores = {row['method']: row['score'] for _, row in df_sub.iterrows()}

            scores_matrix = []
            for method in methods:
                mu = method_scores.get(method, 0)
                scores_matrix.append(rng.normal(loc=mu, scale=1e-6, size=n_samples))
            scores_matrix = np.array(scores_matrix)  # shape (n_methods, n_samples)

            ranks = np.argsort(np.argsort(-scores_matrix, axis=0), axis=0) + 1  # lower is better

            for m_idx, method in enumerate(methods):
                for sample_idx in range(n_samples):
                    result.append({
                        sweep: v,
                        'method': method,
                        'sample': sample_idx,
                        'rank': ranks[m_idx, sample_idx]
                    })

        return pd.DataFrame(result)
    
    def html(self):

        # Compute metrics
        res = self.evaluate_experiments()
        
        # Build main container
        ct = pg.CollapsibleContainer("🏅 Performance", layout='vertical')

        # Plot barplot
        if len(res) > 0:
            
            metrics = list(res.items())[0]['metrics']
            ct_bar = pg.CollapsibleContainer("📊 Barplots", layout='vertical')
            # Add it to ct 
            ct.append(ct_bar)

            metrics = ['loss'] + metrics
            for m in metrics:
                fig, ax = ExperimentComparator.plot_barplot(res, kind = 'vus', metric = m, stat = 'mean')

                # Create container just for this 
                ct_bar_m = pg.CollapsibleContainer(f"Metric: {m}", layout='vertical')
                # Make fig
                p = pg.Image(fig, embed = True)
                ct_bar_m.append(p)
                # Add to ct_bar
                ct_bar.append(ct_bar_m)
                # close fig
                plt.close(fig)
            
            # That's it for now
        else:
            # Add text
            ct.append(pg.Text('No results to display'))
        
        return ct
        


        #ExperimentComparator.plot_2d_curves(exp_comp.experiments, metric='categorical_accuracy')
        #ExperimentComparator.plot_2d_curves(exp_comp.experiments, metric='loss')
        #ExperimentComparator.plot_3d_surface(exp_comp.experiments, metric='categorical_accuracy', zlims = (0,1))
        #ExperimentComparator.plot_3d_surface(exp_comp.experiments, metric='loss')
        #ExperimentComparator.plot_barplot(res, kind = 'vus', metric = 'categorical_accuracy', stat = 'mean')
        #ExperimentComparator.plot_barplot(res, kind = 'vus_rel', metric = 'categorical_accuracy', stat = 'mean')
        #ExperimentComparator.plot_radar_comparison(res)