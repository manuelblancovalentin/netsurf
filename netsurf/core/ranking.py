""" Code for weight ranking according to different methods """

""" Modules """
import os
import copy
import time

""" Numpy and pandas """
import numpy as np
import pandas as pd

""" Matplotlib """
import matplotlib.pyplot as plt 

""" Tensorflow """
import tensorflow as tf

#from tqdm.notebook import tqdm
from tqdm import tqdm

""" netsurf modules """
import netsurf

# Fkeras for Hessian ranking and all the dependencies
import fkeras

####################################################################################################################
#
# GENERIC RANKER (PARENT OF ALL)
#
####################################################################################################################

# Generic Weight Ranker (parent of rest)
class WeightRanker:
    _ICON = "🏆"
    def __init__(self, quantization: 'QuantizationScheme',
                  ascending = False, times_weights = False, normalize_score = False, 
                  **kwargs):

        """ Initialize df """
        self.df = None
        self.quantization = quantization
        self.complete_ranking = True
        self.ascending = ascending
        self.times_weights = times_weights
        self.normalize_score = normalize_score
    
    """ Extracting the table of weights and creating the pandas DF is the same 
        for all methods, so we can define it inside the generic weightRanker obj 
    """
    def extract_weight_table(self, model, quantization, verbose = False, **kwargs):
        # Get all variables for this model 
        variables = model.trainable_variables
        vnames = [v.name for v in model.variables]

        # This is what the table should look like:
        # | param name | index (in trainable_variables) | coord (inside variable) | value | rank | susceptibility | bit |
        # +------------+--------------------------------+--------------------------+-------+------+----------------+-----+
        # Create dictionary to store all the values
        df = {'param': [], 'global_param_num': [], 'variable_index': [], 'internal_param_num': [],
              'coord': [], 'bit': [], 'value': [], 'rank': [], 'susceptibility': [], 
            'binary': [], 'pruned': []}
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
            binary = np.apply_along_axis("".join, 1, (quantization.bin(values)*1).astype(str))

            # Finally, init the bit 
            bits = np.repeat(np.arange(quantization.n + quantization.s - 1, -quantization.f-1, -1), len(values))

            # global_param_num
            global_param_num = cum_index + np.arange(len(values))

            # Add cum_index
            cum_index += len(values)

            # We need to repeat this num_bits times
            names = np.tile(names, quantization.m)
            variable_index = np.tile(variable_index, quantization.m)
            internal_param_num = np.tile(internal_param_num, quantization.m)
            binary = np.tile(binary, quantization.m)
            coords = np.tile(coords, (quantization.m, 1))
            values = np.tile(values, quantization.m)
            # Repeat global_param_num
            global_param_num = np.tile(global_param_num, quantization.m)

            # Also, keep track of parameters that have been pruned
            # get the pruned mask 
            pruned_mask_vname = v.name.replace(':','_prune_mask:')
            if pruned_mask_vname in vnames:
                pruned = model.variables[vnames.index(pruned_mask_vname)].numpy().flatten() == 0
                # tile
                pruned = np.tile(pruned, quantization.m)
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


        # Build df 
        df = pd.DataFrame(df)
        # The index is the global parameter number. Explicitly set it as a column as well (in case we
        # re-sort and reindex by mistake or not later)
        df['param_num'] = df.index

        # Let's randomize before ranking so we get rid of locality 
        df = df.sample(frac=1)

        # Store in place and return 
        self.df = df

        return df

    
    # Method to write table to csv file 
    def _save_to_csv(self, df, filepath = None):
        # Make sure parent dir exists 
        os.makedirs(os.path.dirname(filepath), exist_ok = True)
        # Write to file 
        with open(filepath, 'w') as fo:
            df.to_csv(fo)
        
        print(f'Rank dataframe written to file {filepath}')
    
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
        if not isinstance(self.df['binary'][0], str):
            self.df['binary'] = ["".join([str(xx) for xx in x]) for x in (1.0*self.quantization.bin(self.quantization(self.df['value'].values))).astype(int)]
        elif isinstance(self.df['binary'][0], str):
            # But length is not correct
            if len(self.df['binary'][0]) != self.quantization.m:
                self.df['binary'] = ["".join([str(xx) for xx in x]) for x in (1.0*self.quantization.bin(self.quantization(self.df['value'].values))).astype(int)]
        # if impact in ranking, add it too
        if 'impact' in self.df.columns:
            items.append(('impact', 'Impact', lambda x: x, 'brown'))
        if 'gradient' in self.df.columns:
            items.append(('gradient', 'Gradient', lambda x: x, 'brown'))
        if 'hessian' in self.df.columns:
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
            netsurf.utils.plot.plot_avg_and_std(transform(self.df[field]), w, axs[i], shadecolor=color, ylabel=title)

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
class RandomWeightRanker(WeightRanker):
    _ICON = "🎲"
    def __init__(self, quantization, *args, **kwargs):
        """."""
        super().__init__(quantization, *args, **kwargs)

        # Init df 
        self.df = None
        self.quantization = quantization

    # Method to actually rank the weights
    def rank(self, model, *args, exclude_zero = True, **kwargs):
        # Call super method to obtain DF 
        df = self.extract_weight_table(model, self.quantization)

        # Susceptibility here is considered uniform (hence the randomness assigning TMR)
        df['susceptibility'] = [1/(len(df))]*len(df)
        df['rank'] = np.random.permutation(np.arange(len(df)))
        
        # Sort by rank
        df = df.sort_values(by='rank')

        # assign to self 
        self.df = df

        return df
    
    # save to csv
    def save_to_csv(self, *args, **kwargs):
        self._save_to_csv(self.df, *args, **kwargs)
    
    @property
    def alias(self):
        return 'random'


####################################################################################################################
#
# WEIGHT VALUE RANKERS (AbsoluteValue)
#
####################################################################################################################

""" Rank weights according to their absolute value (the larger, the most important) """
class AbsoluteValueWeightRanker(WeightRanker):
    def __init__(self, quantization: 'QuantizationScheme', *args, **kwargs):
        """."""
        super().__init__(quantization, *args, **kwargs)

        # Init df 
        self.df = None
        self.quantization = quantization

    # Method to actually rank the weights
    def rank(self, model, *args, ascending = False, **kwargs):
        # Call super method to obtain DF 
        df = self.extract_weight_table(model, self.quantization)

        # Susceptibility here is considered uniform (hence the randomness assigning TMR)
        #df = df.sort_values(by='value', key=abs, ascending=ascending)
        df['susceptibility'] = np.abs(df['value'].values)
        df = df.sort_values(by=['pruned','bit','susceptibility'], ascending = [True, False, ascending])
        
        df['rank'] = np.arange(len(df))
        
        #df['susceptibility'] = -np.log10(np.abs(df['value'].values))/np.max(np.abs(np.log10(np.abs(df['value'].values))))
        
        # assign to self 
        self.df = df

        return df
    
    # save to csv
    def save_to_csv(self, *args, **kwargs):
        self._save_to_csv(self.df, *args, **kwargs)
    
    @property
    def alias(self):
        return 'weight_abs_value'


####################################################################################################################
#
# POSITIONAL RANKERS (Bitwise MSB_LSB, layerwise, etc)
#
####################################################################################################################

""" Rank weights by layer (top to bottom, bottom to top or custom order) """
class BitwiseWeightRanker(WeightRanker):
    _ICON = "🔢"
    def __init__(self, quantization:'QuantizationScheme', *args, **kwargs):
        """."""
        super().__init__(quantization, *args, **kwargs)

        # Init df 
        self.df = None
        self.quantization = quantization

    # Method to actually rank the weights
    def rank(self, model, *args, ascending = False, **kwargs):
        # Call super method to obtain DF 
        df = self.extract_weight_table(model, self.quantization)

        # Susceptibility here is considered uniform (hence the randomness assigning TMR)
        df = df.sort_values(by=['pruned','bit'], ascending = [True, ascending])
        df['rank'] = np.arange(len(df))
        df['susceptibility'] = 2.0**df['bit']
        
        # assign to self 
        self.df = df
        self.ascending = ascending

        return df
    
    # save to csv
    def save_to_csv(self, *args, **kwargs):
        self._save_to_csv(self.df, *args, **kwargs)
    
    @property
    def alias(self):
        return f'bitwise_{"lsb" if self.ascending else "msb"}'


""" Rank weights by layer (top to bottom, bottom to top or custom order) """
class LayerWeightRanker(WeightRanker):
    _ICON = "🎞️"
    def __init__(self, quantization: 'QuantizationScheme', *args, **kwargs):
        """."""
        super().__init__(quantization, *args, **kwargs)

        # Init df 
        self.df = None
        self.quantization = quantization

    # Method to actually rank the weights
    def rank(self, model, *args, exclude_zero = True, ascending = True, **kwargs):
        # Call super method to obtain DF 
        df = self.extract_weight_table(model, self.quantization)

        # Susceptibility here is considered uniform (hence the randomness assigning TMR)
        # (variable_index is almost like layer index, although there is a preference of kernel vs bias, cause 
        # when listing variables kernel always comes before bias, even for the same layer, but whatever).
        # If required, we could actually enforce layer index computation by grouping variables by their name 
        # (removing the "/..." part of the name) and then sorting by the order of appearance in the model, 
        # but I don't really think this is required right now. 
        df = df.sort_values(by=['pruned', 'bit', 'variable_index'], ascending = [True, False, ascending])
        df['rank'] = np.arange(len(df))
        df['susceptibility'] = 2.0**df['bit'] * (self.quantization.m - df['variable_index'] + 1)
        
        # assign to self 
        self.df = df
        self.ascending = ascending

        return df
    
    # save to csv
    def save_to_csv(self, *args, **kwargs):
        self._save_to_csv(self.df, *args, **kwargs)
    
    @property
    def alias(self):
        return f'layerwise_{"first" if self.ascending else "last"}'



####################################################################################################################
#
# COMPENSATIONAL RANKERS (DiffBitsPerWeight, RecursiveUnevenRanker)
#
####################################################################################################################

""" Rank weights with how different bits per weight"""
class DiffBitsPerWeightRanker(WeightRanker):
    def __init__(self, quantization: 'QuantizationScheme', *args, **kwargs):
        """."""
        super().__init__(quantization, *args, **kwargs)

        # Init df 
        self.df = None
        self.quantization = quantization

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

    def rank(self, model, *args, exclude_zero = True, **kwargs):
        # Call super method to obtain DF 
        def process_value(value):
            b = netsurf.utils.float_to_binary(value,self.quantization.m)
            diff_sum = self.calculate_bit_differences(b)
            return diff_sum

        df = self.extract_weight_table(model, self.quantization.m)

        differences = np.vectorize(process_value)(df['value'].values)
        df['susceptibility'] = differences
        df = df.sort_values(by=['pruned','bit'], ascending = [True, False])
        df['rank'] = np.arange(len(df))

        # assign to self 
        self.df = df

        return df

    # save to csv
    def save_to_csv(self, *args, **kwargs):
        self._save_to_csv(self.df, *args, **kwargs)
    
    @property
    def alias(self):
        return f'diff_bits_per_weight'


""" Rank weights by using proportion Recursively """
class RecursiveUnevenRanker(WeightRanker):
    def __init__(self, quantization: 'QuantizationScheme', *args, **kwargs):
        """."""
        super().__init__(quantization, *args, **kwargs)

        # Init df 
        self.df = None
        self.quantization = quantization

    # Method to actually rank the weights
    def rank(self, model, *args, **kwargs):
        # Call super method to obtain DF 
        df = self.extract_weight_table(model, self.quantization, **kwargs)

        last_level = self.quantization.m - 1
        input_df = df[df['bit']==0]
        rank = self.rec(input_df, last_level)
        rank['rank'] = np.arange(len(rank))

        # assign to self 
        self.df = df

        return df

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
            
    # save to csv
    def save_to_csv(self, *args, **kwargs):
        self._save_to_csv(self.df, *args, **kwargs)
    
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
class GradRanker(WeightRanker):
    _ICON = "💈"
    def __init__(self, quantization: 'QuantizationScheme', *args, use_delta_as_weight = False, **kwargs):
        """."""
        super().__init__(quantization, *args, **kwargs)

        # Init df 
        self.df = None
        self.quantization = quantization
        self.use_delta_as_weight = use_delta_as_weight

    # Method to actually rank the weights
    def rank(self, model, X, Y, ascending = False, **kwargs):
        # Call super method to obtain DF 
        df = self.extract_weight_table(model, X, Y, self.quantization, ascending = ascending, **kwargs)

        # Finally, sort by susceptibility 
        df = df.sort_values(by=['pruned','bit','susceptibility'], ascending = [True, False, ascending])

        # assign to self 
        self.df = df

        return df
    
    """ Extracting the table of weights and creating the pandas DF is the same 
        for all methods, so we can define it inside the generic weightRanker obj 
    """
    def extract_weight_table(self, model, X, Y, quantization: 'QuantizationScheme', 
                                batch_size = 1000, verbose = True, 
                                normalize_score = False, times_weights = False,
                                ascending = False, absolute_value = True,
                                bit_value = None, out_dir = ".", **kwargs):
        
        # Call super to get the basic df 
        df = super().extract_weight_table(model, quantization, verbose = verbose, 
                                          ascending = ascending,
                                          **kwargs)

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


        # Store in place and return 
        self.df = df

        return df

    # save to csv
    def save_to_csv(self, *args, **kwargs):
        self._save_to_csv(self.df, *args, **kwargs)

""" Rank weights by using HiRes (gradcam++) as a weight to compute, on average, how important each weight is """
class HiResCamWeightRanker(GradRanker):
    def __init__(self, quantization: 'QuantizationScheme', *args, **kwargs):
        """."""
        super().__init__(quantization, *args, use_delta_as_weight = False, **kwargs)

        # Init df 
        self.df = None
        self.quantization = quantization
    
    @property
    def alias(self):
        alias = 'hirescam'
        if self.times_weights:
            alias += '_times_weights'
        if self.normalize_score:
            alias += '_norm'
        return alias

""" Rank weights by using HiRes (gradcam++) but instead of using the weights of the model, we use the DELTAS as weights """
class HiResDeltaRanker(GradRanker):
    _ICON="🔮"
    def __init__(self, quantization: 'QuantizationScheme', *args, **kwargs):
        """."""
        super().__init__(quantization, *args, use_delta_as_weight = True, **kwargs)

        # Init df 
        self.df = None
        self.quantization = quantization
    
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
class HessianRanker(WeightRanker):
    _ICON = "👴🏻"
    def __init__(self, quantization: 'QuantizationScheme', *args, use_delta_as_weight = False, **kwargs):
        """."""
        super().__init__(quantization, *args, **kwargs)

        # Init df 
        self.df = None
        self.quantization = quantization
        self.use_delta_as_weight = use_delta_as_weight

    # override extract_weight_table
    def extract_weight_table(self, model, X, Y, quantization: 'QuantizationScheme', 
                             batch_size = 480, inner_ranking_method = 'same',
                             normalize_score = False, times_weights = False,
                             ascending = False, absolute_value = True,
                             delta_ranking = False, verbose = True, **kwargs):
        
        # Call super to get the basic df 
        df = super().extract_weight_table(model, quantization, verbose = verbose, 
                                          ascending = ascending,
                                          **kwargs)

        # Make sure susceptibility is float
        df['susceptibility'] = df['susceptibility'].astype(float)

        # If use_delta as weight, clone the model and replace the weights with the deltas for each bit
        use_delta_as_weight = self.use_delta_as_weight

        # Store the original variable values first
        original_vars = [v.numpy() for v in model.trainable_variables]

        # Loop thru all bits (even if we are not using deltas as weights, this is so we can use the same loop
        # and code for both cases. If use_delta_as_weight is False we will break after the first loop iter)
        for ibit, bit in enumerate(np.arange(quantization.n + quantization.s - 1, -quantization.f-1, -1)):
            
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
            eigenvalues, eigenvectors = self.top_eigenvalues(model, X, Y, k=8)

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
                    for i in np.arange(quantization.n + quantization.s - 1, -quantization.f-1, -1):
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

        # Store in place and return 
        self.df = df

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


    @tf.function
    def _compute_hvp(self, model, x_batch, y_batch, v):
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
        # pbar = tqdm(total=total_iterations, disable=not verbose, 
        #            desc="Computing eigenvalues", unit="iter")
        
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
        pbar = tqdm(total=num_estimates, disable=not verbose, 
                   desc="Estimating trace", unit="sample")
        
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
        
        pbar.close()
        
        if verbose:
            print(f"Time taken: {time.time() - start_time:.2f} seconds")
            print(f"Final trace estimate: {current_mean:.6e}")
            
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
    def rank(self, model, X, Y, ascending = False, **kwargs):
        # Call super method to obtain DF 
        df = self.extract_weight_table(model, X, Y, self.quantization, ascending = ascending, **kwargs)

        # Finally, sort by susceptibility 
        df = df.sort_values(by=['pruned','bit','susceptibility'], ascending = [True, False, ascending])

        # assign to self 
        self.df = df

        return df

    # save to csv
    def save_to_csv(self, *args, **kwargs):
        self._save_to_csv(self.df, *args, **kwargs)


# Hessian based weight ranker 
# This was taken from: https://github.com/oliviaweng/CIFAR10/blob/fkeras/hls4ml/hessian_analysis.py
class HessianWeightRanker(HessianRanker):
    def __init__(self, quantization: 'QuantizationScheme', *args, **kwargs):
        """."""
        super().__init__(quantization, *args, use_delta_as_weight = False, **kwargs)

        # Init df 
        self.df = None
        self.quantization = quantization
    
    @property
    def alias(self):
        alias = 'hessian'
        if self.normalize_score:
            alias += '_norm'
        return alias

""" Same, but using deltas as weights """
class HessianDeltaWeightRanker(HessianRanker):
    def __init__(self, quantization: 'QuantizationScheme', *args, **kwargs):
        """."""
        super().__init__(quantization, *args, use_delta_as_weight = True, **kwargs)

        # Init df 
        self.df = None
        self.quantization = quantization
    
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
class AIBerWeightRanker(WeightRanker):
    _ICON = "🤖"
    def __init__(self, quantization, ascending=False, **kwargs):
        super().__init__(quantization, ascending=ascending, **kwargs)

        # Init df
        self.df = None
        self.quantization = quantization
    
    def extract_weight_table(self, model, X, Y, quantization, ascending = False, verbose=False, **kwargs):
        # Call super to get the basic df 
        df = super().extract_weight_table(model, quantization, verbose = verbose, 
                                          ascending = ascending, **kwargs)
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
        wrapper = netsurf.dnn.aiber.ModelWrapper(model, quantization)

        w_model = wrapper.wrapped_model
        
        # 2) Train the model for a few epochs
        history = wrapper.train_P(X[:100], Y[:100], num_epochs = 10)
        
        # Store in place and return 
        self.df = df

        return df

    # Method to actually rank the weights
    def rank(self, model, X, Y, ascending = False, **kwargs):
        # Call super method to obtain DF 
        df = self.extract_weight_table(model, X, Y, self.quantization, ascending = ascending, **kwargs)

        # Finally, sort by susceptibility 
        df = df.sort_values(by=['pruned','bit','susceptibility'], ascending = [True, False, ascending])

        # assign to self 
        self.df = df

        return df

    @property
    def alias(self):
        return 'aiber'





####################################################################################################################
#
# DELTAPOX RANKING (WORK IN PROGRESS...)
#
####################################################################################################################
class QPolarWeightRanker(WeightRanker):
    _ICON = "🧲"
    def __init__(self, quantization, ascending=False, **kwargs):
        super().__init__(quantization, ascending=ascending, **kwargs)

        # Init df
        self.df = None
        self.quantization = quantization

    def extract_weight_table(self, model, X, Y, quantization, ascending = False, verbose=False, batch_size = 1000, **kwargs):
        # Call super to get the basic df 
        df = WeightRanker.extract_weight_table(self, model, quantization, verbose = verbose, 
                                          ascending = ascending, **kwargs)
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

        # Store in place and return 
        self.df = df

        return df

    # Method to actually rank the weights
    def rank(self, model, X, Y, ascending = False, **kwargs):
        # Make sure the model has extracted the deltas 
        model.compute_deltas()
        
        # Call super method to obtain DF 
        df = self.extract_weight_table(model, X, Y, self.quantization, ascending = ascending, **kwargs)

        # Finally, sort by susceptibility 
        df = df.sort_values(by=['pruned','bit','susceptibility'], ascending = [True, False, ascending])

        # assign to self 
        self.df = df

        return df

    @property
    def alias(self):
        return 'qpolar'


class QPolarGradWeightRanker(QPolarWeightRanker, GradRanker):
    _ICON = "🧲"
    def __init__(self, quantization, ascending=False, **kwargs):
        super(GradRanker, self).__init__(quantization, ascending = ascending, **kwargs)
        super(QPolarGradWeightRanker, self).__init__(quantization, ascending=ascending, **kwargs)
        
        # Init df
        self.df = None
        self.quantization = quantization

    def extract_weight_table(self, model, X, Y, quantization, ascending = False, verbose=False, 
                             batch_size = 1000, **kwargs):
        # At this point we will have the susceptibility in terms of qpolar impact.
        df = QPolarWeightRanker.extract_weight_table(self, model, X, Y, quantization, verbose = verbose, 
                                                    ascending = ascending, batch_size = batch_size,
                                                    **kwargs)
        
        
        # Make sure we create a DEEP copy of df, cause this is pointing to self.df and it will be modified in the 
        # nxt call
        df_polar = df.copy()
        
        # Now let's get the gradients for all trainable variables
        df_grad = GradRanker.extract_weight_table(self, model, X, Y, quantization, verbose = verbose,
                                                    ascending = ascending, batch_size = batch_size, 
                                                    normalize_score = False, 
                                                    times_weights = False,
                                                    absolute_value = False, bit_value = None,
                                                    **kwargs)
        
        # Copy 
        df_grad = df_grad.copy()
        
        # Make sure self.df is None at this point (sanity check)
        self.df = None 

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

        # Set df
        self.df = df_polar

        return df_polar

    def rank(self, model, X, Y, ascending=False, **kwargs):
        return QPolarWeightRanker.rank(self, model, X, Y, ascending, **kwargs)

    @property
    def alias(self):
        return 'qpolargrad'


"""
Fisher weight ranker
"""
class FisherWeightRanker(WeightRanker):
    _ICON = "🐟"
    def __init__(self, quantization, ascending=False, **kwargs):
        super().__init__(quantization, ascending=ascending, **kwargs)

        # Init df
        self.df = None
        self.quantization = quantization

    def extract_weight_table(self, model, X, Y, quantization, ascending = False, verbose=False, batch_size = 1000, **kwargs):
        # Call super to get the basic df 
        df = WeightRanker.extract_weight_table(self, model, quantization, verbose = verbose, 
                                          ascending = ascending, **kwargs)
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

        self.df = df
        return df

    def rank(self, model, X, Y, ascending=False, **kwargs):
        df = self.extract_weight_table(model, X, Y, self.quantization, ascending=ascending, **kwargs)
        df = df.sort_values(by=['pruned', 'bit', 'susceptibility'], ascending=[True, False, ascending])
        self.df = df
        return df

    @property
    def alias(self):
        return 'fisher'





####################################################################################################################
#
# Access point to get rankers according to the method alias
#
####################################################################################################################

# Function to create a weight ranker given a couple of flags 
def build_weight_ranker(method: str, *args, **kwargs):
    options = {'random': RandomWeightRanker, 
                'weight_abs_value': AbsoluteValueWeightRanker,
                'layerwise': LayerWeightRanker,
                'bitwise': BitwiseWeightRanker,
                'hirescam': HiResCamWeightRanker,
                'hiresdelta': HiResDeltaRanker,
                'recursive_uneven': RecursiveUnevenRanker,
                'diffbitperweight': DiffBitsPerWeightRanker,
                # 'gradcrossentropy': GradCrossEntropyWeightRanker,
                #'oracle': OracleWeightRanker,
                'hessian': HessianWeightRanker,
                'hessiandelta': HessianDeltaWeightRanker,
                'qpolar': QPolarWeightRanker,
                'qpolargrad': QPolarGradWeightRanker,
                'fisher': FisherWeightRanker,
                'aiber': AIBerWeightRanker}
                
    ranker = options.get(method.lower(), WeightRanker)(*args, **kwargs)

    return ranker

