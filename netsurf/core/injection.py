""" Injectors of errors """
from typing import Type

# Modules 
#import re 
import numpy as np
#import copy

from tensorflow import keras
import tensorflow as tf
#from keras import backend as K # To get intermediate activations between layers

""" Pandas """
import pandas as pd

""" Custom modules """
import netsurf

class Attack:
    def __init__(self, N, variables):
        
        # If variables is not a dict, turn into one
        if not isinstance(variables, dict):
            variables = {v.name: v for v in variables}

        # If N is not a dict, turn into one
        if not isinstance(N, dict):
            N = {k: N for k in variables}
        
        # Now parse and initialize 
        N = {k: self._parse_N_(N[k], variables[k].shape) for k in variables}

        self.N = N
        # Reset iter 
        self._iter = 0

        # Find max and min of N (flips)
        self._min = np.maximum(0, np.min([np.sum(N[v]) for v in N]))
        self._max = np.maximum(0, np.max([np.sum(N[v]) for v in N]))
        self._varnames = list(N.keys())
    
    def _parse_N_(self, n, shape):
        if n is None:
            return np.zeros(shape)
        elif isinstance(n, float) or isinstance(n, int):
            # If n is between 0 and 1 then it's the probability of each element being 1
            if 0 <= n <= 1:
                # From a binomial
                return np.random.binomial(1, n, shape)
            else:
                # Othwerwise, we want to set to 1 n elements
                n = int(n)
                # If n is bigger than the number of elements, set all to 1
                if n >= np.prod(shape):
                    return np.ones(shape)
                # Otherwise, set n elements to 1
                return np.random.choice([0, 1], n, replace = False).reshape(shape)
        # If n is a numpy array, then it's the mask
        elif isinstance(n, np.ndarray):
            return n

    def __repr__(self):
        s = f'🔥 <Attack> @ ({hex(id(self))})\n'
        s += f'\t- Number of flips (overall): ({self._min}, {self._max})\n'
        s += f'\t- Flips per var:\n'
        for v in self.N:
            s += f'\t\t- {v}: {np.sum(self.N[v])}\n'
        return s

    def __len__(self):
        return len(self._varnames)
    
    def __iter__(self):
        # Reset iter
        self._iter = 0
        return self
    
    def __next__(self):
        if self._iter >= len(self):
            raise StopIteration
        else:
            val = self._varnames[self._iter]
            self._iter += 1
            return val

    def __getitem__(self, key: str):
        if isinstance(key, str):
            return self.N[key]
        elif isinstance(key, int):
            return self[self._varnames[key]]
        else:
            raise ValueError('Key must be either a string or an integer')

class AttackScheme:
    def __init__(self, attacks, stats):
        self.attacks = attacks
        self.stats = stats
        self._varnames = list(attacks.keys())

        # Iter counter
        self._iter_rep = 0
        self._max_iters = int(self.stats['num_attacks'])
    
    def __iter__(self):
        # Reset iter
        self._iter_rep = 0
        return self

    def __next__(self):
        if self._iter_rep >= self._max_iters:
            raise StopIteration
        else:
            # Get attack
            attack = self[self._iter_rep]

            # Update iter
            self._iter_rep += 1

            return attack
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            # Get attack for this particular rep
            attack = {v: 1*(self.attacks[v] == idx) for v in self.attacks}
            return Attack(N = attack, variables = attack)
        elif isinstance(idx, str):
            # Just get whatever property of the stats 
            return self.stats[idx]
        else:
            raise ValueError('Index must be either a string or an integer')

    def __len__(self):
        return self._max_iters

    def __repr__(self):
        s = f'📜 <AttackScheme> @ ({hex(id(self))})\n'
        s += f'\t- Number of attacks: {len(self)}\n'
        s += f'\t- Variables affected: {", ".join(self._varnames)}\n'
        s += f'\t- Protection: {self.stats["protection"]}\n'
        s += f'\t- BER: {self.stats["ber"]}\n'
        s += f'\t- True BER: {self.stats["true_ber"]}\n'
        return s


class AttackContainer:
    def __init__(self, attacks: dict, stats: dict):
        self.attacks = attacks
        self.stats = stats
        self._varnames = list(attacks.keys())

        # Iter counter
        self._iter_comb = 0
    
    # Loop thru combinations
    def __iter__(self):
        # Reset iter
        self._iter_comb = 0
        return self
    
    def __getitem__(self, idx):
        # if idx is integer, get subset 
        if isinstance(idx, int) or isinstance(idx, slice):
            # Get attack scheme for this particular combination
            attack = {v: self.attacks[v][idx] for v in self.attacks}
            # Get stats
            stats = self.stats.iloc[idx]
            # Build attackScheme
            return AttackScheme(attack, stats)
        elif isinstance(idx, str):
            # Just get whatever property of the stats 
            return self.stats[idx]
        else:
            raise ValueError('Index must be either a string or an integer')
    
    def __next__(self):
        if self._iter_comb >= len(self.stats):
            raise StopIteration
        else:
            # Get attack scheme for this particular combination
            scheme = self[self._iter_comb]
            # Update iter
            self._iter_comb += 1
            # Build attackScheme
            return scheme
    
    def __len__(self):
        return len(self.stats)

    def __repr__(self):
        protection = np.unique(self.stats["protection"].to_numpy())
        protection = list(np.unique([f'{t:.3%}' for t in protection]))
        # format to %
        ber_range = np.unique(self.stats["ber"].to_numpy())
        ber_range = list(np.unique([f'{t:.3%}' for t in ber_range]))
        # format to % 
        true_ber = np.unique(self.stats["true_ber"].to_numpy())
        true_ber = list(np.unique([f'{t:.3%}' for t in true_ber]))

        s = f'📦 <AttackContainer> @ ({hex(id(self))})\n'
        s += f'\t- Number of attack schemes (combs): {len(self)}\n'
        s += f'\t- Variables affected: {", ".join(self._varnames)}\n'
        s += f'\t- Total number of attacks (sum, max, min): {self.stats["num_attacks"].sum()}, {self.stats["num_attacks"].max()}, {self.stats["num_attacks"].min()}\n'
        s += f'\t- Protection range: {", ".join(protection)}\n'
        s += f'\t- BER range: {", ".join(ber_range)}\n'
        s += f'\t- True BER range: ({", ".join(true_ber)})\n'
        return s

# General error Injector definition
class ErrorInjector:
    def __init__(self, model, ranking: Type['Ranking'], quantization: 'QuantizationScheme', 
                 ber_range = [1e-4, 1e-1], protection_range = [0.0, 0.2, 0.4, 0.6, 0.8], 
                 **kwargs):
        """."""
        super().__init__()

        # Save elements into structure
        self.model = model 
        self.protection_range = protection_range
        self.ber_range = ber_range
        self.ranking = ranking
        self.quantization = quantization

        # Initialize masks 
        self._mask_deltas = []
        self._mask_susceptibility = []

        # Compute masks 
        self.deltas, self.susceptibility, self.errors, self.w_idx_mapper = self.compute_masks(self.model, **kwargs)

        # Init cloned model and noise 
        self.cloned_model = None
        self.num_affected_bits = None

    """ Compute masks of susceptibility and delta error """
    def compute_masks(self, model, verbose = True):
        """ There are two things we need here: 

                1) We need a mask that will tell us WHICH weights are susceptible. 
                    In other words, we want matrices with the same dimensions as each
                    weight on each layer, with 0s and 1s, which we will multiply by 
                    the "delta" caused by bit flips. Thus, a "1" in this matrix 
                    will effectively apply the bit flip, while a "0" will leave the
                    weight unaffected. 

                2) We need matrices (with the same size as each weight matrix + (nbits,)) with
                    the delta that a flip on each weight/bit would cause. 
        """
        # Compute the delta for each variable
        variables = model.trainable_variables
        # Get variable names (all, for pruned masks)
        vnames = [v.name for v in model.variables]

        # Get the mapper from name to index (to speedup forward pass later)
        w_idx_mapper = {v.name: iv for iv, v in enumerate(self.model.trainable_variables)}

        # Get quantizer for simplicity
        Q = self.quantization

        # Init vars 
        deltas = {}
        susceptibilities = {}
        errors = {}

        # Compute delta for each variable 
        for var in variables:
            # First quantize 
            var_q = Q(var)

            # Just for fun we can measure the error in quantization
            error = np.mean((var - var_q)**2)

            # Get the delta for this variable
            delta = Q.compute_delta_matrix(var_q)

            # Get the susceptibility matrix for this variable
            # this is, the matrix of which bits can be flipped 
            # for instance, pruned weights cannot be flipped.
            # try to get the pruned mask for this variable
            pruned_vname = var.name.replace(':', '_prune_mask:')
            if pruned_vname in vnames:
                var_pruned = model.variables[vnames.index(pruned_vname)]
                # Get the susceptibility matrix
                susceptibility = (var_pruned != 0)
            else:
                # fallback to checking the value itself
                susceptibility = (var != 0) # IMPORTANT, NOTE THAT WE HAVE TO USE var HERE and NOT VAR_Q, BECAUSE THE QUANTIZATION CAN BE INDEED 0 EVEN FOR NON-PRUNED WEIGHTS.

            # Add to structure 
            deltas[var.name] = delta
            susceptibilities[var.name] = susceptibility
            errors[var.name] = error  

        if verbose: 
            err = "\n".join(list(map(lambda x: "\t\t" + f"{x[0]}: {x[1]}", errors.items())))
            err = '\n\t' + " Quantization errors are:\n" + err
            netsurf.utils.log._info(f'[INFO] - Computed deltas for {len(deltas)} variables.{err}')     

        return deltas, susceptibility, errors, w_idx_mapper
        


    def build_injection_models(self, combs, num_reps, verbose = True, **kwargs):
        """ At this point we know the protection range, we know the ber range and we know the number of reps
            This means that we can pre-compute our attack matrices N. 
            The way we will do this is by creating a single attrack matrix N per variable, where every bit 
            will be assigned a number between -1 and num_reps -1. 
            This number represents the iteration/rep number at which this bit will be flipped. 
            -1 means: no flip.
            0 means: flip at first rep
            ...
            num_reps-1 means: flip at last rep

            If num_reps = -1, we will compute the number of reps we need to do to fill all the space,
            that is, so all flips are flipped at least one. This is kinda like cross-validation
        """

        # Get quantizer
        Q = self.quantization

        # Get all the parameters (param_num)
        param_num = self.ranking['global_param_num'].to_numpy()
        # This is the order in which we will protect our bits. 
        # Now, for each case in comb, we know how many of the top bits we will protect
        # and thus, how many will be left unprotected for us to flip. 

        # Init the attack matrix 
        attack = np.full((len(combs), int(len(param_num)/Q.m), Q.m), fill_value = -1, dtype = np.int32)
        stats = {'protection': [], 'ber': [], 'num_attacks': [], 'bits_protected': [], 
                    'bits_susceptible': [], 'bits_flipped': [], 'true_ber': []}
        for ip, ((protection, ber), already_run_reps) in enumerate(combs.items()):
            # get the number of bits we will be protecting 
            # (remember that pruned weights are not considered)
            non_pruned = self.ranking[self.ranking['pruned'] == False]
            num_bits_protected = np.floor(len(non_pruned)*protection).astype(int)
            # Num bits susceptible
            num_bits_susceptible = len(non_pruned) - num_bits_protected

            # Now, let's see how many bits we will flip per attack (ber)
            num_flips_per_attack = np.round(num_bits_susceptible*ber).astype(int)

            # Now compute how many reps we need to fill all the space
            subnum_reps = (num_reps - already_run_reps) if num_reps != -1 else -1
            if num_reps == -1:
                subnum_reps, num_flips_per_attack = netsurf.core.experiments.get_num_reps_crossvalidation(len(non_pruned), ber, protection, factor = 1)
                subnum_reps -= already_run_reps
                subnum_reps = np.maximum(subnum_reps, 0)

            num_susceptible = len(non_pruned) - num_bits_protected

            # Now we can create the attack matrix (initialize to -1, meaning to attack)
            for irep, i in enumerate(range(0, int(np.floor(subnum_reps*num_flips_per_attack)), num_flips_per_attack)):
                # Everytime we cover all parameters (all susceptible bits), we need to randomize the order

                if (i % num_susceptible) == 0:
                    # Get the ranking for the susceptible bits
                    susceptible = non_pruned.iloc[num_bits_protected:]
                    # Shuffle
                    susceptible = susceptible.sample(frac = 1)
                
                # We need to make sure we alter "i" so it wraps around the length of the susceptible bits
                wrapped_i = (i % num_susceptible)
                
                attack_indices = susceptible.iloc[wrapped_i:wrapped_i+num_flips_per_attack]['global_param_num'].to_list()
                # Get the bit indices too 
                bit_indices = susceptible.iloc[wrapped_i:wrapped_i+num_flips_per_attack]['bit'].to_list()
                # Bit indices should be transformed into actual array indices. 
                # This is the difference:
                #
                #       [S]X. x  x  x  x  x
                # bit   [1]0.-1 -2 -3 -4 -5
                # index [0]1. 2  3  4  5  6
                #
                # bit   [][].-1 -2 -3 -4 -5 -6
                # index [][]. 0  1  2  3  4  5
                #
                # Note the difference? 
                array_bit_indices = list(abs(np.array(bit_indices) - Q.n + (1-Q.s)))

                # Both these indices together represent the places where we will flip the bits
                # Set the attack indices
                attack[ip, attack_indices, array_bit_indices] = irep + already_run_reps
            
            # Update stats 
            stats['protection'].append(protection)
            stats['ber'].append(ber)
            stats['num_attacks'].append(subnum_reps)
            stats['bits_protected'].append(num_bits_protected)
            stats['bits_susceptible'].append(num_bits_susceptible)
            stats['bits_flipped'].append(num_flips_per_attack)
            stats['true_ber'].append(num_flips_per_attack/num_bits_susceptible)

        # Transform into pd.DataFrame
        stats = pd.DataFrame(stats)

        # Now that we have the attacks, we need to shape them into the actual size of each variable
        attacks = {}
        cum_param = 0
        for iv, v in enumerate(self.model.trainable_variables):
            # Get the variable
            vname = v.name
            # Get the deltas
            delta = self.deltas[vname][None]
            # Get the indices of the variable
            num_params = np.prod(v.shape)
            subattack = attack[:, cum_param:cum_param + num_params]
            # Reshape attack to shape of delta
            subattack = np.reshape(subattack, (subattack.shape[0], *delta.shape[1:]))
            # this is our attack (we cannot apply the mask here cause if we 
            # have too many num_reps this will blow up memory)
            attacks[vname] = subattack

            # Update cum_param
            cum_param += num_params
        
        # Convert to AttackScheme
        attacks = AttackContainer(attacks, stats)

        # Create clone model
        self.delta_model = self.model.clone()

        # We are done
        return attacks

    """ Evaluate metrics """
    def evaluate(self, ytrue, ypred, metric_fcns = {}, **kwargs):
        # Get the loss 
        # broadcast 
        # Check if y_true is 1D and y_pred is 2D with a last dimension of 1
        if ytrue.ndim == 1 and ypred.ndim == 2 and ypred.shape[-1] == 1:
            ypred = np.squeeze(ypred, axis=-1)
        loss = self.model.loss(ytrue, ypred).numpy()
        if np.ndim(loss) > 0:
            if loss.shape[0] == ytrue.shape[0]:
                loss = np.mean(loss)
        # Evaluate metrics
        vals = {'loss': loss}
        for mname, mfcn in metric_fcns.items():
            mval = mfcn(ytrue, ypred).numpy()
            vals[mname] = mval
            mfcn.reset_states()
        return vals

    # Forward pass 
    def __call__(self, X, attack: Type['Attack'] = None, batch_size = None, verbose = False, **kwargs):
        """ if attack is not None, get the delta for this model and apply the attack matrix N """
        # Do the forward pass
        # Chck if X is a image generator
        if isinstance(X, keras.preprocessing.image.DirectoryIterator):
            num_samples = np.minimum(2,len(X))
            # Loop thru batches
            y = []
            for i in range(num_samples):
                Xb, Yb = X.next()
                yb = self.model.attack(Xb, N = attack, clamp = True, **kwargs) # Clamp here makes sure the values at the output of the applyalpha layer are clamped within quantization range
                y.append(yb)
            # Turn into numpy array
            y = np.concatenate(y, axis = 0)
        else:
            if batch_size:
                num_batches = np.ceil(len(X)/batch_size).astype(int)
                y = []
                for i in range(num_batches):
                    if verbose and i%10 == 0: print(f'[INFO] - Processing batch {i}/{num_batches}')
                    Xb = X[i*batch_size:(i+1)*batch_size]
                    yb = self.model.attack(Xb, N = attack, clamp = True, **kwargs) # Clamp here makes sure the values at the output of the applyalpha layer are clamped within quantization range
                    y.append(yb)
                y = np.concatenate(y, axis = 0)
            else:
                y = self.model.attack(X, N = attack, clamp = True, **kwargs)
        
        return y
    

class BitFlipInjector:
    def __init__(self, model, ranking: Type['Ranking'], num_samples = 100,
                 ber_range = [1e-4, 1e-1], protection_range = [0.0, 0.8], **kwargs):

        # Save elements into structure
        self.protection_range = protection_range
        self.ber_range = ber_range
        self.ranking = ranking
        self.model = model

        lb = (np.min(ber_range), np.min(protection_range))
        ub = (np.max(ber_range), np.max(protection_range))

        self.lhc_sampler = netsurf.utils.LatinHypercubeSampler(num_samples = num_samples, lower_bounds=lb, upper_bounds=ub)

        # Now we can apply the ranking to the model 
        self.apply_ranking(self.model, ranking)

    def sample(self, num_samples = 100):
        """ Sample the space of protection and ber """
        # Get the samples
        samples = self.lhc_sampler.sample(num_samples)
        # Get the protection and ber
        protection = samples[:, 1]
        ber = samples[:, 0]
        # Return as a tuple
        return list(zip(protection, ber))
    
    @staticmethod
    def reset_bit_flip_variables(model):
        """ Reset the bit flip variables to the reset state. 
            This is done by creating a new model with the same architecture
            but with the weights of the original model. 
            The weights are then sorted according to the ranking.
        """
        # Loop thru variables 
        for i, v in enumerate(model.variables):
            # Get the variable name
            vname = v.name
            # Get the variable shape
            vshape = v.shape   
            if "_N:" in vname or '_ranking:' in vname:
                # Assign to zero        
                model.variables[i].assign(tf.zeros(vshape, dtype = tf.float32))
                netsurf.info(f'Bit flip variable {vname} reset to zero')
        
        # Now reset the bit_flip counters
        if hasattr(model, 'reset_bit_flip_counter'):
            model.reset_bit_flip_counter()
            netsurf.info(f'Bit flip counters reset to zero using the method method "reset_bit_flip_counter"')
        else:
            # Loop thru layers and try to find methods
            for layer in model.layers:
                if hasattr(layer, 'bit_flip_counter'):
                    layer.bit_flip_counter.assign(0)
                    netsurf.info(f'Bit flip counter for layer {layer.name} reset to zero')


    @staticmethod
    def apply_ranking(model, ranking):
        """ Apply the ranking to the model. 
            This is done by creating a new model with the same architecture
            but with the weights of the original model. 
            The weights are then sorted according to the ranking.
        """
        # Before applying the ranking, we need to set the noise bit-flip variables to the reset state
        # kernel_N/bias_N/alpha_N/beta_N/gamma_N/beta_N --> 0
        # kernel_ranking/... -> 1 (means unprotected, last in the ranking)
        # bit_flip counters -> 0
        # prune masks -> 0 (# NO, CAUSE WE MIGHT HAVE PRUNED WEIGHTS)!!!!!
        BitFlipInjector.reset_bit_flip_variables(model)

        # Get the bits
        bits = sorted(ranking['bit'].unique())[::-1]
        
        Q = model.quantizer
        total_num_params = len(ranking)
        
        # Initialize a dict of variables per name 
        var_dict = {v.name: i for i, v in enumerate(model.variables)}

        # Loop thru variables 
        for i, v in enumerate(model.trainable_variables):
            # Get the variable name
            vname = v.name
            # Get the variable shape
            vshape = v.shape            
            # Loop thru bits 
            w_bits = []
            for b in bits:
                # Get the subdf that has this vname 
                subdf = ranking[(ranking['param'] == vname) & (ranking['bit'] == b)]
                # Sort by internal_param_num
                subdf = subdf.sort_values(by = 'internal_param_num')
                # Rank is subdf['rank'] + (b-Q.n-Q.s+1)/Q.m
                subrank = subdf['rank'].to_numpy() + (b - Q.n - Q.s + 1)/Q.m
                # Reshape to variable shape
                subrank = subrank.reshape(vshape)
                # Normalize by total number of vars 
                subrank = subrank/(total_num_params + (Q.m-1)/Q.m)
                w_bits.append(tf.cast(subrank, v.dtype))
                
            # Concat over the bits axis 
            subrank = tf.stack(w_bits, axis = len(vshape))
            # Now we need to find the variable with name "vname_ranking"
            rname = vname.replace(':', '_ranking:')
            # Check if it exists in var_dict
            if rname in var_dict:
                # Get the variable
                ridx = var_dict[rname]
                rvar = model.variables[ridx]
                # Check if the shape is correct
                if rvar.shape != subrank.shape:
                    raise ValueError(f'Variable {rname} has shape {rvar.shape} but subrank has shape {subrank.shape}')
                # Set the value
                model.variables[ridx].assign(subrank)
                # Print info
                netsurf.info(f'Applying ranking to {vname} with shape {vshape} and rank {subrank.shape}')
            else:
                netsurf.error(f'Variable {rname} not found in model variables. This is probably because the model was not built with the ranking in mind. Please check the model architecture and make sure it has a variable with name {rname}')
        # Done 
        netsurf.info(f'Applied ranking to model {model.name} with {len(model.trainable_variables)} variables')

    def __call__(self, X, BER: float = 0.0, protection: float = 0.0, batch_size = None, verbose = False, **kwargs):
        """ if attack is not None, get the delta for this model and apply the attack matrix N """
        if isinstance(X, keras.preprocessing.image.DirectoryIterator):
            num_samples = np.minimum(2,len(X))
            # Loop thru batches
            y = []
            for i in range(num_samples):
                Xb, Yb = X.next()
                yb = self.model.inject(Xb, BER = BER, protection = protection, clamp = True, **kwargs) # Clamp here makes sure the values at the output of the applyalpha layer are clamped within quantization range
                y.append(yb)
            # Turn into numpy array
            y = np.concatenate(y, axis = 0)
        else:
            if batch_size:
                num_batches = np.ceil(len(X)/batch_size).astype(int)
                y = []
                for i in range(num_batches):
                    if verbose and i%10 == 0: print(f'[INFO] - Processing batch {i}/{num_batches}')
                    Xb = X[i*batch_size:(i+1)*batch_size]
                    yb = self.model.inject(Xb, BER = BER, protection = protection, clamp = True, **kwargs) # Clamp here makes sure the values at the output of the applyalpha layer are clamped within quantization range
                    y.append(yb)
                y = np.concatenate(y, axis = 0)
            else:
                y = self.model.inject(X, BER = BER, protection = protection, clamp = True, **kwargs)
        
        return y
        
    """ Evaluate metrics """
    def evaluate(self, ytrue, ypred, metric_fcns = {}, **kwargs):
        # Get the loss 
        # broadcast 
        # Check if y_true is 1D and y_pred is 2D with a last dimension of 1
        if ytrue.ndim == 1 and ypred.ndim == 2 and ypred.shape[-1] == 1:
            ypred = np.squeeze(ypred, axis=-1)
        loss = self.model.loss(ytrue, ypred).numpy()
        if np.ndim(loss) > 0:
            if loss.shape[0] == ytrue.shape[0]:
                loss = np.mean(loss)
        # Evaluate metrics
        vals = {'loss': loss}
        for mname, mfcn in metric_fcns.items():
            mval = mfcn(ytrue, ypred).numpy()
            vals[mname] = mval
            mfcn.reset_states()
        return vals
    
            


