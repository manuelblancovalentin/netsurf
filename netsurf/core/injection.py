""" Injectors of errors """
from typing import Type

# Modules 
import re 
import numpy as np
import copy

from tensorflow import keras
from keras import backend as K # To get intermediate activations between layers

""" Qkeras """
import qkeras 
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

""" Pandas """
import pandas as pd

""" Custom modules """
import wsbmr

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
    def __init__(self, model, ranker: Type['WeightRanker'], quantization: 'QuantizationScheme', 
                 ber_range = [1e-4, 1e-1], protection_range = [0.0, 0.2, 0.4, 0.6, 0.8], 
                 **kwargs):
        """."""
        super().__init__()

        # Save elements into structure
        self.model = model 
        self.protection_range = protection_range
        self.ber_range = ber_range
        self.ranker = ranker
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
            susceptibility = (var != 0) # IMPORTANT, NOTE THAT WE HAVE TO USE var HERE and NOT VAR_Q, BECAUSE THE QUANTIZATION CAN BE INDEED 0 EVEN FOR NON-PRUNED WEIGHTS.

            # Add to structure 
            deltas[var.name] = delta
            susceptibilities[var.name] = susceptibility
            errors[var.name] = error  

        if verbose: 
            err = "\n".join(list(map(lambda x: "\t\t" + f"{x[0]}: {x[1]}", errors.items())))
            err = '\n\t' + " Quantization errors are:\n" + err
            wsbmr.utils.log._info(f'[INFO] - Computed deltas for {len(deltas)} variables.{err}')     

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
        # Get the ranking 
        ranking = self.ranker.df

        # Get quantizer
        Q = self.quantization

        # Get all the parameters (param_num)
        param_num = ranking['global_param_num'].to_numpy()
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
            non_pruned = ranking[ranking['pruned'] == False]
            num_bits_protected = np.floor(len(non_pruned)*protection).astype(int)
            # Num bits susceptible
            num_bits_susceptible = len(non_pruned) - num_bits_protected

            # Now, let's see how many bits we will flip per attack (ber)
            num_flips_per_attack = np.round(num_bits_susceptible*ber).astype(int)

            # Now compute how many reps we need to fill all the space
            subnum_reps = (num_reps - already_run_reps) if num_reps != -1 else -1
            if num_reps == -1:
                subnum_reps, num_flips_per_attack = wsbmr.core.experiments.get_num_reps_crossvalidation(len(non_pruned), ber, protection, factor = 1)
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
        
        
        
        # self.model.attack(X, N = attack)
        
        
        
        # S = {}
        # if attack is not None:
        #     # Loop thru all the variables
        #     for vname in attack.N:
        #         # Get the index of the variable
        #         w_idx = self.w_idx_mapper[vname]
        #         # Get the weight
        #         w = self.delta_model.trainable_weights[w_idx]
        #         # Get the delta for this attack from the model 
        #         delta = self.deltas[vname]
        #         # Apply the attack to this layer 
        #         S[vname] = (attack.N[vname] * delta).sum(-1)
        #         # Add the delta
        #         w.assign(w + S[vname])
        
        # # Do the forward pass
        # # Chck if X is a image generator
        # if isinstance(X, keras.preprocessing.image.DirectoryIterator):
        #     num_samples = np.minimum(2,len(X))
        #     # Loop thru batches
        #     y = []
        #     for i in range(num_samples):
        #         Xb, Yb = X.next()
        #         yb = self.delta_model.predict(Xb, **kwargs)
        #         y.append(yb)
        #     # Turn into numpy array
        #     y = np.concatenate(y, axis = 0)
        # else:
        #     y = self.delta_model.predict(X, **kwargs)

        # # Undo the attack
        # if attack is not None:
        #     for vname in attack.N:
        #         # Get the index of the variable
        #         w_idx = self.w_idx_mapper[vname]
        #         # Get the weight
        #         w = self.delta_model.trainable_weights[w_idx]
        #         # Get the delta for this attack from the model 
        #         delta = self.deltas[vname]
        #         # Apply the attack to this layer 
        #         w.assign(w - S[vname])
    
        return y
            


    # def _OLD_compute_masks(self, verbose = True):
    #     """ Compute the masks for the error injection """
    #     """
    #         1) COMPUTATION OF THE DELTA TO BE ADDED TO EACH WEIGHT, ACCORDING TO THEIR INDIVIDUAL BIT VALUE (0 or 1)
    #     """
    #     # Build array with delta per bit (0->1 vs 1->0)
    #     num_bits = self.quantization.m

    #     # Get df from ranker 
    #     df = self.ranker.df

    #     """
    #         2) COMPUTATION OF THE DELTA TO BE ADDED TO EACH WEIGHT, ACCORDING TO THEIR INDIVIDUAL BIT VALUE (0 or 1)
    #     """

    #     # Get supported weights and pruned masks 
    #     results = wsbmr.models.get_supported_weights(self.model.model, numpy = False, pruned = True, verbose = False)
    #     supported_weights, supported_pruned_masks, supported_layers, weights_param_num = results
        
    #     if verbose:
    #         print(f'[INFO] - Found a total of {len(supported_weights)} supported weights: {", ".join(list(supported_weights.keys()))}')

    #     # Get layer index per layer in supported_layers
    #     supported_layers_idxs = {lname: self.model.model.layers.index(supported_layers[lname]) for lname in supported_layers}

    #     # Get deltas per weight
    #     deltas = {kw: wsbmr.models.get_deltas(kv, num_bits = num_bits) for kw, kv in supported_weights.items()}
    #     is_bit_one = {kw: deltas[kw][1] for kw in supported_weights}
    #     deltas = {kw: deltas[kw][0] for kw in supported_weights}

    #     # Build the susceptibility matrix based on PRUNED bits for each layer 
    #     is_susceptible = {kw: np.ones_like(deltas[kw]) for kw in supported_weights}
    #     for kw, w in supported_weights.items():
    #         # Get mask 
    #         msk = supported_pruned_masks[kw]

    #         if msk is not None:
    #             msk = np.reshape(msk, w.shape)
    #             # Remember to add the extra dimension for the bits
    #             msk = np.tile(msk[...,None], num_bits)
    #             is_susceptible[kw] = 1-msk


    #     """
    #         COMPUTATION OF THE MASK THAT TELLS US WHICH WEIGHTS ARE SUSCEPTIBLE TO BIT FLIPS AND WHICH ARE NOT
    #     """

    #     # On top of the weights that have been pruned, we need to mark as NOT-SUSCEPTIBLE, the ones that will be triplicated
    #     # following the protection range. 

    #     # First, get the number of AVAILABLE WEIGHTS (not pruned). Consider from now on that the pruned weights don't exist.
    #     unpruned_df = df[df['pruned'] == False]
    #     total_num_unpruned_bits = len(unpruned_df)

    #     # Pre convert coords into tuples for easy access 
    #     unpruned_df.loc[:,'tuple_coord'] = unpruned_df.apply(lambda x: tuple([int(g) for g in re.findall('\[(\d+)\]', x.coord)]) + (x.bit,), axis = 1)

    #     # Init previous affected masks (so we don't have to compute them every time, cause the protection is increasing)
    #     susceptibility_masks = {0.0: copy.deepcopy(is_susceptible)}
    #     susceptible_indices = {0.0: {kw: np.where(is_susceptible[kw].flatten() == 1)[0] for kw in is_susceptible}}
    #     last_protection = 0.0

    #     for protection in self.protection_range:
    #         if protection > 0:
    #             num_bits_triplicated_so_far = np.floor(total_num_unpruned_bits*last_protection).astype(int)
    #             num_bits_to_triplicate = np.floor(total_num_unpruned_bits*protection).astype(int)

    #             # Get only the first "weights_to_triplicate" weights from the ranked list 
    #             bits_to_triplicate = unpruned_df[num_bits_triplicated_so_far:num_bits_to_triplicate]

    #             # Group by layer and loop thru them 
    #             flat_indices = {}
    #             for ilayer, layer_df in bits_to_triplicate.groupby('layer'):
    #                 # Replace name with actual object name 
    #                 row_name = layer_df.iloc[0]['weight']
    #                 w_name = row_name.split('[')[0].replace('prune_low_magnitude_', '') + '/kernel:0'
    #                 w_name_prune = 'prune_low_magnitude_' + w_name   

    #                 # Get coords
    #                 cs = layer_df['tuple_coord'].to_numpy()
                    
    #                 var_name = None
    #                 if w_name in is_susceptible:
    #                     var_name = w_name
    #                     for c in cs:
    #                         is_susceptible[w_name][c] = 0.0

    #                 elif w_name_prune in is_susceptible:
    #                     var_name = w_name_prune
    #                     for c in cs:
    #                         is_susceptible[w_name_prune][c] = 0.0
                    
    #                 # 
    #                 flat_indices[var_name] = np.where(is_susceptible[var_name].flatten() == 1.0)[0]

    #             # Update susceptible_indices
    #             susceptible_indices[protection] = flat_indices

    #             # Update susceptibility masks
    #             susceptibility_masks[protection] = copy.deepcopy(is_susceptible)

    #             # Update last protection
    #             last_protection = protection

    #     # Now we can get the number of bits that are susceptible to bit flips
    #     num_bits_susceptible_per_var = {kw: {var_name: susceptibility_masks[kw][var_name].sum() for var_name in susceptible_indices[kw]} \
    #         for kw in susceptible_indices}
    #     num_bits_susceptible = {kw: np.sum([len(susceptible_indices[kw][var_name]) for var_name in susceptible_indices[kw]]) \
    #         for kw in susceptible_indices}

    #     """ Now that we have the susceptible weights per protection level and variable, we can compute the number of 
    #         ACTUAL bits we will be flipping (which depends on the ber rate) 
    #     """
    #     # Get the number of unpruned bits per variable
    #     num_unpruned_bits_per_var = {kw: np.sum(1-supported_pruned_masks[kw]) for kw in susceptible_indices[0.0] if 'bias' not in kw}
    #     total_num_unpruned_bits = np.sum(list(num_unpruned_bits_per_var.values()))
        
    #     num_bits_to_flip_per_var = {}
    #     num_bits_to_flip_per_case = {}
    #     true_ber_rates = {}
    #     for protection in self.protection_range:
    #         num_bits_to_flip_per_var[protection] = {}
    #         num_bits_to_flip_per_case[protection] = {}
    #         true_ber_rates[protection] = {}
    #         for ber in self.ber_range:
    #             num_bits_to_flip_per_var[protection][ber] = {}
    #             num_bits_to_flip_per_case[protection][ber] = 0
    #             M = 1
    #             for var_name in susceptible_indices[protection]:
    #                 if 'bias' not in var_name:
    #                     K = np.maximum(M, np.round(num_unpruned_bits_per_var[var_name]*ber).astype(int))
    #                     num_bits_to_flip_per_var[protection][ber][var_name] = K
    #                     num_bits_to_flip_per_case[protection][ber] += K
    #                     if K > 0:
    #                         M = 0
    #             true_ber_rates[protection][ber] = num_bits_to_flip_per_case[protection][ber]/total_num_unpruned_bits


    #     """ Now set stuff into place for error injection """
    #     self.susceptibility_masks = susceptibility_masks
    #     self.susceptible_indices = susceptible_indices
    #     self.true_ber_rates = true_ber_rates
    #     self.num_bits_to_flip_per_var = num_bits_to_flip_per_var
    #     self.num_bits_to_flip_per_case = num_bits_to_flip_per_case

    #     self.num_bits_susceptible_per_var = num_bits_susceptible_per_var
    #     self.num_bits_susceptible = num_bits_susceptible

    #     self.num_unpruned_bits_per_var = num_unpruned_bits_per_var
    #     self.total_num_unpruned_bits = total_num_unpruned_bits

    #     self.is_susceptible = is_susceptible

    #     # Set masks into place in structure
    #     self._mask_deltas = deltas
    #     self._mask_susceptibility = is_susceptible
    #     self._supported_weights = supported_weights 
    #     self._supported_pruned_masks = supported_pruned_masks 
    #     self._supported_layers = supported_layers 
    #     self._weights_param_num = weights_param_num


    # """ Create noise and compile injected model (model with noise) - This is to speed up computation """
    # def OLD_build_injection_models(self, num_reps, verbose = True, **kwargs):
        
    #     # Get name of the weights in the model
    #     w_model = [w.name for w in self.model.model.trainable_weights]
    #     # Get the indexes of each variable in susceptible_indices in w_model for easy access later 
    #     w_idx_mapper = {w_name: w_model.index(w_name) for w_name in self.susceptible_indices[0.0] if w_name in w_model}

    #     """ First of all, let's create a number of NUM_REPS deltas for each protection level and ber rate """
    #     deltas = {}
    #     selectors = []
    #     for iprotection, protection in enumerate(self.protection_range):
    #         deltas[protection] = {}
    #         for iber, ber in enumerate(self.ber_range):
    #             deltas[protection][ber] = [{var_name: np.zeros_like(self._mask_deltas[var_name][...,0]) \
    #                     for var_name in self.susceptible_indices[protection]} \
    #                         for _ in range(num_reps)]
    #             # Append case to selector
    #             selectors += [{'protection': protection, 'protection_idx': iprotection,
    #                             'ber': ber, 'ber_idx': iber, 
    #                             'rep': rep} for rep in range(num_reps)]
    #             # Get indices for susceptible bits
    #             for var_name in self.susceptible_indices[protection]:
    #                 if 'bias' not in var_name:
    #                     # Get affected indices
    #                     affected_idxs = self.susceptible_indices[protection][var_name]
    #                     # Get number of bits to flip
    #                     num_bits_to_flip = self.num_bits_to_flip_per_var[protection][ber][var_name]
    #                     # Get a random permutation
    #                     random_idxs = np.random.permutation(affected_idxs)
    #                     # # Get susceptibility mask (We don't need it, this info is implicit in affected_idxs)
    #                     # S = self.susceptibility_masks[protection][var_name]
    #                     # Get mask delta
    #                     D = self._mask_deltas[var_name]
    #                     # Loop thru reps
    #                     for rep in range(num_reps):
    #                         # Get subset of random indexes
    #                         tmp_idxs = random_idxs[(num_bits_to_flip*rep):(num_bits_to_flip*(rep+1))]
    #                         # Create flatten noise vector
    #                         affected_flat = np.zeros(np.prod(D.shape))
    #                         # Set affected bits to 1
    #                         affected_flat[tmp_idxs] = 1
    #                         # Unravel to original shape
    #                         affected = np.reshape(affected_flat, D.shape)
    #                         # Get total delta
    #                         delta_tmp = (D*affected).sum(axis=-1)
    #                         # Set in place
    #                         deltas[protection][ber][rep][var_name] = delta_tmp

    #     """ Let's clone our original model before modifying the weights """
    #     # Duplicate the original model
    #     co = {}
    #     qkeras.utils._add_supported_quantized_objects(co)
    #     co['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude

    #     delta_model = qkeras.utils.clone_model(self.model.model, custom_objects=co)
    #     delta_model = strip_pruning(delta_model)

    #     """ Recompile model after changing weights """
    #     delta_model.compile(loss = self.model.loss, 
    #             optimizer=self.model.optimizer, 
    #             metrics=self.model.metrics)
        
    #     # Do the same for the activation model 
    #     activation_model = qkeras.utils.clone_model(self.model.activation_model, custom_objects=co)
    #     activation_model = strip_pruning(activation_model)

    #     # Recompile 
    #     activation_model.compile(loss = self.model.loss,
    #             optimizer=self.model.optimizer, 
    #             metrics=self.model.metrics)

    #     """ Set everything into place """
    #     self.activation_model = activation_model


    #     self.delta_model = delta_model
    #     self.deltas = deltas
    #     self.w_idx_mapper = w_idx_mapper

    #     # Now, to facilitate everything, let's create some wrapper functions that we can pass to the user, so they can 
    #     # call "forward_with_inject" like an iterator, without having to think about selecting each case and protection level
    #     return selectors
    

    # """ Inject random bit flips during forward pass """
    # def forward_with_inject(self, X, Y = None, verbose = True, protection = 0.0, ber = None, rep = 0, 
    #                         protection_idx = None, ber_idx = None, 
    #                         metric_fcns = {},
    #                         **kwargs):

    #     """ Init activations (ground truth), deltas (noise), and altered_activations """     
    #     # Last activation is output
    #     #y_gt = self.model.model(X).numpy())

    #     # Assert ber is not None
    #     if ber is None:
    #         raise ValueError('BER rate must be provided')

    #     # Select the delta 
    #     delta = self.deltas[protection][ber][rep]

    #     # Change weights in cloned model 
    #     for var_name in delta:
    #         # Get weight index
    #         w_idx = self.w_idx_mapper[var_name]
    #         # Get weight
    #         w = self.delta_model.trainable_weights[w_idx]
    #         # Add delta
    #         w.assign(w + delta[var_name])
    
    #     # Chck if X is a image generator
    #     if isinstance(X, keras.preprocessing.image.DirectoryIterator):
    #         num_samples = np.minimum(2,len(X))
    #         # Loop thru batches
    #         y = []
    #         for i in range(num_samples):
    #             Xb, Yb = X.next()
    #             yb = self.delta_model.predict(Xb, verbose = verbose, **kwargs)
    #             y.append(yb)
    #         # Turn into numpy array
    #         y = np.concatenate(y, axis = 0)
    #     else:
    #         y = self.delta_model.predict(X, verbose = verbose, **kwargs)

    #     # Undo changes in cloned model
    #     for var_name in delta:
    #         # Get weight index
    #         w_idx = self.w_idx_mapper[var_name]
    #         # Get weight
    #         w = self.delta_model.trainable_weights[w_idx]
    #         # Add delta
    #         w.assign(w - delta[var_name])

    #     # If this is a regression problem, compute r2 as "accuracy"
    #     if Y.ndim == 1:
    #         if y.ndim > 1:
    #             if self.model.type == 'classification':
    #                 y = np.argmax(y, axis = 1)
    #             elif self.model.type == 'regression':
    #                 y = y.flatten()
        
    #     metrics = self.evaluate_metrics(Y, y, metric_fcns)

    #     # Calculate actual radiation
    #     #tot_flipped = np.sum([np.sum(self.affected_per_var[kw]) for kw in self.affected_per_var if 'kernel' in kw])
    #     #tot_w = np.sum([self.num_affected_bits[kw] for ikw, kw in enumerate(self.num_affected_bits) if 'kernel' in kw])
    #     #actual_radiation = tot_flipped/tot_w
    #     actual_radiation = self.true_ber_rates[protection][ber]

    #     return y, metrics, actual_radiation


