""" numpy """
import numpy as np

""" tensorflow """
import tensorflow as tf

""" Attack class """
class Attack:
    def __init__(self, variables, N = None):
        # If variables is not a dict, turn into one
        if not isinstance(variables, dict):
            variables = {v.name: v for v in variables}

        # If N is not a dict, turn into one
        if not isinstance(N, dict):
            N = {k: N for k in variables}
        
        # Now parse and initialize 
        N = {k: self._parse_N_(N[k], variables[k].shape) for k in variables}

        # Finally, set each N as an attribute
        for k in variables:
            setattr(self, k, N[k])

        # if N is a number between 0 and 1, this is the probability of each element being 1
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
        

""" VulnerabilityGroup """
# Because our method requires us to gather vulnerable objects by pairs, so we can 
# "simplify" them, we need to define a class that will hold these pairs. 
# Here's a simple example with 4 dense layers (no activation in between), of what these
# hierarchical pairs would look like:
#
# Model definition
# X -> [fc0] -> [fc1] -> [fc2] -> [fc3] -> Y
#
# Now the groups would be: 
#
# X -> { { {fc0}, fc1 }, fc2 }, fc3 } -> Y
class VulnerabilityGroup:
    def __init__(self, index=()):
        self.index = index
        self.children = []

    def __repr__(self):
        return self.print_hierarchy()

    def print_hierarchy(self, level=0):
        """Print the hierarchical structure for debugging with indexing"""
        indent = "  " * level
        s = f'{indent}{self.index}: VulnerabilityGroup <{self.index}>\n'
        for i, c in enumerate(self.children):
            child_index = self.index + (i,)
            if isinstance(c, VulnerabilityGroup):
                c.index = child_index  # Update the index dynamically
                s += c.print_hierarchy(level=level + 1)
            else:
                s += f'{indent}  - {child_index}: {c}\n'
        return s

    def append(self, child):
        self.children.append(child)

    @staticmethod
    def from_list(lst, index=(0,)):
        """Recursively builds the nested group structure based on occurrences of instances of A"""
        num_As = sum(isinstance(x, DenseLayerVulnerability) for x in lst)

        if num_As <= 1:
            return lst

        first_A = next(i for i, x in enumerate(lst) if isinstance(x, DenseLayerVulnerability))
        root = VulnerabilityGroup(index=index)

        # Add elements before the first 'A'
        for i, l in enumerate(lst[:first_A]):
            root.append(l)

        # Initialize the first group after the first 'A'
        current_group = VulnerabilityGroup(index=index + (len(root.children),))
        root.append(current_group)

        count_A = num_As
        i = first_A
        while i < len(lst):
            item = lst[i]
            if isinstance(item, DenseLayerVulnerability):
                count_A -= 1
                if count_A > 0:  # Create a new group and append 'A' into it
                    new_group = VulnerabilityGroup(index=current_group.index + (len(current_group.children),))
                    new_group.append(item)
                    current_group.append(new_group)
                    current_group = new_group
                else:  # Last 'A' remains as an element
                    current_group.append(item)
            else:
                current_group.append(item)
            i += 1

        return root

    def forward(self, y_0, epsilon_0 = None, **kwargs):
        for ic, c in enumerate(self.children):
            if isinstance(c, VulnerabilityGroup):
                y_0, epsilon_0 = c.forward(y_0, epsilon_0, **kwargs)
            else:
                y_0, epsilon_0 = c.forward(y_0, epsilon_0, **kwargs)
        return y_0, epsilon_0

    def backward(self, grad_epsilon, y_1):
        # The backward always goes in reverse order, so we first need to get
        # to the bottom of the hierarchy, and at the last group, start by the last element
        for ic, c in enumerate(self.children[::-1]):
            grad_epsilon, y_1 = c.backward(grad_epsilon, y_1)
        return grad_epsilon, y_1
    
    def __call__(self, X, Y = None):
        # First of all, we need to do the forward pass, while storing the intermediate results for each group 
        # in the hierarchy
        Y, epsilon = self.forward(X, epsilon_0 = None, Y=Y)
        
        # Given the epsilon, we can now check the global vulnerability per output
        # So, reshape to (-1, output_dim)
        f_epsilon = tf.reshape(epsilon, (-1, epsilon.shape[-1]))
        pos_sum = np.sum(np.maximum(f_epsilon,0), axis = 0)
        neg_sum = np.sum(np.maximum(-f_epsilon,0), axis = 0)

        d_eps = np.sign(pos_sum - neg_sum)

        # if pos_sum > neg_sum:
        #     deps = tf.cast((epsilon > 0), tf.float32)
        # else:
        #     deps = tf.cast((epsilon < 0), tf.float32)

        # Plot the distribution of the epsilon
        if False:
            import matplotlib.pyplot as plt 

            fig, axs = plt.subplots(1,1, figsize = (10, 5))
            # Histogram manually 
            bins = np.linspace(0, max(np.max(epsilon), -np.min(epsilon)), 100)
            # Positive part 
            counts_pos, edges_pos = np.histogram(epsilon[epsilon > 0], bins = bins)
            # Negative part
            counts_neg, edges_neg = np.histogram(-epsilon[epsilon < 0], bins = bins)

            

            # Plot 
            axs.bar(edges_pos[:-1], counts_pos, width = edges_pos[1] - edges_pos[0], color = 'g', alpha = 0.5, label = f'(+) Total: {pos_sum}')
            axs.bar(edges_neg[:-1], -counts_neg, width = edges_neg[1] - edges_neg[0], color = 'r', alpha = 0.5, label = f'(-) Total: {neg_sum}')
            axs.legend()

            axs.set_title('Epsilon distribution')

            # Activate major and minor grid
            axs.grid(which='major', linestyle='-', linewidth=0.5)
            axs.grid(which='minor', linestyle=':', linewidth=0.5)
            axs.minorticks_on()

            plt.show()

        
        # Now backward
        Y = self.backward(d_eps, Y)
        
        
        

""" Vulnerability for specific layers (so we can perform forward and backpass correctly) """
class LayerVulnerability:
    def __init__(self, layer):
        # Keep layer reference
        self.layer = layer
        # Because we are gonna wrap every layer with this LayerVulnerability, we need 
        # to set anything that doesn't have a specific vulnerability class, to unvulnerable
        # not because it's not really vulnerable, but as an indicator that in our vulnerability
        # propagation pass, it will act as if it was invulnerable.
        self.vulnerable = False
        # Init epsilon_0 and y_0 (from the previous layer)
        self.epsilon_0 = None
        self.y_0 = None
    
    @staticmethod
    def from_type(layer):
        if layer.__class__.__name__ == 'QQDense':
            return DenseLayerVulnerability(layer)
        elif layer.__class__.__name__ == 'QQSoftmax':
            return SoftmaxLayerVulnerability(layer)
        return LayerVulnerability(layer)


    ################################################
    #                +-------------+
    #        y_0 --> |             | --> y_1
    #                | ???         |
    #  epsilon_0 --> |             | --> epsilon_1
    #                +-------------+
    ################################################
    def forward(self, y_0, epsilon_0, **kwargs):
        # Compute uncorrupted output
        y_1 = self.layer(y_0) # This is the uncorrupted output (for the uncorrupted input)
        # Here epsilon_1 is assumed to be the same as epsilon_0, but we will override this in the subclasses
        epsilon_1 = epsilon_0
        # Set in place
        self.y_0 = y_0
        self.epsilon_0 = epsilon_0
        return y_1, epsilon_1
    
    def backward(self, grad_epsilon, y_1):
        # Compute the gradient of the loss with respect to the input
        #grad_input = tf.gradients(y_1, self.y_0, grad_epsilon)
        #return grad_input[0]
        return grad_epsilon, y_1


""" Dense """
class DenseLayerVulnerability(LayerVulnerability):
    def __init__(self, layer):
        super().__init__(layer)
        # Make sure to set vulnerable to True
        self.vulnerable = True
        # Init epsilon_0 and y_0 (from the previous layer)
        self.epsilon_0 = None
        self.y_0 = None
        self.kernel_N = None
        self.kernel_P = None
    
    ################################################
    #                +-------------+
    #        y_0 --> |             | --> y_1
    #                | W, delta, N |
    #  epsilon_0 --> |             | --> epsilon_1
    #                +-------------+
    ################################################
    def forward(self, y_0, epsilon_0, **kwargs):
        # Compute uncorrupted output
        y_1 = self.layer(y_0) # This is the uncorrupted output (for the uncorrupted input)
        
        # Compute corruption from this layer (N=1)
        S_d = tf.reduce_sum(self.layer.kernel_delta, axis=-1)
        W_d = self.layer.kernel + S_d
        
        # if epsilon_0 is None, init to zeros with the same shape as y_0
        if epsilon_0 is None:
            epsilon_0 = tf.zeros_like(y_0)

        #epsilon_1 = self.layer.attack(epsilon_0, N=1)
        epsilon_1 = tf.matmul(epsilon_0, W_d) + tf.matmul(y_0, S_d)
        
        # Store in place  for later use during backprop
        self.y_0 = y_0
        self.epsilon_0 = epsilon_0
        
        return y_1, epsilon_1

    def backward(self, grad_epsilon, y_1):
        # The formula for epsilon_1 = epsilon_1(+) + epsilon_1(-)
        # Where epsilon_1(+) = [ W_1(+) * epsilon_0(+) + W_1(-) * epsilon_0(-) ] + 
        #                      [ d_1(+) * y_0(+) + d_1(-) * y_0(-) ] * N_1 +
        #                      [ d_1(+) * epsilon_0(+) + d_1(-) * epsilon_0(-) ] * N_1
        #
        # And epsilon_1(-) = [ W_1(+) * epsilon_0(-) + W_1(-) * epsilon_0(+) ] +
        #                      [ d_1(+) * y_0(-) + d_1(-) * y_0(+) ] * N_1 +
        #                      [ d_1(+) * epsilon_0(-) + d_1(-) * epsilon_0(+) ] * N_1
        
        # We want to activate the N_1 elements that are aligned with the grad_epsilon (for each output). 
        # For instance, if y_1 (output) at element 0 is positive, then we want to activate the N_1 elements
        # for epsilon_1(+). If y_1 (output) at element 1 is negative, then we want to activate the N_1 elements
        # for epsilon_1(-).
        
        # Compute epsilon_1(+)
        # Note that the dimensions (where [<>] means, expanding into)
        # W_1:          ([bs], i, j, [k]) 
        # epsilon_0:    (bs, i, [j], [k])
        # d_1:          ([bs], i, j, k)
        # N_1:          (i, j, k)
        # epsilon_1:    ([bs], [i], j, [k])
        # y_0:          (bs, i, [j], [k])
        # y_1:          (bs, [i], j, [k])
        
        # term0 is d_1*y_0
        term0 = self.y_0[...,tf.newaxis,tf.newaxis] * self.layer.kernel_delta[tf.newaxis,...]
        term1 = self.epsilon_0[...,tf.newaxis,tf.newaxis] * self.layer.kernel_delta[tf.newaxis,...]

        # Compute epsilon_1(+)
        terms = term0 + term1

        # num outputs 
        num_inputs = self.y_0.shape[-1]

        # N1(+)
        N1_plus = tf.reduce_mean(tf.cast(terms > 0, tf.float32),axis = 0)
        # N1(-)
        N1_minus = tf.reduce_mean(tf.cast(terms < 0, tf.float32),axis = 0)
        
        # Expand grad_epsilon to N1 shape 
        grad_epsilon_exp = tf.repeat(tf.repeat(grad_epsilon[tf.newaxis,...,tf.newaxis], 
                                            num_inputs, axis = 0),
                                 self.layer.quantizer.m, axis = -1)

        # Pick for output 
        N1 = tf.where(grad_epsilon_exp > 0, N1_plus, N1_minus)

        # Only keep elements with an avg > 0.5
        N1 = tf.where(N1 > 0.5, 1, 0)

        # Update N1
        self.kernel_N = N1.numpy()
        
        # Now, going backwards, we need to compute what the "grad_epsilon" should be. 
        # Note that right now grad_epsilon has a shape (j,) (one per output), 
        # while the following layer in the backprop will require this grad to be (i,) (one per input)
        # Usually, we would simply compute the gradient wrt the input, but the actual value of the grad
        # here doesn't really mean anything in terms of the actual gradient (it's only telling us the sign
        # of the terms we need to pick). So what should we do? 
        # Well, assume we have one output j that has a value of +1 in the grad_epsilon. 
        # This means that we want to maximize the positive terms and set to zero the negative ones for this output.
        # Doing this at this level is easy: just check which values of terms=(term0 + term1) are aligned with this 
        # term (positive, in this case), and set to 1. 

        # Now, how do we translate this to the input? Well, what are the inputs that affect this output?
        # 
        return super().backward(grad_epsilon, y_1)


class SoftmaxLayerVulnerability(LayerVulnerability):
    def __init__(self, layer):
        super().__init__(layer)
        # Make sure to set vulnerable to True
        self.vulnerable = True
        # Init epsilon_0 and y_0 (from the previous layer)
        self.epsilon_0 = None
        self.y_0 = None
    
    ################################################
    #                +-------------+
    #        y_0 --> |             | --> y_1
    #                | W, delta, N |
    #  epsilon_0 --> |             | --> epsilon_1
    #                +-------------+
    ################################################
    def forward(self, y_0, epsilon_0, **kwargs):
        # Call super
        y_1, epsilon_1 = super().forward(y_0, epsilon_0, **kwargs)
        return y_1, epsilon_1

    def backward(self, grad_epsilon, y_1):
        # Softmax doesn't change the sign of the epsilon, so we can just pass it through
        return grad_epsilon, y_1

""" Vulnerability object """
class ModelVulnerability:
    def __init__(self, model):
        
        """ Step 1) Make sure we have the deltas for the weights for each layer """
        model.compute_deltas(verbose = True)

        # Store
        self.model = model 

        # Loop thru layers and build 
        self.vlayers = [LayerVulnerability.from_type(layer) for layer in model.layers]

        # Create the hierarchy
        self.hierarchy = VulnerabilityGroup.from_list(self.vlayers)
    
    
    """ This is what effectively computes the vulnerability to an input X """
    def to(self, X, Y = None):
        # Compute
        return self.hierarchy(X, Y = Y)




        

        
        

