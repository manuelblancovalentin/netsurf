# Reg-exp
import re

# Typing
from typing import List, Union

# Numpy
import numpy as np

# Matplotlib
import matplotlib.pyplot as plt

# import pergamos
import pergamos as pg

# Pandas 
import pandas as pd

""" Method to parse a quantization scheme  in the form q<m,n,s> where m is the total number of bits, n 
    is the number of bits of the integer part and s is the sign bit (1 for signed, 0 for unsigned) 
"""
def parse_quantization_scheme(q_string):
  pattern = r"q<(\d+)(?:,(\d+))?(?:,(\d+))?>"
  match = re.match(pattern, q_string)

  if not match: 
    # Try format qm_n_s
    pattern = r"q(\d+)_(\d+)_(\d+)"
    match = re.match(pattern, q_string)
    if not match: raise ValueError("Invalid quantization format")

  m = int(match.group(1))
  n = int(match.group(2)) if match.group(2) is not None else m  # Default n = m
  s = int(match.group(3)) if match.group(3) is not None else 0  # Default s = 0

  # Now we can infer the number of float bits f
  f = m - n - s

  return m, n, f, s


class CustomString(str):
  def __new__(cls, string: str):
    # Create the new tring instance using the class __new__ method 
    # (Immutable types like str don't allow using __init__, or overriding things)
    return super().__new__(cls, string)

  def no_special_chars(self):
    return self.replace(',','_').replace('<','').replace('>','')

# Definition of a quantization scheme 
class QuantizationScheme:
  def __init__(self, quantization_scheme: str):
    # Make sure the scheme is in the format q<m,n,s> where
    # m: total number of bits
    # n: number of bits of the integer part
    # s: sign bit (1 for signed, 0 for unsigned)
    self._scheme_str = CustomString(quantization_scheme)
    self.m, self.n, self.f, self.s = parse_quantization_scheme(quantization_scheme)
    self.scheme = {'total': self.m, 'integer': self.n, 'float': self.f, 'sign': self.s}

    # We need to make sure this quantization scheme makes sense. 
    # First of all, m must be greater than 0.
    assert self.m > 0, "Total number of bits must be greater than 0"
    # Second of all, n and s must be equal or greater than 0
    assert self.n >= 0, "Integer bits must be greater or equal to 0"
    assert self.s >= 0, "Sign bit must be greater or equal to 0"
    # Third of all m must be greater or equal to n. 
    assert self.m >= self.n, "Total number of bits must be greater or equal to the integer number of bits"
    # Fourth, self.s must be either 0 or 1.
    assert self.s in [0, 1], "Sign bit must be either 0 or 1"
    # Fifth, if sign is 1, n has to be equal or smaller than (self.m-1)
    if self.s == 1:
      assert self.n <= (self.m - 1), "Integer bits must be smaller or equal to (total number of bits - 1) in the case of signed quantizations."
    # And fiiiinally, n + f + s MUST add to m
    assert self.n + self.f + self.s == self.m, "Total number of bits must add to n + f + s"

    # Compute range of values 
    self.range, self.min_step = self._compute_range()
    self.min_value, self.max_value = self.range

  @classmethod
  def from_config(cls, config):
    # Assert 
    assert 'total' in config, "Total number of bits must be in the config"
    assert 'integer' in config, "Integer number of bits must be in the config"
    assert 'sign' in config, "Sign bit must be in the config"
    return cls(f"q<{config['total']},{config['integer']},{config['sign']}>")
  
  def get_config(self):
    return self.scheme

  def parse_quantization_scheme(self):
    return parse_quantization_scheme(self.scheme)

  def _compute_range(self):
    # Get params
    m, n, f, s = self.m, self.n, self.f, self.s
    
    # min step
    min_step = 2**(-f)

    if s:
      min_value = -2**n
      max_value = 2**n - min_step
    else:
      min_value = 0
      max_value = 2**n - min_step
    
    return (min_value, max_value), min_step
  
  def __str__(self):
    return self.__repr__()
  
  @property
  def bins(self):
    # return array of bins 
    return np.arange(self.min_value, self.max_value + self.min_step, self.min_step)
      
  @property
  def rbins(self):
    bins = 2.0**np.arange(0, self.n + self.s)
    # Add the negative /floats
    bins = np.concatenate((2.0**np.arange(-self.f, 0), bins))
    # Add [0] to the bins
    bins = np.concatenate(([0.0], bins)) if not self.s else np.concatenate((-bins[::-1], [0.0], bins))
    return bins
  
  @property 
  def qbins(self):
    bits_delta = 2.0**(np.arange(self.n-1+self.s,-self.f-1,-1))
    # Concat the negative values (mirrored)
    bits_delta = np.concatenate((-bits_delta, [0.0], bits_delta[::-1]))
    return bits_delta


  @property
  def __fmt__(self):
    fmt = '' if self.s == 0 else 'S'
    fmt += "".join(['x']*self.n)
    fmt += '.'
    fmt += "".join(['x']*self.f)
    return fmt
  
  def format(self, x = List[Union[np.ndarray | list | tuple]]):
    if x is None:
      return self.__fmt__
    else:
      # init string 
      s = ''
      # Loop thru elements
      i = 0
      if self.s:
        s += f"[{x[i]}]"
        i += 1
      for _ in range(self.n):
        s += f"{x[i]}"
        i += 1
      s += '.'
      for _ in range(self.f):
        s += f"{x[i]}"
        i += 1
      return s
  
  def __repr__(self):
    s = f'🧮 <QuantizationScheme({self._scheme_str})> obj @ ({hex(id(self))}):\n'
    s += f'    Total number of bits (m): {self.m}\n'
    s += f'    Integer bits         (n): {self.n}\n'
    s += f'    Float bits           (f): {self.f}\n'
    s += f'    {"Signed" if self.s else "Unsigned"}\n'
    s += f'    Range: ({self.min_value}, {self.max_value})\n'
    s += f'    Min step: {self.min_step}\n'
    s += f'    Format: {self.__fmt__}\n'
    return s
  
  def sample(self, shape, fcn = np.random.uniform): 
    # Returns a sample of data within the quantization range 
    return self(fcn(self.min_value, self.max_value, shape))
  
  def __call__(self, x): 
    # Performs quantization 
    x = np.round(x / self.min_step) * self.min_step
    x = np.clip(x, self.min_value, self.max_value)
    return x
  
  def __backbone__(self, shape):
    # Representation in int of the values that will be used for the comparison to get 
    # the binary of a number 
    bone = 2**np.arange(self.m-self.s-1, -1, -1)
    bone = np.tile(bone, shape + (1,))
    return bone

  def bin(self, x, return_str = False):
    # returns the binary representation of x according to our quantization scheme

    # Get the backbone for x
    bone = self.__backbone__(x.shape)

    # Find the values that are negative forehand
    is_neg = (x < 0)

    # Shift the values to be positive
    x_shifted = (x/self.min_step).astype(int)

    # Shift the negative values to be positive by adding max_value
    # However remember that we need to shift max_value by f
    x_shifted[is_neg] = 2**(self.f)* (self.max_value + x[is_neg] + self.min_step)

    # Now we can perform the comparison
    x_bin = (x_shifted[...,None] & bone) == bone

    if self.s == 1:
      # Now, we need to add the sign bit if this is a signed quantization
      # Init all to zeros first
      sign = np.zeros(x.shape + (1,), dtype=bool)
      # Now set the is_neg to self.s 
      sign[is_neg] = self.s
      x_bin = np.concatenate((sign, x_bin), axis=-1)

    if not return_str:
      return x_bin

    # flatten except bit dimension
    x_bin_flatten = x_bin.reshape(-1, x_bin.shape[-1])
    # Finally  we can get the binary representation in string in case we want to display it
    x_bin_str = []
    # Loop thru elements
    for _, el in enumerate(x_bin_flatten):
      x_bin_str.append(self.format(el.astype(int)))
    
    # Reshape
    x_bin_str = np.array(x_bin_str).reshape(x.shape)

    return x_bin, x_bin_str
  
  # Function to perform the debinarization of a binary number
  def float(self, x_bin):
    # Get the backbone for x
    bone = self.__backbone__(x_bin.shape[:-1])

    # Get the sign bit
    if self.s:
      sign = x_bin[...,0]

    # Get the rest of the bits
    bits = x_bin[...,self.s:]

    # Get the integer part
    x_int = np.sum(bits * bone, axis=-1)

    # Get the float part
    x_float = x_int * self.min_step

    # If the sign bit is 1, we need to subtract the max_value
    x_float[sign] = x_float[sign] - self.max_value - self.min_step

    return x_float
  
  @property
  def colormap(self):
    bit_colors = plt.get_cmap("tab10", self.m)
    return {i: bit_colors(i) for i in range(self.m)}
  
  # Function to compute delta matrix
  def compute_delta_matrix(self, W_q):
    """
    Computes the bit-flip delta values for a quantized weight matrix.

    :param W_q: Quantized weight matrix
    :param quantization_params: Tuple (total_bits, int_bits, signed)
    :return: Delta matrix of the same shape as W_q but with an extra dimension for bits
    """
    m, n, f, s = self.m, self.n, self.f, self.s
    shape = W_q.shape

    # Get the binary representation of W_q
    W_bin = self.bin(W_q)

    # Placeholder for the values of the deltas (regardless of weights)
    # (WITHOUT THE SIGN BIT)
    bits_delta = 2.0**(np.arange(n-1,-f-1,-1))
    bits_delta = np.reshape(bits_delta, tuple([1]*W_q.ndim) + (n+f,))
    bits_delta = np.tile(bits_delta, shape + (1,))

    # For each False value of W_bin, assign bits_delta, for each True value, assign -bits_delta
    # Remember to remove the sign bit though (if any)
    bits_delta = bits_delta*(1-W_bin[...,self.s:]) + (-1*bits_delta)*(W_bin[...,self.s:])

    # Now just remember to add the bit for the sign, which will be -2**n if the sign is 1, and 2**n if the sign is 0
    if self.s:
      bits_delta_sign = -2**(n) * np.ones(W_q.shape)
      bits_delta_sign[W_bin[...,0]] *= -1
      # concat
      bits_delta_sign = bits_delta_sign[...,None]
      bits_delta = np.concatenate((bits_delta_sign, bits_delta), axis=-1)
    
    assert bits_delta.shape == W_q.shape + (m,)
    return bits_delta

    """ Now we need to check each weight and see whether their bits are 0 or 1 to apply the correct delta """
    # The first thing to do, which might seem a bit counter intuitive, is to shift
    # everything
    #  by m (this is so all our values are integers, and not floats).
    # We'll undo this later. We need to shift them to integers so we can do the
    # comparison later. 
    delta_bit_flip_shifted = ((2**f)*bits_delta).astype(int)

    # Check where weights are negative
    is_neg_weight = (W_q < 0)
    # shift the weights also so they are also integers
    weights_shifted = ((2**f)*np.abs(W_q)).astype(int)

    is_neg_weight_shifted = (((2**f)*(self.max_value+W_q)).astype(int))
    weights_shifted[is_neg_weight] = is_neg_weight_shifted[is_neg_weight]

    # Add extra dimension for nbits
    weights_shifted = np.tile(weights_shifted[...,None], (n+f))

    # Perform the comparison
    is_bit_one = ((delta_bit_flip_shifted & weights_shifted) == delta_bit_flip_shifted)

    # Now that we know which bits are one and which not, we can calculate the deltas for each bit
    # Remember that 0 --> 1 means ADDING value except for the bit before the period, in which case 
    # it means, subtracting 1.
    bits_delta = 1.0*bits_delta
    #bits_delta[...,0] *= -1
    bits_delta = bits_delta*(1-is_bit_one) + (-1*bits_delta)*is_bit_one

    # finally, we can add the sign. We know that if the sign is 1, it means the number 
    # is negative, and thus the delta should be delta = +max_value
    # while if sign is 0, delta = -max_value
    bits_delta_sign = -2**(n) * np.ones(W_q.shape)

    # 
    bits_delta_sign[is_neg_weight] *= -1

    # Add bit dimension before concatenating to bits_delta
    bits_delta_sign = bits_delta_sign[...,None]
    delta = np.concatenate((bits_delta_sign, bits_delta), axis=-1)

    return delta

  def serialize(self, include_emojis = False):
    content = pd.DataFrame({'Total number of bits': [self.m],
                            'Integer bits': [self.n],
                            'Float bits': [self.f],
                            'Signed': [self.s],
                            'Range': [f'({self.min_value}, {self.max_value})'],
                            'Min step': [self.min_step],
                            'Format': [self.__fmt__]
                            })
    return {fr'{"🧮" if include_emojis else ""} Quantization Scheme ({self._scheme_str})': content}

  @pg.printable
  def html(self):
    return self.serialize(include_emojis=True)
    
