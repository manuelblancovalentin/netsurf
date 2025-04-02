# Function to convert float to binary
# Author: Alan Guo
def float_to_binary(x, nbits):
    binary_rep = ''
    sign = 1 if x < 0 else 0
    x = 1 + x if sign == 1 else x
    for b in range(1, nbits):
        bit_value = 2.0**(-b)
        if x >= bit_value:
            binary_rep += '1'
            x -= bit_value
        else:
            binary_rep += '0'
    return str(sign) + binary_rep

import numpy as np

def freedman_diaconis_bins(data):
  """
  Calculates the optimal number of bins using the Freedman-Diaconis rule.

  Args:
    data: A 1D numpy array of data.

  Returns:
    The optimal number of bins as an integer.
  """
  iqr = np.percentile(data, 75) - np.percentile(data, 25)
  n = len(data)
  if iqr > 0:
    bin_width = 2 * iqr / (n**(1/3))
    num_bins = int(np.ceil((np.max(data) - np.min(data)) / bin_width))
  else:
    num_bins = int(np.ceil(np.sqrt(n)))
  return num_bins