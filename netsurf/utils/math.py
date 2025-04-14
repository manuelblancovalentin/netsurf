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

from dataclasses import dataclass
from typing import Optional

@dataclass
class LatinHypercubeSampler:
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    num_samples: Optional[int] = 100

    def sample(self, num_samples: int = None) -> np.ndarray:
        """
        Generate Latin Hypercube samples.
        
        Returns:
            samples (np.ndarray): Array of shape (num_samples, d) where d is the dimensionality.
        """
        if num_samples is not None:
            self.num_samples = num_samples
        lb = np.array(self.lower_bounds)
        ub = np.array(self.upper_bounds)
        d = lb.size
        samples = np.empty((self.num_samples, d))
        for i in range(d):
            # Create equally spaced intervals in [0, 1]
            intervals = np.linspace(0, 1, self.num_samples + 1)
            # Sample one random value from each interval
            points = np.random.uniform(low=intervals[:-1], high=intervals[1:], size=self.num_samples)
            # Shuffle the points to randomize the assignment in this dimension
            np.random.shuffle(points)
            # Scale the points to the actual range for this dimension
            samples[:, i] = lb[i] + points * (ub[i] - lb[i])
        return samples