
# Ignore annoying errors and warnings in numpy
import numpy
numpy.seterr(all = 'ignore') 

# Get available benchmarks
import yaml

# Get the directory where this file is 
import os
__config_dir__ = os.path.dirname(os.path.realpath(__file__))

# First of all, make sure the config file is loaded and exists 
CONFIG_FILE = os.path.join(__config_dir__, 'benchmarks.yaml')
if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(f"Config file {CONFIG_FILE} not found")

# Load the config file
BENCHMARKS_CONFIG = dict()
with open(CONFIG_FILE, 'r') as f:
    BENCHMARKS_CONFIG = yaml.safe_load(f)

# Define a constructor for the tuple tag
def tuple_constructor(loader, node):
    # Load the sequence of values from the YAML node
    values = loader.construct_sequence(node)
    # Return a tuple constructed from the sequence
    return tuple(values)

# Register the constructor with PyYAML
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)

# Read datasets config too 
DATASETS_CONFIG_FILE = os.path.join(__config_dir__, 'datasets.yaml')
if not os.path.exists(DATASETS_CONFIG_FILE):
    raise FileNotFoundError(f"Config file {DATASETS_CONFIG_FILE} not found")

# Load the config file
DATASETS_CONFIG = dict()
with open(DATASETS_CONFIG_FILE, 'r') as f:
    DATASETS_CONFIG = yaml.load(f, Loader=yaml.SafeLoader)

# Available benchmarks are entries in config
AVAILABLE_BENCHMARKS = list(BENCHMARKS_CONFIG.keys())

# Import definitions from defs 
from .defs import *