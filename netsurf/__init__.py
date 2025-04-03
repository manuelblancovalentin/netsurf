import os
from glob import glob
import importlib
# Tensorflow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import sys 
# Add current path to sys.path for netsurf
# Get path for parent of current file 
netsurf_dir = os.path.abspath(__file__)
netsurf_dir = os.path.dirname(os.path.dirname(netsurf_dir))
sys.path.append(netsurf_dir)
print(f'Adding {netsurf_dir} to sys.path')

# Add fkeras and qkeras (expected to be ../)
scripts_dir = os.path.dirname(os.path.dirname(netsurf_dir))
# Check if fkeras exists as a submodule, and if it's valid. If not, check if we can import it directly from environment. 
def check_module(module_name, dir):
    # Find __init__.py in fkeras
    module_init = glob(os.path.join(dir, module_name, "**", "__init__.py"), recursive = True)
    # Find the most top level __init__.py in fkeras_init list
    module_init = sorted(module_init, key=lambda x: len(x.replace(dir,'').split('/')))
    # Get the most top level __init__.py
    module_init = module_init[0] if len(module_init) > 0 else None
    # Check if fkeras_init is not None and exists
    if module_init is not None:
        module_path = os.path.dirname(module_init)
        # Add to path 
        sys.path.append(module_path)
        print(f'[INFO] - Added {module_name} to sys.path from {module_path}')
        return module_path
    else:
        # Check if fkeras is in the environment
        try:
            my_module = importlib.import_module(module_name)
            module_path = os.path.dirname(my_module.__file__)
            return module_path
        except ImportError:
            fkeras_path = None
            raise ImportError('[ERROR] - fkeras not found as submodule or in environment')

check_module('fkeras', netsurf_dir)
import fkeras 
check_module('qkeras', netsurf_dir)
import qkeras
check_module('pergamos', netsurf_dir)
import pergamos
check_module('nodus', netsurf_dir)
import nodus



import tensorflow as tf

# Logger 
from nodus import __logger__ as logger

# Import os to check if the env variable "netsurf_VERBOSITY_LEVEL" is set, if not, set to 1
if "netsurf_VERBOSITY_LEVEL" not in os.environ:
    os.environ["netsurf_VERBOSITY_LEVEL"] = "0"
netsurf_VERBOSITY_LEVEL = int(os.environ["netsurf_VERBOSITY_LEVEL"])
_print_initialization_tasks = netsurf_VERBOSITY_LEVEL > 0

# Get the directory where this file is 
import os
__dir__ = os.path.dirname(os.path.realpath(__file__))
""" Print initialization message"""
if _print_initialization_tasks: print(f"[LOG] - Initializing netsurf from dir {__dir__}")

""" Import config for easy access """
from . import config
if _print_initialization_tasks: print(f"[LOG] - Importing submodule config")

""" Import nodus to manage jobs """
import nodus  # type: ignore
# Create Nodus Session
nodus_session = nodus.NodusSession()
nodus_db = nodus_session.add_nodus_db("netsurf_db", db_path = config.NETSURF_NODUS_DB_NAME)  # Add a default NodusDB instance

# Import utils first of all (so we can use log)
if _print_initialization_tasks: print(f"[LOG] - Importing submodule utils")
from . import utils
# Directly import from logger for easy printing
from .utils.log import _log as log, _error as error, _warn as warn, _info as info, _ok as ok, _nope as nope, _print as print

""" Import argparser for easy access by run_experiment """
if _print_initialization_tasks: utils.log._log("Importing submodule args")
from . import args
from .args import parse_arguments

""" Import core for easy access """
if _print_initialization_tasks: utils.log._log("Importing submodule core")
from . import core
# Shortcuts
from .core import explorer
from .core.experiments import Experiment
from .core import injection
from .core import ranking
from .core.quantization import QuantizationScheme

""" Import dnn for easy access """
if _print_initialization_tasks: utils.log._log("Importing submodule dnn")
# Init MODELS
MODELS = {}

from . import dnn
# Shortcuts
from .dnn import models
from .core import datasets
from .core import benchmarks
from .core.benchmarks import get_benchmark, get_training_session
from .core.benchmarks import load_session
from .dnn import metrics
from .dnn.metrics import METRICS
from .dnn import losses
from .dnn.losses import LOSSES
from .dnn.models import QModel, load_model
from .dnn.layers import QQLAYERS
from .core import WeightRanker, RankingComparator # Ranking
from .core import Experiment # Expeirments
from .core import UncertaintyProfiler, ProfileDivergence # Uncertainty and profilers

# Import documentation
from . import doc

# Import the rest of the modules
#from . import plots

""" Import gui for easy access """
if _print_initialization_tasks: utils.log._log("Importing submodule gui")
from . import gui

""" At exit to determine when the program is closing """
import atexit

def exit():
    nodus_session.close()

# Register the function to be called on exit
atexit.register(exit)