""" Basic """
import os 
import yaml

""" Platform for identifying the platform we're on """
import platform 

""" Generic definitions """
DEFAULT_PROTECTION = (0.0, 0.2, 0.4, 0.6, 0.8)
DEFAULT_BER = (0.001, 0.00167, 0.00278, 0.00464, 0.00774, 0.01292, 0.02154, 0.03594, 0.05995, 0.1)
DEFAULT_PRUNINGS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
DEFAULT_QUANTIZATIONS = ['q<6,0,1>']
DEFAULT_BENCHMARKS = ['mnist_hls4ml', 'ECONT_AE']
DEFAULT_METHODS = ['bitwise_msb', 'random', 'layerwise_first', 'layerwise_last', 'weight_abs_value', 
                   'hirescam_norm', 'hiresdelta', 'hessian', 'hessiandelta',
                   'qpolar', 'qpolargrad', 'aiber','fisher']
# Default simulation config values
DEFAULT_NUM_REPS = 10

AVAILABLE_METHODS = ['bitwise_msb', 'bitwise_lsb', 'random', 'layerwise_first', 'layerwise_last', 'weight_abs_value', 
                    'hirescam', 'hirescam_norm', 'hirescam_times_weights', 'hirescam_times_weights_norm', 
                    'hiresdelta', 'hiresdelta_norm', 'hiresdelta_times_weights', 'hiresdelta_times_weights_norm',
                    'hessian', 'hessiandelta', 'aiber', 'qpolar', 'qpolargrad','fisher']

METHODS_NAMES = {'bitwise': 'bitwise', 
                 'bitwise_msb': 'bitwise', 
                 'bitwise_lsb': 'bitwise', 
                 'random': 'random', 
                 'layerwise': 'layerwise',
                 'layerwise_first': 'layerwise',
                 'layerwise_last': 'layerwise', 
                 'weight_abs_value': 'weight_abs_value', 
                 'hirescam': 'hirescam', 
                 'hirescam_norm': 'hirescam',
                 'hirescam_times_weights': 'hirescam', 
                 'hirescam_times_weights_norm': 'hirescam', 
                 'hiresdelta': 'hiresdelta', 
                 'hiresdelta_norm': 'hiresdelta', 
                 'hiresdelta_times_weights': 'hiresdelta', 
                 'hiresdelta_times_weights_norm': 'hiresdelta',
                 'hessian': 'hessian', 
                 'hessiandelta': 'hessiandelta',
                 'qpolar': 'qpolar',
                 'qpolargrad': 'qpolargrad',
                 'aiber': 'aiber',
                 'fisher': 'fisher',
                 }


""" Config per method """
config_per_method = {
    'random': {'method': 'random', 'method_suffix': None, 'method_kws': 'ascending=False'},
    
    'layerwise': {'method': 'layerwise', 'method_suffix': 'first_to_last', 'method_kws': 'ascending=True'},
    'layerwise_last': {'method': 'layerwise', 'method_suffix': 'last_to_first', 'method_kws': 'ascending=False'},
    'layerwise_first': {'method': 'layerwise', 'method_suffix': 'first_to_last', 'method_kws': 'ascending=True'},
    
    'bitwise': {'method': 'bitwise', 'method_suffix': 'msb_to_lsb', 'method_kws': 'ascending=True'},
    'bitwise_msb': {'method': 'bitwise', 'method_suffix': 'msb_to_lsb', 'method_kws': 'ascending=False'},
    'bitwise_lsb': {'method': 'bitwise', 'method_suffix': 'lsb_to_msb', 'method_kws': 'ascending=True'},

    'weight_abs_value': {'method': 'weight_abs_value', 'method_suffix': None, 'method_kws': 'ascending=False'},

    'hirescam': {'method': 'hirescam', 'method_suffix': None, 'method_kws': 'times_weights=False ascending=False normalize_score=False batch_size=100'},
    'hirescam_norm': {'method': 'hirescam', 'method_suffix': None, 'method_kws': 'times_weights=False ascending=False normalize_score=True batch_size=100'},
    'hirescam_times_weights': {'method': 'hirescam', 'method_suffix': None, 'method_kws': 'times_weights=True ascending=False normalize_score=False batch_size=100'},
    'hirescam_times_weights_norm': {'method': 'hirescam', 'method_suffix': None, 'method_kws': 'times_weights=True ascending=False normalize_score=True batch_size=100'},

    'hiresdelta': {'method': 'hiresdelta', 'method_suffix': None, 'method_kws': 'times_weights=False ascending=False normalize_score=False batch_size=100'},
    'hiresdelta_norm': {'method': 'hiresdelta', 'method_suffix': None, 'method_kws': 'times_weights=False ascending=False normalize_score=True batch_size=100'},
    'hiresdelta_times_weights': {'method': 'hiresdelta', 'method_suffix': None, 'method_kws': 'times_weights=True ascending=False normalize_score=False batch_size=100'},
    'hiresdelta_times_weights_norm': {'method': 'hiresdelta', 'method_suffix': None, 'method_kws': 'times_weights=True ascending=False normalize_score=True batch_size=100'},

    'hessian': {'method': 'hessian', 'method_suffix': None, 'method_kws': 'batch_size=96 eigen_k_top=3 max_iter=1000'},
    'hessiandelta': {'method': 'hessiandelta', 'method_suffix': None, 'method_kws': 'batch_size=96 eigen_k_top=3 max_iter=1000'},
    
    'aiber': {'method': 'aiber', 'method_suffix': None, 'method_kws': 'ascending=False batch_size=96'},

    'qpolar': {'method': 'qpolar', 'method_suffix': None, 'method_kws': 'ascending=False batch_size=96'},
    'qpolargrad': {'method': 'qpolargrad', 'method_suffix': None, 'method_kws': 'ascending=False batch_size=96'},

    'fisher': {'method': 'fisher', 'method_suffix': None, 'method_kws': 'ascending=False batch_size=96'},
}



###############################################################################################################################
# GUI DEFINITIONS 
###############################################################################################################################
DEFAULT_LEVEL_MAP = {0: 'Root', 1: 'Benchmark', 2: 'Quantization', 3: 'Model', 4: 'Method', 5: 'Experiment'} #5: 'Run'}
DEFAULT_CHILDREN_PROP = {'Root': 'benchmark', 'Benchmark': 'quantization', 'Quantization': 'pruning', 'Pruning': 'method', 'Model': 'method', 'Method': None, 'Experiment': None} #'Run': None}

""" Define some parameters """
key_map = {'hiresdelta_None': 'HiResDelta', 'hiresdelta': 'HiResDelta', 
            'hirescam_None': 'HiResCam', 'hirescam': 'HiResCam', 
            
            'qpolar_None': 'QPolar', 'qpolar': 'QPolar',
            'qpolargrad_None': 'QPolarGrad', 'qpolargrad': 'QPolarGrad',

            'hessian_original_msb_ranking_method_msb': 'Hessian (Original - MSB)', 'hessian_original_msb': 'Hessian (Original - MSB)', 
            'hessian_ranking_same_ranking_method_same' : 'Hessian (Original - Same)', 'hessian_ranking_same': 'Hessian (Original - Same)',
            'hessian_original_same_ranking_method_same': 'Hessian (Original - Same)','hessian_original_same': 'Hessian (Original - Same)', 
            'hessian_ranking_hierarchical_ranking_method_hierarchical': 'Hessian (Ours)', 'hessian_hierarchical': 'Hessian (Ours)', 'hessian_ranking_hierarchical': 'Hessian (Ours)', 
            'hessian_ranking_lsb_ranking_method_msb' : 'Hessian (Original - LSB)', 'hessian_ranking_lsb': 'Hessian (Original - LSB)', 'hessian_ranking_method_lsb': 'Hessian (Original - LSB)',
            'hessian_ranking_msb_ranking_method_msb' : 'Hessian (Original - MSB)', 'hessian_ranking_msb': 'Hessian (Original - MSB)', 'hessian_ranking_method_msb': 'Hessian (Original - MSB)',
            'hessian_': 'Hessian',
            'hessiandelta': 'Hessian Delta', 'hessiandelta_None': 'Hessian Delta',
            
            'random': 'Random',
            'recursive_uneven': 'Recursive Uneven',
            'diffbitperweight': 'Diff Bit',
            'layerwise_first_to_last': 'Layerwise (First)',
            'layerwise_last_to_first': 'Layerwise (Last)',
            'weight_abs_value': 'Weight Abs. Value',
            'bitwise_msb_to_lsb': 'MSB to LSB',
            'bitwise_lsb_to_msb': 'LSB to MSB',

            'fisher': 'Fisher', 'fisher_None': 'Fisher',
            
            '_None': '',
            '_ranking_msb_ranking_method_msb': ' (Original - MSB)', 
            '_ranking_msb_ranking_method_lsb': ' (Original - LSB)',
            '_ranking_method_msb': ' (Original - MSB)', '_ranking_msb': ' (Original - MSB)', 
            '_ranking_same_ranking_method_same' : ' (Original - Same)', '_ranking_method_same': ' (Original - Same)', '_ranking_same': ' (Original - Same)', 
            '_ranking_hierarchical_ranking_method_hierarchical' : ' (Ours)', '_ranking_method_hierarchical': ' (Ours)', '_ranking_hierarchical': ' (Ours)', 
            '_times_weights_true': ' x W', '_times_weights_false': '',
            '_times_weights': ' x W',
            '_ascending_true': '', '_ascending_false': '',
            '_normalize_score_true': ' (norm)', '_normalize_score_false': '',
            '_batch_size_10000': '', '_batch_size_1000': '', '_batch_size_100': '', 
            '_batch_size_96': '', '_batch_size_32': '', '_batch_size_256': '',
            '_6bits_0int': '',
            '_normalized': '',
            '_bit_value_0': '', '_bit_value_1': '', '_bit_value_2': '', '_bit_value_3': '', '_bit_value_4': '', '_bit_value_5': '',
            }

# Define the default colors
DEFAULT_COLOR_CYCLE = ['royalblue', 'darkorange', 'forestgreen', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

DEFAULT_DATA_COLUMN_WIDTHS = {'id': 50,
                  'status': 76,
                  'progress': 67,
                  'benchmark': 150,
                  'quantization': 80,
                  'model': 120,
                  'pruning': 58,
                  'method': 120, 
                  'timestamp': 150,
                  'experiment': 175,
                  'experiment_dir': 785,
                  #'run': 175, 
                  #'run_dir': 785,
                  'method_alias': 200,
                  'pid_file': 150,
                  'pid': 80,
                  }


""" Platform dependent definitions """
def check_platform():
    system = platform.system()

    if system == "Darwin":
        return "macOS"
    elif system == "Windows":
        return "Windows"
    elif system == "Linux":
        return "Linux"
    else:
        return "Unknown"

# Current system
current_system = check_platform()

def _get_gui_defaults(system):
    if system == 'macOS':
        # LAYOUT DEFAULTS
        DEFAULT_MAIN_WINDOW_HEIGHT = 500
        DEFAULT_MAIN_WINDOW_WIDTH = 1000
        DEFAULT_TEXTEDIT_HEIGHT = 20
        DEFAULT_BUCKET_PANEL_HEIGHT = 250
    elif system == 'Linux':
        # LAYOUT DEFAULTS
        DEFAULT_MAIN_WINDOW_HEIGHT = 800
        DEFAULT_MAIN_WINDOW_WIDTH = 1400
        DEFAULT_TEXTEDIT_HEIGHT = 20
        DEFAULT_BUCKET_PANEL_HEIGHT = 350
    elif system == 'Windows':
        # TODO: CHECK IF WE WANT SOMETHING DIFFERENT FOR WINDOWS
        # LAYOUT DEFAULTS
        DEFAULT_MAIN_WINDOW_HEIGHT = 800
        DEFAULT_MAIN_WINDOW_WIDTH = 1400
        DEFAULT_TEXTEDIT_HEIGHT = 20
        DEFAULT_BUCKET_PANEL_HEIGHT = 350
    else:
        # LAYOUT DEFAULTS
        DEFAULT_MAIN_WINDOW_HEIGHT = 800
        DEFAULT_MAIN_WINDOW_WIDTH = 1400
        DEFAULT_TEXTEDIT_HEIGHT = 20
        DEFAULT_BUCKET_PANEL_HEIGHT = 350

    return DEFAULT_MAIN_WINDOW_HEIGHT, DEFAULT_MAIN_WINDOW_WIDTH, DEFAULT_TEXTEDIT_HEIGHT, DEFAULT_BUCKET_PANEL_HEIGHT

""" Get GUI layout setup """
DEFAULT_MAIN_WINDOW_HEIGHT, DEFAULT_MAIN_WINDOW_WIDTH, \
    DEFAULT_TEXTEDIT_HEIGHT, DEFAULT_BUCKET_PANEL_HEIGHT = _get_gui_defaults(current_system)

# Check if ~/.netsurf/config exists 

_netsurf_config_file = os.path.expanduser('~/.netsurf/config')
if os.path.exists(_netsurf_config_file):
    # Read yaml
    with open(_netsurf_config_file, 'r') as f:
        _config = yaml.safe_load(f)
        if 'benchmarks_dir' in _config.keys():
            DEFAULT_BENCHMARKS_DIR = _config['benchmarks_dir']
        if 'datasets_dir' in _config.keys():
            DEFAULT_DATASETS_DIR = _config['datasets_dir']
    
    print(f"Found config file: {_netsurf_config_file}")
else:

    # Default directory values
    if current_system == 'Linux':
        DEFAULT_BENCHMARKS_DIR = "/asic/projects/NU/netsurf/manuelbv/benchmarks"
        DEFAULT_DATASETS_DIR = "/asic/projects/NU/netsurf/manuelbv/datasets"
    elif current_system == 'macOS':
        DEFAULT_BENCHMARKS_DIR = '/Users/mbvalentin/scripts/netsurf/benchmarks'
        DEFAULT_DATASETS_DIR = '/Users/mbvalentin/scripts/netsurf/datasets'
    else:
        DEFAULT_BENCHMARKS_DIR = None
        DEFAULT_DATASETS_DIR = None
    
    # Create the file and write the default values
    if DEFAULT_BENCHMARKS_DIR is not None and DEFAULT_DATASETS_DIR is not None:
        if not os.path.exists(os.path.expanduser('~/.netsurf')):
            os.makedirs(os.path.expanduser('~/.netsurf'), exist_ok=True)
        with open(_netsurf_config_file, 'w') as f:
            yaml.dump({'benchmarks_dir': DEFAULT_BENCHMARKS_DIR, 'datasets_dir': DEFAULT_DATASETS_DIR}, f)
        print(f"Created config file: {_netsurf_config_file}")
        
# Def constraints and padding for widgets 
# Left, Top, Right, Bottom
DEFAULT_WIDGET_PADDINGS = (0, 2, 0, 2)


########################################################################################
# NODUS DB / JOBS DEFINITONS
########################################################################################
NETSURF_NODUS_DB_NAME = "netsurf_db"

def _get_db_defaults(system):
    if system == 'macOS' or system == 'Linux':
        return os.path.join(os.path.expanduser('~/.nodus'), NETSURF_NODUS_DB_NAME)
    elif system == 'Windows':
        return os.path.join(os.path.expanduser('~/.nodus'), NETSURF_NODUS_DB_NAME)

NETSURF_NODUS_DB_PATH = _get_db_defaults(current_system)