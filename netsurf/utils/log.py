""" Basic """
import os 

""" Datetime """
from datetime import datetime

""" Colorama for colored output """
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style

""" Import wsbmr """
import wsbmr

# Init colorama
colorama_init()

""" Utilities """
def _print(msg, type, end = '\n', tab = 0, prefix = ''):
    # Add tabulation
    tab = '  ' * tab
    type = type.lower().replace('logging','log').replace('log','l')
    type = type.lower().replace('warning','warn').replace('warn','w')
    type = type.lower().replace('error','err').replace('err','e')
    type = type.lower().replace('information','info').replace('info','i')
    type = type.lower().replace('okay','ok').replace('ok','k')
    type = type.lower().replace('nope','no').replace('no','n')
    color = {'i': Fore.WHITE, 'e': Fore.RED, 'w': Fore.YELLOW, 'l': Fore.CYAN, 'k': Fore.GREEN, 'n': Fore.RED}.get(type, Fore.WHITE)
    icon = {'i': '', 'e': '❌', 'w': "⚠️", 'l': '', 'k': "", 'n': ""}.get(type, 'i')
    type = {'i': 'INFO', 'e': 'ERROR', 'w': 'WARNING', 'l': 'LOG', 'k': "✅", 'n': "❌"}.get(type, type.upper()[:4])
    # Get timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
    print(f'{tab}{prefix}[{color}{icon}{type} @ {timestamp}{Style.RESET_ALL}] - {msg}', end = end)

# Logger
def _log(msg, **kwargs):
    _print(msg, 'log', **kwargs)
def _info(msg, **kwargs):
    _print(msg, 'info', **kwargs)
def _error(msg, **kwargs):
    _print(msg, 'error', **kwargs)
def _warn(msg, **kwargs):
    _print(msg, 'warning', **kwargs)
def _ok(msg, **kwargs):
    _print(msg, 'ok', **kwargs)
def _nope(msg, **kwargs):
    _print(msg, 'nope', **kwargs)
def _custom(level, msg, **kwargs):
    level = level.lower()[:4]
    _print(msg, level, **kwargs)

# Here, if we detect that nodus was imported, we will add the nodus logger to the log functions
try:
    def _log(msg, **kwargs):
        wsbmr.logger.custom('WSBMR', msg, **kwargs)
    def _info(msg, **kwargs):
        wsbmr.logger.info(msg, **kwargs)
    def _error(msg, **kwargs):
        wsbmr.logger.error(msg, **kwargs)
    def _warn(msg, **kwargs):
        wsbmr.logger.warn(msg, **kwargs)
    def _ok(msg, **kwargs):
        wsbmr.logger.ok(msg, **kwargs)
    def _nope(msg, **kwargs):
        wsbmr.logger.nope(msg, **kwargs)
    def _custom(level, msg, **kwargs):
        wsbmr.logger.custom(level, msg, **kwargs)
except ImportError:
    _error('Nodus not found. Logging functions will be used instead of nodus logger.')


def recursive_dict_printer(d, tab = 0):
    s = ''
    if len(d) > 0:
        for k, v in d.items():
            if isinstance(v, dict):
                s += '\t' * tab + k + ': {\n'
                s += recursive_dict_printer(v, tab + 1)
                s += '\t' * tab + '}\n'
            else:
                s += '\t' * tab + f'{k}: {v}' + '\n'
    else:
        s += '{}'
    return s


def open_session_log():
    # Get log from wsbmr nodus 
    try:
        filename = wsbmr.nodus.__nodus_log_file__
        if os.path.exists(filename):
            wsbmr.utils.open_file_with_default_viewer(filename)
    except:
        _error('Failed to open log file')