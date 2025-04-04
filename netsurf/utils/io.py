# Basic modules
import sys
import os 
import re 
import tempfile

# Extended os-attributes
import xattr

""" System """
import platform
import subprocess

""" For loading/saving binary objects """
import pickle as pkl
import dill
#import keras 
#[@max]
from tensorflow import keras 

# Log utils
from . import log
from .time import seconds_to_hms

from time import time

""" Create temporary file (at /tmp) """
def create_temporary_file(prefix = '', suffix = '', ext = '.tmp'):
    return tempfile.mkstemp(prefix=prefix, suffix=suffix+ext, text=False)[1]

""" Check if path exists """
def path_exists(path):
    if path is None:
        return False
    return os.path.exists(path)

""" Check if directory is valid and exists """
def is_valid_directory(directory):
    if directory is None:
        return False
    return os.path.isdir(directory)

""" Get latest name for a given pattern (for a string, not a file) """
def get_latest_suffix(name, name_list, next = False, divider = '_', return_index = False, verbose = True, next_if_empty = False):
    # Replace "." with "\." to avoid regex issues
    name = name.replace(".", "\.")
    pattern = f"{name}{divider}(\d+)?"
    
    # Get the latest file (max number)
    max_suffix = None
    regex = re.compile(pattern)

    try:
        # List all files in the directory
        for _name in name_list:
            match = regex.fullmatch(_name)
            if match:
                # Extract the suffix (if any)
                suffix = match.group(1)
                # Convert to integer if suffix exists, otherwise consider it as 0
                suffix = int(suffix) if suffix else 0
                max_suffix = max(max_suffix, suffix) if max_suffix is not None else suffix
    except Exception as e:
        if verbose: log._error(f"An error occurred while trying to get the latest name pattern: {e}")

    # Thus the latest file will be:
    if max_suffix is not None:
        if next:
            max_suffix += 1
        if max_suffix > 0:
            results = f"{name}{divider}{max_suffix}"
        else:
            results = (name,)
    else:
        results = (None,) if not next_if_empty else (name,)
        max_suffix = max_suffix if not next_if_empty else 0

    if return_index:
        results = results + (max_suffix,)
    return results

""" Get latest file given a pattern, assuming it follows something like <prefix>.<extension>(.<number>?) """
def get_latest_suffix_for_file(file, next = False, return_index = False, verbose = True, next_if_empty = False):

    # Get parent directory of file 
    directory = os.path.abspath(os.path.dirname(file))

    # Add the part of the suffix that adds the number
    filename = os.path.basename(file)
    # Replace "." with "\." to avoid regex issues
    filename = filename.replace(".", "\.")
    pattern = f"{filename}\.?(\d+)?"
    
    # Get the latest file (max number)
    max_suffix = None
    regex = re.compile(pattern)

    try:
        # List all files in the directory
        for file_name in os.listdir(directory):
            match = regex.fullmatch(file_name)
            if match:
                # Extract the suffix (if any)
                suffix = match.group(1)
                # Convert to integer if suffix exists, otherwise consider it as 0
                suffix = int(suffix) if suffix else 0
                max_suffix = max(max_suffix, suffix) if max_suffix is not None else suffix
    except FileNotFoundError:
        if verbose: log._error(f"Directory '{directory}' does not exist.")
    except Exception as e:
        if verbose: log._error(f"An error occurred while trying to get the latest file: {e}")

    # Thus the latest file will be:
    if max_suffix is not None:
        if next:
            max_suffix += 1
        if max_suffix > 0:
            results = (f"{file}.{max_suffix}",)
        else:
            results = (file,)
    else:
        results = (None,) if not next_if_empty else (file,)
        max_suffix = max_suffix if not next_if_empty else 0

    if return_index:
        results = results + (max_suffix,)
    return results


# Function to open file with default application
def open_file_with_default_viewer(filename):
    system = platform.system()
    
    if system == "Darwin":  # macOS
        subprocess.run(["open", filename])
    elif system == "Windows":  # Windows
        os.startfile(filename)
    else:  # Linux
        subprocess.run(["xdg-open", filename])
# Function to open the saved PNG file using the default OS image viewer

def open_image_with_default_viewer(plot_path):
    open_file_with_default_viewer(plot_path)


""" Open directory using the default file explorer """
def open_directory(directory):
    # First check directory exists and it's valid 
    if not is_valid_directory(directory):
        log._log(f"Invalid directory: {directory}")
        return
    
    if platform.system() == "Windows":
        log._log(f"Opening directory {directory} with Windows file explorer")
        os.startfile(directory)
    elif platform.system() == "Darwin":  # macOS
        log._log(f"Opening directory {directory} with macOS file explorer")
        subprocess.run(["open", directory])
    else:  # Linux/Unix
        log._log(f"Opening directory {directory} with Linux file explorer")
        subprocess.run(["xdg-open", directory])

""" Delete directory """
def delete_directory(directory):
    # First check directory exists and it's valid 
    if not is_valid_directory(directory):
        log._log(f"Invalid directory: {directory}")
        return
    
    log._log(f"Deleting directory {directory}")
    # Force delete directory
    subprocess.run(["rm", "-rf", directory])



""" Snippet to save custom objects to pkl file """
def save_object(obj, filename, meta_attributes = {}, custom_objects = {}, remove_log = False, verbose = False):
    # Redirect output to a log file 
    original_stdout = sys.stdout
    log_filename = filename.replace('.' + filename.split('.')[-1],'.log') if '.' in filename else filename + '.log'
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    errored = False
    with open(log_filename, 'w') as flog:
        # [@manuelbv]: THIS IS EXTREMELY IMPORTANT!!!!!! EARLIER, I WAS REASSIGNING sys.stdout 
        #   TO sys.__stdout__ WHICH WAS MESSING UP THE WHOLE IPYTHON ENVIRONMENT, AND I WOULDN'T
        #   BE ABLE TO DEBUG ANYTHING AFTERWARDS. WE COULDNT' DEBUG ANYTHING AFTER THIS LINE. 
        #   WE WOULD JUST GET A NOTIFICATION FROM VSCODE SAYING WE NEEDED IPYKERNEL SETUP WAS 
        #   REQUIRED. SO THE TRICK IS TO STORE THE ORIGINAL sys.stdout AND THEN REASSIGN IT BACK!!!!!!!
        # Store the original stdout
        sys.stdout = flog  # Redirect stdout to the file
        try:
            with keras.utils.custom_object_scope(custom_objects):
                with open(filename, 'wb') as f:
                    dill.dump(obj, f, pkl.HIGHEST_PROTOCOL)
                msg = f"Object saved to {filename}"
                if meta_attributes:
                    msg += f" with metadata attributes: "
                    for key, value in meta_attributes.items():
                        xattr.setxattr(filename, f'user.{key}', str(value).encode())
                        msg += f"{key}={value}, "
        except Exception as e:
            log._error(f"Error saving object to {filename}: {e}")
            msg = f"Error saving object to {filename}: {e}"
            errored = True
    # Reset the stdout
    sys.stdout = original_stdout
    if verbose:
        if errored:
            log._error(msg)
        else:
            log._info(msg)
    if remove_log:
        # Remove the log file
        if os.path.exists(log_filename):
            os.remove(log_filename)
    return msg


""" Snippet to load custom objects from pkl file """
def load_object(filename, custom_objects = {}, remove_log = False, verbose = False):
    # Redirect output to a log file 
    log_filename = filename.replace('.' + filename.split('.')[-1],'.log') if '.' in filename else filename + '.log'
    with open(log_filename, 'w') as flog:
        # [@manuelbv]: THIS IS EXTREMELY IMPORTANT!!!!!! EARLIER, I WAS REASSIGNING sys.stdout 
        #   TO sys.__stdout__ WHICH WAS MESSING UP THE WHOLE IPYTHON ENVIRONMENT, AND I WOULDN'T
        #   BE ABLE TO DEBUG ANYTHING AFTERWARDS. WE COULDNT' DEBUG ANYTHING AFTER THIS LINE. 
        #   WE WOULD JUST GET A NOTIFICATION FROM VSCODE SAYING WE NEEDED IPYKERNEL SETUP WAS 
        #   REQUIRED. SO THE TRICK IS TO STORE THE ORIGINAL sys.stdout AND THEN REASSIGN IT BACK!!!!!!!
        # Store the original stdout
        original_stdout = sys.stdout
        sys.stdout = flog  # Redirect stdout to the file
        
        obj = None
        try:
            with keras.utils.custom_object_scope(custom_objects):
                with open(filename, 'rb') as f:
                    obj = dill.load(f)
            # Reset the stdout
            sys.stdout = original_stdout
            if verbose: log._info(f"Object loaded from {filename}")
        except Exception as e:
            # Reset the stdout
            sys.stdout = original_stdout
            log._error(f"Error loading object from {filename}: {e}")
        
        if remove_log:
            # Remove the log file
            if os.path.exists(log_filename):
                os.remove(log_filename)
        return obj

""" 
    CREATE UNIQUE HASHS FOR CONFIGURATIONS
    @manuelbv: Note on April/4th/2025 -> We shifted away from SHA256, because it's NOT possible to reverse-engineer the hash
    into the serial configuration. So instead, we just use base64 url-safe (no special characters) to generate a unique hash
    that CAN be reversed-engineer into the original configuration (dictionary) :)
"""

import json
import zlib
import base64

def config_to_compressed_id(config: dict) -> str:
    """Serialize + compress + encode config into a compact folder-safe ID."""
    # Serialize to deterministic string
    config_str = json.dumps(config, sort_keys=True, separators=(",", ":"))
    # Compress the string
    compressed = zlib.compress(config_str.encode("utf-8"), level=9)
    # Encode in base64 (URL-safe), strip padding
    encoded = base64.urlsafe_b64encode(compressed).decode("utf-8")
    return encoded.rstrip("=")

def compressed_id_to_config(encoded_id: str) -> dict:
    """Decode + decompress + deserialize the folder-safe ID back to config."""
    # Add padding
    padding = "=" * ((4 - len(encoded_id) % 4) % 4)
    encoded_padded = encoded_id + padding
    compressed = base64.urlsafe_b64decode(encoded_padded)
    config_str = zlib.decompress(compressed).decode("utf-8")
    return json.loads(config_str)



















def get_metadata(path, filename = '.metadata.netsurf'):
    if not os.path.exists(path):
        return None 
    
    metadata_file = os.path.join(path, filename)

    if not os.path.exists(metadata_file):
        return None
    
    # Read metadata (json)
    try:
        metadata = json.load(open(metadata_file, 'r'))
    except Exception as e:
        log._error(f"Error reading metadata file: {e}")
        return None

    # Make sure metadata is a dictionary and it contains the keys we need 
    if not isinstance(metadata, dict):
        return None

    if 'netsurf' in filename:
        if 'level' not in metadata or 'name' not in metadata or 'config' not in metadata:
            return None
    elif 'nodus' in filename:
        pass 

    # Else, we are good!
    return metadata

            
""" Get extra attributes from a file """
def get_xattrs(filepath):
    # check if file exists 
    if not os.path.exists(filepath):
        return {}
    try:
        x = xattr.xattr(filepath)
        d = {}
        if hasattr(x, 'items'):
            d = {k.replace('user.',''): v.decode() for (k,v) in x.items()}
        return d
    except Exception as e:
        log._error(f"Error reading xattrs from file: {e}")
        return {}

# Save df to csv asking user for place to save it 
def save_table_to_csv(df, filepath):
    """ Save a pandas dataframe to a csv file """
    # Ask user for file name
    # Save the table
    df.to_csv(filepath, index=False)
    # Log 
    log._log(f"Table saved to {filepath}")


def handle_ansi_escape_codes(bytes_data):

    # Get text from bytes
    text = bytes_data.data().decode('utf-8', errors='ignore')

    """Handle all ANSI escape sequences including text formatting and colors."""
    # Regular expression for ANSI escape sequences (simple and extended)
    ansi_escape = re.compile(r'\x1b\[([0-9;]*)m')
    
    # Define color mappings for standard colors (30-37 for foreground, 40-47 for background)
    foreground_colors = {
        30: QColor(0, 0, 0),       # Black
        31: QColor(255, 0, 0),     # Red
        32: QColor(0, 255, 0),     # Green
        33: QColor(255, 255, 0),   # Yellow
        34: QColor(0, 0, 255),     # Blue
        35: QColor(255, 0, 255),   # Magenta
        36: QColor(0, 255, 255),   # Cyan
        37: QColor(255, 255, 255), # White
    }
    
    background_colors = {
        40: QColor(0, 0, 0),       # Black
        41: QColor(255, 0, 0),     # Red
        42: QColor(0, 255, 0),     # Green
        43: QColor(255, 255, 0),   # Yellow
        44: QColor(0, 0, 255),     # Blue
        45: QColor(255, 0, 255),   # Magenta
        46: QColor(0, 255, 255),   # Cyan
        47: QColor(255, 255, 255), # White
    }

    formatted_text = ""
    last_pos = 0
    fg_color = QColor(255, 255, 255)  # Default to white for foreground
    bg_color = QColor(0, 0, 0)        # Default to black for background
    bold = False
    underline = False
    dim = False
    reverse_video = False
    crossed_out = False
    italic = False

    # Iterate through all ANSI escape sequences
    for match in ansi_escape.finditer(text):
        # Append the text before the escape sequence
        formatted_text += text[last_pos:match.start()]
        
        # Process the ANSI escape sequence
        codes = match.group(1).split(';')

        # Handle reset codes
        if '0' in codes:
            fg_color = QColor(255, 255, 255)  # White (default foreground)
            bg_color = QColor(0, 0, 0)        # Black (default background)
            bold = False
            underline = False
            dim = False
            reverse_video = False
            crossed_out = False
            italic = False

        # Handle bold and dim
        if '1' in codes:  # Bold
            bold = True
        if '2' in codes:  # Dim
            dim = True
        if '3' in codes:  # Italic
            italic = True
        if '4' in codes:  # Underline
            underline = True
        if '5' in codes:  # Blink
            pass  # Blink not widely supported
        if '7' in codes:  # Reverse video
            reverse_video = not reverse_video  # Toggle reverse video
        if '9' in codes:  # Strikethrough
            crossed_out = True

        # Handle foreground color (30-37)
        for code in range(30, 38):
            if str(code) in codes:
                fg_color = foreground_colors.get(code, QColor(255, 255, 255))

        # Handle background color (40-47)
        for code in range(40, 48):
            if str(code) in codes:
                bg_color = background_colors.get(code, QColor(0, 0, 0))

        # Handle 24-bit colors (38 for foreground, 48 for background)
        if '38' in codes:  # Foreground color (RGB)
            color_idx = codes.index('38')
            if len(codes) > color_idx + 1:
                color_values = codes[color_idx + 1:]
                # There are two formats: 
                # 1. 38;2;r;g;b
                # 2. 38;5;index
                if color_values[0] == '2':
                    try:
                        cols = (int(color_values[1]), int(color_values[2]), int(color_values[3]))
                        fg_color = QColor(*cols)
                    except ValueError:
                        fg_color = QColor(255, 255, 255)  # Fallback to white
                elif color_values[0] == '5':
                    try:
                        fg_color = QColor(int(color_values[1]))
                    except ValueError:
                        fg_color = QColor(255, 255, 255)
            
        if '48' in codes:  # Background color (RGB)
            color_idx = codes.index('48')
            if len(codes) > color_idx + 1:
                color_values = codes[color_idx + 1:]
                # There are two formats: 
                # 1. 48;2;r;g;b
                # 2. 48;5;index
                if color_values[0] == '2':
                    try:
                        cols = (int(color_values[1]), int(color_values[2]), int(color_values[3]))
                        bg_color = QColor(*cols)
                    except ValueError:
                        bg_color = QColor(0, 0, 0)
                elif color_values[0] == '5':
                    try:
                        bg_color = QColor(int(color_values[1]))
                    except ValueError:
                        bg_color = QColor(0, 0, 0)

        # Apply the text formatting with the current fg, bg, and other styles
        text_format = QTextCharFormat()
        text_format.setForeground(fg_color)
        text_format.setBackground(bg_color)
        
        if bold:
            text_format.setFontWeight(75)  # Bold weight (75 is the value for bold)
        if underline:
            text_format.setFontUnderline(True)
        if dim:
            text_format.setFontItalic(True)  # In many terminals, "dim" maps to italic
        if italic:
            text_format.setFontItalic(True)
        if reverse_video:
            text_format.setForeground(bg_color)  # Swap foreground and background for reverse video
            text_format.setBackground(fg_color)
        if crossed_out:
            text_format.setFontStrikeOut(True)

        # Add the formatted text to the result
        formatted_text += f"<span style='color:{fg_color.name()}; background-color:{bg_color.name()};"
        if bold:
            formatted_text += " font-weight: bold;"
        if underline:
            formatted_text += " text-decoration: underline;"
        if dim:
            formatted_text += " font-style: italic;"
        if crossed_out:
            formatted_text += " text-decoration: line-through;"
        formatted_text += f"'>{text[match.end():]}</span>"
        
        last_pos = match.end()

    # Append the remaining text after the last escape sequence
    formatted_text += text[last_pos:]
    return formatted_text

def open_terminal_with_command(command, generic = False):
    
    if generic:
        subprocess.run(["xterm", "-e", command])  # xterm
    else:

        system_platform = platform.system()

        if system_platform == "Windows":
            # Windows: Use Command Prompt or PowerShell
            subprocess.run(["cmd", "/K", command])  # Command Prompt
            # or
            # subprocess.run(["powershell", "-NoExit", "-Command", command])  # PowerShell

        elif system_platform == "Darwin":
            # macOS: Use AppleScript to open Terminal.app
            subprocess.run(["osascript", "-e", f'tell app "Terminal" to do script "{command}"'])

        elif system_platform == "Linux":
            # Linux: Open GNOME Terminal or xterm
            subprocess.run(["gnome-terminal", "--", "bash", "-c", f"{command}; exec bash"])  # GNOME Terminal
            # or
            # subprocess.run(["xterm", "-e", command])  # xterm

import sys

class ProgressBar:
    """
    A simple text-based progress bar for console output.
    Example:
        bar = ProgressBar(total=60, prefix="Computing batch")
        for i in range(60):
            bar.update(i + 1)
    """
    def __init__(self, total, width=30, prefix="Progress", stream=sys.stdout):
        self.total = total            # Total number of steps
        self.width = width            # Width of the progress bar
        self.prefix = prefix          # Text prefix before the bar
        self.stream = stream          # Output stream (e.g., sys.stdout)
        self.last_len = 0             # For overwriting the line
        self.counter = 0
        self.start_time = None

    def update(self, current = None):
        """
        Update the progress bar.
        :param current: current step (1-indexed)
        """
        if current is None:
            # Update counter 
            self.counter += 1
            current = self.counter
        
        progress = current / self.total
        filled = int(self.width * progress)
        if self.start_time is None: self.start_time = time()
        
        elapsed = time() - self.start_time if self.start_time else 0
        eta = (elapsed / current * self.total) - elapsed if current > 0 else 0

        # Format the ETA
        elapsed_str = seconds_to_hms(elapsed) if elapsed > 0 else "N/A"
        t = f'[{elapsed_str}'
        # Format the ETA
        if eta > 0:
            # Format the ETA
            eta_str = seconds_to_hms(eta)
            # append 
            t += f' < {eta_str}'
        # close
        t += ']'

        bar = "[" + "=" * filled + ">" + " " * (self.width - filled - 1) + "]"
        msg = f"{self.prefix:<40s} {current}/{self.total}: {bar} {t}"
        self.stream.write("\r" + msg + " " * max(0, self.last_len - len(msg)))
        self.stream.flush()
        self.last_len = len(msg)

        if current == self.total:
            self.stream.write("\n")
    
    def __enter__(self):
        self.counter = 0
        self.start_time = time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # do nothing
        pass