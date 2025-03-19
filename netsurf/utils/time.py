import os 

# Glob
from glob import glob
# Time management
from datetime import datetime


# Snippet to convert seconds to HH:MM:SS
def seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))

# manuelbv
def find_latest_file(prefix, extension):
    files = glob(f"{prefix}*.{extension}")
    if len(files) == 0:
        return None
    latest_file = max(files, key=os.path.getctime)
    return latest_file

# manuelbv
def generate_filename(prefix, extension):
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    filename = f"{prefix}.{formatted_datetime}.{extension}"
    return filename

""" Method to convert time in seconds to days, hh:mm:ss """
def get_elapsed_time(time_in_seconds):
    days = time_in_seconds // (24 * 3600)
    time_in_seconds = time_in_seconds % (24 * 3600)
    hours = time_in_seconds // 3600
    time_in_seconds %= 3600
    minutes = time_in_seconds // 60
    time_in_seconds %= 60
    seconds = time_in_seconds
    # Now build the string 
    s = ""
    if days > 0:
        s += f"{int(days)} days, "
    # now format of HH:MM:SS
    s += f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    return s
