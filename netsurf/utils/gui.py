""" Basic """
import os

""" Pandas """
import pandas as pd

""" pyqt5 """
from PyQt5.QtWidgets import QFileDialog, QWidget, QMainWindow

""" netsurf """
import netsurf

""" Check coverage """
def check_coverage(df, num_reps = 10, protection = [0.0], ber = [0.001]):
    # If df is empty, return a df with num_reps for each combination of protection, ber
    if len(df) == 0:
        cov = []
        for t in protection:
            for b in ber:
                cov.append({'protection': t, 'ber': b, 'reps_run': 0, 'reps_missing': num_reps, 'reps_total': num_reps, 'progress': 0.0})
        # Convert to pandas df
        cov = pd.DataFrame(cov)
        progress = 0.0
        return progress, cov

    # Get the protection/ber columns only (there's an easy trick to do this by simply groupby protection,radiation and counting)
    tmrber = df[['protection', 'ber', 'method']]
    cov = (num_reps - tmrber.groupby(['protection', 'ber']).count()).reset_index().rename(columns = {'method':'reps_missing'})
    
    # Add the total number of reps, the reps run and the progress columns 
    cov['reps_run'] = num_reps - cov['reps_missing']
    cov['reps_total'] = num_reps
    cov['progress'] = cov['reps_run']/cov['reps_total']

    # Calculate the progress as the sum vs total number of experiments
    total = len(protection)*len(ber)*num_reps
    completed = cov['reps_missing'].sum()
    progress = 1 - completed/total

    # Clip progress to 0, 1
    progress = max(0, min(1, progress))
    return progress, cov


""" Interpolate between colors """
class ColorInterpolator:
    def __init__(self, colors, values):
        # colors are hex
        self.colors = colors
        # Values are between 0 and 1 
        self.values = values

        # Assert values are between 0 and 1 
        for v in values:
            if not 0 <= v <= 1:
                raise ValueError("Values must be between 0 and 1.")
    
    def __call__(self, x):
        # Clip value 
        x = max(0, min(1, x))
        # Find the two colors to interpolate between
        i = 0
        while i < len(self.values) and x > self.values[i]:
            i += 1
        if i == 0:
            return self.colors[0]
        if i == len(self.values):
            return self.colors[-1]
        # Interpolate between the two colors
        a = self.values[i-1]
        b = self.values[i]
        c = self.colors[i-1][1:]
        d = self.colors[i][1:]
        # Convert c and d to rgb
        c = [int(c[j:j+2], 16) for j in (0, 2, 4)]
        d = [int(d[j:j+2], 16) for j in (0, 2, 4)]
        # Interpolate 
        t = ()
        for j in range(3):
            t += (int(c[j] + (d[j] - c[j])*(x - a)/(b - a)),)

        # Convert to hex with double digit (pattern is #rrggbb)
        return f"#{t[0]:02x}{t[1]:02x}{t[2]:02x}"


# def value_to_hex_color(value):
#     """
#     Returns a hex color between red (#ff0000) and green (#00ff00)
#     for a float value between 0 and 1.

#     Args:
#         value (float): A float between 0 and 1.

#     Returns:
#         str: A hex color string (e.g., "#ff8000").
#     """
#     if not 0 <= value <= 1:
#         raise ValueError("Value must be between 0 and 1.")

#     # Compute red and green intensity
#     red = int((1 - value) * 255)
#     green = int(value * 255)

#     # Convert to hex format
#     return f"#{red:02x}{green:02x}00"


def show_save_dialog(parent, title, initial_path, filters):
    # Open the Save File dialog
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog  # Optional: Use PyQt's dialog instead of OS-native
    # Show the dialog and get the file path
    file_path, _ = QFileDialog.getSaveFileName(
        parent,
        title,
        initial_path,                          # Default directory
        filters,  # File filter
        options=options
    )
    return file_path

""" Load dialog """
def show_load_dialog(parent, title, initial_path, filters):
    # Open the Save File dialog
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog  # Optional: Use PyQt's dialog instead of OS-native
    # Show the dialog and get the file path
    file_path, _ = QFileDialog.getOpenFileName(
        parent,
        title,
        initial_path,                          # Default directory
        filters,  # File filter
        options=options
    )
    return file_path

def show_save_file_dialog(parent, initial_path, initial_name = "output.csv"):
    # Set initial path of window
    if initial_path is not None:
        initial_path = os.path.join(initial_path, initial_name)

    # Open the Save File dialog
    file_path = show_save_dialog(parent, "Save File As", initial_path, "CSV Files (*.csv);;All Files (*)")

    if file_path:
        netsurf.utils.save_table_to_csv(parent.df, file_path)
        return file_path
    return None

def show_save_image_dialog(parent, initial_path, initial_name = "output.png"):
    
    # Set initial path of window
    if initial_path is not None:
        initial_path = os.path.join(initial_path, initial_name)

    # Open the Save File dialog
    file_path = show_save_dialog(parent, "Save Image As", initial_path, "PNG Files (*.png);;All Files (*)")

    if file_path:
        parent.figure.savefig(file_path, facecolor='white', transparent=False)
        return file_path
    return None

def show_load_bucket_dialog(parent, initial_path):
    # Open the Load File dialog
    file_path = show_load_dialog(parent, "Load Bucket", initial_path, "Bucket files (*.netsurf.bkt);;All Files (*)")

    if file_path:
        # Load object from file using pkl
        bkt = netsurf.utils.load_object(file_path)
        return file_path, bkt
    return None, None

def show_save_bucket_dialog(parent, initial_path, bucket):
    # Open the Save File dialog
    file_path = show_save_dialog(parent, "Save Bucket As", initial_path, "Bucket files (*.netsurf.bkt);;All Files (*)")

    if file_path:
        # Save object to file using pkl
        netsurf.utils.save_object(bucket, file_path)
        return file_path
    return None


def get_main_window_parent(widget: QWidget):
    """Gets the main window parent of a widget.

    Args:
        widget: The widget whose main window parent is to be found.

    Returns:
        The main window parent of the widget, or None if not found.
    """

    window = widget.window()
    while window:
        if isinstance(window, QMainWindow):
            return window
        window = window.parent()
    return None
