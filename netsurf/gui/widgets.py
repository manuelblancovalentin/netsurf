""" Basic modules """
import math
import os
from copy import deepcopy
import asyncio
from io import TextIOWrapper


""" Numpy """
import numpy as np

""" PyQt5 imports """
from PyQt5.QtCore import Qt, QPoint, QUrl, QTimer, QProcess, QByteArray, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QHBoxLayout, QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox
from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QListWidget, QListWidgetItem, QSpacerItem, QSizePolicy
from PyQt5.QtWidgets import QTableWidget, QAbstractItemView, QTableWidgetItem, QHeaderView, QTreeWidget, QTreeWidgetItem
from PyQt5.QtWidgets import QMenu, QAction, QMessageBox, QTabWidget, QStyle, QToolBar, QToolButton, QComboBox, QCheckBox
from PyQt5.QtWidgets import QTextBrowser, QStackedWidget, QFrame, QGridLayout, QStyledItemDelegate, QTextEdit
from PyQt5.QtGui import QBrush, QColor, QPainter, QPen, QPixmap, QIcon, QFont

""" Matplotlib imports """
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import mplcursors  # For interactive cursor
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

""" Other utils """
from io import BytesIO
from PIL import Image
from PIL.ImageQt import ImageQt

""" Pandas """
import pandas as pd 

# Custom imports
import wsbmr
from wsbmr import gui

""" Custom widgets """
class TextEditWidget(QWidget):
    def __init__(self, label_text, callback, default=None):
        super().__init__()
        self.callback = callback

        """ Create layout """
        layout = QHBoxLayout()

        # Left, top, right, bottom
        layout.setContentsMargins(*wsbmr.config.DEFAULT_WIDGET_PADDINGS)

        self.label = QLabel(label_text)
        self.text_edit = QLineEdit()
        self.text_edit.setFixedHeight(wsbmr.config.DEFAULT_TEXTEDIT_HEIGHT)

        # Set the default value if it is not None
        if default:
            self.text_edit.setText(default)

        """ Add widgets to layout """
        layout.addWidget(self.label)
        layout.addWidget(self.text_edit)
        self.setLayout(layout)

        """ Setup callbacks """
        self.text_edit.returnPressed.connect(self.update_value)

    def update_value(self):
        value = self.text_edit.text()
        self.callback(value)


""" Custom widgets """
class DirectorySelectorWidget(QWidget):
    def __init__(self, label_text, callback, default = None):
        super().__init__()
        self.callback = callback

        # Set this widget to have a fixed height of TEXTEDIT + 10
        #self.setFixedHeight(wsbmr.config.DEFAULT_TEXTEDIT_HEIGHT)

        """ Create layout """
        layout = QHBoxLayout()

        # Left, top, right, bottom
        layout.setContentsMargins(*wsbmr.config.DEFAULT_WIDGET_PADDINGS)

        self.label = QLabel(label_text)
        self.text_edit = QLineEdit()
        self.text_edit.setFixedHeight(wsbmr.config.DEFAULT_TEXTEDIT_HEIGHT)
        self.button = QPushButton("Select")
        self.button.setFixedHeight(wsbmr.config.DEFAULT_TEXTEDIT_HEIGHT)

        # Set dir hint according to default value
        if wsbmr.utils.is_valid_directory(default):
            wsbmr.utils.log._log(f"Default directory for {label_text} is valid: {default}")
            self.text_edit.setText(default)
            # Also call the callback
            self.callback(default)
        else:
            wsbmr.utils.log._log(f"Default directory for {label_text} is invalid: {default}")
            self.text_edit.setPlaceholderText("Select a directory")

        """ Add widgets to layout """
        layout.addWidget(self.label)
        layout.addWidget(self.text_edit)
        layout.addWidget(self.button)

        self.setLayout(layout)
        
        """ Setup callbacks """
        self.button.clicked.connect(self.open_file_dialog)
        self.text_edit.returnPressed.connect(self.update_value)

    def open_file_dialog(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.text_edit.setText(directory)
            self.callback(directory)

    def update_value(self):
        value = self.text_edit.text()
        self.callback(value)

""" Integer spinner """
class IntegerSpinnerWidget(QWidget):
    def __init__(self, label_text, callback, default_value=0, min = 0, max = 1000, labelwidth = 60):
        super().__init__()
        self.callback = callback

        # Set the widget height
        #self.setFixedHeight(40)  # Adjust height as needed

        """ Create layout """
        layout = QHBoxLayout()
        # Left, top, right, bottom
        layout.setContentsMargins(*wsbmr.config.DEFAULT_WIDGET_PADDINGS)

        # Create a label
        self.label = QLabel(label_text)
        self.label.setFixedWidth(labelwidth)

        # Create a spin box (spinner)
        self.spinner = QSpinBox()
        self.spinner.setMinimum(min)  # Minimum value is 0
        self.spinner.setMaximum(max)  # Set a high max value (or no limit)
        self.spinner.setValue(default_value)  # Set default value

        # Add widgets to layout
        layout.addWidget(self.label)
        layout.addWidget(self.spinner)

        self.setLayout(layout)

        """ Setup callbacks """
        self.spinner.valueChanged.connect(self.on_value_changed)

    def on_value_changed(self):
        # Get the value from the spinner and call the callback
        value = self.spinner.value()
        self.callback(value)

class LogScaleSpinBox(QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Set the logarithmic scale properties
        self.setDecimals(5)

    def setMinimum(self, min_value):
        """ All values in log """
        super().setMinimum(math.log10(min_value))
    
    def setMaximum(self, max_value):
        """ All values in log """
        super().setMaximum(math.log10(max_value))

    def setSingleStep(self, step):
        """ All values in log """
        super().setSingleStep(math.log10(step))

    def setValue(self, value):
        """ All values in log """
        super().setValue(math.log10(value))

    def value(self):
        """ Get the current value in linear scale. """
        return round(math.pow(10, super().value()), 5)

    def stepUp(self):
        """ Increase the value by the step in linear scale. """
        self.setValue(math.pow(10, super.value()) + math.pow(10, self.singleStep()))

    def stepDown(self):
        """ Decrease the value by the step in linear scale. """
        self.setValue(math.pow(10, super.value()) - math.pow(10, self.singleStep()))
    
    # To text 
    def textFromValue(self, log_value):
        # the value is in log scale, so convert it to linear scale before converting to string
        value = math.pow(10, log_value)
        # round to 5 decimal places
        return str(round(value, 5))

class FloatSpinnerWidget(QWidget):
    def __init__(self, label_text, callback, default_value=None, min = None, max = None, step = 0.1, log = False, labelwidth = 60):
        super().__init__()
        self.callback = callback

        if log:
            min = min if min else 1e-3
            max = max if max else 1e-1
            default_value = default_value if default_value else 1e-3
        else:
            min = min if min else 0.0
            max = max if max else 1.0
            default_value = default_value if default_value else 0.0

        # Set the widget height
        #self.setFixedHeight(40)  # Adjust height as needed

        """ Create layout """
        layout = QHBoxLayout()
        # Left, top, right, bottom
        layout.setContentsMargins(0,0,0,0)

        # Create a label
        self.label = QLabel(label_text)
        self.label.setFixedWidth(labelwidth)

        # Create a spin box (spinner)
        self.spinner = QDoubleSpinBox() if not log else LogScaleSpinBox()
        self.spinner.setMinimum(min)  # Minimum value is 0
        self.spinner.setMaximum(max)  # Set a high max value (or no limit)
        self.spinner.setValue(default_value)  # Set default value
        self.spinner.setSingleStep(step)

        # Add widgets to layout
        layout.addWidget(self.label)
        layout.addWidget(self.spinner)

        self.setLayout(layout)

        """ Setup callbacks """
        self.spinner.valueChanged.connect(self.on_value_changed)

    def on_value_changed(self):
        # Get the value from the spinner and call the callback
        value = float(self.spinner.value())
        self.callback(value)


""" Tab widget """
class CustomListWidgetTab(QWidget):
    def __init__(self, choices, default_choices, callback):
        super().__init__()
        self.callback = callback
        
        # Create layout for the tab
        layout = QVBoxLayout()
        
        # Create a list widget for multiple selections
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)  # Allow multiple selections

        # Enable vertical and horizontal scrollbars if the list exceeds the available space
        self.list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Add items to the list widget
        for choice in choices:
            item = QListWidgetItem(choice)
            self.list_widget.addItem(item)

        # Set default selected items
        for choice in default_choices:
            item = self.list_widget.findItems(choice, Qt.MatchExactly)[0]
            item.setSelected(True)

        # Connect the item selection change to the callback
        self.list_widget.itemSelectionChanged.connect(self.on_choice_changed)

        # Add the header and list widget directly to the main layout
        #layout.addLayout(header_layout)
        layout.addWidget(self.list_widget)
        self.setLayout(layout)

    def on_choice_changed(self):
        selected_choices = [item.text() for item in self.list_widget.selectedItems()]
        self.callback(selected_choices)
    

""" Quantization tab widget """
class CustomQuantizationListWidgetTab(QWidget):
    def __init__(self, default_choices, callback):
        super().__init__()
        self.callback = callback
        # Init config 
        self.config = wsbmr.config.DEFAULT_QUANTIZATIONS
        
        # Create layout for the tab
        layout = QVBoxLayout()

        # Create two spinners for two integers, one for the total number of bits, and another for the number of bits for the integer part
        Hlayout = QHBoxLayout()

        self.total_bits_spinner = IntegerSpinnerWidget('Total bits (m):', callback = lambda x: x, default_value=6, min = 1, max = 32, labelwidth = 85)
        self.integer_bits_spinner = IntegerSpinnerWidget('Integer bits (n):', callback = lambda x: x, default_value=0, min = 0, max = 32, labelwidth = 95)
        self.signed_checkbox = QCheckBox("Signed")
        self.signed_checkbox.setChecked(True)
        
        self.total_bits_spinner.setFixedHeight(30)
        self.total_bits_spinner.setFixedWidth(200)
        self.integer_bits_spinner.setFixedHeight(30)
        self.integer_bits_spinner.setFixedWidth(200)
        self.signed_checkbox.setFixedHeight(30)
        self.signed_checkbox.setFixedWidth(70)

        # Add a button 
        self.button = QPushButton("Add")
        self.button.setFixedHeight(20)
        self.button.setFixedWidth(40)

        # Add spinners to Hlayout
        Hlayout.addWidget(self.total_bits_spinner)
        Hlayout.addWidget(self.integer_bits_spinner)
        Hlayout.addWidget(self.signed_checkbox)
        # Add spacer between spinners and button
        Hlayout.addWidget(self.button)
        Hlayout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Make everything align left 
        Hlayout.setAlignment(Qt.AlignLeft)

        # Add Hlayout to layout
        layout.addLayout(Hlayout)
        
        # Create a list widget for multiple selections
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)  # Allow multiple selections

        # Enable vertical and horizontal scrollbars if the list exceeds the available space
        self.list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Add items to the list widget
        for choice in default_choices:
            item = QListWidgetItem(choice)
            self.list_widget.addItem(item)
            item.setSelected(True)

        # Connect the item selection change to the callback
        self.list_widget.itemSelectionChanged.connect(self.on_choice_changed)
        # Add callback to button
        self.button.clicked.connect(self.on_button_clicked)

        #self.list_widget.setFixedHeight(50)

        # Add the header and list widget directly to the main layout
        #layout.addLayout(header_layout)
        layout.addWidget(self.list_widget)
        
        # Left, top, right, bottom
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Add vertical spacer
        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        self.setLayout(layout)

    def on_button_clicked(self):
        # Get values from spinners
        m = int(self.total_bits_spinner.spinner.value())
        n = int(self.integer_bits_spinner.spinner.value())
        signed = int(self.signed_checkbox.isChecked())

        # Create string 
        q = f"q<{m},{n},{signed}>"
        # Add to list if not present 
        if not self.list_widget.findItems(q, Qt.MatchExactly):
            item = QListWidgetItem(q)
            self.list_widget.addItem(item)
            item.setSelected(True)
        # Call callback with all selected choices
        self.callback([item.text() for item in self.list_widget.selectedItems()])

    def on_choice_changed(self):
        selected_choices = [item.text() for item in self.list_widget.selectedItems()]
        self.callback(selected_choices)


""" Float tab widget """
class CustomFloatListWidgetTab(QWidget):
    def __init__(self, choices, default_choices, callback, default_value = None, min = None, max = None, step = 0.1, log = False):
        super().__init__()
        self.callback = callback

        # Create layout for the tab
        layout = QVBoxLayout()

        # Create a single spinner for a float value between the max and min values
        Hlayout = QHBoxLayout()

        self.float_spinner = FloatSpinnerWidget('Value:', lambda x: None, default_value=default_value, min = min, max = max, step = step, log = log, labelwidth=50)
        self.float_spinner.setFixedHeight(30)
        self.float_spinner.setFixedWidth(200)

        # Add a button 
        self.button = QPushButton("Add")
        self.button.setFixedHeight(20)
        self.button.setFixedWidth(40)

        # Add spinners to Hlayout
        Hlayout.addWidget(self.float_spinner)
        # Add spacer between spinners and button
        Hlayout.addWidget(self.button)
        Hlayout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Make everything align left 
        Hlayout.setAlignment(Qt.AlignLeft)

        # Add Hlayout to layout
        layout.addLayout(Hlayout)
        
        # Create a list widget for multiple selections
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)  # Allow multiple selections

        # Enable vertical and horizontal scrollbars if the list exceeds the available space
        self.list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Add items to the list widget
        for choice in choices:
            item = QListWidgetItem(choice)
            self.list_widget.addItem(item)
            
        for choice in default_choices:
            item = self.list_widget.findItems(choice, Qt.MatchExactly)[0]
            if item:
                item.setSelected(True)

        # Connect the item selection change to the callback
        self.list_widget.itemSelectionChanged.connect(self.on_choice_changed)
        # Add callback to button
        self.button.clicked.connect(self.on_button_clicked)

        #self.list_widget.setFixedHeight(80)

        # Add the header and list widget directly to the main layout
        #layout.addLayout(header_layout)
        layout.addWidget(self.list_widget)
        
        # Add vertical spacer
        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        self.setLayout(layout)

    def set_value(self, key, value):
        self.config[key] = value

    def on_button_clicked(self):
        # Create string 
        val = str(self.float_spinner.spinner.value())
        # Add to list if not present 
        if not self.list_widget.findItems(val, Qt.MatchExactly):
            item = QListWidgetItem(val)
            self.list_widget.addItem(item)
            item.setSelected(True)
        # Call callback with all selected choices
        self.callback([float(item.text()) for item in self.list_widget.selectedItems()])

    def on_choice_changed(self):
        selected_choices = [float(item.text()) for item in self.list_widget.selectedItems()]
        self.callback(selected_choices)

     
class LegendLineWithDot(QFrame):
    """
    Custom widget to represent a legend line with a dot in the middle.
    """
    def __init__(self, color, linestyle, parent=None):
        super().__init__(parent)
        self.color = QColor(color)  # Convert color string to QColor
        self.linestyle = linestyle
        self.setFixedSize(30, 10)  # Fixed size for the legend item

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw the line
        pen = QPen()
        pen.setColor(self.color)
        pen.setWidth(2)
        if self.linestyle == '--':  # Dashed line
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawLine(0, self.height() // 2, self.width(), self.height() // 2)

        # Draw the dot in the middle
        dot_x = self.width() // 2
        dot_y = self.height() // 2
        dot_radius = 2
        painter.setBrush(QBrush(self.color, Qt.SolidPattern))
        painter.drawEllipse(dot_x - dot_radius, dot_y - dot_radius, dot_radius * 2, dot_radius * 2)

""" custom legend qt object """
class CustomLegend(QWidget):
    def __init__(self, lines, type, parent=None, num_columns = 2, metric = None, ymax = None):
        """
        Initializes the custom legend from a list of Matplotlib line objects.
        
        Parameters:
        - lines: List of Matplotlib Line2D objects
        """
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self.num_columns = num_columns if type == '2d' else 1
        self.metric = metric
        self.ymax = ymax
        self.type = type

        # Add a title to the legend
        title = QLabel("<h1><b>Legend</b></h1>")
        title.setAlignment(Qt.AlignCenter)
        self.layout().addWidget(title)

        # Explanation of the labels of the legend
        explanation = QLabel("<h3>TMR (AUC%)</h3>") if self.type == '2d' else QLabel("<h3>Methods</h3>")
        explanation.setAlignment(Qt.AlignCenter)
        self.layout().addWidget(explanation)
        
        # Loop thru lines and add checkboxes
        self.legend_items = self.add_legend_items(lines)
        self.layout().addWidget(self.legend_items)

        # Add spacer to push the legend to the top
        self.layout().addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Minimum))

        # Add some text to explain the legend
        explanation = QLabel("Click on the checkboxes to toggle the visibility of the lines.")
        explanation.setWordWrap(True)
        explanation.setAlignment(Qt.AlignCenter)
        self.layout().addWidget(explanation)

        # Add a little bar plot showing the AUC per line (TMR)
        # Create a figure and axis
        if self.type == '2d':
            self.auc_bar_mini_plot = self.create_auc_bar_miniplot(lines)
            # Add the mini plot to the layout
            self.layout().addWidget(self.auc_bar_mini_plot)
            self.auc_bar_mini_plot.setFixedHeight(300)
        self.layout().setContentsMargins(0, 0, 0, 0)  # No margins around the layout

    """ Add legend item """
    def add_legend_items(self, lines):
        """
        Creates a legend item for a given Matplotlib line object.
        
        Parameters:
        - line: A dict containing:
            "line": A Matplotlib Line2D object
            "fill": A Matplotlib Line2D object for the fill between the line and the x-axis
            "auc": The area under the curve (float)
        """

        legend = QWidget()

        # Create a grid layout 
        layout = QGridLayout()
        legend.setLayout(layout)

        # Loop thru lines and add a checkbox for each
        num_columns = self.num_columns
        num_lines = len(lines)
        for iline, (line_name, obj) in enumerate(lines.items()):
            
            line = obj['line']
            # fill = obj['fill']
            # auc = obj['auc']

            # Extract line properties
            label = line.get_label()
            color = line.get_color()
            linestyle = line.get_linestyle()
            
            # Legend entry
            legend_entry = QWidget()

            # Create a horizontal layout for the legend entry
            sublayout = QHBoxLayout()
            legend_entry.setLayout(sublayout)
            
            # Create the line with the dot in the middle
            line_with_dot = LegendLineWithDot(color, linestyle)
            sublayout.addWidget(line_with_dot)
            
            # Create a checkbox for toggling visibility
            checkbox = QCheckBox(label)
            checkbox.setChecked(line.get_visible())  # Sync initial state with line visibility
            
            # Connect checkbox state to line visibility
            checkbox.toggled.connect(lambda state, l=obj: self.toggle_line(l, state))
            
            # Add checkbox to layout
            sublayout.addWidget(checkbox)

            # Add to main layout
            layout.addWidget(legend_entry, iline // num_columns, iline % num_columns)
        
        return legend
    
    """ Toggle line visibility """
    def toggle_line(self, obj, state):
        """
        Toggles the visibility of a Matplotlib line.
        
        Parameters:
        - obj: A dict with :
            - "line": Matplotlib Line2D object
            - "fill": Matplotlib Line2D object for the fill between the line and the x-axis
            - "auc": The area under the curve (float)
        - state: The visibility state (True/False)
        """
        line = obj['line']
        line.set_visible(state)
        if self.type == '2d':
            fill = obj['fill']
            #auc = obj['auc']
            fill.set_visible(state)
        line.figure.canvas.draw_idle()  # Redraw the canvas

    """ Create a mini plot showing the AUC per line """
    def create_auc_bar_miniplot(self, line):
        """
        Creates a mini plot showing the AUC per line.
        
        Parameters:
        - lines: List of Matplotlib Line2D objects
        
        Returns:
        - A Matplotlib Figure object
        """
        # Create a Matplotlib figure and axis
        fig, ax = plt.subplots(figsize=(3,3))

        # Mark figure as deletable
        wsbmr.utils.mark_figure_as_deletable(fig)

        # Loop thru lines and add a bar for each
        xt = []
        _ym = -np.inf
        for i, (line_name, elements) in enumerate(line.items()):
            auc = elements['auc']
            if self.metric == 'accuracy':
                auc = 100*auc
                fmt = f'{100*float(line_name):3.1f}'
            else:
                fmt = f'{float(line_name):.2f}'
            if auc > _ym:
                _ym = auc
            ax.bar(i, auc, color=elements['line'].get_color(), alpha = 0.8)
            xt.append(fmt)
        
        # Set the xticks
        ax.set_xticks(range(len(xt)))
        ax.set_xticklabels(xt, rotation=45, ha='right')

        # Add grid 
        ax.grid(True)

        # Set the title and labels
        ax.set_title("AUC per line")
        ax.set_ylabel("AUC (%)" if self.metric == 'accuracy' else "AUC")
        ax.set_xlabel("TMR (%)")
        ymax = self.ymax if self.ymax else _ym 
        ymax = ymax if self.metric != 'accuracy' else 100
        ax.set_ylim(0, ymax)

        # Apply tight layout
        fig.tight_layout()

        # Create the FigureCanvas
        canvas = FigureCanvas(fig)

        return canvas


""" Plot """
class PlotWindow(QWidget):
    def __init__(self, type, parent=None, figure = None, ax = None, title = None, info_tag = None, 
                 legend_lines = {}, info = {}, metric = None, ymax = None, ymin = None, bucket = None):
        super().__init__(parent)
        self.title = title
        self.setWindowTitle(title)
        self.setAttribute(Qt.WA_DeleteOnClose)  # Ensure the widget is deleted on close
        self.setWindowFlags(Qt.Window)  # Make this a standalone window
        self.info = info
        self.type = type
        self.metric = metric
        self.ymax = ymax
        self.ymin = ymin
        self.bucket = bucket

        self.main_layout = QVBoxLayout(self)
        
        # We will add the interactive legend to the right of the plot, so the user can select and unselect lines 
        # to be plotted
        self.plot_layout = QHBoxLayout()
        # Add this to the main layout
        self.main_layout.addLayout(self.plot_layout)

        """ Plot and legend """
        # Create a Matplotlib figure and canvas
        if figure is None or ax is None:
            self.figure, self.ax = plt.subplots()
        else:
            self.figure = figure
            self.ax = ax
            # If ax has legend, delete
            if self.ax.legend_:
                self.ax.legend_.remove()
        
        # Create a vertical layout for info label + plot
        self.plot_and_info_widget = QWidget()
        self.plot_and_info_layout = QVBoxLayout()
        self.plot_and_info_layout.setSpacing(0)  # No space between widgets
        self.plot_and_info_layout.setContentsMargins(0, 0, 0, 0)  # No margins around the layout

        # Create a canvas for the info label
        info_fig, info_ax = plt.subplots() 
        
        # Mark info_fig as deletable figure 
        wsbmr.utils.mark_figure_as_deletable(info_fig)

        self.info_label = FigureCanvas(info_fig)
        self.plot_and_info_layout.addWidget(self.info_label)

        # set background to transparent
        info_fig.patch.set_facecolor('none')  # or fig.patch.set_alpha(0)
        self.figure.patch.set_facecolor('none')  # or fig.patch.set_alpha(0)
        # Set transparent axis background
        info_ax.set_facecolor('none')
        self.ax.set_facecolor('none')

        # Populate info label
        if info_tag:
            padding = 0.3
            border_color="black"
            text_properties = {
                "x": 0.5, #info_tag.get_position()[0],
                "y": 0.0, #info_tag.get_position()[1],
                "s": info_tag.get_text(),
                "fontsize": info_tag.get_fontsize(),
                "color": info_tag.get_color(),
                "ha": "center", #info_tag.get_ha(),
                "va": "bottom", #info_tag.get_va()
                "bbox": dict(
                        boxstyle=f"round,pad={padding}",
                        edgecolor=border_color,
                        facecolor="white")
            }
            info_ax.text(**text_properties)
            self.info_label.setFixedHeight(100)
            # Remove axis outline 
            info_ax.axis('off')
        else:
            self.info_label.setFixedHeight(0)

        # Create canvas and layout for the main plot 
        # Make sure self.figure is tight layout
        info_fig.tight_layout()
        self.figure.tight_layout()
        self.canvas = FigureCanvas(self.figure)
        # Add the canvas to the layout
        self.plot_and_info_layout.addWidget(self.canvas)

        # Add the plot and info layout to the main layout
        self.plot_and_info_widget.setLayout(self.plot_and_info_layout)
        self.plot_layout.addWidget(self.plot_and_info_widget)

        # Add the interactive_legend to the right of the plot, in the plot_layout
        if self.type in ['2d', 'vus_vs_pruning']:
            self.interactive_legend = CustomLegend(legend_lines, self.type, metric = self.metric, ymax = self.ymax)
            self.plot_layout.addWidget(self.interactive_legend)
            # Set default width of interactive_legend
            self.interactive_legend.setFixedWidth(300)
            
        # Create toolbar 
        self.toolbar = self.create_toolbar()
        # Add the toolbar to the main layout
        self.main_layout.addWidget(self.toolbar)

    """ Interactive legend """
    def create_interactive_legend(self, info_tag, legend_lines):
        # Add a checkbox for each line in the legend
        interactive_legend = QWidget()
        # Create a layout for the checkboxes
        legend_layout = QVBoxLayout()
        interactive_legend.setLayout(legend_layout)
        # Loop thru lines and add a checkbox for each
        for line_name, elements in legend_lines.items():
            checkbox = QCheckBox(str(line_name), self)
            checkbox.setChecked(True)
            checkbox.toggled.connect(lambda state, line_name = line_name, elements = elements: self.toggle_line(line_name, elements, state))
            # Add the checkbox to the layout
            legend_layout.addWidget(checkbox)
        
        return interactive_legend


    """ Navigation toolbar """
    def create_toolbar(self):

        # Create a toolbar and add it to the layout
        toolbar = QToolBar(self)

        # Create a checkbox to toggle y-axis scale
        self.y_scale_checkbox = QCheckBox("ylog-scale", self)
        self.y_scale_checkbox.setChecked(self.ax.get_yscale() == 'log')  # Initially linear scale
        self.y_scale_checkbox.toggled.connect(lambda: self.toggle_scale(ax='y'))

        self.x_scale_checkbox = QCheckBox("xlog-scale", self)
        # Check if axis is log
        self.x_scale_checkbox.setChecked(self.ax.get_xscale() == 'log')  # Initially linear scale
        self.x_scale_checkbox.toggled.connect(lambda: self.toggle_scale(ax='x'))
        
        # Add a save button 
        self.save_button = QPushButton("Save", self)
        self.save_button.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.save_button.setToolTip("Save plot to file")
        #self.save_button.setCursor(Qt.PointingHandCursor)
        self.save_button.clicked.connect(self.on_save)

        # Add the checkbox to the toolbar
        toolbar.addWidget(self.y_scale_checkbox)
        toolbar.addWidget(self.x_scale_checkbox)
        toolbar.addWidget(self.save_button)

        return toolbar
        
    def toggle_scale(self, ax='y'):
        """Toggles the y-axis scale between linear and log."""
        if getattr(getattr(self,f'{ax}_scale_checkbox'),'isChecked')():
            getattr(self.ax,f'set_{ax}scale')('log')  # Set log scale
        else:
            getattr(self.ax,f'set_{ax}scale')('linear')  # Set linear scale
        self.canvas.draw()  # Redraw the canvas to reflect the changes

    """ Toggle line """
    def toggle_line(self, line_name, elements, state):
        """Toggles the visibility of a line in the plot."""
        # Find the line in the plot
        for i,l in enumerate(self.ax.lines):
            lbl = l.get_label().split(" ")[0]
            if str(lbl).strip() == str(line_name).strip():
                # Get line from elements
                line = elements['line']
                # Set the visibility of the line
                line.set_visible(state)
                # Set linestyle to solid if visible, else None
                line.set_linestyle('-' if state else 'None')
                # same for fill
                fill = elements['fill']
                fill.set_visible(state)
                fill.set_linestyle('-' if state else 'None')
                wsbmr.utils._info(f"Setting visibility of {line_name} to {state}")

        self.canvas.draw()  # Redraw the canvas to reflect the changes

    def show(self):
        super().show()
        # Enable interactive cursors
        mplcursors.cursor(self.ax.lines, hover=True)  # Adds a cursor for the plotted line
        # Refresh the canvas
        self.canvas.draw()
        
    def closeEvent(self, event):
        """Called when the window is closed."""
        if self.parent():
            self.parent().remove_closed_window(self)  # Notify parent to remove the window
        event.accept()  # Accept the close event

    def on_save(self):
        # Try to get bucket for initial dir 
        initial_dir = None
        if self.bucket:
            initial_dir = self.bucket.dir
        initial_name = self.type.lower() + '.png'
        # save 
        wsbmr.utils.show_save_image_dialog(self, initial_dir, initial_name=initial_name)
        
""" This is a Window that can hold multiple PlotWindow instances, and switch between them """
class MultiPlotWindow(QWidget):
    def __init__(self, parent=None, title = "Multi Plot Viewer"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowFlags(Qt.Window)

        # Main layout 
        self.layout = QHBoxLayout(self)

        # Left panel is a list showing miniatures of each plot 
        self.thumbnail_list_widget = QListWidget()
        self.thumbnail_list_widget.setSelectionMode(QListWidget.SingleSelection)  # Allow single selection
        self.thumbnail_list_widget.setIconSize(QPixmap(100, 100).size())  # Set size for thumbnails
        self.thumbnail_list_widget.itemClicked.connect(self.on_item_clicked)
        self.thumbnail_list_widget.setFixedWidth(200)
        self.layout.addWidget(self.thumbnail_list_widget)

        # Main layout
        self.panel = QWidget()

        self.plot_layout = QVBoxLayout(self.panel)
        
        # Create QStackedWidget to hold multiple PlotWindow instances
        self.stacked_widget = QStackedWidget(self)
        self.plot_layout.addWidget(self.stacked_widget)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("←", self)
        self.prev_button.clicked.connect(self.show_previous)
        nav_layout.addWidget(self.prev_button)
        
        self.next_button = QPushButton("→", self)
        self.next_button.clicked.connect(self.show_next)
        nav_layout.addWidget(self.next_button)
        self.plot_layout.addLayout(nav_layout)

        # Assign layout
        self.panel.setLayout(self.plot_layout)

        # Add panel to layout
        self.layout.addWidget(self.panel)

        self.current_index = 0  # Track current plot index

    def add_plot_window(self, plot_window: PlotWindow):
        """Adds a PlotWindow instance to the stacked widget."""
        self.stacked_widget.addWidget(plot_window)
        if self.stacked_widget.count() == 1:
            plot_window.show()  # Show the first plot window by default
        
        # Add thumbnail
        thumbnail, thumbnail_title = self.generate_thumbnail(plot_window)
        item = QListWidgetItem()
        item.setIcon(QIcon(thumbnail))
        item.setText(thumbnail_title)
        self.thumbnail_list_widget.addItem(item)
        # Change callback of this item to show the corresponding plot window
        i = self.stacked_widget.count() - 1
        #item.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(self.thumbnail_list_widget.currentRow()))
        #item.clicked.connect(lambda i=i, item=item: self.on_item_clicked(i, item))
        
        
    def generate_thumbnail(self, plot_window):
        # Get figure from the plot window
        fig = plot_window.figure

        # # Render the figure to a memory buffer using FigureCanvasAgg
        # canvas = FigureCanvas(fig)
        # buf = BytesIO()
        # canvas.print_png(buf)  # Save figure as a PNG into the buffer
        # buf.seek(0)
        
        # # Convert buffer into a QPixmap
        # image = Image.open(buf)
        # qt_image = QPixmap.fromImage(ImageQt(image))
        # buf.close()

        # Create a new figure
        fig_copy = plt.figure(figsize=(2, 2), dpi=50)
        canvas_copy = FigureCanvas(fig_copy)
        
        # Clone the original figure's axes into the new figure
        for ax in fig.axes:
            if plot_window.type == '3d':
                new_ax = fig_copy.add_subplot(111, projection='3d')
            else:
                new_ax = fig_copy.add_subplot(111)
                
            for line in ax.get_lines():
                new_ax.plot(line.get_xdata(), line.get_ydata(), label=line.get_label())
            
            # Also plot polys 
            if plot_window.type != '3d':
                polygons = [poly for poly in ax.get_children() if isinstance(poly, PolyCollection)]
                for poly in polygons:
                    # Extract paths (vertices) and properties
                    paths = [path.vertices for path in poly.get_paths()]
                    properties = poly.get_facecolor()  # Get color (or use poly.get_edgecolor())

                    # Recreate a new PolyCollection
                    new_poly = PolyCollection(paths, facecolors=properties, alpha=poly.get_alpha())
                    new_ax.add_collection(new_poly)
            elif plot_window.type == '3d':
                # Not implemented yet
                pass
                # # Get all the "artists" (elements like lines, surfaces, etc.) from the original axis
                # for artist in ax.get_children():
                #     # If the artist is a Poly3DCollection (a surface in 3D)
                #     if isinstance(artist, Poly3DCollection):
                #         # Extract the polygons (faces) and other attributes
                #         faces = artist.get_paths()
                #         facecolors = artist.get_facecolor()
                #         edgecolors = artist.get_edgecolor()
                #         alpha = artist.get_alpha()

                #         # Recreate the Poly3DCollection on the new axis
                #         new_poly = Poly3DCollection(faces, facecolors=facecolors, edgecolors=edgecolors, alpha=alpha)
                #         new_ax.add_collection3d(new_poly)


            # Make sure ax scale is the same 
            if plot_window.type != '3d':
                new_ax.set_xlim(ax.get_xlim())
                new_ax.set_ylim(ax.get_ylim())
            # add grid 
            new_ax.grid(True)
        
        # Render the copied figure into a buffer
        buf = BytesIO()
        canvas_copy.print_png(buf)
        buf.seek(0)
        
        # Convert buffer into an image
        image = Image.open(buf)
        qt_image = QPixmap.fromImage(ImageQt(image))
        buf.close()
        
        # Close the copied figure to save memory
        plt.close(fig_copy)

        # Generate the title with the plot_window info
        info = plot_window.info
        thumbnail_title = ""
        for key in ['benchmark', 'quantization', 'model', 'method', 'run']:
            if key in info:
                thumbnail_title += f'{info[key]}\n'
       
        return qt_image, thumbnail_title
    
    def on_item_clicked(self, item):
        """
        Handle thumbnail click events.
        """
        #print(f"Clicked on: {i} {item.text()}")
        # Update the main plot view here as needed
        i = self.thumbnail_list_widget.currentRow()
        self.show_index(i)

    def show_index(self, index):
        """Show the plot at the given index."""
        if 0 <= index < self.stacked_widget.count():
            self.stacked_widget.setCurrentIndex(index)
            self.current_index = index

    def show_previous(self):
        """Navigate to the previous plot."""
        if self.stacked_widget.count() > 0:
            current_index = (self.current_index - 1) % self.stacked_widget.count()
            self.show_index(current_index)
            # Set currentRow of self.thumbnail_list_widget
            self.thumbnail_list_widget.setCurrentRow(current_index)

    def show_next(self):
        """Navigate to the next plot."""
        if self.stacked_widget.count() > 0:
            current_index = (self.current_index + 1) % self.stacked_widget.count()
            self.show_index(current_index)
            # Set currentRow of self.thumbnail_list_widget
            self.thumbnail_list_widget.setCurrentRow(current_index)



""" Dummy item widget to store subbucket for easy access later """
class BucketTreeItemWidget(QTreeWidgetItem):
    def __init__(self, parent, *args, bucket = None, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.bucket = bucket

""" Tree Widget (recursive) """
class BucketTreeWidget(QTreeWidget):
    def __init__(self, parent=None, ):
        self._parent = parent
        super().__init__(parent)
        self.setHeaderLabels(["Name", "Type"])
        self.setColumnCount(2)
        self.plot_windows = self._parent.plot_windows

        # Enable custom context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.open_context_menu)

        self.update_tree(None)

    def open_context_menu(self, position):
        # Get the cell at the position
        item = self.itemAt(position)
        if item:
            # Create the context menu
            context_menu = wsbmr.gui.menu.OpenRowContextMenu(self, item)
            context_menu += wsbmr.gui.menu.SaveBucketContextMenu(self, item)
            context_menu += wsbmr.gui.menu.ReloadBucketContextMenu(self, item)
            context_menu += wsbmr.gui.menu.PlotCoverageRowContextMenu(self, item)
            context_menu += wsbmr.gui.menu.PlotResultsRowContextMenu(self, item)
            
            # check type of row to add the proper context menu 
            print(f"Item type: {item.bucket.type}")
            if item.bucket.type == 'experiment':
                #context_menu += PlotCoverageRowContextMenu(self, item)
                context_menu += wsbmr.gui.menu.RunJobRowContextMenu(self, item) 

            #context_menu = OpenRowContextMenu(self,item) + DeletableRowContextMenu(self, item) + PlotCoverageRowContextMenu(self, item) + RunJobRowContextMenu(self, item) + PlotResultsRowContextMenu(self, item)

            # Show the context menu at the cursor position
            context_menu.exec_(self.viewport().mapToGlobal(position))
    
    def _update_tree_item_recursively(self, bucket, parent_tree_widget_item):
        if parent_tree_widget_item is None:
            # This is root 
            bucketcopy = deepcopy(bucket)
            root_item = BucketTreeItemWidget([bucket.name, bucket.type], bucket = bucketcopy)
            for c in bucket:
                self._update_tree_item_recursively(bucket[c], root_item)
            return root_item

        # Create bucket item 
        # Make a copy of the bucket, otherwise it will be updated in place
        bucketcopy = deepcopy(bucket)
        subitem = BucketTreeItemWidget([bucket.name, bucket.type], bucket = bucketcopy)
        
        # Loop thru children 
        # unless this is a run, then we don't need to go deeper
        if bucket.type != 'experiment':
            for c in bucket:
                self._update_tree_item_recursively(bucket[c], subitem)
        # Add to parent
        parent_tree_widget_item.addChild(subitem)

    # recursively update tree structure according to the bucket
    def update_tree(self, bucket):
        # Update bucket
        self.bucket = bucket

        # Clear the tree
        self.clear()

        # Update the tree
        if bucket is not None:
            # Loop thru children
            root_item = self._update_tree_item_recursively(bucket, None)
            self.addTopLevelItem(root_item)
        
         # Set default width of tree_widget
        self.setColumnWidth(0, 300)
        self.setColumnWidth(1, 100)
    
    def cleanup_closed_windows(self):
        # Call super
        self._parent.cleanup_closed_windows()
    
    


""" Toolbar widget for the Tab views """
class ToolBarWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        # Main layout for the toolbar
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)  # Margins around the toolbar
        layout.setSpacing(10)  # Space between buttons

        # Button: Save
        self.save_button = QPushButton()
        self.save_button.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.save_button.setToolTip("Save changes")
        #self.save_button.setCursor(Qt.PointingHandCursor)
        self.save_button.clicked.connect(lambda *args, **kwargs: self.on_action('on_save', *args, **kwargs))

        # Button: Reset
        self.reset_button = QPushButton()
        self.reset_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.reset_button.setToolTip("Reset changes")
        #self.reset_button.setCursor(Qt.PointingHandCursor)
        self.reset_button.clicked.connect(lambda *args, **kwargs: self.on_action('on_reset', *args, **kwargs))

        # Add buttons to the layout
        layout.addWidget(self.save_button)
        layout.addWidget(self.reset_button)

        # Add a spacer to push the buttons to the left
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addSpacerItem(spacer)

        # Set the layout
        self.setLayout(layout)

    def on_action(self, action, *args, **kwargs):
        if self.parent:
            if hasattr(self.parent, action):
                wsbmr.utils.log._log(f"Calling action {action} with args {args} and kwargs {kwargs}")
                getattr(self.parent, action)(*args, **kwargs)

""" This is so we can make the results/coverage rows change bg color based on 
    some entry of the table itself.
"""
class RowColorDelegate(QStyledItemDelegate):
    def __init__(self, parent=None, 
                 column_name = 'coverage', 
                 color_fcn = None):
        super().__init__(parent)
        self.column_name = column_name
        self.color_fcn = color_fcn

    def set_background(self, option, index):
        # Determine which row the item is in
        row = index.row()
        table = index.model().parent()  # Get the QTableWidget

        # Find index of column_name in table
        # Get header from table 
        headers = []
        for col in range(table.columnCount()):
            header_item = table.horizontalHeaderItem(col)
            if header_item:
                headers.append(header_item.text().lower().strip().split(' ')[0])
            else:
                headers.append("")  # For empty or undefined header
        
        column_index = headers.index(self.column_name) if self.column_name in headers else None
        
        if column_index is not None:
            # Get the value from the "Status" column (assuming column 2)
            status_item = table.item(row, column_index)
            if status_item:
                try:
                    if hasattr(status_item, 'value'):
                        status = status_item.value
                    else:
                        status = status_item.text().replace('%', '')
                        status = float(status)
                    option.backgroundBrush = QColor(self.color_fcn(status))
                except Exception as e:
                    wsbmr.utils.log._log(f"Error: {e}")
                    option.backgroundBrush = QColor("#ffffff")

                # if status in self.color_dict:
                #     option.backgroundBrush = self.color_dict[status]
                # else:
                #     option.backgroundBrush = QColor("#ffffff")
                #     wsbmr.utils.log._log(f"Status {status} not found in color_dict. Setting to white.")
        else:
            option.backgroundBrush = QColor("#ffffff")
            wsbmr.utils.log._log(f"Column {self.column_name} not found in table columns. Setting to white.")

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        self.set_background(option, index)
        
        
""" Show percentage / store value as float """
# class PercentTableWidgetItem(QTableWidgetItem):
#     def __init__(self, value: float):
#         super().__init__(f"{100*value:3.2f}%")  # Set the formatted text
#         self.value = value  # Store the raw numeric value
#         self.text = f"{100*value:3.2f}%"

#     def data(self, role):
#         if role == Qt.DisplayRole:
#             return super().data(role)  # Return the formatted string for display
#         if role == Qt.UserRole:
#             return self.value  # Return the raw numeric value for sorting or other operations
#         return super().data(role)
    

""" Custom table item """
class CustomTableWidgetItem(QTableWidgetItem):
    def __init__(self, display_text, raw_value):
        super().__init__(display_text)
        self.raw_value = raw_value  # Store the raw value for sorting

    def __lt__(self, other):
        if isinstance(other, CustomTableWidgetItem):
            return self.raw_value < other.raw_value
        return super().__lt__(other)

""" Custom table with custom delegate to be able to color rows based on some value 
    as well as to show percentages
"""
class CustomTableWidget(QTableWidget):
    def __init__(self, data, percent_columns, color_column, color_fcn = None, column_width = {}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color_fcn = color_fcn
        self.column_width = column_width
        self.percent_columns = percent_columns
        self.color_column = color_column
        self.set_data(data, percent_columns, color_column)

    def set_data(self, data, percent_columns = None, color_column = None):
        self.data = data

        if percent_columns is not None and self.percent_columns is None:
            if len(percent_columns) > 0:
                self.percent_columns = percent_columns # Columns to be formatted as percentages
        
        if color_column is not None and self.color_column is None:
            self.color_column = color_column        # Column used for coloring rows
        ncols = 0
        nrows = 0
        if data is not None:
            cols = data.columns
            ncols = len(cols)
            nrows = len(data)
            self.setHorizontalHeaderLabels(cols)
            
        self.setColumnCount(ncols)
        self.setRowCount(nrows)
        
        if data is not None:
            self.populate_table()

    def populate_table(self):
        
        # Reset whole table widget
        self.clear()
        
        # Horizontal header labels
        cks = ['accuracy', 'protection', 'ber', 'coverage']
        cl = [c.replace('_', ' ') for c in self.data.columns]
        cl = [c.title() for c in cl]
        cl = [c + ' (%)' if any([ck in c.lower() for ck in cks]) else c for c in cl]
        self.setHorizontalHeaderLabels(cl)

        # We will use specific width for each column
        data = self.data
        cols = data.columns
        num_cols = len(cols)
        num_rows = len(data)

        sum_width = 0
        for i in range(num_cols):
            if cols[i] in self.column_width:
                self.setColumnWidth(i, self.column_width[cols[i]])
                sum_width += self.column_width[cols[i]]
            else:
                self.resizeColumnToContents(i)
                sum_width += self.columnWidth(i)

        for i in range(num_rows):
            row_data = data.iloc[i]
            for j in range(num_cols):
                # Get col name 
                key = cols[j]
                # Get value 
                value = data.iat[i, j]
                # Format percentage columns
                if self.percent_columns is not None:
                    if key in self.percent_columns and isinstance(value, (float, int)):
                        value_display = f"{value * 100:3.2f}%"
                    elif isinstance(value, float):
                        # Check if this is a float or int
                        value_display = f'{value:.4f}'
                    else:
                        value_display = str(value)
                elif isinstance(value, float):
                    value_display = f'{value:.4f}'
                else:
                    value_display = str(value)
                
                item = CustomTableWidgetItem(value_display, value)
                item.setTextAlignment(Qt.AlignCenter)
                self.setItem(i, j, item)
            
            # Apply row coloring based on the progress column
            if self.color_column is not None:
                progress_value = row_data.get(self.color_column, 0)
                self.apply_row_color(i, progress_value)


    def apply_row_color(self, row_idx, progress_value):
        if self.color_fcn is not None:
            color = QColor(self.color_fcn(progress_value))
            for col_idx in range(self.columnCount()):
                self.item(row_idx, col_idx).setBackground(color)
            return
        

""" Data group panel """
class PandasDataFrameGroupWidget(QGroupBox):
    """ Pandas DataFrame table widget """
    def __init__(self, title, df, columns = None):
        super().__init__(title)
        if columns is not None:
            if df is not None:
                # Keep only valid columns in df
                columns = [c for c in columns if c in df.columns]
                df = df[columns]
        else:
            columns = df.columns if df is not None else []
        
        # Store the DataFrame
        self.df = df
        self.columns = columns
        self.title = title

        # Store the plot windows references (this is required so pyqt5 doesn't close them automatically, 
        # cause if we don't keep a ref, it treats them as garbage even before they are displayed)
        self.plot_windows = []

        # Create layout for the group box
        layout = QVBoxLayout()

        # Create row color delegate
        color_interpolator = None
        color_column = None
        if self.title == 'Coverage':
            # Red, yellow, green (pastel)
            color_interpolator = wsbmr.utils.ColorInterpolator(colors = ['#ffcccc', '#ffffcc', '#ccffcc'], values = [0, 0.5, 1])
            color_column = 'coverage'
            #self.row_color_delegate = RowColorDelegate(self.table, column_name = 'coverage', color_fcn = color_interpolator)
            #self.table.setItemDelegate(self.row_color_delegate)
        elif self.title == 'Results':
            # Orange to Blue (pastel)
            color_interpolator = wsbmr.utils.ColorInterpolator(colors = ['#ffddcc', '#ffffff', '#ccccff'], values = [0, 0.5, 1])
            color_column = 'accuracy'
            #self.row_color_delegate = RowColorDelegate(self.table, column_name = 'accuracy', color_fcn = color_interpolator)
            #self.table.setItemDelegate(self.row_color_delegate)
        elif self.title == "Stats":
            # Green to Red (pastel)
            color_interpolator = wsbmr.utils.ColorInterpolator(colors = ['#ccffcc', '#ffffff', '#ffcccc'], values = [0, 0.5, 1])
            color_column = 'mean'
            #self.row_color_delegate = RowColorDelegate(self.table, column_name = 'protection', color_fcn = color_interpolator)
            #self.table.setItemDelegate(self.row_color_delegate)

        # Create a table widget to display the DataFrame
        #self.table = QTableWidget()
        self.table = CustomTableWidget(df, percent_columns = ['accuracy', 'protection', 'ber', 'coverage'], 
                                       color_column = color_column, color_fcn = color_interpolator)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)  # Disable editing
        #self.table.setAlternatingRowColors(True)  # Alternate row colors

        # Options
        self.table.setSortingEnabled(True)  # Enable sorting by columns
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)  # Select entire rows
        
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)  
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)  # Stretch columns to fill the available space        
        
        # Ensure main window (parent) doesn't change size when table is resized
        self.table.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
        self.table.setVerticalScrollMode(QTableWidget.ScrollPerPixel)

        # Enable horizontal scrollbar policy
        self.table.setSizePolicy(self.table.sizePolicy().horizontalPolicy(), self.table.sizePolicy().verticalPolicy())
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Enable custom context menu
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.open_context_menu)

        # Add the table widget to the layout
        layout.addWidget(self.table)

        # Add a horizontal tool panel to do stuff to the table (like save current table to CSV, reset the table columns hidden, etc.)
        toolbox = ToolBarWidget(self)

        # Add the toolbox to the layout
        layout.addWidget(toolbox)

        # Set the layout for the group box
        self.setLayout(layout)

        # Update the table with the DataFrame
        self.update_table()

    def on_save(self, *args, **kwargs):

        # If bucket is not None, get the initial path 
        initial_path = os.getcwd()
        if hasattr(self, 'parent'):
            p = self.parent()
            if hasattr(p,'parent'):
                p = p.parent()
                if hasattr(p, 'bucket'):
                    p = p.bucket
                    # Get the initial path
                    initial_path = p.dir

        # Get the filepath 
        initial_name = f"{self.title.lower()}.csv"
        path = wsbmr.utils.show_save_file_dialog(self, initial_path, initial_name)
    
    def on_reset(self, *args, **kwargs):
        # Reset the table columns and unhide all of them 
        for i in range(self.table.columnCount()):
            self.table.setColumnHidden(i, False)
        # Reset column var
        if self.df is not None:
            self.columns = list(self.df.columns)
        # Re-draw data in table
        self.update_table()
    
    def on_action(self, action, *args, **kwargs):
        if self.parent:
            if hasattr(self.parent, action):
                wsbmr.utils.log._log(f"Calling action {action} with args {args} and kwargs {kwargs}")
                getattr(self.parent, action)(*args, **kwargs)


    """ Update the table with the DataFrame """
    def update_table(self, column_width = {'protection': 100, 'ber': 100, 'run_reps': 100, 'coverage': 100, 'experiment': 100}):
        self.table.set_data(self.df)
       
    def set_data(self, df, config = {}, column_width = {}):
        self.df = df
        self.config = config
        self.update_table(column_width = column_width)

    def get_data(self):
        return self.df

    def clear_data(self):
        self.df = None
        self.config = {}
        self.update_table()
    
    """ Context menu """
    def open_context_menu(self, position: QPoint):
        # Get the column index where the right-click happened
        header = self.table.horizontalHeader()
        column_index = header.logicalIndexAt(position)

        if column_index < 0:  # Out of bounds check
            return

        # Create a context menu
        menu = QMenu(self)
        # Get name of this column in df 
        column_name = self.df.columns[column_index]

        # Create an action to hide the column
        hide_action = QAction(f"Hide column {column_name}", self)
        # Create an action to plot the distribution of the column
        plot_action = QAction(f"Plot distribution of {column_name}", self)

        # Connect the action to hide the column
        hide_action.triggered.connect(lambda: self.hide_column(column_name, column_index))
        # Connect the action to plot the distribution of the column
        plot_action.triggered.connect(lambda: self.plot_dist_column(column_name, column_index))

        # Add the action to the menu
        menu.addAction(hide_action)
        menu.addAction(plot_action)

        # Show the menu at the position relative to the header
        menu.exec_(header.mapToGlobal(position))

    def hide_column(self, column_name, column_index):
        """Hide the specified column."""
        # Remove from self.columns 
        if column_name in self.columns:
            self.columns.remove(column_name)
        self.table.setColumnHidden(column_index, True)

    """ Plot the distribution of the specified column. """
    def plot_dist_column(self, column_name, column_index):
        """Plot the distribution of the specified column."""
        # Get the column values
        values = self.df[column_name]
        
        wsbmr.utils.log._log(f"Plotting distribution of column {column_name}")
        # Clean up previous closed windows 
        self.cleanup_closed_windows()

        # Create a multi plot window 
        plot_window = PlotWindow('dist', title = f'Distribution of {column_name}')

        # Create a histogram
        plot_window.ax.hist(values, bins=20, color='blue', edgecolor='black')

        # Add labels and title
        plot_window.ax.set_xlabel(column_name)
        plot_window.ax.set_ylabel('Frequency')
        plot_window.ax.set_title(f'Distribution of {column_name}')

        # Add reference so pyqt5 doesn't delete it...
        self.plot_windows.append(plot_window)

        # Print number of active windows 
        wsbmr.utils.log._log(f"Number of active plot windows: {len(self.plot_windows)}")

        # Show the plot window
        self.plot_windows[-1].show()
    
    # Function to clean up closed windows
    def remove_closed_window(self, window):
        """Remove a closed window from the list."""
        if window in self.plot_windows:
            self.plot_windows.remove(window)
    
    def cleanup_closed_windows(self):
        """Check all windows and remove references to those that are closed or deleted."""
        #self.plot_windows = [win for win in self.plot_windows if not sip.isdeleted(win) and win.isVisible()]
        idxs2keep = []
        for i, win in enumerate(self.plot_windows):
            try:
                if win.isVisible():
                    idxs2keep.append(i)
            except RuntimeError:
                pass
        self.plot_windows = [self.plot_windows[i] for i in idxs2keep]


class JobsGroupWidget(QGroupBox):
    def __init__(self, title, results, coverage, bucket = None):
        super().__init__(title)
        self.results = results
        self.coverage = coverage
        self.bucket = bucket
        # We need to get the jobs that are missing from the results and coverage, and generate a table (df) with it
        self.potential_jobs = self.get_potential_jobs()

        # Create layout 
        layout = QVBoxLayout()
        # Set this layout to ensure everything created here is set in this group box
        self.setLayout(layout)

        # Create a table widget to display the DataFrame
        self.table = CustomTableWidget(self.potential_jobs, percent_columns = [], color_column = None)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)  # Disable editing
        #self.table.setAlternatingRowColors(True)  # Alternate row colors

        # Set maximum height for table 
        self.table.setMinimumHeight(50)
        self.table.setMaximumHeight(300)

        # Options
        self.table.setSortingEnabled(True)  # Enable sorting by columns
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)  # Select entire rows
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)  # Stretch columns to fill the available space

        # Ensure main window (parent) doesn't change size when table is resized
        self.table.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
        self.table.setVerticalScrollMode(QTableWidget.ScrollPerPixel)

        # Enable horizontal scrollbar policy
        self.table.setSizePolicy(self.table.sizePolicy().horizontalPolicy(), self.table.sizePolicy().verticalPolicy())
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Enable custom context menu
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        #self.table.customContextMenuRequested.connect(self.open_context_menu)
        
        """ . """
        # Add a horizontal tool panel to do stuff to the table (like save current table to CSV, reset the table columns hidden, etc.)
        self.toolbox = self._build_toolbox()

        """ Add a QLabel for displaying a <No jobs found> if df is empty """
        self.no_data_label = QLabel("No jobs missing", self.table)
        self.no_data_label.setStyleSheet("font-size: 18px; color: gray;")
        self.no_data_label.setAlignment(Qt.AlignCenter)
        self.no_data_label.hide()  # Hide by default

        # Add the toolbox to the layout
        layout.addWidget(self.table)
        layout.addWidget(self.toolbox)
        layout.addWidget(self.no_data_label)

        # Set layout
        self.layout = layout

        # Update the table with the DataFrame
        self.update_table()

    def _build_toolbox(self):
        
        widget = QWidget(self)

        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)  # Margins around the toolbar
        layout.setSpacing(10)  # Space between buttons

        # Button: Run jobs
        self.run_button = QPushButton("Run jobs")
        self.run_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.run_button.setToolTip("Run jobs")
        #self.save_button.setCursor(Qt.PointingHandCursor)
        self.run_button.clicked.connect(lambda *args, **kwargs: self.run_jobs(*args, **kwargs))

        # Update table 
        self.update_button = QPushButton("Update table")
        self.update_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.update_button.setToolTip("Update table")
        self.update_button.clicked.connect(lambda *args, **kwargs: self.update_table(*args, **kwargs))

        # Add buttons to the layout
        layout.addWidget(self.run_button)
        layout.addWidget(self.update_button)

        # Add a spacer to push the buttons to the left
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addSpacerItem(spacer)

        # Set the layout of the widget
        widget.setLayout(layout)
        return widget

    """ Get potential jobs (missing) """
    def get_potential_jobs(self):
        
        config = {}
        if self.bucket is not None:
            # if bucket is root, get all jobs without filtering
            if self.bucket.type == 'root':
                config = {}
            else:
                config = self.bucket.structure_config
        
        # Get jobs 
        jobs_df = wsbmr.utils.get_nodus_jobs_for_config(config)
        
        # Initialize the ranges for the jobs
        if self.bucket is None:
            return jobs_df
        
        # Get missing jobs
        missing_jobs = self.bucket.get_missing_jobs()
        
        if 'benchmark' not in missing_jobs:
            if 'benchmark' in self.bucket.structure_config:
                missing_jobs['benchmark'] = self.bucket.structure_config['benchmark']
        if 'quantization' not in missing_jobs:
            if 'quantization' in self.bucket.structure_config:
                missing_jobs['quantization'] = self.bucket.structure_config['quantization']
        if 'pruning' not in missing_jobs:
            if 'pruning' in self.bucket.structure_config:
                missing_jobs['pruning'] = self.bucket.structure_config['pruning_factor']

        # Init job_string pattern, which will be the same for all jobs (like benchmarks_dir, etc)
        job_str_0 = f'python wsbmr --benchmarks_dir \"{self.bucket.hyperspace_global_config["benchmarks_dir"]}\"'
        job_str_0 += f' --datasets_dir \"{self.bucket.hyperspace_global_config["datasets_dir"]}\"'

        # Everytime that we have a new model (combo of benchmark, quantization & pruning),
        # we need to add an extra job just for training this model. This new job will be 
        # the first and all the rest of the jobs for this group are dependent on this one. 
        g = missing_jobs.groupby(['benchmark', 'quantization', 'pruning'])

        # Basic default config
        default_args = '--plot'

        # Init jobs 
        jobs = []

        # Loop thru group combos 
        for name, group in g:
            # Initialize the common part 
            job_str = f'{job_str_0}'
            # Parse name 
            bmk, quant, pruning = name
            # Add benchmark
            job_str += f' --benchmark {bmk}'
            # Add quantization
            bits_config = quant.split('_')
            bits_config = f'num_bits={bits_config[0].replace("bits", "")} integer={bits_config[1].replace("int", "")}'
            job_str += f' --bits_config {bits_config}'
            # Add pruning
            job_str += f' --prune {pruning}'
            # Add default args
            # Protection range and ber 
            job_str += f' --protection_range {" ".join(list(map(str,self.bucket.hyperspace_global_config["protection"])))}'
            job_str += f' --ber_range {" ".join(list(map(str,self.bucket.hyperspace_global_config["ber"])))}'
            # Init subjobs
            subjobs = []
            # Loop thru rows in group 
            for _, row in group.iterrows():
                # Init job string 
                subjob = f'{job_str}'
                # Add config per method 
                if row['method'] in wsbmr.config.config_per_method:
                    for key, value in wsbmr.config.config_per_method[row['method']].items():
                        subjob += f' --{key} {value}'

                # Add default args 
                subjob += f' {default_args}'

                # Entry dictionary (so we can convert it to pandas df later)
                entry = {'status': 'pending', 'benchmark': bmk, 'quantization': quant, 'pruning': pruning, 'method': row['method'], 'command': subjob, 'parent': job_str}

                jobs.append(entry)

        # Convert to df
        jobs_df = pd.DataFrame(jobs)
        
        return jobs_df

    def update_table(self, column_width = {}):

        # Get potential jobs 
        self.potential_jobs = self.get_potential_jobs()

        # If potential jobs is empty 
        if self.potential_jobs.empty:
            self.table.clear()
            self.no_data_label.show()
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.table.hide()
            self.toolbox.hide()
        else:
            self.no_data_label.hide()
            self.toolbox.show()
            self.table.show()
            self.table.set_data(self.potential_jobs)
            self.table.setColumnCount(len(self.potential_jobs.columns))
            self.table.setHorizontalHeaderLabels(self.potential_jobs.columns)

            for i, row in self.potential_jobs.iterrows():
                for j, (key, value) in enumerate(row.items()):
                    item = CustomTableWidgetItem(str(value), value)
                    item.setTextAlignment(Qt.AlignCenter)
                    self.table.setItem(i, j, item)

                    # If we right-click on this item, we want to show a context menu allowing the user to copy the command string 
                    # to the clipboard
                    item.setFlags(item.flags() | Qt.ItemIsSelectable)

            # Callback for copying the command string to the clipboard
            self.table.itemDoubleClicked.connect(self.copy_command_to_clipboard)
        
    
    def copy_command_to_clipboard(self, item):
        # Avoid this function from being called twice (due to the double click event)
        self.table.itemDoubleClicked.disconnect(self.copy_command_to_clipboard)

        """Copy the command string to the clipboard."""
        # Get the command string from the item
        command = item.text()
        # Copy the command string to the clipboard
        clipboard = QApplication.clipboard()
        clipboard.setText(command)
        print(f"Command copied to clipboard: {command}")
        # Get the item X position (to know the column)
        x = item.column()
        # get the column name
        column_name = self.potential_jobs.columns[x]
        # Show a message box to indicate that the command has been copied
        QMessageBox.information(self, f"{column_name.capitalize()} Copied", f"Field {column_name.capitalize()} copied to clipboard:\n\n{command}")

        # Reconnect the signal
        self.table.itemDoubleClicked.connect(self.copy_command_to_clipboard)
        

    """ Set data """
    def set_data(self, results, coverage, bucket, config = {}, column_width = {}):
        self.results = results
        self.coverage = coverage
        self.bucket = bucket
        self.config = config
        # Get the potential jobs
        self.update_table(column_width = column_width)

    def get_data(self):
        return self.potential_jobs

    def clear_data(self):
        self.results = None
        self.coverage = None
        self.potential_jobs = None
        self.config = {}
        self.update_table()

    def run_jobs(self, *args, **kwargs):
        """ Run the jobs """
        # Get the selected rows
        selected_rows = self.table.selectionModel().selectedRows()
        # Get the indices of the selected rows
        selected_indices = [index.row() for index in selected_rows]
        # Get the command strings for the selected rows
        commands = []
        for i in selected_indices:
            cc = {}
            for j, k in enumerate(self.potential_jobs.columns):
                cc[k] = self.table.item(i,j).text()
            commands.append(cc)
        # Convert to df 
        commands = pd.DataFrame(commands)

        # Let's create the jobs using nodus job manager
        r = wsbmr.utils.run_jobs_with_nodus(commands)


""" Data Panel with two tabs, one for the results and another for the coverage """
class PandasDataFrameWidget(QTabWidget):
    def __init__(self, parent=None, results = None, stats = None, coverage = None, bucket = None):
        super().__init__()

        # Layout
        self.info_tab = QTextBrowser()
        self.info_tab.setOpenExternalLinks(False)  # Ensure we handle clicks internally
        # Connect signal to slot
        self.info_tab.anchorClicked.connect(self.open_folder_html)
        self.info_tab.setHtml("")

        # Init bucket to None
        self.bucket = None
        
        """ Other tabs """
        self.results_tab = PandasDataFrameGroupWidget("Results", results)
        self.stats_tab = PandasDataFrameGroupWidget("Stats", stats)
        self.coverage_tab = PandasDataFrameGroupWidget("Coverage", coverage)
        self.jobs_tab = JobsGroupWidget("Jobs", results, coverage, bucket)
        
        self.addTab(self.info_tab, "Info")
        self.addTab(self.results_tab, "Results")
        self.addTab(self.stats_tab, "Stats")
        self.addTab(self.coverage_tab, "Coverage")
        self.addTab(self.jobs_tab, "Jobs")

    def set_data(self, df, stats, coverage, html, bucket, config = {}, column_width = {}):
        # Set bucket 
        self.bucket = bucket
        self.info_tab.setHtml(html)
        self.results_tab.set_data(df, config = config, column_width = column_width)
        self.stats_tab.set_data(stats, config = config, column_width = column_width)
        self.coverage_tab.set_data(coverage, config = config, column_width = column_width)
        self.jobs_tab.set_data(df, coverage, bucket, config = config, column_width = column_width)

    def get_data(self):
        return self.results_tab.get_data(), self.coverage_tab.get_data()

    def clear_data(self):
        self.bucket = None
        self.results_tab.clear_data()
        self.stats_tab.clear_data()
        self.coverage_tab.clear_data()
        self.jobs_tab.clear_data()

    def update_table(self, column_width = {}):
        self.results_tab.update_table(column_width = column_width)
        self.stats_tab.update_table(column_width = column_width)
        self.coverage_tab.update_table(column_width = column_width)
        self.jobs_tab.update_table(column_width = column_width)
    
    def open_folder_html(self, url: QUrl):
        folder_path = url.toLocalFile()
        wsbmr.utils.open_directory(folder_path)
        # Prevent the QTextBrowser from navigating away
        self.info_tab.setSource(QUrl())  # Clear navigation
    
        
""" Log Viewer """
class LogViewer(QWidget):
    def __init__(self, log_file: str):
        super().__init__()
        self.log_file = log_file
        self.line_number = 0
        
        self.init_ui()

        # Set up a QTimer to periodically check for new log data
        self.timer = QTimer(self)
        self.timer.setInterval(100)  # Check every 100 ms
        self.timer.timeout.connect(self.read_log_file)
        self.timer.start()

        # Track the file's current position (initially at the end of the file)
        self.file_position = 0
        
        # # Create and start the LogReaderThread
        # self.log_reader_thread = wsbmr.gui.asynchronous.LogReaderThread(self.log_file)
        # self.log_reader_thread.new_log_data.connect(self.append_log_data)
        # self.log_reader_thread.start()

    def read_log_file(self):
        """Reads new content from the log file and updates the text area."""
        try:
            with open(self.log_file, 'r') as f:
                f.seek(self.file_position)  # Move the pointer to the last read position
                new_data = f.read()  # Read new data
                if new_data:
                    self.text_edit.insertPlainText(new_data)  # Update the text area
                    self.file_position = f.tell()  # Update the position for the next read
        except Exception as e:
            print(f"Error reading file: {e}")

    def init_ui(self):
        self.setWindowTitle("Log Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        # Add a label to show the log file path
        self.log_file_label = QLabel(f"Log file: {self.log_file}")

        # Add text edit 
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)  # Make the text area non-editable

        # Set terminal-like font (monospace)
        self.text_edit.setFont(QFont("Courier", 10))  # Monospace font

        # Set background color to black and text color to white
        self.text_edit.setStyleSheet("background-color: black; color: white;")

        # Remove margins and padding (mimicking a terminal)
        self.text_edit.setContentsMargins(0, 0, 0, 0)  # Remove margins around the text
        self.text_edit.setLineWidth(0)  # Remove line width padding

        # Add widgets 
        self.layout.addWidget(self.log_file_label)
        self.layout.addWidget(self.text_edit)

        #Set layout
        self.setLayout(self.layout)


# import re
# from PyQt5.QtGui import QColor, QTextCharFormat

# """ Terminal """
# class InteractiveTerminalWidget(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Interactive Terminal Widget")

#         # Layout and QTextEdit to display the terminal output
#         layout = QVBoxLayout()
#         self.text_area = QTextEdit()
#         self.text_area.setReadOnly(False)  # Allow text input
#         layout.addWidget(self.text_area)

#         # Button to start the terminal process
#         self.start_button = QPushButton("Start Interactive Terminal")
#         layout.addWidget(self.start_button)

#         self.setLayout(layout)

#         # Initialize QProcess to run the terminal application (interactive session)
#         self.process = QProcess(self)
#         self.process.readyReadStandardOutput.connect(self.handle_output)
#         self.process.readyReadStandardError.connect(self.handle_error)
#         self.start_process()

#         # Connect the button to start the process
#         self.start_button.clicked.connect(self.start_process)

    



#     def handle_output(self):
#         """Handle new output from the running process and display it in the QTextEdit."""
#         output = self.process.readAllStandardOutput()
#         self.text_area.clear()
#         #self.text_area.append(self.handle_ansi_escape_codes(output))  # Append new output to the text area
#         self.text_area.append(output)  # Append new output to the text area

#     def handle_error(self):
#         """Handle any errors from the running process."""
#         error = self.process.readAllStandardError()
#         if error:  # Only process if there is any error output
#             self.text_area.clear()
#             formatted_error = self.handle_ansi_escape_codes(error)  # Apply the ANSI escape code handler
#             self.text_area.append(formatted_error)

#     def keyPressEvent(self, event):
#         """Capture user key presses and send them to the process."""
#         if event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
#             text = self.text_area.toPlainText()
#             last_line = text.split("\n")[-1]
#             self.process.write(QByteArray(last_line.encode("utf-8")) + b"\n")
#         else:
#             super().keyPressEvent(event)

#     def start_process(self):
#         """Start the interactive process when the button is clicked."""
#         self.process.start("python", ["-m", "nodus", "--db", "/Users/mbvalentin/scripts/wsbmr/dev/wsbmr_db"])
