""" Basic modules """
import os
from copy import deepcopy

""" PyQt5 imports """
from PyQt5.QtWidgets import QGroupBox, QSizePolicy, QVBoxLayout, QHBoxLayout, QTabWidget, QPushButton, QSpacerItem
from PyQt5.QtWidgets import QListWidget, QSplitter, QMessageBox, QStyle, QMenu, QWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

# netsurf imports
import netsurf

# Custom imports
from . import widgets

""" Generic panel class """
class netsurfPanel(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.config = {}
    
    """ Set dirs callback """
    def set_value(self, key, value, verbose = True):
        self.config[key] = value
        if verbose:
            netsurf.utils.log._log(f"Setting {key} to {value}")


""" Define inputs panel """
class InputsPanel(netsurfPanel):
    def __init__(self, **kwargs):
        super().__init__("Inputs")
        
        # Initialize configuration to store all relevant values 
        self.config = {"benchmarks_dir": "",
                       "datasets_dir": "",
                       "bucket_name": ""}

        # Initialize UI
        self.init_ui(**kwargs)

    """ Initialize UI """
    def init_ui(self, benchmarks_dir = netsurf.config.DEFAULT_BENCHMARKS_DIR, datasets_dir = netsurf.config.DEFAULT_DATASETS_DIR, **kwargs):
        self.setStyleSheet("QGroupBox { border: 1px solid gray; margin-top: 10px; }")
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        layout = QVBoxLayout()
        # Set constraints 
        # Left, top, right, bottom
        layout.setContentsMargins(10, 10, 10, 10)

        # Create widgets
        benchmarks_widget = widgets.DirectorySelectorWidget("Benchmarks dir:", lambda x: self.set_dir("benchmarks_dir", x), default = benchmarks_dir)
        datasets_widget = widgets.DirectorySelectorWidget("Datasets dir:", lambda x: self.set_dir("datasets_dir", x), default = datasets_dir)
        bucket_name_widget = widgets.TextEditWidget("Bucket name:", lambda x: self.set_value("bucket_name", x), default = "root")

        # Add a horizontal layout for buttons 
        tbox = QHBoxLayout()
        # Left, top, right, bottom
        tbox.setContentsMargins(0, 0, 0, 0)

        # Add "Create bucket" button
        create_bucket_button = QPushButton("Create bucket", self)
        create_bucket_button.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        # Store button in structure so we can access it from outside to set the callback
        self.create_bucket_button = create_bucket_button

        # Load bucket button
        self.load_bucket_button = QPushButton("Load bucket", self)
        self.load_bucket_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogEnd))
        self.load_bucket_button.setToolTip("Load bucket from file")

        # Open session log button
        self.open_session_log_button = QPushButton("Open session log", self)
        self.open_session_log_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        self.open_session_log_button.setToolTip("Open session log")

        # Open nodus session terminal
        self.open_nodus_terminal_button = QPushButton("Open nodus terminal", self)
        self.open_nodus_terminal_button.setIcon(self.style().standardIcon(QStyle.SP_CommandLink))
        self.open_nodus_terminal_button.setIcon(QIcon(os.path.join(netsurf.__dir__,"res/terminal-icon.png")))
        self.open_nodus_terminal_button.setToolTip("Open nodus terminal")

        # Add button to layout
        tbox.addWidget(self.create_bucket_button)
        tbox.addWidget(self.load_bucket_button)
        tbox.addWidget(self.open_session_log_button)
        tbox.addWidget(self.open_nodus_terminal_button)

        # Add a horizontal spacer
        tbox.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # Add to layout
        layout.addWidget(benchmarks_widget)
        layout.addWidget(datasets_widget)
        layout.addWidget(bucket_name_widget)
        layout.addLayout(tbox)

        # Set size
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

    """ Set directory if valid """
    def set_dir(self, key, value, verbose = True):
        if netsurf.utils.is_valid_directory(value):
            netsurf.utils.log._log(f"Setting valid directory {value} for {key}")
            self.set_value(key, value, verbose = False)
        else:
            # If the directory is invalid, ask the user with a popup if they want to create it 
            reply = QMessageBox.question(self, 'Confirmation', f"The directory {value} does not exist. Do you want to create it?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                netsurf.utils.log._log(f"Creating directory {value} for {key}")
                os.makedirs(value, exist_ok = True)
                self.set_value(key, value, verbose = False)
            else:
                netsurf.utils.log._log(f"Invalid directory {value} for {key}")

    
    
""" Define simulation config panel """
class SimulationConfigPanel(netsurfPanel):
    def __init__(self):
        super().__init__("Simulation Config")
        self.config = {'num_reps': -1, 'benchmarks': [], 'methods': [], 'quantizations': [], 'protections': [], 'pruning': []}
        self.init_ui()
    
    # Initialize UI
    def init_ui(self):
        self.setStyleSheet("QGroupBox { border: 1px solid gray; margin-top: 10px; }")
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        layout = QVBoxLayout()

        # Create widgets
        num_reps_spinner = widgets.IntegerSpinnerWidget("Num reps:", lambda x: self.set_value("num_reps", x), default_value = netsurf.config.DEFAULT_NUM_REPS)

        # Create tab widget for different configs
        tab_widget = QTabWidget()
        tab_benchmarks = widgets.CustomListWidgetTab(netsurf.config.AVAILABLE_BENCHMARKS, netsurf.config.DEFAULT_BENCHMARKS, 
                                lambda x: self.set_selection('benchmarks', x))
        tab_methods = widgets.CustomListWidgetTab(netsurf.config.AVAILABLE_METHODS, netsurf.config.DEFAULT_METHODS, 
                                lambda x: self.set_selection('methods',x))
        tab_quantizations = widgets.CustomQuantizationListWidgetTab(netsurf.config.DEFAULT_QUANTIZATIONS, 
                                lambda x: self.set_selection('quantizations', x))
        tab_protections = widgets.CustomFloatListWidgetTab([str(t) for t in netsurf.config.DEFAULT_PROTECTION], [str(t) for t in netsurf.config.DEFAULT_PROTECTION], 
                                lambda x: self.set_selection('protections', x), min = 0.0, max = 1.0, step = 0.2) 
        tab_pruning = widgets.CustomFloatListWidgetTab([str(t) for t in netsurf.config.DEFAULT_PRUNINGS], [str(t) for t in netsurf.config.DEFAULT_PRUNINGS],
                                lambda x: self.set_selection('pruning', x), min = 0.0, max = 1.0, step = 0.125)
        tab_bers = widgets.CustomFloatListWidgetTab([str(t) for t in netsurf.config.DEFAULT_BER], [str(t) for t in netsurf.config.DEFAULT_BER],
                                lambda x: self.set_selection('bers', x), min = 1e-3, max = 1e-1, step = 10**(2/9), log = True)
        # Add the custom tab widgets to the QTabWidget
        tab_widget.addTab(tab_benchmarks, "Benchmarks")
        tab_widget.addTab(tab_methods, "Methods")
        tab_widget.addTab(tab_quantizations, "Quantizations")
        tab_widget.addTab(tab_protections, "TMR - Protection")
        tab_widget.addTab(tab_pruning, "Pruning - Sparsity")
        tab_widget.addTab(tab_bers, "BER - Error Rate Injection")

        # Update config with default values
        self.set_value('num_reps', netsurf.config.DEFAULT_NUM_REPS)
        self.set_selection('benchmarks', netsurf.config.DEFAULT_BENCHMARKS)
        self.set_selection('methods', netsurf.config.DEFAULT_METHODS)
        self.set_selection('quantizations', netsurf.config.DEFAULT_QUANTIZATIONS)
        self.set_selection('protections', netsurf.config.DEFAULT_PROTECTION)
        self.set_selection('pruning', netsurf.config.DEFAULT_PRUNINGS)
        self.set_selection('bers', netsurf.config.DEFAULT_BER)

        # Add to layout
        layout.addWidget(num_reps_spinner)
        layout.addWidget(tab_widget)

        # Set size
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)

        # Set default height of tab_widget
        tab_widget.setMinimumHeight(150)
        #tab_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        #tab_widget.resize(-1, 150)


        #self.setFixedHeight(250)
    
    # Set selection of values 
    def set_selection(self, key, value):
        self.config[key] = value
        netsurf.utils.log._log(f"Setting {key} to {value}")


""" Buckets panel """
class BucketsPanel(netsurfPanel):
    def __init__(self, title, buckets):
        super().__init__(title)
        self.buckets = buckets
        self.plot_windows = []

        self.init_ui()
    
    # Initialize UI
    def init_ui(self):
        self.setStyleSheet("QGroupBox { border: 1px solid gray; margin-top: 10px; }")
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        # The buckets panel has three widgets organized horizontally:
        #  - A list of buckets (left), only one selectable
        #  - A tree widget showing the contents of the selected bucket (center)
        #  - A Pandas Df widget showing the contents of the selected level of the tree widget (right)
        layout = QHBoxLayout()

        # Create a QSplitter for horizontal resizing
        splitter = QSplitter(Qt.Horizontal)

        # Create widgets
        # First a list of buckets 
        list_bucket = QListWidget()
        self.list_bucket = list_bucket
        
        # Create tree widget
        tree_widget = widgets.BucketTreeWidget(self)
        tree_widget.setColumnCount(2)
        tree_widget.setHeaderLabels(["Name", "Type"])
        self.tree_widget = tree_widget

        # finally, add a PandasDataFrameGroupWidget 
        dgp = widgets.PandasDataFrameWidget("Data", None, None)
        self.table_panel = dgp

        # Connect list widget to tree widget
        list_bucket.itemClicked.connect(lambda x: tree_widget.update_tree(self.buckets[x.text()]))
        # Whenever we click on a tree item, update the dataframe
        tree_widget.itemClicked.connect(self.update_table_on_tree_item_selection)
        #list_bucket.itemClicked.connect(lambda x: dgp.update_df(self.buckets[x.text()].df))

        # Add to layout
        splitter.addWidget(list_bucket)
        splitter.addWidget(tree_widget)
        splitter.addWidget(dgp)

        # Optional: Set initial sizes
        splitter.setSizes([20, 300, 400])  # Initial Size adjusting Default wieght Widget1!
        layout.addWidget(splitter)

        # Set size
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        #self.setFixedHeight(350)
        self.setMinimumHeight(350)

        # Set default width of tree_widget
        self.tree_widget.setColumnWidth(0, 300)
        self.tree_widget.setColumnWidth(1, 100)
    
    """ When an item in the tree is selected, update the table """
    def update_table_on_tree_item_selection(self, item):
        # See which item is selected in the tree widget
        if hasattr(item, 'bucket'):
            netsurf.utils.log._log(f"Selected item/bucket is: {item.bucket}")
            # The selected item from the tree item is "item"
            selected_bucket = item.bucket
            # get html
            html = selected_bucket.html
            # Item is a custom BucketTreeItemWidget which is a subclass of QTreeWidgetItem, but it contains the sub-bucket
            # Get results that are not NaN
            df = deepcopy(selected_bucket.results)
            # Get mode metric if any 
            _metric = 'loss'
            if _metric in df.columns:
                if (~df['loss'].isna()).any():
                    _metric = df['loss'].mode()[0]
            stats = df.groupby(['protection','ber']).agg({_metric: ['min', 'max', 'mean', 'std', 'skew']})
            stats = deepcopy(stats)
            # Reset index 
            stats = stats.reset_index()
            # combine multiindex columns
            stats.columns = ['_'.join([cc for cc in col if cc.strip() != ""]).strip() for col in stats.columns.values]
            df = df.dropna(axis=0, how='any')
            # Get coverage
            cov = selected_bucket.coverage
            self.table_panel.set_data(df, stats, cov, html, selected_bucket)

    """ Check if item exists in the QListWidget """
    def item_in_bucket_list(self, search_text):
        # Check if the item exists in the QListWidget
        for i in range(self.list_bucket.count()):
            if search_text == self.list_bucket.item(i).text():
                return True
        return False

    # Update function 
    def update_buckets(self, buckets):
        self.buckets = buckets

        # Update views
        for bucket in self.buckets.values():
            if not self.item_in_bucket_list(bucket.name):
                self.list_bucket.addItem(bucket.name)
                # Connect this item so that if it's clicked with the right button, we can delete it via a
                # context menu
                self.list_bucket.setContextMenuPolicy(Qt.CustomContextMenu)
                self.list_bucket.customContextMenuRequested.connect(lambda *args, **kwargs: self.show_context_menu(bucket.name, *args, **kwargs))
        
        # Update tree widget
        # if there's any selected item, update the tree widget
        if self.list_bucket.currentItem():
            # Get the selected bucket
            selected_bucket = self.buckets[self.list_bucket.currentItem().text()]
            self.tree_widget.update_tree(selected_bucket)
    
    # Show context menu
    def show_context_menu(self, bucket_name, pos):
        # Create a context menu for when this item is clicked with a right button
        menu = QMenu(self)
        delete_action = menu.addAction("Remove bucket")
        # connect delete action to a function
        delete_action.triggered.connect(lambda: self.remove_bucket_from_list(bucket_name))
        # Show the menu
        menu.exec_(self.list_bucket.mapToGlobal(pos))
        
    
    # Remove bucket from list
    def remove_bucket_from_list(self, bucket_name):
        # Get the selected item
        item = self.list_bucket.currentItem()
        # Remove the item from the list
        self.list_bucket.takeItem(self.list_bucket.row(item))
        # Remove the bucket from the buckets dict
        if item.text() in self.buckets:
            del self.buckets[item.text()]
        elif bucket_name in self.buckets:
            del self.buckets[bucket_name]
        # Update the tree widget
        self.tree_widget.update_tree(None)

    # Clean up winows 
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

""" The log panel is a pyqt5 panel that has a text area display where 
    this session's log is displayed in real time (while it's being generated) 
"""
class LogPanel(QWidget):
    def __init__(self, parent = None, log_file = None):
        super().__init__(parent)

        # Initialize the log file 
        if log_file is None:
            log_file = netsurf.nodus.__nodus_log_file__
            if not os.path.exists(log_file):
                log_file = None
        self.log_file = log_file

        # Initialize UI
        self.init_ui()
    
    # Initialize UI
    def init_ui(self):
        layout = QVBoxLayout()
        self.text_area = widgets.LogViewer(self.log_file)
        
        layout.addWidget(self.text_area)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.setMinimumHeight(250)
        self.setMinimumWidth(500)

