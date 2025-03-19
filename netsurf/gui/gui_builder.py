""" Basic modules """
import sys
import os
import asyncio

""" PyQt5 imports """
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QSpacerItem, QSizePolicy, QApplication, QProgressBar, QTabWidget
from PyQt5.QtWidgets import QLabel, QSplitter
from PyQt5.QtCore import Qt

""" Import gui """
import wsbmr
from wsbmr import gui

""" Main window class """
class MainWindow(QMainWindow):
    _last_progress = 0.0
    def __init__(self, **kwargs):
        super().__init__()
        self.setWindowTitle("WSBMR GUI")
        self.setGeometry(100, 100, wsbmr.config.DEFAULT_MAIN_WINDOW_WIDTH, wsbmr.config.DEFAULT_MAIN_WINDOW_HEIGHT)

        # Init config dict to empty
        self.config = {'texp': None, 'coverage': None}
        self.buckets = {}

        # Initialize UI
        self.init_ui(**kwargs)

    """ Initialize UI """
    def init_ui(self, **kwargs):

        """ Layout will consist of a tabbed interface. 
            The first tab will contain a vertical layout with the inputs panel, 
                the simulation config panel, and the actions panel.
            The second tab will contain a text viewer for the log for the current session.
            The third tab will contain a "terminal-like" widget for the user to interact with nodus 
        """

        # Create the tab widget 
        tab_widget = QTabWidget()
        

        """ Tab 1. Main tab """
        # Create layout """
        container = QWidget()
        container_layout = QVBoxLayout()

        # Create a splitter for vertical resizing
        splitter = QSplitter(Qt.Vertical)
        # Add to container_layout
        container_layout.addWidget(splitter)
        
        """ Inputs panel """
        # Instantiate inputs panel
        ip = gui.panels.InputsPanel(**kwargs)
        # Connect callback to button 
        ip.create_bucket_button.clicked.connect(lambda *args, **kwargs: self.process_experiments(text = "Processing experiments"))
        # Connect load and save buttons
        ip.load_bucket_button.clicked.connect(self.load_bucket)
        # Connect open session log button
        ip.open_session_log_button.clicked.connect(wsbmr.utils.open_session_log)
        # connect "open nodus terminal" button
        ip.open_nodus_terminal_button.clicked.connect(self.open_nodus_terminal)
        
        # Add to container
        splitter.addWidget(ip)

        """ Simulation config panel """
        # Instantiate simulation config panel
        scp = gui.panels.SimulationConfigPanel()
        # Add to container
        splitter.addWidget(scp)

        """ Add group with a button to display the config """
        # Instantiate group
        #accp = gui.panels.ActionsPanel("Actions", lambda *args, **kwargs: self.process_experiments())
        # Add to container
        #container_layout.addWidget(accp)

        """ Add a group for the buckets """
        # Instantiate group
        buckets_panel = gui.panels.BucketsPanel("Buckets", self.buckets)
        self.buckets_panel = buckets_panel
        #buckets_panel.setFixedWidth(1100)
        # Add to container
        splitter.addWidget(buckets_panel)

        # Keep children in a list
        self.children = [ip, scp, buckets_panel]

        # Add spacer to push the QGroupBox up
        #container_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Let's add a progress bar at the bottom of the window
        # Create a QProgressBar and add it to the QStatusBar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setAlignment(Qt.AlignCenter)

        # Add the progress bar and label to the status bar
        self.status_bar = self.statusBar()
        self.status_bar.addWidget(self.progress_bar, 1)  # Stretch factor = 1
        # height 
        self.status_bar.setFixedHeight(20)

        # Reset progress bar
        self.reset_progressbar() # initially hidden

        # Set layout
        container.setLayout(container_layout)

        # Add tab to tab widget
        tab_widget.addTab(container, "Main")

        """ Tab 2. Log tab """
        # Create layout """
        log_container = QWidget()
        log_container_layout = QVBoxLayout()

        # Instantiate log panel
        log_panel = gui.panels.LogPanel()
        # Add to container
        log_container_layout.addWidget(log_panel)

        # Set layout
        log_container.setLayout(log_container_layout)

        # Add tab to tab widget
        tab_widget.addTab(log_container, "Log")

        
        """ Finally, add the tab widget to the central widget """
        self.setCentralWidget(tab_widget)

    
    # Add a callback to update the config from children objects 
    def pull_config(self, child):
        if hasattr(child, "config"):
            # We have to LINK the variables, not copy them
            # This way, if the child updates the config, the parent will also see the changes
            for key, value in child.config.items():
                self.config[key] = value
            
    # Display the config
    def print_config(self):
        wsbmr.utils.log._log(f"Current config: ")
        for key, value in self.config.items():
            print(f"\t{key}: {value}")
    
    """ Progress bar gui.callbacks """
    def reset_progressbar(self):
        self.progress_bar.setVisible(True)
        self.progress_bar.setFormat("%p%")

    def start_progressbar(self, text = ""):
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.progress_bar.setFormat(f"{text} - 0%")
    
    def update_progressbar(self, value, text = ""):
        #print(f'Updating progress bar to {value}')
        # Make sure value is int 
        value = int(value)
        if value < 100:
            self.progress_bar.setValue(value)
            self.progress_bar.setFormat(f"{text} - {value}%")
        else:
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat(f"{text} - 100%")
            self.reset_progressbar()
        
        if value == 0.0 or value == 100.0:
            self._last_progress = 0.0

        if (value - self._last_progress) > 0.5:
            self._last_progress = value
            QApplication.processEvents()  # Force UI refresh

    # Run experiments
    def process_experiments(self, text = "Processing experiments"):
        # Init progress bar
        self.start_progressbar(text = text)

        # Pull config from children
        for child in self.children:
            self.pull_config(child)

        # Pass progress bar to callback
        root_bucket = gui.callbacks.process_experiments(self.config, pbar = lambda value, text: self.update_progressbar(value, text))
        
        # Add to buckets 
        old_name = root_bucket.name
        if root_bucket.name in self.buckets:
            # Find next suffix
            new_names = wsbmr.utils.get_latest_suffix(old_name, self.buckets, next = True, divider = '_', return_index = False, next_if_empty = True)
            new_name = new_names[0]

            # Make sure new_name is not None
            if new_name is None or new_name == "" or new_name in self.buckets: 
                # brute force 
                next_dig = 0
                while True:
                    if next_dig == 0:
                        new_name = old_name 
                    else:
                        new_name = old_name + f"_{next_dig}"
                    if new_name not in self.buckets:
                        break
                    next_dig += 1
            
            # Update name in bucket now 
            root_bucket.name = str(new_name)
                
            # Log info
            wsbmr.utils.log._log(f"Bucket with name {old_name} already exists. Renaming to {new_name}")
        
        self.buckets[root_bucket.name] = root_bucket

        # Update tree panel 
        self.buckets_panel.update_buckets(self.buckets)
    
    def load_bucket(self):
        # Try to get initial dir from benchmarks dir from ipanel
        initial_dir = self.children[0].config.get('benchmarks_dir', os.getcwd())
        # Load a bucket from file
        filename, bkt = wsbmr.utils.show_load_bucket_dialog(self, initial_dir)
        if filename:
            # We need to make sure that this bucket's name is not already in the list, 
            # if it is, we need to rename it
            if bkt.name in self.buckets:
                # Find next suffix
                new_name = wsbmr.utils.get_latest_suffix(bkt.name, self.buckets, next = True, divider = '_', return_index = False, next_if_empty = True)
                if isinstance(new_name, list) or isinstance(new_name, tuple):
                    new_name = new_name[0]
                # Make sure new_name is not None
                if new_name is None or new_name == "" or new_name in self.buckets: 
                    # brute force 
                    next_dig = 0
                    while True:
                        if next_dig == 0:
                            new_name = bkt.name 
                        else:
                            new_name = bkt.name + f"_{next_dig}"
                        if new_name not in self.buckets:
                            break
                        next_dig += 1
                # Update name in bucket now 
                bkt.name = str(new_name)
                # Log info
                wsbmr.utils.log._log(f"Bucket with name {bkt.name} already exists. Renaming to {new_name}")

            # Add to buckets
            self.buckets[bkt.name] = bkt
            # Update tree panel
            self.buckets_panel.update_buckets(self.buckets)
            # Log info
            wsbmr.utils.log._log(f"Bucket loaded from file {filename}")
            #self.set_value('bucket_name', filename, verbose = False)
    
    def open_nodus_terminal(self):
        # Run command line "python -m nodus --db wsbmr_db" in a new terminal window
        db_path = wsbmr.config.WSBMR_NODUS_DB_NAME
        wsbmr.utils.open_terminal_with_command(f"python -m nodus --db {db_path}", generic = True)

    # def close(self):
    #     # Make sure to close nodus session before exiting 
    #     wsbmr.exit()
    #     return super().close()

""" Build GUI """
def build_gui(**kwargs):
    # Add fusion as style
    QApplication.setStyle("Fusion")
    app = QApplication(sys.argv)
    window = MainWindow(**kwargs)
    window.show()
    sys.exit(app.exec_())

