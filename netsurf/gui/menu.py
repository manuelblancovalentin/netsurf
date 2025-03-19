""" Numpy """
import numpy as np

""" Pyqt5 """
from PyQt5.QtWidgets import QMenu, QAction, QMessageBox

""" Matplotlib """
import matplotlib.pyplot as plt

""" netsurf """
import netsurf

""" Define some custom context menus """
class GenericContextMenu(QMenu):
    def __init__(self, parent, item):
        super().__init__(parent)
        self.bucket = parent.bucket
        self._parent = parent

    def __add__(self, menu):
        # Add actions to the menu
        for action in menu.actions():
            self.addAction(action)
        # same for submenus 
        for submenu in menu.findChildren(QMenu):
            self.addMenu(submenu)
        return self

            
""" Open row """
class OpenRowContextMenu(GenericContextMenu):
    def __init__(self, parent, item):
        super().__init__(parent, item)

        # Add actions
        open_action = QAction("Open", parent)
        self.addAction(open_action)
        
        # Connect actions to methods
        open_action.triggered.connect(lambda: self.open_item(item))

    """ Open directory using the file manager """
    def open_item(self, item):
        if item:
            # Map visual row to actual model row
            netsurf.utils.open_directory(self.bucket.dir)

""" reload bucket """
class ReloadBucketContextMenu(GenericContextMenu):
    def __init__(self, parent, item):
        super().__init__(parent, item)

        # Add action 
        reload_bucket_action = QAction("Reload bucket", parent)
        self.addAction(reload_bucket_action)

        # Connect actions to methods
        reload_bucket_action.triggered.connect(lambda: self.reload_bucket(item))
    
    def reload_bucket(self, item):
        if item:
            if item.bucket is not None:            
                # Ask the user if they are sure they want to delete the directory
                # Create a confirmation dialog 
                reply = QMessageBox.question(self, 'Reload bucket', f"Are you sure you want to reload the bucket {item.bucket.name}?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.Yes:
                    # Get main window
                    wd = netsurf.utils.get_main_window_parent(self)
                    
                    # Update the progress bar fcn
                    pbar = lambda value, text: wd.update_progressbar(value, text)
                    
                    # Create a progress tracker that allows us to keep track of the progress and update the progress bar in the GUI
                    custom_pbar = netsurf.core.explorer.RecursiveProgressTracker(pbar, offset = 0.0, factor = 100)
    
                    # Reload the bucket
                    item.bucket.reload(pbar = custom_pbar)
                    

""" Save bucket """
class SaveBucketContextMenu(GenericContextMenu):
    def __init__(self, parent, item):
        super().__init__(parent, item)

        # Add action 
        save_bucket_action = QAction("Save bucket", parent)
        self.addAction(save_bucket_action)

        # Connect actions to methods
        save_bucket_action.triggered.connect(lambda: self.save_bucket(item))
    
    def save_bucket(self, item):
        if item:
            # Try to get initial path
            initial_path = None
            if item.bucket is not None:
                initial_path = item.bucket.dir
                netsurf.utils.show_save_bucket_dialog(self, initial_path, item.bucket)


""" Delete row """
class DeletableRowContextMenu(GenericContextMenu):
    def __init__(self, parent, item):
        super().__init__(parent, item)

        # Add actions
        delete_action = QAction("Delete from system", parent)
        self.addAction(delete_action)

        # Connect actions to methods
        delete_action.triggered.connect(lambda: self.delete_item(item))
    
    def delete_item(self, item):
        if item:
            # Map visual row to actual model row
            model_row = self.table.row(item)
            model_col = self.table.column(item)
            run_dir = self.table.item(model_row, list(self._parent.df.columns).index('run_dir')).text()
            # Ask the user if they are sure they want to delete the directory
            # Create a confirmation dialog 
            reply = QMessageBox.question(self, 'Delete directory', f"Are you sure you want to delete the directory {run_dir}?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                # Delete the directory
                netsurf.utils.delete_directory(run_dir)
                # Update the DataFrame
                # First find the row in the DataFrame by checking the id column
                df_row = self._parent.df[self._parent.df['run_dir'] == run_dir].index
                self._parent.df.drop(df_row, inplace=True)
                # Remove the whole row 
                self.table.removeRow(model_row)
                
""" Run job """
class RunJobRowContextMenu(GenericContextMenu):
    def __init__(self, parent, item):
        super().__init__(parent, item)

        # Add actions
        run_job_action = QAction("Run job", parent)
        self.addAction(run_job_action)

        # Connect actions to methods
        run_job_action.triggered.connect(lambda: self.run_job(item))
    
    def run_job(self, item):
        if item:
            # Map visual row to actual model row
            model_row = self.table.row(item)
            model_col = self.table.column(item)
            run_dir = self.table.item(model_row, list(self._parent.df.columns).index('run_dir')).text()
            # Ask the user if they are sure they want to delete the directory
            # Create a confirmation dialog 
            reply = QMessageBox.question(self, 'Run job', f"Are you sure you want to run the job {run_dir}?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                # Run the job
                #utils.run_job(run_dir)
                print(f"Running job {run_dir}")

    
""" Plot coverage """
class PlotCoverageRowContextMenu(GenericContextMenu):
    def __init__(self, parent, item):
        super().__init__(parent, item)

        # Add actions
        plot_coverage_action = QAction("Plot coverage", parent)
        self.addAction(plot_coverage_action)

        # Connect actions to methods
        plot_coverage_action.triggered.connect(lambda: self.plot_coverage(item))

    """ Plot coverage """
    def plot_coverage(self, item):
        if item:
        # Map visual row to actual model row
            run_dir = item.bucket.dir
            # Ask the user if they are sure they want to delete the directory
            # Create a confirmation dialog 
            #reply = QMessageBox.question(self, 'Plot results', f"Are you sure you want to plot the results for {run_dir}?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            reply = QMessageBox.Yes
            if reply == QMessageBox.Yes:

                # Clean up previous closed windows 
                self._parent.cleanup_closed_windows()

                figsize = (7, 7)
                fig, ax = plt.subplots(figsize = figsize)

                # Mark figure as deleatable
                netsurf.utils.mark_figure_as_deletable(fig)

                item.bucket.coverage_pie.plot_coverage(ax = ax, title = '')

                # Add plotwindow
                pw = netsurf.gui.widgets.PlotWindow('coverage', title = "", figure = fig, ax = ax, bucket = item.bucket)

                # Add reference so pyqt5 doesn't delete it...
                self._parent.plot_windows.append(pw)

                # Print number of active windows 
                netsurf.utils.log._log(f"Number of active plot windows: {len(self._parent.plot_windows)}")

                # Show the plot window
                self._parent.plot_windows[-1].show()


""" Plot results """
class PlotResultsRowContextMenu(GenericContextMenu):
    def __init__(self, parent, item):
        super().__init__(parent, item)

        # Add actions
        plot_results_submenu = QMenu("Plot results", parent)

        plot_2d_line_action = QAction("2D line", parent)
        plot_3d_volume_action = QAction("3D volume", parent)
        plot_barplot_action = QAction("Barplot", parent)
        plot_boxplot_action = QAction("Boxplot", parent)
        plot_pruning_relationship = QAction("VUS/AUC vs Pruning", parent)

        plot_results_submenu.addAction(plot_2d_line_action)
        plot_results_submenu.addAction(plot_3d_volume_action)

        # Connect actions to methods
        plot_2d_line_action.triggered.connect(lambda: self.plot_results(item, '2d'))
        plot_3d_volume_action.triggered.connect(lambda: self.plot_results(item, '3d'))

        is_at_least_model_level = item.bucket <= netsurf.explorer.ModelContainer
        is_at_least_quant_level = item.bucket <= netsurf.explorer.QuantizationContainer

        if is_at_least_model_level: 
            plot_results_submenu.addAction(plot_barplot_action)
            plot_barplot_action.triggered.connect(lambda: self.plot_results(item, 'barplot'))
            plot_results_submenu.addAction(plot_boxplot_action)
            plot_boxplot_action.triggered.connect(lambda: self.plot_results(item, 'boxplot'))
        
        if is_at_least_quant_level:
            plot_results_submenu.addAction(plot_pruning_relationship)
            plot_pruning_relationship.triggered.connect(lambda: self.plot_results(item, 'vus_vs_pruning'))

        # Add the submenu to the context menu
        self.addMenu(plot_results_submenu)
    
    def plot_results(self, item, plot_type):
        if item:
            # Map visual row to actual model row
            run_dir = item.bucket.dir

            # Get benchmark and method for this run
            sc = item.bucket.structure_config

            # ylog for mse
            title = None
            ylog = False
            cat_name = ''
            methods = None
            if 'benchmark' in sc:
                b = sc['benchmark']
                ylog = 'tinyml_anomaly_detection' in b
                cat_name = b

            # Build filename 
            #filename = os.path.join(run_dir, f"{b}_{m}_{plot_type}.png")
            filename = None

            # Plot the results
            #utils.plot_results(run_dir)
            #print(f"Plotting results for {run_dir}")

            # Clean up previous closed windows 
            self._parent.cleanup_closed_windows()

            # Set plot interactive OFF 
            plt.ioff()

            # 3d volumes for all experiments and models
            max_attr = None
            if plot_type == '3d':
                figsize = (7, 7)
                figs, axs, _, texts, lines, infos = item.bucket.plot_3D_volumes(filename = filename, cat_name = cat_name, methods = methods, title = '', figsize = figsize, standalone = True)
                max_attr = 'vus'
            elif plot_type == '2d':
                figsize = (7, 7)
                figs, axs, _, texts, lines, infos = item.bucket.plot_2D_curves(filename = filename, cat_name = cat_name, methods = methods, ylog = ylog, title = '', figsize = figsize, standalone = True)
                max_attr = 'auc'
            elif plot_type == 'barplot' or plot_type == 'boxplot':
                figsize = (7, 7)
                figs, axs, _, texts, lines, infos = getattr(item.bucket,f'plot_{plot_type}')(filename = filename, cat_name = cat_name, methods = methods, title = '', figsize = figsize, standalone = True)
                max_attr = 'auc'
            elif plot_type == 'vus_vs_pruning':
                figsize = (7, 7)
                figs, axs, _, texts, lines, infos = item.bucket.plot_vus_vs_pruning(filename = filename, cat_name = cat_name, methods = methods, title = '', figsize = figsize, standalone = True)
                max_attr = 'vus'
            
            
            # Get metric from bucket 
            metric = None
            if 'metric' in item.bucket.hyperspace_global_config._keys:
                metric = item.bucket.hyperspace_global_config['metric']

            # Create a multi plot window 
            plot_window = netsurf.gui.widgets.MultiPlotWindow(title = "Results Viewer")
            
            # Extract max y value (auc) from lines names
            ymax, ymin = None, None
            if len(lines) > 0 and max_attr is not None and plot_type == '2d':
                ymax = np.max([np.max([ll[max_attr] for ll in l.values()]) for l in lines])
                ymin = np.min([np.min([ll[max_attr] for ll in l.values()]) for l in lines])
                #k = np.ceil(np.log10(ymax))
                #ymax = np.round(ymax/10**k, 1)*10**k
                ymax = 1.1*ymax
            elif plot_type == 'barplot' or plot_type == 'boxplot' or plot_type == 'vus_vs_pruning':
                ymax = np.max([axx.get_ylim()[1] for axx in axs])
                ymin = np.min([axx.get_ylim()[0] for axx in axs])

            for fig, ax, t, l, info in zip(figs, axs, texts, lines, infos):
                plot_window.add_plot_window(netsurf.gui.widgets.PlotWindow(plot_type, title = title, 
                                                        figure = fig, ax = ax, 
                                                        info_tag = t if plot_type not in ['barplot', 'boxplot','vus_vs_pruning'] else None, 
                                                        legend_lines = l,
                                                        info = info, 
                                                        metric = metric,
                                                        ymax = ymax, ymin = ymin,
                                                        bucket = item.bucket))

            # Add reference so pyqt5 doesn't delete it...
            self._parent.plot_windows.append(plot_window)

            # Print number of active windows 
            netsurf.utils.log._log(f"Number of active plot windows: {len(self._parent.plot_windows)}")

            # Show the plot window
            self._parent.plot_windows[-1].show()

            # Now try to delete all deletable figures
            netsurf.utils.close_deletable_figures()

                