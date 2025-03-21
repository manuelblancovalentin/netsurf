# Basic modules 
import os
from copy import deepcopy

""" Numpy """
import numpy as np

""" Pandas """
import pandas as pd

""" Matplotlib """
import matplotlib.pyplot as plt 
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
from matplotlib.patches import Arc
import matplotlib.patches as mpatches

""" Sklearn """
import sklearn

""" Tensorflow """
import tensorflow as tf

""" Custom modules """
import netsurf

####################################################################################################
# EXPLORER FUNCTIONS / PLOTTERS
####################################################################################################

""" Plot coverage for Run objects (horizontal bar plot) """
def plot_coverage_barplot(cv, protection_values, ber_values, ax = None, dir_path = None, title = None):

    # Convert protection values to floats
    protection = [float(t) for t in protection_values]
    # Convert bir values to floats
    ber = [float(b) for b in ber_values]

    # Define plot values 
    bar_h = 0.5
    bar_w = 0.8
    
    init_figure = ax is None
    if init_figure: 
        fig, ax = plt.subplots(figsize = (10, 7))
        # Mark figure as deletable
        netsurf.utils.plot.mark_figure_as_deletable(fig)

    # Global vertical bar for protection completion
    patch = mpl.patches.Rectangle((-bar_w, -0.5), bar_w, len(protection), color = 'gray', alpha = 0.6)
    ax.add_patch(patch)

    # Loop thru protection values
    for i in range(len(protection)):
        # This horizontal bar will be placed at x values: [1 to len(ber)]
        # the y value will be [i-bar_h, i+bar_h]
        # the width of the bar will be len(ber)
        # create a horizontal patch 
        # Total 
        patch = mpl.patches.Rectangle((0.5, i-bar_h/2), len(ber), bar_h, color = 'gray', alpha = 0.6)
        ax.add_patch(patch)

        subcv = cv[(cv['protection'] == protection[i])]
        for j in range(len(ber)):
            subsubcv = subcv[subcv['ber'] == ber[j]]
            if len(subsubcv) == 0:
                # No data for this protection and ber
                continue
            
            # Now the patch for the amount that has actually been completed 
            p = subsubcv['coverage'].values[0]
            # clamp p to 1
            p = min(1, p)
            patch = mpl.patches.Rectangle((j + 1 - p/2, i-bar_h/2), p, bar_h, 
                        alpha = 0.6, hatch='//')
            ax.add_patch(patch)
        
        # add vertical line symbolizing the percentage of coverage per protection value
        ct = subcv['coverage'].mean()
        # clamp ct to 1
        ct = min(1, ct)
        col = 'green' if ct == 1.0 else 'red'
        patch = mpl.patches.Rectangle((-bar_w, i - ct/2), bar_w, ct, 
                        alpha = 0.6, hatch='//', facecolor = col)
        ax.add_patch(patch)

    # set lims 
    ax.set_xlim(-1-bar_w/2, len(ber) + 1 + bar_w/2)
    ax.set_ylim(-1, len(protection))

    # set ticks and labels
    ax.set_yticks(np.arange(len(protection)))
    ax.set_yticklabels([f'{t*100:3.1f}%' for t in protection])
    ax.set_xticks(np.arange(len(ber)) + 1)
    ax.set_xticklabels([f'{b*100:3.1f}%' for b in ber])

    # Set xtick labels rotation to 45
    plt.xticks(rotation=45)

    # Set x and y labels
    ax.set_xlabel('BER (Bit-Error Rate)')
    ax.set_ylabel('TMR (Protection)')

    # grid on 
    ax.grid(True)

    # set title 
    if title is not None: ax.set_title(title)

    # Plot tight 
    plt.tight_layout()

    # Save to file 
    if dir_path is not None:
        plot_path = os.path.join(dir_path, "coverage_plot.png")
        fig.savefig(plot_path)
        plt.close(fig)  # Close the plot after saving to file
        return plot_path
    else:
        if init_figure:
            plt.show()
        else:
            fig = ax.figure
        return fig, ax


# Get size of text in plot units
def get_text_size_in_plot_units(ax, text, fontsize=12, linespacing=1.2):
    """
    Calculate the size of a multi-line text string in plot (data) units.

    Parameters:
        ax: The matplotlib axis on which the text will be plotted.
        text: The string of text to measure (can include "\n" for multiple lines).
        fontsize: The font size of the text.
        linespacing: Spacing factor between lines (default is 1.2).

    Returns:
        A tuple (width, height) of the text in plot (data) units.
    """
    # Create a dummy text artist to measure text size
    renderer = ax.figure.canvas.get_renderer()
    text_artist = plt.text(0, 0, text, fontsize=fontsize, linespacing=linespacing)
    bbox = text_artist.get_window_extent(renderer=renderer)

    # Get axis transformation
    trans_data_inv = ax.transData.inverted()

    # Convert text bbox size from display (pixels) to plot units
    bbox_plot_units = trans_data_inv.transform([[bbox.x0, bbox.y0], [bbox.x1, bbox.y1]])
    width = bbox_plot_units[1][0] - bbox_plot_units[0][0]
    height = bbox_plot_units[1][1] - bbox_plot_units[0][1]

    # Remove the dummy text artist
    text_artist.remove()

    return width, height


def arc_patch(theta1, theta2, ax=None, center = (0,0), radius = 1.0, resolution=50, explode = 0.0, label = '', fontsize=12, linespacing=1.2, **kwargs):
    # make sure ax is not empty
    if ax is None:
        ax = plt.gca()

    # Avg theta
    theta_avg = np.radians((theta1 + theta2) / 2)

    # Apply explode
    center = (center[0] + np.cos(theta_avg)*explode, center[1] + np.sin(theta_avg)*explode)

    # generate the points
    theta = np.linspace(np.radians(theta1), np.radians(theta2), resolution)
    points = np.vstack((radius*np.cos(theta) + center[0], 
                        radius*np.sin(theta) + center[1]))

    # Last point before center
    rt1 = (points[0][-1], points[1][-1])
    # Connect last point to center
    points = np.hstack((points, np.array([[center[0]], [center[1]]])))
    # Connect center to first point 
    points = np.hstack((points, np.array([[points[0][0]], [points[1][0]]])))
    # build the polygon and add it to the axes
    poly = mpatches.Polygon(points.T, closed=True, **kwargs)
    ax.add_patch(poly)

    # Add label
    if len(label) > 0:
        # Compute label position
        x = center[0] + radius * np.cos(theta_avg)
        y = center[1] + radius * np.sin(theta_avg)

        # Offset is text length in dpi
        xlabel_offset, ylabel_offset = get_text_size_in_plot_units(ax, label, fontsize=fontsize, linespacing=linespacing)
        offset_factor = 1.5
        xlabel_offset *= offset_factor
        ylabel_offset *= offset_factor

        spacex, spacey = 0.05, 0.05
        # Connect text with arc with a line
        r0 = radius*np.cos(theta_avg) + center[0]
        r1 = radius*np.sin(theta_avg) + center[1]

        if np.sin(theta_avg) > 0:
            va = 'bottom'
            y += ylabel_offset
            spacey = -spacey
            if np.cos(theta_avg) > 0:
                ha = 'left'
                x += xlabel_offset
                spacex = -spacex
            else:
                ha = 'right'
                x -= xlabel_offset
        else:
            va = 'top'
            y -= ylabel_offset
            if np.cos(theta_avg) > 0:
                ha = 'left'
                x += xlabel_offset
                spacex = -spacex
            else:
                ha = 'right'
                x -= xlabel_offset
        # Align text to center
        t = ax.text(x, y, label, ha = 'center', va = va)

        ax.plot([r0, x+spacex], [r1, y+spacey], color = 'black', linewidth = 0.5)

    return poly


def plot_coverage_pie(p, explode = {}, ax = None, title = None):

    for kw in p:
        if kw not in explode:
            explode[kw] = 0.0

    init_plot = ax is None
    if init_plot:
        fig, ax = plt.subplots(figsize=(8, 8))
    
        # Make figure deleatable
        mark_figure_as_deletable(fig)
    
    # Make sure ax.figure is deletable in any case
    mark_figure_as_deletable(ax.figure)

    delta_theta = 360/len(p)
    theta0 = 0
    for ipp, (pname, pp) in enumerate(p.items()):
        # Total angle
        theta1 = theta0 + delta_theta
        _xp = explode[pname]
        # Add progress
        theta_s = (theta0 + theta1) / 2 - delta_theta * pp/2
        theta_e = theta_s + delta_theta * pp
        color = netsurf.config.DEFAULT_COLOR_CYCLE[ipp % len(netsurf.config.DEFAULT_COLOR_CYCLE)]
        # Fill
        arc = arc_patch(theta_s, theta_e, ax=ax, fill=True, facecolor=color, explode = _xp, hatch = '////', label = f'{pname}\n{pp*100:.0f}%')
        # Outline
        arc = arc_patch(theta0, theta1, ax=ax, fill=True, facecolor = color, alpha = 0.2)
        arc = arc_patch(theta0, theta1, ax=ax, fill=False, edgecolor='black', linewidth = 0.5, alpha = 1.0)

        # update theta0
        theta0 += delta_theta

    # Set aspect ratio to equal
    ax.set_aspect('equal')

    # Set axis limits
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    # Remove axis outline and labels
    ax.axis('off')

    if title is not None: ax.set_title(title)

    # plt_tight_layout()
    plt.tight_layout()

    # Show the plot
    if init_plot:
        plt.show()
    else:
        return ax.figure, ax

""" Coverage Pie class """
class CoveragePie:
    def __init__(self, obj, name = None, type = None, factor = 1.0, level = 0, 
                 global_progress = 0.0, local_progress = 0.0, 
                 hyperspace_global_config = None, structure_config = None,
                 verbose = True):
        # Get all values 
        self.obj = obj
        self.factor = factor
        self.level = level
        self.global_progress = global_progress
        self.local_progress = local_progress

        # Deduce type and name
        if type is None:
            type = obj.type if obj is not None else 'experiment'
        self.type = type
        if name is None:
            name = obj.name if obj is not None else 'experiment'
        self.name = name
        
        # Set global metrics 
        if hyperspace_global_config is None:
            hyperspace_global_config = obj.hyperspace_global_config if obj is not None else None
        self.global_metrics = hyperspace_global_config
        if structure_config is None:
            structure_config = obj.structure_config if obj is not None else None
        self.structure_config = structure_config

        # Get this type's children type 
        self.children_type, self.children_property = self.get_children_type()
        tabs = '  ' * self.level
        if verbose: print(f'{tabs}Initializing CoveragePie for {self.name} of type {self.type} with children type {self.children_type} and children property {self.children_property}')

        # # Populate children 
        self.unique_children, self.unique_global_categories = [], []
        if self.type  != 'experiment':
            # Get the unique values (from the global metrics) for this type
            self.unique_children, self.unique_global_categories = self.get_unique_children()
            # get progress recursively if this is not a run
            ps, subpies = self.get_recursive_progress(verbose = verbose)
            local_progress = ps[0]
            global_progress = ps[1]
        self.local_progress = local_progress
        self.global_progress = global_progress
        self.subpies = subpies


    def get_recursive_progress(self, verbose = True):
        # Init progress
        global_progress = 0.0
        # Init subpies dict
        subpies = {}
        # Recalculate subfactor for children
        nunique = 1
        if self.unique_global_categories is not None:
            nunique = len(self.unique_global_categories)
        subfactor = self.factor / nunique

        # First check if this is a run, if so, we can get the progress directly from the subcoverage
        if self.children_type.lower() != 'experiment':
            # Loop thru the unique global categories
            for child in self.unique_global_categories:
                # Try to get child name 
                if hasattr(self.obj, 'get_children_name'):
                    child = self.obj.get_children_name(child, self.obj.dir)[0]

                # Check if this child exists 
                if child not in self.unique_children:
                    # Generate new self.structure_config
                    sc = deepcopy(self.structure_config)
                    try:
                        children_property = self.global_metrics['children_prop'][self.children_type.capitalize()]
                    except:
                        children_property = None
                    sc['children_property'] = children_property
                    sc['children_type'] = children_property.capitalize() if children_property is not None else 'experiment'

                    # This is an empty child, we still have to go down the pipe though
                    #subpie = CoveragePie(self.global_metrics, child, self.children_type, obj = None, factor = subfactor, level = self.level + 1, global_progress = 0.0, local_progress = 0.0)
                    subpie = CoveragePie(None, name = child, type = self.children_type, factor = subfactor, 
                                         level = self.level + 1, global_progress = 0.0, local_progress = 0.0,
                                         hyperspace_global_config = self.global_metrics, 
                                         structure_config = sc, verbose = verbose)
                else:
                    # Get the child's progress
                    #subpie = CoveragePie(self.global_metrics, child, self.children_type, obj = self.obj[child], factor = subfactor, level = self.level + 1)
                    subpie = CoveragePie(self.obj[child], factor = subfactor, level = self.level + 1, verbose = verbose)
                # Add to subpies
                subpies[child] = subpie
                global_progress += subpie.global_progress
        
        else:
            # Get the progress directly from subcoverage
            p = 0.0
            if hasattr(self.obj, 'coverage') and len(self.obj._keys) > 0:
                p = self.obj.coverage['coverage'].mean() if 'coverage' in self.obj.coverage.columns else 0.0
            
            local_progress = p
            global_progress += local_progress*self.factor
            
        # Calculate local progress
        local_progress = global_progress/self.factor
        return (local_progress, global_progress), subpies

    """ Entry point for plotting coverage depending on the type """
    def plot_coverage(self, ax = None, title = None):
        # Get coverage
        cv = self.obj.coverage
        protection_values = self.global_metrics['protection']
        ber_values = self.global_metrics['ber']

        # If this is a run, plot bar of protection/ber coverage 
        if self.type == 'method':
            return netsurf.utils.plot.plot_coverage_barplot(cv, protection_values, ber_values, ax = ax, dir_path = None, title = title)
        else:
            # If this is not a run, plot pie of progress
            p = {key: self.subpies[key].local_progress for key in self.subpies}
            explode = {kw: 0.1 if p[kw] < 10 and p[kw] > 0.0 else 0.0 for kw in p}
            return netsurf.utils.plot.plot_coverage_pie(p, explode = explode, ax = ax, title = title)

    def get_children_type(self):
        return self.structure_config['children_type'], self.structure_config['children_property']

    def get_unique_children(self):
        return [] if self.obj is None else self.obj._keys, self.global_metrics[self.children_property]

    def __repr__(self):
        tabs = '  ' * self.level
        return f'{tabs}{self.name}:[{self.type.capitalize()}Container]'
    

""" Unique colors given structural global metrics """
def get_unique_colors(values, cmap = None):
    if cmap is None:
        cmap = netsurf.config.DEFAULT_COLOR_CYCLE
    # Get unique colors for the metric
    nvals = len(values)
    # repeat colors, cause we will wrap around
    colors = cmap * (nvals // len(cmap) + 1)
    # Get unique colors
    colors = colors[:nvals]
    # Return as a dict
    return dict(zip(values, colors))

# Get status color given some progress/status flag 
def get_status_color(status, progress = 0.5):
    if status == "completed":
        #color = (0, 255, 0, 120)
        color = (129, 228, 127, 120)
    elif status == "redundant":
        # if progress == redundant, blue
        #color = (0, 0, 255, 120)
        color = (127, 188, 228, 120)
    elif status == "deletable":
        # if progress == deletable, red
        #color = (255, 0, 0, 120)
        color = (231, 85, 83, 120)
    else:
        # in between, yellow gradient, from more yellow to more green according to progress
        progress = max(0, min(1, progress))

        # Set color according to progress
        C0 = (255, 255, 0, 120)  # yellow
        C1 = (147, 197, 114) # greenish yellow
        color = (C0[0] + progress*(C1[0] - C0[0]), 
                    C0[1] + progress*(C1[1] - C0[1]), 
                    C0[2] + progress*(C1[2] - C0[2]), 
                    120)
    return color

""" Plot coverage """
def plot_coverage(cv, protection_values, bir_values, num_reps, dir_path = None, tit = None):

    # Convert protection values to floats
    protection = [float(t) for t in protection_values]
    # Convert bir values to floats
    bir = [float(b) for b in bir_values]

    # Define plot values 
    bar_h = 0.5
    bar_w = 0.8

    fig, ax = plt.subplots(figsize = (10, 7))

    # Global vertical bar for protection completion
    patch = mpl.patches.Rectangle((-bar_w, -0.5), bar_w, len(protection), color = 'gray', alpha = 0.6)
    ax.add_patch(patch)

    # Loop thru protection values
    for i in range(len(protection)):
        # This horizontal bar will be placed at x values: [1 to len(bir)]
        # the y value will be [i-bar_h, i+bar_h]
        # the width of the bar will be len(bir)
        # create a horizontal patch 
        # Total 
        patch = mpl.patches.Rectangle((0.5, i-bar_h/2), len(bir), bar_h, color = 'gray', alpha = 0.6)
        ax.add_patch(patch)

        subcv = cv[cv[:,0] == protection[i], 1:]
        for j in range(len(bir)):
            
            # Now the patch for the amount that has actually been completed 
            #ct = subcv[subcv[:,1] == bir[j], -1]
            ct = subcv[j,-1]
            p = ct/num_reps
            # clamp p to 1
            p = min(1, p)
            patch = mpl.patches.Rectangle((j + 1 - p/2, i-bar_h/2), p, bar_h, 
                        alpha = 0.6, hatch='//')
            ax.add_patch(patch)
        
        # add vertical line symbolizing the percentage of coverage per protection value
        ct = np.sum(subcv[:,-1])/(num_reps*len(bir))
        # clamp ct to 1
        ct = min(1, ct)
        col = 'green' if ct == 1.0 else 'red'
        patch = mpl.patches.Rectangle((-bar_w, i - ct/2), bar_w, ct, 
                        alpha = 0.6, hatch='//', facecolor = col)
        ax.add_patch(patch)

    # set lims 
    ax.set_xlim(-1-bar_w/2, len(bir) + 1 + bar_w/2)
    ax.set_ylim(-1, len(protection))

    # set ticks and labels
    ax.set_yticks(np.arange(len(protection)))
    ax.set_yticklabels([f'{t*100:3.1f}%' for t in protection])
    ax.set_xticks(np.arange(len(bir)) + 1)
    ax.set_xticklabels([f'{b*100:3.1f}%' for b in bir])

    # Set xtick labels rotation to 45
    plt.xticks(rotation=45)

    # Set x and y labels
    ax.set_xlabel('BIR (Bit-Injection Rate)')
    ax.set_ylabel('TMR (Protection)')

    # grid on 
    ax.grid(True)

    # set title 
    if tit is not None: ax.set_title(tit)

    # Save to file 
    if dir_path is not None:
        plot_path = os.path.join(dir_path, "coverage_plot.png")
        fig.savefig(plot_path)
        plt.close(fig)  # Close the plot after saving to file
        return plot_path
    else:
        plt.show()


""" WEIGHTS PLOTTERS """
def unravel_weights(mod):
    _ws = mod.weights
    ws = dict()
    for w in _ws:
        if 'batch_normalization' in w.name:
            continue
        new_name = w.name.replace('prune_low_magnitude_','').split('/')[:-1]
        new_name = '/'.join(new_name)
        if new_name not in ws:
            ws[new_name] = {'weights': None, 'bias': None}
        if 'bias' not in w.name:
            ws[new_name]['weights'] = np.array(w.numpy())
        else:
            ws[new_name]['bias'] = np.array(w.numpy())
    return ws

def compute_histogram(weight, num_bits = 6, max_range = None, min_range = None, verbose = True):
    # Get the weights and noise
    _kernel = weight

    # Find the maximum range between weights and noise
    max_range = np.max(_kernel) if max_range is None else max_range
    min_range = np.min(_kernel) if min_range is None else min_range

    # compute the step for our dist
    step = 2**(-num_bits+1)

    # Now let's build the bins according to this range and the step
    # in order to do that, first we need to convert max and min range 
    # to the CLOSEST multiple of step (otherwise our computation of 
    # the bins will be wrong)
    max_range = np.ceil(max_range/step)*step
    min_range = np.floor(min_range/step)*step
    if verbose:
        print(f'Max range: {max_range} - Min range: {min_range}')

    # Now we can build the bins
    bins = np.arange(min_range, max_range + step, step)

    # And now get the histograms
    weights_dist, _  = np.histogram(_kernel.flatten(), bins=bins)

    # Normalize the distributions (divide by area of bins)
    weights_dist_norm = weights_dist/np.sum(step*weights_dist)

    return bins, weights_dist_norm

def plot_model_weights_pie(model, filepath = None, show = True, verbose = True):

    ws = unravel_weights(model)

    # Compute shapes:
    shapes = {k: {'weights': np.prod(np.shape(v['weights'])), 'bias': np.prod(np.shape(v['bias']))} for k,v in ws.items()}

    fig, ax = plt.subplots(figsize = (9,9))
    pts = pd.DataFrame(shapes).T.plot.pie(ax = ax, subplots = True)

    for iax, ax in enumerate(pts):
        # Get legend from ax 
        txts = ax.get_legend().texts
        ax.get_legend().remove()
        
        if iax == 0:
            ax.legend(bbox_to_anchor=(1.0, 1.0), loc='upper center', ncol=int(len(txts)//2), fontsize='small')

    plt.tight_layout()

    if filepath is not None:
        fig.savefig(filepath, bbox_inches='tight')
        if verbose:
            netsurf.utils.log._custom('PLOT', f'Saved figure to {filepath}')

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_confusion_matrix(y, yhat, labels = None, filepath = None, show = True, verbose = True):
    # Get number of classes
    n_classes = y.shape[1]

    if not isinstance(labels, list):
        labels = ['%i' % nr for nr in range(0, n_classes)]  # If you want to look at all the labels
    # labels = ['0','1','9'] # Look at only a few labels, here for digits 0, 1 and 9
    netsurf.utils.log._custom('PLOT',f'Plotting confusion matrix for labels {labels}')

    # Get confusion matrix
    cm = sklearn.metrics.confusion_matrix(y.argmax(axis=1), yhat.argmax(axis=1))
    # Let's expand the cm to add the sum over rows, and the sum over cols 
    cm = np.vstack((cm, cm.sum(axis=0)))
    cm = np.hstack((cm, cm.sum(axis=1)[:, np.newaxis]))

    # Change type
    cm = cm.astype('float')

    # Normalize the bottom row 
    cm[-1,:-1] = np.diag(cm)[:-1] / cm[-1,:-1]
    # Normalize the rightmost column
    cm[:-1,-1] = np.diag(cm)[:-1] / cm[:-1,-1]

    # Last element is the global accuracy
    cm[-1,-1] = np.sum(np.diag(cm)[:-1]) / cm[-1,-1]

    # Normalize the inner square
    cm[:n_classes,:n_classes] = cm[:n_classes,:n_classes] / cm[:n_classes,:n_classes].sum(axis=1)[:, np.newaxis]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(cm, cmap='viridis', vmin=0, vmax=1)

    # Add colorbar
    fig.colorbar(cax)

    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Color of text will depend on the value. If it's above 0.5, it will be black, otherwise white
            color = 'black' if cm[i, j] > 0.5 else 'white'
            ax.text(j, i, f'{cm[i, j]:3.2%}', ha="center", va="center", color=color)

    # Bold rectangle around last row and last column
    rect = mpatches.Rectangle((n_classes-0.5, -0.5), 1, n_classes+1, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    rect = mpatches.Rectangle((-0.5, n_classes-0.5), n_classes+1, 1, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    # Labels should be at the left and top
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')

    # Set ylabel and xlabel
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

    # Set labels
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticklabels(labels + ['Sensitivity'])
    ax.set_yticklabels(labels + ['Precision'])

    # Set title
    ax.set_title('Confusion matrix')

    # Save to file 
    if filepath is not None:
        fp = filepath.format('confusion_matrix')
        fig.savefig(fp)
        if verbose:
            netsurf.utils.log._custom('PLOT', f'Saved confusion matrix to {fp}')

    if show:
        plt.show()
    else:
        plt.close(fig)

# Plot ROC for classification problems 
def plot_ROC(y, yhat, labels = None, filepath = None, show = True,
             ylim = None, xlim = None, xlog = True, ylog = False, verbose = True):

    # Get number of classes
    n_classes = y.shape[1]

    if not isinstance(labels, list):
        labels = ['%i' % nr for nr in range(0, n_classes)]  # If you want to look at all the labels
    # labels = ['0','1','9'] # Look at only a few labels, here for digits 0, 1 and 9
    #print('Plotting ROC for labels {}'.format(labels))

    test_score = (yhat.argmax(axis = 1) == y.argmax(axis=1)).mean()
    netsurf.utils.log._custom('PLOT', f'QKeras accuracy = {test_score:3.2%}')

    df = pd.DataFrame()
    fpr = {}
    tpr = {}
    auc1 = {}
    colors = ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061']
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, label in enumerate(labels):
        df[label] = y[:, int(label)]
        df[label + '_pred'] = yhat[:, int(label)]
        fpr[label], tpr[label], threshold = sklearn.metrics.roc_curve(df[label], df[label + '_pred'])
        auc1[label] = sklearn.metrics.auc(fpr[label], tpr[label])

        # df_q[label] = Y_test[:, int(label)]
        # df_q[label + '_pred'] = predict_qkeras[:, int(label)]
        # fpr_q[label], tpr_q[label], threshold_q = metrics.roc_curve(df_q[label], df_q[label + '_pred'])
        # auc1_q[label] = metrics.auc(fpr_q[label], tpr_q[label])

        plt.plot(
            fpr[label],
            tpr[label],
            label=f'{label}, AUC QKeras = {auc1[label]:3.2%})',
            linewidth=1.5,
            c=colors[i],
            linestyle='solid',
        )
        #plt.plot(fpr_q[label], tpr_q[label], linewidth=1.5, c=colors[i], linestyle='dotted')

    # Plot random line 
    plt.plot([0, 1], [0, 1], color='black', lw=1.5, linestyle='dotted')

    if xlim is None:
        if xlog:
            xlim = (0.01, 1.0)
        else:
            xlim = (0.0, 1.0)
    if ylim is None:
        if ylog:
            ylim = (0.01, 1.0)
        else:
            ylim = (0.0, 1.0)

    #plt.semilogx()
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.xlim(*xlim)
    plt.ylim(*ylim)

    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')

    # Add grid (both minor and major)
    plt.grid(True, which='both')
    plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
    # turn on
    plt.minorticks_on()

    plt.legend(loc='lower right')
    plt.figtext(
        0.2,
        0.83,
        f'Accuracy QKeras bit = {test_score:3.2%}',
        wrap=True,
        horizontalalignment='left',
        verticalalignment='center',
    )
    
    lines = [Line2D([0], [0], ls='-'), Line2D([0], [0], ls='--')]
    
    leg = Legend(ax, lines, labels=['Keras'], loc='lower right', frameon=False)
    ax.add_artist(leg)

    if filepath is not None:
        fp = filepath.format('ROC')
        plt.savefig(fp)
        if verbose: netsurf.utils.log._custom('PLOT', f"Saved ROC plot to {fp}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    
# Plot scatter for regression problems 
def plot_scatter(y, yhat, title = 'Predictions', xlabel = 'True', ylabel = 'Predicted', labels = None, 
                 filepath = None, show = True, verbose = True):
    
    # Ensure y is a numpy array
    y = np.array(y)
    yhat = np.array(yhat)

    # Get number of regressed lines
    n_props = y.shape[1] if y.ndim > 1 else 1

    # If nprops == 1, make sure we add extra dimension to y and yhat
    if n_props == 1:
        y = y[:, np.newaxis]
        yhat = yhat[:, np.newaxis]
    
    # Cols
    colors = ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061']
    
    # Create figure and axis
    fig, axs = plt.subplots(figsize=(7, n_props*7), nrows = n_props, ncols = 1)
    
    # Make sure axs is a list 
    axs = [axs] if n_props == 1 else axs

    # Loop thru properties
    for i in range(n_props):
        # Get current axis
        ax = axs[i]

        # Get color for this plot 
        col = colors[i%len(colors)]

        lbl = labels[i] if labels is not None else f'Property {i}'

        suby = y[:,i]
        subyhat = yhat[:,i]

        mae = np.mean(np.abs(subyhat-suby))
        mse = np.sqrt(np.mean((subyhat-suby)**2))
        netsurf.utils.log._custom('PLOT', f'QKeras mae  = {mae} | mse = {mse}')

        # Compute the r2 score and correlation 
        r2 = sklearn.metrics.r2_score(suby, subyhat)
        corr = np.corrcoef(suby, subyhat.flatten())[0,1]
        netsurf.utils.log._custom('PLOT', f'QKeras r2 score = {r2} | corr = {corr}')

        # Plot scatter
        ax.scatter(suby, subyhat, color = col, alpha = 0.5, label = lbl)
        ax.set_title(lbl)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        # Set lims 
        lims = (np.min([suby.min(), subyhat.min()]), np.max([suby.max(), subyhat.max()]))
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        # Plot 1:1 line
        _ = ax.plot(lims, lims, color = 'black', linestyle = 'dotted', label = '1:1')
        # Add legend 
        # skip latex for now
        #leg_title = r'$mae = ' + f'{mae:.2f}' + r'$' + '\n'
        #leg_title += r'$r^2 = ' + f'{r2*100:3.2f}' + r'\%$' + '\n' 
        #leg_title += r'$corr = ' + f'{corr*100:3.2f}' + r'\%$'
        leg_title = f'MAE = {mae:.2f}\n'
        leg_title += f'R2 = {r2:3.2%}\n'
        leg_title += f'Corr = {corr:3.2%}'
        ax.legend(title = leg_title,loc='upper left')

    # Set suptitle
    if title is not None: fig.suptitle(title)

    if filepath is not None:
        plt.savefig(filepath)
        if verbose: netsurf.utils.log._custom('PLOT', f"Saved Regression plot to {filepath}")
    if show:
        plt.show()
    else:
        plt.close(fig)



# Plot sparsity of weights 
def plot_sparsity(model, filepath = None, show = True, separated = False, verbose = True, 
                  bins = None, xlabel = "Values", ylabel = "Frequency", **kwargs):

    # Get the variables 
    variables = model.trainable_variables

    # Now let's clean up this list by removing some that we don't want to plot
    # because pruning only applies to weight layers, basically let's only keep
    # stuff that has either "kernel" or "bias" in its name
    variables = [v for v in variables if 'kernel' in v.name or 'bias' in v.name 
                 or 'alpha' in v.name or 'gamma' in v.name or 'beta' in v.name]

    var_names = [v.name for v in model.variables]

    # First, compute the range of all variables
    # instead of using max and min, use +- 3 stds
    max_range = -tf.constant(np.inf)
    min_range = tf.constant(np.inf)
    for v in variables:
        max_range = tf.maximum(max_range, tf.reduce_mean(v) + 2*tf.math.reduce_std(v))
        min_range = tf.minimum(min_range, tf.reduce_mean(v) - 2*tf.math.reduce_std(v))

    # Get the bins 
    if bins is None: bins = 50
    edges = tf.linspace(min_range, max_range, bins + 1)

    # Even if separated, we will plot kernel/bias of each layer in the same axs, so 
    # let's find out how many weight layers we have (in other words, for each variable
    # name, remove the part after the "/" and count how many unique names we have)
    nrows = 1
    groups = ['']
    if separated:
        groups = list(np.unique([v.name.split("/")[0] for v in variables]))
        nrows = len(groups)

    # We want to keep the coloring consistent, so let's build a color map, regardless of whether 
    # the plot is separated or not
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in range(len(variables))]
    colors = {v.name: colors[i] for i, v in enumerate(variables)}

    # Init the figure
    fig, axs = plt.subplots(nrows = nrows, ncols = 1, figsize=(12, 6*nrows), sharex = True)

    # if axs is not a list, make it
    if not isinstance(axs, list):
        if nrows == 1:
            axs = [axs]
        axs = list(axs)

    # If this is not separated, we will reorder so we plot the biggest distributions first (so they will be behind)
    order = np.arange(len(variables))
    if not separated:
        order = np.argsort([tf.size(v).numpy() for v in variables])[::-1]

    # Compute the histogram for each variable now according to these bins
    for iv, ii in enumerate(order):
        # Get the variable
        v = variables[ii]
        
        # get vname
        vname = v.name

        # Check if we have the pruning mask for this variable in var_names
        mask_name = vname.replace(':','_prune_mask:')

        if mask_name in var_names:
            mask = model.variables[var_names.index(mask_name)]
            v = tf.boolean_mask(v, mask)
            zeros = tf.reduce_sum(tf.cast(tf.equal(mask, 0), tf.int32))
            sparsity = zeros/tf.size(mask)
        else:
            # Keep track of how many zeros we have in this variable
            zeros = tf.reduce_sum(tf.cast(tf.equal(v, 0), tf.int32))
            # Compute the sparsity
            sparsity = zeros/tf.size(v)
        v = tf.reshape(v, [-1])
        
        # Compute the histogram
        counts = tf.histogram_fixed_width(v, [min_range, max_range], nbins = bins)
        
        # Print the sparsity
        if verbose: netsurf.utils.log._custom('PLOT', f"Variable {vname} has sparsity {sparsity:3.2%}")

        # Get the axs where we should plot this 
        if separated:
            ax = axs[groups.index(vname.split("/")[0])]
        else:
            ax = axs[0]

        # Label
        label = f"{vname} ({sparsity:3.2%})"

        # Plot the histogram
        ax.stairs(counts, edges, label = label, linewidth = 1, fill = True, color = colors[vname], 
                edgecolor='black', alpha = 0.7)

        if separated or iv == 0:
            # ensure grid is set
            ax.grid(True)
            ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
            # turn on
            ax.minorticks_on()
            # Set x and y labels
            ax.set_ylabel(ylabel)
        
        if separated or iv == len(variables) - 1:
            # Set legend and legend-title
            ax.legend(title = 'Variable (Sparsity)', bbox_to_anchor=(1.05, 0.5), loc='center left')
            ax.set_xlabel(xlabel)

    if not separated:
        # Make sure that we re-order the legend so tht they follow the normal order (1,2,3...)
        handles, labels = ax.get_legend_handles_labels()
        new_labels = np.ndarray(len(labels), dtype = object)
        new_handles = np.ndarray(len(labels), dtype = object)
        for i in range(len(labels)):
            new_labels[order[i]] = labels[i]
            new_handles[order[i]] = handles[i]
        # Put legend outside the box (bbox_to_anchor, right middle)
        ax.legend(new_handles, new_labels, title = 'Variable (Sparsity)', 
                bbox_to_anchor=(1.05, 0.5), loc='center left')

    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)
        if verbose: 
            netsurf.utils.log._custom('PLOT', f"Saved sparsity plot to {filepath}")
            netsurf.utils.log._custom('PLOT', f'Plot range is {min_range} to {max_range}')

    if show:
        plt.show()
    else:
        plt.close(fig)

# Function to plot the training history
def plot_training_history(logs, ylog = False, filename = None, show = True, return_fig = False, verbose = True):
    history = logs.history

    # Get unique metrics 
    metrics = [k for k in logs.history.keys() if 'loss' not in k and 'time' not in k ]
    metrics = list(set([m.replace('val_', '') for m in metrics]))

    # Let's plot each metric separatedly
    nplots = 1 + len(metrics)
    fig, axs = plt.subplots(nrows = nplots, ncols = 1, figsize = (13, 3*nplots), sharex = True, height_ratios=[1]*nplots)

    # Plot loss on top
    ax0 = axs[0]

    # Get every term that has 'loss' in it, then we will get the unique terms (without 'val_')
    u_losses = np.unique([k.replace('val_', '') for k in logs.history.keys() if 'loss' in k])
    
    # Plot losses
    for k in u_losses:
        ax0.plot(logs.epoch, history[k],  '.-', label=k)
        if f'val_{k}' in history.keys():
            ax0.plot(logs.epoch, history[f'val_{k}'], '.-', label=f'val_{k}')
    ax0.legend(loc='best')
    if ylog:
        ax0.set_yscale('log')

    # Decide how many ticks you want
    n_ticks = max(5, int(fig.get_size_inches()[0]))
    step = int(np.ceil(len(logs.epoch) / n_ticks))

    # Thin out the epochs and times
    thin_epochs = logs.epoch[::step]
    thin_times = [history['time'][i] for i in range(0, len(logs.epoch), step)]

    # Now set the top ticks using the thinned lists
    ax0_top = ax0.twiny()
    ax0_top.set_xticks(thin_epochs)
    ax0_top.set_xticklabels(thin_times)
    ax0_top.set_xlim(ax0.get_xlim())
    ax0_top.xaxis.set_tick_params(rotation=45)

    # Add timestamps on top of graph
    ax0_top.set_xlabel('Elapsed time (hh:mm:ss)')
    ax0_top.xaxis.set_tick_params(rotation=45)

    """ Metrics """
    # If accuracy exists in metrics, plot that first 
    if 'accuracy' in metrics:
        metrics.remove('accuracy')
        metrics = ['accuracy'] + metrics

    # Plot metrics
    for ix, m in enumerate(metrics):

        ln1 = axs[1+ix].plot(logs.epoch, history[m],  '.-', label=m)
        if 'val_'+m in history.keys():
            ln1 += axs[1+ix].plot(logs.epoch, history['val_'+m], '.-', label='val_'+m)
        axs[1+ix].set_ylabel(m)
        axs[1+ix].legend(loc='best')
        if ylog:
            axs[1+ix].set_yscale('log')
        if m == 'accuracy':
            axs[1+ix].set_ylim([0,1.1])
        #axs[1+ix].grid()
    axs[1+ix].set_xlabel('Epochs')

    for ax in axs:
        ax.set_xticks(thin_epochs)
        # Add grids (both major and minor)
        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2, color='black', linestyle=':')
        # Turn minor on
        ax.minorticks_on()


    plt.tight_layout()

    if filename is not None:
        fig.savefig(filename)
        if verbose: netsurf.utils.log._custom('PLOT', f"Saved training history plot to {filename}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def mark_figure_as_deletable(figure: plt.Figure):
    # Set suffix for fig, so it's deleted later 
    fn = plt.get_fignums()
    if len(fn) > 0:
        i = np.max(fn) + 1
    else:
        i = 0
    name = f'deletable_fig_' + str(i)
    figure.set_label(name)

def close_deletable_figures():    
    prefix = 'deletable_fig_'
    for num in plt.get_fignums():  # Get all active figure numbers
        fig = plt.figure(num)
        label = fig.get_label()
        if label and label.startswith(prefix):
            plt.close(fig)  # Close figures with matching prefix


""" Plot weights histogram according to quantization range """
def plot_quantized_histogram(data: np.ndarray, quantizer: 'QuantizationScheme', 
                             ax: plt.Axes = None, figsize = (7,7), 
                             bins = None, min_value = None, max_value = None,
                             title = None, legend = True, xlabel = None, ylabel = None,
                             filename = None, show = True, flatten = True,
                             type = None):
    
    if hasattr(data,'numpy'):
        data = data.numpy()
    
    # flatten
    num_lines = 1
    # Also get labels for each line 
    labels = ['feat0']
    # Get colors for each line (valid and invalid)
    colors = ['g', 'r']
    if type:
        if type == 'class':
            flatten = False
    if data.ndim == 2 and not flatten:
        # This is a 2D multioutput array, keep as such
        num_lines = data.shape[1]
        # colors from tab20
        colors = plt.cm.tab20.colors[:2*num_lines]
        if isinstance(data, pd.DataFrame):
            labels = list(data.columns)
            # Transform to numpy
            data = data.to_numpy()
        else:
            labels = [f'feat{i}' for i in range(num_lines)] 
    elif data.ndim > 2 or flatten:
        if hasattr(data, 'to_numpy'):
            data = data.to_numpy()
        # This is an image or a volume, so just flatten
        data = data.flatten()

    if max_value is None:
        max_value = data.max(0)
    
    if min_value is None:
        min_value = data.min(0)
    
    if not hasattr(min_value, '__len__'):
        min_value = [min_value]*num_lines
    if not hasattr(max_value, '__len__'):
        max_value = [max_value]*num_lines
    
    # Plot histogram of each layers' weights
    show_me = ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # Compute histogram manually per line
    for i in range(num_lines):
        # Get the bins
        subbins = 1.0*bins if bins is not None else None
        if bins is None or isinstance(bins, int):
            if bins is None:
                bins = 50
            submin = min_value[i]
            submax = max_value[i]
            if submin == submax:
                submax = quantizer.max_value
                submin = quantizer.min_value
            
            subbins = np.linspace(submin, submax, bins)
        
        subd = data[:,i] if (num_lines > 1) and not flatten else data
        counts, edges = np.histogram(subd, bins=subbins)
        edges_centers = (edges[1:] + edges[:-1])/2

        # Divide between elements in the valid range and outside
        valid = np.where((edges_centers >= quantizer.min_value) & (edges_centers <= quantizer.max_value))[0]
        invalid = np.where((edges_centers < quantizer.min_value) | (edges_centers > quantizer.max_value))[0]

        label = f'{labels[i]} ' if num_lines > 1 else ""

        # Now plot bars for each part. Valid in green, invalid in red
        if len(valid) > 0:
            ax.bar(edges_centers[valid], counts[valid], width=np.diff(edges)[0], color=colors[2*i], edgecolor = 'k', alpha=0.5, 
                    label = f'{label}Valid')
        if len(invalid) > 0:
            ax.bar(edges_centers[invalid], counts[invalid], width=np.diff(edges)[0], color=colors[2*i+1], edgecolor = 'k', alpha=0.5, 
                    label = f'{label}Invalid')

    # Add grids, both major and minor
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2, color='black', linestyle=':')
    # turn minor on
    ax.minorticks_on()

    # labels
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)

    # Add title
    if title: ax.set_title(title)

    # Add legend
    if legend: ax.legend()

    if filename:
        fig.savefig(filename)
        netsurf.utils.log._custom('PLOT', f"Saved histogram plot to {filename}")

    if show_me:
        plt.show()
    else:
        return fig, ax
    

""" Define a function to plot the histogram of the activations at the output of each 
    one of the layers 
"""
def plot_histogram_activations(model, X = None, Y = None, show = True, bins = 100, sharex = True):
    # Get quantizer
    Q = model.quantizer
    
    if X is None:
        # random Q sample
        # Now create some random input data, do fwd pass and plot output
        # Create random input using Q
        batch_size = 256
        X = Q.sample((batch_size, *model.in_shape))

    else:
        batch_size = X.shape[0]
    
    # Do forward pass
    if Y is None:
        with tf.device("/CPU:0"):
            if not isinstance(X, tf.Tensor):
                Y = model(tf.convert_to_tensor(X))
            else:
                Y = model(X)
    
    if not isinstance(Y, list):
        Y = [Y]
    
    if hasattr(Y[0], 'numpy'):
        Y = [y.numpy() for y in Y]
        
    # Find max and min values of Y to use the same bins for all 
    # [Errata]: Instead of max/min, use +- 3 stds

    # min_value = min([y.min() for y in Y])
    # max_value = max([y.max() for y in Y])
    min_value = min([np.mean(y) - 3*np.std(y) for y in Y])
    max_value = max([np.mean(y) + 3*np.std(y) for y in Y])
    if isinstance(bins, int) and sharex:
        bins = np.linspace(min_value, max_value, bins)

    # Plot histogram of each layers' weights
    num_outputs = len(Y)
    fig, axs = plt.subplots(num_outputs, 1, figsize=(10, 5*num_outputs), sharex=sharex)

    if not isinstance(axs, np.ndarray):
        axs = [axs]

    for i, ax in enumerate(axs):
        _max = np.mean(Y[i]) + 3*np.std(Y[i]) if not sharex else max_value
        _min = np.min(Y[i]) - 3*np.std(Y[i]) if not sharex else min_value
        plot_quantized_histogram(Y[i], Q, ax = ax, bins = bins, 
                                title = f'Layer {i+1} {model.outputs[i].name} (Out)',
                                min_value = _min, max_value = _max)

    if show:
        plt.show()
    else:
        return fig, axs
    
""" Display samples from data in a grid, if image dataset """
def display_data_img(X, title = None, show = True, filename = None, axs = None, 
                     cmap = 'gray', vmin = 0, vmax = 1, **kwargs):


    num_samples = X.shape[0]
    if axs is None:
        fig, axs = plt.subplots(1, num_samples, figsize=(7, 7))
    else:
        # Make sure axs is a list 
        axs = [axs] if num_samples == 1 else axs
        fig = axs[0].figure
    
    # MAke sure axs is a list 
    if num_samples == 1:
        axs = [axs]
    axs = list(axs)

    for i in range(num_samples):
        axs[i].imshow(X[i], cmap = cmap, vmin = vmin, vmax = vmax)
        axs[i].set_title(title)
        axs[i].axis('off')
        axs[i].grid(False)

    if title is not None: plt.suptitle(title)
    
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)
        netsurf.utils.log._custom('DATA', f'Saved dataset sample to {filename}')

    if show:
        plt.show()


from scipy import ndimage
def turn_grids_on(ax):
    # Add grid (Both)
    ax.grid(True)
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    # Make sure to turn minor ticks on
    ax.minorticks_on()

def run_avg_and_std(values, window):
    #running_avg = np.convolve(values, np.ones(window)/window, mode='same')
    # Convolve with 'reflect' boundary and fill value
    running_avg = ndimage.convolve(values, np.ones(window)/window, mode='reflect', cval=0.0)
    std = ((values - running_avg)**2)/window
    # Append zeros to the beginning and end
    window = int(window/2)
    #running_std = np.sqrt(np.convolve(std, np.ones(window)/window, mode='same'))
    running_std = np.sqrt(ndimage.convolve(std, np.ones(window)/window, mode='reflect', cval=0.0))
    return running_avg, running_std

def plot_avg_and_std(values, window, ax, shadecolor = 'red', alpha = 0.5, ylabel = None):
    running_avg, running_std = run_avg_and_std(values, window)
    ax.plot(running_avg, color = 'black', label='Running Average', lw=.5)
    for i in [1,3,5]:
        ax.fill_between(np.arange(len(running_avg)), running_avg - i*running_std, running_avg + i*running_std, 
                            color=shadecolor, alpha=alpha/i, label=f'±{i}std')
    # legend outside the plot 
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    turn_grids_on(ax)
    if ylabel: ax.set_ylabel(ylabel)
