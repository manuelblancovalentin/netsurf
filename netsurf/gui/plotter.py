""" Numpy """
import numpy as np

""" Pandas """
import pandas as pd

""" netsurf """
import netsurf

""" Matplotlib """
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines

""" warnings """
import warnings

""" scipy """
from scipy.interpolate import griddata # For 3D plots interpolation

""" Random utils """
def create_label(ax, d, x0, y0, fontsize=8, border_color="black", padding=0.3, num_columns=2):
    """
    Create a label in pyplot from a dictionary.

    Parameters:
    - ax: The matplotlib Axes object where the label will be drawn.
    - d: Dictionary containing key-value pairs to display.
    - x0, y0: Coordinates of the bottom-left corner of the label.
    - fontsize: Font size of the text.
    - border_color: Color of the border around the label.
    - padding: Padding around the text inside the border.
    """
    # Format the dictionary into lines
    # Group key-value pairs into rows based on the number of columns
    items = list(d.items())
    rows = [
        "  ║  ".join(
            r"$\bf{" + str(key) + r"}$: " + str(value)  # Format key as bold, value as normal
            for key, value in items[i:i + num_columns]
        )
        for i in range(0, len(items), num_columns)
    ]
    text_content = "\n".join(rows)
    
    # Create a text box with a border
    t = ax.figure.text(
        x0, y0, text_content,
        fontsize=fontsize,
        va="bottom",
        ha="center",
        bbox=dict(
            boxstyle="round,pad={}".format(padding),
            edgecolor=border_color,
            facecolor="white",
        )
    )

    return t

""" Plot boxplot function """
def plot_boxplot(subplotters, 
                 ax = None, y = 'mean', metric = None, colors = None,
                 ylims = None, title = None, xlog = False, ylog = False,
                 show = False, info_label = None, standalone = True, 
                 baseline = None, remove_baseline = False, single_out = 'random', 
                 cmap = 'seismic', filename = None, ylabel = None,
                 sorter = 'mean',
                 **kwargs):

    # Assert sorter 
    assert sorter in ['median', 'mean', 'max', 'min', 'std'], 'Invalid sorter'

    # Loop thru each plotter and get the VUSs
    VUCs = []
    for method in subplotters:
        # Loop thru configs 
        for config in subplotters[method]:
            # Get the plotter obj
            plotter = subplotters[method][config]
            # Get the vuc
            VUCs += [{'method': method, 'config': config, **plotter.vus.loc['vus'].to_dict()}]

    # Convert VUCs to a dataframe
    df = pd.DataFrame(VUCs)
    # Sort by vus 
    df = df.sort_values(sorter, ascending = metric.lower() not in ['accuracy', 'acc'])

    # Xrange is always the number of methods
    # ylims
    if ylims is None:
        ylims = (0.95*df['min'].min(), 1.05*df['max'].max())

    # Create color mapper 
    # Get the min and max values
    vmin = (df['median'] - df['std']).min()
    vmax = (df['median'] + df['std']).max()
    # Create a color palette
    cmap = plt.get_cmap(cmap)
    # Normalize the values
    norm = plt.Normalize(vmin, vmax)
    # Create lambda function to map any value to color later 
    color_mapper = lambda x: cmap(norm(x))
    hatch_styles = {'auc': '**', 'vus': '//'}

    # If ax is none, create a new figure
    if ax is None or standalone:
        fig, ax = plt.subplots(figsize = (7, 10))
        netsurf.utils.mark_figure_as_deletable(fig)
    else:
        fig = ax.figure
    
    # Initialize the width, x of the bar
    bar_w = 0.2
    bar_space = 0.1 # Space between method bars (same method are not spaced)
    lw = 0.0125

    # Store old value of 'hatc.linewidth'
    old_linewidth = plt.rcParams['hatch.linewidth']
    # Set the linewidth of the hatch lines
    plt.rcParams['hatch.linewidth'] = 0.3
    plt.rcParams["lines.solid_capstyle"] = "butt"
    old_grid_color = plt.rcParams['grid.color']
    plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.3)

    # Group by method 
    g = df.groupby('method', sort = False)

    # Set ylims 
    ax.set_ylim(ylims)
    # Set ticks params here (we need them to get the size of the xticklabels)
    ax.tick_params(axis='x', labelsize=9) 

    xticks = []
    xticklabels = []

    # Find the position of the corresponding xtick label
    label = ax.get_xticklabels()[0]  # Get the Text object for the label

    # Use `get_window_extent` to find the label's extent in display space (optional)
    renderer = fig.canvas.get_renderer()
    bbox = label.get_window_extent(renderer=renderer)

    # Convert the bounding box to data coordinates
    inv = ax.transData.inverted()
    bbox_data = inv.transform(bbox)

    # Use the bottom of the bbox as the y-coordinate for the line
    label_y_offset = bbox_data[1][1]  # The top edge of the label in data coordinates
    label_new_line_height = bbox_data[1][1] - bbox_data[0][1]  # The height of the label in data coordinates

    # Now loop thru methods
    boxes = []
    x = 0
    for i, (method, group) in enumerate(g):
        # Loop thru configs, vus
        nconfigs = len(group)
        for j, (_, row) in enumerate(group.iterrows()):
        
            config = row['config']
            median = row['median']
            std = row['std']
            max = row['max']
            min = row['min']

            # Get the x position of the box
            x = i*(bar_w + bar_space) + j*bar_w
            
            # Now get all four coordinates, for simplicity 
            x0, x1, y0, y1, ym = x, x + bar_w, median - std, median + std, median

            # Get this bar's value
            c = color_mapper(y1)
            hs = hatch_styles['vus']
            if single_out is not None:
                if single_out == method:
                    #c = (1, 1, 1, 1) if btype == 'AUC' else (0, 0, 0, 1)
                    c = (0, 0, 0, 1)
                    hs = 'o'

            # Create a box going from the median to the top of the box
            # The color of this box will be the equivalent of the bottom value of the box 
            ptop = ax.fill_between([x0, x1], ym, y1, 
                                    color=c[:-1] + (0.3,), 
                                    edgecolor = 'k', 
                                    linewidth = 0.5)
            # Add hatch to this patch we just created 
            ptop.set_hatch('//')

            # And now one from the median down to the bottom of the box
            pbot = ax.fill_between([x0, x1], y0, ym, 
                                    color=c[:-1] + (0.7,), 
                                    edgecolor = 'k', 
                                    linewidth = 0.5)
            # Add hatch to this patch we just created
            pbot.set_hatch('\\\\')

            # Add a line for the median
            m = ax.plot([x0, x1], [ym, ym], color='k', linewidth = 1.4)

            """ Add whiskers now """
            # Top whisker 
            wx0, wx1, wy0, wy1 = x + bar_w/2 - lw/2, x + bar_w/2 + lw/2, median + std, max
            # try this with a patch instead of a line 
            warnings.filterwarnings("ignore")
            ax.imshow([[wy1, wy1], [wy0, wy0]], 
                cmap = cmap, 
                extent = [wx0, wx1, wy0, wy1],
                interpolation = 'bicubic', 
                vmin = vmin, vmax = vmax,
                alpha = 0.8
            )
            
            # Create a Rectangle patch with the desired border color
            rect = plt.Rectangle((wx0, wy0), lw, wy1-wy0, 
                                edgecolor='k', linewidth = 0.4, facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
            # Finally, add the whisker line (horizontal line)
            ax.plot([x + bar_w/2 - lw, x + bar_w/2 + lw], [wy0, wy0], color='k', linewidth = 0.1)

            # Bottom whisker 
            wx0, wx1, wy0, wy1 = x + bar_w/2 - lw/2, x + bar_w/2 + lw/2, min, median - std
            # try this with a patch instead of a line 
            warnings.filterwarnings("ignore")
            ax.imshow([[wy1, wy1], [wy0, wy0]], 
                cmap = cmap, 
                extent = [wx0, wx1, wy0, wy1],
                interpolation = 'bicubic', 
                vmin = vmin, vmax = vmax,
                alpha = 0.8
            )

            # Create a Rectangle patch with the desired border color
            rect = plt.Rectangle((wx0, wy0), lw, wy1-wy0, 
                                edgecolor='k', linewidth = 0.4, facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
            # Finally, add the whisker line (horizontal line)
            ax.plot([x + bar_w/2 - lw, x + bar_w/2 + lw], [wy1 + 0.01, wy1 - 0.01], color='k', linewidth = 1)
            line = mlines.Line2D(
                    [x + bar_w/2 - lw, x + bar_w/2 + lw],           # x-coordinates
                    [wy1, wy1],        # y-coordinates (outside the plot area)
                    color="black",
                    lw=0.8
                )
            # Add the line to the plot
            ax.add_artist(line)

            # # Now let's get the actual data points for this method 
            # dfm = subplotters[method][config].curves
            # # Make sure len(dfm) > 1, otherwise just pick the only point
            # points = dfm['auc'].values
            # points_tmrs = dfm['tmr'].values
            # if 'tmr_color' in dfm:
            #     pointcols = dfm['tmr_color'].values
            # else:
            #     pointcols = [cmapper(p) for p in points_tmrs]
            
            # # Get the color for each point 
            # #pointcols = [cmapper(p) for p in points]
            # ax.scatter([i+1.5]*len(points), points, edgecolor = 'k', linewidth = 0.4, alpha = 0.8, color = pointcols, s = 30)
            
            # Append to boxes
            #boxes.append([ptop, pbot])

            # Add a label on top of the box with the value of the VUS
            ax.text(x + bar_w/2, max + label_new_line_height/2, f'{median:.3f}', ha='center', va='bottom', fontsize=9)

            # Add xlabel to list 
            xticks += [x + bar_w/2]
            sp = "".join(['\n']*((i+j)%2))
            mstr = method.replace('_', '\n').replace(' ', '\n')
            mstr = mstr.replace('delta', r'$\Delta$')
            xticklabels += [f'{sp}{mstr}\n{config}' if nconfigs > 1 else f'{sp}{mstr}']

            # Add a line to connect the bar to the xticklabel underneath (only if i+j is odd)
            if (i+j) % 2 == 1:
                line = mlines.Line2D(
                    [x + bar_w/2, x + bar_w/2],           # x-coordinates
                    [-label_new_line_height, 2*label_new_line_height],        # y-coordinates (outside the plot area)
                    color="black",
                    lw=0.8
                )
                line.set_clip_on(False)  # Ensure the line is not clipped by the axis
                ax.add_artist(line)     # Add the line as an artist

    # Set grid to dashed and also turn minor grid on
    ax.grid(which='major', linestyle='--')
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':')

    # Update xlims
    ax.set_xlim(-bar_w/2 - bar_space, x + bar_w + bar_space)

    # Set xticks and xticklabels
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation = 0, ha = 'center')

    # Setup labels correctly
    # parse ylabel
    if ylabel:
        ylabel = ylabel.replace('mae', 'Mean Absolute Error').replace('mse', 'Mean Squared Error').replace('accuracy', 'Accuracy')
        ax.set_ylabel(ylabel)

    # Set scale
    if xlog: ax.set_xscale('log')
    if ylog: ax.set_yscale('log')

    # Setup the title 
    if title: ax.set_title(title)

    ax.set_aspect('auto')
    # if vmin == vmax:
    #     vmin = 0
    #     vmax = 1
    # if np.isnan(vmin) or np.isnan(vmax):
    #     vmin = 0
    #     vmax = 1
    # if np.isinf(vmin) or np.isinf(vmax):
    #     vmin = 0
    #     vmax = 1
    # ax.set_ylim(0.95*vmin, 1.05*vmax)
    ax.set_ylim(ylims)

    # Add the info label
    t = None
    if len(info_label) > 0:
        # Get axis position in figure-relative coordinates
        axis_position = ax.get_position()  # Returns (x0, y0, width, height)
        y_top = axis_position.y1 + 0.03  # Slightly above the top of the axis (in figure-relative coordinates)
        # Create label
        t = create_label(ax, info_label, 0.5, y_top, fontsize=9, border_color="black", padding=0.5, num_columns=2)
    
    if show:
        plt.show(block=False)  # Show without blocking the PyQt5 event loop
    else:
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
            plt.close()

    """ Restore the default configuration """
    # Set the 'hatch.linewidth' back to its original value
    plt.rcParams['hatch.linewidth'] = old_linewidth
    # Set the 'grid.color' back to its original value
    plt.rcParams['grid.color'] = old_grid_color

    return ax.figure, ax, t, boxes

""" Plot barplot function """
def plot_barplot(subplotters, 
                 ax = None, y = 'mean', metric = None,  
                 ylims = None, title = None, xlog = False, ylog = False,
                 show = False, info_label = None, standalone = True, 
                 baseline = None, remove_baseline = False, single_out = 'random', 
                 cmap = 'viridis', filename = None, ylabel = None,
                 **kwargs):

    # Loop thru each plotter and get the VUSs
    VUCs = []
    for method in subplotters:
        # Loop thru configs 
        for config in subplotters[method]:
            # Get the plotter obj
            plotter = subplotters[method][config]
            # Get the vuc
            VUCs += [{'method': method, 'config': config, 'vus': plotter.vus.loc['vus'][y]}]

    # Convert VUCs to a dataframe
    df = pd.DataFrame(VUCs)
    # Sort by vus 
    df = df.sort_values('vus', ascending = metric.lower() not in ['accuracy', 'acc'])

    if remove_baseline:
        if baseline in df['method'].values:
            # subtract baseline values 
            df['vus'] = df['vus'] - df[df['method'] == baseline]['vus'].values[0]

    # Xrange is always the number of methods
    # ylims
    if ylims is None:
        ylims = (0, 1.1*df['vus'].max())

    # Create color mapper 
    # Get the min and max values
    vmin = df['vus'].min()
    vmax = df['vus'].max()
    # Create a color palette
    cmap = plt.get_cmap(cmap)
    # Normalize the values
    norm = plt.Normalize(vmin, vmax)
    # Create lambda function to map any value to color later 
    color_mapper = lambda x: cmap(norm(x))
    hatch_styles = {'auc': '**', 'vus': '//'}

    # If ax is none, create a new figure
    if ax is None or standalone:
        fig, ax = plt.subplots()
        netsurf.utils.mark_figure_as_deletable(fig)
    else:
        fig = ax.figure
    
    # Initialize the width, x of the bar
    bar_w = 0.4
    bar_space = 0.1 # Space between method bars (same method are not spaced)

    """ Pyplot configuration for hatches """
    # Store old value of 'hatc.linewidth'
    old_linewidth = plt.rcParams['hatch.linewidth']
    # Set the linewidth of the hatch lines
    plt.rcParams['hatch.linewidth'] = 0.3
    plt.rcParams["lines.solid_capstyle"] = "butt"
    old_grid_color = plt.rcParams['grid.color']
    plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.3)

    # Group by method 
    g = df.groupby('method', sort = False)

    # Set ylims 
    ax.set_ylim(ylims)
    # Set ticks params here (we need them to get the size of the xticklabels)
    ax.tick_params(axis='x', labelsize=9) 

    xticks = []
    xticklabels = []

    # Find the position of the corresponding xtick label
    label = ax.get_xticklabels()[0]  # Get the Text object for the label

    # Use `get_window_extent` to find the label's extent in display space (optional)
    renderer = fig.canvas.get_renderer()
    bbox = label.get_window_extent(renderer=renderer)

    # Convert the bounding box to data coordinates
    inv = ax.transData.inverted()
    bbox_data = inv.transform(bbox)

    # Use the bottom of the bbox as the y-coordinate for the line
    label_y_offset = bbox_data[1][1]  # The top edge of the label in data coordinates
    label_new_line_height = bbox_data[1][1] - bbox_data[0][1]  # The height of the label in data coordinates

    # Now loop thru methods
    bars = []
    for i, (method, group) in enumerate(g):
        # Loop thru configs, vus
        nconfigs = len(group)
        for j, (config, vus) in enumerate(zip(group['config'], group['vus'])):
            # Get the x position of the bar
            x = i*(bar_w + bar_space) + j*bar_w
            
            # Now get all four coordinates, for simplicity 
            x0, x1, y0, y1 = x, x + bar_w, 0, vus

            # Get this bar's value
            c = color_mapper(y1)
            hs = hatch_styles['vus']
            if single_out is not None:
                if single_out == method:
                    #c = (1, 1, 1, 1) if btype == 'AUC' else (0, 0, 0, 1)
                    c = (0, 0, 0, 1)
                    hs = 'o'

            # Add rectangle (we will delete the old bar)
            bar = ax.fill_between([x0, x1], y0, y1, color=c[:-1] + (0.3,), edgecolor = 'k', linewidth = 0.5, label = method)
            # Add hatch to this patch we just created 
            bar.set_hatch(hs)
            bars.append(bar)

            # Add a label on top of the bar with the value of the VUS
            ax.text(x + bar_w/2, vus + label_new_line_height/2, f'{vus:.3f}', ha='center', va='bottom', fontsize=9)

            # Add xlabel to list 
            xticks += [x + bar_w/2]
            sp = "".join(['\n']*((i+j)%2))
            mstr = method.replace('_', '\n').replace(' ', '\n')
            mstr = mstr.replace('delta', r'$\Delta$')
            xticklabels += [f'{sp}{mstr}\n{config}' if nconfigs > 1 else f'{sp}{mstr}']

            # Add a line to connect the bar to the xticklabel underneath (only if i+j is odd)
            if (i+j) % 2 == 1:
                line = mlines.Line2D(
                    [x + bar_w/2, x + bar_w/2],           # x-coordinates
                    [0, label_y_offset - label_new_line_height*0.7],        # y-coordinates (outside the plot area)
                    color="black",
                    lw=0.8
                )
                line.set_clip_on(False)  # Ensure the line is not clipped by the axis
                ax.add_artist(line)     # Add the line as an artist

    # Set grid to dashed and also turn minor grid on
    ax.grid(which='major', linestyle='--')
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':')

    # Set xticks and xticklabels
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation = 0, ha = 'center')

    # Setup labels correctly
    # parse ylabel
    if ylabel:
        ylabel = ylabel.replace('mae', 'Mean Absolute Error').replace('mse', 'Mean Squared Error').replace('accuracy', 'Accuracy')
        ax.set_ylabel(ylabel)

    # Set scale
    if xlog: ax.set_xscale('log')
    if ylog: ax.set_yscale('log')

    # Setup the title 
    if title: ax.set_title(title)

    # Add the info label
    t = None
    if len(info_label) > 0:
        # Get axis position in figure-relative coordinates
        axis_position = ax.get_position()  # Returns (x0, y0, width, height)
        y_top = axis_position.y1 + 0.03  # Slightly above the top of the axis (in figure-relative coordinates)
        # Create label
        t = create_label(ax, info_label, 0.5, y_top, fontsize=9, border_color="black", padding=0.5, num_columns=2)
    
    if show:
        plt.show(block=False)  # Show without blocking the PyQt5 event loop
    else:
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
            plt.close()

    """ Restore the default configuration """
    # Set the 'hatch.linewidth' back to its original value
    plt.rcParams['hatch.linewidth'] = old_linewidth
    # Set the 'grid.color' back to its original value
    plt.rcParams['grid.color'] = old_grid_color

    return fig, ax, t, bars


    # OLD CODE
    VUCs = self.global_df.groupby('method')['vuc'].mean().to_dict()
    AUCs = self.global_df.groupby('method')['auc'].mean().to_dict()
    #AUCs = self.curves.groupby('simplified_title')[self.metric].mean().to_dict()

    if baseline is not None:
        if baseline not in VUCs or baseline not in AUCs:
            if verbose:
                print(f'[INFO] - Baseline {baseline} not found in methods, using global baseline.')
            baseline = None

    auc_baseline = 0.0
    vuc_baseline = 0.0
    if baseline is None:
        if remove_baseline:
            # get baseline value
            if 0.0 not in self.global_df['tmr'].unique():
                if verbose:
                    print(f'[INFO] - No baseline value found, using global minimum.')
                auc_baseline = np.nanmin(self.global_df[self.global_df['tmr'] == 0.2]['auc'])
                #vuc_baseline = self.global_df[self.global_df['tmr'] == 0.0]['vuc'].mean()
                # vuc for zero tmr doesn't make sense, so just use auc 
                vuc_baseline = auc_baseline
            else:
                auc_baseline = np.nanmin(self.global_df[self.global_df['tmr'] == 0.0]['auc'])
                #vuc_baseline = self.global_df[self.global_df['tmr'] == 0.0]['vuc'].mean()
                # vuc for zero tmr doesn't make sense, so just use auc 
                vuc_baseline = auc_baseline
    else:
        # get auc and vuc
        auc_baseline = AUCs[baseline]
        vuc_baseline = VUCs[baseline] 

    # Subtract baseline (note that if remove_baseline is False, this will have no effect)
    if remove_baseline:
        if verbose:
            print(f'[INFO] - Removing baseline values: AUC={auc_baseline:.3f}, VUC={vuc_baseline:.3f}')
        VUCs = {k: v - vuc_baseline for k,v in VUCs.items()}
        AUCs = {k: v - auc_baseline for k,v in AUCs.items()}               


    # Emojis for first second and third place in ranking
    emojis = ['🥇', '🥈', '🥉']
    # Increase emojis to match the number of methods (append with "None")
    emojis += [None] * (len(VUCs) - len(emojis))
    # If not accuracy, reverse the order
    emojis = emojis if 'accuracy' not in self.metric else emojis[::-1]
    
    def offset_image(cords, emoji, ax, zoom = 0.08):
        img = plt.imread(imojify.get_img_path(emoji))
        im = OffsetImage(img, zoom=zoom)
        im.image.axes = ax
        ab = AnnotationBbox(im, (cords[0], cords[1]),  frameon=False, pad=0)
        ax.add_artist(ab)


    fig, ax = plt.subplots(figsize = (12,10))

    df = pd.DataFrame(VUCs, index = [0])
    df = df.T.rename(columns={0:'VUC'}).sort_values('VUC')
    df2 = pd.DataFrame(AUCs, index = [0])
    df2 = df2.T.rename(columns={0:'AUC'})

    df = pd.concat((df,df2), axis = 1)

    maxy = 1.0
    miny = 0.0
    fontsize = 8
    #if remove_baseline:
    maxy = np.nanmax([np.nanmax(list(VUCs.values())), np.nanmax(list(AUCs.values()))])
    maxy = 1.25*maxy if maxy > 0 else 0.8*maxy
    miny = np.nanmin([np.nanmin(list(VUCs.values())), np.nanmin(list(AUCs.values()))])
    miny = 1.25*miny if miny < 0 else 0.8*miny

    if ylog:
        maxy = np.ceil(np.log10(np.maximum(maxy, 1e-1)))
        miny = np.log10(np.maximum(maxy, 1e-1))
        if miny < 0:
            miny = -np.ceil(np.abs(miny))
        else:
            miny = -1

    if 'accuracy' in self.metric and not remove_baseline:
        maxy = 1.2
        miny = 0.0

""" plot_vus_vs_pruning """
def plot_vus_vs_pruning(subplotters, 
                 ax = None, x = 'pruning', y = 'vus', hue = 'method', 
                 metric = None, colors = None,
                 ylims = None, title = None, xlog = False, ylog = False,
                 show = False, info_label = None, standalone = True, 
                 baseline = None, remove_baseline = False, single_out = 'random', 
                 cmap = 'viridis', filename = None, ylabel = None,
                 **kwargs):

    # Loop thru each plotter and get the VUSs
    VUCs = []
    for comb in subplotters:
        # parse 
        pruning_factor, model_name, method = comb
        # Loop thru configs 
        for config in subplotters[comb]:
            # Get the plotter obj
            plotter = subplotters[comb][config]
            # Get the vuc
            VUCs += [{'model_name': model_name, 
                        'pruning': pruning_factor, 
                        'method': method, 
                        'config': config, 
                        'vus': plotter.vus.loc['vus']['mean']}]

    # If colors is none, just get the default colors
    if colors is None and len(VUCs) > 0 and hue is not None:
        if hue in plotter.curves:
            # Get unique colors for the metric
            nvals = len(plotter.curves[hue].unique())
            # repeat colors, cause we will wrap around
            cmap = netsurf.config.DEFAULT_COLOR_CYCLE
            colors = cmap * (nvals // len(cmap) + 1)
            # Get unique colors
            colors = colors[:nvals]
            # Create a dictionary with the colors
            colors = {val: color for val, color in zip(plotter.curves[hue].unique(), colors)}

    # Convert VUCs to a dataframe
    df = pd.DataFrame(VUCs)
    # Sort by vus 
    df = df.sort_values('vus', ascending = metric.lower() not in ['accuracy', 'acc'])

    # Xrange is always the number of methods
    # ylims
    if ylims is None:
        ylims = (0, 1.1*df['vus'].max())

    
    # If ax is none, create a new figure
    if ax is None or standalone:
        fig, ax = plt.subplots()
        netsurf.utils.mark_figure_as_deletable(fig)
    
    # Group by method 
    g = df.groupby('method', sort = False)

    # Set ylims 
    ax.set_ylim(ylims)
    # Set ticks params here (we need them to get the size of the xticklabels)
    ax.tick_params(axis='x', labelsize=9) 

    # Now loop thru methods
    lines = {}
    for i, (method, group) in enumerate(g):
        # Loop thru configs
        gg = group.groupby('config')
        nconfigs = len(gg)
        for j, (config, dta) in enumerate(gg):
            
            # Sort dta by pruning factor
            dta = dta.sort_values(x)

            # Plot line for pruning factors vs VUS
            dx = dta[x].values
            dy = dta[y].values

            # Plot 
            fmt = f'{method}' if nconfigs <= 1 else f'{method} - {config}'
            l = ax.plot(dx, dy, 'o-', label = fmt, markersize = 3)

            # Add line 
            lines[fmt] = {'line': l[0], 'method': method, 'config': config}
 
    # Set grid to dashed and also turn minor grid on
    ax.grid(which='major', linestyle='--')
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':')

    # Set xticks and xticklabels to unique pruning factors
    xticks = df[x].unique()
    xticks.sort()
    xticklabels = [f'{100*xx:3.1f}%' if 'pruning' in x else f'{xx:.1f}' for xx in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation = 0, ha = 'center')

    # Setup labels correctly
    # parse ylabel
    if ylabel:
        ylabel = ylabel.replace('mae', 'Mean Absolute Error').replace('mse', 'Mean Squared Error').replace('accuracy', 'Accuracy')
        ax.set_ylabel(ylabel)

    # Set scale
    if xlog: ax.set_xscale('log')
    if ylog: ax.set_yscale('log')

    # Setup the title 
    if title: ax.set_title(title)

    # Add the info label
    t = None
    if len(info_label) > 0:
        # Get axis position in figure-relative coordinates
        axis_position = ax.get_position()  # Returns (x0, y0, width, height)
        y_top = axis_position.y1 + 0.03  # Slightly above the top of the axis (in figure-relative coordinates)
        # Create label
        t = create_label(ax, info_label, 0.5, y_top, fontsize=9, border_color="black", padding=0.5, num_columns=2)
    
    if show:
        plt.show(block=False)  # Show without blocking the PyQt5 event loop
    else:
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
            plt.close()

    return ax.figure, ax, t, lines
                        

""" Experiments Plotter class """
class ExperimentsPlotter:
    def __init__(self, data, metric = 'accuracy', structure_config = {}, **kwargs):
        self.data = data
        self.metric = metric
        self.structure_config = structure_config

        # Rename true radiation to true_ber
        cs = [('true_radiation', 'true_ber'), ('tmr', 'protection'), ('noise_accuracy', 'accuracy'),
             ('avg_mse', 'mse')]
        self.data = self.data.rename(columns = {k: v for k,v in cs if k in self.data.columns})

        # We want to get the std and avg for each protection, ber combination
        curves = self.data.groupby(['protection', 'ber', 'true_ber'])[self.metric].agg(['median', 'mean', 'std', 'min', 'max']).reset_index()
        # Make sure all values are interpreted as numbers
        self.curves = curves.apply(pd.to_numeric)

        # Loop thru the protections groups and compute the auc for each protection and ber
        self.aucs = self.compute_auc(self.curves)

        # Compute the volume under the curve for all points using the tetrahedron method
        self.vus = self.compute_vus(self.curves)
    
    # Method to plot 2D curves 
    def plot_2D_curves(self, x = 'ber', y = 'mean', hue = 'protection', style = 'protection', ax = None, colors = None, 
                            xrange = None, yrange = None, title = None, xlog = True, ylog = False, 
                            xlabel = None, ylabel = None, xlims = None, ylims = None,
                            info_label = {}, standalone = True, **kwargs):
        
        # If data is empty just skip 
        if len(self.aucs) == 0:
            return None, None, None, None
        
        # If colors is none, just get the default colors
        if colors is None:
            # Get unique colors for the metric
            nvals = len(self.curves[hue].unique())
            # repeat colors, cause we will wrap around
            cmap = netsurf.config.DEFAULT_COLOR_CYCLE
            colors = cmap * (nvals // len(cmap) + 1)
            # Get unique colors
            colors = colors[:nvals]
            # Create a dictionary with the colors
            colors = {val: color for val, color in zip(self.curves[hue].unique(), colors)}

        # Get the data 
        data = self.curves
        aucs = self.aucs.groupby(['x','y']).get_group((x,y))

        # If xrange is none, get the unique of x
        if xrange is None:
            xrange = data[x].unique()
        if yrange is None:
            yrange = data[y].unique()
        
        # Make sure xrange and yrange are np.arrays
        xrange = np.array(xrange)
        yrange = np.array(yrange)

        # If ax is none, create a new figure
        if ax is None or standalone:
            fig, ax = plt.subplots()
            netsurf.utils.mark_figure_as_deletable(fig)
    
        # Store the lines for the legend so we can link them to Qt widgets to turn them on and off
        lines = {}

        # Loop thru values of hue 
        for i, (val, group) in enumerate(data.groupby(hue)):
            # Get the color
            color = colors[val]

            # Plot the curve
            # get auc 
            auc = aucs[aucs[hue] == val]['auc'].values[0]
            fmt = f'{val} ({100*auc:3.1f}%)' if self.metric.lower() == 'accuracy' else f'{val} ({auc:3.3f})'
            l = ax.plot(group[x], group[y], '.-', color = color, label = fmt)

            # Fill the curve between -std and +std
            if group['std'].max() > 0:
                fill = ax.fill_between(group[x], (group[y]-group['std']), (group[y]+group['std']), color = color, alpha=.25)
            else:
                fill = None
            # Store the line
            lines[val] = {'line': l[0], 'fill': fill, 'auc': auc}

        # Set the correct limits 
        xlims = (xrange.min()*0.5, xrange.max()*1.1) if not xlims else xlims
        ylims = (yrange.min()*0.5, yrange.max()*1.1) if not ylims else ylims
        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)

        # Add hline for random choice at 1/10
        if self.metric.lower() == 'accuracy':
            ax.hlines(1/10, *xlims, linestyle='--', lw = 1.5, color='black', alpha=1.0, label = 'Random Choice')
        #ax.hlines(maxmetric, *xlims, linestyle='--', lw = 1.5, color='black', alpha=1.0, label = f'Max = {maxmetric:.3f}')

        # Add grid
        ax.grid()                                     # draw grid for major ticks
        ax.grid(which='minor', alpha=0.3)             # draw grid for minor ticks on x-axis

        # Setup labels correctly
        if xlabel: ax.set_xlabel(xlabel)
        # parse ylabel
        if ylabel is not None:
            ylabel = ylabel.replace('mae', 'Mean Absolute Error').replace('mse', 'Mean Squared Error').replace('accuracy', 'Accuracy')
        if ylabel: ax.set_ylabel(ylabel)
        # Add legend at the left, outside of the axis
        legend = ax.legend(title="", #"TMR - AUC(%)", 
                            loc="lower center",                # Position the legend at the top center
                            bbox_to_anchor=(0.5, 1.02),         # Anchor the legend slightly above the plot
                            ncol=3,                            # Let Matplotlib decide the number of columns
                            borderaxespad=0.1,                   # Adjust padding between the legend and plot
                            fontsize=9)

        # Set scale
        if xlog:
            ax.set_xscale('log')
        if ylog:
            ax.set_yscale('log')

        # Setup the title 
        if title: ax.set_title(title)

        # Add the info label
        t = None
        if len(info_label) > 0:
            # Get legend bbox 
            legend_bbox = legend.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
            y2 = legend_bbox.y1
            t = create_label(ax, info_label, -0.3, y2 if not title else 0.95, fontsize=9, border_color="black", padding=0.5, num_columns=2)

        return ax.figure, ax, t, lines
    

    # Method to plot 2D curves 
    def plot_3D_volumes(self, x = 'ber', y = 'mean', hue = 'protection', style = 'protection', ax = None, colors = None, 
                            xrange = None, yrange = None, title = None, xlog = True, ylog = False, 
                            xlabel = None, ylabel = None, xlims = None, ylims = None, zlims = None,
                            zlabel = None, info_label = {}, standalone = True, **kwargs):

       
        # Get the data 
        data = self.curves
        vus = self.vus[y]['vus']

        # I am using the same parent function for both 2D and 3D plots, 
        # which means in this case we actually need to swap hue and y,
        # as well as Zlabel and Ylabel
        y, hue = hue, y
        ylabel = 'Protection(%)'
        zrange, yrange = yrange, None

        # If xrange is none, get the unique of x
        if xrange is None:
            xrange = data[x].unique()
        if yrange is None:
            yrange = data[y].unique()
        if zrange is None:
            zrange = data[hue].unique()
        
        # Make sure xrange and yrange are np.arrays
        xrange = np.array(xrange)
        yrange = np.array(yrange)
        zrange = np.array(zrange)

        xlims = (xrange.min()*0.5, xrange.max()*1.1) if xlims is None else xlims
        if xlog: xlims = np.log10(xlims)
        ylims = (yrange.min()*0.5, yrange.max()*1.1) if ylims is None else ylims
        if ylog: ylims = np.log10(ylims)
        zlims = (np.minimum(0, zrange.min()*0.5), zrange.max()*1.1) if zlims is None else zlims


        # If ax is none, create a new figure
        # if ax is None or standalone:
        #     fig = plt.figure(figsize = (5, 5))
        #     ax = fig.add_subplot(111, projection = '3d')
        # elif not hasattr(ax, 'plot_surface'):
        #     utils._error("The axis passed is not a 3D axis")
        #     fig = plt.figure(figsize = (5, 5))
        #     ax = fig.add_subplot(111, projection = '3d')

        if ax is not None:
            fig = ax.figure
        # If ax is none, create a new figure
        if ax is None or standalone:
            fig = plt.figure(figsize = (5, 5))
            ax = fig.add_subplot(111, projection = '3d')
    
        # Mark figure as deletable 
        netsurf.utils.mark_figure_as_deletable(fig)

        # Store the lines for the legend so we can link them to Qt widgets to turn them on and off
        lines = {}

        # Loop thru values of hue 
        if True:
            # Plot the curve

            # 1. Create a regular grid for X and Y
            x_grid = np.linspace(min(data[x]), max(data[x]), 50)  # Define X grid points
            y_grid = np.linspace(min(data[y]), max(data[y]), 50)  # Define Y grid points
            X, Y = np.meshgrid(x_grid, y_grid)        # Create meshgrid

            # 2. Interpolate Z values onto the grid
            Z = griddata((data[x], data[y]), data[hue], (X, Y), method='cubic')  # Interpolate to 2D grid

            # Plot the 3D surface
            edgecolor = '#0000001f'
            #facecolor = colors[i%len(colors)] # 'royalblue'
            if xlog: X = np.log10(X)
            if ylog: Y = np.log10(Y)
            surface = ax.plot_surface(X, Y, Z, edgecolor=edgecolor, cmap = 'bwr', lw=0.8) #, facecolor = 'royalblue', rstride=1, cstride=1, alpha=0.5)
            #ax.set_title(method)

            # Plot projections of the contours for each dimension.  By choosing offsets
            # that match the appropriate axes limits, the projected contours will sit on
            # the 'walls' of the graph
            # if any([s == 1 for s in X.shape]) or any([s == 1 for s in Y.shape]) or any([s == 1 for s in Z.shape]):
            #     # delete this axis and continue 
            #     fig.delaxes(ax)
            #     return fig, None, None, None
            
            proj_z = ax.contourf(X, Y, Z, zdir='z', offset=0, cmap='coolwarm', alpha = 0.5)
            proj_x = ax.contourf(X, Y, Z, zdir='x', offset=xlims[0], cmap='coolwarm', alpha = 0.5)
            proj_y = ax.contourf(X, Y, Z, zdir='y', offset=ylims[1], cmap='coolwarm', alpha = 0.5)

            # Set limits
            ax.set(xlim=xlims, ylim=ylims, zlim=zlims, xlabel='BER(%)', ylabel='Protection(%)', zlabel = zlabel)#, zlabel=metric)
            
            def log_tick_formatter(val, pos=None):
                return f"$10^{{{int(val)}}}$"

            if xlog:
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
                ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            if ylog:
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
                ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            # Get vuc 
            #axs[i].legend([f'VUC={VUC:.3f}'], fontsize = 10)
            ax.set_title(f'VUC = {vus:.3f}')

            ax.set_box_aspect([1, 1, 0.5])  # Equal aspect ratio


            # Store the line
            lines[0] = {'line' : surface, 'surface': surface, 'vus': vus} #'proj_x': proj_x, 'proj_y': proj_y, 'proj_z': proj_z, 

        # # Set the correct limits 
        # xlims = (xrange.min()*0.5, xrange.max()*1.1) if not xlims else xlims
        # ylims = (yrange.min()*0.5, yrange.max()*1.1) if not ylims else ylims
        # ax.set_xlim(*xlims)
        # ax.set_ylim(*ylims)

        # Add hline for random choice at 1/10
        #ax.hlines(1/10, *xlims, linestyle='--', lw = 1.5, color='black', alpha=1.0, label = 'Random Choice')
        #ax.hlines(maxmetric, *xlims, linestyle='--', lw = 1.5, color='black', alpha=1.0, label = f'Max = {maxmetric:.3f}')

        # Add grid
        #ax.grid()                                     # draw grid for major ticks
        #ax.grid(which='minor', alpha=0.3)             # draw grid for minor ticks on x-axis

        # Setup labels correctly
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        if zlabel: ax.set_zlabel(zlabel)

        # Add legend at the left, outside of the axis
        # legend = ax.legend(title="", #"TMR - AUC(%)", 
        #                     loc="lower center",                # Position the legend at the top center
        #                     bbox_to_anchor=(0.5, 1.02),         # Anchor the legend slightly above the plot
        #                     ncol=3,                            # Let Matplotlib decide the number of columns
        #                     borderaxespad=0.1,                   # Adjust padding between the legend and plot
        #                     fontsize=9)

        # Set scale
        # if xlog:
        #     ax.set_xscale('log')
        # if ylog:
        #     ax.set_yscale('log')
    

        # Setup the title 
        if title: ax.set_title(title)

        # Add the info label
        t = None
        # if len(info_label) > 0:
        #     # Get legend bbox 
        #     legend_bbox = legend.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
        #     y2 = legend_bbox.y1
        #     t = create_label(ax, info_label, -0.3, y2 if not title else 0.95, fontsize=9, border_color="black", padding=0.5, num_columns=2)

        return fig, ax, t, lines


    # With this data we can now compute the average area under the curve for each tmr.
    # This can be computed using the trapezoidal rule.
    # which is: (x1 - x0) * (y1 + y0) / 2
    def _compute_auc(self, d, x = 'ber', y = 'mean'):
        def _auc(d, x = 'ber', y = 'mean'):
            trapzs = (d[y][1:].values + d[y][:-1].values) * (d[x][1:].values - d[x][:-1].values) / 2
            # compute maximum possible (1, 0) trapz
            ref = (d[x].max() - d[x].min()) * (1 - 0)
            # relative max 
            rel_ref = (d[x].max() - d[x].min()) * (d[y].max() - 0)
            # sum trapzs
            return trapzs.sum()/ref, trapzs.sum()/rel_ref
        
        # One for normal ber
        auc = _auc(d, x = x, y = y)
        # One for true 
        auc2 = _auc(d, x = x.replace('ber','true_ber'), y = y.replace('ber','true_ber'))
        return auc + auc2
    
    def compute_auc(self, curves):
        aucs = []
        for (g,x) in [['ber','protection'], ['protection','ber']]:
            for val, group in curves.groupby(g):
                for y in ['mean', 'max', 'min', 'std']:
                    _aucs = self._compute_auc(group, x = x, y = y)
                    aucs.append({g: val, x: 'all', 'auc': _aucs[0], 'rel_auc': _aucs[1], 'true_auc': _aucs[2], 'true_rel_auc': _aucs[3],
                                 'x': x, 'y': y})
        return pd.DataFrame(aucs)


    # Compute the volume under the curve using the tetrahedron method
    def tetrahedron_volume(self, points):
        """
        Calculate the volume of a tetrahedron given its four vertices in 3D space.
        
        Args:
            points (list of tuples or ndarray): A list or array of four (x, y, z) vertices.
        
        Returns:
            float: The absolute value of the tetrahedron's volume.
        """
        if len(points) == 0:
            return 0.0
        # Ensure the input is a NumPy array
        points = np.array(points)

        # Ensure we have the right dimensions
        if len(points) < 4:
            return 0.0
        
        # Compute vectors relative to the first point
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        v3 = points[3] - points[0]
        
        # Scalar triple product: v1 · (v2 × v3)
        volume = np.abs(np.dot(v1, np.cross(v2, v3))) / 6
        return volume

    """ Compute the volume under the curve using the tetrahedron method """
    def compute_vus(self, curves):
        
        # Get unique tmr values
        protections = curves['protection'].unique()
        # Get unique ber values
        bers = curves['ber'].unique()
        # true 
        true_bers = curves['true_ber'].unique()

        # Make sure they are sorted
        protections.sort()
        bers.sort()
        true_bers.sort()

        dats = {'protection': protections, 'ber': bers, 'true_ber': true_bers}

        # Loop thru the protection groups and compute the auc for each protection and ber
        volumes = {}
        for m in ['median', 'mean', 'max', 'min', 'std']:
            for xn,yn,zn in [['protection', 'ber',''], ['protection', 'true_ber','true_']]:
                x = dats[xn]
                y = dats[yn]
                yref = dats['ber']
                _volume = 0
                for i in range(len(x)-1):
                    for j in range(np.minimum(len(y),len(yref))-1):
                        # Now we can form the tetrahedron
                        # We need to get the 4 points
                        # 1. (protection[i], ber[i], mean)
                        # 2. (protection[i], ber[i+1], mean)
                        # 3. (protection[i+1], ber[i], mean)
                        # 4. (protection[i+1], ber[i+1], mean)
                        points = []
                        minz = np.inf
                        for xx,yy in [(x[i], yref[j]), (x[i], yref[j+1]), (x[i+1], yref[j+1]), (x[i+1], yref[j])]:
                            # Get the mean value
                            mean = curves[(curves[xn] == xx) & (curves['ber'] == yy)][m].values
                            if len(mean) > 0:
                                mean = mean[0]
                                points.append((xx, yy, mean))
                                minz = min(minz, mean)
                        
                                # We need to add the minimum volume of the column below the lowest point in Z
                                _min_vol = minz * abs(x[i+1] - x[i]) * abs(y[j+1] - y[j])

                                # Compute the volume
                                _volume += (self.tetrahedron_volume(points) + _min_vol)

                if _volume > 0:
                    # We need to normalize over the maximum possible
                    dx = x.max() - x.min()
                    dy = y.max() - y.min()
                    _volume_max = abs(dx) * abs(dy) * 1
                    _volume_rel = abs(dx) * abs(dy) * curves[m].max()
                    _volume_min = abs(dx) * abs(dy) * curves[m].min()
                    # Normalize 
                    _volume_range = (_volume-_volume_min)/(_volume_max-_volume_min)
                    _volume_rel = _volume / _volume_rel
                    _volume = _volume / _volume_max
                else:
                    _volume_rel = 0
                    _volume_range = 0
                            
                # Store the volume
                if m not in volumes:
                    volumes[m] = {}

                volumes[m].update({f'{zn}vus': _volume, f'{zn}vus_rel': _volume_rel, f'{zn}vus_range': _volume_range})

        # Dataframe
        volumes = pd.DataFrame(volumes)

        return volumes

        



        
